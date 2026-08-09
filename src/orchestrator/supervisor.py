"""The epoch supervisor: single-writer membership for multi-node training.

Replaces torchrun's dynamic c10d rendezvous (a version-dependent black box that
recovered in 5s locally on torch 2.4 but hung >180s on the DLAMI's torch) with
explicit central orchestration. The orchestrator is the ONLY writer of the
membership document ``runs/<run_id>/epoch.json``; every box's sidecar
(``sidecar.py``) polls it and runs STATIC torchrun for the current epoch. One
writer, N readers, monotonic epochs — every recovery step is our code emitting
our logs, and nothing waits on a timeout to infer what the supervisor decided.

This module splits cleanly into:
  - :func:`decide` — a PURE reducer (Observation, Policy) -> list[Action]. All
    the membership logic, table-testable without AWS.
  - :class:`Supervisor` — the imperative shell: builds an Observation each tick
    (AWS + S3), executes the actions, and emits the profile marks the W&B
    world-size staircase / degraded phase already consume.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from dataclasses import dataclass, field

from spot_train import events, s3_store

from . import aws
from .config import LEADER_VICTIM, OrchestratorConfig
from .profile import RunProfile

# --------------------------------------------------------------------------- #
# Observation (what the supervisor sees) and Policy (how it should react)
# --------------------------------------------------------------------------- #

# AWS instance states that mean a node is gone (or going), not a live member.
_DEAD_STATES = frozenset({"shutting-down", "terminated", "stopping", "stopped"})


@dataclass(frozen=True)
class NodeObs:
    """One node's health, as the supervisor can observe it from OUTSIDE the box.

    Health is observation-only on purpose: even a node the orchestrator itself
    terminated (a scheduled kill standing in for a spot reclaim) is discovered
    dead the same way a real reclaim would be — its AWS state flips into
    ``_DEAD_STATES`` or its log heartbeat goes stale. The orchestrator never
    shortcuts membership with "I know I killed it"."""

    node: int
    aws_state: str  # DescribeInstances state, or "unknown"
    registered: bool  # node<i>.json present in S3 (the box announced its IP)
    log_age_s: float | None = None  # seconds since the log key last changed (heartbeat)


@dataclass(frozen=True)
class Observation:
    """Everything ``decide`` needs for one tick — a pure snapshot."""

    node_count: int  # desired full group size
    nodes: tuple[NodeObs, ...]
    epoch: int  # current published epoch (0 = none yet)
    members: frozenset[int]  # node indices in the current epoch
    metrics_exists: bool  # run's done signal landed
    no_progress_s: float | None  # seconds since the checkpoint step last advanced (None = n/a yet)
    due_kills: frozenset[int] = frozenset()  # scheduled victims whose time has arrived
    # Epochs published in a row that never produced a checkpoint. The stall clock
    # restarts on every epoch (a new world legitimately needs time before its
    # first checkpoint), so this is what still catches a world that FLAPS —
    # re-forming endlessly without ever training. Without it, resetting the clock
    # would make the deadlock-breaker unreachable.
    epochs_without_progress: int = 0


@dataclass(frozen=True)
class Policy:
    """The knobs that make one supervisor a shrink experiment and another a
    preempt experiment — passed to the pure reducer so behavior is explicit."""

    replace_on_loss: bool  # relaunch a lost node (preempt) vs let the group shrink (shrink)
    recovery_timeout_s: float  # no checkpoint progress this long -> whole-group restart
    heartbeat_timeout_s: float = 90.0  # log key stale this long -> node presumed dead
    # Consecutive epochs with no checkpoint before we stop believing the world can
    # form. Each epoch gets a full recovery_timeout_s, so this many failures is
    # already several minutes of evidence — enough to conclude the membership
    # itself is the problem and only a clean slate will do.
    #
    # Budget it in PREEMPTIONS, not epochs: one preemption publishes TWO epochs
    # (shrink onto the survivors, then grow when the replacement joins). At 3,
    # ~1.5 normal recoveries exhausted it — so a counter meant to catch "this
    # world can NEVER form" could be spent by the mechanism working exactly as
    # designed, and its penalty is the most destructive action available:
    # discarding every healthy survivor. 6 = three full preemptions.
    max_epochs_without_progress: int = 6


def _healthy(n: NodeObs, policy: Policy) -> bool:
    """A node counts toward membership iff it announced itself, AWS still shows it
    running, and its log heartbeat isn't stale — all observation, no foreknowledge
    of who we killed."""
    if not n.registered or n.aws_state in _DEAD_STATES:
        return False
    if n.aws_state != "running":
        return False  # pending/unknown: not yet a member
    # Fresh boot has no log yet (age None) — treat as alive; only a stale, once-
    # live log means wedged-but-alive.
    return not (n.log_age_s is not None and n.log_age_s > policy.heartbeat_timeout_s)


# --------------------------------------------------------------------------- #
# Actions (what the reducer decides; the shell executes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PublishEpoch:
    epoch: int
    members: tuple[int, ...]  # node indices, sorted; rank = position, master = members[0]


@dataclass(frozen=True)
class TerminateNode:
    node: int


@dataclass(frozen=True)
class LaunchReplacement:
    node: int


@dataclass(frozen=True)
class WholeGroupRestart:
    # WHY the floor fired. This is the most destructive action the supervisor can
    # take — it discards every healthy survivor and relaunches the fleet — and it
    # used to be logged as a bare "whole-group restart (floor)". When an 8-node
    # run amplified 6 injected kills into 22 node slots, the log could not say
    # which of the three conditions was responsible, so the cause had to be
    # guessed. Never again: the reason travels with the action.
    # Compared by identity of the reason too, so tables stay explicit.
    reason: str = ""


@dataclass(frozen=True)
class Done:
    pass


Action = PublishEpoch | TerminateNode | LaunchReplacement | WholeGroupRestart | Done


# --------------------------------------------------------------------------- #
# The pure reducer
# --------------------------------------------------------------------------- #
def decide(obs: Observation, policy: Policy) -> list[Action]:
    """Observed state -> the actions that reconcile it toward "N healthy members
    training". Pure: no AWS, no clock, no I/O — every branch is table-testable.

    The heart is trivial by design (that's the point of central orchestration):
    the membership that SHOULD be published is just the currently-healthy set,
    and an epoch is (re)published whenever that set differs from what's live.

    Membership is driven ONLY by the observed ``healthy`` set. A scheduled kill
    (``due_kills``) merely emits ``TerminateNode`` — it does NOT itself shrink
    the group. The shrink happens a tick or two later when the terminated box is
    OBSERVED gone (AWS state flips into _DEAD_STATES, or its heartbeat goes
    stale), exactly as a real, un-orchestrated spot reclaim would be discovered.
    """
    if obs.metrics_exists:
        return [Done()]

    healthy = frozenset(n.node for n in obs.nodes if _healthy(n, policy))

    # Whole-group restart floor: the group stopped making progress, kept
    # re-forming without ever training, or every member is observed gone (no
    # survivor to shrink onto).
    #
    # NB the no-progress clock is restarted by the shell on every epoch
    # publication. It has to be: a preemption arriving late in a checkpoint
    # interval would otherwise inherit a nearly-spent budget and trip this
    # instantly, discarding healthy survivors to "recover" from a world that was
    # recovering fine. The clock measures "this world has had its chance", not
    # "time since some checkpoint" — and epochs_without_progress is what keeps
    # the deadlock-breaker reachable once the clock can be reset.
    #
    # Under a HIGH failure rate the first two conditions need care: every kill
    # publishes two epochs (shrink, then grow), so a naive per-epoch counter
    # spends its whole budget on ~1.5 preemptions and nukes a fleet that is
    # recovering exactly as designed. Both are therefore judged against a world
    # that still has healthy members — if survivors are present and the world is
    # re-forming around them, that is the mechanism working, not a deadlock.
    if obs.epoch > 0:
        reason = ""
        if not healthy:
            # Nothing left to shrink onto: the only way back is a full relaunch.
            reason = "no healthy members"
        elif obs.no_progress_s is not None and obs.no_progress_s > policy.recovery_timeout_s:
            reason = (
                f"no checkpoint progress for {obs.no_progress_s:.0f}s "
                f"(> recovery_timeout {policy.recovery_timeout_s:.0f}s)"
            )
        elif obs.epochs_without_progress >= policy.max_epochs_without_progress:
            reason = (
                f"{obs.epochs_without_progress} epochs published with no checkpoint "
                f"(>= max {policy.max_epochs_without_progress})"
            )
        if reason:
            return [WholeGroupRestart(reason)]

    # Startup: don't begin training degraded — wait for the whole group to
    # register and reach running, then publish epoch 1.
    if obs.epoch == 0:
        if len(healthy) >= obs.node_count:
            return [PublishEpoch(1, tuple(sorted(healthy))[: obs.node_count])]
        return []

    actions: list[Action] = []
    # Scheduled kills: terminate the box (a stand-in for the reclaim). This does
    # NOT touch membership — the shrink below only reacts to what's observed.
    for v in sorted(obs.due_kills):
        actions.append(TerminateNode(v))

    # Reconcile membership to the observed healthy set. A member that is no
    # longer healthy (a reclaimed/terminated box observed dead) drops out and
    # the group shrinks; a newly-healthy non-member (a replacement that booted)
    # grows it back.
    if healthy and healthy != obs.members:
        actions.append(PublishEpoch(obs.epoch + 1, tuple(sorted(healthy))))

    if policy.replace_on_loss:
        for v in sorted(obs.members - healthy):  # members observed gone
            actions.append(LaunchReplacement(v))

    return actions


# --------------------------------------------------------------------------- #
# The imperative shell
# --------------------------------------------------------------------------- #
def restore_state(st: SupervisorState, doc: dict | None) -> SupervisorState:
    """Re-adopt a live world from the durable epoch document.

    SupervisorState is cross-tick memory held in the PROCESS, so a restarted
    supervisor starts at epoch 0 / members {} -- and ``decide`` reads
    ``obs.epoch == 0`` as "nothing has ever been published" and republishes
    EPOCH 1 over a live world. Sidecars kill and relaunch torchrun on an epoch
    change, so all N boxes churn at once, their heartbeats go quiet, and
    ``replace_on_loss`` then launches a replacement for every one of them. That
    is how a supervisor SIGKILL grew an 8-node fleet to 14 in the rehearsal.

    The authority already lives in S3 -- epoch.json is written on every
    publication. This just reads it back, so a restart is a no-op instead of a
    fleet rebuild. Which is what makes "if the supervisor throws, reboot it" a
    cheap strategy rather than a $3-and-four-minutes one.

    Deliberately total: a missing document is a genuine cold start (epoch 0 is
    then the truth and the startup branch SHOULD run), and a malformed one --
    truncated, half-written, hand-edited -- degrades to the same, because
    raising here would turn a recoverable restart into a dead control plane.
    """
    if not doc:
        return st
    try:
        epoch = int(doc.get("epoch") or 0)
        raw = doc.get("members") or []
        members = {int(m["node"]): m.get("ip") for m in raw if m.get("node") is not None}
    except (TypeError, ValueError, AttributeError):
        return st
    if epoch <= 0 or not members:
        return st
    st.epoch = epoch
    st.members = frozenset(members)
    st.ips = {n: ip for n, ip in members.items() if ip}
    # epoch_doc() builds `ordered = [master, *rest]`, so rank 0 IS the master --
    # the ordering is the authority and there is no separate field to trust.
    ranked = sorted(
        (m for m in raw if m.get("node") is not None),
        key=lambda m: m.get("rank", 0),
    )
    st.master = int(ranked[0]["node"]) if ranked else None
    return st


def elect_master(
    members: tuple[int, ...], prev_members: frozenset[int], current_master: int | None
) -> int:
    """Pick the rendezvous master (which becomes rank 0 — it hosts the c10d
    TCPStore every rank bootstraps against). STICKY to a proven survivor:

    1. keep the current master if it's still a member — it already has a running
       torchrun / store up, so never migrate the store onto a fresh box;
    2. else the lowest-index member that was in the PREVIOUS epoch (a survivor
       that's been training, not a just-booted replacement);
    3. else (startup, or the whole group was replaced) the lowest-index member.

    This is why a master(rank-0) preemption recovers like a worker preemption
    instead of deadlocking the grow-back: when the killed master's replacement
    re-takes its node index, the master role does NOT follow it — it stays on a
    survivor whose store is already live, and the replacement joins as a worker.
    No leader election needed; the single-writer supervisor just assigns it."""
    ordered = sorted(members)
    if current_master is not None and current_master in members:
        return current_master
    survivors = [n for n in ordered if n in prev_members]
    return survivors[0] if survivors else ordered[0]


def epoch_doc(
    run_id: str,
    epoch: int,
    members: tuple[int, ...],
    ips: dict[int, str],
    port_base: int,
    master: int,
) -> dict:
    """The membership document, built from the node->ip map the boxes registered.
    The elected ``master`` is placed at rank 0 (torchrun needs master_addr to be
    rank 0's host, since that agent starts the rendezvous store); the remaining
    members keep sorted order at ranks 1..N-1."""
    ordered = [master, *(n for n in sorted(members) if n != master)]
    ranked = [{"node": n, "ip": ips[n], "rank": r} for r, n in enumerate(ordered)]
    return {
        "epoch": epoch,
        "members": ranked,
        "node_count": len(members),
        "master_addr": ips[master],
        "master_port": port_base + epoch,
    }


def status_doc(
    run_id: str,
    obs: Observation,
    policy: Policy,
    *,
    epoch: int,
    members: frozenset[int],
    ips: dict[int, str],
    node_ids: dict[int, str],
    logs: dict[int, dict],
    orch_log_key: str,
    prev: dict | None,
    now: float,
    ckpt_step: int = -1,
    done: bool = False,
) -> dict:
    """The observability document (status.json), rewritten every tick so ANY
    process — chiefly the ``spot-orchestrate logs`` viewer — can discover each
    (node, attempt)'s log key and liveness without the driver's in-memory state.

    Entries are keyed by (node, attempt): a replacement reuses the node index
    but is a different box with a different log, so it gets its OWN entry and
    the superseded attempt is carried forward from ``prev`` frozen at "dead".
    Death is sticky per (node, attempt) — once dead, an entry never resurrects
    (its log is frozen for the viewer), matching how replacements actually
    arrive as fresh attempts."""
    prev_nodes = {(e["node"], e["attempt"]): e for e in (prev or {}).get("nodes", [])}
    by_node = {n.node: n for n in obs.nodes}
    entries: dict[tuple[int, int], dict] = {}
    for node, entry in logs.items():
        attempt = entry.get("attempt", 0)
        n = by_node.get(node)
        pe = prev_nodes.get((node, attempt))
        if pe is not None and pe["state"] == "dead":
            state = "dead"
        elif n is None:
            state = "pending"
        elif _healthy(n, policy):
            state = "alive"
        elif n.aws_state in _DEAD_STATES:
            state = "dead"
        elif pe is not None and pe["state"] == "alive":
            state = "dead"  # was alive, now unhealthy: a stale-heartbeat wedge
        else:
            state = "pending"  # booting/joining — never was alive
        entries[(node, attempt)] = {
            "node": node,
            "attempt": attempt,
            "log_key": entry["key"],
            "state": state,
            "aws_state": n.aws_state if n is not None else "unknown",
            "instance_id": node_ids.get(node),
            "ip": ips.get(node),
            "log_age_s": n.log_age_s if n is not None else None,
        }
    # Superseded attempts (their node now maps to a newer attempt) stay
    # enumerable forever, frozen dead — the viewer keeps their log readable.
    for k, pe in prev_nodes.items():
        if k not in entries:
            entries[k] = {**pe, "state": "dead"}
    return {
        "version": 1,
        "run_id": run_id,
        "updated_at": now,
        "epoch": epoch,
        "members": sorted(members),
        # Checkpoint progress: the viewer uses this to tell when survivors have
        # actually resumed training after an epoch change (their torchrun crashed
        # on the NCCL abort and re-rendezvoused at the new world size) vs. are
        # still restarting/restoring.
        "ckpt_step": ckpt_step,
        "done": done,
        "orchestrator": {"log_key": orch_log_key},
        "nodes": [entries[k] for k in sorted(entries)],
    }


@dataclass
class SupervisorState:
    """Cross-tick memory the pure reducer deliberately doesn't hold."""

    epoch: int = 0
    members: frozenset[int] = frozenset()
    master: int | None = None  # current rendezvous master (rank 0); sticky across grows
    replacing: set[int] = field(default_factory=set)  # a LaunchReplacement is in flight
    ips: dict[int, str] = field(default_factory=dict)  # node -> private IP (from registration)
    ckpt_step: int = -1
    ckpt_changed_at: float = 0.0
    # Epochs published since the last checkpoint advanced. Reset by real progress;
    # bumped by each publication. See Observation.epochs_without_progress.
    epochs_without_progress: int = 0
    shrink_baseline: int | None = None  # ckpt step at the last kill (for shrink_resume)
    full_ws: int | None = None  # world size before the last kill (for full_world)
    marks: set[str] = field(default_factory=set)  # one-shot marks already emitted per epoch cycle


class Supervisor:
    """The imperative shell around :func:`decide`: one observe->decide->act tick
    per ``log_stream_seconds``. Startup (launch N, wait for registrations) is the
    experiment driver's job; this owns the steady state — publishing epoch 1 once
    everyone's healthy, reacting to losses, streaming every node's log so the
    console is never blind, and emitting the profile marks the W&B world-size
    staircase / degraded phase consume. Effectful bits that live in the
    experiment (launching a box) are injected as callbacks to avoid an import
    cycle."""

    def __init__(
        self,
        cfg: OrchestratorConfig,
        profile: RunProfile,
        *,
        run_id: str,
        policy: Policy,
        node_ids: dict[int, str],  # node index -> instance id (mutated on replacement)
        logs: dict[int, dict],  # node -> {"key", "state"}: the live log streams
        launch_node,  # (node_index) -> instance_id (waits running, records cost)
        pull_logs,  # () -> None: pull every node's log into the profile
        kill_schedule: list[tuple[float, int]] | None = None,  # (secs after train start, victim)
    ):
        self.cfg = cfg
        self.profile = profile
        self.run_id = run_id
        self.policy = policy
        self.node_ids = node_ids
        self.logs = logs
        self._launch_node = launch_node
        self._pull_logs = pull_logs
        self.kill_schedule = list(kill_schedule or [])
        self.st = SupervisorState()
        self.ckpt_prefix = f"{cfg.run_prefix}/{run_id}/checkpoints/"
        self.metrics_key = cfg.run_metrics_key(run_id)
        self._train_start: float | None = None
        # Wall-clock twin of _train_start. Persisted, because monotonic resets
        # with the process and the kill clock must not.
        self._train_start_wall: float = 0.0
        # Kill scheduling is EDGE-triggered per schedule ENTRY (not per node):
        # once entry i has been issued it never fires again, even after its
        # victim is replaced and re-added to the group. (A level trigger on
        # "elapsed >= secs" plus per-node dedup would re-kill every replacement
        # the instant it rejoined — an infinite kill loop.)
        self._fired_kills: set[int] = set()
        self._terminated_iids: set[str] = set()  # guard against double-terminating a box
        self.metrics: dict | None = None
        # Observability: status.json rewritten every tick + the supervisor's own
        # decision log uploaded next to the box logs, so the `logs` viewer can
        # show live per-node tabs plus the control plane's narrative.
        self.status_key = cfg.run_status_key(run_id)
        self.orch_log_key = cfg.run_orch_log_key(run_id)
        self._last_status: dict | None = None
        self._orch_lines: list[str] = []
        self._orch_dirty = False
        self._downed: set[tuple[int, str]] = set()  # (node, iid) already emitted down/killed

    # -- durable chaos-schedule progress ------------------------------------ #
    def _save_schedule(self, wall: float) -> None:
        """Persist which kill entries have fired, and when training began.

        Housekeeping: a failure here must never end the run. The cost of losing
        it is a replayed schedule, which is bad but survivable; the cost of
        raising is a dead control plane.
        """
        if self._train_start is None:
            return
        with contextlib.suppress(Exception):
            aws.put_text(
                self.cfg.bucket,
                self.cfg.run_schedule_key(self.run_id),
                json.dumps(
                    {
                        "version": 1,
                        "train_start_wall": self._train_start_wall,
                        "fired": sorted(self._fired_kills),
                    }
                ),
            )

    def _restore_schedule(self, now: float, wall: float) -> None:
        """Re-arm the kill clock from S3 instead of from this process's birth.

        `_train_start` is a time.monotonic() reading, and monotonic resets with
        the process -- which is precisely the bug. So the WALL start is what is
        persisted, and the monotonic field is reconstructed by offsetting it, so
        `elapsed` keeps measuring time since training actually began rather than
        since the supervisor last restarted.
        """
        doc = None
        with contextlib.suppress(Exception):
            doc = aws.get_json(self.cfg.bucket, self.cfg.run_schedule_key(self.run_id))
        if not doc:
            return
        try:
            fired = {int(i) for i in doc.get("fired") or []}
            started = float(doc.get("train_start_wall") or 0.0)
        except (TypeError, ValueError):
            return
        if started <= 0:
            return
        self._fired_kills = fired
        self._train_start_wall = started
        self._train_start = now - max(0.0, wall - started)
        remaining = [s for i, (s, _v) in enumerate(self.kill_schedule) if i not in fired]
        self._event(
            f"re-armed kill schedule: {len(fired)} of {len(self.kill_schedule)} already "
            f"fired, {len(remaining)} pending, training began {wall - started:.0f}s ago "
            f"(not replaying)"
        )

    # -- observability ------------------------------------------------------ #
    def _event(self, msg: str) -> None:
        """A supervisor decision line: stderr (as always) + the orchestrator.log
        buffer the next _write_status uploads."""
        print(f"[supervisor] {msg}", file=sys.stderr)
        self._orch_lines.append(f"[{time.strftime('%H:%M:%S')}] [supervisor] {msg}")
        self._orch_dirty = True

    def _emit_event(self, state: str, **fields) -> None:
        """A structured ``[event]`` record (the event-sourced timeline's orch
        half): appended to orchestrator.log — which the viewer parses — AND
        echoed to stderr. ts is stamped at the source here, not on parse."""
        buf = io.StringIO()
        events.emit(state, by="orch", stream=buf, **fields)
        line = buf.getvalue().rstrip("\n")
        print(line, file=sys.stderr, flush=True)
        self._orch_lines.append(line)
        self._orch_dirty = True

    def _write_status(self, obs: Observation, wall: float, *, done: bool = False) -> None:
        """Upload status.json (every tick — updated_at doubles as the supervisor's
        own heartbeat) and orchestrator.log (only when new events accrued).
        Observability must never kill the run: any failure is a stderr line."""
        try:
            doc = status_doc(
                self.run_id,
                obs,
                self.policy,
                epoch=self.st.epoch,
                members=self.st.members,
                ips=self.st.ips,
                node_ids=self.node_ids,
                logs=self.logs,
                orch_log_key=self.orch_log_key,
                prev=self._last_status,
                now=wall,
                ckpt_step=self.st.ckpt_step,
                done=done,
            )
            body = json.dumps(doc)
            aws.put_text(self.cfg.bucket, self.status_key, body)
            # HISTORY, not just current state. status.json is overwritten in
            # place, so the world-size staircase and the Gantt exist only if
            # something captured every tick -- and today that something is
            # live.py polling from a LAPTOP. Close the lid for an hour and that
            # hour of fleet history is unrecoverable: it was never written down.
            # Step/loss/cost survive (node logs are append-only in S3); world
            # size and slot occupancy do not.
            #
            # The supervisor IS the writer, so it publishes each tick as its own
            # immutable object and `aws s3 sync` on the prefix reconstructs the
            # full history from anywhere. One object per tick rather than one
            # growing jsonl: rewriting a 13 MB file every 10s would be ~78 GB of
            # PUTs over a 24h run.
            if doc.get("updated_at") != (self._last_status or {}).get("updated_at"):
                aws.put_text(
                    self.cfg.bucket,
                    self.cfg.run_status_tick_key(self.run_id, wall),
                    body,
                    quiet=True,
                )
            self._last_status = doc
            # Feed the live dashboard the state only the supervisor has. The
            # profile parses trainer logs, so without this push `ckpt_step` --
            # durable progress, the one measure a failure cannot inflate -- is
            # absent from every live run and present only in replays.
            self.profile.observe(
                ckpt_step=self.st.ckpt_step,
                members=list(doc.get("members") or []),
                target_world=self.cfg.node_count,
                whole_group_restarts=getattr(self.st, "whole_group_restarts", 0),
            )
            if self._orch_dirty:
                aws.put_text(self.cfg.bucket, self.orch_log_key, "\n".join(self._orch_lines) + "\n")
                self._orch_dirty = False
        except Exception as exc:  # noqa: BLE001
            print(f"[supervisor] status write failed (non-fatal): {exc}", file=sys.stderr)

    # -- observation ------------------------------------------------------- #
    def _node_ip(self, node: int) -> str | None:
        """Private IP the box registered (node<i>.json), cached once seen."""
        if node in self.st.ips:
            return self.st.ips[node]
        # Full s3:// URI, not the bare key: s3_store.read_bytes treats a
        # prefix-less string as a LOCAL path, so a bare key silently reads as
        # "absent" and the node never counts as registered (epoch 1 never fires).
        raw = s3_store.read_bytes(self.cfg.run_node_uri(self.run_id, node))
        if raw is None:
            return None
        try:
            doc = json.loads(raw)
            ip = doc["ip"]
        except (ValueError, KeyError):
            return None
        # A registration is only OURS if the box that wrote it is the box now
        # occupying this slot. nodes/node<i>.json is keyed by node INDEX and
        # persists in S3, so after a kill the DEAD occupant's doc is still there:
        # clearing the in-memory cache just forced a re-read of it, and the slot
        # read as registered again within one tick. That is what let the reducer
        # regrow onto a corpse at t+288s while the replacement did not boot until
        # t+440s — 204s of survivors idling. Comparing instance ids makes the slot
        # unregistered until the REPLACEMENT writes its own doc.
        # ("unknown" = IMDS unavailable, i.e. the localhost E2E; trust it there.)
        expected = self.node_ids.get(node)
        got = doc.get("instance_id")
        if expected and got and got != "unknown" and got != expected:
            return None
        self.st.ips[node] = ip
        return ip

    def _observe(self, now: float, wall: float) -> Observation:
        nodes = []
        for node, iid in self.node_ids.items():
            state = aws.instance_state(iid) if iid else "unknown"
            log_lm = aws.object_last_modified(self.cfg.bucket, self.logs[node]["key"])
            nodes.append(
                NodeObs(
                    node=node,
                    aws_state=state,
                    registered=self._node_ip(node) is not None,
                    log_age_s=(wall - log_lm) if log_lm is not None else None,
                )
            )
        # Checkpoint progress: track when the max step last advanced.
        step = aws.max_checkpoint_step(self.cfg.bucket, self.ckpt_prefix)
        if step > self.st.ckpt_step:
            # Real progress: the world is training. Clears BOTH stall signals.
            self.st.ckpt_step, self.st.ckpt_changed_at = step, now
            self.st.epochs_without_progress = 0
        no_progress = (now - self.st.ckpt_changed_at) if self._train_start is not None else None

        due = set()
        if self._train_start is not None:
            elapsed = now - self._train_start
            for i, (secs, victim) in enumerate(self.kill_schedule):
                if elapsed >= secs and i not in self._fired_kills:
                    # LEADER_VICTIM resolves HERE, not when the schedule was
                    # built: elect_master is sticky to a survivor, so after any
                    # kill the master may no longer be node 0 and a fixed index
                    # would stop exercising re-election. If there is no master
                    # yet, leave the entry unfired so it lands on a later tick
                    # rather than being silently dropped.
                    if victim == LEADER_VICTIM:
                        if self.st.master is None:
                            continue
                        victim = self.st.master
                    due.add(victim)
                    self._fired_kills.add(i)  # fire this entry exactly once
                    self._save_schedule(wall)

        return Observation(
            node_count=self.cfg.node_count,
            nodes=tuple(nodes),
            epoch=self.st.epoch,
            members=self.st.members,
            metrics_exists=aws.object_exists(self.cfg.bucket, self.metrics_key),
            no_progress_s=no_progress,
            due_kills=frozenset(due),
            epochs_without_progress=self.st.epochs_without_progress,
        )

    # -- current world size (for full_world), read from the profile stream -- #
    def _latest_ws(self) -> int | None:
        for s in reversed(self.profile.samples):
            if s.world_size:
                return s.world_size
        return None

    # -- effects ----------------------------------------------------------- #
    def _publish_epoch(self, epoch: int, members: tuple[int, ...]) -> None:
        shrinking = self.st.members and len(members) < len(self.st.members)
        # Elect BEFORE updating st.members/st.master (elect_master reads the prior
        # epoch's set + current master to stay sticky to a survivor).
        master = elect_master(members, self.st.members, self.st.master)
        doc = epoch_doc(self.run_id, epoch, members, self.st.ips, self.cfg.rdzv_port, master)
        aws.put_text(self.cfg.bucket, self.cfg.run_epoch_key(self.run_id), json.dumps(doc))
        self.st.epoch, self.st.members, self.st.master = epoch, frozenset(members), master
        self.st.replacing -= set(members)  # any admitted member is no longer "in flight"
        # Restart the stall clock: this world was just created and has not had a
        # chance to checkpoint yet. Survivors must tear down and re-form the
        # collective, and a replacement has to boot AND pull the dataset (~4-5 min
        # for a 17 GB train.bin) before it can take a single step. Charging that
        # against the PREVIOUS world's clock is what made every preemption look
        # like a wedged group and trigger a whole-group restart. The counter below
        # is what still catches a world that never trains at all.
        self.st.ckpt_changed_at = time.monotonic()
        self.st.epochs_without_progress += 1
        self._event(
            f"published epoch {epoch}: members {sorted(members)} master=node{master} "
            f"({doc['master_addr']}:{doc['master_port']})"
        )
        # World-size sample for the timeline (the N -> N-1 -> N staircase) +
        # who's rank-0 leader this epoch — authoritative from the decision.
        self._emit_event("epoch", epoch=epoch, world=len(members), leader=master)
        if shrinking:
            # The kill mark + baselines were captured in _terminate; a grow resets
            # the shrink markers so the next kill re-arms them.
            pass
        elif len(members) == self.cfg.node_count and self.st.shrink_baseline is not None:
            self.st.shrink_baseline = None  # back to full; full_world handled in _emit_marks

    def _terminate(self, node: int) -> None:
        # Stand in for a spot reclaim: kill the box. Membership is NOT touched
        # here — the shrink comes later, when the reducer OBSERVES the box gone.
        iid = self.node_ids[node]
        if iid in self._terminated_iids:
            return
        self.st.full_ws = self._latest_ws()
        self.st.shrink_baseline = self.st.ckpt_step
        self.st.marks.discard("shrink_resume")
        self.st.marks.discard("full_world")
        aws.terminate(iid)
        self._terminated_iids.add(iid)
        # Forget the box's identity IMMEDIATELY. `_healthy` requires a
        # registration, and st.ips is keyed by node INDEX, so a slot kept its
        # dead occupant's IP after the kill. Combined with EC2 briefly still
        # reporting `running` and a log only ~26s old (under the 90s heartbeat
        # timeout), a node we had just terminated still read as healthy — so the
        # reducer regrew the world onto a corpse and then waited for the
        # replacement that eventually filled the slot. Clearing the IP makes the
        # slot unregistered until its REPLACEMENT announces itself.
        self.st.ips.pop(node, None)
        self.profile.instance_stopped(iid)
        self.profile.mark("kill")
        self._event(f"terminated node {node} ({iid})")
        # Orchestrator-initiated stop => "killed" (vs a spot reclaim => "down").
        # attempt attaches the event to the exact box's Gantt row (node vs r1).
        attempt = self.logs.get(node, {}).get("attempt", 0)
        self._emit_event("killed", node=node, attempt=attempt, cause="scheduled-kill")
        self._downed.add((node, iid))

    def _launch_replacement(self, node: int) -> None:
        if node in self.st.replacing:
            return
        self.st.replacing.add(node)
        self._event(f"launching replacement for node {node}")
        # Do NOT wait for the dead instance to release its quota first. That was
        # an unconditional serialization — terminate, poll every 5s until AWS
        # moves it out of shutting-down (tens of seconds), only then launch —
        # paid on every replacement even when the account has ample headroom.
        # wait_vcpu_headroom already covers the case it was guarding: it returns
        # immediately when used+needed fits, and blocks only when it does not,
        # which is exactly when the dead instance releasing is what makes room.
        # Measured: 4 nodes + 1 replacement = 20 of a 64 vCPU quota. It never had
        # to wait. It matters most in the total-loss case, where every
        # replacement used to queue behind a corpse AND no survivor is training.
        aws.wait_vcpu_headroom(self.cfg.instance_vcpu_count(), self.cfg.vcpu_quota)
        self.st.ips.pop(node, None)  # force re-read of the replacement's fresh registration
        self.node_ids[node] = self._launch_node(node)
        self.profile.mark("relaunch")

    def _whole_group_restart(self, reason: str = "") -> None:
        self._event(f"whole-group restart (floor): {reason or 'unspecified'}")
        aws.delete_object(self.cfg.bucket, self.cfg.run_epoch_key(self.run_id))
        for iid in self.node_ids.values():
            aws.terminate(iid)
            self.profile.instance_stopped(iid)
        self.st = SupervisorState()
        # Same reasoning as _launch_replacement: no unconditional per-instance
        # quota wait. Here the whole fleet's worth of vCPUs is needed at once, so
        # headroom genuinely may be short — and wait_vcpu_headroom blocks exactly
        # then, and only then, instead of serializing on every corpse in turn.
        aws.wait_vcpu_headroom(
            self.cfg.node_count * self.cfg.instance_vcpu_count(), self.cfg.vcpu_quota
        )
        for node in list(self.node_ids):
            self.node_ids[node] = self._launch_node(node)

    def _execute(self, actions: list[Action]) -> None:
        for a in actions:
            if isinstance(a, TerminateNode):
                self._terminate(a.node)
            elif isinstance(a, PublishEpoch):
                if all(n in self.st.ips or self._node_ip(n) is not None for n in a.members):
                    self._publish_epoch(a.epoch, a.members)
            elif isinstance(a, LaunchReplacement):
                self._launch_replacement(a.node)
            elif isinstance(a, WholeGroupRestart):
                self._whole_group_restart(a.reason)
            elif isinstance(a, Done):
                pass

    def _emit_marks(self) -> None:
        """Observational profile marks (the ones that depend on downstream state
        crossing a threshold, not on a decision): survivors checkpointing again,
        and the world returning to full."""
        base = self.st.shrink_baseline
        if base is not None and self.st.ckpt_step > base and "shrink_resume" not in self.st.marks:
            self.profile.mark("shrink_resume")
            self.st.marks.add("shrink_resume")
        if (
            self.st.full_ws is not None
            and "full_world" not in self.st.marks
            and any(
                s.world_size == self.st.full_ws and s.step > (base or -1)
                for s in self.profile.samples
            )
        ):
            self.profile.mark("full_world")
            self.st.marks.add("full_world")

    # -- loop -------------------------------------------------------------- #
    def run(self, *, deadline_s: float) -> dict | None:
        """Drive the run to completion (metrics.json) or the deadline. Returns the
        parsed metrics, or None on timeout."""
        import sys

        # RE-ADOPT before the first tick. run() is what the systemd unit
        # re-enters after a crash, so this is the one place that sees every
        # restart. On a cold start there is no epoch document and this is a
        # no-op; on a restart it is the difference between continuing a live
        # world and rebuilding one that was training fine.
        prior = self.st.epoch
        with contextlib.suppress(Exception):  # a failed read must not block a restart
            self.st = restore_state(
                self.st, aws.get_json(self.cfg.bucket, self.cfg.run_epoch_key(self.run_id))
            )
        if self.st.epoch > prior:
            self._event(
                f"re-adopted epoch {self.st.epoch}: members {sorted(self.st.members)} "
                f"master=node{self.st.master} (restarted onto a live world, not re-forming it)"
            )
        # Re-arm the chaos schedule from S3 too. Without this a restart replays
        # every kill from zero -- the fleet survives it, but the run's chaos
        # accounting no longer matches what was scheduled, and it costs real
        # replacements.
        self._restore_schedule(time.monotonic(), time.time())

        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            self._pull_logs()
            now, wall = time.monotonic(), time.time()
            obs = self._observe(now, wall)
            # First checkpoint => training has started; arm the kill schedule clock.
            if self._train_start is None and self.st.ckpt_step >= 0 and self.st.epoch > 0:
                self._train_start = now
                self._train_start_wall = wall
                self._save_schedule(wall)
                self.profile.mark("train_start")
            # Reclaim detection: a member observed gone that we did NOT terminate
            # is a spot reclaim => emit "down" (stamped now, ~one tick after the
            # box actually vanished). A member we killed already emitted "killed".
            healthy = frozenset(n.node for n in obs.nodes if _healthy(n, self.policy))
            for node in sorted(self.st.members - healthy):
                iid = self.node_ids.get(node, "")
                if (node, iid) in self._downed:
                    continue
                self._downed.add((node, iid))
                if iid not in self._terminated_iids:
                    attempt = self.logs.get(node, {}).get("attempt", 0)
                    self._emit_event("down", node=node, attempt=attempt, cause="reclaimed")
            actions = decide(obs, self.policy)
            if any(isinstance(a, Done) for a in actions):
                self._pull_logs()
                self._event("run complete (metrics.json)")
                self._write_status(obs, wall, done=True)
                self.metrics = json.loads(aws.get_text(self.cfg.bucket, self.metrics_key))
                return self.metrics
            # Write status BEFORE executing this tick's actions: _execute mutates
            # self.logs / self.node_ids (a replacement bumps node N to a new
            # attempt + instance), and pairing those with THIS tick's older `obs`
            # would tag the fresh attempt with the dead predecessor's AWS state —
            # stamping node N·rK [dead] the instant it's born, which sticky-dead
            # then locks in forever. Observed obs + as-observed logs stay
            # consistent here; the new attempt surfaces next tick with its real
            # state. (Cost: this tick's decision events reach orchestrator.log one
            # tick later — negligible vs. a permanently-wrong dead badge.)
            self._write_status(obs, wall)
            self._execute(actions)
            self._emit_marks()
            time.sleep(self.cfg.log_stream_seconds)
        print("[supervisor] deadline reached without metrics.json", file=sys.stderr)
        return None
