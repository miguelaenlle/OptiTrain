"""The DURABLE REMOTE ORCHESTRATOR — ``spot-orchestrate orch up|status|logs|down``.

The epoch supervisor used to run on your laptop, which made a 36-hour training
run a 36-hour commitment for the laptop: sleep it, lose the wifi, close the lid,
and the training boxes' dead-man's switch (``MAX_INSTANCE_LIFETIME_SECONDS``)
eventually reaps the fleet. This moves the supervisor onto a cheap always-on
``t3.micro`` and leaves the laptop with nothing but a read-only view.

Deliberately the dumbest thing that works — no Auto Scaling Group, no
self-healing controller, no RPC layer, no custom protocol:

  * **one on-demand box** launched by ``orch up``, running the SAME experiment
    entrypoints the laptop ran, under a systemd unit that restarts on failure;
  * **S3 is the whole control plane API.** The box publishes four small
    documents under ``orchestrators/<orch_id>/`` — ``orch.json`` (what it was
    asked to run + the run it owns), ``progress.json`` (boot phases),
    ``heartbeat.json`` (liveness + live step/loss/cost), ``orchestrator.log``.
    The laptop only ever GETs them;
  * **EC2 tags are the whole discovery API.** ``orch_role=controller`` finds the
    box; ``Name=spot-train-<run_id>`` finds the fleet it is driving. No local
    state file to lose, nothing to keep in sync.

``orch up`` launches and then attaches to the existing full-screen dashboard
(``logview.run_logs`` — the same code path as ``spot-orchestrate logs``), so the
headline command both starts the run and shows it. Ctrl-C only detaches: the
viewer reads S3 and holds no authority over anything.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

from . import aws
from .config import OrchestratorConfig

# EC2 tags = discovery. The control-plane box carries both; training boxes carry
# neither, so `orch down` can never confuse the two.
ORCH_TAG = "orch"
ORCH_ROLE_TAG = "orch_role"
CONTROLLER = "controller"

# What `orch up --experiment` accepts. ``kind`` is the run_id prefix the
# experiment mints (so the agent can mint it FIRST and hand it in — that is what
# makes a restart resume rather than restart). ``resumable`` marks experiments
# that own exactly one run_id: those the agent may re-enter after a crash,
# because the trainer's one resume path picks the run's checkpoints back up.
# Sweeps (scaling-*) drive several runs and have no single resume point, so a
# crashed sweep is NOT silently redone — see run_agent.
_EXPERIMENTS: dict[str, tuple[str, bool]] = {
    "multinode": ("multinode", True),
    "multinode-shrink": ("multinode-shrink", True),
    "multinode-preempt": ("multinode-preempt", True),
    "baseline": ("baseline", True),
    "ddp": ("ddp", True),
    "spot": ("spot", False),
    "preempt": ("preempt", False),
    "ddp-preempt": ("preempt", False),
    "calibrate": ("calibrate", False),
    "scaling-experiment": ("scaling", False),
    "scaling-clean": ("scaling", False),
    "scaling-preempt": ("scaling", False),
}

# Which env var carries the run budget for each experiment — used to translate
# the operator's `--env TRAIN_BUDGET_SECONDS=...` (the knob they think in) into
# the knob the experiment actually reads, and to show "elapsed vs budget".
_BUDGET_FROM_TOTAL = frozenset({"multinode-shrink", "multinode-preempt", "preempt", "ddp-preempt"})


def experiment_names() -> list[str]:
    return sorted(_EXPERIMENTS)


def _budget_env_key(experiment: str) -> str:
    return "TRAIN_TOTAL_SECONDS" if experiment in _BUDGET_FROM_TOTAL else "BASELINE_SECONDS"


def _budget_seconds(
    cfg: OrchestratorConfig, experiment: str, env: dict[str, str] | None = None
) -> int:
    """The run's budget in seconds. ``env`` is the environment the CONTROL PLANE
    will boot with — on the laptop that differs from this process's own config
    (the operator passed `--env BASELINE_SECONDS=…`), and the recorded budget
    must be the one the run will actually use."""
    key = _budget_env_key(experiment)
    raw = (env or {}).get(key, "")
    if raw.isdigit():
        return int(raw)
    return cfg.train_total_seconds if experiment in _BUDGET_FROM_TOTAL else cfg.baseline_seconds


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(max(0.0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# --------------------------------------------------------------------------- #
# Boot progress: a pure state machine (no AWS, no printing) so `orch up` can
# show real provisioning progress and the whole sequence is table-testable.
# --------------------------------------------------------------------------- #
# Ordered, monotonic: progress only ever moves forward, so a momentarily missing
# document (S3 read-after-write on an overwritten key) can't rewind the display.
BOOT_PHASES = ("launch", "pending", "running", "provision", "clone", "install", "start", "live")

_DEAD_STATES = frozenset({"shutting-down", "terminated", "stopping", "stopped"})


@dataclass
class BootProgress:
    """Folds (instance state, progress.json, heartbeat.json, "is the run
    viewable yet") into a phase + the lines to print for new transitions.

    ``ready`` is the moment ``orch up`` flips into the dashboard: the agent has
    heartbeated a run_id AND that run has something to show (its status.json or
    a first log). ``failed`` is set if the box dies during boot — otherwise the
    laptop would wait out the whole boot timeout on a corpse."""

    phase: str = "launch"
    run_id: str = ""
    ready: bool = False
    failed: str = ""
    seen: set[str] = field(default_factory=set)

    def _index(self, phase: str) -> int:
        return BOOT_PHASES.index(phase) if phase in BOOT_PHASES else -1

    def _advance(self, phase: str, detail: str, out: list[str]) -> None:
        if self._index(phase) <= self._index(self.phase) or phase in self.seen:
            return
        self.phase = phase
        self.seen.add(phase)
        out.append(f"{phase}{f': {detail}' if detail else ''}")

    def observe(
        self,
        *,
        state: str,
        progress: dict | None,
        heartbeat: dict | None,
        run_ready: bool,
    ) -> list[str]:
        out: list[str] = []
        if state in _DEAD_STATES:
            # The control plane must never be terminated during boot; if it is,
            # say so instead of spinning (user-data crash, capacity pull, …).
            self.failed = f"instance went {state} during boot"
            return out
        if state == "pending":
            self._advance("pending", "waiting for the instance", out)
        elif state == "running":
            self._advance("pending", "waiting for the instance", out)
            self._advance("running", "booting", out)
        if progress and progress.get("phase") == "error":
            # user-data gave up (a failed clone, say) — stop waiting on a box
            # that will never heartbeat.
            self.failed = f"provisioning failed: {progress.get('detail', '')}"
            return out
        if progress:
            self._advance(str(progress.get("phase", "")), str(progress.get("detail", "")), out)
        if heartbeat and heartbeat.get("run_id"):
            self.run_id = str(heartbeat["run_id"])
            self._advance("live", f"supervisor heartbeat — run {self.run_id}", out)
        self.ready = bool(self.phase == "live" and self.run_id and run_ready)
        return out


# --------------------------------------------------------------------------- #
# Discovery (EC2 tags + the S3 documents) — read-only, shared by every command.
# --------------------------------------------------------------------------- #
def find_controllers(cfg: OrchestratorConfig, orch_id: str = "") -> list[dict]:
    """Non-terminated control-plane instances, newest first. ``orch_id`` filters
    to one. Tag-based, so it works from any machine with no local state."""
    found = aws.instances_by_tag(ORCH_ROLE_TAG, CONTROLLER)
    if orch_id:
        found = [i for i in found if i["tags"].get(ORCH_TAG) == orch_id]
    return sorted(found, key=lambda i: i["tags"].get(ORCH_TAG, ""), reverse=True)


def resolve_active(cfg: OrchestratorConfig, orch_id: str = "") -> tuple[str, dict | None]:
    """(orch_id, instance) for the control plane a bare `orch status|logs|down`
    should act on: the requested one, else the newest live one. An orch_id with
    no instance still resolves (its S3 documents outlive the box), so you can
    inspect a finished run's control plane."""
    found = find_controllers(cfg, orch_id)
    if found:
        chosen = found[0]
        if len(found) > 1 and not orch_id:
            print(
                f"[orch] {len(found)} control planes are up; showing "
                f"{chosen['tags'].get(ORCH_TAG)} (pass --id for another)",
                file=sys.stderr,
            )
        return chosen["tags"].get(ORCH_TAG, ""), chosen
    return orch_id, None


def _docs(cfg: OrchestratorConfig, orch_id: str) -> tuple[dict | None, dict | None]:
    """(orch.json, heartbeat.json) — the two documents the box publishes."""
    if not orch_id:
        return None, None
    return (
        aws.get_json(cfg.bucket, cfg.orch_state_key(orch_id)),
        aws.get_json(cfg.bucket, cfg.orch_heartbeat_key(orch_id)),
    )


def fleet_of(cfg: OrchestratorConfig, run_id: str) -> list[dict]:
    """The training boxes a run owns. ``aws.launch`` names every instance
    ``spot-train-<run_id>``, so the run id alone identifies the fleet — no
    registry, no local state."""
    if not run_id:
        return []
    return aws.instances_by_tag("Name", f"spot-train-{run_id}")


# --------------------------------------------------------------------------- #
# `orch up`
# --------------------------------------------------------------------------- #
def _detach_banner(orch_id: str, run_id: str, iid: str) -> str:
    return "\n".join(
        [
            "",
            f"[orch] detached — the run keeps going on {iid} ({orch_id}).",
            f"[orch]   run:        {run_id}",
            "[orch]   reattach:   spot-orchestrate orch logs",
            "[orch]   status:     spot-orchestrate orch status",
            "[orch]   tear down:  spot-orchestrate orch down --all",
            "[orch] Nothing was killed: the dashboard only reads S3.",
        ]
    )


def attach(
    cfg: OrchestratorConfig,
    run_id: str,
    *,
    orch_id: str = "",
    instance_id: str = "",
    node: int | None = None,
    interval: float | None = None,
    grid: bool = False,
    plain: bool = False,
) -> None:
    """Hand off to the SAME viewer ``spot-orchestrate logs`` uses — per-node
    tabs, gantt, events, everything. No rendering code lives here on purpose.

    Ctrl-C lands as a KeyboardInterrupt inside the viewer, which restores the
    terminal on its way out; we catch it and print how to get back in. The
    viewer is read-only over S3, so detaching structurally cannot affect the
    run — the banner says so because that is the fear this command exists to
    remove."""
    from . import logview

    print(
        f"[orch] attaching to {run_id} — Ctrl-C DETACHES (it cannot stop the run)",
        file=sys.stderr,
    )
    try:
        logview.run_logs(cfg, run_id, interval=interval, node=node, grid=grid, plain=plain)
    except KeyboardInterrupt:
        pass
    finally:
        print(_detach_banner(orch_id, run_id, instance_id), file=sys.stderr)


def _wait_for_boot(cfg: OrchestratorConfig, orch_id: str, iid: str) -> BootProgress:
    """Poll the instance state + the box's progress/heartbeat documents, printing
    each new phase, until the run is viewable (or the boot times out / the box
    dies). Ctrl-C here detaches exactly like it does in the dashboard."""
    prog = BootProgress()
    t0 = time.monotonic()
    last_note = t0
    while True:
        now = time.monotonic()
        try:
            state = aws.instance_state(iid)
        except Exception:  # noqa: BLE001 — DescribeInstances is eventually
            # consistent right after RunInstances: "not found" means "not
            # visible yet", never "dead".
            state = "pending"
        progress = aws.get_json(cfg.bucket, cfg.orch_progress_key(orch_id))
        heartbeat = aws.get_json(cfg.bucket, cfg.orch_heartbeat_key(orch_id))
        run_ready = _run_viewable(cfg, str((heartbeat or {}).get("run_id", "")))
        for line in prog.observe(
            state=state, progress=progress, heartbeat=heartbeat, run_ready=run_ready
        ):
            print(f"[orch] +{_fmt_dur(now - t0):>6}  {line}", file=sys.stderr)
        if prog.ready or prog.failed:
            return prog
        if now - t0 > cfg.orch_boot_timeout_seconds:
            prog.failed = f"no supervisor heartbeat within {cfg.orch_boot_timeout_seconds}s"
            return prog
        if now - last_note >= 60:
            print(
                f"[orch] +{_fmt_dur(now - t0):>6}  still {prog.phase}… "
                f"(boot log: s3://{cfg.bucket}/{cfg.orch_boot_log_key(orch_id)})",
                file=sys.stderr,
            )
            last_note = now
        time.sleep(5)


def _run_viewable(cfg: OrchestratorConfig, run_id: str) -> bool:
    """True once the dashboard would show something for this run: the
    supervisor's status.json, or any log the listing fallback can find."""
    if not run_id:
        return False
    return aws.object_exists(cfg.bucket, cfg.run_status_key(run_id)) or bool(
        aws.list_keys(cfg.bucket, cfg.run_logs_prefix(run_id))
    )


def up(
    cfg: OrchestratorConfig,
    *,
    experiment: str,
    env_overrides: dict[str, str] | None = None,
    no_attach: bool = False,
    node: int | None = None,
    interval: float | None = None,
    grid: bool = False,
    run_id: str = "",
    force: bool = False,
) -> str:
    """Launch the control plane and (unless ``no_attach``) watch it boot, then
    flip straight into the live dashboard. Returns the orch_id.

    ``run_id`` ADOPTS an existing run instead of minting a new one. systemd
    resurrects a supervisor process, but nothing resurrects the box it runs on --
    so without this, losing the control-plane instance ends the run outright,
    even though the fleet is still training and every checkpoint is in S3.

    The mechanism is already there: ``run_agent`` mints a run_id only when the
    state doc's field is empty, so seeding it is the whole change. The new box
    boots, ``adopt_fleet`` finds the boxes that never stopped training, and
    ``restore_state`` restores the epoch -- a takeover, not a rebuild.
    """
    from . import bootstrap

    if experiment not in _EXPERIMENTS:
        raise SystemExit(
            f"unknown experiment {experiment!r} — choose from: {', '.join(experiment_names())}"
        )
    cfg.require_bucket()
    aws.set_region(cfg.region)

    if run_id:
        _kind, resumable = _EXPERIMENTS[experiment]
        if not resumable:
            raise SystemExit(
                f"--run-id cannot adopt {experiment!r}: it drives several runs and has no "
                f"single resume point, so re-entering it would silently redo GPU hours."
            )
        if not run_id.startswith(_kind):
            raise SystemExit(
                f"--run-id {run_id!r} does not look like a {experiment!r} run "
                f"(expected the {_kind!r} prefix). Adopting the wrong run would point a "
                f"fresh control plane at another run's checkpoints."
            )
        if not aws.is_dry_run():
            if not aws.any_object_under(cfg.bucket, f"{cfg.run_prefix}/{run_id}/checkpoints/"):
                raise SystemExit(
                    f"no checkpoints under runs/{run_id}/checkpoints/ — nothing to adopt. "
                    f"Drop --run-id to start a fresh run."
                )
            # A finished run has its metrics.json. Re-entering would overwrite the
            # result already paid for, so make that an explicit choice.
            if aws.object_exists(cfg.bucket, cfg.run_metrics_key(run_id)) and not force:
                raise SystemExit(
                    f"{run_id} already wrote metrics.json — it finished. Re-entering would "
                    f"overwrite that result; pass --force if you really mean to extend it."
                )
        print(f"[orch] adopting existing run {run_id} (not minting a new one)", file=sys.stderr)

    overrides = dict(env_overrides or {})
    # The operator thinks in TRAIN_BUDGET_SECONDS (the trainer's knob). The
    # experiment reads BASELINE_SECONDS / TRAIN_TOTAL_SECONDS and derives the
    # trainer's budget from it, so translate rather than silently ignore.
    budget_key = _budget_env_key(experiment)
    if "TRAIN_BUDGET_SECONDS" in overrides and budget_key not in overrides:
        overrides[budget_key] = overrides["TRAIN_BUDGET_SECONDS"]
        print(
            f"[orch] TRAIN_BUDGET_SECONDS={overrides['TRAIN_BUDGET_SECONDS']} -> "
            f"{budget_key} (the knob {experiment} reads)",
            file=sys.stderr,
        )
    dropped = cfg.orch_secretish(overrides)
    if dropped:
        print(
            f"[orch] NOT relaying {', '.join(dropped)} — user-data is readable on the box; "
            "give the control plane a role instead",
            file=sys.stderr,
        )
    env = cfg.orch_relay_env(
        {
            # Forced, not inherited: without these four the box cannot even find
            # the bucket or clone the right branch.
            "SPOT_TRAIN_BUCKET": cfg.bucket,
            "AWS_REGION": cfg.region,
            "REPO_URL": cfg.repo_url,
            "REPO_BRANCH": cfg.repo_branch,
            **overrides,
        }
    )

    # Preflight the things that would otherwise fail 3 minutes into a boot.
    if not aws.is_dry_run():
        if not aws.object_exists(cfg.bucket, f"{cfg.data_prefix}/{cfg.dataset}/train.bin"):
            raise SystemExit(
                f"dataset not staged at {cfg.data_uri()} — run `spot-orchestrate stage-data` first"
            )
        for other in find_controllers(cfg):
            print(
                f"[orch] NOTE: control plane {other['tags'].get(ORCH_TAG)} ({other['id']}) is "
                "already up — this launches a second, independent one",
                file=sys.stderr,
            )
    if os.environ.get("WANDB_API_KEY"):
        # No key is relayed (see orch_relay_env), so the remote run writes
        # profile.json to S3 only. Say so rather than let the dashboard look empty.
        print(
            "[orch] NOTE: no W&B key is relayed to the box — the remote run writes "
            f"{cfg.run_profile_uri('<run_id>')} instead (`compare` reads it)",
            file=sys.stderr,
        )

    orch_id = time.strftime("orch-%Y%m%d-%H%M%S")
    print(
        f"[orch] {orch_id}: {experiment} on 1x{cfg.orch_instance_type} (on-demand) "
        f"in {cfg.region}, branch {cfg.repo_branch}",
        file=sys.stderr,
    )
    for k in sorted(env):
        print(f"[orch]   env {k}={env[k]}", file=sys.stderr)

    ud = bootstrap.build_orchestrator_user_data(
        cfg, orch_id=orch_id, experiment=experiment, env=env
    )
    ami = aws.resolve_ami(cfg.orch_ami_id, cfg.orch_ami_name_filter)
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)
    try:
        iid = aws.launch(
            ami_id=ami,
            instance_type=cfg.orch_instance_type,
            profile_name=cfg.orch_instance_profile,
            security_group_id=sg_id,
            user_data=ud,
            market="on-demand",  # NEVER spot: nobody replaces the replacer
            run_id=orch_id,
            key_name=cfg.key_name,
            extra_tags={ORCH_TAG: orch_id, ORCH_ROLE_TAG: CONTROLLER, "experiment": experiment},
        )
    except Exception as e:  # noqa: BLE001 — turn the one predictable AWS error into advice
        if cfg.orch_instance_profile in str(e) or "IamInstanceProfile" in str(e):
            raise SystemExit(
                f"{e}\n\nThe control plane's instance profile "
                f"({cfg.orch_instance_profile}) does not exist yet — run "
                "`spot-orchestrate setup` once to create it (it is new in this "
                "release; see docs/iam/orchestrator-policy.json)."
            ) from e
        raise
    # Seed the state doc from here so `orch status` is useful before the box has
    # finished booting; the agent takes over ownership of this key.
    aws.put_text(
        cfg.bucket,
        cfg.orch_state_key(orch_id),
        json.dumps(
            {
                "version": 1,
                "orch_id": orch_id,
                "experiment": experiment,
                "instance_id": iid,
                "instance_type": cfg.orch_instance_type,
                "hourly_usd": cfg.orch_hourly_usd(),
                "region": cfg.region,
                "repo_branch": cfg.repo_branch,
                "budget_s": _budget_seconds(cfg, experiment, env),
                "created_at": time.time(),
                # Seeding this is the ENTIRE adoption mechanism: run_agent mints
                # a run_id only when it finds this field empty.
                "run_id": run_id,
                "attempts": 0,
                "env": env,
            },
            indent=2,
        ),
    )
    print(f"[orch] instance {iid} — your laptop is now optional", file=sys.stderr)
    print(f"[orch]   status:    spot-orchestrate orch status --id {orch_id}", file=sys.stderr)
    print(f"[orch]   attach:    spot-orchestrate orch logs --id {orch_id}", file=sys.stderr)
    print(f"[orch]   tear down: spot-orchestrate orch down --id {orch_id} --all", file=sys.stderr)

    if aws.is_dry_run():
        print("[orch] dry-run: skipping boot wait + attach", file=sys.stderr)
        return orch_id
    if no_attach:
        return orch_id

    try:
        prog = _wait_for_boot(cfg, orch_id, iid)
    except KeyboardInterrupt:
        print(_detach_banner(orch_id, "(not started yet)", iid), file=sys.stderr)
        return orch_id
    if prog.failed:
        print(f"[orch] {prog.failed}", file=sys.stderr)
        print(
            f"[orch] boot log: s3://{cfg.bucket}/{cfg.orch_boot_log_key(orch_id)}\n"
            f"[orch] run log:  s3://{cfg.bucket}/{cfg.orch_log_key(orch_id)}\n"
            f"[orch] the box was NOT terminated — inspect it, then "
            f"`spot-orchestrate orch down --id {orch_id}`",
            file=sys.stderr,
        )
        return orch_id
    attach(
        cfg,
        prog.run_id,
        orch_id=orch_id,
        instance_id=iid,
        node=node,
        interval=interval,
        grid=grid,
    )
    return orch_id


# --------------------------------------------------------------------------- #
# `orch status`
# --------------------------------------------------------------------------- #
def format_status(
    *,
    orch_id: str,
    instance: dict | None,
    state: dict | None,
    heartbeat: dict | None,
    run_status: dict | None,
    now: float,
    stale_after: float = 60.0,
) -> list[str]:
    """One screen of status, as lines. PURE: every field comes from documents
    the caller fetched, so the extraction is table-testable and the command is
    just fetch + print."""
    state = state or {}
    hb = heartbeat or {}
    lines: list[str] = []

    if instance is not None:
        lines.append(
            f"[orch] {orch_id or '?'}  {instance['id']}  {instance['type']}  "
            f"[{instance['state']}]"
        )
    elif state:
        lines.append(f"[orch] {orch_id}  {state.get('instance_id', '?')}  [gone]")
    else:
        return [f"[orch] no control plane found{f' for {orch_id}' if orch_id else ''}"]

    if hb:
        age = now - float(hb.get("at", 0.0))
        health = "healthy" if age <= stale_after else f"STALE (> {int(stale_after)}s)"
        lines.append(
            f"[orch] heartbeat {_fmt_dur(age)} ago ({health})  attempt {hb.get('attempt', '?')}"
            f"  pid {hb.get('pid', '?')}"
        )
        if hb.get("error"):
            lines.append(f"[orch] last error: {hb['error']}")
    else:
        lines.append("[orch] heartbeat: none yet (box still provisioning?)")

    experiment = hb.get("experiment") or state.get("experiment", "?")
    run_id = hb.get("run_id") or state.get("run_id", "")
    lines.append(f"[orch] experiment={experiment}  run={run_id or '(not started)'}")
    if hb.get("done"):
        # The box deliberately stays up after the run so the log and profile are
        # still there to read; nothing else will stop the meter.
        lines.append("[orch] RUN FINISHED — `spot-orchestrate orch down --all` to stop billing")

    if run_status:
        members = run_status.get("members") or []
        lines.append(
            f"[orch] epoch {run_status.get('epoch', '?')}  world {len(members)}"
            f"  ckpt step {run_status.get('ckpt_step', '?')}"
            f"  {'DONE' if run_status.get('done') else 'training'}"
        )
    if hb.get("step") is not None:
        ws = f" ws {hb['world_size']}" if hb.get("world_size") else ""
        lines.append(f"[orch] step {hb['step']}  loss {hb.get('loss')}{ws}")

    budget = hb.get("budget_s") or state.get("budget_s")
    elapsed = hb.get("elapsed_s")
    if elapsed is None and state.get("created_at"):
        elapsed = now - float(state["created_at"])
    if elapsed is not None:
        pct = f" ({100 * elapsed / budget:.0f}%)" if budget else ""
        lines.append(
            f"[orch] elapsed {_fmt_dur(elapsed)} of {_fmt_dur(budget)} budget{pct} "
            "(wall clock; training time excludes boot/downtime)"
        )

    # Cost: the agent publishes the run ledger's total, which INCLUDES this
    # control-plane box (registered via on_profile) so the number is honest.
    # Before the agent starts, fall back to the box's own accrual.
    if hb.get("cost_usd") is not None:
        cp = hb.get("cost_control_plane_usd")
        extra = f" (control plane ${cp:.4f})" if cp is not None else ""
        lines.append(f"[orch] cost so far ${float(hb['cost_usd']):.4f}{extra}")
    elif state.get("hourly_usd") and state.get("created_at"):
        own = (now - float(state["created_at"])) * float(state["hourly_usd"]) / 3600.0
        lines.append(f"[orch] cost so far ${own:.4f} (control plane only)")

    lines.append("[orch] attach:  spot-orchestrate orch logs")
    return lines


def status(cfg: OrchestratorConfig, orch_id: str = "") -> None:
    cfg.require_bucket()
    aws.set_region(cfg.region)
    orch_id, instance = resolve_active(cfg, orch_id)
    state, heartbeat = _docs(cfg, orch_id)
    run_id = (heartbeat or {}).get("run_id") or (state or {}).get("run_id", "")
    run_status = aws.get_json(cfg.bucket, cfg.run_status_key(run_id)) if run_id else None
    for line in format_status(
        orch_id=orch_id,
        instance=instance,
        state=state,
        heartbeat=heartbeat,
        run_status=run_status,
        now=time.time(),
        stale_after=cfg.orch_stale_seconds,
    ):
        print(line)


# --------------------------------------------------------------------------- #
# `orch logs`
# --------------------------------------------------------------------------- #
def logs(
    cfg: OrchestratorConfig,
    *,
    orch_id: str = "",
    run_id: str = "",
    node: int | None = None,
    interval: float | None = None,
    grid: bool = False,
    plain: bool = False,
) -> None:
    """Reattach without remembering the run id: resolve the active control
    plane, ask it (via its heartbeat) which run it owns, then hand off to the
    shared viewer."""
    cfg.require_bucket()
    aws.set_region(cfg.region)
    instance = None
    if not run_id:
        orch_id, instance = resolve_active(cfg, orch_id)
        state, heartbeat = _docs(cfg, orch_id)
        run_id = (heartbeat or {}).get("run_id") or (state or {}).get("run_id", "")
    if not run_id:
        raise SystemExit(
            "no active run to attach to — `spot-orchestrate orch status` to see why "
            "(the control plane may still be provisioning), or pass --run <run_id>"
        )
    attach(
        cfg,
        run_id,
        orch_id=orch_id,
        instance_id=(instance or {}).get("id", ""),
        node=node,
        interval=interval,
        grid=grid,
        plain=plain,
    )


# --------------------------------------------------------------------------- #
# `orch down`
# --------------------------------------------------------------------------- #
def down(
    cfg: OrchestratorConfig, *, orch_id: str = "", all_: bool = False, yes: bool = False
) -> None:
    """Terminate the control plane; with ``--all`` the training fleet it drives
    too. Prints the kill list BEFORE killing anything, and is a safe no-op when
    nothing is up."""
    cfg.require_bucket()
    aws.set_region(cfg.region)
    controllers = find_controllers(cfg, orch_id)
    orch_ids = [c["tags"].get(ORCH_TAG, "") for c in controllers]
    fleet: list[dict] = []
    run_ids: list[str] = []
    for oid in orch_ids or ([orch_id] if orch_id else []):
        state, heartbeat = _docs(cfg, oid)
        rid = (heartbeat or {}).get("run_id") or (state or {}).get("run_id", "")
        if rid:
            run_ids.append(rid)
            fleet.extend(fleet_of(cfg, rid))

    if not controllers and not fleet:
        print("[orch] nothing to terminate")
        return

    print("[orch] plan:")
    for c in controllers:
        tag = c["tags"].get(ORCH_TAG, "")
        print(f"[orch]   control plane {c['id']} ({c['type']}, {c['state']}) {tag}")
    if all_:
        for i in fleet:
            name = i["tags"].get("Name", "")
            print(f"[orch]   training box  {i['id']} ({i['type']}, {i['state']}) {name}")
        if not fleet:
            print(
                "[orch]   (no training boxes found for " + (", ".join(run_ids) or "this run") + ")"
            )
    elif fleet:
        # Leaving a fleet running without its supervisor is the expensive
        # mistake this warning exists to prevent.
        print(
            f"[orch]   NOT terminating {len(fleet)} training box(es) for "
            f"{', '.join(run_ids)} — they keep billing with no supervisor.\n"
            "[orch]   Re-run with --all to terminate them too.",
            file=sys.stderr,
        )

    victims = [c["id"] for c in controllers] + ([i["id"] for i in fleet] if all_ else [])
    if not victims:
        print("[orch] nothing to terminate")
        return
    # Confirm interactively (destructive + billable-adjacent); scripted callers
    # and --dry-run go straight through.
    interactive = not yes and not aws.is_dry_run() and sys.stdin.isatty()
    if interactive and input(f"[orch] terminate {len(victims)}? [y/N] ").strip().lower() != "y":
        print("[orch] aborted — nothing terminated")
        return
    for iid in victims:
        aws.terminate(iid)
    print(f"[orch] terminated {len(victims)} instance(s)")


# --------------------------------------------------------------------------- #
# The ON-BOX agent (`orch _agent`) — what systemd actually runs.
# --------------------------------------------------------------------------- #
def _imds(path: str) -> str:
    """One IMDSv2 metadata field, or "" — stdlib only, short timeouts. Used for
    the box's own instance id/AZ so its cost row is real (and so a restarted
    agent can identify itself without being told)."""
    import urllib.request

    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
        )
        token = urllib.request.urlopen(req, timeout=2).read().decode()  # noqa: S310
        req = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=2).read().decode()  # noqa: S310
    except Exception:  # noqa: BLE001 — no IMDS (tests/local) => unknown, not fatal
        return ""


def _invoke_experiment(cfg: OrchestratorConfig, experiment: str, *, run_id: str, on_profile):
    """Call the same experiment entrypoint the laptop calls. Resumable
    experiments take the agent's run_id + ledger hook; the sweeps take neither
    (they mint several runs of their own)."""
    from . import experiments as ex

    kw = {"run_id": run_id, "on_profile": on_profile}
    table = {
        "multinode": lambda: ex.run_multinode(cfg, **kw),
        "multinode-shrink": lambda: ex.run_multinode_shrink(cfg, **kw),
        "multinode-preempt": lambda: ex.run_multinode_preempt(cfg, **kw),
        "baseline": lambda: ex.run_baseline(cfg, **kw),
        "ddp": lambda: ex.run_ddp(cfg, **kw),
        "spot": lambda: ex.run_spot(cfg),
        "preempt": lambda: ex.run_preempt(cfg),
        "ddp-preempt": lambda: ex.run_preempt(cfg, ddp=True),
        "calibrate": lambda: ex.run_calibrate(cfg),
        "scaling-experiment": lambda: ex.run_scaling_experiment(cfg),
        "scaling-clean": lambda: ex.run_scaling_clean(cfg),
        "scaling-preempt": lambda: ex.run_scaling_preempt(cfg),
    }
    return table[experiment]()


def _reap_orphans(cfg: OrchestratorConfig, run_id: str) -> None:
    """Terminate training boxes that are NOT in the current epoch's membership.

    This used to terminate the WHOLE fleet on every restarted attempt, on the
    reasoning that the boxes are polling an epoch.json nobody advances any more.
    That is true only of boxes outside the published membership. A member keeps
    TRAINING while the supervisor is gone -- sidecars run static torchrun against
    the last published epoch doc, and the supervisor's authority lives in S3, not
    in its memory -- so a supervisor crash is survivable at zero cost, and the
    old behaviour turned it into a full fleet rebuild: ~4 minutes of 8-node boot
    plus every step since the last checkpoint, every single restart.

    Since a supervisor exception is the most likely failure on a long run (an
    unhandled InsufficientInstanceCapacity from a replacement launch is enough),
    "reboot the control plane" has to be cheap. Now it is: members are re-adopted
    and only genuine orphans are reaped.
    """
    fleet = fleet_of(cfg, run_id)
    if not fleet:
        return
    # The epoch doc is the authority on membership. Its `members` are
    # [{node, ip, rank}] and carry no instance id, so match on PRIVATE IP --
    # the same value the box registered and the same one torchrun dials.
    doc = aws.get_json(cfg.bucket, cfg.run_epoch_key(run_id)) or {}
    member_ips = {m.get("ip") for m in (doc.get("members") or []) if m.get("ip")}
    # No epoch published yet => nothing has been adopted, so every box is an
    # orphan and the old whole-fleet behaviour is exactly right.
    adopted = [i for i in fleet if i.get("private_ip") and i["private_ip"] in member_ips]
    orphans = [i for i in fleet if i not in adopted]
    if adopted:
        print(
            f"[agent] re-adopting {len(adopted)} box(es) still in epoch "
            f"{doc.get('epoch', '?')} — they kept training while the supervisor was gone",
            flush=True,
        )
    if not orphans:
        return
    print(f"[agent] reaping {len(orphans)} orphaned box(es) from the previous attempt", flush=True)
    for inst in orphans:
        aws.terminate(inst["id"])
    for inst in orphans:
        try:
            aws.wait_quota_released(inst["id"])
        except TimeoutError as e:  # a slow shutdown must not block the restart
            print(f"[agent] {e}", flush=True)


def run_agent(cfg: OrchestratorConfig, *, orch_id: str, experiment: str) -> int:
    """The box-side entrypoint (``ExecStart``). Owns exactly three things beyond
    calling the experiment: the state document (so a systemd restart RESUMES the
    same run instead of starting a fresh one), the heartbeat (liveness + live
    step/loss/cost for ``orch status``), and registering itself in the run's cost
    ledger (so a 36h run's reported cost includes its control plane).

    Returns a process exit code: 0 = done (systemd stays stopped), 1 = crashed
    (systemd restarts it after RestartSec and the run resumes)."""
    import threading
    import traceback

    aws.set_region(cfg.region)
    cfg.require_bucket()
    if experiment not in _EXPERIMENTS:
        raise SystemExit(f"unknown experiment {experiment!r} — {', '.join(experiment_names())}")
    kind, resumable = _EXPERIMENTS[experiment]

    state = aws.get_json(cfg.bucket, cfg.orch_state_key(orch_id)) or {
        "version": 1,
        "orch_id": orch_id,
        "experiment": experiment,
        "created_at": time.time(),
        "attempts": 0,
        "run_id": "",
    }
    attempt = int(state.get("attempts", 0)) + 1
    state["attempts"] = attempt
    state["experiment"] = experiment
    instance_id = state.get("instance_id") or _imds("instance-id")
    az = state.get("az") or _imds("placement/availability-zone")
    state["instance_id"], state["az"] = instance_id, az
    state.setdefault("instance_type", cfg.orch_instance_type)
    state.setdefault("hourly_usd", cfg.orch_hourly_usd())
    state["budget_s"] = _budget_seconds(cfg, experiment)
    # Mint the run id HERE, not inside the experiment: owning it is what lets a
    # restarted agent point the next attempt at the same checkpoints.
    if resumable and not state.get("run_id"):
        state["run_id"] = f"{kind}-{int(time.time())}"
    run_id = state.get("run_id", "")
    started_at = float(state.get("started_at") or time.time())
    state["started_at"] = started_at
    aws.put_text(cfg.bucket, cfg.orch_state_key(orch_id), json.dumps(state, indent=2))

    print(
        f"[agent] {orch_id} attempt {attempt}: {experiment} run={run_id or '(sweep)'} "
        f"on {instance_id or '?'} ({az or '?'})",
        flush=True,
    )
    if attempt > 1:
        if not resumable:
            # A sweep drives several runs and has no single resume point.
            # Silently redoing hours of GPU time would be worse than stopping.
            print(
                f"[agent] {experiment} is not resumable and this is attempt {attempt} — "
                "NOT restarting it. Inspect the log, then re-run `orch up`.",
                flush=True,
            )
            return 0
        if aws.object_exists(cfg.bucket, cfg.run_metrics_key(run_id)):
            print(f"[agent] {run_id} already wrote metrics.json — nothing to do", flush=True)
            return 0
        _reap_orphans(cfg, run_id)

    holder: dict[str, object] = {"profile": None, "error": "", "done": False}

    def on_profile(profile) -> None:
        """Register the control plane in the run's cost ledger (its billing
        started when the box booted, not when the run did) and hand the profile
        to the heartbeat thread so `orch status` can report live step/loss/$."""
        profile.instance_started(
            instance_id or "control-plane",
            "on-demand",
            az,
            state.get("hourly_usd"),
            float(state.get("created_at", started_at)),
        )
        holder["profile"] = profile

    def publish() -> None:
        profile = holder["profile"]
        samples = getattr(profile, "samples", None)
        sample = samples[-1] if samples else None
        doc = {
            "version": 1,
            "orch_id": orch_id,
            "at": time.time(),
            "pid": os.getpid(),
            "attempt": attempt,
            "experiment": experiment,
            "run_id": run_id,
            "instance_id": instance_id,
            "instance_type": state.get("instance_type"),
            "started_at": started_at,
            "budget_s": state["budget_s"],
            "elapsed_s": round(time.time() - started_at, 1),
            "step": getattr(sample, "step", None),
            "loss": getattr(sample, "loss", None),
            "world_size": getattr(sample, "world_size", None),
            "cost_usd": round(profile.cost_now(), 6) if profile is not None else None,
            "cost_control_plane_usd": _control_plane_cost(state, time.time()),
            "error": holder["error"],
            "done": holder["done"],
        }
        # quiet: a doc every few seconds for 36h would otherwise dominate the
        # orchestrator's own log (which is capped, see the relay).
        aws.put_text(cfg.bucket, cfg.orch_heartbeat_key(orch_id), json.dumps(doc), quiet=True)

    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                publish()
            except Exception as e:  # noqa: BLE001 — a missed heartbeat must never end the run
                print(f"[agent] heartbeat failed (non-fatal): {e}", flush=True)
            stop.wait(cfg.orch_heartbeat_seconds)

    hb = threading.Thread(target=loop, name="orch-heartbeat", daemon=True)
    hb.start()
    try:
        _invoke_experiment(cfg, experiment, run_id=run_id, on_profile=on_profile)
        print(f"[agent] {experiment} finished — the box stays up for `orch down`", flush=True)
        rc = 0
    except SystemExit as e:  # a config/preflight refusal: restarting won't help
        holder["error"] = f"{e}"
        print(f"[agent] refusing to continue: {e}", flush=True)
        rc = 0
    except Exception:
        holder["error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
        traceback.print_exc()
        rc = 1  # systemd restarts us; the next attempt resumes this run_id
    finally:
        stop.set()
        hb.join(timeout=5)
        holder["done"] = True
        # Final heartbeat from the main thread: `orch status` shows the terminal
        # state (done / error) immediately instead of aging into STALE.
        with contextlib.suppress(Exception):
            publish()
    return rc


def _control_plane_cost(state: dict, now: float) -> float | None:
    rate = state.get("hourly_usd")
    if not rate or not state.get("created_at"):
        return None
    return round((now - float(state["created_at"])) * float(rate) / 3600.0, 6)
