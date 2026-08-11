"""Per-box sidecar: obey the orchestrator's epoch document.

Runs on every training box (launched by the boot script) and, unchanged, in the
localhost E2E test. It is the node half of the epoch protocol: poll
``<run_uri>/epoch.json``, and whenever the current epoch names this node, run
STATIC torchrun for exactly that membership; when the epoch advances (a peer
died or a replacement joined), kill the local torchrun and relaunch for the new
epoch. No rendezvous negotiation happens here — the orchestrator already decided
who is in the group; this process just executes the decision and reports via its
(S3-synced) stdout.

Deliberately dependency-light: stdlib + ``spot_train.s3_store`` only, so the same
code path that runs on the DLAMI runs against local directories in tests.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

from spot_train import events, s3_store

POLL_SECONDS = 3
# Total time to wait with no epoch naming us before giving up (box stays up for
# the orchestrator's whole-group-restart watchdog). Generous: a replacement can
# sit idle a long time between registering and being admitted.
IDLE_BUDGET_SECONDS = 30 * 60
# Consecutive torchrun crashes within ONE epoch before this box gives up. A peer
# death crashes our collective too, so a crash is normal and relaunching is the
# correct response — but only a few times. Past that the fault is deterministic
# (the world simply cannot start with this config) and retrying just re-pays the
# boot + dataset download on every node, forever, at full fleet cost. Resets on
# every epoch change, because a new membership deserves a fresh verdict.
MAX_EPOCH_CRASHES = 5
# Ceiling on the exponential backoff between relaunches. Kept small: the common
# crash IS a preemption, and this delay lands directly in recovery time, which is
# the headline number the whole spot thesis is measured on.
RELAUNCH_BACKOFF_CAP = 30.0


def _log(msg: str) -> None:
    print(f"[epoch] {msg}", file=sys.stderr, flush=True)


def _join(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name


def _imds(path: str) -> str | None:
    """Best-effort IMDSv2 read (private IP / instance id). None off-EC2."""
    try:
        tok = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(tok, timeout=1).read().decode()
        req = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:  # noqa: BLE001 — not on EC2, or IMDS disabled
        return None


def register(
    run_uri: str, node_index: int, *, ip: str | None = None, instance_id: str | None = None
) -> str:
    """Announce this box: write ``nodes/node<i>.json`` {ip, instance_id}. The IP
    is what the orchestrator puts in the epoch doc as this rank's address; the
    write is also the join request (admission = being named in a published
    epoch). Returns the ip used."""
    ip = ip or os.environ.get("E2E_NODE_IP") or _imds("local-ipv4") or "127.0.0.1"
    instance_id = instance_id or _imds("instance-id") or "unknown"
    s3_store.put_bytes(
        json.dumps({"ip": ip, "instance_id": instance_id}).encode(),
        _join(run_uri, f"nodes/node{node_index}.json"),
    )
    _log(f"node {node_index}: registered ip={ip} instance={instance_id}")
    # First lifecycle event: the box booted and the sidecar is alive (provisioning
    # — clone/pip/dataset done). The trainer emits "training" once it clears
    # checkpoint restore and enters the loop.
    events.emit("provisioning", by="sidecar", node=node_index, cause="boot")
    return ip


def default_launch(
    epoch: int, rank: int, node_count: int, master_addr: str, master_port: int
) -> subprocess.Popen:
    """Static torchrun for one epoch — the plumbing proven pre-elastic, with the
    membership fixed by the orchestrator (no --rdzv, no --nnodes range). Its own
    session so the whole worker tree can be killed on an epoch change. The worker
    module is ``SIDECAR_TRAIN_MODULE`` (default ``spot_train.train``) — the seam
    lets the localhost E2E point torchrun at a dummy worker instead."""
    module = os.environ.get("SIDECAR_TRAIN_MODULE", "spot_train.train")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={node_count}",
        f"--nproc_per_node={os.environ.get('NPROC_PER_NODE', 'gpu')}",
        f"--node_rank={rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--max-restarts=0",
        "-m",
        module,
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def kill_tree(proc: subprocess.Popen) -> None:
    """SIGKILL torchrun AND its detached worker sessions. torchrun starts each
    worker with start_new_session=True, so killing only torchrun's group would
    orphan the workers (which keep holding the NCCL sockets). Kill children by
    pgrep, then torchrun's group, then a belt-and-braces pkill."""
    if proc.poll() is not None:
        return
    try:
        kids = subprocess.run(
            ["pgrep", "-P", str(proc.pid)], capture_output=True, text=True
        ).stdout.split()
    except Exception:  # noqa: BLE001 — pgrep missing; fall through to group kill
        kids = []
    for pid in [proc.pid, *[int(k) for k in kids]]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    subprocess.run(["pkill", "-9", "-f", "spot_train.train"], check=False)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)
    # Wait for the worker processes to ACTUALLY exit before we return — a SIGKILL'd
    # trainer's ~19GB of GPU memory isn't freed until the process is reaped, and the
    # sidecar relaunches torchrun within one poll. Relaunching while a dead-but-not-
    # yet-reaped worker still holds the GPU makes the fresh trainer OOM. Bounded so a
    # wedged worker can't hang the sidecar forever.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        left = subprocess.run(
            ["pgrep", "-f", "spot_train.train"], capture_output=True, text=True
        ).stdout.split()
        if not left:
            return
        time.sleep(0.5)
    _log("WARNING: spot_train.train workers still alive after kill — GPU may not be free")


def _my_membership(doc: dict, node_index: int) -> tuple[int, int, str, int] | None:
    """(rank, node_count, master_addr, master_port) if this node is in the epoch,
    else None."""
    for m in doc.get("members", []):
        if m["node"] == node_index:
            return (m["rank"], doc["node_count"], doc["master_addr"], doc["master_port"])
    return None


def run(
    run_uri: str,
    node_index: int,
    *,
    launch=default_launch,
    idle_budget: float = IDLE_BUDGET_SECONDS,
) -> int:
    """The state machine. Returns the process exit code (0 = the run finished, a
    metrics.json appeared; nonzero = idle budget exhausted, box left up)."""
    epoch_uri = _join(run_uri, "epoch.json")
    metrics_uri = _join(run_uri, "metrics.json")
    # Passed to the trainer (via torchrun's inherited env) so its lifecycle events
    # are attributed to this node and epoch — see spot_train.train.
    os.environ["SPOT_NODE_INDEX"] = str(node_index)
    proc: subprocess.Popen | None = None
    running_epoch = -1  # the epoch `proc` was launched for (reset on crash)
    member_epoch = -1  # the last epoch we were admitted to (PERSISTS across crashes,
    # so "realized the world changed" still fires when our torchrun crashed on the
    # peer's death BEFORE the shrunk epoch was published — the common preempt path)
    idle_deadline = time.monotonic() + idle_budget
    # Consecutive crashes WITHIN one epoch, and the earliest time we may relaunch.
    # Relaunching a crashed torchrun is right for a transient fault, but a
    # DETERMINISTIC one (bad env, a config the world can never start with) turns
    # into a hot retry loop: every node rebuilds, re-downloads, re-fails, every
    # POLL_SECONDS. That is what an 8-node world does at ~$8/hr while making no
    # progress — the failure mode that burned an 8-node leg through 6 identical
    # relaunches before it was killed by hand. Back off, then give up and let the
    # box exit so the orchestrator's own teardown reclaims it.
    epoch_crashes = 0
    # The epoch those crashes belong to. Tracked SEPARATELY from running_epoch,
    # which the crash path resets to -1 — using that as "the epoch changed" would
    # make every crash look like a fresh membership and silently zero the counter,
    # restoring the infinite loop this guard exists to stop.
    crash_epoch = -1
    relaunch_after = 0.0
    try:
        while True:
            if s3_store.read_bytes(metrics_uri) is not None:
                _log(f"node {node_index}: metrics.json present — run complete")
                return 0

            raw = s3_store.read_bytes(epoch_uri)
            doc = None
            if raw is not None:
                try:
                    doc = json.loads(raw)
                except ValueError:
                    doc = None

            mine = _my_membership(doc, node_index) if doc else None
            epoch = doc.get("epoch") if doc else None

            if mine is not None:
                rank, node_count, master_addr, master_port = mine
                epoch_changed = epoch != running_epoch
                crashed = proc is not None and proc.poll() is not None
                if member_epoch != -1 and epoch != member_epoch:
                    # REALIZED the world changed: we've read a new epoch doc (vs.
                    # the last one we were admitted to). Distinct from — and
                    # preceding — provisioning; the gap is the teardown of the old
                    # collective. Fires whether or not our torchrun already crashed.
                    events.emit(
                        "reconfiguring",
                        by="sidecar",
                        node=node_index,
                        epoch=epoch,
                        world=node_count,
                        cause=f"epoch {member_epoch}->{epoch}",
                    )
                member_epoch = epoch
                if epoch_changed and proc is not None:
                    _log(f"node {node_index}: killed for epoch {epoch}")
                    kill_tree(proc)
                    proc = None
                if crash_epoch != -1 and epoch != crash_epoch:
                    # A genuinely NEW epoch: the world changed, so earlier crashes
                    # say nothing about whether THIS membership can run (a world
                    # that fails at 8 nodes may be exactly what succeeds at 7).
                    epoch_crashes = 0
                    crash_epoch = -1
                    relaunch_after = 0.0
                if (proc is None or epoch_changed) and time.monotonic() >= relaunch_after:
                    if epoch_crashes >= MAX_EPOCH_CRASHES:
                        _log(
                            f"node {node_index}: {epoch_crashes} consecutive torchrun crashes in "
                            f"epoch {epoch} — giving up so this box stops burning; "
                            "the orchestrator reaps the fleet"
                        )
                        events.emit(
                            "failed",
                            by="sidecar",
                            node=node_index,
                            epoch=epoch,
                            cause=f"crash-loop:{epoch_crashes}",
                        )
                        return 2
                    _log(
                        f"node {node_index}: entering epoch {epoch} as rank {rank}/{node_count} "
                        f"(master {master_addr}:{master_port})"
                    )
                    # Re-rendezvous / restore window: this node is provisioning at
                    # the new world size until its trainer emits "training". Expose
                    # the epoch so the trainer stamps its events with it.
                    os.environ["SPOT_EPOCH"] = str(epoch)
                    events.emit(
                        "provisioning",
                        by="sidecar",
                        node=node_index,
                        epoch=epoch,
                        world=node_count,
                        cause="launching",
                    )
                    proc = launch(epoch, rank, node_count, master_addr, master_port)
                    running_epoch = epoch
                elif crashed:
                    # torchrun died on its own (a peer's death crashed our
                    # collective) — the supervisor will publish the next epoch;
                    # drop the corpse and let the epoch-change branch relaunch.
                    code = proc.poll()
                    epoch_crashes += 1
                    crash_epoch = epoch
                    # Exponential backoff, capped. A peer death (the expected
                    # cause) is resolved by the next epoch doc within a poll or
                    # two, so the first retry stays fast and only a genuine crash
                    # loop slows down — recovery time is what preemption costs us,
                    # and we must not inflate it to guard against a rarer bug.
                    delay = min(RELAUNCH_BACKOFF_CAP, POLL_SECONDS * (2 ** (epoch_crashes - 1)))
                    relaunch_after = time.monotonic() + delay
                    _log(
                        f"node {node_index}: torchrun exited {code} in epoch {epoch} "
                        f"(crash {epoch_crashes}/{MAX_EPOCH_CRASHES}, retry in {delay:.0f}s)"
                    )
                    events.emit(
                        "provisioning",
                        by="sidecar",
                        node=node_index,
                        epoch=epoch,
                        cause=f"torchrun-exit:{code}",
                    )
                    proc = None
                    running_epoch = -1
                idle_deadline = time.monotonic() + idle_budget  # being a member resets idle
            else:
                # Not in the current epoch: a replacement awaiting admission, or a
                # node the group shrank away from. Stop any stale torchrun, idle.
                if proc is not None:
                    _log(f"node {node_index}: excluded from epoch {epoch} — stopping")
                    kill_tree(proc)
                    proc = None
                    running_epoch = -1
                if time.monotonic() > idle_deadline:
                    _log(f"node {node_index}: idle budget exhausted — leaving box up for watchdog")
                    return 1
            time.sleep(POLL_SECONDS)
    finally:
        if proc is not None:
            kill_tree(proc)


def prewarm_dataset(node_index: int) -> None:
    """Pull the corpus BEFORE announcing this box, so registration means
    "I can train" rather than "this instance exists".

    Measured on a 4-node preemption: the supervisor admitted a replacement the
    moment it appeared, at t+448s, but that box could not take a step until
    t+662s because it was still pulling a 17 GB train.bin — the survivors tore
    down their world-3 collective and idled 214s (3 x 214 = 642 node-seconds)
    waiting for it. Doing the pull first costs nothing extra: it is the same
    download, moved ahead of the announcement instead of behind it.

    Best-effort. A failure here must NOT stop the box: the trainer calls the
    same idempotent function and will retry, and a node that registers slightly
    early is the behaviour we already had."""
    data_uri = os.environ.get("DATA_URI", "")
    data_dir = os.environ.get("DATA_LOCAL_DIR", "")
    if not data_uri or not data_dir:
        return
    started = time.monotonic()
    try:
        from spot_train.data import ensure_dataset

        ensure_dataset(data_dir, data_uri)
        _log(
            f"node {node_index}: dataset ready in {time.monotonic() - started:.1f}s (pre-register)"
        )
    except Exception as exc:  # noqa: BLE001 — the trainer retries; never block boot
        _log(f"node {node_index}: dataset prewarm failed ({exc}) — trainer will retry")


def main() -> None:
    ap = argparse.ArgumentParser(prog="orchestrator.sidecar")
    ap.add_argument("--run-uri", required=True, help="s3://bucket/runs/<run_id> (or a local dir)")
    ap.add_argument("--node-index", type=int, required=True)
    args = ap.parse_args()
    prewarm_dataset(args.node_index)
    register(args.run_uri, args.node_index)
    sys.exit(run(args.run_uri, args.node_index))


if __name__ == "__main__":
    main()
