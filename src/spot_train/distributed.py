"""Single-node DDP context for torchrun — the ONLY DDP-specific module.

Dormant unless ``RANK`` is set (i.e. launched via torchrun), so the single-process
training path is byte-identical to Phase 1a. On CPU we use the ``gloo`` backend
(no GPU needed); on a real GPU box it's ``nccl``. torchrun ``--standalone`` sets
MASTER_ADDR/MASTER_PORT/RANK/LOCAL_RANK/WORLD_SIZE for us.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class Dist:
    enabled: bool  # True only under torchrun (RANK set)
    rank: int
    local_rank: int
    world_size: int
    master: bool  # rank 0 — does all logging / checkpointing / metrics
    device: str


def init(device: str) -> Dist:
    """Detect torchrun and (if present) join the process group. Returns a context;
    ``enabled=False`` (single process) when RANK is unset."""
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return Dist(enabled=False, rank=0, local_rank=0, world_size=1, master=True, device=device)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Collective timeout: how long a rank blocks on a peer before aborting. ONE
    # value cannot serve both phases, and using the steady-state value for startup
    # breaks 8-node runs outright:
    #
    #   STARTUP wants a LONG timeout. Bringing up the process group + DDP's first
    #   collectives establishes a mesh of NCCL connections over plain TCP (no EFA
    #   on g4dn/g5) — the setup cost grows ~O(N^2) in nodes. Measured: 2- and
    #   4-node came up inside 20s, 8-node did NOT, and the DDP shape-verification
    #   broadcast aborted at exactly 20000ms. torch then reports that as
    #   "params[0] ... appears not to match sizes ... in process 0", which reads
    #   like a model-config bug and sends you hunting for a vocab mismatch that
    #   isn't there. Every rank had an identical 123.69M-param model.
    #
    #   STEADY STATE wants a SHORT timeout, which is why NCCL_TIMEOUT exists: a
    #   survivor of a node kill must abort fast so the supervisor can re-form the
    #   world. Long here would directly slow preemption recovery.
    #
    # So: come up on NCCL_INIT_TIMEOUT, then tighten to NCCL_TIMEOUT once the
    # startup collectives are done (see ``tighten_timeout``).
    steady_s = int(os.environ.get("NCCL_TIMEOUT", "600"))
    init_s = int(os.environ.get("NCCL_INIT_TIMEOUT", str(max(600, steady_s))))
    timeout = timedelta(seconds=init_s)
    if device.startswith("cuda"):
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        # device_id binds the NCCL communicator to this rank's GPU eagerly, so
        # collectives can't lazily pick the wrong device (and shutdown is clean).
        dist.init_process_group(backend="nccl", device_id=torch.device(device), timeout=timeout)
    else:
        dist.init_process_group(backend="gloo", timeout=timeout)
    return Dist(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        master=(rank == 0),
        device=device,
    )


def tighten_timeout(d: Dist) -> int | None:
    """Shrink the process-group timeout to the steady-state NCCL_TIMEOUT.

    Call once the startup collectives (process-group init + DDP construction)
    are done. Until then we run on the generous NCCL_INIT_TIMEOUT so an 8-node
    world has time to build its TCP connection mesh; from here on we want the
    SHORT timeout, because it is the in-band signal that a peer died — a
    survivor of a preemption must abort fast enough for the supervisor to
    re-form the world.

    Returns the applied timeout in seconds, or None if it could not be changed
    (no process group, no steady value, or the torch build lacks the private
    setter). Failing to tighten is degraded-but-correct — the run simply keeps
    the longer timeout and detects a dead peer more slowly — so this never
    raises into the training path.
    """
    if not d.enabled or not dist.is_initialized():
        return None
    steady = os.environ.get("NCCL_TIMEOUT")
    if not steady:
        return None
    try:
        seconds = int(steady)
        from torch.distributed import distributed_c10d as _c10d

        setter = getattr(_c10d, "_set_pg_timeout", None)
        if setter is None:  # older torch: leave the init timeout in place
            return None
        setter(timedelta(seconds=seconds), _c10d._get_default_group())
        return seconds
    except Exception:  # noqa: BLE001 — a slower death signal must not kill the run
        return None


def shutdown(d: Dist) -> None:
    if d.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_stop(d: Dist, local_stop: bool) -> bool:
    """Agree a stop across ranks (MAX): if ANY rank wants to stop, all stop on the
    same iteration. Must be the last collective in the loop body so no rank is left
    blocking on the next backward's gradient all-reduce (deadlock avoidance)."""
    if not d.enabled:
        return local_stop
    # Collective tensors must live on the backend's device: nccl only reduces
    # CUDA tensors (a CPU tensor raises "No backend type associated with device
    # type cpu"); on gloo d.device is "cpu" so this is a no-op.
    t = torch.tensor([1 if local_stop else 0], dtype=torch.int32, device=d.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def broadcast_flag(d: Dist, local_flag: bool) -> bool:
    """Rank 0's flag, agreed group-wide (non-master inputs are ignored). Used to
    step-align the node-local checkpoint tier: rank 0 decides "checkpoint at this
    step" and every node's LOCAL_RANK-0 snapshots the SAME step, which is what
    lets a shrunken group later agree on a common local resume step. Implemented
    as a MAX all-reduce where only rank 0 contributes, so it composes with the
    same every-rank-every-step call pattern as all_reduce_stop."""
    if not d.enabled:
        return local_flag
    t = torch.tensor([1 if (local_flag and d.master) else 0], dtype=torch.int32, device=d.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def all_reduce_min(d: Dist, value: int) -> int:
    """Group-wide MIN of an integer (identity when not under torchrun). The
    resume path uses this to agree on a checkpoint step every rank can load:
    the minimum over each rank's best locally-available step."""
    if not d.enabled:
        return value
    t = torch.tensor([value], dtype=torch.int64, device=d.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def mean_loss(d: Dist, loss_value: float) -> float:
    """Global mean loss across ranks for the reported log line (all ranks call it)."""
    if not d.enabled:
        return loss_value
    t = torch.tensor([loss_value], dtype=torch.float32, device=d.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / d.world_size
