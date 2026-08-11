"""Full-state checkpoint: everything that affects the next training step.

Missing any one of these makes resume silently diverge:

    - model weights
    - optimizer state
    - step number
    - all RNG states           (see rng.py)
    - data-loader position     (see data.py)

Save path: serialize to a local temp file, then hand off to the store, which
renames atomically (see s3_store.save_atomic). A mid-write kill can only ever
leave a .tmp behind — never a corrupt "latest".

Two tools answer "is this checkpoint comprehensive and valid?":
  - ``verify(ref)``     — loads it, checks the schema is complete and every
                          tensor is finite (catches NaN/inf and truncation).
  - ``smoke_test(...)`` — restores the weights into a fresh model and runs one
                          forward pass, asserting a finite loss (catches a file
                          that loads but is subtly wrong).
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from typing import Any

import torch

from . import distributed, rng, s3_store

# v2 adds trained_seconds (cumulative in-loop wall-clock, the run-level budget's
# progress meter). v3 adds an OPTIONAL "scaler" (the fp16 GradScaler's loss-scale
# state — restoring it keeps a preempt/resume bit-exact instead of re-warming the
# scale from 2**16). Older blobs still load: trained_seconds -> 0, scaler -> None.
CKPT_VERSION = 3
_KNOWN_VERSIONS = (1, 2, 3)
_REQUIRED_KEYS = ("version", "step", "model", "optimizer", "rng", "loader")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, corrupt, or non-finite."""


def _ckpt_name(step: int) -> str:
    # zero-padded so lexicographic sort == numeric sort for `latest()`
    return f"{s3_store.CHECKPOINT_PREFIX}{step:012d}.pt"


def _step_of(ref: str | None) -> int:
    """Step number encoded in a checkpoint ref's basename, or -1 for None."""
    if ref is None:
        return -1
    base = ref.rsplit("/", 1)[-1]
    return int(base[len(s3_store.CHECKPOINT_PREFIX) : -len(".pt")])


def _prune_durable(uri: str, keep: int, log=None) -> None:
    """Trim the durable tier after a successful write.

    A prune failure must NEVER end the run: the worst case of skipping it is
    today's behaviour (an unpruned prefix), which is a cost and latency problem,
    not a correctness one. Same discipline AsyncCheckpointer._write already
    applies to its own exceptions.
    """
    try:
        n = s3_store.prune_checkpoints(uri, keep)
    except Exception as e:  # noqa: BLE001 — housekeeping must not kill training
        if log:
            log(f"[checkpoint] prune failed (non-fatal, continuing): {e!r}")
        return
    if n and log:
        log(f"[checkpoint] pruned {n} old checkpoint(s), keeping {keep}")


def save(
    *,
    model,
    optimizer,
    loader,
    step: int,
    uri: str,
    trained_seconds: float = 0.0,
    scaler=None,
    keep: int = 0,
) -> str:
    """Atomically persist full training state. Returns the final checkpoint ref."""
    blob: dict[str, Any] = {
        "version": CKPT_VERSION,
        "step": step,
        "trained_seconds": trained_seconds,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng.capture(),
        "loader": loader.state_dict(),
        "scaler": _scaler_state(scaler),
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(blob, tmp_path)
        _warn_if_low_disk(os.path.getsize(tmp_path))
        ref = s3_store.save_atomic(tmp_path, uri, _ckpt_name(step))
        # AFTER the write, never before: the new checkpoint must be durable
        # before anything old is deleted, or a crash mid-prune leaves the run
        # with fewer checkpoints than it thinks.
        _prune_durable(uri, keep)
        return ref
    finally:
        # The local backend renames tmp_path away; the S3 backend uploads a
        # copy and leaves it — without this, every save leaks a checkpoint
        # into /tmp until the disk fills.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _scaler_state(scaler) -> dict | None:
    """The GradScaler's state, or None when there's no scaler / it's disabled
    (bf16, fp32, CPU) — a disabled scaler's state_dict is empty and carries no
    information worth persisting, so None keeps blobs uniform across dtypes."""
    if scaler is None or not scaler.is_enabled():
        return None
    return scaler.state_dict()


def _cuda_tensors(obj: Any, out: list) -> None:
    """Collect every CUDA tensor in a state tree, in deterministic walk order."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _cuda_tensors(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _cuda_tensors(v, out)


def _batched_d2h(tensors: list) -> dict[int, torch.Tensor]:
    """One pinned staging buffer, one async copy per tensor, ONE sync at the end.

    The naive ``t.to("cpu", copy=True)`` per tensor is a SYNCHRONISING call: it
    waits for the GPU, which is busy with queued training kernels. GPT-2 124M has
    ~450 such tensors (weights + Adam's two moments each), so the cost is ~450
    stalls rather than bandwidth — raw PCIe Gen4 x16 moves the whole 1.5 GB in
    ~250 ms, but measured stalls were seconds.

    Instead: allocate one pinned (page-locked) host buffer, issue every copy as
    non_blocking into slices of it, then synchronise ONCE. Pinned memory also
    lets the driver DMA directly instead of staging through its own bounce
    buffer, which is worth roughly 3x on its own.

    Returns id(tensor) -> CPU tensor so the tree walk can substitute them.
    """
    if not tensors:
        return {}
    total = sum(t.numel() * t.element_size() for t in tensors)
    try:
        flat = torch.empty(total, dtype=torch.uint8, device="cpu", pin_memory=True)
    except RuntimeError:
        # Pinned allocation can fail under host-memory pressure; fall back to
        # pageable rather than failing the checkpoint.
        flat = torch.empty(total, dtype=torch.uint8, device="cpu")
    out: dict[int, torch.Tensor] = {}
    off = 0
    for t in tensors:
        nbytes = t.numel() * t.element_size()
        view = flat[off : off + nbytes].view(torch.uint8)
        # A contiguous byte view of the destination, reinterpreted as the source
        # dtype/shape, so one copy_ moves the tensor with no per-element work.
        dst = view.view(t.dtype).view(t.shape) if t.is_contiguous() else None
        if dst is None:
            out[id(t)] = t.detach().to("cpu", copy=True)  # rare: non-contiguous
        else:
            dst.copy_(t.detach(), non_blocking=True)
            out[id(t)] = dst
        off += nbytes
    torch.cuda.synchronize()  # the ONE sync: every copy above is now complete
    return out


def _cpu_copy(obj: Any, mapping: dict[int, torch.Tensor] | None = None) -> Any:
    """Deep copy a state tree with every tensor moved to CPU. The copy is the
    point: the optimizer keeps mutating the live tensors while the background
    writer serializes, so the snapshot must not alias them.

    ``mapping`` supplies already-copied CUDA tensors from :func:`_batched_d2h`;
    without it each tensor is copied individually (the CPU-test path)."""
    if isinstance(obj, torch.Tensor):
        if mapping is not None and id(obj) in mapping:
            return mapping[id(obj)]
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_copy(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_copy(v, mapping) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_copy(v, mapping) for v in obj)
    return copy.deepcopy(obj)


def snapshot(
    *, model, optimizer, loader, step: int, trained_seconds: float = 0.0, scaler=None
) -> dict[str, Any]:
    """Point-in-time CPU copy of the full training state (same schema as
    ``save`` writes). This is the only part of an async checkpoint that must run
    on the training critical path, and its cost scales with model + optimizer
    size: ~tens of ms for the small Shakespeare model, but SECONDS for GPT-2-124M
    — the D2H copy of ~1.5 GB (weights + Adam moments) is what shows up as a
    periodic stall on the timeline. Keep CHECKPOINT_INTERVAL_SECONDS well above
    it so the stall is amortized, not per-step. RNG and loader state are captured
    in the same instant as the weights — the invariant that keeps resume from
    silently diverging."""
    model_sd, opt_sd = model.state_dict(), optimizer.state_dict()
    scaler_sd = _scaler_state(scaler)
    # ONE batched device->host transfer for every CUDA tensor across all three
    # trees, then one sync — instead of ~450 individually synchronising copies.
    mapping = None
    if torch.cuda.is_available():
        cuda: list = []
        for tree in (model_sd, opt_sd, scaler_sd):
            _cuda_tensors(tree, cuda)
        if cuda:
            mapping = _batched_d2h(cuda)
    return {
        "version": CKPT_VERSION,
        "step": step,
        "trained_seconds": trained_seconds,
        "model": _cpu_copy(model_sd, mapping),
        "optimizer": _cpu_copy(opt_sd, mapping),
        "rng": rng.capture(),  # capture() already returns copies
        "loader": dict(loader.state_dict()),
        "scaler": _cpu_copy(scaler_sd, mapping),
    }


def save_local(blob: dict[str, Any], local_dir: str, step: int, keep: int = 2) -> str:
    """Write a snapshot blob to the node-local tier (atomic same-dir rename) and
    prune to the ``keep`` newest. Keeping two absorbs one interval of skew when
    a crash lands mid-save somewhere in the group — the group-MIN agreement in
    :func:`load_group_latest` can then still find a step everyone has."""
    os.makedirs(local_dir, exist_ok=True)
    # Temp file IN the destination dir so os.replace never crosses filesystems.
    fd, tmp_path = tempfile.mkstemp(suffix=".pt", dir=local_dir)
    os.close(fd)
    try:
        torch.save(blob, tmp_path)
        final = os.path.join(local_dir, _ckpt_name(step))
        os.replace(tmp_path, final)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    kept = sorted(
        f
        for f in os.listdir(local_dir)
        if f.startswith(s3_store.CHECKPOINT_PREFIX) and f.endswith(".pt")
    )
    for old in kept[:-keep]:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(local_dir, old))
    return final


class AsyncLocalSaver:
    """Async writer for the NODE-LOCAL tier — the same two-phase split
    :class:`AsyncCheckpointer` uses for S3, applied to the local disk write.

    Why this exists: the local tier was fully synchronous, so a 1.5 GB
    ``torch.save`` sat on the training critical path every checkpoint. Measured
    at 4 nodes on GPT-2 124M, that was **99s of 301.7 training-seconds — 33% of
    all training time** (checkpoint steps showed 28.5s vs a 3.0s norm). The S3
    tier had solved this already; the local tier just never got the same
    treatment.

    The caller still snapshots synchronously (:func:`snapshot` — a point-in-time
    CPU copy, which is what makes the write safe while the optimizer keeps
    mutating the live tensors). Only the serialize + write + prune moves off the
    critical path.

    One save in flight, like the S3 saver: ``submit`` returns False while busy,
    so memory stays bounded at one blob and a slow disk cannot queue-pile. The
    durability trade is the same and is deliberate — worst-case lost work on a
    hard kill becomes the checkpoint interval plus one local write, and the
    node-local tier is the FAST-RESTART tier, not the durable one (S3 is
    durable). A survivor that loses one local checkpoint falls back to S3.

    Preempt and final checkpoints stay synchronous in the trainer; call
    :meth:`flush` first so the writer never races them.
    """

    def __init__(self, local_dir: str, *, keep: int = 2, log: Callable[[str], None] | None = None):
        self._dir = local_dir
        self._keep = keep
        self._log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self._thread: threading.Thread | None = None
        self.failures = 0
        self.skipped = 0

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, blob: dict[str, Any], step: int) -> bool:
        """Hand an ALREADY-SNAPSHOTTED blob to the writer. False = previous write
        still in flight (skipped). The caller owns the snapshot because rank 0
        may reuse the same blob it just handed to the S3 tier."""
        if self.busy():
            self.skipped += 1
            return False
        self._thread = threading.Thread(
            target=self._write, args=(blob, step), name="ckpt-local-writer", daemon=True
        )
        self._thread.start()
        return True

    def _write(self, blob: dict[str, Any], step: int) -> None:
        try:
            save_local(blob, self._dir, step, keep=self._keep)
        except Exception as e:  # noqa: BLE001 — background thread: log, count, keep training
            self.failures += 1
            self._log(
                f"[checkpoint] ASYNC LOCAL save at step {step} FAILED: {e!r} — "
                "training continues; the S3 tier is unaffected"
            )

    def flush(self, timeout: float = 300.0) -> None:
        """Wait for the in-flight local write. Called before the synchronous
        preempt/final checkpoints and at shutdown, so a resume can never read a
        directory that a background thread is still pruning."""
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)


class AsyncCheckpointer:
    """Two-phase checkpointing: the caller snapshots on its own thread
    (:func:`snapshot`, ~tens of ms), then a daemon thread serializes, uploads
    (same atomic temp-key -> rename), and verify/smoke-tests off the critical
    path.

    One save in flight at a time: ``submit`` returns False while the previous
    write is still running (the caller retries on a later loop iteration), so
    memory is bounded at one snapshot and a slow S3 day can't queue-pile.
    Durability shifts accordingly: worst-case lost work on a hard kill is the
    checkpoint interval PLUS one upload, not the interval alone. Preempt and
    final checkpoints stay synchronous in the trainer — call ``flush`` first so
    the writer never races them.

    Verify/smoke run on CPU in the background thread (a GPU forward there would
    contend with training); a failed background save is logged and counted, and
    training continues on the previous good checkpoint."""

    def __init__(
        self,
        uri: str,
        *,
        verify_every: int = 1,
        build_model: Callable[[], Any] | None = None,
        sample_batch: tuple | None = None,
        log: Callable[[str], None] | None = None,
        keep: int = 0,
    ):
        self._uri = uri
        self._keep = keep
        self._verify_every = verify_every
        self._build_model = build_model
        self._sample_batch = sample_batch  # CPU tensors (see trainer: RNG-free eval batch)
        self._log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self._thread: threading.Thread | None = None
        self._count = 0
        self.failures = 0

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(
        self,
        *,
        model=None,
        optimizer=None,
        loader=None,
        step: int,
        trained_seconds: float = 0.0,
        scaler=None,
        blob: dict[str, Any] | None = None,
    ) -> bool:
        """Snapshot now and hand off to the writer. False = previous save still
        in flight (skipped; nothing was snapshotted).

        Pass ``blob`` to reuse a snapshot the caller already took. Rank 0 feeds
        BOTH tiers, and the two snapshots were identical by construction (same
        model, optimizer and step, with a collective between them guaranteeing
        training had not advanced) — so taking two was a second full 1.5 GB
        device->host copy for nothing. The writers only ever READ the blob, so
        sharing one is safe; it also halves peak host memory, at the cost of the
        blob living until BOTH writers finish."""
        if self.busy():
            return False
        if blob is None:
            blob = snapshot(
                model=model,
                optimizer=optimizer,
                loader=loader,
                step=step,
                trained_seconds=trained_seconds,
                scaler=scaler,
            )
        self._count += 1
        self._thread = threading.Thread(
            target=self._write, args=(blob, step, self._count), name="ckpt-writer", daemon=True
        )
        self._thread.start()
        return True

    def _write(self, blob: dict[str, Any], step: int, count: int) -> None:
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            try:
                torch.save(blob, tmp_path)
                _warn_if_low_disk(os.path.getsize(tmp_path))
                ref = s3_store.save_atomic(tmp_path, self._uri, _ckpt_name(step))
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            # Rank-0-only on the durable tier (docs/checkpoint-tiers.md), so
            # there is no concurrent-pruner race by construction.
            _prune_durable(self._uri, self._keep, self._log)
            if self._verify_every and count % self._verify_every == 0:
                good = verify(ref, map_location="cpu")
                if self._build_model is not None and self._sample_batch is not None:
                    smoke_test(good, self._build_model, self._sample_batch, "cpu")
                self._log(f"[verify] checkpoint at step {step} passed verify + smoke test (async)")
        except Exception as e:  # noqa: BLE001 — background thread: log, count, keep training
            self.failures += 1
            self._log(
                f"[checkpoint] ASYNC save at step {step} FAILED: {e!r} — "
                "training continues on the previous checkpoint"
            )

    def flush(self, timeout: float = 300.0) -> None:
        """Wait for the in-flight save (if any) to finish. Called before the
        synchronous preempt/final checkpoints and at shutdown."""
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)


def _warn_if_low_disk(ckpt_bytes: int) -> None:
    """One save+verify cycle needs ~2x the checkpoint transiently; warn at 4x
    so the operator sees the disk shrinking before a write fails mid-run."""
    free = shutil.disk_usage(tempfile.gettempdir()).free
    if free < 4 * ckpt_bytes:
        print(
            f"[checkpoint] WARNING: {free / 1e9:.1f} GB free on "
            f"{tempfile.gettempdir()} < 4x checkpoint size "
            f"({ckpt_bytes / 1e9:.2f} GB) — the next save/verify may fail",
            file=sys.stderr,
            flush=True,
        )


def load_latest(uri: str, map_location: str = "cpu") -> dict[str, Any] | None:
    """Return the newest checkpoint blob under ``uri``, or None if none exists."""
    ref = s3_store.latest(uri)
    if ref is None:
        return None
    with s3_store.fetch(ref) as local:  # no-op for local refs; downloads+verifies S3
        return torch.load(local, map_location=map_location, weights_only=False)


def load_group_latest(
    uri: str,
    local_dir: str = "",
    dist_ctx: distributed.Dist | None = None,
    map_location: str = "cpu",
) -> dict[str, Any] | None:
    """The one resume path, now across two tiers and N ranks.

    Every rank offers the newest step it can reach (node-local tier if present,
    else the durable S3 tier); the group takes the MIN so all ranks restore the
    SAME step, then each loads it from the cheapest tier that has it. The two
    interesting cases fall out without any membership detection:

      - group shrank (survivors only): everyone holds the same step-aligned
        local snapshot -> instant disk restore, ~zero lost work;
      - a fresh replacement is present: its best is S3-latest, which becomes the
        group MIN -> everyone restores S3-latest (survivors lose at most one
        interval + one in-flight upload — the existing durability bound).

    Degrades exactly to :func:`load_latest` semantics for a single process with
    no local tier, so single-node paths are unchanged.
    """
    local_ref = s3_store.latest(local_dir) if local_dir else None
    s3_ref = s3_store.latest(uri)
    best = max(_step_of(local_ref), _step_of(s3_ref))
    group_step = distributed.all_reduce_min(dist_ctx, best) if dist_ctx is not None else best
    if group_step < 0:
        return None  # no checkpoint anywhere in the group -> fresh
    name = _ckpt_name(group_step)
    local_path = os.path.join(local_dir, name) if local_dir else ""
    if local_path and os.path.exists(local_path):
        ref = local_path
    else:
        ref = s3_store.ref_for(uri, name)
        if not s3_store.exists(ref):
            # A peer holds a local step the durable tier never received (its
            # upload failed). Crash loudly: the elastic agent restarts us, the
            # group re-agrees, and S3 has usually caught up by then.
            raise CheckpointError(
                f"group agreed on step {group_step} but this rank has no tier "
                f"holding it (local={local_path or 'off'}, s3={ref})"
            )
    with s3_store.fetch(ref) as local:
        return torch.load(local, map_location=map_location, weights_only=False)


def restore_into(blob: dict[str, Any], *, model, optimizer, loader, scaler=None) -> int:
    """Restore all state from ``blob``. Returns the step to resume from.

    ``scaler`` is optional: only fp16 runs carry one, and only a v3+ blob written
    by an enabled scaler holds its state. When either is absent the scaler simply
    keeps its fresh loss-scale (a brief re-warm, never a divergence)."""
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    loader.load_state_dict(blob["loader"])
    rng.restore(blob["rng"])
    saved_scaler = blob.get("scaler")
    if scaler is not None and scaler.is_enabled() and saved_scaler:
        scaler.load_state_dict(saved_scaler)
    return blob["step"]


# --------------------------------------------------------------------------- #
# Validation tools
# --------------------------------------------------------------------------- #
def _all_finite(state: Any) -> bool:
    """Recursively assert every floating-point tensor in a state tree is finite."""
    if isinstance(state, torch.Tensor):
        return (not state.is_floating_point()) or bool(torch.isfinite(state).all())
    if isinstance(state, dict):
        return all(_all_finite(v) for v in state.values())
    if isinstance(state, list | tuple):
        return all(_all_finite(v) for v in state)
    return True


def _verify_blob(blob: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in _REQUIRED_KEYS if k not in blob]
    if missing:
        raise CheckpointError(f"checkpoint missing keys: {missing}")
    if blob["version"] not in _KNOWN_VERSIONS:
        raise CheckpointError(f"unsupported checkpoint version {blob['version']}")
    if not _all_finite(blob["model"]):
        raise CheckpointError("model weights contain NaN/inf")
    if not _all_finite(blob["optimizer"].get("state", {})):
        raise CheckpointError("optimizer state contains NaN/inf")
    return blob


def verify(ref: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load ``ref`` and assert it is complete and finite. Returns the blob.

    ``ref`` may be a local path or an ``s3://`` URI (downloaded + checksum-checked).
    Raises :class:`CheckpointError` on any problem; ``torch.load`` itself raises on
    a truncated/corrupt file.
    """
    with s3_store.fetch(ref) as local:
        return _verify_blob(torch.load(local, map_location=map_location, weights_only=False))


def smoke_test(
    blob: dict[str, Any],
    build_model: Callable[[], Any],
    sample_batch: tuple,
    device: str,
) -> float:
    """Restore weights into a fresh model and run one forward pass.

    Confirms the saved state actually reconstructs a working model, not just a
    loadable file. Returns the (finite) loss; raises :class:`CheckpointError`
    otherwise.
    """
    model = build_model().to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    with torch.no_grad():
        _, loss = model(*sample_batch)
    if loss is None or not bool(torch.isfinite(loss)):
        raise CheckpointError("smoke test produced a non-finite loss")
    return float(loss.item())
