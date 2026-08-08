"""Async checkpointing tests — hermetic (local fs, tiny torch modules).

Pins the properties the two-phase design rests on: the snapshot is a true
point-in-time copy (later mutation can't leak in), the written artifact is a
valid checkpoint (same schema, restores exactly), only one save is ever in
flight (skip-when-busy), and a background failure is counted + survivable
rather than fatal.
"""

from __future__ import annotations

import os
import threading

import torch

from spot_train import checkpoint


class _StubLoader:
    """Just enough loader for snapshot/restore (state_dict of plain ints)."""

    def __init__(self, step: int = 0, epoch: int = 0):
        self.state = {"step": step, "epoch": epoch}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, sd):
        self.state = dict(sd)


def _model_opt(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # One step so the optimizer has real state tensors to snapshot.
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()
    return model, opt


def test_async_write_restores_exactly(tmp_path):
    model, opt = _model_opt()
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=1)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(step=7), step=42)
    writer.flush()
    assert writer.failures == 0

    blob = checkpoint.load_latest(str(tmp_path) + "/")
    fresh_model, fresh_opt = _model_opt(seed=1)  # different init, gets overwritten
    fresh_loader = _StubLoader()
    assert (
        checkpoint.restore_into(blob, model=fresh_model, optimizer=fresh_opt, loader=fresh_loader)
        == 42
    )
    for a, b in zip(model.state_dict().values(), fresh_model.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert fresh_loader.state == {"step": 7, "epoch": 0}


def test_snapshot_is_point_in_time(tmp_path):
    model, opt = _model_opt()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=0)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=1)
    # Mutate the live weights immediately — the background write must not see it.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    writer.flush()
    blob = checkpoint.load_latest(str(tmp_path) + "/")
    for k, v in blob["model"].items():
        assert torch.equal(v, original[k])


def test_one_in_flight_skips_when_busy(tmp_path, monkeypatch):
    release = threading.Event()
    real_save_atomic = checkpoint.s3_store.save_atomic

    def slow_save_atomic(local_path, uri, name):
        release.wait(timeout=30)
        return real_save_atomic(local_path, uri, name)

    monkeypatch.setattr(checkpoint.s3_store, "save_atomic", slow_save_atomic)
    model, opt = _model_opt()
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=0)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=1)
    # Previous upload still blocked => skip, nothing queued.
    assert not writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=2)
    release.set()
    writer.flush()
    blob = checkpoint.load_latest(str(tmp_path) + "/")
    assert blob["step"] == 1  # only the first save happened
    # Idle again => a new submit goes through.
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=3)
    writer.flush()
    assert checkpoint.load_latest(str(tmp_path) + "/")["step"] == 3


def test_background_failure_is_counted_not_fatal(tmp_path, monkeypatch):
    def boom(local_path, uri, name):
        raise OSError("upload exploded")

    monkeypatch.setattr(checkpoint.s3_store, "save_atomic", boom)
    model, opt = _model_opt()
    logs: list[str] = []
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=0, log=logs.append)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=1)
    writer.flush()
    assert writer.failures == 1
    assert any("FAILED" in m for m in logs)
    assert checkpoint.load_latest(str(tmp_path) + "/") is None  # nothing corrupt left behind


def test_checkpoint_async_env_parse(monkeypatch):
    from spot_train.config import TrainConfig

    monkeypatch.delenv("CHECKPOINT_ASYNC", raising=False)
    assert TrainConfig.from_env().checkpoint_async is True
    monkeypatch.setenv("CHECKPOINT_ASYNC", "0")
    assert TrainConfig.from_env().checkpoint_async is False
    monkeypatch.setenv("CHECKPOINT_ASYNC", "false")
    assert TrainConfig.from_env().checkpoint_async is False


# --------------------------------------------------------------------------- #
# AsyncLocalSaver — the node-local tier's writer
# --------------------------------------------------------------------------- #
def test_local_saver_writes_off_the_critical_path(tmp_path):
    """The whole point: submit() returns immediately, the file appears later.

    Synchronously, a 1.5 GB torch.save sat on every checkpoint step — measured
    at 33% of all training time on GPT-2 124M.
    """
    import time

    from spot_train import checkpoint as ckpt

    saver = ckpt.AsyncLocalSaver(str(tmp_path))
    blob = {"version": ckpt.CKPT_VERSION, "step": 7, "payload": torch.zeros(256)}
    t0 = time.monotonic()
    assert saver.submit(blob, 7) is True
    submit_s = time.monotonic() - t0
    saver.flush()
    assert submit_s < 0.5, "submit must hand off, not write"
    files = [f for f in os.listdir(tmp_path) if f.endswith(".pt")]
    assert files, "the background writer never produced a checkpoint"
    assert torch.load(os.path.join(tmp_path, files[0]), weights_only=False)["step"] == 7


def test_local_saver_bounds_memory_to_one_inflight(tmp_path, monkeypatch):
    """One save in flight, like the S3 tier: a second submit is refused rather
    than queued, so a slow disk cannot pile up 1.5 GB blobs in RAM."""
    from spot_train import checkpoint as ckpt

    gate = threading.Event()
    real = ckpt.save_local

    def slow(blob, d, step, keep=2):
        gate.wait(5)
        return real(blob, d, step, keep=keep)

    monkeypatch.setattr(ckpt, "save_local", slow)
    saver = ckpt.AsyncLocalSaver(str(tmp_path))
    assert saver.submit({"step": 1}, 1) is True
    assert saver.submit({"step": 2}, 2) is False, "second save must be refused while busy"
    assert saver.skipped == 1
    gate.set()
    saver.flush()


def test_local_saver_failure_does_not_kill_training(tmp_path, monkeypatch):
    """A background write that throws is logged and counted; the trainer keeps
    going on the previous checkpoint. The S3 tier is unaffected."""
    from spot_train import checkpoint as ckpt

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ckpt, "save_local", boom)
    logs: list[str] = []
    saver = ckpt.AsyncLocalSaver(str(tmp_path), log=logs.append)
    assert saver.submit({"step": 3}, 3) is True
    saver.flush()
    assert saver.failures == 1
    assert any("ASYNC LOCAL save at step 3 FAILED" in m for m in logs)


def test_local_saver_flush_is_safe_when_idle(tmp_path):
    # flush() is called on shutdown paths that may never have submitted.
    from spot_train import checkpoint as ckpt

    ckpt.AsyncLocalSaver(str(tmp_path)).flush()


def test_snapshot_does_not_alias_live_tensors(tmp_path):
    """The whole reason the CPU copy stays on the critical path: the optimizer
    keeps mutating the live tensors while a background thread serializes. If the
    snapshot aliased them the checkpoint would be torn. Batching the copies must
    not weaken this."""
    import torch

    from spot_train import checkpoint

    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(2, 8)).sum().backward()
    opt.step()

    class _L:
        def state_dict(self):
            return {"step": 1, "epoch": 0}

    blob = checkpoint.snapshot(model=model, optimizer=opt, loader=_L(), step=1)
    before = blob["model"]["weight"].clone()
    with torch.no_grad():  # mutate the LIVE weight after snapshotting
        model.weight.add_(100.0)
    assert torch.equal(blob["model"]["weight"], before), "snapshot aliased a live tensor"


def test_shared_blob_serializes_identically_for_both_tiers(tmp_path):
    """Rank 0 hands ONE blob to the S3 writer and the local writer. Both only
    read it, so the two artifacts must be byte-identical in content — that is
    what makes sharing safe instead of a second 1.5 GB copy."""
    import torch

    from spot_train import checkpoint

    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    class _L:
        def state_dict(self):
            return {"step": 3, "epoch": 0}

    blob = checkpoint.snapshot(model=model, optimizer=opt, loader=_L(), step=3)
    a = checkpoint.save_local(blob, str(tmp_path / "a"), 3)
    b = checkpoint.save_local(blob, str(tmp_path / "b"), 3)
    la = torch.load(a, map_location="cpu", weights_only=False)
    lb = torch.load(b, map_location="cpu", weights_only=False)
    assert la["step"] == lb["step"] == 3
    assert torch.equal(la["model"]["weight"], lb["model"]["weight"])


def test_submit_accepts_a_prebuilt_blob(tmp_path):
    """AsyncCheckpointer.submit(blob=...) must skip its own snapshot entirely —
    that skip IS the fix; otherwise rank 0 still pays two device->host copies."""
    from spot_train import checkpoint

    calls = {"n": 0}
    real = checkpoint.snapshot

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    checkpoint.snapshot = counting
    try:
        saver = checkpoint.AsyncCheckpointer(str(tmp_path / "ck"))
        blob = {
            "version": checkpoint.CKPT_VERSION,
            "step": 5,
            "trained_seconds": 0.0,
            "model": {},
            "optimizer": {},
            "rng": {},
            "loader": {"step": 5, "epoch": 0},
            "scaler": None,
        }
        assert saver.submit(step=5, blob=blob) is True
        saver.flush()
        assert calls["n"] == 0, "submit(blob=...) must not snapshot again"
    finally:
        checkpoint.snapshot = real
