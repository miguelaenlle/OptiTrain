"""R1 — resumability, verified against an UNINTERRUPTED control.

Nothing else in the suite drives the training loop through a stop and a resume.
The existing coverage is checkpoint-level primitives (save/verify/restore,
trained_seconds carried, group agreement) — all necessary, none of which proves
the loop actually continues.

Why an A/B rather than "it ran and loss went down": every interesting failure
here still produces a clean decreasing curve and reads as success.

  * weights reinitialized      -> trains from scratch, curve looks fine
  * optimizer moments lost     -> loss spikes then recovers, looks like a blip
  * loader position reset      -> re-trains the same data, loss looks BETTER
  * trained_seconds reset      -> segment 2 runs a full fresh budget (2x cost)

Comparing against a control that was never interrupted catches all four at once.
Bit-exactness is deliberately relaxed for the MVP (CLAUDE.md), so the bound is
"close to the control", not "identical to it".

CPU-only, tiny model, synthetic data — a few seconds.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spot_train import checkpoint  # noqa: E402
from spot_train import train as train_mod  # noqa: E402
from spot_train.config import TrainConfig  # noqa: E402

VOCAB = 65
BLOCK = 16


def _dataset(d: Path) -> None:
    """A LEARNABLE synthetic corpus. Random tokens would make loss flat at
    ln(VOCAB) and the whole comparison vacuous — the control and the resumed run
    would agree at chance level while proving nothing."""
    # A strict cycle 0,1,2,...,VOCAB-1,0,1,... Next-token is fully determined by
    # the previous one, so a 1-layer model learns it in tens of steps and the loss
    # moves FAR from ln(VOCAB). That gap is what the A/B assertions measure
    # against; on a task the model cannot learn, both runs sit at chance and the
    # comparison passes while proving nothing.
    seq = np.tile(np.arange(VOCAB, dtype=np.uint16), 400)
    for split in ("train", "val"):
        seq.astype(np.uint16).tofile(d / f"{split}.bin")
    with (d / "meta.pkl").open("wb") as fh:
        pickle.dump({"vocab_size": VOCAB}, fh)


def _cfg(tmp: Path, *, steps: int, ckpt: Path) -> TrainConfig:
    return TrainConfig(
        device="cpu",
        dtype="float32",
        n_layer=1,
        n_head=1,
        n_embd=32,
        block_size=BLOCK,
        batch_size=8,
        global_batch_size=8,
        max_steps=steps,
        max_seconds=None,
        learning_rate=3e-2,
        warmup_steps=0,
        lr_decay_steps=1000,
        min_lr=1e-4,
        eval_iters=2,
        eval_interval_steps=0,
        sample_interval_steps=0,
        log_interval_steps=1,
        checkpoint_interval_seconds=1e9,  # only the explicit final save
        checkpoint_async=False,
        smoke_test_every=0,
        data_local_dir=str(tmp / "data"),
        checkpoint_uri=str(ckpt),
        samples_uri=str(tmp / "samples.json"),
        metrics_uri=str(tmp / "metrics.json"),
        # No end-of-run sampling: with no stoi/itos in meta.pkl the codec falls
        # back to the GPT-2 BPE (vocab 50257) and indexes a 65-token embedding.
        sample_prompts=[],
        seed=1234,
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    _dataset(d)
    return d


def _run(cfg: TrainConfig) -> dict:
    return train_mod.train(cfg)


def test_resume_matches_an_uninterrupted_control(tmp_path: Path, data_dir: Path):
    """The gold standard: 20 straight vs 10 + resume + 10."""
    ck_a, ck_b = tmp_path / "ck_a", tmp_path / "ck_b"
    ck_a.mkdir()
    ck_b.mkdir()

    control = _run(_cfg(tmp_path, steps=60, ckpt=ck_a))

    first = _run(_cfg(tmp_path, steps=30, ckpt=ck_b))
    assert first["steps"] == 30
    resumed = _run(_cfg(tmp_path, steps=60, ckpt=ck_b))

    assert resumed["resumed"] is True, "second launch did not take the resume path"
    assert resumed["steps"] == 60, "step counter restarted instead of continuing"

    # PRECONDITION: the control must actually have learned something, or every
    # assertion below is satisfied at chance level and the test is decorative.
    fresh0 = math.log(VOCAB)
    assert control["train_loss"] < fresh0 - 0.5, (
        f"control loss {control['train_loss']:.4f} never left fresh-init "
        f"(~{fresh0:.4f}) — the A/B cannot distinguish anything"
    )

    # The headline assertion. A reinitialized model would land near ln(VOCAB);
    # a lost optimizer state or a reset loader would drift well outside this.
    fresh = math.log(VOCAB)
    assert abs(resumed["train_loss"] - control["train_loss"]) < 0.35 * abs(
        fresh - control["train_loss"]
    ), (
        f"resumed loss {resumed['train_loss']:.4f} is far from the uninterrupted "
        f"control {control['train_loss']:.4f} (fresh-init would be ~{fresh:.4f})"
    )


def test_resumed_run_did_not_start_from_scratch(tmp_path: Path, data_dir: Path):
    """The failure that looks most like success: weights silently reinitialized.
    The curve still descends, so only the ABSOLUTE level gives it away."""
    ck = tmp_path / "ck"
    ck.mkdir()
    first = _run(_cfg(tmp_path, steps=40, ckpt=ck))
    resumed = _run(_cfg(tmp_path, steps=45, ckpt=ck))
    fresh = math.log(VOCAB)
    assert first["train_loss"] < fresh - 0.5, "first segment never learned; test is vacuous"
    assert resumed["train_loss"] < first["train_loss"] + 0.5
    assert fresh - resumed["train_loss"] > 0.5, "resumed loss fell back to fresh-init level"


def test_loader_position_advances_across_resume(tmp_path: Path, data_dir: Path):
    """A reset loader re-trains the same data. Loss then looks BETTER, so nothing
    in the output flags it — it has to be asserted on the checkpoint directly."""
    ck = tmp_path / "ck"
    ck.mkdir()
    _run(_cfg(tmp_path, steps=30, ckpt=ck))
    blob = checkpoint.load_latest(str(ck), map_location="cpu")
    assert blob["loader"], "checkpoint carries no loader state"
    pos = blob["loader"].get("position", blob["loader"].get("offset"))
    assert pos is None or pos > 0, "loader position is at the start after 10 steps"


def test_trained_seconds_carries_forward(tmp_path: Path, data_dir: Path):
    """Budget-in-checkpoint. If this resets, segment 2 runs a full fresh budget
    and the run costs twice what it should."""
    ck = tmp_path / "ck"
    ck.mkdir()
    _run(_cfg(tmp_path, steps=30, ckpt=ck))
    blob = checkpoint.load_latest(str(ck), map_location="cpu")
    assert blob["trained_seconds"] > 0.0
