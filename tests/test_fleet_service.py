"""Serving dtype policy + the two-loader dispatch, with no weights on disk.

Everything here is CPU-only and never touches a checkpoint or the HF hub: the
loaders are replaced with sentinels so we test the *decision* (which path, which
dtype), not the several-GB download behind the pretrained path.
"""

import pytest
import torch

from inference.service import ModelNotReady, ModelService, ServeSettings, resolve_serve_dtype
from spot_train.config import TrainConfig

# --- dtype policy ---------------------------------------------------------


def test_cpu_auto_stays_float32():
    """CPU must stay fp32 or the local determinism tests stop being exact."""
    assert resolve_serve_dtype("cpu", "auto") is torch.float32
    assert resolve_serve_dtype("cpu") is torch.float32


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float16", torch.float16),
        ("fp16", torch.float16),
    ],
)
def test_explicit_dtype_is_honored_on_any_device(device, name, expected):
    assert resolve_serve_dtype(device, name) is expected


def test_unknown_dtype_names_the_valid_options():
    with pytest.raises(ValueError) as e:
        resolve_serve_dtype("cuda", "float8")
    msg = str(e.value)
    assert "float8" in msg
    for option in ("auto", "float32", "bfloat16", "float16"):
        assert option in msg


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), torch.bfloat16),  # Hopper
        ((8, 6), torch.bfloat16),  # A10G / Ampere — the g5.xlarge target
        ((7, 5), torch.float16),  # Turing/T4: bf16 runs off the tensor cores
    ],
)
def test_cuda_auto_follows_the_ampere_capability_rule(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert resolve_serve_dtype("cuda", "auto") is expected


# --- settings -------------------------------------------------------------


def test_serve_settings_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("PRETRAINED_MODEL", raising=False)
    monkeypatch.delenv("SERVE_DTYPE", raising=False)
    s = ServeSettings.from_env()
    assert s.pretrained_model == ""
    assert s.serve_dtype == "auto"


def test_serve_settings_reads_env(monkeypatch):
    monkeypatch.setenv("PRETRAINED_MODEL", "gpt2-xl")
    monkeypatch.setenv("SERVE_DTYPE", "bfloat16")
    s = ServeSettings.from_env()
    assert s.pretrained_model == "gpt2-xl"
    assert s.serve_dtype == "bfloat16"


# --- load dispatch --------------------------------------------------------


@pytest.fixture
def loaders(monkeypatch):
    """Replace both loaders with sentinels that record how they were called."""
    calls = {}

    def fake_pretrained(model_type, *, device, dtype, engine="hf"):
        calls["pretrained"] = {
            "model_type": model_type,
            "device": device,
            "dtype": dtype,
            "engine": engine,
        }
        return "PRETRAINED"

    def fake_checkpoint(cfg, *, device, dtype):
        calls["checkpoint"] = {"cfg": cfg, "device": device, "dtype": dtype}
        return "CHECKPOINT"

    monkeypatch.setattr(ModelService, "_load_pretrained", staticmethod(fake_pretrained))
    monkeypatch.setattr(ModelService, "_load_checkpoint", staticmethod(fake_checkpoint))
    return calls


def test_pretrained_env_selects_the_pretrained_loader(monkeypatch, loaders):
    monkeypatch.setenv("PRETRAINED_MODEL", "gpt2-xl")
    monkeypatch.setenv("SERVE_DTYPE", "float16")
    cfg = TrainConfig(device="cpu")

    assert ModelService.load(cfg) == "PRETRAINED"
    assert "checkpoint" not in loaders
    assert loaders["pretrained"] == {
        "model_type": "gpt2-xl",
        "device": "cpu",
        "dtype": torch.float16,
        "engine": "hf",  # KV-cached engine is the default for stock weights
    }


def test_unset_pretrained_falls_back_to_the_checkpoint_loader(monkeypatch, loaders):
    monkeypatch.delenv("PRETRAINED_MODEL", raising=False)
    monkeypatch.delenv("SERVE_DTYPE", raising=False)
    cfg = TrainConfig(device="cpu")

    assert ModelService.load(cfg) == "CHECKPOINT"
    assert "pretrained" not in loaders
    assert loaders["checkpoint"]["cfg"] is cfg
    assert loaders["checkpoint"]["device"] == "cpu"
    assert loaders["checkpoint"]["dtype"] is torch.float32  # cpu + auto


def test_explicit_settings_beat_the_environment(monkeypatch, loaders):
    """An injected ServeSettings wins — the worker can be driven from code."""
    monkeypatch.setenv("PRETRAINED_MODEL", "gpt2-xl")
    ModelService.load(TrainConfig(device="cpu"), ServeSettings())
    assert "pretrained" not in loaders
    assert "checkpoint" in loaders


# --- pretrained preconditions --------------------------------------------


def test_pretrained_without_tiktoken_is_not_ready(monkeypatch):
    """Codec first: no BPE means no serving, and we say so before downloading
    six gigabytes of weights."""
    monkeypatch.setattr("inference.service._bpe_codec", lambda: None)
    monkeypatch.setattr(
        "inference.service.load_pretrained_gpt",
        lambda mt: pytest.fail("must not load weights without a codec"),
    )
    with pytest.raises(ModelNotReady) as e:
        ModelService._load_pretrained("gpt2-xl", device="cpu", dtype=torch.float32)
    assert "tiktoken" in str(e.value)


def test_pretrained_load_failure_becomes_not_ready(monkeypatch):
    monkeypatch.setattr("inference.service._bpe_codec", lambda: (lambda s: [0], lambda i: ""))

    def boom(model_type):
        raise AssertionError(model_type)

    monkeypatch.setattr("inference.service.load_pretrained_gpt", boom)
    with pytest.raises(ModelNotReady) as e:
        ModelService._load_pretrained("gpt2-xxl", device="cpu", dtype=torch.float32)
    assert "gpt2-xxl" in str(e.value)
