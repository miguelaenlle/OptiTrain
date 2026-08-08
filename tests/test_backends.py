"""Backend seam: engine selection, and HF/nanoGPT equivalence.

The equivalence tests need stock GPT-2 weights. Rather than make the suite
download ~500MB, they run only when the weights are already in the HF cache and
skip otherwise, so a clean checkout stays fast. Force them with
``HF_MODEL_TESTS=1``.
"""

from __future__ import annotations

import os

import pytest
import torch

from inference.backends import HFBackend, NanoGPTBackend
from inference.service import ModelService, ServeSettings, resolve_serve_dtype

GREEDY = {"temperature": 1.0, "top_k": 1, "seed": 0}
"""top_k=1 leaves a single unmasked logit, so softmax is one-hot and the
multinomial draw is deterministic — greedy on BOTH engines, which is what makes
them comparable token-for-token."""


def _gpt2_cached() -> bool:
    if os.environ.get("HF_MODEL_TESTS") == "1":
        return True
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(hub, "models--gpt2"))


needs_gpt2 = pytest.mark.skipif(
    not _gpt2_cached(), reason="stock gpt2 weights not in the HF cache (set HF_MODEL_TESTS=1)"
)


# --- engine selection (no weights needed) ---------------------------------


def test_serve_engine_defaults_to_hf(monkeypatch):
    monkeypatch.delenv("SERVE_ENGINE", raising=False)
    assert ServeSettings.from_env().serve_engine == "hf"


def test_serve_engine_from_env(monkeypatch):
    monkeypatch.setenv("SERVE_ENGINE", "nanogpt")
    assert ServeSettings.from_env().serve_engine == "nanogpt"


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="SERVE_ENGINE"):
        ModelService._load_pretrained(
            "gpt2", device="cpu", dtype=torch.float32, engine="tensorflow"
        )


def test_hf_backend_reports_missing_transformers(monkeypatch):
    """The dependency error must name the fix, not surface as an AttributeError."""
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "transformers":
            raise ImportError("no transformers")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="transformers"):
        HFBackend.load("gpt2", device="cpu", dtype=torch.float32)


# --- equivalence (needs weights) ------------------------------------------


@pytest.fixture(scope="module")
def services():
    dtype = resolve_serve_dtype("cpu", "auto")
    hf = ModelService._load_pretrained("gpt2", device="cpu", dtype=dtype, engine="hf")
    ng = ModelService._load_pretrained("gpt2", device="cpu", dtype=dtype, engine="nanogpt")
    return hf, ng


@needs_gpt2
def test_backends_are_the_expected_engines(services):
    hf, ng = services
    assert isinstance(hf.backend, HFBackend) and hf.backend.name == "hf"
    assert isinstance(ng.backend, NanoGPTBackend) and ng.backend.name == "nanogpt"


@needs_gpt2
@pytest.mark.parametrize(
    "prompt",
    ["The capital of France is", "In a shocking finding, scientists", "def bfs(graph, start):"],
)
def test_hf_matches_nanogpt_token_for_token(services, prompt):
    """Same weights, greedy decode => identical output. Not 'similar' — identical.

    This is the test that makes every downstream number trustworthy: it proves
    the KV cache changed the cost of generation, not its result.
    """
    hf, ng = services
    a = hf.complete(prompt, max_new_tokens=32, **GREEDY)
    b = ng.complete(prompt, max_new_tokens=32, **GREEDY)
    assert a["text"] == b["text"]
    assert a["completion_tokens"] == b["completion_tokens"] == 32


@needs_gpt2
def test_backend_returns_completion_only_not_the_prompt(services):
    """The prompt slice happens inside the backend; a regression here would
    silently prepend the prompt to every completion."""
    hf, _ = services
    out = hf.complete("The capital of France is", max_new_tokens=8, **GREEDY)
    assert not out["text"].startswith("The capital of France is")
    assert out["prompt_tokens"] == 5  # correct GPT-2 BPE tokenization


@needs_gpt2
def test_tiktoken_agrees_with_hf_tokenizer():
    """We encode with tiktoken but generate with an HF model. If the two BPEs
    ever disagreed, we would feed the model ids from a different vocabulary."""
    transformers = pytest.importorskip("transformers")
    from spot_train.sampling import _bpe_codec

    codec = _bpe_codec()
    assert codec is not None
    encode, _ = codec
    hf_tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    for s in ["The capital of France is", "def bfs(graph, start):", "hello  world\n\n"]:
        assert encode(s) == hf_tok.encode(s), f"tokenizer mismatch on {s!r}"
