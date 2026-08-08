"""Generation backends — one interface, two engines.

``ModelService`` should not know which engine produced a completion, so both
engines expose the same tiny call and return **only the completion ids** (the
prompt is sliced off inside the backend, where the engine-specific output shape
is known).

Why two engines at all:

* **nanoGPT** (:class:`NanoGPTBackend`) serves *trained checkpoints*. It must
  stay, because a served checkpoint has to be byte-for-byte the artifact the
  training run produced.
* **HuggingFace** (:class:`HFBackend`) serves *stock GPT-2*. nanoGPT's
  ``generate()`` re-runs the **entire sequence every token** — no KV cache — so
  producing 64 tokens from a 5-token prompt costs 2,336 token-positions of work
  instead of 69, ~34x more, and the gap grows quadratically with output length.
  ``third_party/nanoGPT`` is a pinned, read-only submodule ("we import, never
  rewrite"), so rather than reimplement attention we hand stock weights to
  ``transformers``, which has a cache. For stock GPT-2 the nanoGPT class buys us
  nothing anyway: same architecture, same weights.

Greedy decoding is identical across both by construction: ``top_k=1`` leaves a
single unmasked logit, so the softmax is one-hot and the multinomial draw is
deterministic. That is what makes the two engines comparable token-for-token
(see ``tests/test_backends.py``).
"""

from __future__ import annotations

from typing import Protocol

import torch


class Backend(Protocol):
    """Generate a completion from prompt ids. Returns completion ids only."""

    def generate(
        self,
        ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int | None = None,
    ) -> list[int]: ...


class NanoGPTBackend:
    """nanoGPT's ``GPT.generate`` — no KV cache, full recompute per token."""

    name = "nanogpt"

    def __init__(self, model, *, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int | None = None,
    ) -> list[int]:
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)[None, ...]
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k or None)
        return out[0, len(ids) :].tolist()


class HFBackend:
    """``transformers`` ``GPT2LMHeadModel.generate`` with ``use_cache=True``."""

    name = "hf"

    def __init__(self, model, *, device: str):
        self.model = model
        self.device = device

    @classmethod
    def load(cls, model_type: str, *, device: str, dtype: torch.dtype) -> HFBackend:
        """Load stock GPT-2 weights straight into a HF model.

        ``low_cpu_mem_usage`` + ``torch_dtype`` matter more than they look: the
        old path built a nanoGPT ``GPT`` *and* a HF model in fp32 before copying
        weights across, peaking near 12.4GB of host RAM for gpt2-xl. A
        g5.xlarge has 16GiB, and that peak happens on the CPU side before
        ``.to(cuda)``. Loading once, in the serving dtype, keeps the peak near
        the model's own size.
        """
        try:
            from transformers import GPT2LMHeadModel
        except ImportError as e:  # pragma: no cover - dependency guard
            raise RuntimeError(
                f"serving {model_type!r} on the HF engine needs `transformers` "
                "(pip install -e '.[fleet]')"
            ) from e

        # low_cpu_mem_usage needs `accelerate`. It is worth having (it loads
        # shard-by-shard into a meta model, so peak ~= model size instead of 2x),
        # but it must not be a hard requirement: without it we still load in the
        # serving dtype, which is the larger win. gpt2-xl in fp16 peaks ~6.2GB
        # either way, comfortably inside a g5.xlarge's 16GiB.
        try:
            model = GPT2LMHeadModel.from_pretrained(
                model_type, torch_dtype=dtype, low_cpu_mem_usage=True
            )
        except ImportError:
            model = GPT2LMHeadModel.from_pretrained(model_type, torch_dtype=dtype)
        model.to(device)
        model.eval()
        return cls(model, device=device)

    @torch.no_grad()
    def generate(
        self,
        ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int | None = None,
    ) -> list[int]:
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)[None, ...]
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(
            idx,
            attention_mask=torch.ones_like(idx),  # silences HF's no-mask warning
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k or 0,  # HF reads 0 as "no top-k filter"
            use_cache=True,  # the entire point of this backend
            pad_token_id=self.model.config.eos_token_id,
        )
        return out[0, len(ids) :].tolist()
