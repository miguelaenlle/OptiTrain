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

    #: Does this backend batch requests internally?
    #:
    #: False (nanoGPT, HF) means one generate() at a time, so ``ModelService``
    #: serializes behind a lock and the router spreads load across workers.
    #:
    #: True (vLLM) means the engine schedules concurrent sequences itself, and
    #: the lock MUST be bypassed. Holding it would feed vLLM one sequence at a
    #: time, which forfeits continuous batching entirely -- the single reason
    #: for adopting it. That failure is silent: everything still works, just at
    #: roughly the throughput we already have.
    concurrent: bool

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
    concurrent = False

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
    concurrent = False

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
        # attn_implementation is pinned rather than left to the default. E1
        # measured 33ms/token against a ~7.1ms memory-bandwidth bound, i.e. ~26ms
        # of overhead per token, and eager attention is a prime suspect: it runs
        # several separate kernels per layer and materialises the full attention
        # matrix, which at 48 layers and batch 1 is dominated by launch overhead.
        # SDPA fuses that into one kernel. Note the irony worth remembering --
        # nanoGPT already used SDPA, so moving to HF for the KV cache may have
        # traded flash attention away for caching. Fall back if this transformers
        # version has no SDPA path for GPT-2.
        kwargs = {"torch_dtype": dtype, "attn_implementation": "sdpa"}
        for attempt in ({**kwargs, "low_cpu_mem_usage": True}, kwargs, {"torch_dtype": dtype}):
            try:
                model = GPT2LMHeadModel.from_pretrained(model_type, **attempt)
                break
            except (ImportError, ValueError):
                continue
        else:  # pragma: no cover - every variant failed
            raise RuntimeError(f"could not load {model_type!r} through transformers")
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


class VLLMBackend:
    """vLLM — PagedAttention, continuous batching, fused kernels.

    ⚠️ UNVERIFIED: vLLM needs CUDA + Linux, so this cannot be exercised on the
    development Mac. It has never been run. Treat the first GPU boot as the real
    test and expect to iterate.

    Why this exists, from E1/E2 rather than from principle
    (docs/experiments/e1e2-results.md): our HF worker measured 30.5 tok/s and
    $7.96 per 1M tokens -- 4.6x off the memory-bandwidth bound and ~40x off
    market rates. The two causes are per-token overhead and the absence of
    batching, and vLLM is the project that exists to fix exactly those.

    ``concurrent = True`` is the load-bearing part. vLLM schedules many
    sequences itself, so ``ModelService`` must NOT serialize calls to it. If the
    lock were held, vLLM would receive one sequence at a time and continuous
    batching -- the entire reason to adopt it -- would silently do nothing.
    """

    name = "vllm"
    concurrent = True

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def load(
        cls,
        model_type: str,
        *,
        dtype: str = "auto",
        max_model_len: int = 1024,
        gpu_memory_utilization: float = 0.90,
    ) -> VLLMBackend:
        """Build the engine. GPU-only; there is no CPU path worth having.

        ``gpu_memory_utilization`` is how much VRAM vLLM claims up front for
        weights plus the paged KV cache. Higher means more concurrent sequences
        fit, which is where the throughput comes from; too high and it fails to
        allocate. ``max_model_len`` is capped at GPT-2's context of 1024 --
        leaving it at the default would reserve cache for a context the model
        cannot use.
        """
        try:
            from vllm import LLM
        except ImportError as e:  # pragma: no cover - dependency guard
            raise RuntimeError(
                f"serving {model_type!r} on the vLLM engine needs `vllm` "
                "(pip install vllm; CUDA + Linux only)"
            ) from e

        engine = LLM(
            model=model_type,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return cls(engine)

    def generate(
        self,
        ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int | None = None,
    ) -> list[int]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            # vLLM reads -1 as "no top-k filter"; HF and nanoGPT read 0/None.
            top_k=top_k if top_k else -1,
            seed=seed,
        )
        out = self.engine.generate(prompt_token_ids=[ids], sampling_params=params)
        return list(out[0].outputs[0].token_ids)
