"""Model service for the inference worker — load once, generate serially.

Two ways to get a model, one way to serve it:

  - **checkpoint** (default): the newest checkpoint under ``CHECKPOINT_URI``,
    with the char codec from the dataset's ``meta.pkl`` — so the served model is
    byte-for-byte the artifact a training run produced;
  - **pretrained** (``PRETRAINED_MODEL=gpt2-xl``): stock OpenAI GPT-2 weights,
    tokenized with the GPT-2 BPE. Needs no checkpoint, no dataset and no
    ``meta.pkl`` — it exists so the fleet can be benchmarked on a model big
    enough that the numbers measure inference rather than FastAPI overhead.

Both converge on the same ``ModelService`` and the same ``complete()``; only the
*engine* underneath differs (see :mod:`inference.backends`). Stock weights are
served through ``transformers`` because it has a **KV cache** — nanoGPT's
``generate()`` re-runs the whole sequence every token — while checkpoints stay on
nanoGPT so a served model is byte-for-byte the training artifact.

Generation holds a lock: one request at a time per worker, and the router
spreads load across workers. Batching several requests into one forward pass is
still to come (ROADMAP Part 4).
"""

from __future__ import annotations

import contextlib
import os
import pickle
import shutil
import threading
import time
from dataclasses import dataclass, field

import torch

from spot_train import checkpoint, s3_store
from spot_train.config import TrainConfig
from spot_train.sampling import _bpe_codec, _char_codec

from .backends import HFBackend, NanoGPTBackend
from .model_build import build_gpt, load_pretrained_gpt


class ModelNotReady(RuntimeError):
    """The worker cannot serve: missing checkpoint or dataset metadata."""


class PromptError(ValueError):
    """The request prompt cannot be encoded with this model's vocabulary."""


def _ensure_meta(data_local_dir: str, data_uri: str) -> None:
    """Make sure meta.pkl (the char codec + vocab size) exists locally.

    The worker doesn't need train/val bins — only the codec — so this pulls a
    single small file instead of reusing PositionedLoader's full download.
    """
    meta_path = os.path.join(data_local_dir, "meta.pkl")
    if os.path.exists(meta_path):
        return
    if not data_uri:
        raise ModelNotReady(f"no meta.pkl in {data_local_dir} and DATA_URI is unset")
    os.makedirs(data_local_dir, exist_ok=True)
    ref = data_uri.rstrip("/") + "/meta.pkl"
    with s3_store.fetch(ref) as local:
        if local != meta_path:
            shutil.copyfile(local, meta_path)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


_SERVE_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_serve_dtype(device: str, name: str = "auto") -> torch.dtype:
    """Pick the dtype the weights are *stored* in for serving.

    Unlike training (autocast around fp32 master weights) inference can just
    hold the weights in half precision: decoding is memory-bandwidth-bound —
    every token re-reads the whole parameter tensor — so halving the weights
    roughly doubles tokens/s and halves VRAM (gpt2-xl 6.2 GB -> 3.1 GB, which is
    what makes it comfortable on a 24 GB A10G alongside activations).

    "auto" mirrors the capability rule in the trainer's ``_resolve_amp``: bf16
    only on Ampere+ (compute capability >= 8.0) where it has a tensor-core path,
    else fp16 — bf16 on Turing (T4, sm_75) runs off the tensor cores at ~fp32
    speed. CPU stays fp32 so the local tests stay bit-exact.
    """
    name = (name or "auto").lower()
    if name in ("auto", ""):
        if device.split(":")[0] != "cuda":
            return torch.float32
        # bf16 tensor cores are Ampere+ (sm_80); Turing/T4 (sm_75) needs fp16.
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    try:
        return _SERVE_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown SERVE_DTYPE {name!r}; valid options: auto, float32, bfloat16, float16"
        ) from None


@dataclass
class ServeSettings:
    """Serving-only knobs, separate from TrainConfig (which is the trainer's)."""

    pretrained_model: str = ""  # e.g. "gpt2-xl"; empty => serve a checkpoint
    serve_dtype: str = "auto"
    # Engine for the PRETRAINED path only. "hf" gets a KV cache (~34x less work
    # at 64 tokens); "nanogpt" is the old full-recompute path, kept so the two
    # can be compared token-for-token and as a fallback. Checkpoints always use
    # nanoGPT — they must stay byte-for-byte the training artifact.
    serve_engine: str = "hf"

    @classmethod
    def from_env(cls) -> ServeSettings:
        return cls(
            pretrained_model=os.environ.get("PRETRAINED_MODEL", ""),
            serve_dtype=os.environ.get("SERVE_DTYPE", "auto"),
            serve_engine=os.environ.get("SERVE_ENGINE", "hf"),
        )


def gpu_stats() -> dict:
    """Best-effort GPU telemetry for /stats. Empty dict on CPU boxes; memory
    works everywhere CUDA does; utilization needs pynvml (present on the
    DLAMI) and is skipped silently if unavailable."""
    if not torch.cuda.is_available():
        return {}
    out: dict = {}
    with contextlib.suppress(Exception):
        free, total = torch.cuda.mem_get_info()
        out["gpu_mem_used_mb"] = round((total - free) / 1e6)
        out["gpu_mem_total_mb"] = round(total / 1e6)
    with contextlib.suppress(Exception):
        out["gpu_util"] = torch.cuda.utilization()  # % over the last sample window
    return out


@dataclass
class ServiceStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generate_seconds: float = 0.0
    # Gauge: requests inside complete() right now. Generation is serialized
    # behind a lock, so at most one is generating — the rest are the queue.
    in_flight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1

    def leave(self) -> None:
        with self._lock:
            self.in_flight -= 1

    def record(self, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.generate_seconds += seconds

    def snapshot(self) -> dict:
        with self._lock:
            tok_s = self.completion_tokens / self.generate_seconds if self.generate_seconds else 0.0
            return {
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "generate_seconds": round(self.generate_seconds, 3),
                "tokens_per_second": round(tok_s, 1),
                "in_flight": self.in_flight,
                "queued": max(self.in_flight - 1, 0),
            }


class ModelService:
    """A loaded model + codec with a serialized ``complete()``."""

    def __init__(self, backend, encode, decode, *, device: str, model_name: str, ckpt_step: int):
        self.backend = backend
        self.encode = encode
        self.decode = decode
        self.device = device
        self.model_name = model_name
        self.ckpt_step = ckpt_step
        self.stats = ServiceStats()
        self._gen_lock = threading.Lock()

    @classmethod
    def load(cls, cfg: TrainConfig, serve: ServeSettings | None = None) -> ModelService:
        """Load a model for serving — pretrained if asked for, else a checkpoint.

        One dispatcher, two loaders; both return a service with the same
        ``complete()``.
        """
        serve = serve or ServeSettings.from_env()
        device = resolve_device(cfg.device)
        dtype = resolve_serve_dtype(device, serve.serve_dtype)
        if serve.pretrained_model:
            return cls._load_pretrained(
                serve.pretrained_model, device=device, dtype=dtype, engine=serve.serve_engine
            )
        return cls._load_checkpoint(cfg, device=device, dtype=dtype)

    @classmethod
    def _load_pretrained(
        cls, model_type: str, *, device: str, dtype: torch.dtype, engine: str = "hf"
    ) -> ModelService:
        """Serve stock OpenAI GPT-2 weights (``gpt2`` … ``gpt2-xl``).

        No checkpoint, no dataset, no meta.pkl — the tokenizer is the GPT-2 BPE
        these weights were trained with, so the only external artifact is the HF
        weight download (cached after the first boot).

        ``engine="hf"`` (default) serves through ``transformers`` and therefore
        gets a KV cache; ``"nanogpt"`` keeps the original full-recompute path.
        """
        codec = _bpe_codec()
        if codec is None:
            raise ModelNotReady(
                f"serving {model_type} needs the GPT-2 BPE, but tiktoken is not installed"
            )
        encode, decode = codec
        if engine not in ("hf", "nanogpt"):
            raise ValueError(f"unknown SERVE_ENGINE {engine!r}; valid options: hf, nanogpt")
        try:
            if engine == "hf":
                backend = HFBackend.load(model_type, device=device, dtype=dtype)
            else:
                model = load_pretrained_gpt(model_type)
                model.to(device=device, dtype=dtype)
                model.eval()
                backend = NanoGPTBackend(model, device=device)
        except (AssertionError, ImportError, OSError, RuntimeError) as e:
            raise ModelNotReady(
                f"could not load pretrained {model_type!r}: {e} "
                "(valid: gpt2, gpt2-medium, gpt2-large, gpt2-xl; needs `transformers` "
                "installed and network/HF-cache access for the weights)"
            ) from e
        print(
            f"[worker] serving pretrained {model_type} on {device} as {dtype} "
            f"via {engine} (kv_cache={engine == 'hf'})",
            flush=True,
        )
        return cls(
            backend,
            encode,
            decode,
            device=device,
            model_name=model_type,
            ckpt_step=0,
        )

    @classmethod
    def _load_checkpoint(cls, cfg: TrainConfig, *, device: str, dtype: torch.dtype) -> ModelService:
        """Load the newest checkpoint under ``cfg.checkpoint_uri`` for serving."""
        _ensure_meta(cfg.data_local_dir, cfg.data_uri)
        with open(os.path.join(cfg.data_local_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        vocab_size = int(meta["vocab_size"])
        codec = _char_codec(cfg.data_local_dir)
        if codec is None:
            raise ModelNotReady(f"meta.pkl in {cfg.data_local_dir} has no stoi/itos codec")
        encode, decode = codec

        blob = checkpoint.load_latest(cfg.checkpoint_uri, map_location=device)
        if blob is None:
            raise ModelNotReady(f"no checkpoint found under {cfg.checkpoint_uri}")
        model = build_gpt(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            vocab_size=vocab_size,
        )
        model.load_state_dict(blob["model"])
        model.to(device=device, dtype=dtype)
        model.eval()
        step = int(blob["step"])
        print(f"[worker] serving {cfg.run_id} step {step} on {device} as {dtype}", flush=True)
        return cls(
            NanoGPTBackend(model, device=device),
            encode,
            decode,
            device=device,
            model_name=f"{cfg.run_id}@step{step}",
            ckpt_step=step,
        )

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int | None = None,
    ) -> dict:
        """Generate one completion. Returns text + token counts.

        ``temperature`` is clamped to a small positive value (nanoGPT divides
        logits by it, so exactly 0 is undefined; near-0 ≈ greedy).
        """
        text = prompt or "\n"  # generate() requires a non-empty idx
        try:
            ids = self.encode(text)
        except KeyError as e:
            raise PromptError(f"prompt contains characters outside the model vocab: {e}") from e
        temperature = max(float(temperature), 1e-5)

        # Gauge covers the wait on the lock too: in_flight - 1 = queue depth.
        self.stats.enter()
        try:
            start = time.monotonic()
            with self._gen_lock:
                completion_ids = self.backend.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k or None,
                    seed=seed,
                )
            elapsed = time.monotonic() - start
        finally:
            self.stats.leave()

        completion = self.decode(completion_ids)
        self.stats.record(len(ids), len(completion_ids), elapsed)
        return {
            "text": completion,
            "prompt_tokens": len(ids),
            "completion_tokens": len(completion_ids),
            "generate_seconds": elapsed,
        }
