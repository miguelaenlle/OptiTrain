"""nanoGPT model construction for the inference worker — no training imports.

Why this exists: the worker used to reach into ``spot_train.train`` for its
~14-line ``build_model``. Importing that module *executes* it, which drags in
``distributed``, ``events``, ``data.PositionedLoader`` and
``interruption.InterruptionListener`` — a whole training dependency tree the
serving path never uses. That bloats the worker container image and slows cold
start, and cold start is a number this platform intends to measure honestly (see
docs/inference-platform-plan.md). So serving gets its own tiny builder and the
inference package stops depending on the training module tree.

This deliberately MIRRORS ``spot_train.train.build_model`` /
``_ensure_nanogpt_on_path``: same nanoGPT submodule, same ``GPTConfig`` fields,
same instantiation. If the trainer's builder changes shape, change it here too —
a served checkpoint must be byte-for-byte the artifact training produced.
"""

from __future__ import annotations

import os
import sys


def ensure_nanogpt_on_path() -> None:
    """Put the nanoGPT submodule on sys.path so ``from model import GPT`` works.

    Works for an editable install (local or on the box): repo root is three
    levels above this file (src/inference/model_build.py -> repo root), and
    nanoGPT lives at <root>/third_party/nanoGPT. Lets ``spot-worker`` and
    ``python -m inference.worker`` run without a manual PYTHONPATH.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ng = os.path.join(root, "third_party", "nanoGPT")
    if os.path.isdir(ng) and ng not in sys.path:
        sys.path.insert(0, ng)


def build_gpt(
    *,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    dropout: float,
    vocab_size: int,
):
    """Instantiate nanoGPT's GPT from the submodule. We import, never rewrite."""
    ensure_nanogpt_on_path()
    from model import GPT, GPTConfig  # type: ignore

    gpt_cfg = GPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        vocab_size=vocab_size,
    )
    return GPT(gpt_cfg)


def load_pretrained_gpt(model_type: str):
    """Load stock OpenAI GPT-2 weights into nanoGPT's GPT.

    ``model_type`` is one of ``gpt2 | gpt2-medium | gpt2-large | gpt2-xl``.
    nanoGPT pulls the weights through ``transformers`` (HF-cached after the
    first download) and forces ``block_size=1024``.
    """
    ensure_nanogpt_on_path()
    from model import GPT  # type: ignore

    return GPT.from_pretrained(model_type)
