"""Stage the dataset to S3 once, so every training box pulls identical bytes.

Runs nanoGPT's ``prepare.py`` locally (if the bins aren't already present), then
uploads ``train.bin``/``val.bin``/``meta.pkl`` to ``s3://<bucket>/data/<dataset>/``.
Idempotent: if the objects already exist in S3, it does nothing. shakespeare_char
is tiny; the same flow works for OpenWebText (prepared offline).

Sized for the FULL OpenWebText bin (~17 GB): uploads go through boto3's managed
transfer (threaded multipart — a single S3 PUT caps at 5 GB), and each file is
skipped individually when the object already matches the local size, so a
staging run interrupted after ``train.bin`` doesn't push those 17 GB twice.
"""

from __future__ import annotations

import os
import subprocess
import sys

from . import aws
from .config import OrchestratorConfig

# train/val bins are required; meta.pkl is only for char-level datasets (the BPE
# corpora like OpenWebText have a fixed vocab and ship no meta).
_REQUIRED = ("train.bin", "val.bin")
_OPTIONAL = ("meta.pkl",)


def _local_dir(cfg: OrchestratorConfig) -> str:
    """Where this dataset's ``prepare.py`` + bins live. Prefer a repo-level
    ``data/<dataset>/`` (our own preps, e.g. the capped OpenWebText slice) over
    nanoGPT's ``third_party/nanoGPT/data/<dataset>/`` (the submodule fixtures)."""
    ours = f"data/{cfg.dataset}"
    if os.path.exists(os.path.join(ours, "prepare.py")):
        return ours
    return f"third_party/nanoGPT/data/{cfg.dataset}"


def _already_staged(cfg: OrchestratorConfig, path: str, key: str) -> bool:
    """True when S3 already holds this exact file. An S3 object only appears once
    its (multipart) upload completes, so a matching size means a finished upload;
    a differing size means the local bins were re-prepared and must be re-pushed."""
    remote = aws.object_size(cfg.bucket, key)
    if remote is None:
        return False
    local = os.path.getsize(path)
    if remote == local:
        return True
    print(
        f"[stage-data] s3://{cfg.bucket}/{key} is {remote:,} B but the local file is "
        f"{local:,} B — re-uploading",
        file=sys.stderr,
    )
    return False


def stage_data(cfg: OrchestratorConfig) -> None:
    cfg.require_bucket()
    prefix = f"{cfg.data_prefix}/{cfg.dataset}"

    if all(aws.object_exists(cfg.bucket, f"{prefix}/{f}") for f in _REQUIRED):
        print(f"[stage-data] {cfg.data_uri()} already present — nothing to do", file=sys.stderr)
        return

    data_dir = _local_dir(cfg)
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in _REQUIRED):
        print(f"[stage-data] running prepare.py in {data_dir}", file=sys.stderr)
        subprocess.run([sys.executable, "prepare.py"], cwd=data_dir, check=True)

    uploaded, skipped = [], []
    for f in (*_REQUIRED, *_OPTIONAL):
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            continue
        key = f"{prefix}/{f}"
        # Per-file, not all-or-nothing: at OpenWebText scale one bin is 17 GB, so
        # a resumed staging must not re-send what already landed.
        if _already_staged(cfg, path, key):
            skipped.append(f)
            continue
        aws.upload_file(path, cfg.bucket, key)
        uploaded.append(f)
    tail = f" (already staged: {skipped})" if skipped else ""
    print(f"[stage-data] uploaded {uploaded} to {cfg.data_uri()}{tail}", file=sys.stderr)
