# Inference worker — FastAPI + torch serving one model.
#
# One Dockerfile, two bases. Locally (k3d, Apple silicon) we want a small CPU
# image; on a g5.xlarge we need the CUDA runtime. Rather than fork the file,
# BASE_IMAGE is an ARG:
#
#   CPU   docker build -f deploy/docker/worker.Dockerfile .
#   GPU   docker build -f deploy/docker/worker.Dockerfile \
#           --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime .
#
# ⚠️ Cross-arch: a Mac builds arm64 by default, EC2 g5 is amd64. An arm64 image
# fails on EC2 with "exec format error". Always build cloud images with
# --platform linux/amd64 (see deploy/docker/build.sh).

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

# torch is the bulk of the image and the slowest layer to build, so it is
# installed before any source is copied -- editing our code must not invalidate
# it. The CPU wheel index keeps the default image ~2GB instead of ~7GB; on the
# CUDA base this is skipped because torch is already present.
ARG TORCH_SPEC="torch>=2.2 --index-url https://download.pytorch.org/whl/cpu"
RUN python -c "import torch" 2>/dev/null || pip install --no-cache-dir ${TORCH_SPEC}

WORKDIR /app

# Runtime deps, still ahead of the source copy for the same caching reason.
RUN pip install --no-cache-dir \
      fastapi uvicorn numpy boto3 requests tiktoken \
      transformers accelerate prometheus-client

# nanoGPT is a git submodule, needed only for the trained-checkpoint path
# (stock GPT-2 goes through transformers and never touches it). It is copied
# rather than cloned so the image is reproducible from the pinned commit.
COPY third_party/nanoGPT /app/third_party/nanoGPT

COPY src /app/src
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PORT=8001 \
    HF_HOME=/models

# Model weights land here. Mount a volume to persist them across restarts --
# otherwise every pod restart re-downloads gpt2-xl (~6GB), which would show up
# in the cold-start number this platform reports.
VOLUME ["/models"]

EXPOSE 8001

# Non-root: nothing here needs privileges, and the k8s manifests set
# runAsNonRoot to match.
RUN useradd -u 10001 -m worker && mkdir -p /models && chown -R worker /models /app
USER worker

# No HEALTHCHECK on purpose -- Kubernetes owns liveness/readiness via probes
# against /healthz, and two competing health systems is one too many.
CMD ["python", "-m", "inference.worker"]
