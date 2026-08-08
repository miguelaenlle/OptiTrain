#!/usr/bin/env bash
# Build the router + worker images.
#
# Why this script exists rather than two docker build lines: this repo is
# developed on Apple silicon (arm64) and deployed to g5.xlarge (amd64). An
# image built with the local default runs fine in k3d and then dies on EC2 with
# "exec format error". Getting that wrong costs a cloud debugging cycle at
# ~$2/hr, so the target platform is explicit here, always.
#
#   ./build.sh                 local dev: native arch, CPU worker
#   ./build.sh --cloud         amd64 + CUDA worker base, for EC2
#   ./build.sh --push <repo>   also push (e.g. an ECR repo URI)
#
# Run from the repo root (the build context must include src/ and third_party/).

set -euo pipefail

TAG="${TAG:-dev}"
PLATFORM=""
WORKER_BASE="python:3.11-slim"
TORCH_SPEC="torch>=2.2 --index-url https://download.pytorch.org/whl/cpu"
PUSH=""
REPO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --cloud)
      PLATFORM="linux/amd64"
      # The CUDA base already ships torch, so the pip install is a no-op there.
      WORKER_BASE="pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
      TORCH_SPEC="torch>=2.2"
      shift ;;
    --push) PUSH="yes"; REPO="${2:?--push needs a repo prefix}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/../.."   # repo root = build context
[ -f pyproject.toml ] || { echo "must run from the repo (pyproject.toml not found)" >&2; exit 1; }

if [ ! -f third_party/nanoGPT/model.py ]; then
  echo "third_party/nanoGPT is empty — run: git submodule update --init" >&2
  exit 1
fi

PLAT_ARGS=()
[ -n "$PLATFORM" ] && PLAT_ARGS=(--platform "$PLATFORM")
PREFIX=""
[ -n "$REPO" ] && PREFIX="${REPO%/}/"

echo "==> router  (${PLATFORM:-native})"
docker buildx build "${PLAT_ARGS[@]}" \
  -f deploy/docker/router.Dockerfile \
  -t "${PREFIX}fleet-router:${TAG}" \
  ${PUSH:+--push} ${PUSH:---load} \
  .

echo "==> worker  (${PLATFORM:-native}, base=$WORKER_BASE)"
docker buildx build "${PLAT_ARGS[@]}" \
  -f deploy/docker/worker.Dockerfile \
  --build-arg BASE_IMAGE="$WORKER_BASE" \
  --build-arg TORCH_SPEC="$TORCH_SPEC" \
  -t "${PREFIX}fleet-worker:${TAG}" \
  ${PUSH:+--push} ${PUSH:---load} \
  .

echo
echo "built:"
echo "  ${PREFIX}fleet-router:${TAG}"
echo "  ${PREFIX}fleet-worker:${TAG}"
[ -z "$PLATFORM" ] && cat <<'EOF'

NOTE: built for the native architecture. These run in k3d but NOT on EC2.
Use --cloud for anything that will land on a g5.xlarge.
EOF
