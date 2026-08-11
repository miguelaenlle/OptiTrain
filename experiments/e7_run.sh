#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
. scripts/fleetctl.sh
export MARKET=on-demand NODES=2 VCPU_QUOTA=64 MAX_EPOCHS_WITHOUT_PROGRESS=12 \
       MAX_INSTANCE_LIFETIME_SECONDS=3600 METRICS_OVERHEAD=1800 \
       WARMUP_STEPS=100 LR_DECAY_STEPS=2000 EVAL_INTERVAL_STEPS=0 EVAL_ITERS=20 \
       LOG_INTERVAL_STEPS=1 CHECKPOINT_INTERVAL_SECONDS=30 \
       TRAIN_TOTAL_SECONDS=300 FIRST_KILL=60
# region + tag scoped; can never touch us-east-2 or anything untagged
trap 'echo "[e7] trap -> reaping"; reap_ours' EXIT INT TERM
python3 experiments/e7_one_node.py 2>&1
echo "[e7] driver rc=$?"
echo "[e7] final fleet: ours=$(ours_count)"
