#!/bin/bash
cd "$(dirname "$0")/.."
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
export MARKET=on-demand NODES=4 VCPU_QUOTA=64 MAX_INSTANCE_LIFETIME_SECONDS=7200 \
       METRICS_OVERHEAD=1800 WARMUP_STEPS=100 LR_DECAY_STEPS=2000 \
       EVAL_INTERVAL_STEPS=0 EVAL_ITERS=50 LOG_INTERVAL_STEPS=1
reap() { . "$(dirname "$0")/../scripts/fleetctl.sh"; reap_ours; }
trap reap EXIT INT TERM
TRAIN_TOTAL_SECONDS=300 PREEMPT_COUNT=1 PREEMPT_VICTIMS=3 PREEMPT_AFTER=120 \
  PYTHONPATH=src python3 -m orchestrator multinode-preempt
