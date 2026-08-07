# Scaling benchmark — runbook

A **fast, cheap, controlled** measurement of how throughput scales with node
count. Small world first (1, 2, 4), then optionally 8. Designed to finish in
~25 minutes for ~$1.20 so it can be re-run whenever the system changes, rather
than being a once-a-project event.

Everything below has been measured on this repo except where marked *projected*.

---

## What it measures, and what it deliberately does not

**Measures:** steady-state **ms/step** and **tok/s** at each world size, with an
identical model, dataset, seed and — critically — an identical **global batch**.
Gradient accumulation is recomputed per world size (`K = ceil(global / (world x
micro))`), so 1, 2, 4 and 8 nodes all take the *same optimizer steps on the same
batch*. The only variable is how many machines share the work, so any difference
in wall-clock is scaling behaviour and nothing else.

**Does not measure:** convergence. This runs in **throughput mode** (no
`TARGET_LOSS`), which is what makes it minutes instead of hours. A
time-to-target sweep has to let the *slowest* world reach the target, so the
1-node leg alone runs ~8x the 8-node leg — that is a ~5 hour, ~$20 experiment.
Use time-to-target when you need a convergence claim; use this for scaling.

**Expected shape**, from the time-to-target ladder already run on this repo:

| nodes | speedup | efficiency | $/unit work |
|------:|--------:|-----------:|------------:|
| 1 | 1.00x | 100% | 1.00x |
| 2 | 1.79x |  90% | 1.12x |
| 4 | 2.80x |  70% | 1.43x |
| 8 | 4.01x |  50% | 2.00x |

Efficiency decays because gradient all-reduce grows with world size while
per-node compute shrinks, over plain TCP (no EFA on g4dn/g5). If this run
reproduces roughly that curve, the system is behaving; a sharp deviation is the
signal worth chasing.

---

## Preconditions

```bash
cd <repo>
git checkout phase1/gpt2-owt-baseline && git pull   # boxes clone THIS branch
python3 -m pytest tests/ -q                          # expect: all pass
aws ec2 describe-instances \
  --filters Name=instance-state-name,Values=running,pending \
  --query 'length(Reservations[].Instances[])' --output text   # expect: 0
```

- `.env` present with `SPOT_TRAIN_BUCKET`, and `REPO_BRANCH=phase1/gpt2-owt-baseline`.
- Dataset staged: `s3://<bucket>/data/openwebtext/{train,val}.bin` (~17 GB + 8.5 MB).
- G/VT on-demand quota >= 16 vCPU for 1/2/4 (4 nodes x 4 vCPU). 8 nodes needs 32.
- `WANDB_API_KEY` in `.env` if you want the W&B mirror. Optional —
  `profile.json` in S3 is the source of truth and the run works without it.

---

## Run it

```bash
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
export MARKET=on-demand NODE_COUNTS=1,2,4 VCPU_QUOTA=64 \
       SCALING_CAP_SECONDS=300 MAX_INSTANCE_LIFETIME_SECONDS=7200 \
       WARMUP_STEPS=100 LR_DECAY_STEPS=2000
unset TARGET_LOSS                 # <- absence of a target IS throughput mode

spot-orchestrate scaling-clean --dry-run   # verify the recipe echo first
spot-orchestrate scaling-clean
```

`--dry-run` is not optional discipline — it prints the resolved recipe, and a
wrong `eval_interval` or `global_batch` there is the difference between a valid
experiment and 25 wasted minutes. Confirm `global_batch: 480`, `throughput_only:
True`, `node_counts: 1,2,4`.

**Budget:** ~25 min wall clock, ~$1.20. Per leg: ~160s boot (EC2 + ~35s dataset
pull + setup) + 300s training + ~60s teardown. Legs run **sequentially** —
deliberately, so no two worlds compete for S3 bandwidth or NIC while being
timed.

**To extend to 8 nodes** once the small sweep looks right:
`NODE_COUNTS=1,2,4,8`, needs 32 vCPU, adds ~9 min and ~$1.

---

## Artifacts, and where they land

The sweep writes a report directory locally and the durable record to S3.

```
reports/scaling-clean-<stamp>/
├── summary.txt                              # the table + per-run detail + W&B URLs
└── runs/
    ├── <run_id>-timeline.png                # per-node gantt: prov / train / stalled / down
    ├── <run_id>-events.txt                  # control-plane decisions, timestamped
    └── <run_id>-valcurve.png                # val loss vs step

s3://<bucket>/runs/<run_id>/
├── profile.json      # SOURCE OF TRUTH: loss_samples, val_samples, events, cost ledger
├── metrics.json      # final: steps, val_loss, trained_seconds_total, grad_accum, ...
└── logs/boot-node*.log
```

`summary.txt` already embeds the absolute paths to each PNG and the W&B run URL
per leg, so it is the single index — link it, and everything else is reachable
from it.

**W&B:** each leg is its own run in project `spot-train`, grouped by
`WANDB_GROUP` (the sweep sets `scaling-clean-<stamp>`). Open the group to see the
world-size staircase and per-leg throughput on shared axes. W&B is a **mirror**;
if it is disabled the experiment is unaffected and `profile.json` still has
every number.

### Collect the links into one page

```bash
python3 .context/ladder/collect.py            # table straight from S3 profiles
sed -n '1,40p' reports/scaling-clean-<stamp>/summary.txt
```

For a writeup, quote `summary.txt`'s table, embed the three PNGs per leg, and
link the W&B group. Keep the `run_id`s — they are the join key between the local
report, S3, and W&B.

---

## Reading the result

Check these in order; the first two are validity gates, not results.

1. **Control held?** In `summary.txt` / `metrics.json`, every leg must show
   `effective_global_batch: 480` and `grad_accum` halving as nodes double
   (40 / 20 / 10 / 5). If not, the legs did different work and the speedups are
   meaningless.
2. **Clean run?** `restart_count: 0`, `resumed: false`, no whole-group restarts
   in the events file. A recovery mid-measurement contaminates ms/step.
3. **Speedup + efficiency** vs the table above.
4. **$/unit work** = `nodes / speedup`. This is the number most scaling writeups
   omit, and it is the one that shows 8 nodes costing 2x per unit of work — the
   honest cost of buying wall-clock with parallelism.

---

## Known gotchas (each of these has bitten this repo)

- **`.env` beats your exports if you source it last.** The CLI's loader uses
  `setdefault` and is safe; a manual `. ./.env` after your `export`s silently
  overwrites them. Source `.env` *first*, overrides after — the order in the
  command above is deliberate.
- **`eval_interval` from the recipe is 1000.** Harmless here only by accident: a
  300s leg never reaches step 1000, so no eval fires and nothing is timed that
  shouldn't be. In a *target* sweep the same value is fatal — the target
  crossing is never observed and every leg reports INCONCLUSIVE. Set
  `EVAL_INTERVAL_STEPS=25` for any run with a `TARGET_LOSS`.
- **`WARMUP_STEPS` from the recipe is 2000.** A short leg spends its entire life
  in LR warmup. Overridden to 100 above; keep it that way for short runs and use
  the recipe's 2000 only for multi-hour runs.
- **Timeouts are derived, not fixed** — `NCCL_INIT_TIMEOUT` scales with node
  count, `RECOVERY_TIMEOUT` covers a replacement's boot + dataset pull, and the
  metrics deadline is `budget + overhead`. If you shorten the dataset or change
  instance types, re-check them; every one of these was found by a run dying.
- **Leave `PREEMPT_CHECKPOINT_SECONDS` alone unless killing nodes.** At 5s with a
  1.5 GB checkpoint it cost +58% per step. It is 60s now.

---

## Variants

| goal | change |
|---|---|
| add 8 nodes | `NODE_COUNTS=1,2,4,8` (needs 32 vCPU, +~9 min, +~$1) |
| convergence claim | set `TARGET_LOSS=<value>`, raise `SCALING_CAP_SECONDS` to ~8x the 8-node time-to-target. Hours, not minutes. |
| under preemption | `spot-orchestrate scaling-preempt` — same recipe, kills half of each world at `PREEMPT_AT_SECONDS`, reports recovery cost |
| laptop-independent | drive it from the t3.micro: `spot-orchestrate orch up --experiment scaling-clean --env NODE_COUNTS=1,2,4` |
