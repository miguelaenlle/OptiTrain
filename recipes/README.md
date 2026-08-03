# Recipes — pinned env presets

Env files you `source` before a run. They exist so a run's hyperparameters are a
reviewable artifact instead of a shell-history incantation, and so every run in a
comparison provably used the same numbers.

```bash
set -a && . recipes/gpt2-owt.env && set +a     # note the -a: exports every var
NODES=8 TRAIN_BUDGET_SECONDS=3600 spot-orchestrate multinode
```

`set -a` matters: the orchestrator relays hyperparameters to the training boxes
by reading its **own environment**, so an unexported shell variable is silently
dropped and the box quietly runs defaults.

Anything you set on the command line still wins — the experiment drivers use
`os.environ.setdefault`, so a sourced recipe overrides the driver's built-ins and
an explicit `VAR=x` on the command line overrides the recipe.

| file | for | horizon |
|---|---|---|
| `gpt2-owt.env` | Karpathy's `train_gpt2.py` config, verbatim | the 1h / 4h / 36h runs |
| `gpt2-owt-ladder.env` | same model + batch, schedule compressed | the 1/2/4/8-node sweeps |

## The two rules that break the experiment if you get them wrong

**1. `BATCH_SIZE` must divide 60.** It's the *per-rank micro-batch*; the trainer
derives `grad_accum = ceil(GLOBAL_BATCH_SIZE / (world * micro))`. That `ceil` is
a silent trap — when `world * micro` doesn't divide 480, the effective global
batch inflates at *some* node counts and not others, so the node counts are no
longer step-matched and the whole comparison is invalid. The failure is invisible:
no error, just curves that don't line up.

| micro | accum @ 1 / 2 / 4 / 8 nodes | effective global batch | |
|---|---|---|---|
| 12 (Karpathy) | 40 / 20 / 10 / 5 | 480 / 480 / 480 / 480 | ✅ |
| 10 | 48 / 24 / 12 / 6 | 480 / 480 / 480 / 480 | ✅ |
| 6 | 80 / 40 / 20 / 10 | 480 / 480 / 480 / 480 | ✅ |
| **8** | 60 / 30 / 15 / 8 | 480 / 480 / 480 / **512** | ❌ |
| 16 | 30 / 15 / 8 / 4 | 480 / 480 / **512** / **512** | ❌ |

If micro-batch 12 OOMs on the 24GB A10G (Karpathy had 40GB A100s), step down to
**10, then 6** — never to the intuitive 8.

**2. Use one `LR_DECAY_STEPS` for every run in a comparison.** Karpathy decays
over 600k steps (~5 days on 8×A100). Our budget is ~36h, so the canonical recipe
never leaves the high-LR plateau. Either keep 600000 (schedule-faithful, worse
final loss) or set it to the step count `calibrate` projects for your budget
(better loss, deviates from Karpathy) — but changing it *between* runs makes
their loss curves incomparable.

## Sizing the ladder

Ideal scaling means the 1-node run takes ~8× the 8-node run, and
`SCALING_CAP_SECONDS` applies to **every** run in the sweep. Size it off the
*slowest* run: if the cap only fits the 8-node run, the 1-node entry returns
INCONCLUSIVE and there's no baseline to compute speedup against.

With `T` = 8-node time-to-target, the sweep costs `32·T` node-hours and takes
`15·T` wall-clock. At T = 20 min that's ≈ **$11 and ~5 hours**.

**Run throughput mode first.** Omit `TARGET_LOSS` and every run trains to the cap
instead of to a loss, so the sweep costs `15·cap` node-hours — at an 8-minute cap
that's ≈ **$2 and ~30 minutes**. It answers "does 8 nodes work, and what's the
ms/step scaling curve" cheaply, and its measured throughput tells you what to set
`TARGET_LOSS` and the cap to for the real run. Only then pay for target mode.

```bash
set -a && . recipes/gpt2-owt-ladder.env && set +a

# 1. cheap: scaling efficiency + a real 8-node smoke test  (~$2, ~30 min)
MARKET=on-demand NODE_COUNTS=1,2,4,8 SCALING_CAP_SECONDS=480 \
  spot-orchestrate scaling-clean

# 2. headline: time-to-target-loss ladder  (~$11, ~5 h)
MARKET=on-demand NODE_COUNTS=1,2,4,8 TARGET_LOSS=<from step 1 / calibrate> \
  spot-orchestrate scaling-clean

# 3. fault tolerance: kill 4 of 8 nodes at t+60s, replace, keep training
MARKET=on-demand NODE_COUNTS=8 TARGET_LOSS=<same> \
  spot-orchestrate scaling-preempt
```
