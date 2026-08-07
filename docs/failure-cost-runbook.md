# Experiment — what does one node failure cost, and does elastic resize pay?

Two runs, 4 nodes, 5-minute training budget each. Identical in every respect
except that one kills a node mid-run. **~22 min, ~$1.44.**

It answers two questions at once:

1. **What does a failure cost?** — the A/B difference in wall clock, steps, dollars.
2. **Does elastic resize earn its keep?** — do the survivors actually train while
   the replacement boots, or do they spend the whole window restarting?

Question 2 is currently *unanswered*, not answered-badly: the Gantt's
"degraded 86s, world dipped to 4" is `len(members)` from the epoch document —
it proves the supervisor shrank membership, not that anyone trained. The
trainer's own step lines across two preemption runs show **only `ws 8`**, with
361s and 566s stretches logging nothing. That is suggestive but not conclusive,
because at `LOG_INTERVAL_STEPS=10` a short degraded window can pass without
emitting a single line. Logging every step removes the ambiguity.

---

## Why the A/B is controlled

`TRAIN_BUDGET_SECONDS` / `TRAIN_TOTAL_SECONDS` budget **training seconds, not
wall clock** — downtime is never charged against them (proven: a run where all 8
nodes were killed still recorded `trained_seconds_total=900.58` against a 900s
budget). Both arms therefore train for the same 300s, and the failure shows up in
exactly two places:

| measure | captures |
|---|---|
| **Δ wall clock** | downtime — detect, re-form, boot a replacement, rejoin |
| **Δ steps** | work destroyed — rollback to last checkpoint + reduced-world time |

Held fixed: model, dataset, seed, global batch (480), node count, **checkpoint
interval**.

> **The checkpoint interval is the confound to guard.** Clean arm uses
> `CHECKPOINT_INTERVAL_SECONDS=30`; the preempt arm uses
> `min(30, PREEMPT_CHECKPOINT_SECONDS=60) = 30`. Identical today — verify before
> trusting results. A preempt arm that checkpoints more often pays a throughput
> tax the clean arm doesn't, and the comparison silently becomes meaningless.
> That exact bug (5s vs 30s, at 1.5 GB per checkpoint) cost **+58% per step** and
> made "preemption" look three times more expensive than it is.

---

## Setup

```bash
cd <repo>
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
export MARKET=on-demand NODES=4 VCPU_QUOTA=64 \
       MAX_INSTANCE_LIFETIME_SECONDS=7200 METRICS_OVERHEAD=1800 \
       WARMUP_STEPS=100 LR_DECAY_STEPS=2000 \
       EVAL_INTERVAL_STEPS=0 EVAL_ITERS=50 \
       LOG_INTERVAL_STEPS=1
```

- `EVAL_INTERVAL_STEPS=0` — a val pass is pure overhead here and would land at
  different points in each arm. This measures throughput, not convergence.
- `LOG_INTERVAL_STEPS=1` — **this is what makes question 2 answerable.** Every
  step line carries its world size, so a single step at `ws 3` cannot hide.

## Arm A — clean (~9 min)

```bash
BASELINE_SECONDS=300 PYTHONPATH=src python3 -m orchestrator multinode
```

## Arm B — one node killed at t+120s (~13 min)

```bash
TRAIN_TOTAL_SECONDS=300 \
PREEMPT_COUNT=1 PREEMPT_VICTIMS=3 PREEMPT_AFTER=120 \
PYTHONPATH=src python3 -m orchestrator multinode-preempt
```

t+120s of a 300s budget: past warmup, with room to recover and train again
before the budget expires. Node 3 is a plain worker (kill node 0 for the
master/re-election case — see *Next*).

---

## Analysis

Both numbers come from `profile.json`. With `LOG_INTERVAL_STEPS=1`, every step
is a sample carrying its `world_size`.

```bash
set -a && . ./.env && set +a
RUN_A=<clean run_id>; RUN_B=<preempt run_id>
for R in $RUN_A $RUN_B; do
aws s3 cp "s3://$SPOT_TRAIN_BUCKET/runs/$R/profile.json" - 2>/dev/null | python3 -c "
import json,sys
from collections import Counter
d=json.load(sys.stdin); ls=sorted(d.get('loss_samples') or [], key=lambda s:s['t_rel'])
print('run:', d['run_id'])
print('  steps by world size:', dict(Counter(s.get('world_size') for s in ls)))
print('  wall_s:', (d.get('durations') or {}).get('total_s'),
      ' cost:', round((d.get('cost') or {}).get('total_usd',0),2))
prev=None
for s in ls:
    if prev and s.get('world_size')!=prev.get('world_size'):
        print(f\"  world {prev['world_size']} -> {s['world_size']}: \"
              f\"gap {s['t_rel']-prev['t_rel']:.0f}s  (step {prev['step']} -> {s['step']})\")
    prev=s
"
done
```

### Q1 — cost of a failure

```
Δ wall clock = wall_B − wall_A        (downtime)
Δ steps      = steps_A − steps_B      (rollback + reduced-world time)
Δ cost       = cost_B − cost_A
```

Report **absolute seconds per failure, not percentages.** Rollback is bounded by
the checkpoint interval in absolute terms (≤30s) — 10% of a 300s run but 0.02%
of a 36h run. The per-event seconds transfer between run lengths; the percentage
does not.

### Q2 — does elastic pay?

Look at `steps by world size` for arm B:

- **`{4: n, 3: m}` with m > 0** → survivors trained during the window. Elastic
  pays. Report `m` steps and the seconds spent at `ws 3`.
- **`{4: n}` only** → the two restarts consumed the whole window. Elastic
  produced nothing.

The same output gives the two inputs to the decision rule:

- **`T_restart`** = the `world 4 -> 3` gap (last step before the change, first
  step after).
- **`window`** = time from that change until the world is back at 4.

> **shrink-and-continue wins iff `T_restart < window × (N−1)/N`**
> — at N=4 that is `T_restart < 0.75 × window`.

Why: one membership change restarts *every* node (kill torchrun, reload a 1.5 GB
checkpoint, re-init NCCL), and a preemption causes **two** — shrink to N−1, then
grow back to N. Survivors only net out ahead if the window is long enough to
amortise both.

---

## Next, by outcome

- **Elastic pays** → keep it. You now have a measured claim rather than a design
  intention, and `T_restart` to quote.
- **Elastic does not pay** → implement a third policy: when a replacement is
  already in flight, **hold the world and pay one restart instead of two**. Note
  this is not "disable fault tolerance" — a broken collective must be re-formed
  either way; the only question is whether you re-form once or twice. Today the
  code offers `replace_on_loss=True` (shrink + replace) and `False` (shrink, no
  replace); "hold and wait" does not exist yet.

Either way the result inverts the intuition worth stating: **shrink-and-continue
only pays when replacements are SLOW.** Add hot spares (~10s replacement) and
shrinking becomes strictly worse than pausing — you would pay two restarts to
salvage a ten-second window.

Then repeat at **8 nodes** (~$2.88): losing 1 of 8 costs half the throughput of
losing 1 of 4, so the rule predicts elastic looks *better* at 8. Confirming that
is the scaling story.

---

## Artifacts

- **Timeline / Gantt** — `reports/<sweep>/runs/<run_id>-timeline.png`. Per-node
  bars, ✕ kill markers, world-size track. Lead with this; it is the most legible
  single image. Remember its world track is epoch membership, not training.
- **Events** — `reports/<sweep>/runs/<run_id>-events.txt`
- **Profile (source of truth)** — `s3://<bucket>/runs/<run_id>/profile.json`
- **Metrics** — `s3://<bucket>/runs/<run_id>/metrics.json`
- **Per-node logs** — `s3://<bucket>/runs/<run_id>/logs/boot-node*.log`
- **W&B** — URL printed at launch; project `spot-train`; shows the world-size
  staircase live. `profile.json` is authoritative, W&B is a mirror.

## Validity checks before believing anything

1. `trained_seconds_total` ≈ 300 in **both** arms. If not, the A/B is invalid.
2. `restart_count = 0` and `resumed = true` in arm B — recovery actually happened.
3. **No whole-group restart** in arm B. If one fired, the supervisor discarded
   healthy survivors; that is a bug to fix, not a cost to report.
4. Both arms on the same branch — boxes clone `REPO_BRANCH` at boot.

## Guardrails

- `MAX_INSTANCE_LIFETIME_SECONDS=7200` — boxes self-terminate after 2h regardless
  of what happens to your laptop.
- Confirm teardown after each arm:
  ```bash
  aws ec2 describe-instances --filters Name=instance-state-name,Values=running,pending \
    --query 'length(Reservations[].Instances[])' --output text
  ```
