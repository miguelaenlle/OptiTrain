# Runbook — what does one node failure cost?

A controlled A/B: run the identical job twice, once clean and once with a node
killed mid-run. The difference **is** the cost of the failure. Run it at 4 nodes
first, then 8, to see whether that cost grows with the size of the world.

Fast and cheap by design: **~22 min and ~$1.50 for the 4-node pair**, same again
plus ~$3 for the 8-node pair.

---

## Why this is controlled

`TRAIN_BUDGET_SECONDS` / `TRAIN_TOTAL_SECONDS` budgets **training seconds, not
wall clock** — downtime is never billed against it (proven: a total-loss run
still recorded `trained_seconds_total=900.58` against a 900s budget). So both
arms do the *same amount of training*, and the failure shows up in exactly two
places:

| measure | what it captures |
|---|---|
| **Δ wall clock** | downtime — detect, re-form, boot a replacement, rejoin |
| **Δ steps** | work destroyed — rollback to the last checkpoint + time at reduced world |

Everything else is held fixed: same model, same dataset, same seed, same global
batch (480), same node count, same checkpoint interval.

**Checkpoint interval is the one confound to watch, and it is currently clean:**
the clean arm uses `CHECKPOINT_INTERVAL_SECONDS=30`; the preempt arm uses
`min(30, PREEMPT_CHECKPOINT_SECONDS=60) = 30`. Identical. If you ever change
either, re-check this — a preempt arm that checkpoints more often is paying a
throughput tax the clean arm isn't, and the comparison silently stops meaning
anything. (That exact bug cost ~58% per step before it was found.)

---

## Phase 1 — 4 nodes (~22 min, ~$1.50)

Shared env for **both** arms:

```bash
cd <repo>
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
export MARKET=on-demand VCPU_QUOTA=64 MAX_INSTANCE_LIFETIME_SECONDS=7200 \
       WARMUP_STEPS=100 LR_DECAY_STEPS=2000 EVAL_INTERVAL_STEPS=0 EVAL_ITERS=50 \
       LOG_INTERVAL_STEPS=1 METRICS_OVERHEAD=1800
```

`EVAL_INTERVAL_STEPS=0` — a full val pass is pure overhead here and would land at
different points in each arm. We are measuring throughput, not convergence.

`LOG_INTERVAL_STEPS=1` — **this setting is the second experiment, for free.** Every
step line carries its world size (`ws N`), so the preempt arm answers a question
the existing data cannot: *do the survivors actually train while a replacement
boots?* See "Does elastic actually pay?" below. At the default of 10, a short
degraded window can pass without emitting a single line, which is exactly why
the current evidence is ambiguous.

**Arm A — clean baseline**

```bash
NODES=4 PYTHONPATH=src python3 -m orchestrator multinode
# BASELINE_SECONDS is what `multinode` reads for its budget:
#   export BASELINE_SECONDS=300
```

**Arm B — one node killed at t+120s**

```bash
NODES=4 TRAIN_TOTAL_SECONDS=300 \
PREEMPT_COUNT=1 PREEMPT_VICTIMS=3 PREEMPT_AFTER=120 \
PYTHONPATH=src python3 -m orchestrator multinode-preempt
```

Kill at t+120s of a 300s budget: late enough to be past warmup, early enough
that the world recovers and trains again before the budget expires. Victim 3 is
a plain worker — see *Variants* for the master case.

## Phase 2 — 8 nodes (~22 min, ~$3)

Identical, with `NODES=8` and `PREEMPT_VICTIMS=7`. Run it only after Phase 1
looks sane; it is twice the money for the same wall clock.

---

## What to record

For each of the four runs, from `s3://<bucket>/runs/<run_id>/metrics.json`:

| field | why |
|---|---|
| `steps` | the headline: work completed |
| `trained_seconds_total` | must be ~300 in **both** arms, else the A/B is invalid |
| `val_loss`, `train_loss` | sanity — the failure must not change the math |
| `restart_count`, `resumed` | `resumed=true` in arm B is the recovery actually happening |
| `world_size` | must return to N |

and from `profile.json`: `durations.total_s` (wall clock) and `cost.total_usd`.

Then:

```
failure cost (time)  = wall_B − wall_A
failure cost (work)  = steps_A − steps_B          (rollback + degraded window)
failure cost (money) = cost_B − cost_A
```

**Report the absolute numbers, not the percentages.** Rollback is bounded by the
checkpoint interval in absolute terms (≤30s), which is 10% of a 300s run and
0.02% of a 36h run. A percentage from this experiment does not transfer; the
per-event cost in seconds does.

---

## Artifacts to link

Written automatically by each run:

- **Timeline / Gantt** — `reports/<sweep>/runs/<run_id>-timeline.png`
  Per-node bars (provision / train / stalled / down), the ✕ kill markers, and
  the world-size track underneath showing the dip and its recovery. This is the
  single most legible artifact; lead with it.
- **Events** — `reports/<sweep>/runs/<run_id>-events.txt`
- **Profile (source of truth)** — `s3://<bucket>/runs/<run_id>/profile.json`
  loss samples, per-instance cost ledger, phase durations.
- **Metrics** — `s3://<bucket>/runs/<run_id>/metrics.json`
- **Per-node logs** — `s3://<bucket>/runs/<run_id>/logs/boot-node*.log`
- **W&B** — run URL is printed at launch and stored in the sweep summary;
  project `spot-train`. Shows the world-size staircase live.

`profile.json` is authoritative; W&B is a mirror that no-ops without a key.

---

## How to read it

- **Δ wall clock ≈ 4–5 min per failure** is the expected shape: ~100s EC2 launch
  (not ours), ~35s dataset pull (post-tuning), ~25s setup, plus detection and two
  epoch transitions. If it is much larger, look at the timeline for a
  whole-group restart — that means the supervisor discarded healthy survivors,
  which is a bug, not a cost.
- **Δ steps** should be roughly `rollback (≤30s) + degraded window × (1/N)`. At
  8 nodes losing one worker, the degraded window costs only 1/8 of throughput,
  so Δ steps should be *smaller* at 8 nodes than at 4 — that is the interesting
  result, and the reason to run both.
- **`trained_seconds_total` differing between arms invalidates the run.** Check
  it before believing anything else.

---

## Does elastic actually pay? (the second result, free)

The design shrinks membership onto the survivors so training continues while a
replacement boots. Whether that produces *work* is unverified — and the current
evidence says probably not:

- the Gantt's "degraded 86s, world dipped to 4" is `ws = len(members)`, i.e. the
  **epoch document's membership count**. It proves the supervisor shrank. It does
  not prove anyone trained.
- the trainer's own step lines in two preemption runs contain **only `ws 8`**,
  with 361s and 566s stretches where no step was logged at all.

**Why it might not pay.** A membership change restarts *every* node, so one
preemption costs **two** full world restarts (shrink to N−1, then grow back to
N) — each one reloading a 1.5 GB checkpoint and re-initialising NCCL. Survivors
only get useful work if

```
degraded window  >  2 x T_restart
```

At 8 nodes T_restart is plausibly 60–90s (NCCL init alone needed >20s), so
2 x T_restart is about the size of the whole window.

**What to look for in arm B's log:** any `step N: ... ws 3` (4-node) or `ws 7`
(8-node) lines. Count them.

- **Some** -> elastic pays; report steps completed at reduced world.
- **None** -> the double restart eats the entire window. The fix is a supervisor
  policy, not tuning: when a replacement is already in flight and expected within
  ~2 x T_restart, **do not shrink** — hold the world and pay one restart instead
  of two.

Worth stating plainly because it inverts the intuition: **shrink-and-continue is
only worth it when replacements are SLOW.** Add hot spares (~10s replacement) and
shrinking becomes strictly worse than pausing.

While you are here, `T_restart` falls out of the same log for free: the gap
between the last step before a membership change and the first step after it.

## Variants worth one extra run each

- `PREEMPT_VICTIMS=0` — kill the **master**. Forces leader re-election; expected
  to be the more expensive case.
- `PREEMPT_COUNT=2 PREEMPT_VICTIMS=3,1 PREEMPT_AFTER=100` — two failures, to
  check the per-event cost is constant rather than compounding.

## Guardrails

- `MAX_INSTANCE_LIFETIME_SECONDS=7200` — every box self-terminates after 2h no
  matter what happens to your laptop.
- Confirm teardown after each pair:
  `aws ec2 describe-instances --filters Name=instance-state-name,Values=running,pending --query 'length(Reservations[].Instances[])' --output text`
- Both arms must run on the same branch; boxes clone `REPO_BRANCH` at boot.
