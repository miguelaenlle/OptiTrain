# E4 — rolling pair failures: 6 kills, 3 rounds, zero degradation

**Verdict: PASS on every criterion.** Half a 4-node world destroyed three times in
a row — including the master, and including replacements — and recovery cost did
**not** grow. Each round banked exactly the same amount of work.

Run `multinode-preempt-1786199662` · 900s training budget · **$1.84, 27.8 min**.

```
round 1  t+120s  kill 2,3  ->  survivors 0,1  ->  recover to 4
round 2  t+420s  kill 0,1  ->  survivors 2,3  ->  recover to 4   (the MASTER dies)
round 3  t+720s  kill 2,3  ->  survivors 0,1  ->  recover to 4   (kills round 1's REPLACEMENTS)
```

## The diagram

![E4 rolling pair failures](img/e4-rolling-pairs.png)

Ten node rows: four originals plus six replacements (`·r1`, `·r2`). The world
track shows **three clean dips — `degraded 598s across 3 dip(s)`** — and every
survivor row is **green straight through them** (201s, 183s, 184s of training at
world 2, not idling).

Two details worth catching:

- **The leader star moves three times** — `node0` → `node2·r1` → `node0·r1`.
  Master re-election happened twice, unprompted, because round 2 killed node 0.
- **Round 2's dip is two-stage** (`2 → 3 → 4`). The replacements registered at
  different moments and the supervisor grew the world as each became ready
  rather than waiting for both — so capacity returned sooner than a
  wait-for-all policy would have allowed.

## Durable progress over time

![E4 durable progress](img/e4-progress.png)

`ckpt_step` from the supervisor's own status history — already timestamped, so
nothing is reconstructed. It is the **furthest step that survives a failure**,
which is the number that actually matters.

**The curve never goes backwards, and it climbs inside every shaded window.**

| world-2 window | ckpt_step | rate |
|---|---|---|
| t+470..648s | 38 → 59 (+21) | 0.118 steps/s |
| t+771..910s | 59 → 89 (+30) | 0.215 steps/s |
| t+1068..1223s | 98 → 119 (+21) | 0.135 steps/s |
| **world 4 reference** | 119 → 195 (+76) | **0.190 steps/s** |

Half the fleet dead costs roughly a third to a half of the step rate, not all of
it — gradient accumulation holds the global batch at 480, so two nodes do the
same work per step, just with more micro-batches each. (The middle window's
0.215 exceeds the world-4 reference because that reference spans checkpoint
stalls; treat the three world-2 rates as the comparable set.)

Per-step rollback at each membership change is small and bounded by the
checkpoint interval — node 0's log shows `steps 1..47` then a resume at 38, then
`39..66` then a resume at 59: **9 and 7 steps**, ~30s of work at ~3s/step.

> A first version of this plot was reconstructed by anchoring step lines to
> `training` events and cumulatively summing `ms/step`. It showed false plateaus,
> because node index 0 has training events from both the original box and its
> replacement and the segments were mis-assigned. `ckpt_step` needs no
> reconstruction and is what the supervisor actually acts on.

### Per-step granularity

![E4 true per-step progress](img/e4-true-steps.png)

`ckpt_step` above moves only every ~30s. This is every logged step, placed in wall
clock by anchoring each log segment to its `training` event — those events carry
`{node, epoch}`, so original boxes and their replacements are never confused, and
only one clock is used. All 219 step lines are placed, none dropped, and the final
step is 195, matching `metrics.json`.

**Progress is essentially continuous.** The thin line is the current step (per
node) and the thick line the furthest reached. There are exactly **two rollbacks
of any size — 8 steps and 6 steps** — each bounded by the 30s checkpoint interval
at ~3s/step. The slope flattens inside the shaded world-2 windows but never
reaches zero.

Shaded spans here are approximate: they are placed in the training-event clock,
whereas the epoch boundaries were recorded in the status clock. Slope changes are
the reliable indicator, not the exact edges.

**For future runs: put a timestamp on the step log line.** Every reconstruction
above exists only because `step N: loss …, Nms/step` has none, and a first
attempt at this plot was wrong precisely because of that.

## The headline: recovery does not degrade

| round | `T_restart` | world-2 window | idle after regrow | **steps banked** |
|---|---|---|---|---|
| 1 | 9.9s | 201.0s | 42.1s | **21** |
| 2 | 7.9s | 168.5s | 60.2s | **21** |
| 3 | 7.8s | 184.3s | — | **21** |

Every prior measurement was a *single* failure, so nothing showed whether the
Nth recovery costs more than the first. **It does not.** `T_restart` is flat
(9.9 / 7.9 / 7.8s — the first is marginally slower, consistent with warm caches
afterwards), and the world-2 windows differ only by replacement boot jitter.

## The work is banked — identically, every round

Resume points prove reduced-world training was kept, not redone:

```
round 1   shrink resumed @ step 38  ->  regrow resumed @ step 59   = 21 steps banked
round 2   shrink resumed @ step 68  ->  regrow resumed @ step 89   = 21 steps banked
round 3   shrink resumed @ step 98  ->  regrow resumed @ step 119  = 21 steps banked
```

If the reduced-world work were being discarded, each regrow would resume at the
same step its shrink did — which is exactly what the pre-E2b runs did.

## Results

| metric | value |
|---|---|
| steps | 195 |
| val_loss | 5.6473 |
| `trained_seconds_total` | **901.74** (budget 900) |
| `resumed` / `restart_count` | True / 0 |
| final `world_size` | 4 |
| **whole-group restarts** | **0** |
| instances used | **10** = 4 + exactly 1 per kill |
| wall clock | 1667.8s |
| cost | $1.84 |

Epoch progression, clean throughout:
`1 [0,1,2,3] → 2 [0,1] → 3 [0,1,2,3] → 4 [2,3] → 5 [0,2,3] → 6 [0,1,2,3] →
7 [0,1] → 8 [0,1,3] → 9 [0,1,2,3]`

- W&B — https://wandb.ai/bot-developer3-Miguel%20Aenlle/spot-train/runs/5kdn29l3

## What six kills actually cost — vs. a clean 4-node run

Everything above is internal to E4: rounds compared against each other. This is
the external comparison — **what would the same 900 training-seconds have
produced with nothing dying?**

### The baseline, and why it is interpolated

No clean 4-node run exists at a 900s budget. The nearest is **E1b**
(`multinode-1786166169`) — 4 nodes, **300s**, and critically the *first* run with
async checkpointing, so it is the only clean control on the same code E4 ran.
Identical recipe, verified from both `profile.json`s: `openwebtext`,
`world_size 4`, `grad_accum_steps 10`, `effective_global_batch 480`.

So E1b is extrapolated 300s → 902s. Extrapolation is done **two independent
ways**, because a single linear scaling would hide the startup transient:

| method | steps @ 901.7 train-s |
|---|---|
| naive — E1b's average rate (0.2691 steps/s) × 902 | 242.7 |
| modelled — one-time warmup + steady 3.123 s/step + amortised checkpoint cycles | 246.7 |

They agree to **1.6%**, so the baseline is taken as **~245 steps**. The model
comes from decomposing E1b's 81 `ms_per_step` samples: median 3039 ms, a 4.8s
one-time warmup on step 1, and checkpoint stalls arriving in **pairs** (steps
10/13, 39/42, 67/71 — ~14.6s excess every ~28.5 steps). Wall clock scales only
in its training term; provisioning (196.5s), final saves (74.3s), eval and
sampling are fixed costs that a longer run amortises.

### The comparison

| metric | clean 4-node (interpolated) | **E4 — 6 kills** | delta |
|---|---|---|---|
| steps | ~245 | **195** | **−50 (−20.3%)** |
| wall clock | 1188.7s | **1667.8s** | **+479.2s (+40.3%)** |
| cost | $1.358 | **$1.838** | **+$0.480 (+35.4%)** |
| instances | 4 | 10 | +6 |
| instance-seconds | 4860 | 6579 | +1719 (+35.4%) |
| **instance-s / step** | 19.86 | **33.74** | **+69.9%** |
| **$ / step** | $0.00555 | **$0.00943** | **+69.9%** |
| goodput (`trained/wall`) | 75.9% | **54.1%** | −21.8 pp |
| train loss at end | ~5.49 *(est.)* | 5.665 | +0.175 |

`trained_seconds_total` is 901.74 against a 900s budget, so the *training* budget
was fully honoured — the loss deficit is not lost training time. It is that a
world-2 step is slower than a world-4 step, so the same 900 seconds buys fewer
of them.

The loss estimate fits `loss = 9.7069 − 0.7669·ln(step)` on E4's own tail
(step ≥ 78, max residual 0.065). It predicts **5.6628** at step 195 against an
actual **5.6648**, so the ~5.49 at step 245 is a modest extrapolation, not a
guess. E1 established that loss-vs-step superimposes across these configs.

### Normalised per failure

| | wall clock | cost | steps |
|---|---|---|---|
| per kill (6) | +79.9s | +$0.080 | −8.3 |
| per pair-event (3) | +159.7s | +$0.160 | −16.6 |

**This is the number that improved most.** The pre-fix failure-cost A/B measured
a *single* node loss at **+314.4s and +$0.351**. E4 loses **two nodes at once**
for **+159.7s and +$0.160** — so per node lost, 314s → 80s, a **3.9× reduction**,
attributable to E1/E1b (async checkpointing) and E2b (survivors train through the
replacement's boot instead of idling).

### Reading it honestly

Two true statements, and both belong in any writeup:

- **Per unit of work, fault tolerance is not free: +70% cost per step.** Six
  destructive events on a 15-minute run is an extreme rate — one every ~2.5
  minutes, each removing half the fleet — so this is a stress-test upper bound,
  not a steady-state spot tax. Real spot interruption rates would spread the same
  fixed recovery cost across far more steps.
- **The run finished.** 20% fewer steps and 35% more money is the price of
  *surviving* six kills. The alternative without this machinery is not a cheaper
  run; it is a dead run and 900 seconds of discarded work, three times over.

The fixed costs are what dominate: 196.5s provisioning and 74.3s of final saves
are ~23% of a 1189s clean run and would be ~1.3% of a 4-hour one. **Both columns
improve with run length, and the gap between them narrows** — which is the case
for measuring this again at the 1h/4h scale before quoting a spot-vs-on-demand
headline.

## The risk that did not fire

Three rounds × 2 epochs = 6 epoch publications, against
`max_epochs_without_progress = 6`. The plan flagged this as safe *only* because
checkpoints land between rounds and reset the counter — and if the floor fired
anyway, that would have been the finding.

**It did not fire.** Checkpoints landed in every window (the banking above proves
it), the counter reset each round, and the floor stayed dormant across six kills.

> Reporting note: an early automated check said `1` whole-group restart. That was
> the collector's `grep -ci "whole-group restart"` matching the driver's own
> warning banner about this risk, not a supervisor event. Grepping for the actual
> emitted string (`[supervisor] whole-group restart`) returns **0**.

## What this run covers that earlier ones did not

| | covered by | E4 |
|---|---|---|
| single node loss | E2b | — |
| all nodes at once | total-loss expt | — |
| **half the world, repeatedly** | — | ✅ 3 rounds |
| **master dying as part of a pair** | — | ✅ round 2 |
| **killing a replacement** | — | ✅ round 3 kills round 1's `·r1` boxes |
| **does recovery cost accumulate?** | — | ✅ answered: no |

## Validity

- `trained_seconds_total` 901.74 against a 900s budget — honoured across six kills
- 10 instances = 4 + exactly one replacement per kill; no churn, no crash loops
- `restart_count=0`, `resumed=True`, zero whole-group restarts
- reduced-world counts and resume points read from **raw node logs**, not
  `profile.json` (which dedupes by step and drops rolled-back work)
- driver ran under a `trap reap` guard; fleet terminated, 0 instances billing

## Reproducing the comparison

```bash
set -a && . ./.env && set +a
python3 .context/e4/compare_clean.py --fetch    # re-pull the three profiles from S3
```

Every number in the comparison table above is printed by that script. It asserts
the recipe match between control and E4 and **exits rather than compare** if they
differ, prints both extrapolation methods with their spread, and reports the loss
fit's residual so the projection can be judged rather than trusted.

## Driver

`.context/e4/rolling_pairs.py`. A bespoke driver was necessary because
`multinode-preempt` builds `[((k+1)*interval, victims[k]) ...]` — one victim per
entry at strictly increasing times — so it cannot express a simultaneous pair,
and a stagger would leave three survivors and quietly become the already-measured
single-node case. The schedule builder is unit-tested with no AWS: three rounds,
both victims sharing one timestamp, master included in round 2, all four indices
covered.
