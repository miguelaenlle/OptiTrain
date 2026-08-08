# E5 — 8-node chaos ladder: calibrate the 36h run before spending $300

**Not yet run. Pre-registered predictions below — the point is that they can be wrong.**

One 8-node run, four failure events of escalating severity (**k = 1, 2, 4, 7**),
each measured independently. **~$8, ~55 min.** Its output is the three constants
the 36h projection currently guesses at.

## Why this run exists

The 36h model (`.context/e5/model.py`) has one unmeasured constant,
`RESTART_SCALE_8N`, and the headline moves a lot with it:

| `RESTART_SCALE_8N` | overrun @ 18 events | @ 36 events |
|---|---|---|
| 1.0 | 3.6% | 7.1% |
| **1.3 (current guess)** | **4.0%** | **8.0%** |
| 2.0 | 5.1% | 10.1% |
| 3.0 | 6.6% | 13.1% |

Across a plausible 1.0–3.0 range that is **1.3–2.4h** of overrun at 18 events, or
**2.6–4.7h** at 36. Too wide to quote.

**8-node post-fix recovery has never been measured.** The two 8-node preempt runs
predate both the async-checkpoint fix and the replacement-launch de-serialization
fix, and they are unusable even as a *ratio* check: in
`multinode-preempt-1785780835`, `shrink_resume` lands **exactly** on the last of
four relaunches spaced 18.8s apart. That 87s kill→resume measures a bug we
removed, not 8-node rendezvous.

## Hypothesis: what an ideal chaos test looks like

> **Fix the variable you want to attribute cost to; randomise everything else.**

Pure randomness is not rigour — it produces an uncontrolled mix of failure sizes
from which no per-`k` cost can be recovered. A fully fixed schedule is not
realism — E4 already showed its rollback numbers were "phase luck between fixed
kill times and the checkpoint cycle."

So:

| | | why |
|---|---|---|
| **fixed** | `k` per round (1, 2, 4, 7), ascending | `k` is the independent variable; one round per value is what makes cost attributable |
| **fixed** | round spacing (~500s) | rounds must not overlap, or they measure cascade instead |
| **random** | *which* nodes die | removes any suspicion of convenient victims; naturally samples master and replacement kills |
| **random** | kill time, ±60s jitter | randomises **checkpoint phase**, so rollback is *sampled* rather than a single lucky draw |
| **random** | which node survives at k=7 | the survivor is arbitrary, as in a real reclaim |

Seeded and printed before launch, so it is reproducible and auditable — genuinely
random draws, not hand-picked, and re-runnable by seed. **That is how the run
gets nondeterminism and rigour at the same time.**

Ascending rather than shuffled order is deliberate: k=1 and k=2 are the values
the 36h model actually needs, so they are banked first. If k=7 destabilises
something, the deliverable survives.

## Engineering cost: near zero

`_run_supervised(kill_schedule=[(t, node), ...])` already expresses everything —
**simultaneous kills are just entries sharing a timestamp**, which is exactly how
E4 built pairs. No orchestrator change. The driver is a ~60-line adaptation of
`.context/e4/rolling_pairs.py`.

Three things do need attention, all small:

1. **`max_epochs_without_progress = 6` is hardcoded** (`supervisor.py:95`), and
   4 rounds × 2 epochs = **8 publications**. It resets only when a *checkpoint
   step advances* (`supervisor.py:529-532`). E4 sat exactly at 6 and survived on
   that reset. Raise it to 12 for this run, or a whole-group restart mid-ladder
   destroys the measurement. **This is the one code change.**
2. **Timestamp the step log.** `train.py:440` emits `step N: loss …, Nms/step`
   with no clock. Every E4 plot was reconstructed because of that, and *two were
   wrong before they were right*. One-line change, removes the entire class of
   analysis error.
3. **`LOG_INTERVAL_STEPS=1`** — reduced-world steps are invisible at the default
   of 10 when a window is short (E4 lesson).

## Schedule

```
E5 CHAOS LADDER — one run, four escalating failures

 world
   8  ▇▇▇▇▇▇▇▇┐      ┌▇▇▇▇▇▇┐      ┌▇▇▇▇▇┐      ┌▇▇▇▇┐      ┌▇▇▇▇▇▇▇▇▇▇
   7          └▇▇▇▇▇▇┘      │      │     │      │    │      │
   6                        └▇▇▇▇▇▇┘     │      │    │      │
   4                                     └▇▇▇▇▇▇┘    │      │
   1                                                 └▇▇▇▇▇▇┘
      └───────┬─────────────┬────────────┬───────────┬──────────────> t
      lead-in │ R1 k=1      │ R2 k=2     │ R3 k=4    │ R4 k=7
       180s   t+180±60      t+680±60     t+1180±60   t+1680±60
              kill 1 of 8   kill 2       kill 4      kill 7

              ▲ every kill: victims drawn at random from the LIVE set
                (master and prior replacements both eligible),
                time jittered ±60s to randomise checkpoint phase
```

Budget **2100 training-seconds**. 500s spacing: recovery is expected at
200–300s, leaving ~200s of full-world training between rounds — enough to land
≥2 checkpoints and reset the epoch counter.

Budget **2100 training-seconds**. 500s spacing: recovery is expected at
200–300s, leaving ~200s of full-world training between rounds — enough to land
≥2 checkpoints and reset the epoch counter.

## What each failure costs, and where

This is the model E5 is calibrating. **The two costs land in different
currencies, and conflating them is the easiest way to misreport this system:**

```
ANATOMY OF ONE FAILURE

            kill                              replacement joins
              │                                       │
  world 8 ▇▇▇▇┤                                       ├▇▇▇▇▇▇▇▇▇
              │                                       │
  world 7     └────────▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇┘
              ├───┤                                ├───┤
           T_restart         degraded window      regrow
            10-25s             200-300s           10-25s
              │        (survivors KEEP TRAINING)     │
              ├──── rollback ~50s (redo work) ───────┤

  WALL-CLOCK cost = T_restart + regrow + rollback        ~120s
      no training seconds accrue -> the 36h budget is NOT consumed,
      the run simply ends later

  STEP cost       = degraded_window x (1 - rate_ratio)   ~7% of 250s at k=1
      training seconds DO accrue, just buying fewer steps
```

Because the budget counts **training seconds**, a failure never steals budget —
it costs wall clock (downtime) and steps (degraded throughput), separately.

## Pre-registered predictions

**Throughput at reduced world** is the strongest prediction, because the clean
ladder measured two of these points directly (1n/2n/4n/8n time-to-target:
625.6 / 348.8 / 223.6 / 156.2s):

| round | world | predicted step rate vs world 8 | basis |
|---|---|---|---|
| k=1 | 7 | **0.93** | log-interp between measured 4n and 8n |
| k=2 | 6 | **0.86** | log-interp |
| k=4 | 4 | **0.70** | **measured directly** (223.6 vs 156.2s) |
| k=7 | 1 | **0.25** | **measured directly** (625.6 vs 156.2s) |

Note this supersedes the linear `ETA = 0.653` in the model, which was fit at a
single point (k/N = 0.5) and predicts a nonsensical 0.43 at world 1.

| quantity | prediction | confidence |
|---|---|---|
| `T_restart` (kill → survivors training) | **12–25s**, roughly k-independent | medium — 4n measured 7.8–9.9s; 8n adds NCCL init |
| recovery (kill → world 8) | **200–300s, k-INDEPENDENT** | **this is the de-serialization test** |
| per-event wall cost | 150–250s | low — the constant we are here to measure |
| rollback | 0–104s, mean ~52s, *varying across rounds* | high — jitter should produce spread |
| whole-group restarts | **0** | medium — depends on fix #1 above |

**The sharpest single result:** if k=7 recovery ≈ k=1 recovery, replacements
boot in parallel and the de-serialization fix holds at scale. If k=7 takes ~7×
longer, it does not — and that is a bug worth finding before a 36h run, not
during one.

## Risks

| risk | mitigation |
|---|---|
| **`world_size=1` at k=7 untested** under the epoch protocol | torchrun sets RANK, so `distributed.init` takes the normal path with a 1-rank group — should work. Verify free on localhost via `test_epoch_e2e.py` first. |
| **vCPU peak 60 of 64** — at k=7, 8 boxes (7 terminating) + 7 replacements = 15 × 4 | tightest moment of the run. Guard in the driver; abort if quota headroom is short. |
| epoch floor fires mid-ladder | fix #1 |
| k=7 step time ~14s (accum 40 at world 1) | expected, not a fault — budget the round for few steps |

## Success criteria

- world returns to 8 after **every** round
- **zero** whole-group restarts
- `trained_seconds_total` ≈ 2100
- reduced-world steps > 0 **and banked** in all four rounds
- rollback varies across rounds (confirms jitter sampled checkpoint phase)
- 22 instances = 8 + (1+2+4+7); fleet terminated

## Deliverable

`RESTART_SCALE_8N`, per-`k` throughput ratios, and recovery duration at 8 nodes —
which collapses the sensitivity band above and lets the 36h scenario tables be
quoted rather than hedged.

---

# The 36h run this calibrates — hypothesised ideal configuration

```
┌─ 36 HOUR RUN ─────────────────────────────────────────────────────────────┐
│                                                                            │
│  8 x g5.xlarge   ·   budget = 36h of TRAINING seconds   ·   ckpt 480s      │
│  failures: Poisson, seeded, MTBF 2h  ->  ~18 events                        │
│  size mix: 70% k=1   25% k=2   5% k=4        (never k>=7 unattended)       │
│                                                                            │
│  world                                                                     │
│   8 ▇▇▇▇▇╲╱▇▇▇▇▇▇▇╲╱▇▇▇╲__╱▇▇▇▇▇▇▇╲╱▇▇▇╲╱▇▇▇▇▇╲____╱▇▇▇╲╱▇▇▇▇╲__╱▇▇▇▇▇▇   │
│   7      ╰╯       ╰╯                ╰╯    ╰╯                ╰╯             │
│   6                  ╰──╯                       ╰────╯  (k=2)              │
│   4                                    ╰────────╯       (k=4)              │
│     └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┤     │
│     0h        6h        12h       18h       24h       30h       36h        │
│                                                                            │
│  EXPECTED:  wall 37.9h (+5.3%)   ·   72.8k steps (-0.34%)   ·   ~$305 OD   │
└────────────────────────────────────────────────────────────────────────────┘
```

**Why these settings**

| knob | value | reasoning |
|---|---|---|
| budget in **training seconds** | 36h | downtime never bills against the budget, so every scenario does identical work and slowdown is measurable purely as wall clock + steps |
| **checkpoint interval** | **480s** | optimum for ~18 failures. Shorter wastes steps on checkpoint tax; longer pays more rollback per failure. Floor is ~104s regardless (save+verify+smoke saturates below that) |
| MTBF | 2h (~18 events) | far above real spot rates — a genuine 36h run would likely see **0** interruptions, so the demo must inject them |
| k mix | 70/25/5 | weighted to the realistic case while still exercising multi-node loss |

**The interval is failure-rate dependent** — this is the one setting that must be
chosen *after* E5, not before:

| failures over 36h | best interval | steps/wall-h |
|---|---|---|
| 0 | 1200s (20 min) | 2063 |
| 9 | 668s (11 min) | 1976 |
| **18** | **480s (8 min)** | **1920** |
| 36 | 344s (5.7 min) | 1833 |
| 72 | 252s (4.2 min) | 1701 |

**Sensitivity of the headline** (interval × failure count, `RESTART_SCALE_8N=1.3`):

| interval | F=12 | F=18 | F=24 | F=36 |
|---|---|---|---|---|
| 300s | 2.7% | 4.0% | 5.4% | 8.0% |
| 420s | 3.2% | 4.9% | 6.5% | 9.7% |
| **480s** | 3.5% | **5.3%** | 7.0% | 10.5% |
| 600s | 4.1% | 6.1% | 8.1% | 12.2% |

Step deficit is nearly flat across all of these (**0.23–0.69%**) — the failures
cost wall clock, not progress. That asymmetry is the result worth showing, and it
is exactly what E5 either confirms or refutes.

> Every number in this section rests on `RESTART_SCALE_8N`, which is a guess
> until E5 runs. Treat the whole block as a hypothesis, not a projection.
