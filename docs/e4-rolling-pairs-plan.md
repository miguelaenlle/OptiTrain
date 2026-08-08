# E4 — rolling pair failures: 2 die, recover, 2 others die, recover, again

Three rounds on a 4-node world. Each round kills **two nodes simultaneously**,
lets the world recover to full, then kills the *other* pair.

**~$2.50, ~25 min.** Do not run until reviewed.

```
round 1   t+120s   kill nodes 2,3   -> survivors 0,1   -> recover to 4
round 2   t+420s   kill nodes 0,1   -> survivors 2,3   -> recover to 4
round 3   t+720s   kill nodes 2,3   -> survivors 0,1   -> recover to 4
```

## What this tests that nothing before it has

| | covered by | E4 adds |
|---|---|---|
| single node loss | E2b | — |
| all nodes at once | total-loss expt | — |
| **half the world, repeatedly** | — | ✅ 3 rounds |
| **the master dying as part of a pair** | — | ✅ round 2 kills node 0 |
| **killing a replacement** | — | ✅ round 3 kills round 1's replacements |
| **does recovery cost stay constant?** | — | ✅ 3 comparable rounds in one run |

The last one is the real question. Every measurement so far is a *single*
failure; this is the first that can show whether the Nth recovery costs the same
as the first, or whether something accumulates.

## Why it needs a custom driver

`multinode-preempt` cannot express simultaneous pairs — its schedule is
`[((k+1)*interval, victims[k]) …]`, one victim per entry at strictly increasing
times. Same limitation the total-loss experiment hit.

So: a small driver calling `_run_supervised` directly, modelled on
`.context/ladder/total_loss.py`:

```python
kill_schedule = [
    (120.0, 2), (120.0, 3),     # round 1 — pair dies together
    (420.0, 0), (420.0, 1),     # round 2 — the OTHER pair, incl. the master
    (720.0, 2), (720.0, 3),     # round 3 — round 1's replacements
]
_run_supervised(cfg, kind="multinode-preempt", budget=900,
                replace_on_loss=True, kill_schedule=kill_schedule,
                return_profile=True)
```

**300s between rounds** is deliberate: E2b took ~210s from kill to full world, so
this leaves ~90s of full-world training between recoveries. Any tighter and the
rounds overlap, which measures something different (cascading failure) and is
worth its own experiment later.

## Settings

```bash
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
export MARKET=on-demand NODES=4 VCPU_QUOTA=64 \
       MAX_INSTANCE_LIFETIME_SECONDS=7200 \
       METRICS_OVERHEAD=2400 \
       WARMUP_STEPS=100 LR_DECAY_STEPS=2000 \
       EVAL_INTERVAL_STEPS=0 EVAL_ITERS=50 LOG_INTERVAL_STEPS=1
# driver sets TRAIN_TOTAL_SECONDS=900
```

`METRICS_OVERHEAD=2400` (not the 1200 default): budget 900s + three recoveries.
Deadline becomes 3300s against an expected ~1250–1850s wall clock. The default
would give 2100s — probably enough, but this is exactly the failure that killed a
healthy 1h fleet at 30 minutes, so size it deliberately.

`LOG_INTERVAL_STEPS=1` again — reduced-world steps are invisible at the default
of 10 when a window is short.

## The risk to watch

**The restart floor is exactly at its cap.** Each round publishes two epochs
(shrink, then grow), so three rounds = **6 epochs**, and
`max_epochs_without_progress = 6`. It is only safe because a checkpoint lands
between rounds and resets the counter — with 300s spacing and a 30s checkpoint
interval, several will.

If a whole-group restart fires anyway, that is the finding: the floor is counting
normal recoveries again and the budget needs to be per-round rather than
cumulative. Watch for `whole-group restart (floor)` in the log; it names its
reason now.

Peak vCPU: 4 nodes + 2 simultaneous replacements = **24 of 64**. Fine.

## What to measure

Per round, from the event stream (`t+` offsets) and node logs:

| metric | why |
|---|---|
| kill → survivors training at world 2 | `T_restart` — should stay ~10–12s every round |
| **duration of the world-2 window** | survivors must train through it (E2b: 173s) |
| **steps at `ws 2`, and whether banked** | resume point after regrow must be *ahead* of the pre-kill step |
| replacement register → world restored | admission latency |
| **round 1 vs 2 vs 3** | the headline: does any of the above grow? |

Plus overall: `steps`, `val_loss`, `trained_seconds_total` (must be ~900),
`resumed`, `restart_count`, wall clock, cost, instances used (expect 4 + 6 = 10).

Read `ws` counts from **raw node logs**, never `profile.json` — it dedupes by
step and silently drops rolled-back work.

## Control

No new control run needed. Compare per-round recovery against E2b's single-node
numbers, and rounds against each other *within* this run — which is the stronger
comparison anyway, since everything else is held identical.

If a clean 900s baseline is wanted for total wall clock, that is a separate
~$1.40 run; I would skip it initially.

## Success criteria

- world returns to 4 after **every** round
- **zero** whole-group restarts
- `trained_seconds_total` ≈ 900 (budget honoured across six kills)
- steps at `ws 2` > 0 **and banked** in all three rounds
- `T_restart` and recovery duration flat across rounds (no accumulation)
- instances used = 10; fleet terminated at the end

## Deliverables

`docs/e4-results.md` with the timeline PNG (three dips in the world track — that
image is the result), the per-round table, and W&B links.

## Guardrails

- driver under `trap reap EXIT INT TERM`
- `MAX_INSTANCE_LIFETIME_SECONDS=7200` — boxes self-terminate regardless
- verify 0 instances after the run
- runs on `phase1/gpt2-owt-baseline` (boxes clone `REPO_BRANCH`), which now
  includes the E2b registration fix and the replacement-launch de-serialization
