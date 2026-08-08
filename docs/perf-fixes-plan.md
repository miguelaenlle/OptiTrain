# Plan — async checkpointing + stop idling survivors

Two fixes, each validated by a controlled A/B against a baseline **we already
have**. Prove at 4 nodes, then scale to 8.

**Total new compute: ~$5.** Existing runs supply the control arms, so most of the
comparison is already paid for.

## Measured baselines (already banked, 4 nodes, 300s training budget)

| | run_id | steps | wall | training-s | notes |
|---|---|---|---|---|---|
| **clean** | `multinode-1786117954` | 67 | 667.1s | 301.7 | control for E1 |
| **1 preemption** | `multinode-preempt-1786118730` | 75 | 981.5s | 327.2 | control for E2 |

Recipe for both (must be reproduced exactly in every new arm):
`GLOBAL_BATCH_SIZE=480 BATCH_SIZE=12`, GPT-2 124M, `openwebtext`, `NODES=4`,
`MARKET=on-demand`, `WARMUP_STEPS=100 LR_DECAY_STEPS=2000`,
`EVAL_INTERVAL_STEPS=0 EVAL_ITERS=50`, **`LOG_INTERVAL_STEPS=1`**,
`CHECKPOINT_INTERVAL_SECONDS=30`.

---

# Fix 1 — async node-local checkpointing (~33% of time-to-loss)

## The problem

Arm A's 301.7 training-seconds produced only 67 steps:

```
67 steps x 3.02s ideal      = 202s
checkpoint stall            =  99s   <- 33% of all training time
final_saves (separate phase)= 108s   <- another 16% of wall clock
```

`train.py:511` runs on the critical path every checkpoint:

```python
if cfg.local_checkpoint_dir and distributed.broadcast_flag(ddp, submitted):
    if ddp.local_rank == 0:
        checkpoint.save_local(checkpoint.snapshot(...), cfg.local_checkpoint_dir, step)
```

Two compounding costs, and they need different fixes:

1. **The write is synchronous** — `save_local` serializes + writes 1.5 GB inline.
2. **It writes to the slow disk** — `LOCAL_CHECKPOINT_DIR=/tmp/spot-ckpt` is on
   the 30 GB gp3 root (~125 MB/s) while the 229 GB instance-store NVMe sits idle
   hosting the dataset. 1.5 GB / 125 MB/s = **~12s**.

## The change

**1a. Move the local tier to the NVMe.** Reuse `_data_dir_block`'s existing probe
(`bootstrap.py`) — same mount that already holds the dataset, same
fallback-to-root when no instance store exists. Ephemeral is correct here:
node-local checkpoints only ever serve a survivor restarting in place; if the box
dies they are worthless anyway (that is what S3 is for).

**1b. Make the local write async**, mirroring the S3 tier exactly
(`checkpoint.py:174` `AsyncSaver`): one save in flight, `submit()` returns False
while busy, background thread does serialize+write. The correctness question —
*what if the model mutates mid-save?* — is already solved by `snapshot()`, which
takes a **point-in-time CPU copy**; training then mutates GPU tensors freely.

**Keep synchronous:** preempt and final checkpoints (already the case). Call
`flush()` before them so the writer never races.

**Not in scope:** the D2H copy itself. `snapshot()`'s docstring notes it is
"SECONDS for GPT-2-124M" — pinned memory + a dedicated CUDA stream would overlap
it with compute, but that is a separate, riskier change. Do it only if E1 shows
the residual stall still matters.

## Tests (hermetic, $0)

- async local save: training continues while the write is in flight; the file
  lands complete and atomically renamed
- one-in-flight bound: a second `submit` during a slow write is refused, memory
  stays at one snapshot
- **resume correctness**: extend `tests/test_kill_resume.py` — kill mid-async-write,
  resume, assert loss continues from the checkpoint (not from scratch, not diverged)
- final/preempt checkpoints remain synchronous
- NVMe path chosen when the probe finds an instance store, root when it does not,
  and both call sites agree (`tests/test_bootstrap.py` already has this shape)

## E1 — does async checkpointing pay? (~$0.75, ~11 min)

**One new run.** The control already exists.

```bash
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
export MARKET=on-demand NODES=4 VCPU_QUOTA=64 MAX_INSTANCE_LIFETIME_SECONDS=7200 \
       WARMUP_STEPS=100 LR_DECAY_STEPS=2000 EVAL_INTERVAL_STEPS=0 EVAL_ITERS=50 \
       LOG_INTERVAL_STEPS=1 METRICS_OVERHEAD=1800
BASELINE_SECONDS=300 PYTHONPATH=src python3 -m orchestrator multinode
```

| metric | control (measured) | predicted with fix | pass |
|---|---|---|---|
| **steps in 300 training-s** | 67 | **~100** | >= 90 |
| wall clock | 667.1s | ~460s | <= 520s |
| max step time (from `ws` log lines) | 28.5s | ~3.5s | < 6s |
| `final_saves_s` | 108.2s | < 20s | < 30s |
| val_loss at equal steps | 6.527 @67 | within +/-0.05 @67 | correctness gate |

**The val_loss gate is the important one.** More steps is worthless if the
checkpoints are wrong; compare loss at the *same step number*, not at the end.

**FAIL -> stop.** If steps < 90, the stall is not (only) the local write — most
likely the D2H copy dominates, which points at the pinned-memory change instead.

---

# Fix 2 — do not regrow until the replacement can actually join (~270s/failure)

## The problem (confirmed from the event stream)

```
t+422.6s  node0 -> reconfiguring  epoch 1->2      KILL detected
t+434.5s  node0 -> training                       world 3 (T_restart = 11.9s)
t+448.0s  node0 -> reconfiguring  epoch 2->3      REGROW published
t+512.3s  node3 -> provisioning  boot             replacement only NOW starts
t+662.7s  all   -> training                       world 4 restored
```

Survivors trained at world 3 for **13.5s**, then idled **214s** waiting for a
node that had not begun booting. **3 x 214 = 642 node-seconds wasted per failure.**

**Root cause** — `supervisor.py:_healthy()`:

```python
if not n.registered or n.aws_state in _DEAD_STATES: return False
if n.aws_state != "running": return False
# Fresh boot has no log yet (age None) — treat as alive
return not (n.log_age_s is not None and n.log_age_s > policy.heartbeat_timeout_s)
```

A node we terminated ~26s ago still passes: its cached IP makes `registered`
true, EC2 briefly still reports `running`, and its log age (26s) is under the
90s heartbeat timeout. So `healthy` includes node 3, `healthy != members` fires,
and the reducer regrows onto a corpse. The world then waits for the replacement
that eventually fills the slot.

Note `T_restart = 11.9s` — re-forming an already-booted world is *cheap*. The
214s is not the cost of resizing; it is the cost of resizing at the wrong moment.

## The change

A slot must not count as healthy between "we terminated it" and "its replacement
can train". Two parts:

**2a. Invalidate the cached identity on terminate.** `_terminate` clears
`st.ips[node]` so `registered` goes false immediately — closing the AWS-lag +
stale-log race that let a corpse look alive.

**2b. Readiness, not registration.** The replacement's sidecar registers at boot
(t+512) but cannot train until the dataset is present (t+662). Registration is
the wrong signal. Have the slot become a member only once the node reports it is
ready to join the collective — the trainer already emits a `training` state
event, so gate admission on that rather than on the instance existing.

Threaded through `Observation` (keep `decide()` pure): the reducer sees a
`ready` set, not merely `registered`.

**Guard against the opposite failure:** if a replacement never becomes ready, the
world must not shrink forever. The existing `WholeGroupRestart` floor and
`max_epochs_without_progress=6` already cover it — verify they still fire.

## Tests (hermetic, $0)

- terminated node stops counting as healthy on the very next tick, even with a
  fresh log age and `aws_state="running"` (the exact observed race)
- a registered-but-not-ready replacement does **not** trigger regrow
- regrow fires on the tick the replacement first reports ready
- a replacement that never becomes ready still hits the restart floor
- existing supervisor tables stay green (`tests/test_supervisor.py`, 36 cases)

## E2 — do survivors actually train through a failure? (~$1.15, ~16 min)

**One new run**, versus the banked preempt control.

```bash
# same env as E1, plus:
TRAIN_TOTAL_SECONDS=300 PREEMPT_COUNT=1 PREEMPT_VICTIMS=3 PREEMPT_AFTER=120 \
  PYTHONPATH=src python3 -m orchestrator multinode-preempt
```

| metric | control (measured) | predicted with fix | pass |
|---|---|---|---|
| **steps at reduced world (`ws 3`)** | 3, all rolled back | **>= 20, banked** | >= 15 |
| survivor idle after regrow | 214s | < 30s | < 60s |
| wall clock | 981.5s | ~800s | <= 850s |
| whole-group restarts | 0 | 0 | must be 0 |
| `resumed` | True | True | must be True |

**`ws 3` step count is the metric that isolates Fix 2** — it is unaffected by
Fix 1, so this arm is valid even though it carries both changes. Wall clock
mixes the two; do not attribute it to either alone.

> **Read `ws` counts from the raw node log, not `profile.json`.** The profile
> dedupes by step number, so reduced-world steps that were later re-executed at
> full world are silently overwritten — that is what made "elastic never trained"
> look true. `aws s3 cp .../logs/boot-node0.log - | grep -oE "ws [0-9]+" | sort | uniq -c`

**Also check the work is BANKED**, not merely performed: in the control, both
resumes restored from step 37 and all reduced-world work was discarded because
the 25s degraded window never outlived the 30s checkpoint interval. With a ~240s
window at least one checkpoint must land while degraded. If `ws 3` steps happen
but the regrow still resumes from the pre-kill step, the remaining gap is
**checkpoint-on-membership-change**, which is then the next fix.

---

# E3 — scale to 8 nodes (~$3, ~35 min)

Only after E1 and E2 pass. Same recipe, `NODES=8`, `PREEMPT_VICTIMS=7`.

Two runs: clean + one preemption. What 8 nodes tests that 4 cannot:

- **Fix 1 at scale** — `broadcast_flag` is a collective across 8 ranks; async
  saves must not desynchronise it.
- **Fix 2 at scale** — losing 1 of 8 costs half the throughput of 1 of 4, so
  survivors should retain ~87% of full-world throughput while degraded.
- **The scaling claim** — per-failure cost should *fall* as the world grows.
  That is the interview-grade result, and it needs both sizes to state.

| metric | 4-node (from E2) | 8-node expectation |
|---|---|---|
| throughput while degraded | ~75% | ~87% |
| wall-clock cost per failure | ~130s | <= 130s |
| steps at reduced world | >= 20 | >= 20 |

---

# Cost and sequencing

| step | compute | wall clock |
|---|---|---|
| Fix 1 + tests | $0 | — |
| **E1** (1 run) | ~$0.75 | ~11 min |
| Fix 2 + tests | $0 | — |
| **E2** (1 run) | ~$1.15 | ~16 min |
| **E3** (2 runs, 8 nodes) | ~$3.00 | ~35 min |
| **total** | **~$4.90** | **~1h** |

Gate at each step: **E1 fails -> do not build Fix 2**, the checkpoint stall would
dominate anything Fix 2 buys.

## Risks

- **Async checkpoint + resume is correctness-critical.** A bug silently diverges
  rather than crashing. The CPU-snapshot pattern is already proven in the S3
  tier, and the kill/resume test is the gate.
- **E2 carries both fixes**, so wall clock cannot be attributed to one. The `ws 3`
  step count isolates Fix 2. A third arm (Fix 1 only, with preemption, ~$1.15)
  would give clean isolation if that matters for the writeup.
- **`profile.json` loses rolled-back steps.** Every reduced-world measurement
  must come from raw node logs.
- **Instance-store checkpoints vanish on terminate** — already true of `/tmp`
  on a terminated box, and S3 remains the durable tier. No new exposure.

## Guardrails (every run)

- `MAX_INSTANCE_LIFETIME_SECONDS=7200` — boxes self-terminate after 2h regardless
- driver runs under a `trap reap EXIT INT TERM` that terminates all instances
- verify `0` instances after each arm before starting the next
- both arms of any comparison on the same branch — boxes clone `REPO_BRANCH`
