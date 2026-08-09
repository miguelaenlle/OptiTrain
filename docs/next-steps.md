# Next steps — from here to the 24h run

Written 2026-08-08, after the 1h chaos rehearsal (`multinode-preempt-1786239456`)
and the two control-plane restart fixes. Companion to
[final-run-plan.md](./final-run-plan.md), which still holds for the 24h run
itself.

**On-demand only. No spot, anywhere.** See the banner in the run plan.

**A7 (bake the dataset into the AMI) is dropped** at the operator's direction.
Recovery therefore stays ~154s, of which ~118s is the dataset pull. Report it as
measured; do not imply it is optimised.

---

## Where we are

| | Status |
|---|---|
| G1 checkpoint pruning | done, 9 tests |
| G2 dead-man timers | done, wired at `bootstrap.py:1013` |
| G4-lite narrowed reap | done, verified in the cloud |
| `restore_state` + `adopt_fleet` | done — supervisor **process** restart is a proven no-op (epoch 3→3, fleet 2→2) |
| CH1 simultaneous kills + `L` leader token | done, exercised for real |
| Gap1 per-tick status objects | done — `live.py` already syncs `runs/<id>/status/` |
| A8 `extend` + resume-continuity tests | done |
| D1 val-loss panel, D2 model/param tiles | done |
| Suite | 480 passed, 2 skipped, ruff clean |

**Proven by the rehearsal:** worker kill (12.4s gap / 156s recovery), leader kill
(10.0s / 156s), simultaneous 2-node incl. the master (167s), sticky master across
grow-back, staircase regrow, zero whole-group restarts, live dashboard through
all of it.

**Never ran:** mass loss (6 of 8), kill-during-checkpoint-write. The rehearsal
stopped at the supervisor-kill cascade before reaching them.

---

## Step 0 — pre-flight, before ANY further cloud run (~5 min, $0)

These were originally filed after the rehearsal. Wrong order: they apply to every
paid run from here, and getting them wrong would confound the tests that follow.

**Set `ORCH_INSTANCE_TYPE=t3.medium`** in `.env`. 4 GB instead of 1. Over 24h the
supervisor accumulates a RunProfile (events, loss/val samples) and pulls all 8
node logs every tick; OOM on `t3.micro` is a real risk, and every restart it
causes replays the chaos schedule until Step 1 lands. +$0.75 over 24h.

Two items that were on this list are **already satisfied** — verified by reading
before scheduling work against them:

- **A8 archives the prior segment.** `run_extend` walks
  `metrics-seg1.json`, `-seg2`, … to the first free slot before re-entering, and
  refuses when `budget <= trained`. Extending cannot clobber a paid-for segment.
- **G2's derived lifetime is safe.** `instance_lifetime_slack_seconds = 3600`, so
  a 20-minute test box gets an 80-minute timer. It cannot fire mid-test. Still
  worth eyeballing `systemctl list-timers` once during Step 3 — reading is not
  running.

## Step 1 — persist the kill schedule (~30 min, $0)

**The last known correctness gap, and it must land before any further chaos run.**

`_fired_kills` and `_train_start` are process memory (`supervisor.py:467`).
`_train_start` re-arms to the restart moment, so after **any** supervisor restart
the entire `PREEMPT_SCHEDULE` replays from zero — every kill fires again at its
original offset. A restart at hour 8 of the 24h run would re-fire the mass loss
of 6 of 8, and `PREEMPT_SCHEDULE` is baked into user-data, so there is no live
knob to stop it.

Same pattern as the other two fixes: the authority belongs in S3, not memory.
Persist `{fired: [...], train_start_wall: <epoch seconds>}` next to the run's
other durable state and restore it in `run()` beside `restore_state`.

Use a **wall clock** for train-start, not `time.monotonic()` — monotonic resets
with the process, which is the whole bug.

**Verify:** unit tests (pure, seconds) — a restored schedule does not re-fire
spent entries; unspent entries still fire at the right offset; a cold start is
unaffected. Then it is covered end-to-end by Step 3.

## Step 2 — `orch up --run-id` (~20 min, $0)

Required by Step 3, and the only thing standing between "control-plane box died"
and "the run is over".

`spot-orchestrate extend` runs the supervisor **on the laptop**, which is the
wrong shape for a 24h run. `orch up` has no way to re-enter an existing run: it
always mints a fresh run_id.

The plumbing already exists — `run_agent` mints a run_id **only when the state
doc's field is empty** (`orch.py:817`), and `orch up` seeds that doc
(`orch.py:434`). So this is one CLI flag and one seeded field:

```bash
spot-orchestrate orch up --experiment multinode-preempt --run-id <existing> \
  --env TRAIN_TOTAL_SECONDS=<new_total>
```

The new control plane then boots, `adopt_fleet` finds the still-running boxes,
`restore_state` restores the epoch, and it continues. Guard: refuse if the run
already wrote `metrics.json` unless `--force`, so this cannot silently clobber a
finished run.

---

## Step 2b — render "control plane down" as a positive signal (~40 min, $0)

**Prerequisite for Step 3 being interpretable.** Today a control-plane outage
makes the world-size line and the Gantt simply *stop changing* — which is
pixel-identical to "the fleet was stable". That is the more dangerous reading of
the two, and a viewer has no way to tell them apart.

### Detect it from the S3 object clock, not the poll clock

Gap1 writes one status object per tick, named `<epoch_ms>.json`. So the gap is
visible **in the object names themselves**, independent of when `live.py`
happened to fetch them. A gap > ~3 tick intervals means nobody was writing.

This is what makes the detector correct rather than merely plausible, because it
distinguishes the two outages that look the same on a stale dashboard:

| | Status objects in S3 | Detector says |
|---|---|---|
| **Supervisor / box down** | never written — gap is permanent | **down band** ✅ |
| **Laptop / `live.py` down** | written all along, just fetched late | after backfill, **no band** ✅ |

Same code, opposite answers, which is exactly right. And it gives Steps 3 and 4
**opposite expected outcomes** — a strong mutual check. If Step 4 draws a down
band, the backfill is broken. If Step 3 does not, the detector is.

### Absence must be synthesised into rows

There are no status ticks during the outage, so there are no rows to carry the
signal — the same problem the trailing "now" row solved. The exporter must
**emit explicit rows at the gap boundaries** rather than leaving a hole:

- new `supervisor_up` column in `timeseries.csv` (1 normally, **0** inside a gap),
  with synthesised rows at gap start and gap end;
- inside a gap, carry cost and the last known world size forward but leave
  `train_loss` / `ms_per_step` empty — same discipline as `_fleet_row`: only what
  is still asserted, never invented.

### Render it twice

1. **A thin `supervisor_up` strip** in the Sentinels row — a hard series, greppable
   in the CSV, obvious when it drops to 0.
2. **A shaded region across every panel**, reusing the `degraded.json` machinery
   (`write_degraded` + `_annotations` already do exactly this shape). Write a
   second region set, `control_plane_down.json`, in a distinct colour from the
   degraded-world shading so the two are never confused.

The shading is what makes the story legible at a glance: the band sits over a
**still-advancing loss curve**, which is the whole result — training outlived its
control plane.

### Verify for $0

Replay it offline. `.live/multinode-preempt-1786239456/status/` holds the
rehearsal's real per-tick objects; delete a contiguous window into a copy, run the
exporter, and assert `supervisor_up` goes 0 across exactly that window and 1 either
side. No cloud needed, and it pins the detector before it is trusted in Step 3.

## Step 3 — supervisor BOX failure → resume (~35 min, ~$3)

Distinct from the SIGKILL test already passing. systemd resurrects a *process*;
nothing resurrects a *box*. This is the uncovered case.

**Shape:** 2 nodes, on-demand, no chaos schedule, `TRAIN_TOTAL_SECONDS=1200`.

| t+ | Action | Expected |
|---|---|---|
| ~5m | training steady at epoch 1 | dashboard shows world 2, steps advancing |
| ~8m | **`aws ec2 terminate-instances` the t3.micro** | control plane gone for good |
| 8–13m | observe | **steps keep advancing** — sidecars obey the last epoch doc. World size and the Gantt FREEZE (no supervisor => no status ticks). Cost keeps climbing |
| ~13m | `orch up --run-id <same>` | new box adopts the live fleet |
| ~16m | observe | `adopted N running box(es)` + `re-adopted epoch N` in the log; **fleet still 2**, no relaunch; status ticks resume |
| end | run completes | one continuous loss curve across the whole thing |

**Pass criteria**

1. Fleet count never changes (2 throughout).
2. Steps advance during the outage — training never noticed.
3. After resume: `adopted 2 running box(es)` and `re-adopted epoch N`, N unchanged.
4. No `published epoch 1`.
5. Loss curve continuous end to end.
6. Same run_id throughout; checkpoints continue from where they were.

### Making it autonomously monitorable

This is the interesting half, and it needs one honest caveat surfaced in the
dashboard rather than explained away.

Two data paths, and they behave **differently** during the outage:

| Panel source | During control-plane death |
|---|---|
| node logs → S3 (loss, steps, step time, tokens) | **keeps updating** — boxes write straight to S3 |
| supervisor status ticks (world size, Gantt, work-at-risk, fleet counters) | **frozen** — nobody is writing them |

That contrast is the *result*, not a defect: it is a picture of training
outliving its control plane. **Step 2b is what makes it say so** — without the
down band, a frozen world-size line reads as "the fleet was stable", which is
the opposite of the truth.

Run `live.py` on the laptop for the whole test (it polls S3 and is unaffected by
the control plane dying).

**Dashboard pass criteria, additional to the run's own:**

7. `supervisor_up` drops to **0** for the outage window and returns to 1 on resume.
8. The **shaded down band** covers the outage on every panel, sitting over a loss
   curve that is still advancing.
9. The Gantt and world size flatline through the band rather than blanking out.

**Cost:** 2 × g5.xlarge × ~25 min ≈ $0.85, plus 2 × t3.micro ≈ $0.02. Call it $3
with slack.

---

## Step 4 — laptop disconnect / reconnect (~30 min, ~$3)

Operator-in-the-loop. Verifies Gap1 end to end: the run must be untouched, and
the dashboard must have **no hole** afterwards.

**Shape:** 2 nodes, on-demand, **no chaos**, `TRAIN_TOTAL_SECONDS=1200`. A clean
job — the point is the observer, not the fleet.

| t+ | Who | Action |
|---|---|---|
| 0 | agent | launch via `orch up`, start `live.py`, confirm dashboard live |
| ~6m | agent | **tell the operator to disconnect** — wifi off or lid closed |
| 6–14m | operator | stay offline ≥8 min (≈50 missed 10s polls) |
| ~14m | operator | reconnect |
| ~15m | agent | restart `live.py`; it `s3 sync`s `runs/<id>/status/` and backfills |

**Pass criteria**

1. Run unaffected — steps advanced throughout, fleet 2, same run_id.
2. **No gap in `world.csv` or the Gantt across the offline window** — this is the
   whole point. The supervisor wrote a status object every tick; sync backfills
   them.
2b. **`supervisor_up` stays 1 throughout, and NO down band is drawn.** The
   supervisor never stopped writing, so the Step 2b detector must say so. A band
   here means the backfill is broken — the inverse of Step 3, and the reason the
   two tests check each other.
3. Loss/step panels continuous (node logs are append-only in S3).
4. Cost curve continuous.
5. `orch status` reports a healthy run immediately on reconnect.

**Why this is the real test of Gap1:** before per-tick objects, `status.json` was
overwritten in place, so an offline hour was world-size and Gantt history that
never existed and could never be recovered. Watch specifically for `world.csv`
rows *inside* the offline window — their presence is the proof.

`live.py` can also simply be left running through a wifi drop; killing and
restarting it is the harsher version and the one worth doing.

**Cost:** ~$3.

---

## Step 5 — re-run the 1h chaos rehearsal (~1.5h, ~$10)

Only after Steps 1–4 pass. Same schedule as before — it is already in
[final-run-plan.md §7](./final-run-plan.md) — but now it should reach the two
events that never ran:

- **t+40m mass loss, 6 of 8** — the headline event. Expect ~4× step time at world
  2 and a staircase regrow inside ~300s.
- **t+50m kill during a checkpoint write** — atomic rename.

Replace the t+32m supervisor SIGKILL with the **box-failure + `--run-id` resume**
from Step 3, since that is now the stronger test and the SIGKILL path is already
proven.

All ten go/no-go gates from the run plan apply.

---

## Step 6 — operator items before the 24h run

Config and code pre-flight moved to **Step 0**; what remains genuinely belongs
last, because both need a live box or are the operator's to do.

| | Item | When |
|---|---|---|
| a | **AWS Budgets ceiling + deny-EC2 action.** The only guardrail that survives our code being wrong in a way we did not imagine | operator, before the 24h run |
| b | Execute each kill switch once from a clean shell — `reap_ours`, `orch status`, `orch down`. One never run is not a kill switch | during Step 3 |
| c | Eyeball `systemctl list-timers` for `spot-autokill` — G2's value is verified by reading, not by running | during Step 3 |

---

## Deliberately NOT doing

- **A7 / AMI dataset bake** — dropped by the operator.
- **ASG for the control plane.** The design invariant is one writer, N readers;
  an ASG guarantees eventual replacement, not exclusivity, and two supervisors
  publishing epochs is split brain. Doing it safely needs a lease with a fencing
  token — a day of new, untested machinery to prevent a ~0.5%/24h event whose
  cost is now bounded by Steps 2–3. Reducing recovery time beats reducing failure
  probability for a one-off run.
- **Real 2-minute-notice preemption** (IMDS/FIS). Parked; controlled kills are
  harsher. Say so rather than implying the notice path is proven.

---

## Sequence and cost

| Step | Effort | Cost |
|---|---|---|
| 0 — pre-flight config | 5m | $0 |
| 1 — persist kill schedule | 30m | $0 |
| 2 — `orch up --run-id` | 20m | $0 |
| 2b — "control plane down" signal | 40m | $0 |
| 3 — box failure → resume | 35m | ~$3 |
| 4 — laptop disconnect | 30m | ~$3 |
| 5 — re-run 1h rehearsal | 1.5h | ~$10 |
| 6 — operator items | 20m | $0 |
| **To the 24h start line** | **~4h** | **~$16** |
| 24h run | — | ~$196 |

**Ordering constraints, and only these:**

- Step 0 before **any** paid run — it is config for all of them.
- Step 2 before Step 3 — without `--run-id` there is no way to resume a dead
  control plane, which is the entire point of that test.
- Step 1 before Step 5 — a restart during the rehearsal would otherwise replay
  the whole chaos schedule and make the run uninterpretable.
- Step 2b before Step 3 — without it a control-plane outage is indistinguishable
  from a stable fleet, and Step 3 has nothing to show.

Steps 3 and 4 are both clean 2-node runs and independent of each other; if time
is short they can go back to back off one build.
