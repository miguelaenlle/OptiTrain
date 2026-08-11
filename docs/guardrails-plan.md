# Guardrails for a 36h run — implementation plan

## What this is

Four changes that bound the **cost of a failure** without preventing the
failure. They are explicitly *not* fault tolerance: none of them makes the run
survive anything it doesn't survive today. They make the failures we already
have bounded, loud, and cheap instead of open-ended.

The motivating case: an 8-node, 36h job where the control plane dies at an
arbitrary moment. Today that already resumes (`orch.py:769 run_agent` owns the
run_id in a durable state doc, systemd restarts the agent, the trainer's one
resume path picks up the S3 checkpoint with `TRAIN_BUDGET_SECONDS -
trained_seconds` intact). What it does *not* do is bound what happens when the
resume itself misbehaves, or when nothing is left alive to notice.

A guardrail here has three properties, and every item below is judged against
them:

1. **It fails toward stopping the spend.** Ambiguity resolves to "shut down".
2. **It is independent of what it guards.** A root-owned systemd timer on the
   GPU box, not logic inside the orchestrator that is supposed to be watching
   it. Sharing a failure domain with the guarded thing disqualifies it.
3. **It is small enough to be obviously correct** without a test matrix.

Total: ~3 hours. Ordered by value, not dependency — G1 and G2 can land alone.

| | Guardrail | Bounds | Effort |
|---|---|---|---|
| **G1** | Prune S3 checkpoints to the last N | unbounded storage + O(n) LIST growth | ~1 hr |
| **G2** | Arm the instance lifetime timers | unbounded GPU burn when nothing reaps | ~15 min |
| **G3** | Halt flag honored in `aws.launch()` | any launch path, durably, across restarts | ~1 hr |
| **G4** | Cascade guard — meter launches at the chokepoint | fleet-rebuild loops **and** in-flight amplification | ~1 hr |

---

## G1 — prune S3 checkpoints

### The problem

The node-local tier prunes to the 2 newest (`checkpoint.py:213 save_local`).
**The durable S3 tier prunes nothing.** Both `save()` (`checkpoint.py:84`) and
`AsyncCheckpointer._write` (`checkpoint.py:395`) call `s3_store.save_atomic` and
never delete.

At the default `CHECKPOINT_INTERVAL_SECONDS=30` and ~1.5 GB per blob for GPT-2
124M, 36h is on the order of **4,000 objects / ~6 TB** (~$140/month until it is
cleaned up). Note `_run_supervised` *clamps the interval down further* when a
kill schedule is present (`experiments.py:529`), so a preempt-flavored run lands
on the dense end of that estimate.

Second-order, and the reason this is rank 1 rather than a housekeeping chore:
`aws.max_checkpoint_step` (`aws.py:229`) runs a full paginated LIST over that
prefix on **every supervisor tick**, and `s3_store._s3_latest` (`s3_store.py:92`)
does the same on every trainer resume. Both get monotonically slower for the
entire run.

This is the only item on the list that bites with **certainty** — it needs
nothing to fail.

### The change

New primitive in `s3_store.py`, next to `latest()`:

```python
def prune_checkpoints(uri: str, keep: int) -> int:
    """Delete all but the ``keep`` newest checkpoints under ``uri``. Returns the
    number removed. Mirrors save_local's prune on the durable tier."""
```

- Handles S3 and local behind the one interface, like every other function in
  this module.
- Reuses `_s3_latest`'s filter exactly: name starts with `CHECKPOINT_PREFIX`,
  does **not** end with `_TMP_SUFFIX`. An in-flight atomic save must never be a
  prune candidate.
- Sorts by name. `_ckpt_name` zero-pads the step to 12 digits
  (`checkpoint.py:54`), so lexicographic order is numeric order.
- Batches deletes (`delete_objects` takes up to 1000 keys per call), so the
  first prune on an existing run is a handful of requests, not 4,000.

Call it after each successful S3 write:

- `checkpoint.py:84` — after `save_atomic` in `save()`
- `checkpoint.py:395` — after `save_atomic` in `AsyncCheckpointer._write`

Both sites are rank-0-only for the S3 tier (see `docs/checkpoint-tiers.md`), so
there is no concurrent-pruner race by construction.

New config in `spot_train/config.py`: `checkpoint_keep: int = 10`
(`CHECKPOINT_KEEP`). `0` disables pruning, for anyone who wants the old
behaviour.

### Why 10, not 2

The local tier keeps 2 for a specific reason — one interval of skew, so
`load_group_latest`'s group-MIN agreement can still find a step every survivor
holds. The S3 tier is the *durable* tier and its job is different: it must
survive losing every box, and it is what a replacement restores from. 10 is
~5 minutes of history at the dense interval and costs ~15 GB. Cheap insurance
against a bad checkpoint being the newest one.

### Failure discipline

A prune failure must never end the run. Wrap in `try/except`, log, continue —
the same discipline `AsyncCheckpointer._write` already applies to its own
exceptions (`checkpoint.py:404`). Worst case we are back to today's behaviour.

### Tests

`tests/test_checkpoint_prune.py`, against a local directory (the pattern the
existing s3_store tests use):
- 15 checkpoints, `keep=10` → the 10 highest steps survive, in order.
- A `.tmp` file present → untouched.
- `keep=0` → nothing deleted.
- Fewer than `keep` present → no-op, no error.
- Empty/absent dir → no-op, no error.

---

## G2 — arm the instance lifetime timers

### The problem

Both dead-man switches already exist and both default to **off**:

| Timer | Built at | Config | Default |
|---|---|---|---|
| Training box | `bootstrap.py:1012` (`spot-autokill`) | `MAX_INSTANCE_LIFETIME_SECONDS` | `0` = off (`config.py:175`) |
| Control plane | `bootstrap.py:411` (`orch-autokill`) | `ORCH_MAX_LIFETIME_SECONDS` | `0` = off (`config.py:371`) |

They are root-owned systemd timers armed as the first action of user-data, which
makes them the most independent mechanism available — they survive the boot
script exiting, the sidecar exiting, and the orchestrator ceasing to exist.
`poweroff` + `InstanceInitiatedShutdownBehavior=terminate` means billing actually
stops.

Without them: if the control plane is gone and a node then dies, survivors'
torchrun crash-loops, sidecars exhaust `MAX_EPOCH_CRASHES` and exit 2 —
"leaving the box up for the watchdog" (`sidecar.py:319`) — a watchdog that no
longer exists. The fleet idles at full GPU rate until someone notices.

### The change

Derive a default instead of leaving it at 0. In `config.py`:

```python
def instance_lifetime_for(self, max_seconds: int) -> int:
    """Explicit env override if set; otherwise the run's own budget plus slack
    for boot, dataset pull and the post-budget eval tail."""
    if self.max_instance_lifetime_seconds > 0:
        return self.max_instance_lifetime_seconds
    return max_seconds + self.instance_lifetime_slack_seconds  # default 3600
```

`bootstrap.build_user_data` already receives `max_seconds`, so the autokill block
at `bootstrap.py:1012` becomes a call to that instead of a bare `> 0` check.
Do the same for the control plane using the experiment budget
(`orch._budget_seconds` already computes it).

Set `ORCH_MAX_LIFETIME_SECONDS` explicitly for the 36h run rather than relying on
the derivation — the control plane should outlive the fleet by a comfortable
margin so it can do the final teardown.

### Known limitation (do not paper over it)

The timer is per-box and starts at **that box's boot**. A replacement launched at
hour 30 of a 36h run receives the same `max_seconds` (the trainer clamps its own
budget internally via budget-in-checkpoint) and so gets a timer running ~36h from
*its* boot — far past the end of the run. The timer is therefore a backstop for
the *fleet-abandoned-early* case, not a tight bound late in a run.

The lease (`status.json.valid_until`, renewable by any supervisor, enforced by
the sidecar) is what fixes this properly, and it needs no new IAM because the
worker role's S3-only policy (`docs/iam/worker-policy.json`) already suffices.
It is deliberately out of scope here — G2 is 15 minutes and covers the same
failure bluntly.

### Tests

`build_user_data` emits the `systemd-run --on-active=<n>s` line with the derived
value; an explicit env override wins over the derivation; `0` + no budget still
emits nothing.

---

## G3 — halt flag honored in `aws.launch()`

### Why this insertion point

`aws.py:924` is the **only** `run_instances` call in the orchestrator package.
Every launch path funnels through `aws.launch()`:

| Caller | Launches |
|---|---|
| `experiments.py:129` | baseline / ddp single box |
| `experiments.py:456` | multinode nodes — **and** supervisor replacements, and whole-group restarts |
| `orch.py:412` | the control plane itself |
| `prep.py:289`, `bake.py:99` | dataset prep, AMI bake |
| `fleet.py:265/345/366` | inference fleet |

One check covers all of them, including paths nobody is thinking about right
now. This is a direct payoff from the invariant CLAUDE.md already enforces
("`aws.py` — the ONLY module that calls AWS").

### The change

A document, `control/halt.json` (global) and/or `runs/<run_id>/halt.json`
(per-run):

```json
{"until": 1754700000, "reason": "crash-loop: 4 rebuilds, ckpt_step stuck at 12400",
 "by": "agent|operator", "run_id": "multinode-1754..."}
```

Checked in `aws.launch()` immediately after the existing `_DRY_RUN` early return
(`aws.py:873`), so a dry run still narrates without consulting S3:

```python
if _DRY_RUN:
    return "i-DRYRUN"
halt = halt_active(run_id)
if halt:
    raise HaltedError(f"launches halted until {_fmt(halt['until'])}: {halt['reason']}")
```

`halt_active(run_id)` consults the per-run key first, then the global one, and
treats an expired `until` as absent. New config helpers `cfg.halt_key()` /
`cfg.run_halt_key(run_id)`.

CLI: `spot-orchestrate halt [--hours 48] [--reason "..."] [--run <id>]`,
`--clear`, and `--status`.

### Two decisions worth stating explicitly

**Fail open, not closed.** If the S3 read itself errors (throttle, transient),
launch anyway. A flaky GET must not block a legitimate replacement mid-run, and
G2 + G4 still bound the damage. The flag is a deliberate stop, not a health
check.

**The flag never terminates anything.** "Halt" has two meanings and conflating
them makes G4 destructive:

| | Stops new launches | Kills running boxes |
|---|---|---|
| Halt flag | yes | **no** |
| Halt + terminate (operator / Lambda ceiling) | yes | yes |

G4 wants only the first — stop rebuilding, but leave alone a fleet that may
still be training fine. Termination stays a separate, explicit action.

### Coverage gap

`lambda_spotwatch.py:753` calls `run_instances` independently, from a different
principal, for spot probes. It is outside this fence. Small instances, low
stakes, but it should be documented rather than silently assumed covered.

### Tests

`tests/test_halt.py` — pure where possible:
- active flag → `aws.launch` raises `HaltedError`
- expired `until` → launches
- absent → launches
- per-run flag set, different run_id → launches
- global flag set → every run_id blocked
- S3 read raises → launches (fail-open), one warning logged

---

## G4 — cascade guard: meter launches at the chokepoint

### The problem, part 1: the rebuild loop

`run_agent` restarts on failure and, from attempt 2 onward, calls
`_reap_orphans` (`orch.py:752`) — terminate every training box — then rebuilds
the fleet from scratch.

Worse, the fleet destruction is not only that policy. `_run_supervised` ends
with an unconditional teardown:

```python
finally:
    for iid in node_ids.values():
        aws.terminate(iid)
```
`experiments.py:597`

So on the most likely crash — a Python exception in the supervisor tick — all
8 boxes are terminated on the way out, *before* systemd even restarts.

Now the loop: systemd has `StartLimitIntervalSec=600` / `StartLimitBurst=10`
(`bootstrap.py:534`), but each reap→rebuild→boot→dataset-pull→crash cycle takes
well over 10 minutes. **The limiter never trips.** A deterministic supervisor
bug rebuilds an 8-node fleet forever, at full fleet cost, making zero progress.

That is the *inter*-attempt loop. There is a second, nastier one.

### The problem, part 2: the cascade this system has actually had

`supervisor.py:135` records an 8-node run that **amplified 6 injected kills into
22 node slots**. Expected for 6 faults is `8 + 6 = 14`; it launched 22. The
supervisor never crashed — it kept firing whole-group restarts, each discarding
healthy survivors and relaunching the fleet. The whole cascade happened *inside
one agent lifetime*, so `attempts` never left 1.

That is what makes attempt-counting the wrong meter. It sees exactly one of the
two loops, and not the one with the measured precedent.

The fix for that specific cascade already shipped (budget the floor in
preemptions rather than epochs, 3→6; restart the stall clock on every epoch
publication; carry the `reason` on `WholeGroupRestart` so the log can say which
condition fired). **E5 validates it:** run `multinode-preempt-1786207072`, 14
kills across four rounds (k = 1, 2, 4, 7 of 8), **22 instances = 8 + exactly one
per kill**, zero whole-group restarts (`docs/e5-results.md`). What is missing is
not the fix — it is a runtime check that would have *caught* the pre-fix
behaviour instead of leaving it to be diagnosed afterwards.

### The signal: launches, not attempts

Amplification is the definition of cascade here, and it is already the number
this repo uses to declare a run healthy — E5's headline health line is literally
`22 = 8 + exactly 1 per kill`. Promote that post-hoc verdict into a live check:

```
amplification = (launches_total - node_count) / max(1, observed_faults)
```

| run | faults | slots | ratio |
|---|---|---|---|
| pre-fix 8-node (`supervisor.py:135`) | 6 | 22 | **2.33** |
| E5 (`multinode-preempt-1786207072`) | 14 | 22 | **1.00** |

Healthy is 1.0 by construction: `node_count` at start, one replacement per
observed loss. Anything sustained above ~2 is the fleet churning.

Both inputs already exist. `observed_faults` is emitted today —
`_emit_event("killed", ...)` (`supervisor.py:631`) and `_emit_event("down", ...)`
(`supervisor.py:730`). Launches get counted at `aws.launch()`.

**Why the meter belongs at `aws.launch()`:** it is mechanism-agnostic. One
counter catches supervisor restarts, whole-group restart storms, replacement
churn, and any future path, because they all have to pass through the single
`run_instances` call (`aws.py:924`). Same chokepoint as G3, same reasoning.

Keep durable checkpoint progress (`aws.max_checkpoint_step`) as the *secondary*
signal — it is the one measure a failure cannot inflate (`supervisor.py:539`),
and it catches the crash-loop-before-any-launch case that the launch meter
cannot see.

### The change

**1. A launch ledger.** `runs/<run_id>/launches.json`, appended in
`aws.launch()` beside the G3 halt check: `{ts, run_id, node, market}`. Cheap —
`aws.launch` is called per box, not per tick.

**2. A pure verdict**, matching the discipline of `supervisor.decide`:

```python
def cascade_verdict(ledger: dict, faults: int, ckpt_step: int, policy) -> tuple[str, dict]:
    """-> ("proceed" | "halt", updated_state). No AWS, no clock (now passed in)."""
```

Trips on any of:
- `amplification > policy.max_amplification` (default **2.0**), with a floor of
  at least `node_count` extra launches observed so small-N noise cannot trip it;
- more than `node_count` launches in a rolling hour beyond the initial fleet
  (absolute rate ceiling, for the case where faults are also inflated);
- `attempts_since_progress >= policy.max_rebuild_attempts` (default **4**) — the
  secondary, attempt-based check, using `last_progress_step` on the state doc.

**3. Wiring.** The attempt-based check goes in `run_agent`'s existing
`if attempt > 1:` block (`orch.py:818`), **before** `_reap_orphans` — the whole
point is to not reap. The amplification check runs each supervisor tick, where
`ckpt_step` and the event stream are already in hand.

On a halt verdict:
1. Write the **per-run** halt flag (G3) with the reason and a 48h expiry.
2. `_emit_event` so it lands in `orchestrator.log` and the timeline.
3. Final heartbeat carrying the reason, so `orch status` shows it.
4. **Return 0, not 1** — systemd must stay stopped. Returning 1 restarts the
   agent, which would consult the flag it just wrote and exit again; correct,
   but noisy in the log.

Writing the flag is what makes the decision stick across a manual `orch up` —
precisely the moment a tired operator would otherwise restart the loop by hand.

### Companion damping (small, same sitting)

Circuit breakers stop a cascade; damping keeps it from starting.

- **Back off the agent restart.** `RestartSec=30` is fixed (`bootstrap.py:548`).
  Make it 30s → 2m → 8m, exactly as the sidecar already does per-epoch
  (`RELAUNCH_BACKOFF_CAP`). Two lines, and it converts a tight rebuild loop into
  a slow one before any breaker trips.
- **Cool down whole-group restart.** Two can currently fire back to back. The
  most destructive action available should be rate-limited independently of the
  conditions that trigger it — one per `recovery_timeout_s`, minimum.

Noted but **not** fixed now: `wait_vcpu_headroom` serializes exactly when the
whole fleet is relaunching, lengthening the degraded window, which itself feeds
the no-progress condition that caused the rebuild. Mild self-reinforcement;
watch it in the logs before touching it.

### Why these thresholds

Ratio **2.0**: healthy is 1.0 by construction and the measured cascade was 2.33,
so 2.0 sits between them with room for one legitimately-retried launch.

Attempts **4**: each costs a full fleet rebuild — boot plus a 17 GB `train.bin`
pull (measured at 214s in `sidecar.prewarm_dataset`'s docstring). Four is ~40+
minutes of evidence that rebuilding does not help, and bounds the burn at well
under an hour of fleet cost.

Deliberately **not** triggers (each fires on the system working as designed):

- **a whole-group restart** — that is the recovery mechanism.
  `Policy.max_epochs_without_progress` documents this exact trap one level down:
  budgeting in epochs rather than preemptions let ~1.5 normal recoveries exhaust
  the counter and fire the most destructive action available
  (`supervisor.py:88-95`). Repeating that mistake at the guardrail layer would
  halt healthy runs.
- **node deaths or a high preemption rate** — that is the thesis. E5 lost 7 of 8
  nodes in one round and recovered; faults are the denominator, not the trigger.
- **diverging loss / NaN** — a training bug, not a spend bug, and halting
  destroys the state you would want to inspect. Alert, do not halt.
- **a single supervisor exception** — systemd already handles it.

### Tests

`tests/test_cascade_guard.py`, table-driven over `cascade_verdict` — pure, no
AWS, `now` injected:

| launches | node_count | faults | ckpt progress | verdict | note |
|---|---|---|---|---|---|
| 22 | 8 | 14 | advancing | proceed | E5: ratio 1.0 |
| 22 | 8 | 6 | advancing | **halt** | the measured cascade: ratio 2.33 |
| 10 | 8 | 1 | advancing | proceed | ratio 2.0, under the small-N floor |
| 8 | 8 | 0 | advancing | proceed | initial fleet only |
| 8 | 8 | 0 | stuck, 4 attempts | **halt** | secondary check: never launched, never progressed |
| 8 | 8 | 0 | stuck, 3 attempts | proceed | 3 < 4 |
| 8 | 8 | 0 | resumed at a lower step | proceed | a step going backwards is not progress, but is not a cascade either |
| 20 | 8 | 0 | advancing | **halt** | rate ceiling: 12 extra launches in an hour with no faults |

---

## Order of work

1. **G2** (~15 min) — one config helper, two call sites. Ship first; it is the
   only item that is pure downside protection with no design surface.
2. **G1** (~1 hr) — self-contained in `spot_train`, no orchestrator changes.
3. **G3** (~1 hr) — must land before G4, which depends on the flag.
4. **G4** (~1 hr) — the launch ledger, the pure verdict, its wiring, and the two
   damping changes (agent restart backoff, whole-group-restart cooldown).

Then `ruff check --fix . && ruff format .` and `pytest tests/`.

## Out of scope (named so they are not silently assumed)

- **Fleet adoption** — rebuilding `SupervisorState` from S3 + EC2 instead of
  reaping. ~1.5 days. Requires making the `experiments.py:597` `finally`
  conditional on the run actually being over, not on the function returning.
- **The lease** (`status.json.valid_until`) — the proportional version of G2.
- **`orch up --resume <orch_id>`** — recovery when the t3.micro itself is gone.
- **A true dollar ceiling** — a spotwatch-shaped Lambda (`spotwatch.py:163`
  already has the whole deploy path). Note that G2 already bounds a
  dead-supervisor burn with a clock, and clock × known fleet rate *is* a dollar
  bound; a ceiling only earns its keep when the supervisor is alive and making
  bad decisions.
- **Per-attempt `profile-attempt<N>.json`** — cost under-reporting after a
  restart is real but reconstructable from EC2 records afterward.

## Verification before the 36h run

The highest-information test is not a shorter version of the same job — a short
run does not exercise these paths. Run **full 8-node width for ~2h and
deliberately `systemctl kill spot-orch` once.** That exercises reap → rebuild →
resume at real width with real checkpoints, which is the exact path the 36h run
depends on. ~$25.
