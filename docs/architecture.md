# Architecture

How this system is put together: the planes, the processes, the protocol between
them, and the invariants that make a training run survive machines disappearing
underneath it.

**Scope.** This is the whole-system view. Three existing documents go deeper on
one piece each and are not repeated here:

| Document | Covers |
|---|---|
| [multinode-design.md](./multinode-design.md) | The epoch protocol's rationale + the torchrun-elastic post-mortem |
| [checkpoint-tiers.md](./checkpoint-tiers.md) | Why there are two checkpoint tiers and what a blob costs |
| [region-split.md](./region-split.md) | The us-east-1 / us-east-2 account split contract |

---

## 1. The problem the architecture is shaped by

Spot instances are 60–90% cheaper than on-demand and can be reclaimed at any
moment, sometimes with no warning. Synchronous data-parallel training is the
worst possible workload for that: every rank must participate in every
all-reduce, so **one node dying stalls all N**.

Every structural decision below follows from one goal — make node loss a routine,
bounded event rather than a run-ending one:

| Invariant | Enforced by |
|---|---|
| **One resume code path.** Startup always restores-latest-or-starts-fresh; there is never a separate "resume" branch to drift. | `spot_train/train.py:228` |
| **Checkpoint everything that affects the next step** — weights, optimizer, step, *all* RNG, *and* loader position. | `spot_train/checkpoint.py:1-21` |
| **Atomic writes.** Temp key → rename, so a mid-write kill cannot corrupt the last good checkpoint. | `spot_train/s3_store.py:1-17` |
| **Assume no warning.** Checkpoint on a timer regardless of any interruption notice. | `TrainConfig.checkpoint_interval_seconds` |
| **Single writer for membership.** Exactly one process decides who is in the group; everyone else obeys. | `orchestrator/supervisor.py` |
| **Observation, never foreknowledge.** Even a node the orchestrator itself killed is discovered dead the same way a real reclaim would be. | `supervisor.py:44-50`, `decide()` |
| **The decision core is pure.** `decide(Observation, Policy) -> [Action]` — no clock, no I/O, table-testable. | `supervisor.py:154` |
| **Don't write the model.** nanoGPT is a pinned read-only submodule; we own the fault-tolerance layer around it. | `third_party/nanoGPT`, `train.py:50` |

---

## 2. The shape in one picture

```
  LAPTOP (read-only)                    AWS us-east-1
  ┌────────────────┐
  │ spot-orchestrate│  launches    ┌──────────────────────────────┐
  │  orch up        ├─────────────▶│  CONTROL PLANE  t3.micro     │
  │  orch status    │              │  systemd: spot-orch.service  │
  │  logs  (TUI)    │◀── GET ──┐   │  ┌────────────────────────┐  │
  └────────────────┘          │   │  │  EPOCH SUPERVISOR      │  │
  ┌────────────────┐          │   │  │  observe → decide → act│  │
  │ Grafana (local)│◀── GET ──┤   │  └───────────┬────────────┘  │
  │ live.py loop   │          │   └──────────────┼───────────────┘
  └────────────────┘          │                  │ writes epoch.json
                              │                  ▼
                    ┌─────────┴──────────────────────────────────────┐
                    │                  S3  (the entire API)          │
                    │  runs/<id>/ epoch.json status.json checkpoints/│
                    │             logs/ profile.json nodes/ schedule │
                    │  data/<dataset>/{train,val}.bin                │
                    └─────────┬──────────────────────────────────────┘
                              │ polled every 3s
       ┌──────────────┬───────┴───────┬──────────────┐
       ▼              ▼               ▼              ▼
  ┌─────────┐    ┌─────────┐     ┌─────────┐    ┌─────────┐
  │ sidecar │    │ sidecar │     │ sidecar │    │ sidecar │   GPU boxes
  │ static  │    │ static  │     │ static  │    │ static  │   (spot or
  │torchrun │    │torchrun │     │torchrun │    │torchrun │    on-demand)
  │  └trainer   │  └trainer    │  └trainer   │  └trainer  │
  └─────────┘    └─────────┘     └─────────┘    └─────────┘
```

**No node hosts a rendezvous store, so every node is killable — including rank
0.** The supervisor assigns rank 0 per epoch; a survivor keeps the role.

---

## 3. Processes and where they run

| Process | Host | Lifetime | Entry point |
|---|---|---|---|
| CLI / viewer | laptop | seconds–hours, detachable | `orchestrator/__main__.py` |
| Epoch supervisor | `t3.micro` on-demand controller | the whole run (systemd `Restart=`) | `orch.run_agent` → `experiments._run_supervised` → `supervisor.Supervisor.run` |
| Sidecar | every GPU box | box lifetime | `orchestrator/sidecar.py` |
| `torchrun` (static) | every GPU box | one **epoch** | `sidecar.default_launch` |
| Trainer | one process per GPU | one epoch | `spot_train/train.py` |
| Grafana + live loop | laptop (docker) | while watching | `deploy/grafana/live.py` |
| spotwatch collector | AWS Lambda | cron, ~$1–3/mo | `orchestrator/lambda_spotwatch.py` |
| Inference router / workers | EC2 or local | ROADMAP Part 1 | `src/inference/` |

The laptop holds **no authority**. `orch up` attaches to a dashboard; Ctrl-C only
detaches (`orch.py:1-27`). Discovery is EC2 tags (`orch_role=controller`,
`Name=spot-train-<run_id>`) — there is no local state file to lose.

---

## 4. Control plane

### 4.1 The pure reducer

`decide(Observation, Policy) -> list[Action]` (`supervisor.py:154`) is the entire
membership logic. It has no clock and no AWS calls, so every branch is a table
test.

**Observation** (`supervisor.py:58`) is a snapshot: per-node AWS state,
registration, log-heartbeat age; the current epoch and its members; whether
`metrics.json` exists; seconds since the checkpoint step last advanced; which
scheduled kills are due; how many epochs published without a checkpoint.

**Policy** (`supervisor.py:77`) is what makes one run a shrink experiment and
another a preempt experiment: `replace_on_loss`, `recovery_timeout_s` (600s),
`heartbeat_timeout_s` (90s), `max_epochs_without_progress` (6).

**Actions**: `PublishEpoch`, `TerminateNode`, `LaunchReplacement`,
`WholeGroupRestart(reason)`, `Done`.

The core is deliberately trivial — that is the payoff of central orchestration:

```python
healthy = {n for n in obs.nodes if _healthy(n, policy)}
if healthy and healthy != obs.members:
    publish(obs.epoch + 1, sorted(healthy))
```

A node is healthy iff it **registered**, AWS reports `running`, and its log
object in S3 was modified within `heartbeat_timeout_s` (`supervisor.py:99`).

Two subtleties that were bugs first:

- **A scheduled kill does not shrink the group.** It emits `TerminateNode` only;
  the shrink happens a tick or two later when the box is *observed* gone — the
  same path a real reclaim takes (`supervisor.py:161-166`).
- **The stall clock restarts on every epoch publication** (`supervisor.py:722`),
  because a fresh world legitimately needs time before its first checkpoint.
  `epochs_without_progress` is what still catches a world that *flaps*.

### 4.2 The whole-group restart floor

The most destructive action available — it discards every healthy survivor. It
fires on exactly three conditions, and the reason travels with the action so the
log can never say "restart (floor)" without saying why (`supervisor.py:131-140`):

1. no healthy members at all (nothing to shrink onto);
2. no checkpoint progress for > `recovery_timeout_s`;
3. `epochs_without_progress >= 6`.

Six is budgeted in *preemptions*, not epochs: one preemption publishes two epochs
(shrink, then grow), so 3 would be spent by ~1.5 normal recoveries.

### 4.3 Master election

`elect_master` (`supervisor.py:283`) is sticky to a proven survivor:

1. keep the current master if it is still a member — its c10d store is already up;
2. else the lowest-index member that was in the *previous* epoch;
3. else the lowest-index member.

This is why killing rank 0 recovers like killing a worker: the replacement takes
the dead node's *index* but not its *role*.

### 4.4 The epoch document

Written by the supervisor, read by every sidecar
(`runs/<run_id>/epoch.json`, `supervisor.py:307`):

```json
{"epoch": 7,
 "members": [{"node": 2, "ip": "10.0.1.9", "rank": 0},
             {"node": 0, "ip": "10.0.1.4", "rank": 1}],
 "node_count": 2,
 "master_addr": "10.0.1.9",
 "master_port": 29407}
```

Ordering *is* the authority — rank 0 is the master, there is no separate field to
trust. `master_port = RDZV_PORT + epoch`, so a new epoch never collides with a
half-dead listener from the old one.

### 4.5 The sidecar state machine

`sidecar.py` is stdlib + `s3_store` only, so the code that runs on the DLAMI is
the code that runs in the localhost E2E test. Loop, every 3s:

```
metrics.json exists?              → exit 0 (run complete)
read epoch.json
  named in it?
    epoch changed  → kill_tree(torchrun); relaunch static torchrun for this epoch
    torchrun died  → count crash, exponential backoff (cap 30s), relaunch
    ≥5 crashes in ONE epoch → emit "failed", exit 2 (stop burning money)
  not named?
    stop any stale torchrun; idle (30 min budget) awaiting admission
```

Three details that matter:

- **`kill_tree`** (`sidecar.py:121`) kills torchrun *and* its detached worker
  sessions, then **waits for the workers to actually exit** — a SIGKILLed
  trainer's ~19 GB of GPU memory is not freed until reaped, and relaunching too
  early OOMs the fresh trainer.
- **`prewarm_dataset`** (`sidecar.py:327`) pulls the corpus *before* registering,
  so "registered" means "I can train", not "this instance exists". Measured cost
  of getting this backwards: 214s of survivors idling on a 17 GB `train.bin`.
- **Crash counting is per-epoch**, tracked separately from `running_epoch`, so a
  crash cannot silently zero the counter (`sidecar.py:196-201`).

### 4.6 A preemption, end to end

```
t+0.0   node 3 vanishes (reclaim, or supervisor TerminateNode)
t+0.x   survivors' NCCL collective aborts (NCCL_TIMEOUT, steady-state short)
        → their torchrun exits → sidecars drop the corpse
t+≤3    supervisor tick: node 3 not registered / not running / log stale
        → decide() → PublishEpoch(N+1, survivors) [+ LaunchReplacement(3)]
t+≤6    sidecars read the new epoch → static torchrun at world N−1
        → load_group_latest: every survivor has the same step-aligned LOCAL
          snapshot → group MIN → instant disk restore
t+~11   survivors are taking steps again          ← "training gap"
...     replacement boots, pulls dataset, registers
t+~154  supervisor observes it healthy → PublishEpoch(N+2, all N)
        → group restores from S3-latest, world back to N   ← "full recovery"
```

The two numbers are different things and both are reported: **training gap**
(≈11s measured at 8 nodes) is how long *nobody* was training; **full recovery**
(≈154s, of which ~118s was the dataset pull) is how long the world ran degraded.

### 4.7 Control-plane durability

The supervisor's cross-tick memory lives in a process, so every piece of it that
matters is mirrored to S3 and restored on entry to `run()`:

| State | Restored by | Bug it prevents |
|---|---|---|
| epoch + members + master | `restore_state` (`supervisor.py:239`) | A restart reads `epoch == 0`, republishes epoch 1 over a live world, all N sidecars churn, every one gets "replaced" — this grew an 8-node fleet to 14. |
| instance ids + log keys | `adopt_fleet` (`experiments.py:469`) | `_run_supervised` relaunches the whole fleet on top of one that never stopped training. |
| fired kills + train start | `_restore_schedule` (`supervisor.py:506`) | A restart at hour 8 replays the entire `PREEMPT_SCHEDULE` from zero. Persists a **wall** clock, because `time.monotonic()` resetting with the process *is* the bug. |

Plus, on the boxes: an unconditional systemd dead-man timer armed as the first
boot action (`bootstrap.py:1007`), so a fleet whose control plane dies still
terminates itself instead of billing forever.

### 4.8 Experiment drivers

`experiments.py` owns launch/teardown; the supervisor owns steady state. Every
multi-node experiment is one function, `_run_supervised` (`experiments.py:535`),
parameterised by `(kind, budget, replace_on_loss, kill_schedule)`:

| Verb | `replace_on_loss` | Kills |
|---|---|---|
| `multinode` | — | none |
| `multinode-shrink` | `False` | one (+ PASS/FAIL verdict) |
| `multinode-preempt` | `True` | `PREEMPT_SCHEDULE` |
| `scaling-{experiment,clean,preempt}` | varies | sweeps over world size |

Sharing one path is what keeps the cost ledger, the world-size staircase and the
degraded-phase accounting identical across all of them.

`PREEMPT_SCHEDULE` expresses arbitrary chaos as a string
(`config.py:846`): `"480:3;960:L;1440:1,4;2400:0,1,2,3,4,5"` — simultaneous
victims, `L` = whoever is leader when the entry fires (resolved at fire time, so
sticky re-election keeps getting exercised), validated before any box launches.

---

## 5. Data plane — the trainer

### 5.1 One resume path

```python
blob = checkpoint.load_group_latest(cfg.checkpoint_uri, cfg.local_checkpoint_dir, ddp)
if blob is not None: start_step = checkpoint.restore_into(blob, ...)
else:                start_step = 0
```

There is no "am I a replacement?" branch anywhere (`train.py:228-241`).

`load_group_latest` (`checkpoint.py:478`) is the two-tier, N-rank generalisation:
each rank offers the best step it can reach (node-local disk, else S3), the group
takes the **MIN**, everyone restores that step from the cheapest tier holding it.
Both interesting cases fall out with no membership detection:

- **survivors only** — everyone holds the same step-aligned local snapshot →
  instant disk restore, ~zero lost work;
- **a fresh replacement is present** — its best is S3-latest, which becomes the
  MIN → the whole group restores S3-latest.

If a rank cannot reach the agreed step it raises loudly rather than silently
diverging.

### 5.2 What a checkpoint contains

`CKPT_VERSION = 3` (`checkpoint.py:43`):

```
version · step · trained_seconds · model · optimizer · rng · loader · scaler
```

Two tools answer "is this checkpoint real?": `verify()` (schema complete + every
float tensor finite) and `smoke_test()` (restore into a *fresh* model, one
forward, assert finite loss). Both run every Nth checkpoint, on CPU, on the
background writer thread.

### 5.3 Two tiers, both async

| Tier | Writer | Who | Purpose | Keep |
|---|---|---|---|---|
| Node-local disk | `AsyncLocalSaver` | every node's `local_rank 0` | instant survivor restore | 2 |
| S3 | `AsyncCheckpointer` | rank 0 only | durable; replacements | `CHECKPOINT_KEEP` (10) |

The split is two-phase. The **snapshot** — a point-in-time CPU copy — stays on
the critical path, because it is what makes the write safe while the optimizer
keeps mutating live tensors. Serialize + upload + verify + prune all move off it.

Two optimizations in that snapshot are load-bearing at GPT-2 scale:

- **`_batched_d2h`** (`checkpoint.py:146`): GPT-2 124M has ~450 tensors, and the
  naive `t.to("cpu")` per tensor *synchronises* on a busy GPU each time. One
  pinned staging buffer + async copies + **one** sync turns ~450 stalls into a
  bandwidth-bound transfer.
- **One blob feeds both tiers** (`checkpoint.py:382-402`). Rank 0 used to take
  two byte-identical 1.5 GB copies.

Measured impact of making the local tier async: it had been **99s of 301.7
training-seconds — 33% of all training time**.

Both writers allow exactly **one write in flight**; `submit()` returns `False`
while busy, so memory is bounded at one blob and a slow S3 day cannot queue-pile.
The durability trade is explicit: worst-case lost work is the checkpoint interval
*plus one upload*. Preempt and final checkpoints stay synchronous, with a
`flush()` first so the writers never race them.

### 5.4 Budget-in-checkpoint

`trained_seconds` is cumulative **in-loop** wall-clock, carried in the blob. On
resume:

```python
cfg.max_seconds = cfg.train_budget_seconds - trained_before   # train.py:246
```

Boot, dataset pulls, NCCL stalls and crash teardown are therefore **never billed
against the run's training budget** — a 24h budget buys 24h of stepping, however
many times the fleet was rebuilt. It is also what makes `extend` well-defined:
`--budget` is the new *total*, not an increment (`experiments.py:673`).

### 5.5 Constant global batch

`grad_accum_steps(global, world, batch) = ceil(global / (world × batch))`
(`train.py:130`) is recomputed at every launch, so 1, 2, 4 and 8 nodes take the
**same optimizer steps on the same batch**. A membership change alters wall-clock
per step, not gradient statistics — which is what keeps the LR schedule valid
across a preemption and makes scaling numbers attributable.

The LR schedule itself is a pure function of the step number (`train.py:142`), so
no schedule state has to be checkpointed.

### 5.6 DDP details that only matter under failure

- **Two collective timeouts.** Startup runs on `NCCL_INIT_TIMEOUT` (long: the
  TCP connection mesh cost grows ~O(N²) and 8-node startup does *not* fit in 20s
  — torch misreports that abort as a parameter-shape mismatch). Once startup
  collectives are done, `tighten_timeout` drops to the short steady-state
  `NCCL_TIMEOUT`, because that timeout **is** the in-band signal that a peer died
  (`distributed.py:37-108`).
- **Coordinated stop.** `all_reduce_stop` is the *last* collective in every loop
  body, so all ranks break on the same iteration and none is left blocking in the
  next backward (`distributed.py:116`). The stop *reason* is then agreed with a
  second balanced all-reduce.
- **Step-aligned local tier.** `broadcast_flag` lets rank 0 decide "checkpoint at
  this step" and every node's `local_rank 0` snapshot the *same* step — which is
  precisely what lets a shrunken group later agree on a common local resume step
  (`distributed.py:130`).
- **bf16 gradient compression** on the all-reduce, NCCL only (multi-node comms
  are TCP-bandwidth-bound with no EFA on g4dn/g5); gloo stays fp32 so CPU tests
  remain bit-exact.
- **Every rank executes eval and sampling blocks** even though only rank 0 prints
  — a rank that skipped them would run ahead and stall the group at the next
  backward.

### 5.7 Stall watchdog

A rank blocked inside NCCL cannot emit anything. A daemon thread watches forward
progress and, past `STALL_THRESHOLD_SECONDS` (45s), emits `stalled` **stamped at
the last good step** — the true onset, not when it was noticed (`train.py:388`).
45s clears the longest legitimate pause (a checkpoint snapshot, a full-val eval)
while still flagging a dead peer ~10× sooner than NCCL's own timeout.

---

## 6. Storage plane — S3 is the whole API

There is no RPC layer, no message bus, no database. Every cross-process contract
is an S3 object, which means the entire live state of a run is inspectable with
`aws s3 ls` and reconstructible after the fact.

```
s3://<bucket>/
  data/<dataset>/{train.bin, val.bin, meta.pkl}      staged once
  runs/<run_id>/
    epoch.json          membership   — supervisor writes, sidecars read
    nodes/node<i>.json  registration — box writes {ip, instance_id}
    status.json         observability, rewritten every tick
    status/<epoch_ms>.json   one immutable object per tick = fleet HISTORY
    schedule.json       durable chaos-schedule progress {train_start_wall, fired}
    checkpoints/ckpt-<step:012d>.pt   durable tier (atomic; SHA-256)
    logs/node<i>.log, node<i>-r<k>.log, orchestrator.log
    metrics.json        the run's DONE signal
    samples.json, samples/<step>.json
    profile.json        the run's source of truth (timeline, curves, cost)
  orchestrators/<orch_id>/{orch,progress,heartbeat}.json, orchestrator.log, boot.log
  fleet/<fleet_id>/workers/<worker_id>.json          inference heartbeats
  spotwatch/…                                        availability shards
```

Conventions worth knowing:

- **`metrics.json` is the done signal.** It is written *only* on a completed
  budget — a preempted trainer checkpoints and exits without one, so its presence
  is unambiguous for supervisor and sidecar alike.
- **Checkpoint names are zero-padded to 12 digits**, so lexicographic sort equals
  numeric sort and "latest" is one `list_objects_v2` away.
- **A replacement gets a fresh log key** (`node<i>-r<k>.log`), so it can never
  clobber the dead attempt's log; the viewer keeps both readable forever.
- **`status/` is history, `status.json` is current.** `status.json` is
  overwritten in place, so world-size and slot occupancy would exist only for as
  long as a laptop happened to be polling. The supervisor is the writer, so it
  publishes each tick as its own immutable object — one object per tick rather
  than one growing JSONL, because rewriting a 13 MB file every 10s is ~78 GB of
  PUTs over 24h (`supervisor.py:578-597`).
- **Registrations are ownership-checked.** `nodes/node<i>.json` is keyed by node
  *index* and persists after a kill, so the supervisor compares `instance_id`
  against the slot's current occupant. Without that check the reducer regrew the
  world onto a corpse and idled survivors for 204s (`supervisor.py:631-643`).

---

## 7. Observability

The pipeline is event-sourced, and it reuses the log transport that already
exists rather than adding an ingest path.

```
trainer / sidecar / supervisor
        │  events.emit(...)  → one line: [event] {"ts":…, "node":…, "state":…}
        ▼
   stderr → box log → S3 (synced every few seconds)
        │
        ├── profile.py    regex-parses step/eval lines + control-plane marks
        │      └── runs/<id>/profile.json  ← SOURCE OF TRUTH
        │      └── W&B mirror (optional; no-ops without a key)
        ├── logview.py    full-screen TUI: per-node tabs, grid, Gantt, events
        └── live.py       → export.py → timeseries.csv + occupancy.csv → Grafana
```

**Event states** (`spot_train/events.py`): `provisioning`, `reconfiguring`,
`training`, `stalled`, `down`, `killed`. `ts` is stamped **at the source**, so
the 3s log-sync latency affects when a reader *sees* an event, never when it
*happened*. `down` vs `killed` differ only by cause — reclaimed vs
orchestrator-initiated — which is what lets the timeline distinguish real spot
behaviour from injected chaos.

**Per-step lines carry their own wall clock** (`train.py:449`), appended last so
the existing regex cannot break. Without it, every timeline plot has to
reconstruct step times by anchoring to events and summing ms/step — an error that
produced two wrong E4 plots before it produced a right one. Non-master ranks log
in a deliberately different format so they stay out of the rank-0-authoritative
profile while still proving each node is training.

**Grafana** (`deploy/grafana/`) renders live via an unusual but deliberate path:
the Infinity datasource's browser module 404s in this environment and fails
*silently* (panels read "No data" while backend queries succeed), so `live.py`
regenerates the provisioned dashboard JSON every 10s and lets Grafana's own
10s disk reload be the live update. It costs a 200 KB rewrite per tick and adds
no new failure mode on the day of a paid run.

`scripts/run_with_dashboard.sh` exists because the live loop is a separate
*process* but must not be a separate *step* — having to remember it is how a run
ends up with no dashboard, which already happened once.

---

## 8. Cost model

`RunProfile` keeps a ledger with one row per EC2 box (`profile.py:133`):
`(instance_id, market, az, hourly_usd, started_at, stopped_at)`, billed
per-second from `running` to terminate — matching how AWS bills, rather than
multiplying a fleet size by a duration. The spot rate is the actual per-AZ price
at launch; `None` means unknown, and sums skip such rows while `cost_dict` still
counts them so nothing hides. The control-plane box adds its own row via
`on_profile`, so a multi-day run's cost is honest end to end.

---

## 9. Failure model

| Failure | Detected by | Response | Cost |
|---|---|---|---|
| Worker node lost | AWS state / stale log heartbeat (≤ 90s, usually ≤ 3s) | shrink epoch; survivors restore from local disk; replacement launched | ~11s training gap |
| **Master (rank 0) lost** | same | role stays with a survivor; replacement joins as a worker | same as a worker |
| k nodes lost at once | same | one shrink epoch to the survivors | measured per-k in E5 |
| All nodes lost | `healthy` empty | `WholeGroupRestart("no healthy members")` | full relaunch from S3 |
| Node alive but wedged | log heartbeat stale > 90s | treated as dead | as a loss |
| Rank blocked on a dead peer | in-process watchdog, 45s | emits `stalled` at true onset | observability only |
| torchrun crash-loops in one epoch | sidecar counter ≥ 5 | box emits `failed` and exits; fleet reaped | bounded |
| World never forms | 6 epochs with no checkpoint | `WholeGroupRestart` with the reason | bounded |
| Supervisor process dies | systemd | `restore_state` + `adopt_fleet` + `_restore_schedule` → restart is a no-op | ~0 |
| Control-plane box dies | heartbeat age | `orch up --run-id <id>` re-enters the live run | boot time |
| Laptop disconnects | — | nothing; the laptop holds no authority | 0 |
| Everything dies | systemd dead-man timer on each box | boxes self-terminate | bounded billing |
| Async checkpoint write fails | background thread | logged + counted; training continues on the previous checkpoint | ≤ one interval |
| Corrupt/partial checkpoint | atomic rename + SHA-256 + `verify` + smoke test | cannot be observed as "latest" | 0 |

---

## 10. Safety rails

Because this system's failure modes cost money, not just correctness:

- **Region is pinned and enforced.** `aws.set_region()` *raises* on anything but
  `us-east-1` (`aws.py:51`); the account is shared with an inference platform
  that owns `us-east-2`, and G/VT quota is per-region. Override is explicit:
  `ALLOW_REGION_OVERRIDE=1`.
- **`aws.py` is the only module that calls AWS.** Every mutating call logs first
  and honours `--dry-run`, which walks the full control flow minus the waiting.
- **Fleet operations are tag-scoped.** Source `scripts/fleetctl.sh`; never write
  a bare `describe-instances`/`terminate-instances` in a driver.
- **vCPU gate.** `wait_vcpu_headroom` blocks only when `used + needed` exceeds
  quota — replacements are *not* unconditionally serialized behind a dying box's
  quota release, which used to cost tens of seconds on every recovery.
- **Two dead-man timers.** One on every training box, one on the prep box, both
  armed by systemd as the first boot action so they survive the boot script.
- **Least-privilege IAM**, per principal, in `docs/iam/` (controller / worker /
  one-time setup). Credentials live in a git-ignored `.env`; no code reads them —
  boto3 resolves them at call time, so the same code works with laptop SSO now
  and an instance-profile role on the controller.
- **Prune after, never before.** A new checkpoint must be durable before an old
  one is deleted, and `.tmp` objects are never prune candidates (deleting one
  would race the copy in an in-flight atomic save).

---

## 11. Boot path

`bootstrap.py` builds EC2 user-data. Two DLAMI realities shape it: the PyTorch
env only auto-activates for the `ubuntu` **login** shell (so everything runs via
`sudo -u ubuntu -i`, with an `import torch` preflight that fails loudly), and we
run from source with `PYTHONPATH` rather than `pip install -e .` to avoid writing
into a root-owned framework env.

```
arm dead-man timer  →  clone repo + nanoGPT submodule  →  install deps
  →  resolve DATA_LOCAL_DIR (instance-store NVMe if present, else root volume)
  →  write source-able env file  →  start the log→S3 sync loop
  →  sidecar: prewarm dataset → register → obey epoch.json
```

No credentials are passed in user-data; the box reads and writes S3 through its
instance profile.

---

## 12. Configuration flow

```
.env  +  recipes/*.env        (operator's shell)
   → OrchestratorConfig       (config.py; every field env-overridable)
   → _TRAINER_PASSTHROUGH     (relayed verbatim, only when set)
   → user-data env file       (on the box)
   → TrainConfig.from_env()   (parsed on the box)
```

The orchestrator deliberately **never branches on** trainer knobs it relays —
they stay untyped strings, parsed once, on the box. Validation that *can* happen
before spending money does: `PREEMPT_SCHEDULE` is parsed and range-checked
against `NODES` before a single instance launches (`config.py:846-908`).

---

## 13. Inference fleet (ROADMAP Part 1)

Same philosophy applied to serving:

| Component | Role |
|---|---|
| `inference/registry.py` | Workers overwrite `<workers_uri>/<id>.json` every few seconds; live = `last_seen` within TTL (~15s). S3 as transport again — no new IAM, and `aws s3 ls` shows fleet state. |
| `inference/worker.py` | FastAPI, loads the newest checkpoint at boot, OpenAI-shaped completions, heartbeats. A spot kill needs no cleanup. |
| `inference/router.py` | The one stable (on-demand) component: round-robins over live workers and **reroutes on failure** — connection error, timeout or 5xx goes to the next worker, not to the client. |
| `inference/service.py` | Reuses the trainer's own `checkpoint`/`s3_store`/`build_model`, so the served model is byte-for-byte the training artifact. |
| `loadgen/` (Go) | **Open-loop** load: dispatch on a fixed schedule regardless of server latency, so a struggling fleet shows as rising latency and drops rather than a politely slowed client. `-kill-after`/`-kill-cmd` injects chaos and stamps the kill into the report. |

## 14. spotwatch

An unattended availability collector (`spotwatch.py` + `lambda_spotwatch.py`,
stdlib + boto3 so it runs in Lambda): EventBridge cron → Spot Placement Scores,
prices and interruption data → JSONL shards in S3 → `spotwatch report` renders
heatmaps, per-pool rankings, relocation scenarios and a calibration check. ~$1–3
per month. It answers "when and where is capacity actually available?" without a
human polling the console — which is what a spot drought made necessary.

---

## 15. Testing architecture

502 tests collected, all CPU, no AWS. Three layers:

1. **Pure-function tables.** `decide()` has no clock and no I/O, so the whole
   membership matrix is a table test (`test_supervisor.py`,
   `test_chaos_schedule.py`, `test_kill_schedule_durable.py`).
2. **Determinism / resume gates.** `test_kill_resume.py` is the one to run before
   any cloud work: kill a local run mid-flight, resume, assert the loss continues
   from the checkpoint rather than resetting. Plus `test_resume_continuity.py`,
   `test_checkpoint_disk.py`, `test_async_checkpoint.py`,
   `test_checkpoint_prune.py`.
3. **Whole-protocol E2E on localhost.** `test_epoch_e2e.py` runs *real* sidecars
   against *real* static torchrun with a dummy worker module
   (`SIDECAR_TRAIN_MODULE`), over local directories instead of S3 — the same code
   paths that run on the DLAMI. `test_supervisor_readopt.py` covers restart
   adoption; `test_control_plane_gap.py` covers the outage-rendering case.

The `s3_store` local/S3 duality and the sidecar's stdlib-only diet exist
specifically so these can be real tests rather than mocks.

---

## 16. Known gaps

- **The Go control plane is not built.** The supervisor's observe/compare/act
  loop is Python today; ROADMAP Parts 3/7 move it to Go/K8s. The pure reducer is
  the seam that makes that a port rather than a rewrite.
- **The 1c headline is unrun.** Fault tolerance is cloud-proven; the
  spot-cheaper-than-on-demand-to-the-same-loss comparison is not (see the
  on-demand-only banner in `docs/final-run-plan.md`, and the spot-drought note in
  `docs/next-steps.md`).
- **Recovery is dataset-pull-bound.** ~118s of the ~154s full recovery is pulling
  a 17 GB `train.bin`. Baking it into the AMI was considered and dropped; report
  the number as measured, not as optimised.
- **Several experiment drivers live outside the repo.** E4/E5/ladder/total-loss
  drivers and their collectors sit in the git-excluded `.context/`, and ~12
  tracked docs reference those paths — so a fresh clone can run the standard CLI
  experiments but cannot reproduce those results. `PREEMPT_SCHEDULE` now
  expresses what most of them needed; promoting the rest into a tracked
  `experiments/` directory would close the hole.
- **Single-writer by construction.** There is exactly one supervisor, and no
  leader election among control planes. That is deliberate for N ≤ 8 — the
  restart path is a no-op and the boxes have a dead-man switch — but it is the
  assumption to revisit before the fleet gets much larger.
