# Final training run — plan and context

**Audience:** the agent picking this up. You are taking a system that is built
and cloud-proven at small scale to a **24-hour, 8-node, chaos-included GPT-2 /
OpenWebText run**. Everything is `us-east-1` (see `docs/region-split.md` — the
inference platform owns us-east-2 and `aws.set_region()` raises on anything
else).

> ## ⚠️ ON-DEMAND ONLY — NO SPOT INSTANCES
>
> **`MARKET=on-demand` everywhere. Every node, every phase, no exceptions.**
> Do not add a spot path, a spot fallback, or a spot price check. If you find a
> reference to spot in this document, it is a bug in the document.
>
> **Consequence, stated plainly: this run cannot make a cost-savings claim.**
> The 1c headline ("spot cheaper than on-demand to the same loss") is *not* what
> this run proves and must not be claimed from it. What this run proves is
> **fault tolerance**: training survives repeated node loss with high goodput and
> bounded lost work. That is a real systems result on its own. Do not dress it up
> as an economic one.

Order is fixed and each gate is a stop: **Batch 1 (code) → Batch 2 (1h
rehearsal) → Batch 3 (24h run).**

**8 nodes. 24 hours.** Duration is 24h with a documented option to extend to 36h
(§8) — every systems claim is a per-event metric already captured in the
rehearsal, so duration only buys event count, endurance coverage and a clean
sentence. A *complete* 24h run beats a 36h run truncated at hour 30.

Numbers marked *measured* came from real runs in this account. Numbers marked
*projected* are arithmetic on those. If you change a parameter, redo the
arithmetic rather than trusting the projection.

---

## 1. Where the system is today

Built and cloud-proven: single-node kill/resume, DDP, multinode epoch supervisor
with node replacement, two-tier checkpoints, budget-in-checkpoint, cost ledger,
run profiles, and a live Grafana dashboard that renders boot, kill, degraded and
regrow phases in real time.

### Measured baseline — your evidence table

| Quantity | Value | Source |
|---|---|---|
| Step time, 8 nodes, OWT | **4.013 s/step** | `multinode-1785795454`, 897 steps / 3599.89 s train |
| Throughput, 8 nodes | **122,470 tok/s** | 491,520 tok/step ÷ 4.013 s |
| Global batch | **480 seq × 1024 tok = 491,520** | `GLOBAL_BATCH_SIZE=480`, `BLOCK_SIZE=1024` |
| Model | **GPT-2 124M** (12L/12H/768d) | node log: `number of parameters: 123.69M` |
| Exact param count | 124,475,904 | 124,354,560 decayed + 121,344 non-decayed |
| **Training gap on node loss** | **11 s** | run `1786231481`: last step 1786231838 → first step 1786231849 |
| **Full-world recovery** | **154 s** | shrink epoch 1786231848 → regrow epoch 1786232002 |
| — of which dataset pull | **118 s (77%)** | `dataset ready in 118.0s` — see A7 |
| Val loss @ 897 steps | 3.9842 | `multinode-1785795454` |
| Val curve available | 35 points, 8.05 → 3.98 | `EVAL_INTERVAL_STEPS=25` |
| Goodput (heavy chaos) | 0.629 | E5, 14 kills in 3336 s |
| Utilization (same run) | 0.853 | Σ ms_per_step / elapsed |

**The two recovery numbers are different things and both matter.** 11 s is the
fault tolerance: survivors re-form and keep training at N−1. 154 s is EC2 boot +
dataset pull for the replacement. Report them separately.

`full_world` in `profile.json` fires 0–3 s after `kill` — it does **not** mean
the world is back to N. Derive recovery from the epoch timeline instead.

### Quota and price, verified

| | Value |
|---|---|
| **On-Demand G/VT vCPU** (L-DB2E81BA) | **64** |
| 8 × g5.xlarge | 32 vCPU — fits, with headroom for replacements |
| g5.xlarge on-demand, us-east-1 | **$1.006/hr** |

Replacements transiently exceed 8 boxes (old one shutting down, new one pending),
so headroom matters. 64 vCPU = 16 boxes; comfortable.

---

## 2. What the 24-hour run proves

Three claims. Every one must be answerable from `profile.json` + the dashboard.

1. **Preemption is survivable at scale.** N=8, injected node losses including a
   simultaneous loss of 6, zero whole-group restarts, loss curve continuous
   across every event.
2. **Lost work is bounded by the checkpoint interval.** `steps_at_risk` never
   exceeds `CHECKPOINT_INTERVAL_SECONDS / 4.013 s`.
3. **Downtime is not billed.** `TRAIN_BUDGET_SECONDS − trained_seconds` means
   wall-clock spent degraded or booting doesn't consume budget.

**Not a claim: cost savings.** See the banner.

### Targets

| Metric | Target | Basis |
|---|---|---|
| Training gap per kill | ≤ 30 s | measured 11 s |
| Full-world recovery | ≤ 240 s | measured 154 s |
| **Mass-loss recovery (2→8)** | ≤ 300 s | E5 precedent; parallel boots, staircase regrow |
| Goodput | ≥ 0.90 | E5's 0.629 was 14 kills in 56 min — far denser than this |
| Utilization | ≥ 0.95 | measured 0.853 under heavy chaos |
| Whole-group restarts | **0** | any non-zero fails the core thesis |
| Steps completed | 19,000–20,600 | projected below |
| Tokens | 9.3–10.1 B | steps × 491,520 |
| Val loss | ≤ 3.2 | extrapolated — a sanity bound, not a promise |
| Degraded time | ≤ 6% of wall | ~1.05 h of 24 h = 4.4% |

---

## 3. Chaos design

All node loss is **injected** via EC2 `TerminateInstances` — an abrupt kill with
no warning, which is *harsher* than a real spot interruption's 2-minute notice.
Since we are not on spot, there is no ambiguity to manage: these are deliberate
fault injections and the writeup should say exactly that.

### The 24-hour schedule

| Event | Count | When | Proves |
|---|---|---|---|
| Single-node kill | ~20, jittered 55–65 min | throughout | the common case |
| **Leader / rank-0 kill** | ≥2, forced | — | re-election |
| **Simultaneous 2-node kill** | 3 | — | multi-slot replacement |
| **MASS LOSS — 6 of 8 at once** | **1** | **~hour 16** | the headline event |
| **Supervisor reboot** | 1 | ~hour 8 | G4-lite: fleet re-adopted, not rebuilt |
| Kill during a checkpoint write | ~3 (happens on its own) | — | atomic rename |

Kills must be spaced further apart than recovery (154 s). Do not exceed ~1 kill
per 20 min sustained — beyond that you are measuring a fleet churning faster than
it can boot, which is a true but uninteresting result.

### The mass-loss event — the money shot

**E5 already survived killing 7 of 8 nodes simultaneously** and finished 899 steps
at val 3.98 — the `1→2→3→4→5→6→7→8` regrow staircase in
`docs/grafana-handoff.md`. A plan without this under-sells the result.

**6 of 8, not 7:** two survivors keep DDP alive, the realistic "an AZ went away"
shape. World 1 is a different path (single process, no collectives) and is
*already* proven by E5. Either is defensible; say which you ran.

At world 2 expect step time ~4× steady state (two nodes carry all 40
gradient-accumulation micro-batches). Recovery is a staircase, not a step — six
replacements boot in parallel but register at slightly different times.

Place it at **~hour 16**: long steady-state history before, visible full recovery
and sustained training after. Not near the end.

---

## 4. Parameters

Start from `recipes/gpt2-owt.env` and override.

```bash
# --- model / data: unchanged, this is the flagship config ---
DATASET=openwebtext
N_LAYER=12 N_HEAD=12 N_EMBD=768 BLOCK_SIZE=1024
GLOBAL_BATCH_SIZE=480          # constant across world size via grad accum
BATCH_SIZE=12

# --- market: ON-DEMAND. NOT NEGOTIABLE. ---
MARKET=on-demand
INSTANCE_TYPE=g5.xlarge
NODES=8

# --- horizon: CHANGED. Recipe ships 600000 for both. ---
LR_DECAY_STEPS=20000           # matched to the 24h horizon
MAX_STEPS=24000
WARMUP_STEPS=2000

# --- eval ---
EVAL_INTERVAL_STEPS=300        # ~65 points over 24h
EVAL_ITERS=200

# --- checkpointing: 30 s is far too dense for 24h ---
CHECKPOINT_INTERVAL_SECONDS=120   # ≈30 steps at risk; 690 ckpts over 24h
CHECKPOINT_KEEP=10                # requires G1 (DONE)

# --- logging / samples ---
LOG_INTERVAL_STEPS=10
SAMPLE_INTERVAL_STEPS=1500

# --- budget ---
TRAIN_BUDGET_SECONDS=82800     # 23h of TRAINING; wall ≈24h with chaos
```

### The LR-schedule decision

The recipe ships `LR_DECAY_STEPS=600000` (nanoGPT's full OWT schedule). At ~20k
steps you'd finish at 3% of it with LR barely decayed and a visibly unfinished
curve. Tuning to 20000 gives a curve that ends at a proper minimum.

It is **irreversible once the run starts** and must be **identical** between any
baseline and the main run. Pin it in the recipe, not on the command line.

### Projection (redo if you change anything)

```
82,800 s training ÷ 4.013 s/step   = 20,633 steps   (at full world)
minus ~5% degraded-world slowdown  ≈ 19,600 steps
× 491,520 tok/step                 ≈ 9.6 B tokens
wall = 82,800 s + ~24 recoveries   ≈ 86,500 s ≈ 24.0 h
```

---

## 5. Scope — only realistic failures

**Descoped deliberately** to failures that occur without simulation.

**In scope:**

| Failure | Why real | Response |
|---|---|---|
| Supervisor throws | Our code. Trigger: `InsufficientInstanceCapacity` on a replacement launch — likely, not hypothetical | **Reboot it.** systemd restarts and resumes the same run_id |
| Laptop disconnect | Certain — the operator will lose wifi repeatedly | `orch.py` already handles it; needs Gap1 for dashboard history |
| Kill during checkpoint write | ~12% per kill × 24 ≈ 3 occurrences | Safe by construction: temp-key → atomic rename |

**Out of scope (needs simulation):** spontaneous t3.micro death, correlated AZ
loss. **Dropped:** G3 halt flag, G4 cascade metering, fault-injection hook.

### G4-lite — what makes "just reboot" cheap

`_reap_orphans` (`orch.py:752`) terminates **every** training box from the
previous attempt. But the fleet keeps training fine without a supervisor —
sidecars run static torchrun against the last published epoch doc, and supervisor
state lives in S3, not memory. Today a reboot destroys 8 healthy boxes and
rebuilds: ~4 min of full-fleet boot plus all work since the last checkpoint.

**Fix: reap only boxes NOT in the current epoch's membership.** Genuinely
orphaned boxes still get cleaned; a healthy fleet gets re-adopted.

---

## 6. Batch 1 — all code, no AWS (~2.5 h, $0)

| | Item | Status |
|---|---|---|
| G1 | Prune S3 checkpoints to `CHECKPOINT_KEEP` | **DONE** — 9 tests, knob relays to the box |
| G2 | Arm both dead-man timers | ~15 min |
| G4-lite | Narrow `_reap_orphans` to non-members | ~30 min |
| A8 | `extend <run_id> --budget <new_total>` | ~20 min |
| **R1** | **Resumability A/B test** | **~20 min — see below** |
| Gap1 | Supervisor publishes per-tick status objects to S3 | ~20 min |
| CH1 | Let `multinode-preempt` express **simultaneous** kills | ~20 min |
| D1 | Val-loss series under the training-loss curve | ~20 min |
| D2 | Model / params / dataset tiles from real log data | ~20 min |

**CH1 is not optional.** `multinode-preempt` builds its kill schedule as
increasing times and *cannot* express a simultaneous pair — which is why
`e4_rolling_pairs.py` exists as a bespoke driver. The mass-loss event needs it.

**D1/D2 validate for $0** against existing artifacts: `multinode-1785795454` has a
real 35-point OWT val curve; `multinode-preempt-1786231481`'s logs carry
`number of parameters: 123.69M` and `DATASET=openwebtext`.

### R1 — how resumability actually gets verified

**No existing test drives the train loop through a kill and resume.** Coverage is
checkpoint-level primitives (`save_verify_restore`,
`test_checkpoint_carries_trained_seconds`, group agreement) — all good, none of
which proves the *loop* resumes.

**The gold-standard test: A/B against an uninterrupted control.**

- Run A: 20 steps straight through.
- Run B: 10 steps → checkpoint → "complete" → extend → 10 more.
- Assert `final_loss(B) ≈ final_loss(A)` within tolerance.

Bit-exactness is deliberately relaxed for the MVP, so this is approximate — but
"within a few percent of the control" exercises weights, optimizer moments,
loader position and RNG simultaneously.

Each failure below **looks like success** if you only check that it ran:

| Failure | Why it's invisible | Assertion |
|---|---|---|
| Weights reinitialized | Still gives a clean decreasing curve, just from scratch | First resumed loss ≈ last loss of seg 1, **not** ≈ `ln(vocab)` |
| Optimizer moments lost | Loss spikes then recovers; reads as a blip | No material spike at the boundary |
| **Loader position reset** | You re-train the same data — loss looks **better** | Restored offset > 0 and advanced |
| `trained_seconds` reset | Segment 2 runs a full fresh budget → **2× cost** | Carried forward |
| `metrics.json` clobbered | Silent; noticed when the 24h result is gone | `metrics-seg1.json` intact |

Not covered by R1, worth a deliberate check in Batch 2: resume across a **code
change** (the patch-and-continue case), since `CKPT_VERSION` gates compatibility.

---

## 7. Batch 2 — the 1-hour rehearsal (~$9)

**One run: a compressed replica of the 24h run.** Same 8 nodes, same config, all
event *types* in the same order, inside ~55 minutes. It is not a scale model of
the chaos *rate* — 20 single-node kills prove nothing new in an hour and are
dropped to two.

```bash
TRAIN_BUDGET_SECONDS=3300      # 55 min
LR_DECAY_STEPS=20000           # SAME as the 24h run — do NOT retune
MAX_STEPS=24000
EVAL_INTERVAL_STEPS=25         # ~30 points in the hour
CHECKPOINT_INTERVAL_SECONDS=120
SAMPLE_INTERVAL_STEPS=200
MARKET=on-demand
```

Launched via `orch up`, so the remote control plane is on the critical path.

| t+ | Event | Validates |
|---|---|---|
| 8 min | single non-leader kill | the common case, CH1 scheduling |
| 16 min | **leader kill** | re-election at N=8 |
| 24 min | **simultaneous 2-node kill** | CH1, via the main driver |
| 32 min | **supervisor reboot** | G4-lite — fleet re-adopted, not rebuilt |
| 40 min | **MASS LOSS — 6 of 8** | the headline event, at full scale |
| 50 min | kill during a checkpoint write | atomic rename |
| throughout | kill `live.py` for 5 min | Gap1 — no history hole |
| throughout | G1 count plateaus; G2 timer present; D1/D2 render live | — |
| after | `extend` +10 min, patching a trivial file first | A8 + resume-across-code-change |

Spacing ≥8 min everywhere, comfortably above the 154 s recovery; the mass loss
gets 10 min for its ~300 s staircase regrow.

### Go/no-go for the 24h run — all must hold

1. Zero whole-group restarts.
2. Training gap ≤ 30 s single-node; ≤ 60 s on the 2-node.
3. Full-world recovery ≤ 240 s; **mass-loss recovery ≤ 300 s**.
4. Loss curve continuous across every event — no reset, no divergence.
5. `steps_at_risk` never exceeds 30.
6. Supervisor reboot resumes the same run_id and does **not** rebuild the fleet.
7. S3 checkpoint count plateaus at `CHECKPOINT_KEEP`.
8. `trained_seconds` ≈ budget — downtime not billed.
9. `live.py` blackout leaves no hole in world size / Gantt.
10. `extend` resumes at the right step with the loss curve continuous.

**Cost:** 8 nodes × ~1.1 h + boot + replacement overlap ≈ 9.5 node-hours ×
$1.006 = **$9.56**, plus t3.micro ≈ **$9.60**.

---

## 8. Batch 3 — the 24-hour run (~$195)

Start only after every gate passes. Fresh run_id — do **not** extend the
rehearsal into it, or the 24h goodput number is polluted by a compressed-chaos
hour.

| | node-hours | on-demand |
|---|---|---|
| 8 nodes × 24 h | 192 | $193.15 |
| boot overhead | 0.53 | $0.53 |
| replacement overlap (~24 × 154 s) | 1.03 | $1.04 |
| t3.micro (26 h) | — | $0.27 |
| S3 storage w/ G1 (15 GB, pro-rata) | — | $0.01 |
| **Total** | | **≈ $195** |
| *S3 without G1* | | *+$24/mo* |

**Abort criteria:**

- Any whole-group restart.
- Two consecutive failed replacements (capacity exhaustion, not fault tolerance).
- Goodput below 0.75 at the 6-hour mark.
- Checkpoint count growing unbounded (G1 regression).

**Operator checkpoints:** t+1h, t+6h, t+18h. The t+1h check matters most — almost
everything that will go wrong is visible by then.

### Extending to 36 h — requires A8

Decide at **hour 20**. Extend only if goodput ≥ 0.90, zero whole-group restarts,
and checkpoint count bounded.

`spot-orchestrate resume` will **not** do this: it derives
`kind = run_id.split("-", 1)[0]`, mapping `multinode-preempt-…` to `"multinode"`,
then raises `SystemExit("resume handles single-box runs")`. It is a single-box
salvage tool. A8 calls `_run_supervised` directly, which already accepts
`run_id: str | None = None`.

Extension cost: **+$97** for ~9,000 more steps, plus one fresh 8-node boot
(~4 min ≈ $1.30). Nothing bills between segments. Two caveats to state: the extra
steps run past `LR_DECAY_STEPS=20000` at **minimum LR** (a constant-LR tail), and
the dashboard shows a real wall-clock gap between segments.

**Reporting consequence:** the cost ledger lives in `profile.json`, so a
two-segment run has two ledgers. Any cost or goodput claim must **sum the
segments**.

---

## 9. Laptop detach / rejoin

**Already exists** — `spot-orchestrate orch up|status|logs|down`. The supervisor
runs on an always-on `t3.micro`; S3 is the entire control-plane API; EC2 tags are
discovery. The laptop only does GETs. `orch up` attaches to the dashboard and
**Ctrl-C only detaches**.

**Gap1 — Grafana history has laptop-shaped holes.** `deploy/grafana/live.py`
polls `status.json` and appends `status_hist.jsonl`. `status.json` is
**overwritten in place** in S3, so an hour with the lid closed is an hour of
world-size and Gantt history that is *permanently unrecoverable*. Step/loss/cost
backfill fine (node logs are append-only); world size and occupancy do not.

**Fix:** the supervisor is the writer — publish each status doc as its own small
object under `runs/<run_id>/status/<ts>.json`. `live.py` then `aws s3 sync`s that
prefix. ~13,000 tiny objects over 24 h; LIST/GET cost negligible. Do **not**
rewrite one growing `jsonl` per tick — that is ~78 GB of PUTs.

**Grafana stays entirely local.** The laptop keeps every capability it has now;
it just stops being the only thing that remembers.

**Gap2 — epoch publications** are printed by the supervisor to the driver's log
(dashboard Trap #6: world-size and Gantt render *empty while every other panel
works*). Under `orch.py` the driver is the t3.micro and its log reaches S3 as
`orchestrator.log`, so this resolves itself — but `live.py --log=` must point at
the downloaded copy. Verify in Batch 2.

---

## 10. Risks

1. **On-demand capacity for 8 × g5.xlarge.** There is an ongoing GPU shortage.
   `InsufficientInstanceCapacity` is the most likely reason a launch or a
   replacement fails. Check availability immediately before Batch 3.
2. **Dataset re-pull dominates recovery** — 118 s of 154 s. See A7: verify OWT
   `train.bin` is baked into the AMI or a pre-warmed snapshot. Highest-leverage
   improvement to the headline metric. (`bake-ami` was previously measured
   *slower* for boot generally — that was about the repo+pip clone, not a 17 GB
   dataset. Re-measure.)
3. **LR schedule mismatch** between any baseline and the main run silently
   invalidates comparison.
4. **S3 checkpoint growth** without G1 — certain, not hypothetical. Now fixed.
5. **`down` vs `provisioning`** is a known-open dashboard item
   (`docs/grafana-handoff.md`); without it you cannot distinguish "replacement
   booting" from "no capacity".

**Not yet built (operator's call):** AWS Budgets ceiling with a deny-EC2 action.
The only guardrail that survives our code being wrong in a way we didn't imagine.
~15 min in the console, $0.

---

## 11. Appendix — kill switches

Fill in and keep reachable from a phone. **Execute each one during Batch 2** —
a kill switch you have never run is not a kill switch.

```bash
# terminate everything we own in us-east-1 (region + tag scoped)
. ./scripts/fleetctl.sh && reap_ours

# what is alive right now
spot-orchestrate orch status

# tear down the control plane too
spot-orchestrate orch down
```

---

## 12. Total cost

| Phase | Cost |
|---|---|
| Batch 1 — code | $0 |
| Batch 2 — 1h rehearsal | ~$10 |
| Batch 3 — 24h run | ~$195 |
| Optional 24h→36h extension | +$97 |
| **Planned total** | **~$205** (~$302 with extension) |

Contingency: **$400**. An abort at hour 6 costs ~$50, and with A8 built it is
resumable rather than a write-off.
