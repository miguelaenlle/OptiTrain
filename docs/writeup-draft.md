# Training and serving LLMs on hardware that disappears

> **Draft.** `[TBD]` = comes from the 36h/8-node run and the Go/K8s fleet stress
> test, not yet run. Everything else is measured and traceable to a run id.
> Figures are the real 4-node plots, standing in for their 36h equivalents.

---

*Miguel Aenlle · `[repo]` · `[contact]`*

Spot GPUs cost 60–90% less than on-demand and can be reclaimed with no warning. I
built both halves of an LLM stack on them — a GPT-2 pretraining cluster and an
inference fleet — on one control pattern: **membership is observed, compared to
intent, and acted on**, with S3 as the only transport. Two implementations, same
failure injector, same cost ledger.

**The result that matters: recovery cost does not grow with the number of
failures.** Six kills over three rounds banked identical work each round, and the
Nth restart was no slower than the first.

## Results

| | |
|---|---|
| **Training** | GPT-2 124M · OpenWebText (9.04B tokens) · 8 nodes `[TBD instance type]` · 36h → val loss `[TBD]` |
| **Failures injected** | `[TBD]` kills across 3 classes: node loss, process crash, hang |
| **Restart latency** | 7.8–9.9s at 4 nodes; `[TBD]` p50/p95 at 8 |
| **Work lost per failure** | 6–9 steps (~30s), bounded by the checkpoint interval |
| **Goodput** (train-seconds ÷ wall-clock) | `[TBD]` |
| **Cost to the same val loss** | `[TBD]` spot vs `[TBD]` on-demand |
| **Serving** | `[TBD]` RPS · `[TBD]`% client-visible errors through worker kills · p99 recovers in `[TBD]`s |

## The system

One orchestrator owns membership by publishing monotonic **epoch documents** to
S3; every box runs a sidecar that obeys them by running *static* `torchrun` per
epoch. No node hosts a rendezvous store, so **any node is killable, including the
master**. Survivors keep training at world N−1 while a replacement boots, and
gradient accumulation holds the global batch at 480, so a smaller world does more
micro-batches rather than bending the loss trajectory. Checkpoints are two-tier —
node-local NVMe for instant survivor restores, async rank-0 S3 for replacements —
and carry the remaining time budget, so downtime is never billed. The decision
core is a pure reducer, `decide(Observation, Policy) → [Action]`: table-tested and
replayable without touching AWS.

This replaced a `torchrun`-elastic design that passed every local test on torch
2.4 and hung >180s on the DLAMI's torch ≥2.8. A version-dependent black box is not
debuggable at 3am on a machine that is about to vanish.

## Durable progress through repeated failure

![Rolling pair failures](img/e4-rolling-pairs.png)

*`multinode-preempt-1786199662`. Ten rows — four originals, six replacements. Half
the world destroyed three times: round 2 kills the master, round 3 kills round 1's
replacements. Survivor rows stay green (training) through every dip; the leader
star moves twice, unprompted.*

![Durable progress](img/e4-progress.png)

*Furthest checkpointed step vs wall clock. The curve never goes backwards and
climbs inside every shaded window — half the fleet dead costs a third to a half of
the step rate, not all of it.*

| round | restart latency | reduced-world window | **steps banked** |
|---|---|---|---|
| 1 | 9.9s | 201.0s | **21** |
| 2 — the master dies | 7.9s | 168.5s | **21** |
| 3 — kills round 1's replacements | 7.8s | 184.3s | **21** |

Every prior measurement was a *single* failure, so none showed whether the Nth
recovery costs more than the first. It does not. Instances consumed: **10 = 4 +
exactly one per kill** — no churn, no crash loops, zero whole-group restarts, and
`trained_seconds_total` of 901.74 against a 900s budget across six kills.

Two fixes got there. Making registration mean *"I hold the corpus and can train"*
cut replacement join time from ~150s to ~50s. Rejecting a dead node's durable S3
registration stopped the supervisor regrowing the world onto a corpse — it had
been doing so 151s before the replacement even began booting. Together, a single
node failure went from **981s to 679s of wall clock (−31%) and −29% in dollars**,
with survivor idle time 214s → 36s (`…1786169550` vs `…1786118730`).

Every experiment above was a controlled A/B against a fixed training-second
budget, and each one cost **$0.79–$1.84** to run.

## Serving on the same substrate

**`[TBD — the fleet stress test.]`** A Go router tracks workers by S3 heartbeat
and reroutes on connection error, 5xx, or timeout; a Go load generator holds fixed
RPS while workers are hard-killed mid-test. The claim to land: **client-visible
error rate stays at ~zero because retries absorb the kill, and p99 returns to
baseline within one heartbeat TTL** — reported through the same cost ledger as the
trainer, in $/1M tokens.

## Capacity, not price, is the binding constraint

Over 23.2h of spot placement scores (43,074 records), only **4 of 25 (AZ, GPU)
pools in us-east-1** cleared AWS's recommended score of 7 for **8 instances in one
AZ**, and the longest unbroken good window was **50 minutes**. Pinning a specific
type was worse: `g4dn.xlarge` never cleared the bar, in any AZ, in any sample —
and single-instance scores run systematically optimistic relative to what a
training world needs. You cannot wait for an 8-node window. You take what exists
and survive losing it.

## What broke

- **The elastic rendezvous.** Green locally on torch 2.4, hung >180s on torch ≥2.8.
- **A fix that was a no-op.** Clearing the dead node's cached IP only forced a
  re-read of the same durable S3 doc. Found in the event stream, not by reading code.
- **22 instances burned** by an 8-node NCCL-init crash loop before a human
  noticed. That is why the crash-loop cap exists.
- **Two wrong performance predictions in a row** on checkpoint cost — ≥90 steps
  then ~100; got 78, then 81. The right response was to stop guessing and instrument.

## Limits

The results above are 300–900s runs at 4 nodes; the 36h/8-node figures generalise
them. Checkpoint stall is still ~18% of training time and not fully explained.
Each box downloads a monolithic 17GB corpus instead of streaming shards, costing
~2.5 min per cold start and per replacement. No bit-exactness claim — the
invariant is that loss continues from the checkpoint.

`[repo link]` · per-experiment writeups: `docs/e1-results.md` … `e4-results.md` ·
`[W&B project link]`

---

<!-- ==================== NOT PART OF THE PAGE ==================== -->

## Notes for the final cut

**Open placeholders, in priority order**

1. **The on-demand denominator.** Every headline claim is "spot beats on-demand
   for the same loss," and nothing in `reports/` closes it — the latest
   `scaling-clean` summary reads `ms/step: None` throughout. Needs a reference arm
   at the same world size and recipe. Cannot be retrofitted after the 36h run.
2. **MFU.** Not currently logged. This audience will ask; a stated MFU is the
   difference between knowing your denominator and not.
3. **Serving section** — entirely `[TBD]` until the Go/K8s fleet runs.
4. **Goodput.** From E4's own table, 901.74 / 1667.8 = **54%**, but that is
   boot-dominated at a 900s budget. Report the 36h figure, definition inline.
5. **Instance type for the 8-node run.** Spotwatch says `g5.xlarge` and
   `g4dn.xlarge` never cleared the placement-score bar; an "any GPU" pool did.
   Decide (and say) whether the run is type-flexible.

**Pre-flight instrumentation** — before the 36h run, not after

- **Timestamp the step log line.** Every plot in `e4-results.md` is a
  reconstruction because it has none, and the first attempt was wrong for exactly
  that reason. At 36h this gets much harder.
- **Log tokens/step and MFU** so $/1M tokens is measured, not derived.
- **Instrument `snapshot()`** (~$0.75, already in the backlog). Otherwise the
  headline run bakes in an unexplained 18% tax you cannot report honestly.
- **Widen the failure taxonomy** to `kill -9` (crash, box alive) and `SIGSTOP`
  (wedged, EC2 still says healthy). Upgrades the claim from "survives node loss"
  to "survives node loss, process crash, and hang" — three classes, one recovery
  path. The machinery exists; this only proves it.
- **Record restart latency as a distribution.** 36h gives enough kills for p50/p95,
  and p95 is what a reviewer actually wants.

**Structural**

- Serve the training run's own checkpoints from the inference fleet, driven by the
  same kill scheduler. One page, one story: the fleet serves a model the spot
  cluster is still training, and both survive the same kills.
- **The final page gets ONE figure** — the 36h staircase with `ckpt_step` overlaid
  and kills annotated. The two E4 plots here are placeholders for it; merge them.
  E1/E2/E2b plots stay in their own docs.
- Cut roadmap/future work to one clause; cut the RL track entirely.
- Current length: **786 words of prose + 2 tables + 2 figures.** That is one dense
  page. Merging the two figures into one buys back the room the 36h numbers will
  need; if it still overruns, the Capacity section compresses to two sentences.
