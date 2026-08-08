# Backlog

Things worth doing, not scheduled. Each entry states why it matters and what it
would cost, so it can be picked up cold.

---

## Fault injection beyond instance loss (~$2.30, two runs)

**Every failure we have injected is `TerminateInstances`.** That is faithful to an
unannounced spot reclaim, and the recovery path is genuinely blind to our
foreknowledge — a scheduled kill only terminates the box; membership changes come
solely from `_healthy()` observing the loss. But it exercises exactly one of the
three failure classes the system already claims to handle:

| failure | detected by | injected? |
|---|---|---|
| box dies | `aws_state` dead, or stale log | ✅ repeatedly |
| **process crashes, box alive** | sidecar sees nonzero exit → relaunch, capped by `MAX_EPOCH_CRASHES` | ❌ never deliberately |
| **box alive, process wedged** | log stale past 90s heartbeat → treated as lost | ❌ never deliberately |

These are not the same failure. **An instance death frees the GPU and the vCPU
quota; a hung process holds both.** A wedged node is the nastier case: EC2 still
reports it healthy, it keeps its slot, and only the heartbeat timeout unmasks it.

We have hit this class accidentally and paid for it — the 8-node NCCL-init crash
loop consumed 22 instances before a human noticed, which is why the crash-loop
cap exists. It has never been tested on purpose.

**Two experiments, same harness as the failure-cost A/B — only the kill *action*
changes, from `TerminateInstances` to an SSM command on the box:**

1. **`kill -9` the trainer, leave the box up.** Does the sidecar relaunch it and
   rejoin the world cleanly? Does the crash-loop cap stay untriggered for a
   single crash?
2. **`SIGSTOP` the trainer.** Does the stale-heartbeat path evict it after 90s
   and replace it, rather than waiting forever on a node EC2 calls healthy?

**Why it is worth doing:** "we survive node loss, process crash, and hang" is a
materially stronger claim than node loss alone, and the machinery for all three
already exists — this only proves it. Cheap: ~$1.15 per run at 4 nodes, reusing
`docs/failure-cost-runbook.md` unchanged apart from the injection step.

---

## Instrument `snapshot()` (~$0.75)

E1/E1b cut checkpoint stall per step 1.47s → 0.68s (−54%) but missed the ≥90-step
gate at 81, and **two predictions in a row were wrong** (≥90, then ~100). 54.9s of
stall over 81 steps is still unexplained, so the model of where the time goes is
wrong.

Leading suspect is self-inflicted: `_batched_d2h` allocates a fresh **1.5 GB
pinned buffer every checkpoint**, and page-locking host memory is far costlier
than ordinary allocation. Reusing one persistent buffer would fix it, but the
blob is handed to background writers still reading it, so reuse must coordinate
with `flush()`.

**Do not guess a third time.** Time the pinned allocation, the copies, the sync,
and the handoff separately, log them, and let one run answer it with evidence.

---

## Streaming data loader (large)

Deferred deliberately — see `docs/perf-fixes-plan.md`. −17% on a 300s run but
**0.09% on a 36h run**, for 600–800 lines on the correctness-critical path plus a
sampling-distribution change that needs a side-by-side loss check. Becomes worth
it when the corpus stops fitting on local disk, not before.

Note the interaction: fixing the idle-survivor bug removes most of streaming's
remaining value, since survivors then train through a replacement's boot and the
116s download no longer blocks progress.

---

## ~~Checkpoint-on-membership-change~~ — NOT NEEDED (E2b)

Closed. E2b widened the degraded window from 24s to 173s, which the normal 30s
checkpoint interval crosses several times, so reduced-world work banks on its
own: the regrow resumed from step 68 rather than the pre-kill step 39. Re-open
only if a future degraded window is shorter than the checkpoint interval.
