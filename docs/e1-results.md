# E1 — making checkpointing cheap (async write → batched D2H → shared blob)

Three runs, 4 nodes, 300s training budget, identical recipe. **~$1.50 total.**

**Verdict: substantial, verified improvement (+21% steps, −54% stall per step,
−45% worst step) — but it still FAILS the ≥90-step gate, and my predictions were
wrong twice. Recommend instrumenting rather than guessing a third time.**

## Results

| | control | E1 | **E1b** |
|---|---|---|---|
| run_id | `multinode-1786117954` | `multinode-1786164581` | `multinode-1786166169` |
| change | sync write, gp3 root | async write, NVMe | **+ batched D2H, shared blob** |
| **steps in 300 training-s** | 67 | 78 (+16%) | **81 (+21%)** |
| stall total | 98.3s | 65.1s | **54.9s** |
| stall per step | 1.47s | 0.83s | **0.68s (−54%)** |
| stall as % of training | 32.6% | 21.5% | **18.2%** |
| worst step | 29.6s | 26.0s | **16.3s (−45%)** |
| `final_saves_s` | 108.2s | 81.3s | **74.3s (−31%)** |
| `trained_seconds_total` | 301.71 | 302.03 | 300.99 |
| val_loss | 6.527 @67 | 6.3675 @78 | **6.3155 @81** |

**Gate `≥90 steps`: 81 → FAIL.**

![control vs E1 vs E1b](img/e1-comparison.png)

- **Left:** all three loss curves superimpose exactly. That is the correctness
  result — neither async writes, batched device-to-host copies, nor a blob shared
  between two writer threads perturbed training. Each fix simply travels further
  along the same trajectory within the same 300 training-seconds.
- **Right (log scale):** checkpoint stalls shrink from ~29.6s to ~16.3s, but they
  are still there and still an order of magnitude above the 3.04s steady state.

## W&B

- control — https://wandb.ai/bot-developer3-Miguel%20Aenlle/spot-train/runs/k34qpxws
- E1 — https://wandb.ai/bot-developer3-Miguel%20Aenlle/spot-train/runs/zxk2nzeo
- **E1b** — https://wandb.ai/bot-developer3-Miguel%20Aenlle/spot-train/runs/28vbdt57
- project — https://wandb.ai/bot-developer3-Miguel%20Aenlle/spot-train

## What was changed

**E1 — async write + NVMe.** The node-local tier did serialize+write inline, on
the 30 GB gp3 root at ~125 MB/s (~12s for 1.5 GB) while the instance-store NVMe
sat idle. Mirrored the S3 tier's existing `AsyncSaver` and moved the tier to the
NVMe. Result: stall/step −43%.

**E1b — batched D2H + shared blob.**

1. `_batched_d2h`: one **pinned** host buffer, every tensor copied
   `non_blocking` into slices of it, then **one** `torch.cuda.synchronize()`.
   Replaces ~450 individually-synchronising `.to("cpu")` calls (GPT-2 124M has
   ~150 parameters × weights + two Adam moments each), every one of which waited
   on a GPU busy with queued training kernels.
2. `AsyncCheckpointer.submit(blob=...)`: rank 0 snapshots **once** and hands the
   same dict to the S3 writer and the local writer. The two snapshots were
   byte-identical by construction — same model, optimizer, step, with a
   collective between them guaranteeing training had not advanced.

Both are guarded by tests pinning the invariant the design rests on: **a snapshot
must not alias live tensors.** Mutating a weight after snapshotting must not
change the blob; otherwise async writes would serialize state changing underneath
them and produce torn checkpoints.

## Why it still misses — and why I am not going to guess again

Predicted ≥90 from (1) alone and ~100 from (1)+(2). Got 81. **Two wrong
predictions in a row means the model of where the time goes is wrong**, and
54.9s of stall over 81 steps is still unexplained.

The leading suspect is something E1b *introduced*: `_batched_d2h` allocates a
fresh **1.5 GB pinned buffer on every checkpoint**. Page-locking host memory is
far more expensive than an ordinary allocation — plausibly seconds — so part of
the win may be self-inflicted. Reusing one persistent buffer would fix it, but
the blob is handed to background writers that may still be reading it, so reuse
must be coordinated with `flush()`.

That is a hypothesis, not a finding. **The right next step is to instrument
`snapshot()`** — time the pinned allocation, the copies, the sync, and the
handoff separately, and log them. One ~$0.75 run then answers it with evidence
instead of a third guess.

## Options from here

1. **Instrument `snapshot()`, then fix what the data shows** (~$0.75). Highest
   confidence, and it stops the guessing.
2. **Bank +21% and move to Fix 2 (idle survivors).** That fix is independent,
   has a measured target (~642 node-seconds wasted per failure), and a confirmed
   root cause. Checkpointing is already 54% cheaper per step than where it started.
3. Reuse the pinned buffer on the hypothesis above (~$0.75) — cheaper than
   instrumenting but risks a third miss.

**Recommendation: (2), with (1) queued behind it.** The remaining checkpoint
stall is 18% of training time; the idle-survivor bug wastes 214 seconds per
failure with a root cause already confirmed from the event stream. Better value
per unit of effort, and E1's gains are already banked.

## Validity

- `trained_seconds_total` 301.7 / 302.0 / 301.0 — all ~300s, A/B valid across all three
- identical recipe, node count, and 30s checkpoint interval in every arm
- loss curves superimpose → no divergence from any of the changes
- 412 tests green; fleet terminated after each run, 0 instances billing
