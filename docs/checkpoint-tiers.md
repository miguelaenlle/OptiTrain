# How checkpointing works (two tiers, and who writes what)

## Two tiers, different jobs

| tier | who writes it | where | purpose |
|---|---|---|---|
| **S3** | **rank 0 only** (the master) | `s3://<bucket>/runs/<id>/checkpoints/` | **durable.** The run's real checkpoint. Survives losing every box. |
| **node-local** | **every node's `local_rank 0`** | `LOCAL_CHECKPOINT_DIR` on that box | **fast restart in place.** A survivor of a membership change reloads from its own disk instead of pulling 1.5 GB back from S3. |

On a 4-node, 1-GPU-per-node run: **all four nodes** write a local checkpoint;
**only node 0** also writes to S3. So node 0 does both jobs.

The local tier is deliberately *not* durable — if the box dies its local
checkpoint dies with it, and S3 is what the replacement restores from. That is
why it can live on the ephemeral instance-store NVMe.

Node-local checkpoints are step-aligned on rank 0's decision (a `broadcast_flag`
collective), so every node snapshots the *same step*. That is what lets a
shrunken group later agree on a common resume step.

## What a "blob" is

`checkpoint.snapshot()` returns one dict — the blob:

```python
{"version", "step", "trained_seconds",
 "model":     <every weight, copied to CPU>,
 "optimizer": <Adam exp_avg + exp_avg_sq, copied to CPU>,
 "rng":       <all RNG states>,
 "loader":    <data position>,
 "scaler":    <AMP scaler state>}
```

~1.5 GB for GPT-2 124M. This dict is what gets serialized into a `.pt` file.

**The CPU copy is the whole trick.** Once the blob exists in host RAM, training
can carry on mutating the GPU tensors while a background thread serializes it.
Without the copy, an async write would be serializing state that is changing
underneath it — a torn checkpoint. This is why `snapshot()` must stay on the
critical path even when the write does not.

## What Fix 1 changed

Each checkpoint has two stages. Only the second moved.

```
stage 1  snapshot()      GPU -> CPU copy      ~1.5 GB   still SYNCHRONOUS
stage 2  serialize+write CPU -> disk/S3       ~1.5 GB   now on a thread
```

Before, the **local** tier did both stages inline: a full `torch.save` on the
training critical path, onto the 30 GB gp3 root at ~125 MB/s (**~12s**). The S3
tier had already been split this way; the local tier had simply never been given
the same treatment.

Fix 1 = mirror that split (`AsyncLocalSaver`) **and** move the tier to the
instance-store NVMe, where the same write is ~1s.

Measured: stall per step **1.47s → 0.83s (−43%)**, `final_saves_s` **108s → 81s**.

## Why "hand the same blob to both tiers" is the next fix

Node 0 currently calls `snapshot()` **twice per checkpoint**:

```python
async_ckpt.submit(model=raw_model, ...)      # snapshot() runs INSIDE this
blob = checkpoint.snapshot(model=raw_model, ...)   # a SECOND, identical copy
async_local.submit(blob, step)
```

Two full 1.5 GB GPU→CPU copies of **identical state** — same model, same
optimizer, same step, microseconds apart, with a collective in between
guaranteeing training has not advanced.

The fix is to snapshot once and pass that one dict to both writers. Nodes 1..N-1
are unaffected (they only ever snapshot once, for their local tier); this is
purely node 0's double cost.

### Why the copy is slow at all

Raw PCIe Gen4 ×16 moves 1.5 GB in **~250 ms**. But `_cpu_copy` walks the state
tree calling `.to("cpu", copy=True)` per tensor — **~450 of them** for GPT-2 124M
(≈150 parameters × weights + two Adam moments). Each is a synchronising call
against a GPU that is busy with queued training kernels, so the cost is dominated
by ~450 stalls rather than by bandwidth.

Hence the two follow-ups, in order of effort:

1. **snapshot once, share the blob** — halves node 0's copy cost; small change.
2. **batch the copies through a pinned buffer** — removes the per-tensor syncs
   entirely, taking the copy toward its ~250 ms floor.
