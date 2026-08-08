# E1 / E2 — the first real GPU numbers

**Run:** 2026-08-08 · **1× g5.xlarge (A10G 24GB), us-east-2** · **Cost: ~$0.80**
(35 min of that was boot; the measurement itself took ~4 min. Terminated
immediately after; both regions verified empty.)
Raw: `.fleet/e1e2-results.json` · Rig: `deploy/aws/e1e2-run.py`

Served as a **bare process**, no Kubernetes, so nothing sat between the
measurement and the model. `gpt2-xl` on CUDA in bf16, HF backend, KV cache on —
3,477 MB VRAM.

---

## The four numbers everything else keys off

| | Measured |
|---|---|
| **L0** (unloaded p50, 64 tokens) | **2,097 ms** |
| **SLO** (= 3 × L0) | p99 ≤ 6,291 ms |
| **Decode rate** | **30.5 tok/s** |
| **Single-worker capacity** | **0.48 req/s** |
| **$/1M tokens** | **$7.96** |

---

## E2 found the knee is *below* the sweep's floor

The sweep started at 1 rps and immediately blew the SLO:

| offered | achieved | p50 | p99 | errors |
|---|---|---|---|---|
| 1 rps | 0.55 rps | 10,148 ms | 17,446 ms | 0 |

That is not a failure of the fleet, it is arithmetic. Service time is 2.10 s and
generation is serialized behind `_gen_lock`, so one worker tops out at
**1 / 2.10 = 0.48 req/s**. Offering 1 rps is **210 % utilisation** — the queue
grows without bound, and p50 reflects waiting, not generating. Achieved
throughput (0.55 rps) matches predicted capacity, which is the tell.

**So `C1 < 1 rps`, and the sweep never had a valid point.** The rig should start
at a fraction of `1/L0`. Fixed for the re-run, but stated here because the
number as printed (`C1 = None`) is a rig limitation, not a measured capacity.

Worth noting: **zero errors** even at 210 % utilisation. Requests queued and
were served late rather than dropped — the failure mode is latency, not loss.

---

## Two independent problems, and they compound

### 1. Per-token overhead — 4.6× off the hardware bound

gpt2-xl in bf16 is ~3.0 GB of weights. An A10G has 600 GB/s, call it 420 GB/s
effective. Memory-bandwidth-bound decode should therefore cost **7.1 ms/token
≈ 140 tok/s**.

We measured **30.5 tok/s = 33 ms/token** — so roughly **26 ms/token is
overhead**, not compute. The GPU is idle most of every token.

Prime suspects, in order:
- `transformers` `generate()` Python overhead per step (logits processors,
  stopping criteria, tensor bookkeeping) — dominant at batch 1
- **`attn_implementation` may be defaulting to eager rather than SDPA** for
  GPT-2 on this version; worth pinning explicitly
- No CUDA graphs (`torch.compile` + `StaticCache` is exactly the fix for
  launch-overhead-bound decode)

This is a *latency* problem and batching will not fix it.

### 2. No batching — the throughput/cost problem

Decode is memory-bandwidth bound: reading the weights costs the same whether
you decode for 1 sequence or 32. Serving one request at a time pays that cost
once per token per request.

**$7.96/1M tokens against a ~$0.20 market rate is 40× off.** That is the number
that makes the batching case unarguable, and it is now measured rather than
predicted.

---

## What this changes

The plan predicted "~$2.00/1M unbatched". The truth is **$7.96 — 4× worse than
predicted**, because the plan assumed the 140 tok/s bandwidth bound and reality
delivered 30.5. Both fixes are now justified on measurement rather than
principle:

| Fix | Attacks | Expected |
|---|---|---|
| `attn_implementation="sdpa"`, `torch.compile` + `StaticCache` | the 26 ms/token overhead | latency ↓, toward 140 tok/s |
| Continuous batching | the 40× cost gap | throughput ↑ ~linearly in batch size |

Ordering matters: **fix per-token overhead first.** Batching multiplies
throughput but does nothing for the fixed cost of each decode step, so batching
a 33 ms/token engine locks the overhead in as a floor.

---

## What this does NOT establish

1. **No batched number yet.** The `$0.19/1M` figure in the plan remains a
   *prediction*. E10 needs an actual batcher in the worker — measuring it
   requires code on the box, not more load from outside.
2. **`C1` is unmeasured.** The sweep floor was above capacity. Re-run needs
   points at 0.1–0.5 rps.
3. **One instance, one run.** No variance estimate; `L0` is a p50 of 12 samples
   after a warm-up.
4. **Nothing about multi-worker scaling** — that is E5, and it is cheaper on CPU
   workers.
5. **The 26 ms/token attribution is a hypothesis.** It is the residual after
   subtracting the bandwidth bound; it has not been profiled. Do not quote a
   cause until a profile confirms it.

## Rig bugs found (fixed)

- Sweep floor above capacity (above).
- `gpu-up.sh` opened port 8001 to the SG itself but **not to the operator's IP**,
  and swallowed the authorize error behind `2>&1` — the worker was healthy and
  serving locally for ~5 minutes while appearing dead from outside.
