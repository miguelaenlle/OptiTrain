# Local validation plan — gpt2-xl serving + Go/Python router A/B

**Status:** planned, not yet run.
**Cost:** $0. Entirely local. No AWS, no credentials, no cloud calls.
**Purpose:** de-risk the GPU run. Every bug found here costs $0; the same bug
found on a g5.xlarge costs $1/hr plus the time to notice it.

---

## Machine constraints (measured, 2026-08-08)

| Resource | Value | Consequence |
|---|---|---|
| RAM | **17.2 GB total, 0.9 GB free+inactive** | `gpt2-xl` fp32 ≈ **6.2 GB per worker process**. Two gpt2-xl workers is impossible; one is tight. |
| Disk | 28 GB free | `gpt2-xl` download ≈ 6 GB. Fits. |
| HF cache | 8.8 GB, has `gpt2` but **not** `gpt2-xl` | First load pays a ~6 GB download. |
| GPU | none (CPU only) | ~5–10 tok/s for a 1.5B model. Functional testing only, never performance. |

---

## The central design decision

**We must NOT use `gpt2-xl` for the router A/B.** It would actively hide the
thing we are trying to measure.

A router adds on the order of **1–3 ms** per request. On CPU, a `gpt2-xl`
completion takes **~10–20 s**. The router would be ~0.02 % of the measurement —
buried under model noise and run-to-run variance. We would "measure" the two
routers as identical no matter what they did, including if one were badly
broken. That is a test that cannot fail, which is a test worth nothing.

So the work splits into three experiments with different jobs:

| # | Question | Upstream | Why that upstream |
|---|---|---|---|
| **X1** | Does `gpt2-xl` load and serve at all? | 1 × gpt2-xl worker | The point *is* the big model. De-risks the GPU run. |
| **X2** | Are the routers functionally equivalent? | 2 × `gpt2` (124M) workers | Need **two** workers to exercise round-robin/reroute; small model fits in RAM and is fast enough to iterate. |
| **X3** | What does each router *cost* per request? | N × stub workers | Only way to make router overhead visible. This is the real answer to "is the Go one better." |

X3 is the one that produces a defensible number. X2 proves correctness. X1
proves the model path.

---

## X1 — `gpt2-xl` loads and serves

**Question:** does `PRETRAINED_MODEL=gpt2-xl` work end to end — download, build,
BPE codec, generate, OpenAI-shaped response?

### Setup
- One worker, no router, no registry (`FLEET_WORKERS_URI=""` → heartbeat off).
- `PRETRAINED_MODEL=gpt2-xl`, `DEVICE=cpu`.
- `max_tokens=16` — we are not measuring speed, only correctness. 64 tokens at
  CPU speed would take minutes per request for no added signal.

### Procedure
1. Start worker on :8001. Time the cold load (download + build + `.to()`).
2. `GET /healthz` → expect `model: "gpt2-xl"`.
3. `GET /v1/models` → expect the same id.
4. `POST /v1/completions`, prompt `"The capital of France is"`, `max_tokens=16`.
5. `GET /stats` → confirm counters moved.
6. Record RSS of the worker process at steady state.

### Pass criteria
- Loads without OOM.
- Completion is **coherent English** (not repeated tokens or gibberish) — this
  is what proves the weights were loaded and transposed correctly, and that we
  are using the GPT-2 BPE rather than a char codec.
- `prompt_tokens == 5` for that prompt (correct BPE tokenization).
- Response matches the OpenAI shape the Python worker already emits.
- RSS ≈ 6–7 GB.

### Risks / fallbacks
| Risk | Fallback |
|---|---|
| OOM at fp32 (only 0.9 GB free) | `SERVE_DTYPE=bfloat16` → ~3.1 GB. **Note:** CPU bf16 in PyTorch is slow and some ops may be unimplemented; if it errors, that is a *fallback* failure, not a gpt2-xl failure — report it as such. |
| Download is slow / rate-limited | Pre-fetch weights once, in the background, before the run. |
| Machine thrashes | Accept it; this is a functional test. Do not report any timing from X1 as performance data. |

### Explicitly NOT concluded from X1
Nothing about latency, throughput, or tokens/s. CPU numbers for a 1.5B model
say nothing about an A10G. **L0 comes from E1 on the GPU, not from here.**

---

## X2 — router functional equivalence

**Question:** is `router-go` a true drop-in for `src/inference/router.py`?

### Setup
- **Two** `gpt2` (124M) workers on :8001 and :8002, sharing a local-dir registry
  (`.fleet/local/workers`). ~0.5 GB each — fits comfortably.
- Workers stay **running and untouched** for the whole experiment. Only the
  router is swapped. This is the control: identical upstreams, identical load.
- Each router in turn on :8000, reading the same registry dir.

### Procedure
For **each** router (Python, then Go):
1. Start it; wait until `/fleet/status` reports `live_workers == 2`.
2. **Contract checks:** `/healthz`, `/fleet/status`, `/fleet/metrics`, `/metrics`
   — capture each response body.
3. **Round-robin:** 10 sequential completions; collect `worker_id` from each.
4. **Reroute:** `kill -9` worker A mid-load; continue sending; then restore.
5. **4xx passthrough:** send a request that a worker rejects (e.g. `max_tokens=0`
   → 422 from the worker's own validation); confirm it is **not** retried.
6. **No workers:** stop both workers; confirm 503.
7. Stop the router.

Then diff the two routers' captured outputs.

### Pass criteria
| Check | Expectation |
|---|---|
| Response shape | Byte-identical JSON structure; same fields |
| Round-robin | Both routers spread across both workers (neither pins to one) |
| Reroute on kill | **Zero client-visible errors**; `rerouted > 0` on `/fleet/status` |
| Dead worker dropped | Within the 15 s TTL |
| 4xx | Passed through unretried by **both** — attempt count 1 |
| No workers | 503 from both |
| `/metrics` | Same metric names, labels, and buckets (cosmetic deltas OK: Python's `*_created` series, `le="1.0"` vs `le="1"`) |

### Known divergences to verify deliberately
The Go router has two intentional differences. X2 must confirm they behave as
designed rather than silently regress:
- **`ROUTER_UPSTREAM_KEEPALIVE=false` (default).** Step 4's `kill -9` is exactly
  the case this exists for. Expect bounded reroute (~connect timeout), not a
  60 s hang.
- **Client disconnect stops the retry loop.** Send a request and abort the
  client mid-flight; confirm the Go router does not keep fanning it across
  workers.

### Risks
| Risk | Mitigation |
|---|---|
| Go router's registry env var name may differ | **Confirm from `router-go/README.md` before running.** Do not guess. |
| Port collision from a previous run | Check :8000–:8002 are free; kill strays first |
| Registry dir has stale docs | Clear `.fleet/local/workers` before starting |

---

## X3 — router overhead (the measurable one)

**Question:** what does each router actually cost per request, and where does
each one saturate?

### Setup
- **Stub workers**: a trivial HTTP server returning a valid, fixed,
  OpenAI-shaped completion immediately, and writing a heartbeat doc so the
  router discovers it normally. No model, no torch.
  → new artifact: `bench/stub_worker.py`
- With upstream latency ≈ 0, essentially **all** measured latency is router
  overhead. That is the entire point.
- Drive with the existing Go `loadgen`.

### Procedure
1. Start 4 stub workers.
2. For each router: RPS sweep (e.g. 50, 100, 200, 400, 800, 1600) at
   `max_tokens` irrelevant (stub ignores it), 30 s per point.
3. Record p50/p95/p99 and achieved-vs-offered RPS at each point.
4. Plot both routers on one chart via `bench/plots.py`.

### Pass criteria
- Both routers correct at every RPS (error rate 0 below saturation).
- Report, as **measured numbers, not predictions**: per-request overhead (p50 at
  low RPS) and the knee (highest RPS holding the SLO) for each.
- Expectation to be tested, not assumed: Go's knee ≥ Python's. If it is not,
  that is a finding worth reporting honestly, not tuning away.

### Why this matters
This is the **router-knee result** that the 8-GPU-vCPU cap would otherwise have
cost us (old E6 needed far more GPU nodes). It runs locally, for $0, and it
isolates the variable properly.

### Risks
| Risk | Mitigation |
|---|---|
| **loadgen is the bottleneck, not the router** | Both are Go on one box, competing for cores. Watch for loadgen `dropped > 0` and CPU saturation — if the client is the limit, say so and cap the claim rather than reporting a fake knee. |
| Single-box contention (router + stubs + loadgen) | Report as "overhead on a shared box"; treat the *comparison* as the result, not the absolute knee |
| Localhost ≠ network | State it. Real reroute latency is dominated by network + connect timeout, not loopback |

---

## Order and time

| Step | Est. |
|---|---|
| Pre-fetch `gpt2-xl` (background) | 5–15 min |
| Confirm Go router env var names from its README | 2 min |
| X2 (fast, small model — do this **first**, it's the highest-signal) | 15 min |
| X3 (stubs + sweep + chart) | 20 min |
| X1 (slow, memory-hungry — do **last**, it can't break the others) | 10–30 min |

**X2 before X1** deliberately: X2 is the highest-information test and needs
little memory. X1 is slow, thrashes RAM, and its failure modes are independent.
Running X1 first would just delay the useful result.

---

## Cleanup (every run)

- Kill all worker/router/stub processes; verify :8000–:8010 are free.
- Clear `.fleet/local/workers`.
- Leave the HF cache (re-downloading 6 GB is wasteful).

---

## What this plan does NOT establish

To be restated in the writeup so no number is over-claimed:

1. **No performance claim about gpt2-xl.** CPU timings are not GPU timings.
   `L0`, `C1`, tokens/s, and $/1M tokens all come from **E1/E2 on a g5.xlarge**.
2. **No scaling claim.** Local workers on one box share CPU; that is not
   horizontal scaling. `E5` on real instances answers that.
3. **No absolute router capacity.** X3's knee is on a contended laptop. The
   *ratio* between the two routers is the durable result; the absolute number
   is not.
4. **Nothing about K8s.** These run as plain processes. Pod-kill behavior is
   T4 on k3d.
