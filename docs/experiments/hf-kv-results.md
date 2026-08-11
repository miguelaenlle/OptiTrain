# Results — HF backend with KV caching (Part 1 + P2.1–P2.4)

**Run:** 2026-08-08 · **Cost: $0** · **AWS resources used: none** — no credentials
loaded, no API calls, in any region.
**Plan:** [hf-kv-k3s-plan.md](./hf-kv-k3s-plan.md)
**Machine:** macOS, 8 cores, CPU only, `OMP_NUM_THREADS=4`. Model: stock `gpt2`
(124M), fp32.

---

## Headline

| | Result |
|---|---|
| **P2.1 equivalence** | **Token-for-token identical** across all 3 prompts × 32 tokens. The HF backend is *equivalent*, not merely similar. |
| **P2.2 KV cache** | **2.3× → 3.5× faster**, and the gap **widens with output length** exactly as theory predicts. |
| **P2.3 memory** | Peak RSS **0.79 GB vs 1.24 GB** — and that is *without* the `accelerate` fast path. |
| **P2.4 regression** | **129 tests pass**, ruff clean. Worker API unchanged. |

---

## What changed

A backend seam so `ModelService` doesn't know which engine it's talking to:

```
Backend.generate(ids, *, max_new_tokens, temperature, top_k, seed) -> completion ids

  NanoGPTBackend   GPT.generate()                        — trained checkpoints
  HFBackend        GPT2LMHeadModel.generate(use_cache=True) — stock GPT-2
```

`ServeSettings.serve_engine` (`SERVE_ENGINE`, default `hf`) picks the engine for
the **pretrained path only**. Checkpoints always use nanoGPT — they must remain
byte-for-byte the training artifact. The lock, `ServiceStats`, and the OpenAI
response shape are untouched.

Files: `src/inference/backends.py` (new), `src/inference/service.py`,
`tests/test_backends.py` (new).

---

## P2.1 — Backend equivalence ✅

The load-bearing test. Same weights, **greedy** decode, both engines:

| Prompt | Result |
|---|---|
| `"The capital of France is"` | **MATCH** — `" the capital of the French Republic, and the capital of the Fren"` |
| `"In a shocking finding, scientists"` | **MATCH** |
| `"def bfs(graph, start):"` | **MATCH** |

Greedy is achieved with `top_k=1`, which leaves a single unmasked logit — softmax
is one-hot and the multinomial draw is deterministic on **both** engines. That is
what makes them comparable at all; with sampling, exact comparison is impossible.

**This is what makes every downstream number trustworthy:** the KV cache changed
the *cost* of generation, not its *result*.

Also verified: tiktoken's ids == HF's tokenizer ids (we encode with tiktoken but
generate with an HF model — a silent divergence would feed the model a different
vocabulary).

---

## P2.2 — KV cache speedup ✅

Best of 3 runs each, same prompt, same machine:

| tokens | HF (cached) | nanoGPT (recompute) | speedup |
|---|---|---|---|
| 16 | 0.190 s · 84.2 tok/s | 0.472 s · 33.9 tok/s | **2.48×** |
| 32 | 0.369 s · 86.7 tok/s | 0.859 s · 37.2 tok/s | **2.33×** |
| 64 | 0.774 s · 82.6 tok/s | 1.840 s · 34.8 tok/s | **2.38×** |
| 128 | 1.811 s · 70.7 tok/s | 5.174 s · 24.7 tok/s | **2.86×** |
| 256 | 4.053 s · 63.2 tok/s | 14.120 s · 18.1 tok/s | **3.48×** |

Note nanoGPT's tok/s **collapses** as length grows (33.9 → 18.1) while HF's holds
far better (84.2 → 63.2). That divergence is the quadratic recompute cost
appearing in wall clock.

### Why 2–3.5× and not the 34× I predicted

The 34× figure is a **token-position (FLOP) ratio**, and wall clock does not
follow it, for three reasons worth stating plainly:

1. **A cache reduces work *per step*, not the *number* of steps.** Both engines
   still run one forward pass per generated token.
2. **Small matmuls are latency-bound, not throughput-bound.** A
   `(133×768)@(768×2304)` matmul is nowhere near 133× slower than
   `(1×768)@(768×2304)` on a CPU — the big one uses cache and vector units far
   better per element.
3. **Fixed per-step costs don't shrink at all** — Python dispatch, layernorms,
   sampling.

So 34× was an upper bound on the *arithmetic*, never a prediction of wall clock.
The honest claim is the measured one: **2.3–3.5× on CPU, growing with length.**

**The GPU number must be measured, not extrapolated** (E1/E2). It could plausibly
be larger — gpt2-xl is 12× the parameters, so the redundant compute is far more
significant — but CPU and GPU sit in different bottleneck regimes and I will not
predict across them.

---

## P2.3 — Memory ✅

Peak RSS, each engine in its **own process** so the peaks can't mix:

| engine | peak RSS (gpt2 124M, fp32) |
|---|---|
| HF | **0.79 GB** |
| nanoGPT | 1.24 GB |

nanoGPT's path holds **two full copies** — it builds a nanoGPT `GPT` *and* an HF
`GPT2LMHeadModel`, then copies weights across. That is what put gpt2-xl at a
~12.4 GB peak and made it a poor fit for a g5.xlarge's 16 GiB. The HF path loads
once, in the serving dtype.

Bonus, unlooked for: **HF loads 3× faster** — 1.3 s vs 4.0 s — because it skips
the second allocation and the weight conversion. That lands directly in the
cold-start number E9 reports.

---

## P2.4 — No regressions ✅

```
129 passed, 2 warnings in 12.15s
ruff check: All checks passed!    ruff format: 42 files already formatted
```

10 new backend tests, and the existing worker/router/registry/metrics/bench
suites unchanged.

---

## ⚠️ Correction to the plan

The plan claimed **"no new dependencies."** That was wrong:
`low_cpu_mem_usage=True` **requires `accelerate`** in transformers 4.44.2 — the
first run failed with `ImportError`.

Handled by making it optional rather than mandatory: `backends.py` tries the fast
path and **falls back to a plain load** if `accelerate` is missing.
`accelerate` is added to the `fleet` extra so we get the better path where it's
installed. The measurements above were taken on the **fallback** path (accelerate
is not installed here), so the real numbers with it will be better, not worse.

---

## What this does NOT establish

1. **No GPU performance claim.** CPU wall-clock is not GPU wall-clock. `L0`,
   `C1`, tokens/s and $/1M tokens come from **E1/E2 on a g5.xlarge**.
2. **`gpt2-xl` has still never been loaded.** Only `gpt2` (124M). The memory
   argument for it is structural (one copy instead of two), not measured at that
   size.
3. **Nothing about batching.** Batch = 1 throughout. The $/1M-token headline is
   E10.
4. **No `torch.compile`.** Deferred until GPU numbers exist — its benefit is
   largest in the overhead-bound regime, and we may be bandwidth-bound after
   this change.
5. **Trained checkpoints keep the slow path.** Only the pretrained path gained a
   cache. The converter that would fix that is deliberately off the critical
   path.

---

## Next

Part 3 — containerize, then k3d locally (Docker, k3d, helm, k9s are all installed
and running), then multi-node k3s on EC2 once the AWS blockers clear.
