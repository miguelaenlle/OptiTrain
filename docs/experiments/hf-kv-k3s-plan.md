# Plan — HF/KV-cache engine → serve it → serve it on multi-node k3s

**Status:** planned, not started.
**Region:** everything cloud is **us-east-2** only. Nothing is created, mutated,
or destroyed in us-east-1.
**Quota ceiling:** 8 GPU vCPU (= 2 × g5.xlarge) · 32 Standard vCPU (= ~14 ×
c7i.large, minus 4 for the k3s server + router).

## Already done (PR #4)

Go router (42 tests, 9.3× the Python router's throughput at ~3× lower
overhead — see [x2-x3-results.md](./x2-x3-results.md)) · Prometheus `/metrics`
on both routers · `bench/` harness · stock-GPT-2 serving path.

---

# Part 1 — HF backend with KV caching

## Why

nanoGPT's `generate()` re-runs the **full sequence every token** — ~34× the
necessary FLOPs at 64 tokens, ~130× at 256. There is no KV cache, and
`third_party/nanoGPT` is a pinned read-only submodule ("we import, never
rewrite"). So we consume HF's implementation rather than write our own.

## Design: one backend interface, two implementations

`ModelService.complete()` should not know which engine it is talking to. Insert
a minimal seam:

```
Backend protocol:  generate(ids, *, max_new_tokens, temperature, top_k, seed) -> list[int]

  NanoGPTBackend   wraps GPT.generate()          — trained checkpoints, unchanged
  HFBackend        wraps GPT2LMHeadModel.generate(use_cache=True)  — stock GPT-2
```

Everything else in `ModelService` (the lock, `ServiceStats`, the OpenAI response
shape) is untouched, so the worker API and all existing tests keep working.

| Path | Backend | Why |
|---|---|---|
| `PRETRAINED_MODEL=gpt2-xl` | **HF** | KV cache; and the nanoGPT class buys nothing for stock weights |
| trained checkpoint (default) | nanoGPT | must stay byte-for-byte the training artifact |

**Deliberately deferred:** converting trained checkpoints into
`GPT2LMHeadModel` so *everything* gets the cache. It's the authenticity demo,
not the benchmark subject — off the critical path.

## Details that will bite if missed

| Detail | Handling |
|---|---|
| HF `generate` returns **prompt + completion** | slice `out[0, len(ids):]` — same as the nanoGPT path |
| `temperature` only applies with `do_sample=True` | set explicitly; don't rely on defaults |
| `top_k=0` | map to `None` |
| GPT-2 has no pad token | `pad_token_id=eos_token_id` to silence warnings (batch=1 today) |
| Memory | `torch_dtype` + `low_cpu_mem_usage=True` → ~3.1 GB peak instead of 12.4 GB. **This is what makes gpt2-xl fit g5.xlarge's 16 GiB.** |
| Tokenizer | keep tiktoken `_bpe_codec` (already a core dep). **Assert it agrees with HF's tokenizer** rather than assuming |
| `use_cache=True` | explicit, not implicit — it is the entire point |

## Files

| File | Change |
|---|---|
| `src/inference/backends.py` | **new** — the protocol + both implementations |
| `src/inference/service.py` | pick a backend in `load()`; `complete()` calls the protocol |
| `tests/test_backends.py` | **new** |
| `pyproject.toml` | no change (`transformers` already in the `fleet` extra) |

**No new dependencies.** Using `torch_dtype`/`low_cpu_mem_usage` avoids pulling
in `accelerate`.

---

# Part 2 — testing we can serve it

## P2.1 Backend equivalence (local, $0) ⭐ the important one

Same weights through both backends must produce the **same tokens**.

- Load `gpt2` (124M) twice: NanoGPTBackend and HFBackend.
- **Greedy decode** (`temperature→0`/`do_sample=False`, fixed seed) — sampling
  would make exact comparison impossible, greedy makes it deterministic.
- Assert **token-for-token identical output** for 3 prompts × 32 tokens.

If this passes, the HF path is not "similar," it is equivalent — and every
downstream number is trustworthy. If it fails, we have a real bug and stop.

Also: assert tiktoken's ids == HF tokenizer's ids on a sample string.

## P2.2 KV-cache speedup, measured (local CPU, $0)

Time 64-token generation on `gpt2` through both backends, same machine, same
prompt.

- **Pass:** HF (cached) is materially faster. Expect several×; report the actual
  ratio, do not predict it.
- CPU wall-clock ≠ GPU wall-clock — this proves the cache *works*, not how fast
  the GPU will be.

## P2.3 Memory ceiling (local, $0)

`resource.getrusage(RUSAGE_SELF).ru_maxrss` around an HF `gpt2-large` load.

- **Pass:** peak well under the old double-allocation, confirming gpt2-xl (2×
  larger) will fit g5.xlarge's 16 GiB.

## P2.4 Worker API unchanged (local, $0)

Serve `gpt2` via HF behind the real FastAPI worker; run the existing worker
contract tests. **Pass:** OpenAI response shape unchanged, `/stats` counters
move, 93-test suite still green.

## P2.5 — E1/E2: gpt2-xl on GPU (us-east-2, ~$2) 🔴 needs AWS go-ahead

**1 × g5.xlarge**, bare process (not containerized yet — so a failure is
unambiguously the engine, not the container).

1. Load `gpt2-xl`, record cold-start seconds and peak host RSS + VRAM.
2. Serve one completion; confirm coherent English.
3. **E1:** `L0` = unloaded p50 at RPS=1, `max_tokens=64`. Define `SLO = 3 × L0`.
4. **E2:** RPS sweep via `bench/` → saturation curve, knee `C1`, p99/p50 ratio.

**Pass:** loads inside 16 GiB; coherent output; `L0` and `C1` recorded. These two
numbers unlock every other target in the platform plan.

> X1 from the earlier plan is **folded in here.** The HF path removed the memory
> risk that justified a separate cheap CPU box, so the `r7i.xlarge` step is
> dropped.

---

# Part 3 — serving on multiple nodes with k3s

## P3.1 Containerize (local, $0)

- Worker image (Python + torch; CPU base locally, CUDA base for GPU nodes)
- Router image (Go, multi-stage → distroless)
- **Pass:** both run locally under Docker and serve exactly as bare processes did.

## P3.2 k3d single-node cluster (local, $0)

`kubectl apply -k deploy/local` → 1 router + 2 CPU worker pods on `gpt2`.

| # | Test | Pass |
|---|---|---|
| a | Router discovers workers via **K8s Endpoints**, heartbeat backend off | `/fleet/status` shows 2 |
| b | Completions served through the K8s Service | 200, correct shape |
| c | **`kubectl delete pod` under load** | **0 client errors**, `rerouted > 0` |
| d | Deployment self-heals | pod replaced with no human action |
| e | **Discovery latency: probe vs heartbeat TTL** | ~2 s vs 13 s (measured in X2) — the concrete K8s win |
| f | Prometheus scrapes both; Grafana panels populate | targets `UP` |

> **Carry-forward from X3:** Go's `client_golang` **omits zero-valued metric
> families**, so a cold router returns *no data*, not `0`. Grafana panels must
> treat absent as zero or the fleet looks broken after every restart. Fix in the
> dashboard JSON here.

## P3.3 Multi-node k3s on EC2 (us-east-2) 🔴 needs AWS go-ahead

**This is the actual "multiple nodes" test.** Everything above ran on one host.

```
   t3.small  ── k3s server + router pod        (CPU, off GPU quota)
   g5.xlarge ── agent node 1 → worker pod      ⎫ 1 GPU per node ⇒
   g5.xlarge ── agent node 2 → worker pod      ⎭ 1 worker pod per node
```

| # | Test | Pass criteria |
|---|---|---|
| a | Agents join the cluster | `kubectl get nodes` → 3 Ready |
| b | GPU scheduling works | pods land via `nvidia.com/gpu: 1`; `nvidia-smi` in-pod |
| c | **Cross-node serving** | completions served by workers on **both** nodes — verified by `worker_id`, not assumed |
| d | **Kill a whole node under load** (`TerminateInstances`) | **0 client errors**, `rerouted > 0`, p99 recovers |
| e | **Cordon + drain** (graceful) | 0 errors; distinct from (d)'s hard kill |
| f | **Black-hole failure mode** ⭐ | The one X2 **could not** test: localhost gives instant `ECONNREFUSED`, a terminating EC2 box **silently drops packets**. This is what the 3 s connect timeout and `ROUTER_UPSTREAM_KEEPALIVE=false` exist for. **First real validation.** |

**Scale variant (CPU, cheap):** swap agents for up to 8 × `c7i.large` from the
separate 32-vCPU Standard pool to run **E5** (scaling 1→2→4→8) without touching
the GPU quota — ~$0.09/hr each.

---

# Cost

| Part | Where | $ |
|---|---|---|
| 1 · engine | local | **$0** |
| 2.1–2.4 · serving tests | local | **$0** |
| 2.5 · E1/E2 on GPU | 1 × g5.xlarge, ~2 h | ~$2 |
| 3.1–3.2 · containers + k3d | local | **$0** |
| 3.3 · multi-node k3s | t3.small + 2 × g5.xlarge, ~2 h | ~$4 |
| 3.3 scale variant | 8 × c7i.large, ~1 h | ~$1 |
| **Total** | | **~$7** |

Peak burn ~$2/hr. Teardown after every run; budget alarm at $150.

# Blockers

| Blocker | Gates |
|---|---|
| 🔴 AWS go-ahead + **S3 bucket name** (us-east-2) | 2.5, 3.3 |
| 🔴 `project=spot-train` hardcoded (`aws.py:515`) | any launch — your teardown snippet can't see my instances until this is `project=inference` |
| 🔴 Region default is `us-east-1` (`config.py:92`) | needs a launch guard |
| 🔴 `fleet down` is `fleet_id`-blind | scoping fix |
| 🟡 `open -a Docker && brew install k3d helm k9s` | 3.1, 3.2 |

**Parts 1 and 2.1–2.4 are unblocked and can start now.**

# What this plan does NOT prove

1. **Nothing about batching.** Batch=1 throughout. The $/1M-token headline
   (~$2.00 → ~$0.19) is E10, after this.
2. **No `torch.compile`.** Deferred until KV-cache numbers exist — its benefit
   is largest in the overhead-bound regime, and after Part 1 we may be
   bandwidth-bound instead. Measure, then decide.
3. **No streaming / TTFT.** We measure end-to-end only. Real platforms report
   TTFT separately; that's a known gap, not an oversight.
4. **CPU timings are not GPU timings.** P2.2's ratio proves the cache works;
   absolute latency comes from 2.5.
5. **Trained checkpoints keep the slow path** until/unless the converter lands.
