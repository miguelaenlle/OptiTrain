# Inference platform plan — Go router + k3s + Grafana

Plan of record for the inference track (supersedes the Part 1 / Part 3 sketches
in [ROADMAP.md](../ROADMAP.md) where they conflict). Goal: a serving fleet whose
speed, reliability, scalability, and cost are **measured and provable**, not
asserted.

---

## 0. Decisions

| Question | Decision | Rationale |
|---|---|---|
| Spot instances? | **No — on-demand.** | No GPU spot capacity available. Fault-tolerance is unaffected: "worker dies under load" is proven by pod delete / node drain / `TerminateInstances`, not by *why* it died. |
| Cost thesis | **Autoscaling + scale-to-zero**, not spot | Broader-applicability story, and it survives the capacity drought. |
| Model served | **GPT-2 XL 1.5B** via nanoGPT `GPT.from_pretrained('gpt2-xl')` | One code path (same `GPT` class, tiktoken GPT-2 BPE already a core dep), zero training cost, and it actually loads an A10G. |
| GPU | **g5.xlarge** (1× A10G 24 GB, 4 vCPU), on-demand | Best GPU-per-vCPU ratio in the g5 family (4 vCPU/GPU); anything larger halves node count per unit of quota. |
| Scale target | **64 nodes** headline, 128 stretch | Constraint is the **$200 budget**, not quota. 6 doublings; the 1→64 sweep costs ~$23. |
| Router language | **Go** (`router-go/`) | The pure `route_completion` policy ports 1:1; `context` models the connect/read timeout split more cleanly than `requests`. Zero training coupling. |
| Local cluster | **k3d** | No separate orchestrator; same k3s that runs in cloud. |
| Cloud cluster | **k3s on EC2**, spun up per-demo | Real multi-node drain/reroute. Never left running. |
| Dashboard | **Prometheus + Grafana** (kube-prometheus-stack) | Industry standard, K8s-native, dashboard-as-code. |

### Why not the 10M char model
On an A10G it serves ~1,700 tok/s — every latency number would measure FastAPI
overhead, not inference. GPT-2 XL gives ~140 tok/s single-stream and ~0.5 s
completions: percentiles that mean something.

Keep the trained GPT-2 124M checkpoint registered as a **second** model. It
preserves the train→serve story and proves the fleet is model-agnostic via the
existing `/v1/models` endpoint.

---

## 1. What exists today (working Python MVP)

| Component | File | State |
|---|---|---|
| Model load + generation | `src/inference/service.py` | Loads latest checkpoint; generation **serialized behind `_gen_lock`** |
| Worker | `src/inference/worker.py` | FastAPI; `POST /v1/completions` (OpenAI-shaped), `/healthz`, `/v1/models`, `/stats`; 5 s heartbeat |
| Router | `src/inference/router.py` | Round-robin + retry-on-failure (transport/5xx → next worker; 4xx passes through); connect 3 s / read 60 s; `route_completion` is a **pure, unit-tested** function |
| Registry | `src/inference/registry.py` | S3-or-local-dir heartbeat docs, 15 s TTL |
| Fleet lifecycle | `src/orchestrator/fleet.py` | `up/down/status/kill-worker`, local subprocesses or EC2 |
| Chaos experiment | `src/orchestrator/fleet_preempt.py` | Kill-under-load, before/disruption/recovered windows, PASS/FAIL |
| Live monitor | `src/orchestrator/monitor.py` | Terminal table, optional W&B mirror |
| Load generator | `loadgen/main.go` | Open-loop fixed-RPS + ramp, percentiles, chaos mode, JSON report |
| Tests | `tests/test_fleet_*.py` | ~445 lines |

**Declared gaps:** no continuous batching, no streaming.

---

## 2. Coupling to the training stack (measured)

**Six import lines across three files.** Direction is one-way: inference depends
on training, never the reverse — so nothing in this track can break training.

| Component | Imports from `spot_train` | Verdict |
|---|---|---|
| `loadgen/` | — | Zero |
| `router.py` | — (only `.registry`) | **Zero** ← the piece going to Go |
| `fleet.py`, `fleet_preempt.py`, `monitor.py` | — (only `.config`) | Zero |
| `registry.py` | `s3_store` | Infra, not training |
| `worker.py` | `TrainConfig` | Config carrier |
| `service.py` | `checkpoint`, `s3_store`, `TrainConfig`, `_char_codec`, `build_model` | **The real coupling** |

Of the six: 2 are shared infrastructure, 2 are a config carrier, and **3 are
genuine model-artifact coupling — all inside `ModelService.load()`**.

**The GPT-2 XL path eliminates all three** (no checkpoint format, no char codec,
no `meta.pkl`), leaving only nanoGPT (a `third_party/` submodule, i.e. a shared
dependency) and tiktoken.

### Cleanups adopted
1. **`build_model` drags in the whole training tree.** `from spot_train.train
   import build_model` executes `train.py`, pulling `distributed`, `events`,
   `data.PositionedLoader`, `interruption.InterruptionListener` — to get a
   14-line function. Bloats the worker image and slows cold start (a metric we
   measure). → extracted to `src/inference/model_build.py`.
2. `_char_codec` / `_bpe_codec` are private functions imported across a package
   boundary — fragile.
3. `s3_store` belongs in a neutral shared module.

---

## 3. Language split

**Go owns the network and control path; Python owns the tensor path.** The
boundary is exactly one HTTP hop.

| | Go | Python |
|---|---|---|
| **Components** | `loadgen/` (exists, ~364 L) · `router-go/` (~550 L) · drain-watcher DaemonSet (~180 L) | worker + model service · fleet CLI · `bench/` + plots · tests |
| **New code** | **~730 lines** | ~300 lines |
| **Under load?** | Every request crosses it | Only the worker (GPU-bound anyway) |

Two ambiguous pieces:
- **`registry.py` splits** — worker-side heartbeat stays Python (keeps
  `fleet up --local` working with no cluster); the *reading* side moves to Go and
  in-cluster is replaced by a K8s Endpoints watch.
- **`router.py` (Python) survives deliberately** — it's the control arm for
  experiment E3 (Go vs Python router under identical load), retired after.

---

## 4. Target architecture

| Layer | Choice | Notes |
|---|---|---|
| Local dev | k3d | Needs Docker running + `brew install k3d helm` |
| Cloud cluster | k3s, per-demo | — |
| k3s server + router | t3.small on-demand | CPU |
| Workers | g5.xlarge on-demand, **1 pod per node** | 1 GPU ⇒ replica count == node count |
| GPU enablement | NVIDIA container toolkit + device plugin DaemonSet, `nvidia.com/gpu: 1` | — |
| Router | Go, distroless, Prometheus `/metrics` | — |
| Discovery | K8s Endpoints in-cluster; S3 heartbeat as local/fallback | — |
| Observability | kube-prometheus-stack + dashboard JSON in `deploy/` | Provisioned via ConfigMap sidecar — **no UI clicking** |

### The constraint that shapes everything: 1 GPU per node

g5.xlarge has a single A10G, so worker pods cannot be packed. Therefore:
- Pre-provision the GPU node pool; HPA scales pods 0→N *inside* it.
- Report **pod-level scaling** (seconds) and **node-level scaling** (~2 min,
  where the dollars are) as separate numbers.
- Node scale-down reuses the tag-based EC2 launch/terminate already in
  `fleet.py` — Cluster-Autoscaler-lite with code we own.

---

## 5. Metrics and targets

Calibrate first, then every target is defensible instead of arbitrary:

- **L0** = unloaded single-request p50 (1 worker, RPS=1, `max_tokens=64`)
- **C1** = single-worker capacity = max RPS holding p99 < SLO
- **SLO** = **p99 ≤ 3 × L0** ← define once; everything keys off it

| Claim | Metric | Target |
|---|---|---|
| **Fast** | e2e p50/p95/p99 vs offered RPS | p99 < SLO at ≤ 80 % of knee |
| | p99 / p50 ratio | **< 3×** (< 2× excellent) |
| | C1 (knee) | report absolute RPS/worker |
| **Scalable** | tokens/s vs N workers | near-linear |
| | scaling efficiency `(tput_N/N) ÷ tput_1` | **≥ 90 % @ N=4, ≥ 80 % @ N=8** |
| **Reliable** | client-visible errors during kill | **exactly 0** |
| | rerouted count | > 0 (proves reroute fired) |
| | p99 recovery time | ≤ 25 s (15 s TTL + 10) |
| | availability, full run | ≥ 99.9 % |
| **Cost** | $/1M tokens (autoscaled) | report absolute |
| | savings vs. static-peak | **≥ 2×** on diurnal load |
| | p99 during ramp | < SLO for ≥ 99 % of samples |
| | pod / node scale-up | < 60 s / < 3 min |
| | cold start from zero | < 60 s, reported honestly |

**Reporting rule:** every experiment reports client-side (loadgen) and
server-side (Prometheus) numbers separately, and states which is being quoted.

### Why batching is the headline finding

At $1.006/hr, serialized generation is ~10× off market rates:

| Config | Aggregate tok/s | $/1M tokens |
|---|---|---|
| GPT-2 XL, no batching | ~140 | ~$2.00 |
| GPT-2 XL, batch 32 | ~1,500 | **~$0.19** |

Hosted Llama-8B APIs run ~$0.20/1M, so the batched number is *credible* and the
serialized `_gen_lock` baseline is a legitimately damning "before" chart.
**Measure serialized first.**

---

## 6. Phases and experiments

| Phase | Work | Experiments | Proves | Cost |
|---|---|---|---|---|
| **0** | GPT-2 XL serving path, bf16, decouple `build_model` | — | — | $0 |
| **1** | Baseline on one g5 (Python router) | **E1** L0/C1 calibration · **E2** saturation curve | **Fast** | ~$2 |
| **2** | Prometheus `/metrics`, `bench/`, `plots.py`, Go router | **E3** Go vs Python router A/B | — | ~$2 |
| **3** | Dockerfiles, k3d, Kustomize | **E4** pod-kill chaos, local | Reliable (local) | $0 |
| **4** | k3s on EC2, GPU device plugin, scale out | **E5** throughput sweep 1→64 · **E6** router-stress sweep | **Scalable** | ~$40 |
| **5** | Drain DaemonSet, PDBs | **E7** node drain + hard kill under load | **Reliable** | ~$8 |
| **6** | HPA + prometheus-adapter + node scale-down | **E8** diurnal autoscale · **E9** scale-to-zero cold start | **Cost** | ~$24 |
| **7** | *(stretch)* dynamic batching | **E10** serialized vs batched | Cost headline | ~$6 |

The three that matter most:
- **E6 (router-stress)** — short completions at 64 nodes push ~9k req/s at a
  single router. The only experiment that finds *our* control-plane limit rather
  than the GPU's.
- **E7 (chaos at scale)** — the "0 client errors, p99 recovered in 12 s"
  timeline. The portfolio screenshot.
- **E10 (batching)** — the largest single number in the project.

### Sweep design (two cost techniques)

**① Boot once at max, scale *down*.** Run N=64, terminate half, run N=32, halve
again. Boots 64 nodes instead of 127; EC2 bills per-second. Also what you'd do
in production.

**② Scale duration inversely with N** — stable p99 needs a fixed *sample count*,
and a 64-node fleet produces samples 64× faster.

| N | Duration | Node-hr |
|---|---|---|
| 1 | 20 min | 0.33 |
| 2 | 20 min | 0.67 |
| 4 | 15 min | 1.00 |
| 8 | 15 min | 2.00 |
| 16 | 10 min | 2.67 |
| 32 | 10 min | 5.33 |
| 64 | 10 min | 10.67 |
| **Total** | | **22.7 ≈ $23** |

---

## 7. Portfolio artifacts

1. **Saturation curve** — capacity + tail discipline
2. **Scaling-efficiency chart** — "92 % linear to 64 workers"
3. **Chaos-recovery timeline** — "0 client errors, p99 recovered in 12 s, 99.9x %"
4. **Autoscaling triple-overlay + cost bar** — "held p99 under SLO idle→peak at
   $Z/1M tokens, 2.4× cheaper than static"

Each is a Grafana panel *and* an offline figure generated from loadgen JSON —
reproducible in CI, not screenshot-only.

---

## 8. Budget

> **⚠️ This plan spends real money. g5.xlarge is ~$1.00/hr each, on-demand.**
> **At 64 nodes the burn rate is ~$64/hr.** Leaving the fleet up overnight
> (12 h) ≈ **$773**, ~4× the budget.

| Item | Node-hr | $ |
|---|---|---|
| E1 calibration | 1 | $1 |
| E2 + E5 throughput sweep 1→64 | 23 | $23 |
| E6 router-stress sweep | 16 | $16 |
| E7 chaos | 8 | $8 |
| E8/E9 diurnal autoscale | 24 | $24 |
| E10 batching | 6 | $6 |
| **Experiments** | **78** | **$78** |
| Debug / reruns / boot overhead (~55 %) | 43 | $43 |
| **Total** | **121** | **$121** |
| **Reserve (of $200)** | | **$79** |

**Non-negotiable controls:** hard teardown after every run, idle-timeout
self-terminate on nodes, and an AWS Budgets alarm at $150.

---

## 9. Who runs what

Following this repo's convention that the user runs every credentialed command:

| Actor | Responsibility |
|---|---|
| **Claude** | All code, manifests, Helm values, Grafana dashboard JSON, prometheus-adapter rules, tests. Runs local CPU tests, lint, k3d + `helm install`, and verifies scrape targets/panels locally. |
| **User** | One-time: start Docker, `brew install k3d helm`. Then anything credentialed or billable — ECR push, EC2 launches, cloud k3s, `fleet up`. Plus taste calls on the dashboard. |

The dashboard is a **version-controlled JSON file** loaded by the
kube-prometheus-stack sidecar (ConfigMap labeled `grafana_dashboard: "1"`), so it
survives cluster teardown and is never configured by hand.
