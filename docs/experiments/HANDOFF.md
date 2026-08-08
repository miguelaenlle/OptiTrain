# Handoff — OptiTrain inference platform

For an agent picking this up cold. Read §0 and §1 before touching anything.

---

## 0. ⚠️ Where the work lives — read this first

**This work is NOT in the main checkout.** It lives in a git worktree:

```
/Users/miguelaenlle/conductor/repos/spot-distributed-llm-training/.claude/worktrees/inference-go-k8s
```

- Branch `worktree-inference-go-k8s`, open as **PR #4** against `main`.
- **`cd` there first.** Running from the main checkout or any other Conductor
  workspace will silently operate on different code.
- **There are ~10 other worktrees on this machine, several with active work.**
  Do not read from or write to them.
- **Never merge, never push to `main`, never force-push.** The user decides.

### The trap that will bite you

The editable install (`__editable__.spot_train-0.0.1.pth`) points at a
**different worktree** (`workspaces/.../san-antonio/src`). So:

```bash
python -c "import inference; print(inference.__file__)"   # -> san-antonio, WRONG
```

**Always export `PYTHONPATH=<this worktree>/src`.** Without it you will test
code you did not write and get confusing passes. Every command below assumes:

```bash
cd /Users/miguelaenlle/conductor/repos/spot-distributed-llm-training/.claude/worktrees/inference-go-k8s
export PYTHONPATH=$PWD/src
set -a; . ./.env; set +a          # INFERENCE_* vars
```

### AWS rules — non-negotiable

| | |
|---|---|
| **Inference region** | **us-east-2 only** |
| Training region | us-east-1 — **never create, mutate, or destroy anything there** |
| Tag | every instance gets `project=inference` |
| IAM/SG names | `inference-*` (IAM is **global**, so the region split does not protect it) |
| Bucket | `optitrain-inference-us-east-2` |

Contract: `workspaces/.../san-antonio/docs/prompts/inference-agent-region.md`.
`OrchestratorConfig.for_inference()` refuses us-east-1 outright.

**Always tear down.** `./deploy/aws/k3s-down.sh --all`, `./deploy/aws/gpu-down.sh`.
Verify zero instances after every session.

---

## 1. State

**Works and is measured:**

| | |
|---|---|
| Go router (`router-go/`) | 50 tests. **9.3× the Python router's throughput**, ~3× lower per-request overhead |
| HF backend + KV cache | Token-for-token identical to nanoGPT under greedy decode |
| k3d local cluster | Pod-kill chaos, 0 errors |
| **Multi-node k3s on EC2** | **76/76 requests, 0 errors through a full instance termination** |
| E1/E2 on a real A10G | The four baseline numbers below |

**Written but NEVER RUN:** `VLLMBackend` (`src/inference/backends.py`). vLLM
needs CUDA+Linux; the dev machine is an arm64 Mac. **Assume it is broken until a
GPU proves otherwise.**

### The baseline to beat (measured, `docs/experiments/e1e2-results.md`)

gpt2-xl, bf16, HF backend, 1× g5.xlarge:

| | |
|---|---|
| L0 (unloaded p50, 64 tok) | **2,097 ms** |
| Decode rate | **30.5 tok/s** |
| Single-worker capacity | **0.48 req/s** |
| $/1M tokens | **$7.96** |
| vs memory-bandwidth bound | **4.6× slower** (~26 ms/token is overhead, not compute) |
| vs market rate | **~40× more expensive** |

### Fault tolerance (measured, `docs/experiments/p33-results.md`)

| Failure | Detection |
|---|---|
| Pod deleted (graceful) | 0.83 s |
| Pod killed (abrupt) | 4.65 s |
| **Whole EC2 instance terminated** | **~38 s** |

The 38 s matters: readiness probes cannot help when a node dies, because the
kubelet that would report NotReady died with it. **The router's retry is the
only thing between that gap and the client.** Zero errors across 38 s of a stale
endpoint is a *router* result, not a Kubernetes one.

### Quotas

- **GPU: 8 vCPU = 2 × g5.xlarge.** User will raise to **64 vCPU (16 nodes)
  after testing** — do not plan large GPU fleets before that lands.
- **Standard: 32 vCPU = 16 × c7i.large.**

---

## 2. Phase A — vLLM (do this first)

**Why:** not fashion. E1/E2 measured two specific deficiencies — per-token
overhead and no batching — and vLLM exists to fix exactly those.

### The one thing that will silently ruin this

`Backend.concurrent`. vLLM schedules sequences itself, so `ModelService` must
**not** serialize calls to it. If the lock is held, vLLM gets one sequence at a
time, continuous batching does nothing, and **there is no error** — throughput
just stays at the unbatched rate and a benchmark will happily "confirm" it.
`VLLMBackend.concurrent = True` handles this; `tests/test_backend_concurrency.py`
pins it. **If vLLM shows <2× throughput, check this first.**

### Success criteria

| Metric | Baseline | Target | Stretch |
|---|---|---|---|
| Single-stream decode | 30.5 tok/s | **≥ 80 tok/s** | 120 |
| L0 (64 tok) | 2,097 ms | **≤ 800 ms** | 550 |
| Aggregate @ concurrency 32 | n/a | **≥ 1,000 tok/s** | 2,500 |
| $/1M tokens | $7.96 | **≤ $0.30** | $0.15 |
| Output sanity | — | coherent English, `prompt_tokens == 5` | — |

Targets are anchored to the ~140 tok/s bandwidth bound: ≥80 single-stream means
most per-token overhead is gone; ≥1,000 aggregate means batching works.

**If vLLM underperforms**, the fallback is cheaper and still valuable: keep the
HF backend and land `torch.compile` + `StaticCache` (already pinned SDPA). Say
so plainly rather than forcing the vLLM number.

### Then re-run E1/E2 — the current rig has a known bug

`deploy/aws/e1e2-run.py` starts its sweep at **1 rps, which is already 210 %
utilisation** (2.10 s service time ⇒ 0.48 req/s capacity). Every point measured
queueing, not capacity, so **`C1` has never been measured.** Start the sweep at
`0.2 × (1/L0)` and step up.

---

## 3. Phase B — Fault tolerance at scale *(before scaling, per the user)*

Mostly proven at 2 nodes; what is missing is **duration and repetition**.

| Test | Success criteria |
|---|---|
| Sustained chaos, ≥30 min, repeated node kills | **0 client-visible errors**; availability **≥ 99.9 %**; report request count |
| Graceful drain vs hard kill, side by side | drain ≤ 5 s; hard kill bounded by connect timeout (~3.4 s/attempt observed) |
| Failure *during* an autoscale event | no errors; the case nobody tests |
| Multiple simultaneous node loss | survives N−2 of N |

Use CPU workers (`c7i.large`, ~$0.09/hr) — **fault tolerance is a router and
cluster property, not a GPU one.** Do not burn GPU quota on it.

Expect ~3.4 s per rerouted request during a node's black-hole window; that is
`ROUTER_CONNECT_TIMEOUT_SECONDS=3` working, not a bug.

---

## 4. Phase C — Scaling (after fault tolerance)

| Metric | Target @4 | @8 | @16 |
|---|---|---|---|
| Scaling efficiency `(tput_N/N)÷tput_1` | ≥ 90 % | ≥ 80 % | **≥ 70 %** |
| p99 at constant load/worker | within 10 % of N=1 | | |
| Router knee (short completions) | measure — this is the control-plane limit | | |

70 % at 16 is deliberately not 90 %: a single router fronting 16 upstreams will
show falloff, and **finding where it falls off is the result**. Do not tune to
hit a number.

GPU scaling is capped at 2 nodes until the quota lands.

---

## 5. Phase D — Dashboard

`deploy/monitoring/` is written and `helm template`-clean but **has never
scraped anything in the cloud**. Two carry-forwards:

- **Panels must treat absent as zero.** Go's `client_golang` omits zero-valued
  metric families, so a cold router returns *no data*, not `0`, and the fleet
  looks broken after every restart.
- **k3s runs one control-plane process**, so the chart's
  `kubeEtcd`/`kubeScheduler`/`kubeControllerManager`/`kubeProxy` ServiceMonitors
  alert on components that are fine. Already disabled in `values.yaml`.

**Missing: GPU metrics.** `kube-prometheus-stack` gives none, and GPU
utilisation is *the* metric that proves batching works (expect ~15 % → ~80 %).
Cheapest path: the worker's `/stats` already reports `gpu_util` and
`gpu_mem_used_mb`; wire them into Prometheus gauges (~20 lines).

---

## 6. Traps already paid for — do not rediscover

| Trap | Symptom |
|---|---|
| **Flannel needs UDP 8472** | Nodes join, pods schedule, cross-node traffic silently vanishes |
| **Subnets not `DefaultForAz`** | `RunInstances`: "No subnets found" despite healthy subnets |
| **Stateless NACLs** | Inbound SSH works, *all outbound hangs* — looks like a dead IGW. `k3s-up.sh` now requires a permissive ingress rule |
| **`--node-external-ip` on the k3s server** | Publishes the public IP as the `kubernetes` Service endpoint; every pod's API call leaves via the IGW and is dropped. Nodes still report `Ready` |
| **Containers bind `0.0.0.0`, not `127.0.0.1`** | Loopback is unreachable across a pod boundary; probes get connection-refused |
| **torch ignores cgroup limits** | One thread per *host* core per pod; measured 1.2 tok/s vs 85. Set `OMP_NUM_THREADS` |
| **distroless + `runAsNonRoot`** | Needs a **numeric** `runAsUser` (65532) |
| **arm64 Mac → amd64 EC2** | `exec format error`. Build `--platform linux/amd64` |
| **No ECR permissions** | Ship images via S3; k3s auto-imports tarballs from `/var/lib/rancher/k3s/agent/images/` |

---

## 7. Method — what makes these numbers trustworthy

1. **Falsifiable setup.** Discovery is tested with the heartbeat *off*, so DNS is
   provably the mechanism rather than plausibly.
2. **Attribute from the right witness.** Cross-node serving is measured from the
   router's record of which upstream it dialled, not the worker's self-report —
   the stubs all call themselves `stub-8001`.
3. **Open-loop load.** A closed loop throttles to the server's pace and can never
   show saturation.
4. **Stop at the first SLO violation.** Capacity is the leading healthy run; a
   lucky sample above a breakdown must not report capacity that does not exist.
5. **Every writeup ends with "what this does NOT establish."** Keep that.
6. **Bare process before Kubernetes** on any new hardware, so an engine bug and
   a container bug stay distinguishable.

## 8. Known weaknesses in our own code

- `bench/` sweep live path is only `--dry-run` tested.
- Worker `/stats` is JSON, not Prometheus — worker metrics reach Prometheus only
  via the router's scrape loop.
- `kubectl logs`/`exec` to k3s **agent** nodes returned 502 (API server →
  kubelet :10250) for a whole session. Never blocked experiments; unexplained.
- Trained checkpoints still use the slow nanoGPT path. Converting them to
  `GPT2LMHeadModel` would give them the KV cache too; deliberately off the
  critical path.
