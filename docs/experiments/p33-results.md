# P3.3 results — multi-node k3s on EC2, and the black-hole failure mode

**Run:** 2026-08-08 · **Region: us-east-2 only** · **Cost: ~$0.15** (t3 instances;
torn down immediately, us-east-1 verified untouched throughout).
**Plan:** [hf-kv-k3s-plan.md](./hf-kv-k3s-plan.md) · Raw data: `.fleet/p33-results.json`

Cluster: 1× t3.small k3s server + 2× t3.medium agents, one worker pod per agent,
router pinned to the server node. Workers are the Go stub — these questions are
about the **network and the cluster**, not the model, and a stub keeps model
latency out of the measurement.

---

## Headline

| Test | Result |
|---|---|
| **(a) Discovery** | 2 live workers via the headless Service, heartbeat provably off |
| **(b) Cross-node** | 10 requests each to pods on **2 different EC2 instances** |
| **(c) Black-hole** | **76/76 requests OK, 0 client-visible errors** while an entire instance was terminated under load |

---

## (a) Discovery ✅

```
live_workers=2
  10.42.1.2   10.42.1.2:8001
  10.42.2.2   10.42.2.2:8001
```

Workers run with `FLEET_WORKERS_URI=""`, so the heartbeat path is off on both
sides. The router still sees both pods, and no other discovery mechanism exists
— so DNS/EndpointSlice is provably what found them, rather than merely
plausibly.

## (b) Cross-node serving ✅

Attribution comes from the **router's own record** of which upstream it dialled,
not from the response body. The stub derives its id from its port, so both pods
call themselves `stub-8001` and the body cannot distinguish them. The router
keys on pod IP, which is unique — and it is the better witness anyway, being the
component that does the choosing.

| pod IP | requests | node |
|---|---|---|
| 10.42.1.2 | 10 | ip-172-31-92-248 |
| 10.42.2.2 | 10 | ip-172-31-86-149 |

Even round-robin across **two separate EC2 instances**.

---

## (c) Black-hole ⭐ the reason any of this ran in the cloud

`TerminateInstances` on a whole agent node, under continuous load.

**Why this could not be tested anywhere cheaper:** killing a local process, or
deleting a pod, produces an immediate `ECONNREFUSED` — the kernel actively
rejects. A **terminating EC2 instance simply stops answering**, and its packets
are silently dropped. There is no error for the router to react to; it must time
the connection out. localhost and k3d structurally cannot produce that.

```
requests=76  ok=76  errors=0
router: requests=116  rerouted=14  failed=0  live=1
```

### The 3.45s signature

Nine requests took 3.38–3.46s. Every other one was ~0.4s.

That is `ROUTER_CONNECT_TIMEOUT_SECONDS=3` doing exactly its job: the
round-robin cursor lands on the dead pod, the SYN goes nowhere, the connect
times out at 3s, and the request is rerouted to the survivor and succeeds. The
client waits — but it is never failed.

```
t=11.7s  3.38s      ← every ~4.7s: cursor hits the dead pod,
t=16.5s  3.46s         pays the connect timeout, reroutes
...
t=49.5s  3.44s      ← last one
then     ≤0.40s     ← 49 further requests, all clean
```

**This is also the first real evidence for `ROUTER_UPSTREAM_KEEPALIVE=false`.**
A pooled connection to a vanished instance has no SYN to time out, so it would
have blocked on the 60s read window instead of the 3s connect window — twenty
times worse, and well past most client timeouts.

### The finding that matters most: node death ≫ pod death

The dead endpoint stayed in rotation for **~38 seconds**.

| Failure | Detection | Mechanism |
|---|---|---|
| Pod deleted (graceful) | **0.83 s** | control plane removes it from Endpoints *before* it dies |
| Pod killed (abrupt) | **4.65 s** | readiness probe fails (2s × 2) |
| **Whole node terminated** | **~38 s** | node-monitor-grace-period, *then* pod eviction |

Readiness probes cannot help when a node vanishes: the kubelet that would report
NotReady died with it. Kubernetes must wait out its node-monitor grace period
before evicting the pods.

**So for the failure mode this platform is actually built around — losing a
machine — Kubernetes is ~8× slower to react than for a pod failure, and the
router's retry is the only thing standing between that gap and the client.**
Zero errors across 38 seconds of a stale endpoint is a router result, not a
Kubernetes one. That is worth stating plainly, because "Kubernetes handles it"
is the obvious wrong conclusion to draw from a green test.

---

## What it cost to get here

Six clusters. Five failed, each for a distinct reason invisible on a laptop:

| # | Failure | Why local testing could not catch it |
|---|---|---|
| 1 | No subnets flagged `DefaultForAz` | No such concept locally |
| 2 | **Stateless NACL** blocked ephemeral inbound | Inbound SSH worked while every outbound hung — read as a dead IGW |
| 3 | Stub bound `127.0.0.1` | Fine as a process; unreachable across a pod network boundary |
| 4 | Stale image tag in containerd | k3d imports fresh every time |
| 5 | **`--node-external-ip` on the server** | Published the *public* IP as the `kubernetes` Service endpoint, so every pod's API access went out through the IGW and was blocked by the SG |

Failure 5 is the one worth remembering. Nodes reported `Ready` throughout —
kubelet heartbeats are outbound and never touched the broken path — while
CoreDNS sat at 0/1, metrics-server and local-path crashlooped, cluster DNS never
answered, and the router reported `live_workers: 0` with **no error anywhere**.
Three layers between cause and symptom.

Also fixed along the way: flannel needs **UDP 8472** (without it nodes join,
pods schedule, and cross-node traffic silently vanishes), and traefik /
metrics-server / local-storage are now disabled — unused here, and on a 2GB
t3.small they crowded out the logs needed to diagnose all of the above.

---

## What this does NOT establish

1. **Nothing about GPU or model performance.** Stub workers, no inference.
   `L0`, `C1`, tokens/s and $/1M tokens still come from E1/E2 on a g5.xlarge.
2. **No scaling claim.** Two workers.
3. **The 38s node-detection figure is k3s's default grace period**, not a tuned
   result. It can be shortened, at the cost of false positives on a slow node.
4. **One trial per failure mode.** The 3.45s connect-timeout signature is
   consistent across 9 samples within a single run, not across runs.
5. **`kubectl logs`/`exec` to agent nodes returned 502** for the whole session
   (API server → kubelet :10250). It never blocked the experiments, which run
   over HTTP through the router, but it is unexplained and would matter for
   production debugging.

## Known weakness this exposed in our own code

`DNSRegistry` treats NXDOMAIN as "no workers" and returns **no error**, so a
genuinely broken resolver is indistinguishable from an empty fleet. During
failure 5 this turned a DNS outage into a silent `live_workers: 0`. Resolution
failure should be surfaced distinctly from "resolved fine, zero ready pods."
