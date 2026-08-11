# Monitoring — OptiTrain inference fleet

Prometheus + Grafana for the fleet, as code. No UI clicking: the dashboard is a
version-controlled JSON file that a sidecar provisions from a ConfigMap, and the
whole stack is one Helm release plus two manifests.

| File | What it is |
|---|---|
| `values.yaml` | kube-prometheus-stack values. k3s-safe, small, short retention. |
| `servicemonitor-router.yaml` | The fleet's **only** Prometheus scrape target. |
| `servicemonitor-workers.yaml` | Why there is no worker scrape (and the block to enable when there is). **Creates nothing today.** |
| `dashboard-fleet.json` | The dashboard. Source of truth. |
| `dashboard-configmap.yaml` | The same JSON wrapped in a ConfigMap labelled `grafana_dashboard: "1"`. Generated — see [Editing the dashboard](#editing-the-dashboard). |

---

## Install

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

kubectl create namespace monitoring

# The Grafana admin password comes from a Secret, never from this repo and
# never from --set (which lands in shell history).
kubectl -n monitoring create secret generic grafana-admin \
  --from-literal=admin-user=admin \
  --from-literal=admin-password="$(openssl rand -base64 24)"

helm upgrade --install optitrain prometheus-community/kube-prometheus-stack \
  --version 88.2.0 -n monitoring -f deploy/monitoring/values.yaml

# The scrape target and the dashboard.
kubectl apply -f deploy/monitoring/servicemonitor-router.yaml
kubectl apply -f deploy/monitoring/dashboard-configmap.yaml
```

`servicemonitor-router.yaml` selects on `app: fleet-router` and the **named**
port `http` — read off the real Service in
[`deploy/k8s/base/router.yaml`](../k8s/base/router.yaml), not assumed. Nothing in
`deploy/k8s/` pins a namespace, so the ServiceMonitor uses
`namespaceSelector: { any: true }` and lives in `monitoring` itself; it finds the
router wherever `kubectl apply -k deploy/k8s/local` puts it.

To run the Go-vs-Python A/B (E3), give the second router's Service a distinct
`app` value (e.g. `app: fleet-router-py`) and widen the selector to a
`matchExpressions … In [fleet-router, fleet-router-py]`. Because `jobLabel: app`,
the two arms then arrive as two separate `job` values and the dashboard's
**Router** variable separates them with no further wiring.

### Reaching Grafana

No Ingress, no LoadBalancer — nothing is exposed on the k3s server's public
interface for a demo dashboard.

```bash
kubectl -n monitoring port-forward svc/optitrain-grafana 3000:80
# then open http://localhost:3000

# username: admin
kubectl -n monitoring get secret grafana-admin \
  -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

Grafana opens directly on **OptiTrain – Inference Fleet**
(`grafana.ini.dashboards.default_home_dashboard_path` points at the file the
sidecar writes). Prometheus itself, if you need the expression browser or the
Targets page:

```bash
kubectl -n monitoring port-forward svc/optitrain-kube-prometheus-prometheus 9090:9090
```

### Verify

```bash
# Every target UP, and none of the k3s phantoms present:
kubectl -n monitoring exec -it prometheus-optitrain-kube-prometheus-prometheus-0 \
  -c prometheus -- wget -qO- localhost:9090/api/v1/targets \
  | python3 -c 'import json,sys; [print(t["health"], t["labels"]["job"]) for t in json.load(sys.stdin)["data"]["activeTargets"]]'
```

---

## Which panel answers which question

Four rows, four claims. Every query reads only the seven metric families
declared in `src/inference/metrics.py` (and mirrored verbatim in
`router-go/metrics.go`, which is what lets one dashboard serve both A/B arms).

### Row 1 — Fleet at a glance

| Panel | Question | Query |
|---|---|---|
| **Live workers** | How big is the fleet right now? | `max(fleet_router_live_workers{job=~"$job"}) or vector(0)` |
| **Served** | Is it doing work? | `sum(rate(fleet_router_requests_total{job=~"$job", outcome=~"ok\|rerouted"}[$rate_window])) or vector(0)` |
| **p99 latency** | Is it fast? | `histogram_quantile(0.99, sum by (le) (rate(fleet_router_request_duration_seconds_bucket{job=~"$job"}[$rate_window])))` |
| **p99 vs SLO budget** | Fast *enough*? | the same p99 `/ $slo_p99_seconds * 100` — green < 80 %, amber to 100 %, red over budget |
| **Rerouted (range)** | Did the reroute path fire? | `sum(increase(fleet_router_requests_total{job=~"$job", outcome="rerouted"}[$__range])) or vector(0)` |
| **Client-visible errors (range)** | Did a client ever pay for it? Target: **exactly 0** | `sum(increase(fleet_router_requests_total{job=~"$job", outcome="failed"}[$__range])) or vector(0)` |

### Row 2 — FAST

| Panel | Question | Query |
|---|---|---|
| **Request rate — offered vs served** | Is served tracking offered, and where is the gap? | `sum(rate(fleet_router_requests_total{job=~"$job"}[$rate_window])) or vector(0)` · same with `outcome=~"ok\|rerouted"` · same with `outcome="failed"` |
| **Latency p50 / p95 / p99 vs SLO** | Where is the tail, and is it inside budget? | `histogram_quantile(0.50\|0.95\|0.99, sum by (le) (rate(fleet_router_request_duration_seconds_bucket{job=~"$job"}[$rate_window])))` plus `vector($slo_p99_seconds)` as the dashed SLO line |
| **Tail ratio p99 / p50** | Is the tail disciplined? Target **< 3×** | the p99 expression `/` the p50 expression |
| **Latency distribution** | What does the whole distribution look like? | `sum by (le) (rate(fleet_router_request_duration_seconds_bucket{job=~"$job"}[$rate_window]))`, heatmap format |

The SLO is defined once in `docs/inference-platform-plan.md` §5 as **p99 ≤ 3 × L0**
(L0 = unloaded single-request p50). Measure L0 first (E1), then set the
`SLO p99 (s)` variable at the top of the dashboard — the budget tile and the
dashed line both follow it, so no panel JSON needs editing.

### Row 3 — RELIABLE

| Panel | Question | Query |
|---|---|---|
| **Reroutes and client-visible errors** | The fault-tolerance story in one plot | `sum(rate(fleet_router_requests_total{job=~"$job", outcome="rerouted"}[$rate_window])) or vector(0)` · same with `outcome="failed"` |
| **Upstream failures by kind** | *How* did an attempt fail? | `sum(rate(fleet_router_upstream_attempts_total{job=~"$job", result="client_error"}[$rate_window])) or vector(0)` · same for `result="server_error"` · same for `result="error"` |
| **Fleet membership** | Which workers does the router believe in? | `max(fleet_router_live_workers{job=~"$job"}) or vector(0)` · `count(fleet_worker_queue_depth{job=~"$job"}) or vector(0)` |

`result="error"` is a **transport** failure — the connection never completed.
That is the shape a preempted spot box makes, and during a chaos run it should
spike at the moment of the kill while client-visible errors stay flat at zero.
The successful `result="ok"` series is deliberately **not** plotted: at 100:1 it
flattens the three failure lines into the axis, and successful throughput is
already the panel to its left.

Caveat carried over from
[`docs/experiments/x2-x3-results.md`](../../docs/experiments/x2-x3-results.md):
a localhost `kill -9` yields an immediate `ECONNREFUSED`, whereas a terminating
EC2 instance **black-holes packets**. The transport-error series therefore
appears faster in a local chaos test than it will in the cloud. That path is
still unverified (E7).

The two membership series answer different questions: `fleet_router_live_workers`
is what the router will *route to* (heartbeat inside TTL);
`count(fleet_worker_queue_depth)` is how many actually returned a `/stats`
scrape. A gap between them is a real, specific failure — a worker with a fresh
heartbeat whose `/stats` is timing out will still receive traffic while its
queue depth is invisible to the HPA.

### Row 4 — SCALABLE & COST

| Panel | Question | Query |
|---|---|---|
| **Queue depth by worker (HPA signal)** | Which worker is backed up right now? | `fleet_worker_queue_depth{job=~"$job"}` (instant, one bar per `worker_id`) |
| **Queue depth over time vs HPA target** | Is the fleet queueing, and is load balanced? | `max(fleet_worker_queue_depth{job=~"$job"}) or vector(0)` · `avg(...) or vector(0)` · `vector($hpa_queue_target)` |
| **Backlog — in flight vs queued** | Is the *router* or the *GPU* the bottleneck? | `sum(fleet_router_in_flight{job=~"$job"}) or vector(0)` · `sum(fleet_worker_queue_depth{job=~"$job"}) or vector(0)` |
| **Decode rate by worker** | Is any worker an outlier? | `fleet_worker_tokens_per_second{job=~"$job"}` (instant) |
| **Fleet decode throughput** | Does throughput scale with workers? | `sum(fleet_worker_tokens_per_second{job=~"$job"}) or vector(0)` |
| **$ / 1M tokens (decode-rate floor)** | What could a token cost? | `($price_per_worker_hour * max(fleet_router_live_workers{job=~"$job"}) * 1000000 / 3600) / (sum(fleet_worker_tokens_per_second{job=~"$job"}) > 0)` |

**Read the two throughput panels honestly.** `fleet_worker_tokens_per_second` is
*lifetime completion tokens ÷ lifetime generate-seconds*, as the worker reports
it on `/stats`. It measures how fast a worker decodes **while decoding**, and it
does not fall when the worker goes idle. So:

- it is a **health/outlier** signal per worker (a worker that fell back to CPU,
  or landed on a slower node, shows up immediately);
- the fleet sum is a **capacity ceiling**, not delivered throughput;
- the `$ / 1M tokens` tile is therefore a **floor** — the price if every worker
  decoded continuously for the whole hour. It is the number to beat, not the
  number to quote. Real $/1M tokens includes idle time and comes from the
  E-series runs on g5 (`docs/inference-platform-plan.md` §5). Expect the
  serialized `_gen_lock` baseline to sit ~10× off market rates; that gap is the
  argument for continuous batching.

The denominator is guarded with `> 0`, so the tile reads blank rather than
infinity when nothing is generating.

### Dashboard variables

| Variable | Default | Why it exists |
|---|---|---|
| `Data source` | `prometheus` | Matches the uid the chart's sidecar provisions. |
| `Router` (`$job`) | All | Pick one arm to compare the Go and Python routers (E3). Populated from `label_values(fleet_router_live_workers, job)` — a gauge, which both client libraries always export, so the variable never empties out (see below). |
| `Rate window` | `30s` | See below. |
| `SLO p99 (s)` | `3` | Set to 3 × your measured L0. |
| `HPA queue target` | `2` | Lives here once, so the dashed target line and any prometheus-adapter rule quote the same number. |
| `$ per worker-hour` | `1.006` | g5.xlarge on-demand. Drop it to the spot price to see the spot floor. |

**Why `$rate_window` instead of `$__rate_interval`.** Prometheus scrapes the
router every **5 s** (`servicemonitor-router.yaml`), so a rate window needs ≥ 20 s
to span four samples. But Grafana derives `$__rate_interval` from the
*datasource's* scrape interval, which the chart sets from the global
`scrapeInterval` (15 s) — giving a 60 s window. A reroute event lasts about one
TTL (15 s); a 60 s window smears it into a flat minute and the chaos timeline
stops being readable. So the window is an explicit, tunable variable rather than
a derived one.

---

## Caveat 1 — the k3s ServiceMonitor trap

**k3s (and k3d, which is k3s in Docker) runs the entire control plane as ONE
process.** `k3s server` embeds the apiserver, scheduler, controller-manager,
kube-proxy and an embedded datastore in a single binary. There is no separate
`kube-scheduler` pod, no `etcd` pod, no `kube-proxy` DaemonSet, and none of them
serve the metrics ports (`10259` / `10257` / `10249` / `2381`) that the chart's
default ServiceMonitors point at.

Left at their defaults, the chart creates four scrape jobs whose Endpoints are
empty or unreachable. The result on a perfectly healthy cluster:

- permanent **"targets down"** in the Prometheus UI;
- `KubeSchedulerDown` / `KubeControllerManagerDown` / `KubeProxyDown` /
  `etcdMembersDown` firing forever — which trains you to ignore alerts, exactly
  the wrong reflex to build *before* a chaos experiment;
- a red kube-\* dashboard that makes a working fleet look broken in a demo.

`values.yaml` disables both halves — the scrapers **and** the rule groups that
alert on them:

```yaml
kubeEtcd:              { enabled: false }
kubeScheduler:         { enabled: false }
kubeControllerManager: { enabled: false }
kubeProxy:             { enabled: false }

defaultRules:
  rules:
    etcd: false
    kubeControllerManager: false
    kubeProxy: false
    kubeSchedulerAlerting: false
    kubeSchedulerRecording: false
```

Disabling the ServiceMonitor alone is not enough: several of those rules are
written against `absent(...)` or `up{job="..."} == 0`, and an absent series is
not `0` — the rule fires anyway. Both layers have to go.

`kubeApiServer`, `kubelet` (+ cAdvisor) and `coreDns` **do** work on k3s and stay
enabled — that is where pod CPU / memory / restarts come from.

Alertmanager is also off (nobody is on call for a cluster that lives 40 minutes),
so its self-check rules and its provisioned datasource are disabled too. Rules
still evaluate and are visible in the Prometheus UI's Alerts tab; they just have
nowhere to route.

---

## Caveat 2 — absent is not zero

**Go's `client_golang` omits metric families that have no children.** A freshly
started or restarted Go router returns **nothing at all** for:

- `fleet_router_requests_total`
- `fleet_router_upstream_attempts_total`

— not zero. Python's `prometheus_client` emits the `# TYPE` header with no
samples, which reaches Prometheus as the same thing: no series. This was
confirmed, not assumed, in
[`docs/experiments/x2-x3-results.md`](../../docs/experiments/x2-x3-results.md)
("Two exposition deltas"); after traffic the Go router exports both counters
correctly, so it is a client-library convention and not a bug.

A naive panel renders **"No data"** — and "no data" and "zero errors" look
identical to a viewer. The fleet looks broken after every restart, at exactly
the moment someone is watching.

**How this dashboard handles it.** Every query over those two counters ends in
`or vector(0)`. When the family is absent, the inner aggregation returns an empty
vector and `or vector(0)` substitutes a scalar zero; when it is present, the left
side wins and the fallback is ignored. Each affected series is also given a
literal `legendFormat` (`"Served"`, `"Failed (5xx to client)"`, …) so the
zero-fill — which carries no labels — is named identically to the real series and
keeps its assigned colour instead of appearing as `{}`.

Three cases behave differently, on purpose:

| Family | Absent when… | Treatment |
|---|---|---|
| `fleet_router_requests_total`, `fleet_router_upstream_attempts_total` (CounterVec) | Go router restarted, no traffic yet | **`or vector(0)`** on every query. This is the trap. |
| `fleet_worker_queue_depth`, `fleet_worker_tokens_per_second` (GaugeVec) | Fleet is empty — the router *clears and rewrites* these on every `/stats` sweep, so a dead worker stops being exported within one sweep. **Both** client libraries go silent here. | Aggregates (`sum` / `max` / `avg` / `count`) get **`or vector(0)`**. The two per-worker bar gauges do **not**: no workers genuinely means no bars, and inventing a zero-valued bar for a worker that does not exist would be worse. |
| `fleet_router_request_duration_seconds` (plain Histogram), `fleet_router_live_workers`, `fleet_router_in_flight` (plain Gauges) | Never — non-vector collectors have themselves as a child and are always exported. | No fallback needed. The percentile panels *do* show gaps with no traffic, because `histogram_quantile` over all-zero buckets is `NaN`. That gap is honest: no observations means there is no latency to report, and zero-filling it would claim a 0 ms p99. |

That last row is also why the `$job` variable is populated from
`label_values(fleet_router_live_workers, job)` — a plain gauge — rather than from
one of the counters. Seed a variable off a family that can vanish and the whole
dashboard empties out after a router restart.

---

## Why the workers are not scraped

`src/inference/worker.py` serves `/healthz`, `/v1/models`, `/stats` and
`/v1/completions`. There is **no `/metrics`**, and `/stats` is application JSON,
not the Prometheus text exposition format. Pointing a ServiceMonitor at it
produces a permanently-DOWN target with a parse error — the same false-alarm
class as the k3s phantoms above. So `servicemonitor-workers.yaml` deliberately
creates nothing; it documents this and carries a ready-to-enable block for the
day the worker speaks Prometheus (ROADMAP Part 4 swaps in vLLM, whose OpenAI
server does expose a real `/metrics`).

Worker metrics reach Prometheus **through the router**: it polls every live
worker's `/stats` on a 2 s loop and rewrites two gauges on its own registry
(`sync_worker_gauges` in `src/inference/metrics.py`, `SyncWorkerGauges` in
`router-go/metrics.go`):

```
fleet_worker_queue_depth{worker_id="…"}        ← /stats "queued"
fleet_worker_tokens_per_second{worker_id="…"}  ← /stats "tokens_per_second"
```

One scrape target; worker identity carried in the `worker_id` label. Two
consequences worth knowing: those gauges are freshness-bounded by the router's
poll period (2 s) **plus** the scrape interval (5 s), not the scrape interval
alone; and because the router clears them every sweep, an empty fleet exports
nothing (see the table above).

Per-worker request outcomes *are* available without a worker scrape —
`fleet_router_upstream_attempts_total` carries a `worker_id` label, so "which box
returned transport errors" is answerable from the router's exposition alone.
GPU utilisation is **not** covered; that needs NVIDIA DCGM-exporter as a
DaemonSet on the g5 nodes, which is a separate deliverable and is deliberately
not faked here.

---

## Editing the dashboard

`dashboard-fleet.json` is the source of truth. `dashboard-configmap.yaml` is
generated from it.

UI edits are **not** persisted —
`grafana.sidecar.dashboards.provider.allowUiUpdates` is `false`, so the file wins
on every reconcile. Prototype in the UI if you like, then export the JSON back
into `dashboard-fleet.json`.

```bash
cd deploy/monitoring

# 1. edit dashboard-fleet.json, then check it:
python3 -m json.tool dashboard-fleet.json > /dev/null

# 2. regenerate the ConfigMap (the header comments are re-added by hand):
kubectl create configmap fleet-dashboard \
  --namespace monitoring \
  --from-file=dashboard-fleet.json \
  --dry-run=client -o yaml \
| kubectl label -f - --local --dry-run=client -o yaml \
    grafana_dashboard=1 \
    app.kubernetes.io/name=fleet-dashboard \
    app.kubernetes.io/part-of=optitrain-inference \
> dashboard-configmap.yaml

# 3. apply. The sidecar picks it up within seconds; no Grafana restart.
kubectl apply -f dashboard-configmap.yaml
```

The ConfigMap **key name matters**: the sidecar writes each data key as a file
under `/tmp/dashboards/`, and `values.yaml` points
`default_home_dashboard_path` at `/tmp/dashboards/dashboard-fleet.json`. Rename
the key and the home dashboard silently reverts to Grafana's default.

### Design notes

Panel colours are not ad hoc. Percentiles are an **ordinal** scale, so p50/p95/p99
are one hue in ordered lightness steps (`#184f95` → `#2a78d6` → `#86b6ef`), with
the lightest — most prominent on a dark surface — reserved for p99. Series that
*mean* good/bad (rerouted, failed, transport errors) wear reserved **status**
colours and never a series colour, and vice versa. Nominal categories (worker
ids) get one hue with magnitude in bar length, never a value ramp. The request
rate panel is an **emphasis** form: Served in the accent hue, Offered in the
de-emphasis grey, Failed in critical red. There are no dual-axis panels
anywhere — "Backlog: in flight vs queued" puts two series on one axis precisely
because they share a unit.

Colours were validated with a CVD/contrast checker against Grafana's dark panel
surface (`#181b1f`): the categorical pair passes all six checks (worst adjacent
ΔE 26.8 protan / 31.8 normal-vision), and the percentile ramp passes the ordinal
checks (monotone lightness, adjacent ΔL ≥ 0.06, light end 2.13:1 vs surface).
The dashboard is designed and tested for **dark** mode; `values.yaml` sets
`users.default_theme = dark` to match.

---

## Verifying this directory without a cluster

Everything here is checkable offline, and was:

```bash
# 1. the chart renders with these values
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm template optitrain prometheus-community/kube-prometheus-stack \
  --version 88.2.0 --namespace monitoring -f deploy/monitoring/values.yaml

# 2. the dashboard is valid JSON
python3 -m json.tool deploy/monitoring/dashboard-fleet.json > /dev/null

# 3. every PromQL expression parses (extract them, feed promtool a rule file)
docker run --rm --entrypoint promtool -v /tmp/promql-check.yaml:/tmp/promql-check.yaml:ro \
  prom/prometheus:v3.1.0 check rules /tmp/promql-check.yaml
```

The ServiceMonitor was additionally validated against the `ServiceMonitor` v1
`openAPIV3Schema` shipped in the chart's own CRD (`charts/crds/crds/
crd-servicemonitors.yaml`, operator v0.93.0), as was the commented-out worker
block once uncommented.
