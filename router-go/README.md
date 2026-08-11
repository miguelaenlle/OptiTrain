# router-go — the fleet router in Go

A **drop-in replacement** for `src/inference/router.py`. Same endpoints, same
env vars, same default port (8000), same request/response JSON, same Prometheus
metric names — so it can be swapped under identical load for experiment **E3**
(Go vs Python router A/B) and served by the same Grafana dashboard.

```
go build ./...        # build
go vet ./...          # vet
go test ./... -race   # policy table + registry + HTTP surface, no sockets needed
FLEET_WORKERS_URI=.fleet/local/workers go run . --port 8000
```

## Layout

| File | What it is |
|---|---|
| `route.go` | **The pure routing policy** — `RouteCompletion`, a 1:1 port of `route_completion`. No I/O, no clock, no metrics. |
| `route_test.go` | The same policy table as `tests/test_fleet_router.py`, plus wrap-around, budget, and body-passthrough cases. |
| `registry.go` | `Registry` interface + `DirRegistry` (local JSON dir) and `S3Registry` (aws-sdk-go-v2 behind an `S3API` interface), TTL filtering. |
| `registry_test.go` | Dir + TTL + torn-doc cases, and the S3 lister driven by an in-memory fake (never a real bucket). |
| `state.go` | Concurrency-safe shared state: worker snapshot, round-robin cursor, counters. |
| `metrics.go` | Every Prometheus metric, declared in one place — names/labels/buckets copied from `src/inference/metrics.py`. |
| `server.go` | Settings-from-env, HTTP handlers, the two background loops, and the real HTTP poster/scraper. |
| `server_test.go` | Endpoints, JSON shapes, exposition contract, sweeps, cursor under `-race`, and one end-to-end run over real sockets. |
| `main.go` | Flags, wiring, graceful shutdown. |

## Why the policy is a pure function

`RouteCompletion(ctx, body, workers, post, policy) RouteResult` takes the
"do one upstream POST" behavior as an injected `Poster` (an interface, with a
`PosterFunc` adapter). It therefore has **no** network, no goroutines, no
`time.Now()`, and no metrics registry — the retry policy is a table test.

Per-attempt metrics are added by wrapping the poster (`Server.instrumented`),
exactly as the Python router wraps `_post` with `_instrumented_post`, so the
policy never learns that metrics exist.

## Behavior

- `POST /v1/completions` — round-robin over live workers; the request body is
  proxied through byte-for-byte and the upstream response is returned unchanged.
- **Retry policy:** transport error or 5xx → next worker. **4xx passes straight
  through with no retry** (every worker would reject it identically). Bounded by
  `ROUTER_MAX_ATTEMPTS` and by the number of live workers. Exhausted, or no
  workers at all → `503 {"detail": ...}`.
- **Timeout split:** connect is `net.Dialer.Timeout`
  (`ROUTER_CONNECT_TIMEOUT_SECONDS`, 3s); the read window is a
  `context.WithTimeout` **per attempt** (`REQUEST_TIMEOUT_SECONDS`, 60s) — same
  shape as `requests`' `(connect, read)` tuple, so a retry after a connect
  failure still gets a full read window. A terminating box black-holes packets
  rather than sending RST, so the short connect timeout is what bounds reroute
  latency.
- `GET /healthz`, `GET /fleet/status`, `GET /fleet/metrics` — same JSON as Python.
- `GET /metrics` — Prometheus exposition from a private registry.
- Two background goroutines on tickers: registry poll (`ROUTER_POLL_SECONDS`)
  and per-worker `/stats` scrape (`ROUTER_STATS_POLL_SECONDS`). A worker whose
  scrape fails still appears with `ok:false` — the monitor shows it dying
  instead of silently dropping it.
- SIGINT/SIGTERM stops the loops, then `http.Server.Shutdown` drains in-flight
  completions (a request's own context is not cancelled by the signal).

## Environment

| Var | Default | Meaning |
|---|---|---|
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Listen address (`--host` / `--port` win) |
| `FLEET_WORKERS_URI` | *(empty → warns)* | Heartbeat prefix: `s3://bucket/fleet/<id>/workers` or a local directory |
| `ROUTER_POLL_SECONDS` | `3` | Registry poll cadence |
| `WORKER_TTL_SECONDS` | `15` | Heartbeat freshness window |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Read window per upstream attempt |
| `ROUTER_CONNECT_TIMEOUT_SECONDS` | `3` | Dial timeout per upstream attempt |
| `ROUTER_MAX_ATTEMPTS` | `3` | Retry budget per client request |
| `ROUTER_STATS_POLL_SECONDS` | `2` | `/stats` scrape cadence |
| `ROUTER_UPSTREAM_KEEPALIVE` | `false` | Go-only. Pool upstream connections (see below) |
| `ROUTER_SHUTDOWN_TIMEOUT_SECONDS` | `10` | Drain grace period |

Unparseable values log a warning and fall back to the default rather than
killing the router (Python would raise at startup).

### `ROUTER_UPSTREAM_KEEPALIVE`

Off by default, matching Python: `requests.post` dials fresh every call, which
is what makes the 3s connect timeout the real bound on reroute latency. A
**pooled** connection to a box that has just been terminated has no SYN to time
out, so the attempt would hang for the full 60s read window instead of
rerouting in 3s. Turn it on for pure-throughput runs against a stable fleet
(E6); leave it off for anything that measures recovery.

## Metric contract

Identical to `src/inference/metrics.py` — same names, labels, help text and
bucket boundaries:

| Metric | Type | Labels |
|---|---|---|
| `fleet_router_requests_total` | counter | `outcome` = `ok` \| `rerouted` \| `failed` |
| `fleet_router_upstream_attempts_total` | counter | `worker_id`, `result` = `ok` \| `client_error` \| `server_error` \| `error` |
| `fleet_router_request_duration_seconds` | histogram | — (buckets `0.05 0.1 0.25 0.5 1 2 3 5 8 12 20 30 60 +Inf`) |
| `fleet_router_live_workers` | gauge | — |
| `fleet_router_in_flight` | gauge | — |
| `fleet_worker_queue_depth` | gauge | `worker_id` (the HPA signal) |
| `fleet_worker_tokens_per_second` | gauge | `worker_id` |

A 4xx counts as `outcome="ok"` (the router did its job; the request was bad) and
as `result="client_error"` per attempt, so it is never invisible. Per-worker
gauges are cleared and rewritten on every scrape, so a terminated worker stops
being exported within one interval instead of pinning an HPA at its last value.

Everything registers into a **private** `prometheus.Registry`, so `/metrics`
carries exactly these series — no `go_*` / `process_*` collectors that the
Python router does not export.

## Known differences from the Python router

All deliberate; none change the client contract.

1. **`_created` series.** Python's `prometheus_client` also emits
   `*_created` gauges (an OpenMetrics artifact); `client_golang` does not.
2. **Bucket label text.** Python writes `le="1.0"`, Go writes `le="1"`.
   Prometheus normalizes `le` on scrape, so both land on the same series.
3. **JSON key order.** Go marshals maps with sorted keys; Python preserves
   insertion order. Heartbeat documents are re-emitted **byte-for-byte** as the
   worker wrote them (including any extra fields), so `/fleet/status` worker
   entries are unchanged.
4. **Malformed body.** FastAPI's `body: dict` rejects a non-object body with a
   422 and a structured `detail`; this returns 422 with
   `{"detail": "body must be a JSON object"}`.
5. **`/stats` scrapes run concurrently** (Python's are serial). Output order
   still follows worker order, so `/fleet/metrics` is deterministic — but a
   64-worker sweep no longer takes 64 × the per-worker timeout.
6. **A non-JSON upstream body** becomes `{"detail": "<first 200 chars>"}`, the
   same fallback Python applies when `r.json()` raises — truncated by rune
   rather than by byte.
7. **Client disconnect stops the retry loop.** If the caller hangs up, the
   request context is cancelled and the router stops instead of fanning one
   abandoned request across every worker (which would blame healthy boxes and
   burn GPU). The per-attempt read timeout is a *child* context, so a slow
   worker is still rerouted normally. Python's sync handler never notices the
   disconnect and keeps retrying.

## Not yet done (ROADMAP Part 3)

- K8s Endpoints watch as a third `Registry` implementation (the interface is
  the seam).
- Dockerfile (multi-stage, distroless) and `deploy/` manifests.
