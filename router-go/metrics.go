package main

// Prometheus metrics — the SAME contract as src/inference/metrics.py.
//
// Metric names, label names and histogram buckets are copied verbatim from the
// Python router so one Grafana dashboard (and one prometheus-adapter HPA rule
// on fleet_worker_queue_depth) serves both implementations, and so the Go-vs-
// Python A/B (experiment E3) compares like with like.
//
// Everything registers into a private registry rather than the process-global
// default: exactly the declared series are exported, matching Python's private
// CollectorRegistry. No go_* / process_* collectors — they would show up as a
// diff against the Python router's scrape.

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// DurationBuckets mirrors metrics.DURATION_BUCKETS. The prometheus defaults
// spend 9 of 14 buckets below 250ms — a band a token-generating LLM request
// never lands in — so these start at 50ms, resolve the 0.5–5s band where
// completions actually live, and top out at 60s to match the default
// REQUEST_TIMEOUT_SECONDS. (+Inf is implicit in client_golang; Python spells
// it out.)
var DurationBuckets = []float64{0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 60.0}

// Request outcomes — the `outcome` label on fleet_router_requests_total.
const (
	OutcomeOK       = "ok"       // served on the first attempt
	OutcomeRerouted = "rerouted" // served, but only after a retry — the spot story
	OutcomeFailed   = "failed"   // 5xx back to the client (no worker answered)
)

// Per-attempt results — the `result` label on fleet_router_upstream_attempts_total.
const (
	AttemptOK          = "ok"
	AttemptClientError = "client_error"
	AttemptServerError = "server_error"
	AttemptTransport   = "error" // transport failure: the shape a preempted box makes
)

// Metrics owns the registry and every collector the router exports.
type Metrics struct {
	Registry *prometheus.Registry

	Requests              *prometheus.CounterVec
	UpstreamAttempts      *prometheus.CounterVec
	RequestDuration       prometheus.Histogram
	LiveWorkers           prometheus.Gauge
	InFlight              prometheus.Gauge
	WorkerQueueDepth      *prometheus.GaugeVec
	WorkerTokensPerSecond *prometheus.GaugeVec
}

// NewMetrics declares every metric in one place, as metrics.py does.
func NewMetrics() *Metrics {
	m := &Metrics{
		Registry: prometheus.NewRegistry(),
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "fleet_router_requests_total",
			Help: "Completion requests the router finished, by outcome.",
		}, []string{"outcome"}),
		UpstreamAttempts: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "fleet_router_upstream_attempts_total",
			Help: "Individual worker attempts the router made, by worker and result.",
		}, []string{"worker_id", "result"}),
		RequestDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "fleet_router_request_duration_seconds",
			Help:    "End-to-end router latency for a completion, retries included.",
			Buckets: DurationBuckets,
		}),
		LiveWorkers: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "fleet_router_live_workers",
			Help: "Workers whose heartbeat is inside the TTL.",
		}),
		InFlight: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "fleet_router_in_flight",
			Help: "Completions the router is proxying right now.",
		}),
		WorkerQueueDepth: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "fleet_worker_queue_depth",
			Help: "Requests waiting behind the generation lock on a worker (HPA signal).",
		}, []string{"worker_id"}),
		WorkerTokensPerSecond: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "fleet_worker_tokens_per_second",
			Help: "Lifetime completion tokens / generate seconds, as the worker reports it.",
		}, []string{"worker_id"}),
	}
	m.Registry.MustRegister(
		m.Requests,
		m.UpstreamAttempts,
		m.RequestDuration,
		m.LiveWorkers,
		m.InFlight,
		m.WorkerQueueDepth,
		m.WorkerTokensPerSecond,
	)
	return m
}

// RecordRequest counts one finished completion and its end-to-end latency.
func (m *Metrics) RecordRequest(outcome string, d time.Duration) {
	m.Requests.WithLabelValues(outcome).Inc()
	m.RequestDuration.Observe(d.Seconds())
}

// RecordAttempt counts one router→worker call.
func (m *Metrics) RecordAttempt(workerID, result string) {
	if workerID == "" {
		workerID = "unknown"
	}
	m.UpstreamAttempts.WithLabelValues(workerID, result).Inc()
}

// SetLiveWorkers publishes the registry snapshot size.
func (m *Metrics) SetLiveWorkers(n int) {
	m.LiveWorkers.Set(float64(n))
}

// SyncWorkerGauges rewrites the per-worker gauges from a /stats sweep.
//
// Clear-then-set, so a worker that vanished between sweeps stops being exported
// instead of freezing at its last value (a stuck queue_depth would otherwise
// keep an HPA scaled up forever). Workers whose scrape failed (ok=false) are
// dropped for the same reason: we have no fresh reading.
func (m *Metrics) SyncWorkerGauges(stats []WorkerStat) {
	m.WorkerQueueDepth.Reset()
	m.WorkerTokensPerSecond.Reset()
	for _, doc := range stats {
		workerID, _ := doc["worker_id"].(string)
		if workerID == "" || !statOK(doc) {
			continue
		}
		if v, ok := statFloat(doc, "queued"); ok {
			m.WorkerQueueDepth.WithLabelValues(workerID).Set(v)
		}
		if v, ok := statFloat(doc, "tokens_per_second"); ok {
			m.WorkerTokensPerSecond.WithLabelValues(workerID).Set(v)
		}
	}
}

// OutcomeFor maps a RouteResult onto the `outcome` label.
//
// A 4xx counts as "ok": the router did its job and a worker answered — the
// request itself was bad. It still shows up as result="client_error" on the
// per-attempt counter, so it is never invisible.
func OutcomeFor(statusCode int, rerouted bool) string {
	if statusCode >= 500 {
		return OutcomeFailed
	}
	if rerouted {
		return OutcomeRerouted
	}
	return OutcomeOK
}

// AttemptResult is the low-cardinality label for one upstream attempt's status.
func AttemptResult(statusCode int) string {
	switch {
	case statusCode >= 200 && statusCode < 300:
		return AttemptOK
	case statusCode >= 400 && statusCode < 500:
		return AttemptClientError
	default:
		return AttemptServerError
	}
}

// statOK reads the "ok" flag, defaulting to true when absent (Python's
// doc.get("ok", True)).
func statOK(doc WorkerStat) bool {
	v, present := doc["ok"]
	if !present {
		return true
	}
	b, isBool := v.(bool)
	return !isBool || b
}

// statFloat reads a numeric field out of a decoded JSON payload. encoding/json
// hands back float64, but tests (and any future typed producer) may use ints.
func statFloat(doc WorkerStat, key string) (float64, bool) {
	switch v := doc[key].(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	default:
		return 0, false
	}
}
