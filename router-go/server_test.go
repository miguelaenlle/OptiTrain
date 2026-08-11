package main

// HTTP surface tests — the endpoints, the JSON shapes, the Prometheus
// contract, and the background sweeps. Everything runs through httptest with
// injected stubs; no worker process and no network.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func noStats(context.Context, string) (map[string]any, error) {
	return nil, errors.New("stats scrape disabled in this test")
}

func testSettings() Settings {
	return Settings{
		MaxAttempts:       DefaultMaxAttempts,
		TTLSeconds:        DefaultTTLSeconds,
		RequestTimeout:    time.Second,
		ConnectTimeout:    time.Second,
		PollInterval:      time.Second,
		StatsPollInterval: time.Second,
		ShutdownTimeout:   time.Second,
	}
}

func newTestServer(t *testing.T, workers []WorkerDoc, p Poster) *Server {
	t.Helper()
	if p == nil {
		p = poster(func(string, []byte) (int, json.RawMessage, error) {
			return 200, json.RawMessage(`{}`), nil
		})
	}
	s := NewServer(testSettings(), DirRegistry{Dir: t.TempDir()}, NewMetrics(), p,
		noStats, log.New(io.Discard, "", 0))
	s.state.SetWorkers(workers, time.Now())
	return s
}

func do(t *testing.T, s *Server, method, path string, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req := httptest.NewRequest(method, path, reader)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

func decode(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response %q is not a JSON object: %v", rec.Body.String(), err)
	}
	return out
}

func TestCompletionsProxiesUpstreamResponseUnchanged(t *testing.T) {
	upstream := `{"id":"cmpl-1","choices":[{"text":" JULIET"}],"usage":{"completion_tokens":3}}`
	var seenBody []byte
	s := newTestServer(t, testWorkers(2), poster(func(_ string, b []byte) (int, json.RawMessage, error) {
		seenBody = b
		return 200, json.RawMessage(upstream), nil
	}))

	body := []byte(`{"prompt":"ROMEO:","max_tokens":64}`)
	rec := do(t, s, http.MethodPost, "/v1/completions", body)
	if rec.Code != 200 {
		t.Fatalf("status = %d (%s), want 200", rec.Code, rec.Body)
	}
	if rec.Body.String() != upstream {
		t.Errorf("body = %s, want the upstream JSON verbatim %s", rec.Body, upstream)
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Errorf("content-type = %q, want application/json", got)
	}
	if string(seenBody) != string(body) {
		t.Errorf("worker saw %s, want %s", seenBody, body)
	}
	if got := testutil.ToFloat64(s.metrics.Requests.WithLabelValues(OutcomeOK)); got != 1 {
		t.Errorf("fleet_router_requests_total{outcome=ok} = %v, want 1", got)
	}
}

func TestCompletionsWithNoWorkersReturns503(t *testing.T) {
	s := newTestServer(t, nil, nil)

	rec := do(t, s, http.MethodPost, "/v1/completions", []byte(`{"prompt":"x"}`))
	if rec.Code != 503 {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	if got := decode(t, rec)["detail"]; got != "no live workers" {
		t.Errorf("detail = %v, want %q", got, "no live workers")
	}
	if got := testutil.ToFloat64(s.metrics.Requests.WithLabelValues(OutcomeFailed)); got != 1 {
		t.Errorf("fleet_router_requests_total{outcome=failed} = %v, want 1", got)
	}
	// Failures are timed too, or the latency histogram silently ignores the
	// worst requests.
	if got := testutil.CollectAndCount(s.metrics.RequestDuration); got != 1 {
		t.Errorf("duration histogram series = %d, want the 503 observed", got)
	}
}

// Same table as tests/test_fleet_metrics.py::test_outcome_and_attempt_label_mapping.
func TestOutcomeAndAttemptLabelMapping(t *testing.T) {
	outcomes := []struct {
		status   int
		rerouted bool
		want     string
	}{
		{200, false, OutcomeOK},
		{200, true, OutcomeRerouted},
		{400, false, OutcomeOK}, // client's fault, we served
		{503, false, OutcomeFailed},
		{500, true, OutcomeFailed},
	}
	for _, tc := range outcomes {
		if got := OutcomeFor(tc.status, tc.rerouted); got != tc.want {
			t.Errorf("OutcomeFor(%d, %v) = %q, want %q", tc.status, tc.rerouted, got, tc.want)
		}
	}
	attempts := map[int]string{
		200: AttemptOK, 204: AttemptOK,
		400: AttemptClientError, 422: AttemptClientError,
		500: AttemptServerError, 503: AttemptServerError,
	}
	for status, want := range attempts {
		if got := AttemptResult(status); got != want {
			t.Errorf("AttemptResult(%d) = %q, want %q", status, got, want)
		}
	}
}

func TestCompletionsRerouteIsCountedAndAttributed(t *testing.T) {
	s := newTestServer(t, testWorkers(2), poster(func(addr string, _ []byte) (int, json.RawMessage, error) {
		if addr == "host1:8001" { // NextStart moves the cursor to 1 first
			return 0, nil, &UpstreamError{Err: errors.New("connection refused")}
		}
		return 200, json.RawMessage(`{"ok":true}`), nil
	}))

	if rec := do(t, s, http.MethodPost, "/v1/completions", []byte(`{"prompt":"x"}`)); rec.Code != 200 {
		t.Fatalf("status = %d (%s), want 200", rec.Code, rec.Body)
	}
	if got := testutil.ToFloat64(s.metrics.Requests.WithLabelValues(OutcomeRerouted)); got != 1 {
		t.Errorf("fleet_router_requests_total{outcome=rerouted} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(
		s.metrics.UpstreamAttempts.WithLabelValues("w1", AttemptTransport)); got != 1 {
		t.Errorf("attempts{worker_id=w1,result=error} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(
		s.metrics.UpstreamAttempts.WithLabelValues("w0", AttemptOK)); got != 1 {
		t.Errorf("attempts{worker_id=w0,result=ok} = %v, want 1", got)
	}

	status := decode(t, do(t, s, http.MethodGet, "/fleet/status", nil))
	if status["rerouted"] != float64(1) || status["requests"] != float64(1) {
		t.Errorf("/fleet/status = %v, want requests=1 rerouted=1", status)
	}
}

func TestCompletions4xxPassesThroughAsDetail(t *testing.T) {
	attempts := 0
	s := newTestServer(t, testWorkers(3), poster(func(string, []byte) (int, json.RawMessage, error) {
		attempts++
		return 400, json.RawMessage(`{"detail":"prompt contains characters outside the model vocab"}`), nil
	}))

	rec := do(t, s, http.MethodPost, "/v1/completions", []byte(`{"prompt":"☃"}`))
	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	detail, _ := decode(t, rec)["detail"].(string)
	if !strings.Contains(detail, "vocab") {
		t.Errorf("detail = %q, want the upstream message", detail)
	}
	if attempts != 1 {
		t.Errorf("attempts = %d, want 1 — a 4xx must not be retried", attempts)
	}
	// A 4xx is the client's fault, not the fleet's: the request counts as ok.
	if got := testutil.ToFloat64(s.metrics.Requests.WithLabelValues(OutcomeOK)); got != 1 {
		t.Errorf("fleet_router_requests_total{outcome=ok} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(
		s.metrics.UpstreamAttempts.WithLabelValues("w1", AttemptClientError)); got != 1 {
		t.Errorf("attempts{result=client_error} = %v, want 1", got)
	}
}

func TestCompletionsRejectsNonObjectBody(t *testing.T) {
	s := newTestServer(t, testWorkers(1), poster(func(string, []byte) (int, json.RawMessage, error) {
		t.Error("poster called for a malformed body")
		return 0, nil, nil
	}))

	for _, body := range []string{`[1,2,3]`, `not json`, ``} {
		rec := do(t, s, http.MethodPost, "/v1/completions", []byte(body))
		if rec.Code != http.StatusUnprocessableEntity {
			t.Errorf("POST %q -> %d, want 422", body, rec.Code)
		}
	}
}

func TestWrongMethodIs405(t *testing.T) {
	s := newTestServer(t, nil, nil)
	if rec := do(t, s, http.MethodGet, "/v1/completions", nil); rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET /v1/completions -> %d, want 405", rec.Code)
	}
}

func TestHealthzAndFleetStatusShapes(t *testing.T) {
	s := newTestServer(t, testWorkers(2), nil)

	health := decode(t, do(t, s, http.MethodGet, "/healthz", nil))
	if health["ok"] != true || health["live_workers"] != float64(2) {
		t.Errorf("/healthz = %v, want {ok:true, live_workers:2}", health)
	}

	status := decode(t, do(t, s, http.MethodGet, "/fleet/status", nil))
	for _, key := range []string{"live_workers", "workers", "requests", "rerouted", "failed", "last_poll"} {
		if _, ok := status[key]; !ok {
			t.Errorf("/fleet/status missing %q (have %v)", key, status)
		}
	}
	workers, ok := status["workers"].([]any)
	if !ok || len(workers) != 2 {
		t.Fatalf("/fleet/status workers = %v, want 2 docs", status["workers"])
	}
	first, _ := workers[0].(map[string]any)
	if first["worker_id"] != "w0" || first["addr"] != "host0:8001" {
		t.Errorf("worker doc = %v, want the heartbeat fields", first)
	}
}

// An empty fleet must serialize as [] — a null would break the monitor and any
// JSON consumer expecting a list.
func TestEmptyFleetSerializesAsEmptyLists(t *testing.T) {
	s := newTestServer(t, nil, nil)

	if body := do(t, s, http.MethodGet, "/fleet/status", nil).Body.String(); !strings.Contains(body, `"workers":[]`) {
		t.Errorf("/fleet/status = %s, want \"workers\":[]", body)
	}
	if body := do(t, s, http.MethodGet, "/fleet/metrics", nil).Body.String(); !strings.Contains(body, `"workers":[]`) {
		t.Errorf("/fleet/metrics = %s, want \"workers\":[]", body)
	}
}

func TestFleetMetricsShape(t *testing.T) {
	s := newTestServer(t, testWorkers(1), nil)
	s.state.SetWorkerStats([]WorkerStat{{
		"worker_id": "w0", "addr": "host0:8001", "ok": true, "queued": 2, "tokens_per_second": 41.5,
	}}, time.Now())

	body := decode(t, do(t, s, http.MethodGet, "/fleet/metrics", nil))
	router, ok := body["router"].(map[string]any)
	if !ok {
		t.Fatalf("/fleet/metrics router = %v, want an object", body["router"])
	}
	for _, key := range []string{"live_workers", "in_flight", "requests", "rerouted", "failed"} {
		if _, ok := router[key]; !ok {
			t.Errorf("/fleet/metrics router missing %q (have %v)", key, router)
		}
	}
	for _, key := range []string{"ts", "workers", "stats_ts"} {
		if _, ok := body[key]; !ok {
			t.Errorf("/fleet/metrics missing %q (have %v)", key, body)
		}
	}
	workers, _ := body["workers"].([]any)
	if len(workers) != 1 {
		t.Fatalf("workers = %v, want the last scrape", body["workers"])
	}
	if w, _ := workers[0].(map[string]any); w["queued"] != float64(2) {
		t.Errorf("worker stat = %v, want the scraped fields passed through", workers[0])
	}
}

// The exposition must carry exactly the names, labels and buckets that
// src/inference/metrics.py declares — one Grafana dashboard serves both
// routers, and the HPA rule reads fleet_worker_queue_depth.
func TestPrometheusExpositionMatchesThePythonContract(t *testing.T) {
	s := newTestServer(t, testWorkers(1), nil)
	if rec := do(t, s, http.MethodPost, "/v1/completions", []byte(`{"prompt":"x"}`)); rec.Code != 200 {
		t.Fatalf("warm-up request failed: %d %s", rec.Code, rec.Body)
	}
	s.metrics.SetLiveWorkers(1)
	s.metrics.SyncWorkerGauges([]WorkerStat{{
		"worker_id": "w0", "ok": true, "queued": 3.0, "tokens_per_second": 12.5,
	}})

	body := do(t, s, http.MethodGet, "/metrics", nil).Body.String()
	want := []string{
		`fleet_router_requests_total{outcome="ok"} 1`,
		`fleet_router_upstream_attempts_total{result="ok",worker_id="w0"} 1`,
		`fleet_router_request_duration_seconds_bucket{le="0.05"}`,
		`fleet_router_request_duration_seconds_bucket{le="60"}`,
		`fleet_router_request_duration_seconds_bucket{le="+Inf"}`,
		`fleet_router_request_duration_seconds_count 1`,
		`fleet_router_live_workers 1`,
		`fleet_router_in_flight 0`,
		`fleet_worker_queue_depth{worker_id="w0"} 3`,
		`fleet_worker_tokens_per_second{worker_id="w0"} 12.5`,
	}
	for _, line := range want {
		if !strings.Contains(body, line) {
			t.Errorf("/metrics missing %q", line)
		}
	}
	// Every declared bucket boundary, in the Python order.
	for _, b := range []string{"0.05", "0.1", "0.25", "0.5", "1", "2", "3", "5", "8", "12", "20", "30", "60"} {
		if !strings.Contains(body, `le="`+b+`"`) {
			t.Errorf("/metrics missing histogram bucket le=%q", b)
		}
	}
	// A private registry: no go_* / process_* noise that Python does not export.
	if strings.Contains(body, "go_goroutines") || strings.Contains(body, "process_cpu") {
		t.Error("/metrics exports default collectors; the Python router does not")
	}
}

// A worker that vanished between sweeps must stop being exported, or a stuck
// queue_depth keeps an HPA scaled up forever.
func TestSyncWorkerGaugesDropsGoneAndUnhealthyWorkers(t *testing.T) {
	m := NewMetrics()
	m.SyncWorkerGauges([]WorkerStat{
		{"worker_id": "w0", "ok": true, "queued": 4.0, "tokens_per_second": 10.0},
		{"worker_id": "w1", "ok": true, "queued": 1.0, "tokens_per_second": 11.0},
	})
	if got := testutil.CollectAndCount(m.WorkerQueueDepth); got != 2 {
		t.Fatalf("queue_depth series = %d, want 2", got)
	}

	m.SyncWorkerGauges([]WorkerStat{
		{"worker_id": "w0", "ok": true, "queued": 0.0, "tokens_per_second": 9.0},
		{"worker_id": "w1", "ok": false, "error": "connection refused"},
		{"worker_id": "", "ok": true, "queued": 5.0},
		{"worker_id": "w3", "ok": true}, // answered, but reported no queue depth
	})
	if got := testutil.CollectAndCount(m.WorkerQueueDepth); got != 1 {
		t.Errorf("queue_depth series = %d, want 1 (dead worker cleared)", got)
	}
	// A zero must still be exported: "the queue drained" is a real reading.
	if got := testutil.ToFloat64(m.WorkerQueueDepth.WithLabelValues("w0")); got != 0 {
		t.Errorf("w0 queue_depth = %v, want 0", got)
	}
}

// A worker that errors still appears, with ok=false — the monitor shows it
// dying rather than silently dropping it.
func TestScrapeWorkerStatsKeepsFailedWorkers(t *testing.T) {
	get := func(_ context.Context, addr string) (map[string]any, error) {
		if addr == "host1:8001" {
			return nil, errors.New("dial tcp 10.0.0.2:8001: connect: connection refused")
		}
		return map[string]any{"worker_id": "w0", "queued": 1.0, "tokens_per_second": 33.0}, nil
	}

	stats := ScrapeWorkerStats(context.Background(), testWorkers(2), get)
	if len(stats) != 2 {
		t.Fatalf("stats = %v, want one entry per worker", stats)
	}
	if stats[0]["ok"] != true || stats[0]["queued"] != 1.0 {
		t.Errorf("healthy stat = %v, want ok=true plus the scraped fields", stats[0])
	}
	if stats[1]["worker_id"] != "w1" || stats[1]["ok"] != false {
		t.Errorf("failed stat = %v, want the worker present with ok=false", stats[1])
	}
	if msg, _ := stats[1]["error"].(string); !strings.Contains(msg, "connection refused") {
		t.Errorf("failed stat error = %q, want the transport failure", msg)
	}
}

func TestScrapeWorkerStatsTruncatesLongErrors(t *testing.T) {
	get := func(context.Context, string) (map[string]any, error) {
		return nil, errors.New(strings.Repeat("x", 500))
	}
	stats := ScrapeWorkerStats(context.Background(), testWorkers(1), get)
	if msg, _ := stats[0]["error"].(string); len(msg) != 120 {
		t.Errorf("error length = %d, want it clipped to 120", len(msg))
	}
}

type stubRegistry struct {
	docs []WorkerDoc
	err  error
}

func (s stubRegistry) ListWorkers(context.Context) ([]WorkerDoc, error) { return s.docs, s.err }

func TestPollOnceFiltersStaleAndSurvivesRegistryFailure(t *testing.T) {
	now := float64(time.Now().UnixNano()) / 1e9
	reg := &stubRegistry{docs: []WorkerDoc{
		{WorkerID: "fresh", Addr: "a:1", LastSeen: now},
		{WorkerID: "stale", Addr: "b:1", LastSeen: now - 3600},
		{WorkerID: "no-addr", LastSeen: now},
	}}
	s := NewServer(testSettings(), reg, NewMetrics(), nil, noStats, log.New(io.Discard, "", 0))

	s.PollOnce(context.Background())
	if got := ids(s.state.Workers()); len(got) != 1 || got[0] != "fresh" {
		t.Fatalf("workers = %v, want [fresh]", got)
	}
	if got := testutil.ToFloat64(s.metrics.LiveWorkers); got != 1 {
		t.Errorf("fleet_router_live_workers = %v, want 1", got)
	}

	// A flaky store must not blank the fleet.
	reg.err = errors.New("AccessDenied")
	s.PollOnce(context.Background())
	if got := ids(s.state.Workers()); len(got) != 1 || got[0] != "fresh" {
		t.Errorf("workers = %v after a failed poll, want the previous snapshot kept", got)
	}
}

func TestNextStartIsRoundRobinAndConcurrencySafe(t *testing.T) {
	s := &State{}
	for i, want := range []int{1, 2, 0, 1, 2, 0} {
		if got := s.NextStart(3); got != want {
			t.Fatalf("NextStart #%d = %d, want %d", i, got, want)
		}
	}
	if got := s.NextStart(0); got != 0 {
		t.Errorf("NextStart(0) = %d, want 0 (no divide by zero on an empty fleet)", got)
	}

	// Run under -race: the cursor and the counters are shared by every request.
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				s.NextStart(4)
				s.Enter()
				s.Record(RouteResult{StatusCode: 200, Rerouted: j%2 == 0})
				s.Leave()
				_ = s.Counters()
				_ = s.Workers()
			}
		}()
	}
	wg.Wait()
	if c := s.Counters(); c.Requests != 6400 || c.Rerouted != 3200 || c.InFlight != 0 {
		t.Errorf("counters = %+v, want requests=6400 rerouted=3200 in_flight=0", c)
	}
}

func TestSettingsFromEnv(t *testing.T) {
	logger := log.New(io.Discard, "", 0)
	// Empty is treated as unset, so this isolates the test from the ambient
	// environment (and t.Setenv restores it).
	for _, key := range []string{
		"HOST", "PORT", "FLEET_WORKERS_URI", "ROUTER_POLL_SECONDS", "WORKER_TTL_SECONDS",
		"REQUEST_TIMEOUT_SECONDS", "ROUTER_CONNECT_TIMEOUT_SECONDS", "ROUTER_MAX_ATTEMPTS",
		"ROUTER_STATS_POLL_SECONDS", "ROUTER_UPSTREAM_KEEPALIVE", "ROUTER_SHUTDOWN_TIMEOUT_SECONDS",
	} {
		t.Setenv(key, "")
	}

	def := SettingsFromEnv(logger)
	if def.Host != "0.0.0.0" || def.Port != 8000 {
		t.Errorf("default listen = %s:%d, want 0.0.0.0:8000", def.Host, def.Port)
	}
	if def.PollInterval != 3*time.Second || def.StatsPollInterval != 2*time.Second {
		t.Errorf("poll intervals = %v/%v, want 3s/2s", def.PollInterval, def.StatsPollInterval)
	}
	if def.RequestTimeout != 60*time.Second || def.ConnectTimeout != 3*time.Second {
		t.Errorf("timeouts = %v/%v, want 60s read / 3s connect",
			def.RequestTimeout, def.ConnectTimeout)
	}
	if def.TTLSeconds != 15 || def.MaxAttempts != 3 || def.UpstreamKeepAlive {
		t.Errorf("ttl=%v attempts=%d keepalive=%v, want 15/3/false",
			def.TTLSeconds, def.MaxAttempts, def.UpstreamKeepAlive)
	}

	t.Setenv("HOST", "127.0.0.1")
	t.Setenv("PORT", "9000")
	t.Setenv("FLEET_WORKERS_URI", "s3://bucket/fleet/f1/workers")
	t.Setenv("ROUTER_POLL_SECONDS", "0.5")
	t.Setenv("WORKER_TTL_SECONDS", "30")
	t.Setenv("REQUEST_TIMEOUT_SECONDS", "120")
	t.Setenv("ROUTER_CONNECT_TIMEOUT_SECONDS", "1")
	t.Setenv("ROUTER_MAX_ATTEMPTS", "5")
	t.Setenv("ROUTER_STATS_POLL_SECONDS", "4")
	t.Setenv("ROUTER_UPSTREAM_KEEPALIVE", "true")

	got := SettingsFromEnv(logger)
	want := Settings{
		Host: "127.0.0.1", Port: 9000, WorkersURI: "s3://bucket/fleet/f1/workers",
		PollInterval: 500 * time.Millisecond, TTLSeconds: 30,
		RequestTimeout: 120 * time.Second, ConnectTimeout: time.Second,
		MaxAttempts: 5, StatsPollInterval: 4 * time.Second,
		UpstreamKeepAlive: true, ShutdownTimeout: 10 * time.Second,
	}
	if got != want {
		t.Errorf("settings = %+v, want %+v", got, want)
	}

	// A garbage value falls back to the default instead of killing the router.
	t.Setenv("ROUTER_MAX_ATTEMPTS", "banana")
	if SettingsFromEnv(logger).MaxAttempts != DefaultMaxAttempts {
		t.Error("a non-numeric ROUTER_MAX_ATTEMPTS should fall back to the default")
	}
}

// The whole stack over a real socket: registry dir -> poll -> route -> upstream.
func TestEndToEndOverRealSockets(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/completions":
			body, _ := io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"echo":` + string(body) + `}`))
		case "/stats":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"worker_id":"live-1","queued":0,"tokens_per_second":50.0}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer worker.Close()

	dir := t.TempDir()
	addr := strings.TrimPrefix(worker.URL, "http://")
	writeFile(t, dir, "live-1.json", heartbeat(t, "live-1", addr, float64(time.Now().Unix())))
	writeFile(t, dir, "dead-1.json", heartbeat(t, "dead-1", "127.0.0.1:1", float64(time.Now().Unix())))

	settings := testSettings()
	settings.WorkersURI = dir
	s := NewServer(settings, DirRegistry{Dir: dir}, NewMetrics(), nil, nil, log.New(io.Discard, "", 0))
	s.PollOnce(context.Background())
	if got := len(s.state.Workers()); got != 2 {
		t.Fatalf("live workers = %d, want 2", got)
	}

	// 127.0.0.1:1 refuses instantly, so whichever worker the cursor picks
	// first, the request still lands on the live one.
	for i := 0; i < 2; i++ {
		rec := do(t, s, http.MethodPost, "/v1/completions", []byte(`{"prompt":"x"}`))
		if rec.Code != 200 {
			t.Fatalf("request %d: status = %d (%s), want 200", i, rec.Code, rec.Body)
		}
		if want := `{"echo":{"prompt":"x"}}`; rec.Body.String() != want {
			t.Errorf("body = %s, want %s", rec.Body, want)
		}
	}

	s.ScrapeOnce(context.Background())
	stats := s.state.WorkerStats()
	if len(stats) != 2 {
		t.Fatalf("stats = %v, want one entry per worker", stats)
	}
	byID := map[string]WorkerStat{}
	for _, st := range stats {
		id, _ := st["worker_id"].(string)
		byID[id] = st
	}
	if byID["live-1"]["ok"] != true || byID["live-1"]["tokens_per_second"] != 50.0 {
		t.Errorf("live worker stat = %v, want the scrape merged in", byID["live-1"])
	}
	if byID["dead-1"]["ok"] != false {
		t.Errorf("dead worker stat = %v, want ok=false", byID["dead-1"])
	}
	if got := testutil.CollectAndCount(s.metrics.WorkerQueueDepth); got != 1 {
		t.Errorf("queue_depth series = %d, want only the live worker", got)
	}
}
