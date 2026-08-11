package main

// HTTP surface + background loops — everything around the pure policy in
// route.go. Endpoint set, JSON shapes, env vars and defaults all match
// src/inference/router.py so this binary is a drop-in swap for it.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Settings is RouterSettings: same env vars, same defaults.
type Settings struct {
	Host              string        // HOST
	Port              int           // PORT
	WorkersURI        string        // FLEET_WORKERS_URI
	PollInterval      time.Duration // ROUTER_POLL_SECONDS
	TTLSeconds        float64       // WORKER_TTL_SECONDS
	RequestTimeout    time.Duration // REQUEST_TIMEOUT_SECONDS (per upstream attempt)
	ConnectTimeout    time.Duration // ROUTER_CONNECT_TIMEOUT_SECONDS
	MaxAttempts       int           // ROUTER_MAX_ATTEMPTS
	StatsPollInterval time.Duration // ROUTER_STATS_POLL_SECONDS

	// UpstreamKeepAlive (ROUTER_UPSTREAM_KEEPALIVE, default off) pools upstream
	// connections. Off by default for parity with the Python router, which
	// dials fresh every request — that is what makes ConnectTimeout the real
	// bound on reroute latency when a terminating box black-holes packets. A
	// pooled connection to a dead box has no SYN to time out, so it would hang
	// for the full RequestTimeout instead. Turn it on for pure throughput runs
	// (E6) where the fleet is stable.
	UpstreamKeepAlive bool

	ShutdownTimeout time.Duration // grace period for in-flight completions
}

// StatsPollTimeout bounds one worker /stats scrape (requests.get(timeout=1.5)).
const StatsPollTimeout = 1500 * time.Millisecond

// SettingsFromEnv mirrors RouterSettings.from_env.
func SettingsFromEnv(logger *log.Logger) Settings {
	return Settings{
		Host:              envString("HOST", "0.0.0.0"),
		Port:              envInt(logger, "PORT", 8000),
		WorkersURI:        envString("FLEET_WORKERS_URI", ""),
		PollInterval:      envSeconds(logger, "ROUTER_POLL_SECONDS", 3),
		TTLSeconds:        envFloat(logger, "WORKER_TTL_SECONDS", DefaultTTLSeconds),
		RequestTimeout:    envSeconds(logger, "REQUEST_TIMEOUT_SECONDS", 60),
		ConnectTimeout:    envSeconds(logger, "ROUTER_CONNECT_TIMEOUT_SECONDS", 3),
		MaxAttempts:       envInt(logger, "ROUTER_MAX_ATTEMPTS", DefaultMaxAttempts),
		StatsPollInterval: envSeconds(logger, "ROUTER_STATS_POLL_SECONDS", 2),
		UpstreamKeepAlive: envBool(logger, "ROUTER_UPSTREAM_KEEPALIVE", false),
		ShutdownTimeout:   envSeconds(logger, "ROUTER_SHUTDOWN_TIMEOUT_SECONDS", 10),
	}
}

// StatsGetter fetches http://<addr>/stats. Injected so the monitoring sweep is
// testable without sockets, like the Poster.
type StatsGetter func(ctx context.Context, addr string) (map[string]any, error)

// Server wires the registry, the policy, the counters and the metrics together.
type Server struct {
	settings Settings
	state    *State
	registry Registry
	metrics  *Metrics
	poster   Poster
	stats    StatsGetter
	logger   *log.Logger
}

// NewServer builds a router. poster and stats may be nil, in which case real
// HTTP clients are built from the settings; tests pass fakes.
func NewServer(
	settings Settings,
	registry Registry,
	metrics *Metrics,
	poster Poster,
	stats StatsGetter,
	logger *log.Logger,
) *Server {
	if logger == nil {
		logger = log.New(os.Stdout, "", log.LstdFlags)
	}
	if metrics == nil {
		metrics = NewMetrics()
	}
	if poster == nil || stats == nil {
		upstream := newUpstreamClient(settings)
		if poster == nil {
			poster = &httpPoster{client: upstream, timeout: settings.RequestTimeout}
		}
		if stats == nil {
			stats = httpStatsGetter(upstream)
		}
	}
	return &Server{
		settings: settings,
		state:    &State{},
		registry: registry,
		metrics:  metrics,
		poster:   poster,
		stats:    stats,
		logger:   logger,
	}
}

// Handler returns the router's mux. Go 1.22 method patterns give the same
// 405-on-wrong-method behavior FastAPI has.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /fleet/status", s.handleFleetStatus)
	mux.HandleFunc("GET /fleet/metrics", s.handleFleetMetrics)
	mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	mux.Handle("GET /metrics", promhttp.HandlerFor(s.metrics.Registry, promhttp.HandlerOpts{}))
	return mux
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":           true,
		"live_workers": len(s.state.Workers()),
	})
}

func (s *Server) handleFleetStatus(w http.ResponseWriter, _ *http.Request) {
	c := s.state.Counters()
	writeJSON(w, http.StatusOK, map[string]any{
		"live_workers": c.LiveWorkers,
		"workers":      nonNilWorkers(s.state.Workers()),
		"requests":     c.Requests,
		"rerouted":     c.Rerouted,
		"failed":       c.Failed,
		"last_poll":    c.LastPoll,
	})
}

// handleFleetMetrics is the live aggregate the fleet monitor polls: router
// counters plus the latest per-worker /stats scrape.
func (s *Server) handleFleetMetrics(w http.ResponseWriter, _ *http.Request) {
	c := s.state.Counters()
	writeJSON(w, http.StatusOK, map[string]any{
		"ts": unixSeconds(time.Now()),
		"router": map[string]any{
			"live_workers": c.LiveWorkers,
			"in_flight":    c.InFlight,
			"requests":     c.Requests,
			"rerouted":     c.Rerouted,
			"failed":       c.Failed,
		},
		"workers":  nonNilStats(s.state.WorkerStats()),
		"stats_ts": c.StatsTS,
	})
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeDetail(w, http.StatusBadRequest, "could not read request body")
		return
	}
	// FastAPI types the body as `dict`; anything else is a 422 there too.
	if !isJSONObject(body) {
		writeDetail(w, http.StatusUnprocessableEntity, "body must be a JSON object")
		return
	}

	workers := s.state.Workers()
	start := 0
	if len(workers) > 0 {
		start = s.state.NextStart(len(workers))
	}

	s.state.Enter()
	s.metrics.InFlight.Inc()
	defer func() { // Python's try/finally: the gauge must never leak
		s.state.Leave()
		s.metrics.InFlight.Dec()
	}()

	started := time.Now()
	result := RouteCompletion(r.Context(), body, workers, s.instrumented(workers), RoutePolicy{
		StartIndex:  start,
		MaxAttempts: s.settings.MaxAttempts,
	})
	s.state.Record(result)
	s.metrics.RecordRequest(OutcomeFor(result.StatusCode, result.Rerouted), time.Since(started))

	if result.StatusCode >= 400 {
		// FastAPI raises HTTPException for both the 503 and the passed-through
		// 4xx, so the client always sees {"detail": ...} on an error.
		writeDetail(w, result.StatusCode, result.Detail)
		return
	}
	// A 2xx upstream is always relayed as 200 with the body unchanged — the
	// Python endpoint returns the payload and lets FastAPI stamp 200.
	writeRaw(w, http.StatusOK, result.Response)
}

// instrumented wraps the injected poster with per-attempt metrics. Wrapping
// here (rather than inside RouteCompletion) is what keeps the policy pure: it
// never sees the metrics registry.
func (s *Server) instrumented(workers []WorkerDoc) Poster {
	byAddr := make(map[string]string, len(workers))
	for _, wk := range workers {
		byAddr[wk.Addr] = wk.WorkerID
	}
	return PosterFunc(func(ctx context.Context, addr string, body []byte) (int, json.RawMessage, error) {
		workerID := byAddr[addr]
		if workerID == "" {
			workerID = addr
		}
		status, payload, err := s.poster.Post(ctx, addr, body)
		if err != nil {
			s.metrics.RecordAttempt(workerID, AttemptTransport)
			return status, payload, err
		}
		s.metrics.RecordAttempt(workerID, AttemptResult(status))
		return status, payload, nil
	})
}

// StartBackground launches the registry poll and the /stats scrape. Both run
// once immediately, then on their ticker, and stop when ctx is cancelled.
func (s *Server) StartBackground(ctx context.Context) *sync.WaitGroup {
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		s.loop(ctx, s.settings.PollInterval, s.PollOnce)
	}()
	go func() {
		defer wg.Done()
		s.loop(ctx, s.settings.StatsPollInterval, s.ScrapeOnce)
	}()
	return &wg
}

func (s *Server) loop(ctx context.Context, every time.Duration, tick func(context.Context)) {
	if every <= 0 {
		every = time.Second
	}
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		tick(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// PollOnce refreshes the registry snapshot. A failed poll keeps the previous
// snapshot: a flaky store must not blank the fleet.
func (s *Server) PollOnce(ctx context.Context) {
	docs, err := s.registry.ListWorkers(ctx)
	if err != nil {
		if ctx.Err() == nil {
			s.logger.Printf("[router] registry poll failed: %v", err)
		}
		return
	}
	live := LiveWorkers(docs, s.settings.TTLSeconds, time.Now())
	s.state.SetWorkers(live, time.Now())
	s.metrics.SetLiveWorkers(len(live))
}

// ScrapeOnce refreshes the per-worker /stats view and the gauges built from it.
func (s *Server) ScrapeOnce(ctx context.Context) {
	stats := ScrapeWorkerStats(ctx, s.state.Workers(), s.stats)
	s.state.SetWorkerStats(stats, time.Now())
	s.metrics.SyncWorkerGauges(stats)
}

// ScrapeWorkerStats runs one monitoring sweep. A worker that errors still
// appears, with ok=false — the monitor shows it dying rather than silently
// dropping it.
//
// The scrapes run concurrently (Python's are serial); output order still
// matches the worker order, so the result is deterministic.
func ScrapeWorkerStats(ctx context.Context, workers []WorkerDoc, get StatsGetter) []WorkerStat {
	out := make([]WorkerStat, len(workers))
	var wg sync.WaitGroup
	for i := range workers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			wk := workers[i]
			doc := WorkerStat{"worker_id": wk.WorkerID, "addr": wk.Addr, "ok": true}
			fields, err := get(ctx, wk.Addr)
			if err != nil {
				doc["ok"] = false
				doc["error"] = truncate(err.Error(), 120)
			} else {
				for k, v := range fields {
					doc[k] = v
				}
			}
			out[i] = doc
		}(i)
	}
	wg.Wait()
	return out
}

// newUpstreamClient builds the client the router dials workers with. The
// connect timeout lives on the dialer; the read window is a per-attempt
// context deadline in httpPoster.
func newUpstreamClient(s Settings) *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   s.ConnectTimeout,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			DisableKeepAlives:     !s.UpstreamKeepAlive,
			MaxIdleConns:          256,
			MaxIdleConnsPerHost:   64,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   s.ConnectTimeout,
			ExpectContinueTimeout: time.Second,
		},
		// No client-level Timeout: the deadline is per attempt, so a retry
		// after a connect failure gets a full read window of its own — exactly
		// what requests' (connect, read) tuple does.
	}
}

type httpPoster struct {
	client  *http.Client
	timeout time.Duration
}

// Post implements Poster: one POST to http://<addr>/v1/completions.
func (p *httpPoster) Post(ctx context.Context, addr string, body []byte) (int, json.RawMessage, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, "http://"+addr+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return 0, nil, &UpstreamError{Err: err}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return 0, nil, &UpstreamError{Err: err}
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, &UpstreamError{Err: err} // truncated body == transport failure
	}
	if !json.Valid(payload) {
		payload = detailPayload(truncate(string(payload), 200))
	}
	return resp.StatusCode, payload, nil
}

func httpStatsGetter(client *http.Client) StatsGetter {
	return func(ctx context.Context, addr string) (map[string]any, error) {
		ctx, cancel := context.WithTimeout(ctx, StatsPollTimeout)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+addr+"/stats", nil)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return nil, fmt.Errorf("stats %d: %s", resp.StatusCode, truncate(string(body), 80))
		}
		var out map[string]any
		if err := json.Unmarshal(body, &out); err != nil {
			return nil, err
		}
		return out, nil
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	payload, err := json.Marshal(v)
	if err != nil {
		payload = detailPayload("response encoding failed")
		status = http.StatusInternalServerError
	}
	writeRaw(w, status, payload)
}

func writeRaw(w http.ResponseWriter, status int, payload []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(payload)
}

func writeDetail(w http.ResponseWriter, status int, detail string) {
	writeRaw(w, status, detailPayload(detail))
}

func detailPayload(detail string) []byte {
	payload, err := json.Marshal(map[string]string{"detail": detail})
	if err != nil { // unreachable: a string always marshals
		return []byte(`{"detail":""}`)
	}
	return payload
}

// isJSONObject reports whether body decodes as a JSON object, the only body
// shape the Python endpoint accepts.
func isJSONObject(body []byte) bool {
	var probe map[string]json.RawMessage
	return json.Unmarshal(body, &probe) == nil && probe != nil
}

// truncate cuts to n runes (Python slices str by character, not byte).
func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// nonNilWorkers keeps /fleet/status emitting [] rather than null when empty.
func nonNilWorkers(docs []WorkerDoc) []WorkerDoc {
	if docs == nil {
		return []WorkerDoc{}
	}
	return docs
}

func nonNilStats(stats []WorkerStat) []WorkerStat {
	if stats == nil {
		return []WorkerStat{}
	}
	return stats
}

func envString(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func envFloat(logger *log.Logger, key string, def float64) float64 {
	raw, ok := os.LookupEnv(key)
	if !ok || raw == "" {
		return def
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		logger.Printf("[router] %s=%q is not a number — using %v", key, raw, def)
		return def
	}
	return v
}

func envSeconds(logger *log.Logger, key string, def float64) time.Duration {
	return time.Duration(envFloat(logger, key, def) * float64(time.Second))
}

func envInt(logger *log.Logger, key string, def int) int {
	raw, ok := os.LookupEnv(key)
	if !ok || raw == "" {
		return def
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		logger.Printf("[router] %s=%q is not an integer — using %d", key, raw, def)
		return def
	}
	return v
}

func envBool(logger *log.Logger, key string, def bool) bool {
	raw, ok := os.LookupEnv(key)
	if !ok || raw == "" {
		return def
	}
	v, err := strconv.ParseBool(raw)
	if err != nil {
		logger.Printf("[router] %s=%q is not a boolean — using %v", key, raw, def)
		return def
	}
	return v
}
