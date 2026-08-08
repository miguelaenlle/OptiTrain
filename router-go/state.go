package main

// Shared router state — the registry snapshot, the round-robin cursor, and the
// counters /fleet/status reports. Port of router.py::RouterState.
//
// One RWMutex guards the lot: reads (every request takes a worker snapshot)
// vastly outnumber writes (one registry poll every 3s), and the cursor bump is
// the only hot write.

import (
	"sync"
	"time"
)

// WorkerStat is one /stats scrape: worker_id/addr/ok plus every field the
// worker reported. A map, not a struct, because the worker's payload grows
// (gpu_util, gpu_mem_used_mb, ...) and /fleet/metrics must pass it through
// unchanged.
type WorkerStat map[string]any

// State is the router's mutable state. The zero value is ready to use.
type State struct {
	mu          sync.RWMutex
	workers     []WorkerDoc
	workerStats []WorkerStat
	rr          int
	requests    int
	rerouted    int
	failed      int
	inFlight    int
	lastPoll    float64 // unix seconds; 0 until the first successful poll
	statsTS     float64
}

// SetWorkers installs a fresh registry snapshot.
func (s *State) SetWorkers(docs []WorkerDoc, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workers = docs
	s.lastPoll = unixSeconds(now)
}

// Workers returns the current snapshot. The slice is never mutated in place
// (SetWorkers swaps a new one in), so callers may hold it for the whole
// request without copying.
func (s *State) Workers() []WorkerDoc {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.workers
}

// SetWorkerStats installs the latest /stats sweep.
func (s *State) SetWorkerStats(stats []WorkerStat, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workerStats = stats
	s.statsTS = unixSeconds(now)
}

// WorkerStats returns the latest /stats sweep.
func (s *State) WorkerStats() []WorkerStat {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.workerStats
}

// Enter marks a completion as in flight.
func (s *State) Enter() {
	s.mu.Lock()
	s.inFlight++
	s.mu.Unlock()
}

// Leave marks a completion as finished.
func (s *State) Leave() {
	s.mu.Lock()
	s.inFlight--
	s.mu.Unlock()
}

// NextStart advances the round-robin cursor and returns the index of the
// worker to try first. Same arithmetic as RouterState.next_start.
func (s *State) NextStart(n int) int {
	if n < 1 {
		n = 1
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rr = (s.rr + 1) % n
	return s.rr
}

// Record folds a finished request into the counters.
func (s *State) Record(r RouteResult) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests++
	if r.Rerouted {
		s.rerouted++
	}
	if r.StatusCode >= 500 {
		s.failed++
	}
}

// Counters is an atomic read of everything the status endpoints report.
type Counters struct {
	LiveWorkers int
	InFlight    int
	Requests    int
	Rerouted    int
	Failed      int
	LastPoll    float64
	StatsTS     float64
}

// Counters returns a consistent snapshot of the counters.
func (s *State) Counters() Counters {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return Counters{
		LiveWorkers: len(s.workers),
		InFlight:    s.inFlight,
		Requests:    s.requests,
		Rerouted:    s.rerouted,
		Failed:      s.failed,
		LastPoll:    s.lastPoll,
		StatsTS:     s.statsTS,
	}
}

func unixSeconds(t time.Time) float64 {
	return float64(t.UnixNano()) / 1e9
}
