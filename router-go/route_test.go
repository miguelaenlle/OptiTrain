package main

// Routing policy tests — the same table as tests/test_fleet_router.py.
//
// Retry-on-failure is the spot-preemption story, so it is verified without
// sockets by injecting the upstream call.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func testWorkers(n int) []WorkerDoc {
	docs := make([]WorkerDoc, n)
	for i := range docs {
		docs[i] = WorkerDoc{WorkerID: fmt.Sprintf("w%d", i), Addr: fmt.Sprintf("host%d:8001", i)}
	}
	return docs
}

// poster adapts a socket-free stub to the Poster interface.
func poster(fn func(addr string, body []byte) (int, json.RawMessage, error)) Poster {
	return PosterFunc(func(_ context.Context, addr string, body []byte) (int, json.RawMessage, error) {
		return fn(addr, body)
	})
}

func TestSuccessFirstTry(t *testing.T) {
	p := poster(func(addr string, _ []byte) (int, json.RawMessage, error) {
		return 200, json.RawMessage(fmt.Sprintf(`{"choices":[{"text":"ok"}],"from":%q}`, addr)), nil
	})

	r := RouteCompletion(context.Background(), []byte(`{"prompt":"x"}`), testWorkers(2), p, RoutePolicy{})
	if r.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", r.StatusCode)
	}
	if r.Attempts != 1 {
		t.Errorf("attempts = %d, want 1", r.Attempts)
	}
	if r.Rerouted {
		t.Error("rerouted = true, want false on a first-try success")
	}
}

// A dead worker's request lands on the next one — the headline behavior.
func TestReroutesOnConnectionError(t *testing.T) {
	var calls []string
	p := poster(func(addr string, _ []byte) (int, json.RawMessage, error) {
		calls = append(calls, addr)
		if addr == "host0:8001" {
			return 0, nil, &UpstreamError{Err: errors.New("connection refused")}
		}
		return 200, json.RawMessage(`{"choices":[{"text":"ok"}]}`), nil
	})

	r := RouteCompletion(context.Background(), []byte(`{"prompt":"x"}`), testWorkers(2), p,
		RoutePolicy{StartIndex: 0, MaxAttempts: 3})
	if r.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", r.StatusCode)
	}
	if !r.Rerouted {
		t.Error("rerouted = false, want true")
	}
	if r.Attempts != 2 {
		t.Errorf("attempts = %d, want 2", r.Attempts)
	}
	if got, want := strings.Join(calls, ","), "host0:8001,host1:8001"; got != want {
		t.Errorf("calls = %s, want %s", got, want)
	}
}

func TestReroutesOn5xx(t *testing.T) {
	p := poster(func(addr string, _ []byte) (int, json.RawMessage, error) {
		if addr == "host0:8001" {
			return 500, json.RawMessage(`{}`), nil
		}
		return 200, json.RawMessage(`{"choices":[]}`), nil
	})

	r := RouteCompletion(context.Background(), []byte(`{"prompt":"x"}`), testWorkers(2), p, RoutePolicy{})
	if r.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", r.StatusCode)
	}
	if !r.Rerouted {
		t.Error("rerouted = false, want true")
	}
}

// A bad request would fail identically everywhere — don't burn workers.
func Test4xxPassesThroughWithoutRetry(t *testing.T) {
	var calls []string
	p := poster(func(addr string, _ []byte) (int, json.RawMessage, error) {
		calls = append(calls, addr)
		return 400, json.RawMessage(
			`{"detail":"prompt contains characters outside the model vocab"}`), nil
	})

	r := RouteCompletion(context.Background(), []byte(`{"prompt":"☃"}`), testWorkers(3), p, RoutePolicy{})
	if r.StatusCode != 400 {
		t.Fatalf("status = %d, want 400", r.StatusCode)
	}
	if !strings.Contains(r.Detail, "vocab") {
		t.Errorf("detail = %q, want it to carry the upstream message", r.Detail)
	}
	if len(calls) != 1 {
		t.Errorf("made %d attempts (%v), want exactly 1 — a 4xx must not retry", len(calls), calls)
	}
	if r.Rerouted {
		t.Error("rerouted = true, want false")
	}
}

func TestAllDeadReturns503WithBoundedAttempts(t *testing.T) {
	p := poster(func(string, []byte) (int, json.RawMessage, error) {
		return 0, nil, &UpstreamError{Err: errors.New("boom")}
	})

	r := RouteCompletion(context.Background(), []byte(`{"prompt":"x"}`), testWorkers(5), p,
		RoutePolicy{MaxAttempts: 3})
	if r.StatusCode != 503 {
		t.Fatalf("status = %d, want 503", r.StatusCode)
	}
	if r.Attempts != 3 {
		t.Errorf("attempts = %d, want 3 (bounded by max_attempts, not worker count)", r.Attempts)
	}
	if !strings.Contains(r.Detail, "boom") {
		t.Errorf("detail = %q, want the last upstream failure in it", r.Detail)
	}
}

func TestNoWorkersIs503(t *testing.T) {
	p := poster(func(string, []byte) (int, json.RawMessage, error) {
		t.Fatal("poster called with no live workers")
		return 0, nil, nil
	})

	r := RouteCompletion(context.Background(), []byte(`{"prompt":"x"}`), nil, p, RoutePolicy{})
	if r.StatusCode != 503 {
		t.Fatalf("status = %d, want 503", r.StatusCode)
	}
	if r.Attempts != 0 {
		t.Errorf("attempts = %d, want 0", r.Attempts)
	}
	if r.Detail != "no live workers" {
		t.Errorf("detail = %q, want %q", r.Detail, "no live workers")
	}
}

func TestRoundRobinStartIndexSpreadsLoad(t *testing.T) {
	var seen []string
	p := poster(func(addr string, _ []byte) (int, json.RawMessage, error) {
		seen = append(seen, addr)
		return 200, json.RawMessage(`{}`), nil
	})

	workers := testWorkers(3)
	for i := 0; i < 3; i++ {
		RouteCompletion(context.Background(), []byte(`{}`), workers, p, RoutePolicy{StartIndex: i})
	}
	if got, want := strings.Join(seen, ","), "host0:8001,host1:8001,host2:8001"; got != want {
		t.Errorf("served by %s, want %s", got, want)
	}
}

func TestResultRecordsServingWorker(t *testing.T) {
	p := poster(func(string, []byte) (int, json.RawMessage, error) {
		return 200, json.RawMessage(`{}`), nil
	})

	r := RouteCompletion(context.Background(), []byte(`{}`), testWorkers(2), p, RoutePolicy{StartIndex: 1})
	if r.WorkerID != "w1" {
		t.Errorf("worker_id = %q, want %q", r.WorkerID, "w1")
	}
}

// The cursor may point at the last worker; the retry must wrap to the first.
func TestRerouteWrapsAroundTheWorkerList(t *testing.T) {
	var seen []string
	p := poster(func(addr string, _ []byte) (int, json.RawMessage, error) {
		seen = append(seen, addr)
		if addr == "host2:8001" {
			return 503, json.RawMessage(`{}`), nil
		}
		return 200, json.RawMessage(`{}`), nil
	})

	r := RouteCompletion(context.Background(), []byte(`{}`), testWorkers(3), p, RoutePolicy{StartIndex: 2})
	if r.StatusCode != 200 || r.WorkerID != "w0" {
		t.Fatalf("status=%d worker=%q, want 200 served by w0", r.StatusCode, r.WorkerID)
	}
	if got, want := strings.Join(seen, ","), "host2:8001,host0:8001"; got != want {
		t.Errorf("tried %s, want %s", got, want)
	}
}

// No worker is tried twice: with 2 workers and a budget of 3, only 2 attempts
// are possible.
func TestAttemptsNeverExceedWorkerCount(t *testing.T) {
	attempts := 0
	p := poster(func(string, []byte) (int, json.RawMessage, error) {
		attempts++
		return 0, nil, &UpstreamError{Err: errors.New("dead")}
	})

	r := RouteCompletion(context.Background(), []byte(`{}`), testWorkers(2), p, RoutePolicy{MaxAttempts: 3})
	if r.StatusCode != 503 {
		t.Fatalf("status = %d, want 503", r.StatusCode)
	}
	if attempts != 2 || r.Attempts != 2 {
		t.Errorf("attempts = %d (result says %d), want 2", attempts, r.Attempts)
	}
}

// A client that hung up must not have its request fanned across the fleet:
// every worker would "fail" it, blaming healthy boxes for the client leaving.
func TestCancelledContextStopsRetrying(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	attempts := 0
	p := poster(func(string, []byte) (int, json.RawMessage, error) {
		attempts++
		cancel() // the client disconnects mid-attempt
		return 0, nil, &UpstreamError{Err: context.Canceled}
	})

	r := RouteCompletion(ctx, []byte(`{}`), testWorkers(5), p, RoutePolicy{MaxAttempts: 3})
	if attempts != 1 {
		t.Errorf("attempts = %d, want 1 — a cancelled request must not be retried", attempts)
	}
	if r.StatusCode != 503 {
		t.Errorf("status = %d, want 503", r.StatusCode)
	}
}

func TestMaxAttemptsDefaultsToThree(t *testing.T) {
	attempts := 0
	p := poster(func(string, []byte) (int, json.RawMessage, error) {
		attempts++
		return 500, json.RawMessage(`{}`), nil
	})

	RouteCompletion(context.Background(), []byte(`{}`), testWorkers(9), p, RoutePolicy{})
	if attempts != DefaultMaxAttempts {
		t.Errorf("attempts = %d, want the default %d", attempts, DefaultMaxAttempts)
	}
}

// The policy proxies the body through untouched — no re-encoding, no field
// filtering. Whatever the client sent is what the worker sees.
func TestBodyReachesTheWorkerUnchanged(t *testing.T) {
	body := []byte(`{"prompt":"ROMEO:","max_tokens":64,"nested":{"a":[1,2,3]}}`)
	var got []byte
	p := poster(func(_ string, b []byte) (int, json.RawMessage, error) {
		got = b
		return 200, json.RawMessage(`{"ok":true}`), nil
	})

	r := RouteCompletion(context.Background(), body, testWorkers(1), p, RoutePolicy{})
	if string(got) != string(body) {
		t.Errorf("worker saw %s, want %s", got, body)
	}
	if string(r.Response) != `{"ok":true}` {
		t.Errorf("response = %s, want the upstream JSON verbatim", r.Response)
	}
}

func TestDetailOf(t *testing.T) {
	cases := []struct {
		name    string
		payload string
		want    string
	}{
		{"string detail", `{"detail":"bad prompt"}`, "bad prompt"},
		{"missing detail", `{"choices":[]}`, ""},
		{"not an object", `[1,2,3]`, ""},
		{"not json", `<html>502</html>`, ""},
		{"structured detail", `{"detail":[{"msg":"field required"}]}`, `[{"msg":"field required"}]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := detailOf(json.RawMessage(tc.payload)); got != tc.want {
				t.Errorf("detailOf(%s) = %q, want %q", tc.payload, got, tc.want)
			}
		})
	}
}
