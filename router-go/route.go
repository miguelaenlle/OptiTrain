package main

// Routing policy — the pure core of the router.
//
// Ported 1:1 from src/inference/router.py::route_completion. The upstream call
// is INJECTED (a Poster), so the policy is exercised by the tests with no
// sockets, no goroutines, and no clock. Everything else in this program is
// plumbing around this file.

import (
	"context"
	"encoding/json"
	"fmt"
)

// DefaultMaxAttempts mirrors ROUTER_MAX_ATTEMPTS' default (and Python's
// route_completion(max_attempts=3) keyword default).
const DefaultMaxAttempts = 3

// UpstreamError marks a retryable transport failure — the shape a preempted
// spot box makes: connection refused, or a SYN into a black hole that times
// out. RouteCompletion treats ANY non-nil error from Poster.Post as retryable
// (Python only catches UpstreamError, but its _post raises nothing else); this
// type exists so the failure detail reads the same in logs and in the 503 body.
type UpstreamError struct{ Err error }

func (e *UpstreamError) Error() string { return e.Err.Error() }

func (e *UpstreamError) Unwrap() error { return e.Err }

// Poster performs one router→worker POST /v1/completions.
//
// A non-nil error is a retryable transport failure. Otherwise the HTTP status
// and the decoded upstream body are returned verbatim — the policy decides
// what they mean.
type Poster interface {
	Post(ctx context.Context, addr string, body []byte) (int, json.RawMessage, error)
}

// PosterFunc adapts a plain function to Poster.
type PosterFunc func(ctx context.Context, addr string, body []byte) (int, json.RawMessage, error)

// Post implements Poster.
func (f PosterFunc) Post(ctx context.Context, addr string, body []byte) (int, json.RawMessage, error) {
	return f(ctx, addr, body)
}

// RoutePolicy is the knob set route_completion took as keyword arguments.
type RoutePolicy struct {
	StartIndex  int // round-robin cursor: which live worker to try first
	MaxAttempts int // <= 0 means DefaultMaxAttempts
}

// RouteResult mirrors the Python dataclass of the same name.
type RouteResult struct {
	// Response is the upstream JSON on success (and on a passed-through 4xx);
	// nil when no worker answered.
	Response   json.RawMessage
	StatusCode int // 200 on success; client 4xx passed through; 503 if exhausted
	Detail     string
	Attempts   int
	Rerouted   bool   // succeeded on a retry — the headline counter
	WorkerID   string // who actually served it
}

// RouteCompletion tries live workers in round-robin order until one answers.
//
// Retryable: transport errors (post returns an error) and 5xx. NOT retryable:
// 4xx — that's the client's request, and every worker would reject it the same
// way, so burning the retry budget on it only makes the fleet look unhealthy.
func RouteCompletion(
	ctx context.Context,
	body []byte,
	workers []WorkerDoc,
	post Poster,
	policy RoutePolicy,
) RouteResult {
	if len(workers) == 0 {
		return RouteResult{StatusCode: 503, Detail: "no live workers"}
	}
	maxAttempts := policy.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = DefaultMaxAttempts
	}
	start := policy.StartIndex
	if start < 0 {
		start = 0
	}

	attempts := 0
	lastDetail := ""
	for k := 0; k < len(workers); k++ {
		if attempts >= maxAttempts {
			break
		}
		w := workers[(start+k)%len(workers)]
		attempts++

		status, payload, err := post.Post(ctx, w.Addr, body)
		if err != nil {
			lastDetail = fmt.Sprintf("%s: %v", w.WorkerID, err)
			if ctx.Err() != nil {
				// The client hung up (or its deadline passed). Retrying would
				// only fan one abandoned request across the whole fleet and
				// blame healthy workers for it. Note the per-attempt read
				// timeout lives on a CHILD context, so it does not land here:
				// a slow worker is still rerouted.
				break
			}
			continue
		}
		switch {
		case status >= 200 && status < 300:
			return RouteResult{
				Response:   payload,
				StatusCode: status,
				Attempts:   attempts,
				Rerouted:   attempts > 1,
				WorkerID:   w.WorkerID,
			}
		case status >= 400 && status < 500:
			return RouteResult{
				Response:   payload,
				StatusCode: status,
				Detail:     detailOf(payload),
				Attempts:   attempts,
			}
		default:
			lastDetail = fmt.Sprintf("%s: upstream %d", w.WorkerID, status)
		}
	}
	return RouteResult{
		StatusCode: 503,
		Detail:     fmt.Sprintf("all attempts failed (%s)", lastDetail),
		Attempts:   attempts,
	}
}

// detailOf pulls the "detail" field out of an upstream error body, mirroring
// Python's `payload.get("detail", "") if isinstance(payload, dict) else ""`.
// A non-string detail (FastAPI allows any JSON there) is passed through as its
// JSON text rather than dropped.
func detailOf(payload json.RawMessage) string {
	var obj map[string]json.RawMessage
	if json.Unmarshal(payload, &obj) != nil {
		return "" // not a JSON object
	}
	raw, ok := obj["detail"]
	if !ok {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return string(raw)
}
