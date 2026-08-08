// Command stubworker is a zero-latency stand-in for an inference worker.
//
// X3 measures what the ROUTER costs per request. A real worker takes hundreds
// of milliseconds to generate, which would bury a router's 1-3ms of overhead
// under model noise -- the two routers would measure as identical no matter
// what they did. This stub answers /v1/completions immediately with a fixed,
// correctly-shaped payload, so essentially all observed latency is the router.
//
// It is written in Go, not Python, for the same reason: a python http.server
// would saturate long before either router does, and we would end up measuring
// the stub.
//
// It heartbeats into the same registry directory a real worker uses, so the
// router discovers it through the normal path with no special-casing.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"time"
)

var (
	requests atomic.Int64
	tokens   atomic.Int64
)

// completionBody is the OpenAI-shaped response the real worker emits. The
// token counts are fixed: loadgen parses usage.completion_tokens, so they must
// be present and non-zero for its tokens/s accounting to work.
func completionBody(workerID string) []byte {
	b, _ := json.Marshal(map[string]any{
		"id":        "cmpl-stub",
		"object":    "text_completion",
		"created":   time.Now().Unix(),
		"model":     "stub",
		"worker_id": workerID,
		"choices": []map[string]any{{
			"text":          " stub completion",
			"index":         0,
			"logprobs":      nil,
			"finish_reason": "length",
		}},
		"usage": map[string]any{
			"prompt_tokens":     5,
			"completion_tokens": 8,
			"total_tokens":      13,
		},
	})
	return b
}

func envInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envStr(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

func main() {
	// Env fallbacks so the same binary works under Kubernetes, which sets
	// configuration through the environment rather than argv.
	port := flag.Int("port", envInt("PORT", 9001), "listen port")
	host := flag.String("host", envStr("STUB_HOST", "0.0.0.0"), "bind address")
	workerID := flag.String("worker-id", "", "worker id (default stub-<port>)")
	registry := flag.String("registry", "", "heartbeat directory")
	flag.Parse()

	id := *workerID
	if id == "" {
		id = fmt.Sprintf("stub-%d", *port)
	}
	// Bind ALL interfaces, not loopback. Inside a container 127.0.0.1 is
	// reachable only from the pod itself, so the kubelet's probe and the router
	// both get "connection refused" while the process looks perfectly healthy
	// from within. That is what this originally did, and it made the stub
	// silently unusable in Kubernetes.
	addr := fmt.Sprintf("%s:%d", *host, *port)
	body := completionBody(id)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		tokens.Add(8)
		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"ok":true,"worker_id":%q,"model":"stub"}`, id)
	})
	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"worker_id": id, "model": "stub", "device": "cpu",
			"requests": requests.Load(), "prompt_tokens": 0,
			"completion_tokens": tokens.Load(), "generate_seconds": 0.0,
			"tokens_per_second": 0.0, "in_flight": 0, "queued": 0,
		})
	})

	if *registry != "" {
		os.MkdirAll(*registry, 0o755)
		go func() {
			path := filepath.Join(*registry, id+".json")
			for {
				doc, _ := json.Marshal(map[string]any{
					"worker_id": id, "addr": addr, "model": "stub",
					"market": "local", "requests_served": requests.Load(),
					"last_seen": float64(time.Now().UnixNano()) / 1e9,
				})
				// Write-then-rename: the router lists this directory
				// concurrently and must never observe a torn document.
				tmp := path + ".tmp"
				if os.WriteFile(tmp, doc, 0o644) == nil {
					os.Rename(tmp, path)
				}
				time.Sleep(2 * time.Second)
			}
		}()
	}

	srv := &http.Server{Addr: addr, Handler: mux}
	log.Printf("[stub] %s listening on %s", id, addr)
	log.Fatal(srv.ListenAndServe())
}
