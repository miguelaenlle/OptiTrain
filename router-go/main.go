// Command router is the fleet router: one public endpoint, N disposable
// workers behind it.
//
// It keeps a registry snapshot (polled from the heartbeat store), round-robins
// completions across live workers, and reroutes on failure: a connection
// error, timeout, or 5xx sends the request to the next live worker instead of
// the client. That retry is the spot story — a terminated worker's in-flight
// requests land somewhere else, and its stale heartbeat drops it from rotation
// within the TTL.
//
// A drop-in replacement for src/inference/router.py: same endpoints, same env
// vars, same JSON, same Prometheus metric names.
//
// Usage:
//
//	FLEET_WORKERS_URI=.fleet/local/workers go run . --port 8000
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

func main() {
	port := flag.Int("port", 0, "listen port (default $PORT, else 8000)")
	host := flag.String("host", "", "listen address (default $HOST, else 0.0.0.0)")
	flag.Parse()

	logger := log.New(os.Stdout, "", log.LstdFlags|log.Lmicroseconds)
	settings := SettingsFromEnv(logger)
	if *port > 0 {
		settings.Port = *port
	}
	if *host != "" {
		settings.Host = *host
	}

	// A signal cancels this context: the background loops stop, then the HTTP
	// server drains. Requests keep their own contexts, so in-flight
	// completions are not cut off mid-generation.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if settings.WorkersURI == "" {
		logger.Print("[router] WARNING: FLEET_WORKERS_URI unset — no workers will be found")
	}
	registry, err := NewRegistry(ctx, settings.WorkersURI)
	if err != nil {
		logger.Fatalf("[router] registry: %v", err)
	}

	router := NewServer(settings, registry, NewMetrics(), nil, nil, logger)
	background := router.StartBackground(ctx)

	srv := &http.Server{
		Addr:              net.JoinHostPort(settings.Host, strconv.Itoa(settings.Port)),
		Handler:           router.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		// No WriteTimeout: a long generation legitimately holds the response
		// open for the full REQUEST_TIMEOUT_SECONDS window (plus retries).
	}

	serveErr := make(chan error, 1)
	go func() {
		logger.Printf("[router] listening on %s (workers_uri=%q, ttl=%.0fs, max_attempts=%d)",
			srv.Addr, settings.WorkersURI, settings.TTLSeconds, settings.MaxAttempts)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
			return
		}
		serveErr <- nil
	}()

	select {
	case err := <-serveErr:
		if err != nil {
			logger.Fatalf("[router] serve: %v", err)
		}
	case <-ctx.Done():
		logger.Print("[router] signal received — draining")
	}

	stop() // a second signal now kills immediately instead of hanging the drain
	shutdownCtx, cancel := context.WithTimeout(context.Background(), settings.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Printf("[router] shutdown: %v", err)
	}
	background.Wait()
	logger.Print("[router] stopped")
}
