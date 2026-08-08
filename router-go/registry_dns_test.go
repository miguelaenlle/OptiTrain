package main

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"
)

func lookupReturning(addrs []string, err error) func(context.Context, string) ([]string, error) {
	return func(context.Context, string) ([]string, error) { return addrs, err }
}

func TestDNSRegistryMapsEachReadyPodToAWorker(t *testing.T) {
	r := DNSRegistry{
		Host:       "fleet-worker",
		Port:       "8001",
		LookupHost: lookupReturning([]string{"10.42.0.7", "10.42.0.5"}, nil),
	}
	docs, err := r.ListWorkers(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("want 2 workers, got %d", len(docs))
	}
	// Sorted, so the round-robin cursor is not reshuffled by resolver ordering.
	if docs[0].Addr != "10.42.0.5:8001" || docs[1].Addr != "10.42.0.7:8001" {
		t.Fatalf("addrs not sorted/joined as expected: %q %q", docs[0].Addr, docs[1].Addr)
	}
	if docs[0].WorkerID != "10.42.0.5" {
		t.Fatalf("worker id should be the pod IP, got %q", docs[0].WorkerID)
	}
}

func TestDNSRegistryDocsAlwaysSurviveTheTTLFilter(t *testing.T) {
	// DNS carries no timestamps, and it does not need to: a headless Service
	// only publishes Ready pods, so the kubelet already did the liveness
	// filtering. If LastSeen were left zero, LiveWorkers would expire every
	// healthy worker immediately and the fleet would look empty.
	r := DNSRegistry{
		Host:       "fleet-worker",
		Port:       "8001",
		LookupHost: lookupReturning([]string{"10.42.0.5"}, nil),
	}
	docs, err := r.ListWorkers(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	live := LiveWorkers(docs, 15, time.Now())
	if len(live) != 1 {
		t.Fatalf("DNS-sourced worker was filtered out by the TTL: %+v", docs[0])
	}
}

func TestDNSRegistryTreatsNXDOMAINAsZeroWorkers(t *testing.T) {
	// A headless Service with no Ready pods has no records at all. That is a
	// normal rollout state, not a failure: report no workers (router answers
	// 503) instead of erroring on every poll.
	r := DNSRegistry{
		Host:       "fleet-worker",
		Port:       "8001",
		LookupHost: lookupReturning(nil, &net.DNSError{Err: "no such host", IsNotFound: true}),
	}
	docs, err := r.ListWorkers(context.Background())
	if err != nil {
		t.Fatalf("NXDOMAIN should not be an error, got %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("want 0 workers, got %d", len(docs))
	}
}

func TestDNSRegistrySurfacesRealResolverFailures(t *testing.T) {
	r := DNSRegistry{
		Host:       "fleet-worker",
		Port:       "8001",
		LookupHost: lookupReturning(nil, errors.New("connection refused to nameserver")),
	}
	if _, err := r.ListWorkers(context.Background()); err == nil {
		t.Fatal("a genuine resolver failure must not be silently swallowed")
	}
}

func TestParseDNSURI(t *testing.T) {
	cases := []struct {
		uri, host, port string
		ok              bool
	}{
		{"dns://fleet-worker:8001", "fleet-worker", "8001", true},
		{"dns://fleet-worker.default.svc.cluster.local:8001", "fleet-worker.default.svc.cluster.local", "8001", true},
		{"dns://fleet-worker", "", "", false}, // no port: nothing to dial
		{"dns://:8001", "", "", false},
		{"s3://bucket/prefix", "", "", false},
		{"/var/run/workers", "", "", false},
		{"", "", "", false},
	}
	for _, c := range cases {
		host, port, ok := parseDNSURI(c.uri)
		if ok != c.ok || host != c.host || port != c.port {
			t.Errorf("parseDNSURI(%q) = (%q,%q,%v), want (%q,%q,%v)",
				c.uri, host, port, ok, c.host, c.port, c.ok)
		}
	}
}

func TestNewRegistryPicksDNSBackend(t *testing.T) {
	reg, err := NewRegistry(context.Background(), "dns://fleet-worker:8001")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	dns, ok := reg.(DNSRegistry)
	if !ok {
		t.Fatalf("want DNSRegistry, got %T", reg)
	}
	if dns.Host != "fleet-worker" || dns.Port != "8001" {
		t.Fatalf("bad wiring: %+v", dns)
	}
}

func TestNewRegistryStillPicksDirForPlainPaths(t *testing.T) {
	// Guard against the dns:// case accidentally swallowing local-mode URIs.
	reg, err := NewRegistry(context.Background(), "/var/run/workers")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := reg.(DirRegistry); !ok {
		t.Fatalf("want DirRegistry, got %T", reg)
	}
}
