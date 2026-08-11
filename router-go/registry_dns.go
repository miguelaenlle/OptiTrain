package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sort"
	"time"
)

// DNSRegistry discovers workers from a Kubernetes headless Service.
//
// In-cluster the heartbeat registry is redundant and worse. A headless Service
// publishes one A record per **Ready** pod, and readiness is decided by the
// kubelet's probe -- so the control plane removes a sick pod for us, on the
// probe's schedule (~4s with our settings), instead of us waiting 15s for a
// heartbeat document to go stale.
//
// That difference matters most in the case this fleet exists to survive: a pod
// that dies abruptly cannot publish "I am gone". With heartbeats the router
// must wait out the TTL. With readiness, something *else* notices.
//
// Honest about the mechanism: this is still POLLING (we re-resolve every
// ROUTER_POLL_SECONDS), not a watch. It is not push. The win is the source of
// truth -- kubelet probes rather than self-reported liveness -- not the
// transport. A true Endpoints watch would remove the poll interval too, at the
// cost of client-go, RBAC and a ServiceAccount; DNS needs none of those and
// behaves identically in k3d, k3s and EKS.
//
// Go's resolver does not cache, so every poll reflects current readiness
// rather than a stale record.
type DNSRegistry struct {
	Host string // e.g. "fleet-worker" or "fleet-worker.default.svc.cluster.local"
	Port string // worker port, e.g. "8001"

	// LookupHost is the seam tests use; nil means the real resolver.
	LookupHost func(ctx context.Context, host string) ([]string, error)
}

// ListWorkers implements Registry.
//
// Every returned doc is stamped LastSeen=now on purpose. DNS carries no
// timestamps, and it does not need to: presence in a headless Service's records
// already means "Ready". The TTL filter in LiveWorkers is a heartbeat-era
// concept, so we make it a no-op here rather than inventing a fake age that
// would silently expire healthy workers.
func (r DNSRegistry) ListWorkers(ctx context.Context) ([]WorkerDoc, error) {
	lookup := r.LookupHost
	if lookup == nil {
		lookup = net.DefaultResolver.LookupHost
	}
	addrs, err := lookup(ctx, r.Host)
	if err != nil {
		// ONLY NXDOMAIN counts as "no workers". A headless Service with zero
		// Ready pods genuinely has no records, so that is a normal rollout
		// state, not a failure -- reporting it as an error would log on every
		// poll while a deployment rolls.
		//
		// Everything else must surface. This previously also swallowed
		// IsTemporary, which is the opposite condition: it means the RESOLVER
		// could not be reached, not that the name is absent. When cluster DNS
		// went down in P3.3 the router reported live_workers=0 with no error
		// anywhere, and a dead resolver was indistinguishable from an empty
		// fleet -- three layers between cause and symptom. Timeouts and
		// connection failures are infrastructure faults and are now returned.
		var dnsErr *net.DNSError
		if errors.As(err, &dnsErr) && dnsErr.IsNotFound {
			return nil, nil
		}
		return nil, fmt.Errorf("resolve %s: %w", r.Host, err)
	}

	// Sorted so round-robin order is stable across polls; otherwise resolver
	// shuffling would reshuffle our cursor and skew per-worker load.
	sort.Strings(addrs)

	now := float64(time.Now().UnixNano()) / 1e9
	docs := make([]WorkerDoc, 0, len(addrs))
	for _, ip := range addrs {
		docs = append(docs, WorkerDoc{
			WorkerID: ip, // pod IP: stable for the pod's lifetime, unique in-cluster
			Addr:     net.JoinHostPort(ip, r.Port),
			LastSeen: now,
		})
	}
	return docs, nil
}
