"""Prometheus metrics for the fleet router — every metric declared in one place.

Split out of ``router.py`` so the names/labels/buckets are reviewable on their
own and so a future Go router can be held to the same contract (same metric
names ⇒ the same Grafana dashboards and the same HPA rules keep working).

``prometheus_client`` is an OPTIONAL dependency. If it is missing this module
hands back no-op stand-ins: the router still serves traffic, ``/metrics``
returns a plain-text explanation instead of an exposition payload, and nothing
else in the process has to know. ``PROMETHEUS_AVAILABLE`` says which path is
live.

Metrics register into a private ``CollectorRegistry`` rather than the
process-global default. The router is a single process and nothing else here
exports, so the isolation costs nothing — and it makes this module safe to
reload (re-registering the same name in the global default registry raises).

Cardinality: only ``worker_id`` is unbounded-ish, and fleet size is O(10). The
per-worker gauges are cleared and rewritten on every stats scrape, so a
terminated spot worker stops being exported within one scrape interval instead
of lingering forever at its last value.
"""

from __future__ import annotations

# Bucket boundaries (seconds) for router-observed completion latency. The
# prometheus defaults spend 9 of their 14 buckets below 250ms — the range a
# token-generating LLM request essentially never lands in — and give only
# 2.5/5/7.5/10 above one second, which is exactly the band we care about. These
# start at 50ms (a trivially short generation / an immediate 4xx), resolve the
# 0.5–5s band where nanoGPT completions actually live, and top out at 60s to
# match the default ``REQUEST_TIMEOUT_SECONDS``: anything in +Inf timed out or
# burned its whole retry budget, which is the signal worth alerting on.
DURATION_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
    30.0,
    60.0,
    float("inf"),
)

# Request outcomes (the ``outcome`` label on fleet_router_requests_total).
OUTCOME_OK = "ok"  # served on the first attempt
OUTCOME_REROUTED = "rerouted"  # served, but only after a retry — the spot story
OUTCOME_FAILED = "failed"  # 5xx back to the client (no worker answered)

_UNAVAILABLE_BODY = (
    b"# prometheus_client is not installed, so router metrics are disabled.\n"
    b"# Install it (pip install prometheus-client) to expose the exposition format.\n"
)
_UNAVAILABLE_CONTENT_TYPE = "text/plain; charset=utf-8"


class _NoopMetric:
    """Stand-in with the prometheus metric surface, minus the recording."""

    def labels(self, *args, **kwargs) -> _NoopMetric:
        return self

    def inc(self, amount: float = 1.0) -> None:
        pass

    def dec(self, amount: float = 1.0) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, amount: float, exemplar: dict | None = None) -> None:
        pass

    def clear(self) -> None:
        pass

    def remove(self, *labelvalues: str) -> None:
        pass


try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
    from prometheus_client import generate_latest as _generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by reloading with the dep hidden
    PROMETHEUS_AVAILABLE = False

if PROMETHEUS_AVAILABLE:
    REGISTRY = CollectorRegistry()

    ROUTER_REQUESTS = Counter(
        "fleet_router_requests_total",
        "Completion requests the router finished, by outcome.",
        ["outcome"],
        registry=REGISTRY,
    )
    UPSTREAM_ATTEMPTS = Counter(
        "fleet_router_upstream_attempts_total",
        "Individual worker attempts the router made, by worker and result.",
        ["worker_id", "result"],
        registry=REGISTRY,
    )
    REQUEST_DURATION = Histogram(
        "fleet_router_request_duration_seconds",
        "End-to-end router latency for a completion, retries included.",
        buckets=DURATION_BUCKETS,
        registry=REGISTRY,
    )
    LIVE_WORKERS = Gauge(
        "fleet_router_live_workers",
        "Workers whose heartbeat is inside the TTL.",
        registry=REGISTRY,
    )
    IN_FLIGHT = Gauge(
        "fleet_router_in_flight",
        "Completions the router is proxying right now.",
        registry=REGISTRY,
    )
    WORKER_QUEUE_DEPTH = Gauge(
        "fleet_worker_queue_depth",
        "Requests waiting behind the generation lock on a worker (HPA signal).",
        ["worker_id"],
        registry=REGISTRY,
    )
    WORKER_TOKENS_PER_SECOND = Gauge(
        "fleet_worker_tokens_per_second",
        "Lifetime completion tokens / generate seconds, as the worker reports it.",
        ["worker_id"],
        registry=REGISTRY,
    )
else:  # pragma: no cover - exercised by reloading with the dep hidden
    REGISTRY = None
    ROUTER_REQUESTS = _NoopMetric()
    UPSTREAM_ATTEMPTS = _NoopMetric()
    REQUEST_DURATION = _NoopMetric()
    LIVE_WORKERS = _NoopMetric()
    IN_FLIGHT = _NoopMetric()
    WORKER_QUEUE_DEPTH = _NoopMetric()
    WORKER_TOKENS_PER_SECOND = _NoopMetric()


def outcome_for(status_code: int, rerouted: bool) -> str:
    """Map a ``RouteResult`` onto the ``outcome`` label.

    A 4xx counts as ``ok``: the router did its job and a worker answered — the
    request itself was bad. It still shows up as ``result="client_error"`` on
    the per-attempt counter, so it is never invisible.
    """
    if status_code >= 500:
        return OUTCOME_FAILED
    return OUTCOME_REROUTED if rerouted else OUTCOME_OK


def attempt_result(status_code: int) -> str:
    """Low-cardinality label for one upstream attempt's HTTP status."""
    if 200 <= status_code < 300:
        return "ok"
    if 400 <= status_code < 500:
        return "client_error"
    return "server_error"


def record_request(outcome: str, duration_seconds: float) -> None:
    ROUTER_REQUESTS.labels(outcome=outcome).inc()
    REQUEST_DURATION.observe(duration_seconds)


def record_attempt(worker_id: str, result: str) -> None:
    """One router→worker call. ``result`` is ``ok``/``client_error``/
    ``server_error``/``error`` (the last being a transport failure — the shape
    a preempted spot box makes)."""
    UPSTREAM_ATTEMPTS.labels(worker_id=worker_id or "unknown", result=result).inc()


def set_live_workers(count: int) -> None:
    LIVE_WORKERS.set(count)


def sync_worker_gauges(worker_stats: list[dict]) -> None:
    """Rewrite the per-worker gauges from a ``/stats`` sweep.

    Clear-then-set, so a worker that vanished between sweeps stops being
    exported rather than freezing at its last value (a stuck queue_depth would
    otherwise keep an HPA scaled up forever). Workers that failed their scrape
    (``ok=False``) are dropped for the same reason — we have no fresh reading.
    """
    WORKER_QUEUE_DEPTH.clear()
    WORKER_TOKENS_PER_SECOND.clear()
    for doc in worker_stats:
        worker_id = doc.get("worker_id") or ""
        if not worker_id or not doc.get("ok", True):
            continue
        if "queued" in doc:
            WORKER_QUEUE_DEPTH.labels(worker_id=worker_id).set(float(doc["queued"]))
        if "tokens_per_second" in doc:
            WORKER_TOKENS_PER_SECOND.labels(worker_id=worker_id).set(
                float(doc["tokens_per_second"])
            )


def render() -> tuple[bytes, str]:
    """``(payload, content_type)`` for the router's ``/metrics`` endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return _UNAVAILABLE_BODY, _UNAVAILABLE_CONTENT_TYPE
    return _generate_latest(REGISTRY), CONTENT_TYPE_LATEST
