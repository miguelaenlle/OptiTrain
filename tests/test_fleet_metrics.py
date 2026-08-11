"""Prometheus exposition for the router: the numbers a Grafana panel (and a
future HPA reading queue depth) will read. No model, no sockets, no cluster."""

import importlib
import sys
import threading
import time

import pytest

pytest.importorskip("fastapi")  # router imports fastapi at module scope

from inference import metrics  # noqa: E402


def _text() -> str:
    payload, _ = metrics.render()
    return payload.decode()


def _split(series: str) -> tuple[str, frozenset]:
    name, brace, rest = series.partition("{")
    return name, frozenset(rest.rstrip("}").split(",")) if brace else frozenset()


def _sample(text: str, key: str) -> float:
    """Value of one exposition series, e.g. ``fleet_router_in_flight`` or
    ``fleet_router_requests_total{outcome="ok"}``; missing series read 0.
    Labels are compared as a set — the exposition format sorts them."""
    want = _split(key)
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        series, _, value = line.rpartition(" ")
        if _split(series) == want:
            return float(value)
    return 0.0


# --- module surface -------------------------------------------------------


def test_render_returns_bytes_and_content_type():
    payload, content_type = metrics.render()
    assert isinstance(payload, bytes)
    assert content_type.startswith("text/plain")


def test_all_metric_families_are_declared():
    pytest.importorskip("prometheus_client")
    text = _text()
    for name in (
        "fleet_router_requests_total",
        "fleet_router_upstream_attempts_total",
        "fleet_router_request_duration_seconds",
        "fleet_router_live_workers",
        "fleet_router_in_flight",
        "fleet_worker_queue_depth",
        "fleet_worker_tokens_per_second",
    ):
        assert f"# HELP {name} " in text


def test_duration_buckets_cover_llm_latencies():
    """Completions run hundreds of ms to seconds; the default buckets waste
    resolution below 250ms and stop at 10s. Ours top out at the 60s timeout."""
    assert metrics.DURATION_BUCKETS[0] == 0.05
    finite = [b for b in metrics.DURATION_BUCKETS if b != float("inf")]
    assert finite[-1] == 60.0
    assert sum(1 for b in finite if 0.5 <= b <= 5.0) >= 4  # resolution where it matters
    assert list(finite) == sorted(finite)


# --- counters / gauges move ----------------------------------------------


def test_ok_and_rerouted_outcomes_are_separate_counters():
    pytest.importorskip("prometheus_client")
    before = _text()
    metrics.record_request(metrics.OUTCOME_OK, 0.4)
    metrics.record_request(metrics.OUTCOME_REROUTED, 2.5)
    after = _text()
    ok = 'fleet_router_requests_total{outcome="ok"}'
    rerouted = 'fleet_router_requests_total{outcome="rerouted"}'
    assert _sample(after, ok) - _sample(before, ok) == 1
    assert _sample(after, rerouted) - _sample(before, rerouted) == 1
    count = "fleet_router_request_duration_seconds_count"
    assert _sample(after, count) - _sample(before, count) == 2
    # 2.5s lands above the 2.0 boundary and at/below 3.0 — cumulative buckets.
    lo = 'fleet_router_request_duration_seconds_bucket{le="2.0"}'
    hi = 'fleet_router_request_duration_seconds_bucket{le="3.0"}'
    assert _sample(after, lo) - _sample(before, lo) == 1  # only the 0.4s one
    assert _sample(after, hi) - _sample(before, hi) == 2


def test_outcome_and_attempt_label_mapping():
    assert metrics.outcome_for(200, False) == metrics.OUTCOME_OK
    assert metrics.outcome_for(200, True) == metrics.OUTCOME_REROUTED
    assert metrics.outcome_for(400, False) == metrics.OUTCOME_OK  # client's fault, we served
    assert metrics.outcome_for(503, False) == metrics.OUTCOME_FAILED
    assert metrics.attempt_result(200) == "ok"
    assert metrics.attempt_result(422) == "client_error"
    assert metrics.attempt_result(500) == "server_error"


def test_upstream_attempts_are_labeled_per_worker():
    pytest.importorskip("prometheus_client")
    before = _text()
    metrics.record_attempt("w-alpha", "error")
    metrics.record_attempt("w-beta", "ok")
    after = _text()
    dead = 'fleet_router_upstream_attempts_total{worker_id="w-alpha",result="error"}'
    live = 'fleet_router_upstream_attempts_total{worker_id="w-beta",result="ok"}'
    assert _sample(after, dead) - _sample(before, dead) == 1
    assert _sample(after, live) - _sample(before, live) == 1


def test_live_workers_gauge_is_absolute():
    pytest.importorskip("prometheus_client")
    metrics.set_live_workers(4)
    assert _sample(_text(), "fleet_router_live_workers") == 4
    metrics.set_live_workers(1)
    assert _sample(_text(), "fleet_router_live_workers") == 1


def test_worker_gauges_track_the_latest_scrape():
    """Queue depth is the HPA signal, so a vanished worker must stop exporting
    rather than pin the gauge at its last (possibly huge) value."""
    pytest.importorskip("prometheus_client")
    metrics.sync_worker_gauges(
        [
            {"worker_id": "w0", "ok": True, "queued": 3, "tokens_per_second": 42.5},
            {"worker_id": "w1", "ok": True, "queued": 0, "tokens_per_second": 10.0},
            {"worker_id": "w2", "ok": False, "error": "refused"},  # scrape failed
        ]
    )
    text = _text()
    assert _sample(text, 'fleet_worker_queue_depth{worker_id="w0"}') == 3
    assert _sample(text, 'fleet_worker_tokens_per_second{worker_id="w0"}') == 42.5
    assert _sample(text, 'fleet_worker_queue_depth{worker_id="w1"}') == 0
    assert 'worker_id="w2"' not in text  # no fresh reading -> no gauge

    metrics.sync_worker_gauges([{"worker_id": "w1", "ok": True, "queued": 7}])
    text = _text()
    assert 'worker_id="w0"' not in text  # dropped out of the sweep
    assert _sample(text, 'fleet_worker_queue_depth{worker_id="w1"}') == 7


# --- router integration ---------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.text = ""
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def router():
    """Two live workers injected; TestClient built without ``with`` so the
    lifespan poll/stats threads never start (same trick as test_fleet_monitor)."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from inference.router import RouterSettings, RouterState, create_app

    state = RouterState()
    state.set_workers(
        [
            {"worker_id": "w0", "addr": "host0:8001", "last_seen": time.time()},
            {"worker_id": "w1", "addr": "host1:8001", "last_seen": time.time()},
        ]
    )
    app = create_app(RouterSettings(workers_uri="unused", max_attempts=3), state)
    return TestClient(app)


def _stub_upstream(monkeypatch, handler):
    """Replace the router's outbound requests.post with ``handler(addr, body)``."""
    from inference import router as router_module

    def post(url, json=None, timeout=None):
        addr = url[len("http://") :].split("/", 1)[0]
        return handler(addr, json)

    monkeypatch.setattr(router_module.requests, "post", post)


def test_metrics_endpoint_serves_prometheus(router):
    r = router.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    if metrics.PROMETHEUS_AVAILABLE:
        assert "fleet_router_live_workers" in r.text
    else:
        assert "prometheus_client is not installed" in r.text


def test_completion_records_ok_outcome_and_worker_attempt(router, monkeypatch):
    pytest.importorskip("prometheus_client")
    _stub_upstream(monkeypatch, lambda addr, body: _FakeResponse(200, {"choices": []}))
    before = router.get("/metrics").text

    assert router.post("/v1/completions", json={"prompt": "x"}).status_code == 200

    after = router.get("/metrics").text
    ok = 'fleet_router_requests_total{outcome="ok"}'
    assert _sample(after, ok) - _sample(before, ok) == 1
    # Round-robin advances before the first request, so w1 serves it.
    served = 'fleet_router_upstream_attempts_total{worker_id="w1",result="ok"}'
    assert _sample(after, served) - _sample(before, served) == 1
    assert _sample(after, "fleet_router_in_flight") == 0  # decremented on the way out


def test_reroute_records_both_workers_and_the_rerouted_outcome(router, monkeypatch):
    pytest.importorskip("prometheus_client")
    import requests as requests_lib

    def handler(addr, body):
        if addr == "host1:8001":  # the one round-robin picks first
            raise requests_lib.ConnectionError("connection refused")
        return _FakeResponse(200, {"choices": []})

    _stub_upstream(monkeypatch, handler)
    before = router.get("/metrics").text

    assert router.post("/v1/completions", json={"prompt": "x"}).status_code == 200

    after = router.get("/metrics").text
    rerouted = 'fleet_router_requests_total{outcome="rerouted"}'
    dead = 'fleet_router_upstream_attempts_total{worker_id="w1",result="error"}'
    live = 'fleet_router_upstream_attempts_total{worker_id="w0",result="ok"}'
    assert _sample(after, rerouted) - _sample(before, rerouted) == 1
    assert _sample(after, dead) - _sample(before, dead) == 1
    assert _sample(after, live) - _sample(before, live) == 1


def test_exhausted_fleet_records_failed_outcome(router, monkeypatch):
    pytest.importorskip("prometheus_client")
    import requests as requests_lib

    def handler(addr, body):
        raise requests_lib.ConnectionError("boom")

    _stub_upstream(monkeypatch, handler)
    before = router.get("/metrics").text

    assert router.post("/v1/completions", json={"prompt": "x"}).status_code == 503

    after = router.get("/metrics").text
    failed = 'fleet_router_requests_total{outcome="failed"}'
    assert _sample(after, failed) - _sample(before, failed) == 1
    count = "fleet_router_request_duration_seconds_count"
    assert _sample(after, count) - _sample(before, count) == 1  # failures are timed too


class _OnceStop(threading.Event):
    """Stop token that lets a background loop run its body exactly once: the
    first ``wait()`` (the sleep at the bottom of the loop) trips the flag."""

    def wait(self, timeout=None):
        self.set()
        return True


def test_poll_and_stats_loops_feed_the_gauges(monkeypatch):
    """The real loop bodies, one iteration each, no threads and no sockets."""
    pytest.importorskip("prometheus_client")
    from inference import registry as registry_module
    from inference import router as router_module

    docs = [{"worker_id": "w0", "addr": "host0:8001", "last_seen": time.time()}]
    monkeypatch.setattr(registry_module, "list_workers", lambda uri: docs)
    monkeypatch.setattr(
        router_module.requests,
        "get",
        lambda url, timeout=None: _FakeResponse(200, {"queued": 5, "tokens_per_second": 33.0}),
    )

    state = router_module.RouterState()
    settings = router_module.RouterSettings(workers_uri="unused")
    router_module._poll_loop(state, settings, _OnceStop())
    router_module._stats_loop(state, settings, _OnceStop())

    text = _text()
    assert _sample(text, "fleet_router_live_workers") == 1
    assert _sample(text, 'fleet_worker_queue_depth{worker_id="w0"}') == 5
    assert _sample(text, 'fleet_worker_tokens_per_second{worker_id="w0"}') == 33.0


# --- graceful degradation -------------------------------------------------


def _assert_router_serves_without_prometheus(monkeypatch):
    """With the dep hidden the router still routes; /metrics just explains."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from inference.router import RouterSettings, RouterState, create_app

    state = RouterState()
    state.set_workers([{"worker_id": "w0", "addr": "host0:8001", "last_seen": time.time()}])
    client = TestClient(create_app(RouterSettings(workers_uri="unused"), state))
    _stub_upstream(monkeypatch, lambda addr, body: _FakeResponse(200, {"choices": []}))

    assert client.post("/v1/completions", json={"prompt": "x"}).status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "prometheus_client is not installed" in r.text


def test_module_degrades_without_prometheus_client(monkeypatch):
    """Hide the dep, reload, and confirm nothing the router calls explodes."""
    monkeypatch.setitem(sys.modules, "prometheus_client", None)  # forces ImportError
    try:
        degraded = importlib.reload(metrics)
        assert degraded.PROMETHEUS_AVAILABLE is False
        payload, content_type = degraded.render()
        assert isinstance(payload, bytes)
        assert b"prometheus_client is not installed" in payload
        assert content_type.startswith("text/plain")
        # Every call the router makes must be a safe no-op.
        degraded.record_request(degraded.OUTCOME_OK, 1.25)
        degraded.record_attempt("w0", "error")
        degraded.set_live_workers(2)
        degraded.sync_worker_gauges([{"worker_id": "w0", "ok": True, "queued": 1}])
        degraded.IN_FLIGHT.inc()
        degraded.IN_FLIGHT.dec()
        assert degraded.outcome_for(503, False) == degraded.OUTCOME_FAILED
        _assert_router_serves_without_prometheus(monkeypatch)
    finally:
        monkeypatch.undo()
        importlib.reload(metrics)  # fresh private registry; later tests unaffected
    assert metrics.PROMETHEUS_AVAILABLE is _prometheus_installed()


def _prometheus_installed() -> bool:
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        return False
    return True
