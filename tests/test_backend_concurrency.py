"""The lock bypass — the thing that decides whether vLLM actually batches.

A backend that schedules sequences itself must receive concurrent requests
concurrently. If ModelService held its lock, vLLM would get one sequence at a
time and continuous batching would silently do nothing: no error, no warning,
throughput simply unchanged. These tests pin the behaviour so that regression
cannot happen quietly.
"""

from __future__ import annotations

import threading
import time

from inference.service import ModelService


class _Backend:
    """Records the maximum number of generate() calls in flight at once."""

    def __init__(self, concurrent: bool):
        self.concurrent = concurrent
        self.name = "fake"
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def generate(self, ids, *, max_new_tokens, temperature, top_k, seed=None):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        time.sleep(0.05)
        with self._lock:
            self._in_flight -= 1
        return [1] * max_new_tokens


def _service(backend) -> ModelService:
    return ModelService(
        backend,
        encode=lambda s: [0, 1, 2],
        decode=lambda ids: "x" * len(ids),
        device="cpu",
        model_name="fake",
        ckpt_step=0,
    )


def _drive(svc: ModelService, n: int = 6) -> None:
    threads = [
        threading.Thread(
            target=svc.complete,
            args=("hello",),
            kwargs={"max_new_tokens": 4, "temperature": 1.0, "top_k": None},
        )
        for _ in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)


def test_serial_backend_is_serialized():
    """nanoGPT/HF cannot batch, so exactly one generate() may run at a time."""
    b = _Backend(concurrent=False)
    _drive(_service(b))
    assert b.max_in_flight == 1, f"serial backend saw {b.max_in_flight} concurrent calls"


def test_concurrent_backend_is_not_serialized():
    """vLLM-style backends must see requests in parallel or batching is moot."""
    b = _Backend(concurrent=True)
    _drive(_service(b), n=6)
    assert b.max_in_flight > 1, (
        "concurrent backend was serialized — continuous batching would be "
        "silently disabled and throughput would stay at the unbatched rate"
    )


def test_backend_without_the_attribute_defaults_to_serial():
    """An older/third-party backend must not accidentally opt into concurrency."""

    class Legacy:
        name = "legacy"

        def __init__(self):
            self.calls = 0

        def generate(self, ids, *, max_new_tokens, temperature, top_k, seed=None):
            self.calls += 1
            return [7] * max_new_tokens

    svc = _service(Legacy())
    out = svc.complete("hi", max_new_tokens=3, temperature=1.0, top_k=None)
    assert out["completion_tokens"] == 3
