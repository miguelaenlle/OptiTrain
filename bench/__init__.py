"""bench/ — the inference benchmark harness (docs/inference-platform-plan.md §3).

Drives the Go load generator (``loadgen/``) across sweeps and turns the reports
it writes into the two portfolio figures:

* **E2 — saturation curve.** Fixed worker count, sweep offered RPS. Produces
  p50/p95/p99 vs offered RPS and locates the *knee* (``C1``): the highest
  offered RPS whose p99 still holds the SLO.
* **E5 — scaling sweep.** Near-saturation load, sweep worker count
  ``N = 1, 2, 4, 8 …``. Produces sustained tokens/s vs N and the scaling
  efficiency ``(tput_N / N) / tput_1``.

Layout mirrors the repo's split between pure logic and effects:

``bench.analyze``
    PURE — parses loadgen reports, derives ``L0`` / ``SLO`` / the knee /
    efficiency. No I/O, no subprocess, no matplotlib; unit-tested in
    ``tests/test_bench.py``.
``bench.sweep``
    Planning is pure (``plan_saturation`` / ``plan_scaling`` /
    ``planned_commands``); only a handful of thin functions actually shell out
    to ``go build`` and the loadgen binary. ``--dry-run`` prints the plan.
``bench.plots``
    Renders the two figures (PNG + SVG) plus a CSV table view.

``bench`` is a plain package at the repo root — run it as
``python -m bench.sweep …`` from there (it is deliberately not installed by
``pyproject.toml``; only its matplotlib dependency is, under the ``bench``
extra).
"""

__all__ = ["analyze", "plots", "sweep"]
