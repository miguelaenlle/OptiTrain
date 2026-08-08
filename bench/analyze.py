"""Pure analysis of Go-loadgen reports — no I/O, no subprocess, no plotting.

Everything here takes plain dicts / dataclasses and returns plain dataclasses,
so the whole benchmark story is unit-testable without a fleet (same discipline
as ``orchestrator.fleet_preempt.analyze`` and ``orchestrator.monitor``).

**Schema.** Field names come straight from the ``report`` struct in
``loadgen/main.go`` — do not rename them here:

``url``, ``start_unix``, ``rps``, ``concurrency``, ``duration_s``,
``requests``, ``succeeded``, ``failed``, ``dropped``, ``error_rate``,
``completion_tokens``, ``tokens_per_second``, ``mean_ms``, ``p50_ms``,
``p90_ms``, ``p95_ms``, ``p99_ms``, ``kill_at_s`` (omitempty), ``kill_cmd``
(omitempty), and ``per_second``: a list of buckets
``{t, sent, ok, errors, dropped, p99_ms, mean_ms}``.

**Definitions** (docs/inference-platform-plan.md §3 — one place, everything
keys off it):

``L0``
    unloaded single-request p50 latency (1 worker, RPS=1).
``SLO``
    ``p99 <= 3 x L0``.
``C1`` (the knee)
    the highest offered RPS whose p99 still holds the SLO.
scaling efficiency
    ``(tput_N / N) / tput_1`` — targets >= 90% at N=4, >= 80% at N=8.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

# --------------------------------------------------------------------------- #
# Constants (the plan's numbers, in one place)
# --------------------------------------------------------------------------- #
SLO_MULTIPLIER = 3.0
"""SLO = p99 <= SLO_MULTIPLIER x L0."""

P99_P50_TARGET = 3.0
"""Tail-to-median target: p99/p50 < 3x (< 2x is excellent)."""

EFFICIENCY_TARGETS: dict[int, float] = {4: 0.90, 8: 0.80}
"""Scaling-efficiency targets, keyed by worker count."""

DEFAULT_MAX_ERROR_RATE = 0.01
"""A sweep point only counts as holding the SLO if its error rate is under
this. Without the guard a point that failed 90% of its requests would "pass"
on the fast 10% that survived. Dropped requests are *not* gated: in the
open-loop generator a drop means the client's own worker pool was saturated,
so it is reported (see ``Point.dropped``) rather than used as a verdict."""


# --------------------------------------------------------------------------- #
# Normalized records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Point:
    """One loadgen report, normalized into the fields the analysis needs.

    ``workers`` and ``label`` are not in the loadgen report (the generator does
    not know how big the fleet is) — the sweep manifest supplies them.
    """

    label: str
    workers: int
    offered_rps: float
    concurrency: int
    duration_s: float
    requests: int
    succeeded: int
    failed: int
    dropped: int
    error_rate: float
    completion_tokens: int
    tokens_per_second: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    start_unix: float = 0.0
    kill_at_s: float | None = None
    url: str = ""

    @property
    def achieved_rps(self) -> float:
        """Successful completions per second — the *served* rate, which falls
        below ``offered_rps`` once the fleet saturates."""
        return self.succeeded / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def p99_p50_ratio(self) -> float:
        """Tail-to-median spread. ``inf`` when there is no median to divide by
        (a point where nothing succeeded)."""
        return self.p99_ms / self.p50_ms if self.p50_ms > 0 else float("inf")


@dataclass(frozen=True)
class Knee:
    """Where the saturation curve crosses the SLO.

    ``rps`` is ``C1``: the highest swept offered RPS whose p99 still holds.
    ``beyond_sweep`` means *every* swept point held, so the true knee is at or
    above the largest RPS tested — the sweep did not find it, and the chart
    must say so rather than implying the curve ends there.
    """

    rps: float | None
    p99_ms: float | None
    beyond_sweep: bool = False
    point: Point | None = None

    @property
    def found(self) -> bool:
        return self.rps is not None


@dataclass(frozen=True)
class SaturationRow:
    point: Point
    within_slo: bool
    p99_p50_ratio: float
    is_knee: bool = False


@dataclass(frozen=True)
class Saturation:
    """E2 — the saturation curve, analyzed."""

    rows: tuple[SaturationRow, ...]
    l0_ms: float
    slo_ms: float
    slo_multiplier: float
    knee: Knee
    workers: int

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(r.point for r in self.rows)


@dataclass(frozen=True)
class ScalingRow:
    workers: int
    tokens_per_second: float
    ideal_tokens_per_second: float
    efficiency: float
    target: float | None
    meets_target: bool | None
    p99_ms: float
    error_rate: float
    offered_rps: float
    point: Point


@dataclass(frozen=True)
class Scaling:
    """E5 — the scaling sweep, analyzed."""

    rows: tuple[ScalingRow, ...]
    baseline_workers: int
    baseline_tokens_per_second: float
    targets: dict[int, float]

    @property
    def passed(self) -> bool:
        """True when every swept point that has a target meets it (and at
        least one target was actually exercised)."""
        checked = [r for r in self.rows if r.meets_target is not None]
        return bool(checked) and all(r.meets_target for r in checked)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _f(d: dict, key: str, default: float = 0.0) -> float:
    value = d.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def _i(d: dict, key: str, default: int = 0) -> int:
    value = d.get(key, default)
    return int(value) if isinstance(value, int | float) else default


def parse_report(
    report: dict,
    *,
    workers: int = 1,
    label: str = "",
    offered_rps: float | None = None,
) -> Point:
    """Normalize one loadgen report dict into a :class:`Point`.

    Missing keys fall back to zero so a degenerate run (nothing succeeded, so
    the Go side left every latency at 0) still parses instead of exploding —
    ``holds_slo`` is what refuses to count such a point as passing.
    """
    rps = _f(report, "rps") if offered_rps is None else float(offered_rps)
    kill = report.get("kill_at_s")
    return Point(
        label=label or f"rps{rps:g}-n{workers}",
        workers=int(workers),
        offered_rps=rps,
        concurrency=_i(report, "concurrency"),
        duration_s=_f(report, "duration_s"),
        requests=_i(report, "requests"),
        succeeded=_i(report, "succeeded"),
        failed=_i(report, "failed"),
        dropped=_i(report, "dropped"),
        error_rate=_f(report, "error_rate"),
        completion_tokens=_i(report, "completion_tokens"),
        tokens_per_second=_f(report, "tokens_per_second"),
        mean_ms=_f(report, "mean_ms"),
        p50_ms=_f(report, "p50_ms"),
        p90_ms=_f(report, "p90_ms"),
        p95_ms=_f(report, "p95_ms"),
        p99_ms=_f(report, "p99_ms"),
        start_unix=_f(report, "start_unix"),
        kill_at_s=float(kill) if isinstance(kill, int | float) else None,
        url=str(report.get("url", "")),
    )


def points_from_manifest(manifest: dict, *, include_warmup: bool = False) -> list[Point]:
    """Parse every point of a ``sweep.json`` manifest (see ``bench.sweep``).

    Warmup points are excluded by default — their whole job is to absorb model
    init / CUDA-context cost so it never lands in ``L0``.
    """
    points: list[Point] = []
    for entry in manifest.get("points", []):
        if entry.get("warmup") and not include_warmup:
            continue
        report = entry.get("report")
        if not isinstance(report, dict):
            continue
        points.append(
            parse_report(
                report,
                workers=int(entry.get("workers", 1)),
                label=str(entry.get("label", "")),
                offered_rps=entry.get("offered_rps"),
            )
        )
    return points


# --------------------------------------------------------------------------- #
# The definitions: L0, SLO, the knee
# --------------------------------------------------------------------------- #
def l0_ms(points: Sequence[Point]) -> float:
    """``L0`` — unloaded single-request p50, in ms.

    Picked from the sweep rather than passed in: the 1-worker point with the
    lowest offered RPS (ties broken by the lower p50). Points where nothing
    succeeded carry no latency and are skipped.
    """
    candidates = [p for p in points if p.workers <= 1 and p.succeeded > 0 and p.p50_ms > 0]
    if not candidates:
        raise ValueError(
            "cannot derive L0: no single-worker point with successful requests "
            "(sweep a 1-worker, low-RPS point, or pass l0_override)"
        )
    best = min(candidates, key=lambda p: (p.offered_rps, p.p50_ms))
    return best.p50_ms


def slo_ms(l0: float, multiplier: float = SLO_MULTIPLIER) -> float:
    """The latency SLO: ``p99 <= multiplier x L0``."""
    return multiplier * l0


def holds_slo(point: Point, slo: float, *, max_error_rate: float = DEFAULT_MAX_ERROR_RATE) -> bool:
    """Did this sweep point hold the SLO? Requires real successes and an error
    rate under ``max_error_rate`` — see :data:`DEFAULT_MAX_ERROR_RATE`."""
    return point.succeeded > 0 and point.error_rate <= max_error_rate and point.p99_ms <= slo


def find_knee(
    points: Sequence[Point], slo: float, *, max_error_rate: float = DEFAULT_MAX_ERROR_RATE
) -> Knee:
    """``C1`` — the last offered RPS that still holds the SLO.

    Walks the sweep in ascending offered-RPS order and stops at the first
    violation, so the knee is the top of the *leading* run of healthy points.
    (Taking a global max instead would let a lucky high-RPS point sit above a
    real breakdown and report a knee the fleet cannot actually sustain.)

    Degenerate cases are reported, not smoothed over:

    * every point holds -> ``rps`` = the largest swept RPS with
      ``beyond_sweep=True`` (the real C1 is >= that; extend the sweep).
    * no point holds -> ``rps is None`` (C1 is below the smallest swept RPS).
    """
    ordered = sorted(points, key=lambda p: p.offered_rps)
    if not ordered:
        raise ValueError("cannot find a knee in an empty sweep")
    knee_point: Point | None = None
    for point in ordered:
        if not holds_slo(point, slo, max_error_rate=max_error_rate):
            break
        knee_point = point
    if knee_point is None:
        return Knee(rps=None, p99_ms=None, beyond_sweep=False, point=None)
    return Knee(
        rps=knee_point.offered_rps,
        p99_ms=knee_point.p99_ms,
        beyond_sweep=knee_point is ordered[-1],
        point=knee_point,
    )


# --------------------------------------------------------------------------- #
# E2 — saturation
# --------------------------------------------------------------------------- #
def analyze_saturation(
    points: Iterable[Point],
    *,
    slo_multiplier: float = SLO_MULTIPLIER,
    l0_override: float | None = None,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> Saturation:
    """Analyze an RPS sweep at fixed worker count into :class:`Saturation`."""
    ordered = sorted(points, key=lambda p: p.offered_rps)
    if not ordered:
        raise ValueError("analyze_saturation needs at least one point")
    l0 = float(l0_override) if l0_override is not None else l0_ms(ordered)
    slo = slo_ms(l0, slo_multiplier)
    knee = find_knee(ordered, slo, max_error_rate=max_error_rate)
    rows = tuple(
        SaturationRow(
            point=p,
            within_slo=holds_slo(p, slo, max_error_rate=max_error_rate),
            p99_p50_ratio=p.p99_p50_ratio,
            is_knee=knee.point is p,
        )
        for p in ordered
    )
    workers = ordered[0].workers
    return Saturation(
        rows=rows,
        l0_ms=l0,
        slo_ms=slo,
        slo_multiplier=slo_multiplier,
        knee=knee,
        workers=workers,
    )


# --------------------------------------------------------------------------- #
# E5 — scaling
# --------------------------------------------------------------------------- #
def analyze_scaling(
    points: Iterable[Point],
    *,
    targets: dict[int, float] | None = None,
) -> Scaling:
    """Analyze a worker-count sweep into :class:`Scaling`.

    Efficiency is ``(tput_N / N) / tput_1``, so N=1 is 1.0 by construction.
    When a worker count appears more than once (repeat runs) the best
    throughput observed for that N wins.
    """
    targets = dict(EFFICIENCY_TARGETS if targets is None else targets)
    best: dict[int, Point] = {}
    for p in points:
        current = best.get(p.workers)
        if current is None or p.tokens_per_second > current.tokens_per_second:
            best[p.workers] = p
    if not best:
        raise ValueError("analyze_scaling needs at least one point")
    baseline_workers = min(best)
    baseline = best[baseline_workers]
    if baseline.tokens_per_second <= 0:
        raise ValueError(
            f"cannot compute scaling efficiency: the N={baseline_workers} baseline "
            "produced no tokens/s"
        )
    per_worker_baseline = baseline.tokens_per_second / baseline_workers

    rows = []
    for n in sorted(best):
        p = best[n]
        ideal = per_worker_baseline * n
        efficiency = (p.tokens_per_second / n) / per_worker_baseline
        target = targets.get(n)
        rows.append(
            ScalingRow(
                workers=n,
                tokens_per_second=p.tokens_per_second,
                ideal_tokens_per_second=ideal,
                efficiency=efficiency,
                target=target,
                meets_target=None if target is None else efficiency >= target,
                p99_ms=p.p99_ms,
                error_rate=p.error_rate,
                offered_rps=p.offered_rps,
                point=p,
            )
        )
    return Scaling(
        rows=tuple(rows),
        baseline_workers=baseline_workers,
        baseline_tokens_per_second=baseline.tokens_per_second,
        targets=targets,
    )


# --------------------------------------------------------------------------- #
# Table views (the accessible twin of every chart) + terminal rendering
# --------------------------------------------------------------------------- #
def saturation_table(sat: Saturation) -> list[list[str]]:
    """Header + rows, all strings — written beside the figure as CSV."""
    rows = [
        [
            "offered_rps",
            "achieved_rps",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "p99_p50",
            "tokens_per_second",
            "error_rate",
            "dropped",
            "within_slo",
            "knee",
        ]
    ]
    for r in sat.rows:
        p = r.point
        ratio = "inf" if r.p99_p50_ratio == float("inf") else f"{r.p99_p50_ratio:.2f}"
        rows.append(
            [
                f"{p.offered_rps:g}",
                f"{p.achieved_rps:.2f}",
                f"{p.p50_ms:.1f}",
                f"{p.p95_ms:.1f}",
                f"{p.p99_ms:.1f}",
                ratio,
                f"{p.tokens_per_second:.1f}",
                f"{p.error_rate:.4f}",
                str(p.dropped),
                "yes" if r.within_slo else "no",
                "yes" if r.is_knee else "",
            ]
        )
    return rows


def scaling_table(sc: Scaling) -> list[list[str]]:
    rows = [
        [
            "workers",
            "offered_rps",
            "tokens_per_second",
            "ideal_tokens_per_second",
            "efficiency",
            "target",
            "meets_target",
            "p99_ms",
            "error_rate",
        ]
    ]
    for r in sc.rows:
        rows.append(
            [
                str(r.workers),
                f"{r.offered_rps:g}",
                f"{r.tokens_per_second:.1f}",
                f"{r.ideal_tokens_per_second:.1f}",
                f"{r.efficiency:.3f}",
                "" if r.target is None else f"{r.target:.2f}",
                "" if r.meets_target is None else ("yes" if r.meets_target else "no"),
                f"{r.p99_ms:.1f}",
                f"{r.error_rate:.4f}",
            ]
        )
    return rows


def render_saturation(sat: Saturation) -> str:
    """Terminal table for E2 (same shape as ``fleet_preempt.render``)."""
    lines = [
        "=== E2 saturation curve ===",
        f"L0 {sat.l0_ms:.1f} ms (1 worker, lowest RPS) | "
        f"SLO p99 <= {sat.slo_multiplier:g} x L0 = {sat.slo_ms:.1f} ms | "
        f"workers {sat.workers}",
        f"{'rps':>7} {'served':>8} {'p50ms':>8} {'p95ms':>8} {'p99ms':>8} "
        f"{'p99/p50':>8} {'tok/s':>8} {'err':>7} {'drop':>6}  slo",
    ]
    for r in sat.rows:
        p = r.point
        ratio = "  inf" if r.p99_p50_ratio == float("inf") else f"{r.p99_p50_ratio:.2f}"
        mark = "OK " if r.within_slo else "OVER"
        knee = "  <- knee (C1)" if r.is_knee else ""
        lines.append(
            f"{p.offered_rps:>7g} {p.achieved_rps:>8.2f} {p.p50_ms:>8.1f} {p.p95_ms:>8.1f} "
            f"{p.p99_ms:>8.1f} {ratio:>8} {p.tokens_per_second:>8.1f} "
            f"{p.error_rate * 100:>6.2f}% {p.dropped:>6}  {mark}{knee}"
        )
    if sat.knee.rps is None:
        lines.append("C1 (knee): NOT FOUND — every swept RPS broke the SLO; sweep lower")
    elif sat.knee.beyond_sweep:
        lines.append(
            f"C1 (knee): >= {sat.knee.rps:g} rps — every swept point held the SLO; sweep higher"
        )
    else:
        lines.append(
            f"C1 (knee): {sat.knee.rps:g} rps @ p99 {sat.knee.p99_ms:.1f} ms "
            f"({sat.knee.rps / max(sat.workers, 1):g} rps/worker)"
        )
    return "\n".join(lines)


def render_scaling(sc: Scaling) -> str:
    """Terminal table for E5."""
    lines = [
        "=== E5 scaling sweep ===",
        f"baseline N={sc.baseline_workers}: {sc.baseline_tokens_per_second:.1f} tok/s | "
        "efficiency = (tput_N / N) / tput_1",
        f"{'N':>3} {'tok/s':>10} {'ideal':>10} {'eff':>7} {'target':>7} "
        f"{'p99ms':>9} {'err':>7}  verdict",
    ]
    for r in sc.rows:
        target = "-" if r.target is None else f"{r.target * 100:.0f}%"
        verdict = "" if r.meets_target is None else ("MEETS" if r.meets_target else "BELOW")
        lines.append(
            f"{r.workers:>3} {r.tokens_per_second:>10.1f} {r.ideal_tokens_per_second:>10.1f} "
            f"{r.efficiency * 100:>6.1f}% {target:>7} {r.p99_ms:>9.1f} "
            f"{r.error_rate * 100:>6.2f}%  {verdict}"
        )
    checked = [r for r in sc.rows if r.meets_target is not None]
    if checked:
        lines.append(f"RESULT: {'PASS' if sc.passed else 'FAIL'}")
    else:
        lines.append("RESULT: no target worker count (N=4 / N=8) in this sweep")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Small helpers used by the plotting layer (kept pure and here on purpose)
# --------------------------------------------------------------------------- #
def relabel(point: Point, label: str) -> Point:
    """A copy of ``point`` with a different label (frozen dataclasses)."""
    return replace(point, label=label)


def as_dict(obj: Any) -> dict:
    """Shallow dict view of an analysis dataclass, for JSON artifacts.

    ``Point`` objects are dropped (they are already in the manifest's raw
    reports) so the artifact stays a summary rather than a second copy.
    """
    from dataclasses import fields, is_dataclass

    if not is_dataclass(obj):
        raise TypeError(f"as_dict expects a dataclass, got {type(obj)!r}")
    out: dict = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, Point) or f.name == "point":
            continue
        if isinstance(value, tuple):
            out[f.name] = [as_dict(v) if is_dataclass(v) else v for v in value]
        elif is_dataclass(value):
            out[f.name] = as_dict(value)
        else:
            out[f.name] = value
    return out
