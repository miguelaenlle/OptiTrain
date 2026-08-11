"""The two portfolio figures: E2 saturation curve, E5 scaling sweep.

    python -m bench.plots --sweep .fleet/bench/e2-saturation/sweep.json --out figures/

Rendering only — every number on these charts comes from
:mod:`bench.analyze`, which comes from the loadgen's own reports. Nothing is
smoothed, extrapolated, or invented; a sweep that never crossed the SLO says so
on the chart instead of drawing a knee that was not measured.

Design rules (all deliberate, none matplotlib defaults):

* **Palette** — the validated categorical slots 1-3, in fixed order, with a
  dark-mode set stepped for the dark surface (not an inverted flip). Threshold
  lines wear the reserved *status* red; every other non-data mark is chart
  chrome. Validated with the data-viz palette checker in both modes: all
  checks pass, except light-mode slot 3 (aqua) at 2.74:1 contrast — that WARN
  obligates a relief channel, so every figure is written alongside a CSV table
  view and carries direct labels.
* **Marks** — 2px lines, 8px markers with a 2px surface ring, hairline solid
  gridlines on one axis, no top/right spines, one y-scale per chart (never a
  second axis).
* **Text** — labels, values and legends wear ink tokens, never the series
  colour; identity comes from the coloured mark beside them. Labels are
  selective: series ends, the knee, the efficiency at each N.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless: these are artifacts, not windows

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from . import analyze  # noqa: E402


# --------------------------------------------------------------------------- #
# Theme tokens (validated palette; see module docstring)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]
    threshold: str  # reserved status colour — thresholds only, never a series


THEMES: dict[str, Theme] = {
    "light": Theme(
        name="light",
        surface="#fcfcfb",
        text_primary="#0b0b0b",
        text_secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
        threshold="#d03b3b",
    ),
    "dark": Theme(
        name="dark",
        surface="#1a1a19",
        text_primary="#ffffff",
        text_secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=("#3987e5", "#d95926", "#199e70"),
        threshold="#d03b3b",
    ),
}

FIGSIZE = (9.0, 5.4)
DPI = 200
LINE_W = 2.0
MARKER_SIZE = 8.0
RING_W = 2.0
HAIRLINE = 0.9


def _theme(theme: str | Theme) -> Theme:
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r} — pick one of {sorted(THEMES)}") from None


def _new_figure(t: Theme):
    # One UI sans everywhere — no display or serif face, not even on big numbers.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(t.surface)
    ax.set_facecolor(t.surface)
    return fig, ax


def _style_axes(ax, t: Theme, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=t.grid, linewidth=HAIRLINE, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
        ax.spines[side].set_linewidth(HAIRLINE)
    ax.tick_params(colors=t.muted, labelsize=9, length=0, pad=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(t.text_secondary)


def _chrome(fig, t: Theme, title: str, subtitle: str, footnote: str, *, x: float = 0.065) -> None:
    """Title / subtitle / source note, left-aligned to the plot area."""
    fig.text(x, 0.945, title, fontsize=15, fontweight="semibold", color=t.text_primary)
    fig.text(x, 0.888, subtitle, fontsize=10, color=t.text_secondary)
    fig.text(x, 0.035, footnote, fontsize=8.5, color=t.muted)


def _legend(ax, t: Theme, handles: list, *, loc: str = "upper left"):
    """Legend with clean line keys.

    Handles are rebuilt rather than reused: matplotlib's default key draws the
    marker in the middle of the line segment, which reads as a dashed line and
    would falsely distinguish the solid data series from the dashed reference
    lines.
    """
    return ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        fontsize=9.5,
        handlelength=1.9,
        handletextpad=0.7,
        borderaxespad=0.0,
        labelspacing=0.7,
        labelcolor=t.text_secondary,
    )


def _line_key(color: str, label: str, *, dashed: bool = False) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=color,
        linewidth=LINE_W if not dashed else 1.6,
        linestyle=(0, (6, 4)) if dashed else "-",
        solid_capstyle="round",
        label=label,
    )


def _line(ax, x, y, color: str, t: Theme, label: str, *, zorder: int = 3):
    return ax.plot(
        x,
        y,
        color=color,
        linewidth=LINE_W,
        solid_capstyle="round",
        solid_joinstyle="round",
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor=color,
        markeredgecolor=t.surface,  # 2px surface ring keeps overlaps legible
        markeredgewidth=RING_W,
        label=label,
        zorder=zorder,
    )


def _axes_fraction(value: float, lo: float, hi: float, *, log: bool = False) -> float:
    """Where ``value`` sits between the limits, 0..1 — scale aware, so the
    end-label collision check stays honest on a log axis."""
    if hi <= lo:
        return 0.0
    if log:
        if min(value, lo) <= 0:
            return 0.0
        return (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return (value - lo) / (hi - lo)


def _save(fig, out_dir: str, stem: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, facecolor=fig.get_facecolor(), format=ext)
        paths.append(path)
    plt.close(fig)
    return paths


def _write_table(rows: list[list[str]], out_dir: str, stem: str) -> str:
    """The table view every chart ships with — the WCAG-clean twin, and the
    relief channel the light-mode palette WARN requires."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.csv")
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


# --------------------------------------------------------------------------- #
# E2 — saturation curve
# --------------------------------------------------------------------------- #
def plot_saturation(
    sat: analyze.Saturation,
    out_dir: str,
    *,
    theme: str | Theme = "light",
    stem: str = "e2-saturation",
    subtitle: str = "",
) -> list[str]:
    """p50 / p95 / p99 vs offered RPS, with the SLO line and the knee marked."""
    t = _theme(theme)
    points = sat.points
    x = [p.offered_rps for p in points]
    # Series in story order: the tail is the headline, so it takes slot 1.
    series = [
        ("p99", [p.p99_ms for p in points], t.series[0]),
        ("p95", [p.p95_ms for p in points], t.series[1]),
        ("p50", [p.p50_ms for p in points], t.series[2]),
    ]

    fig, ax = _new_figure(t)
    for name, ys, color in series:
        _line(ax, x, ys, color, t, name)

    # --- axes ---------------------------------------------------------------
    # Doubling sweeps get log axes: on linear scales one post-knee blow-up
    # squashes the whole pre-knee region — the part the SLO argument lives in —
    # into the baseline. Both axis labels say so.
    log_x = len(x) > 1 and min(x) > 0 and max(x) / min(x) >= 8
    values = [v for _, ys, _ in series for v in ys if v > 0]
    log_y = bool(values) and max(values) / min(values) >= 20
    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(x) / 1.3, max(x) * 1.35)
    else:
        ax.set_xlim(0, max(x) * 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.xaxis.set_minor_locator(plt.NullLocator())

    top = max(max(ys) for _, ys, _ in series)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(min(values) / 1.7, max(top, sat.slo_ms) * 1.7)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=16))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        # A decade between labels is a long way to interpolate by eye, so the
        # 2x / 5x minor gridlines stay — a shade fainter, still hairline.
        ax.grid(axis="y", which="minor", color=t.grid, linewidth=0.6, alpha=0.55)
    else:
        ax.set_ylim(0, max(top, sat.slo_ms) * 1.18)
    ax.set_xlabel(
        "offered load (requests/s" + (", log scale)" if log_x else ")"),
        fontsize=10,
        color=t.text_secondary,
        labelpad=8,
    )
    ax.set_ylabel(
        "latency (ms" + (", log scale)" if log_y else ")"),
        fontsize=10,
        color=t.text_secondary,
        labelpad=8,
    )
    _style_axes(ax, t)

    # --- SLO threshold: reserved status colour, dashed because it IS a
    # threshold (gridlines stay solid), always shipped with its label ---------
    ax.axhline(sat.slo_ms, color=t.threshold, linewidth=1.6, linestyle=(0, (6, 4)), zorder=2)
    ax.text(
        0.012,
        sat.slo_ms,
        f"SLO  p99 ≤ {sat.slo_multiplier:g} × L0 = {sat.slo_ms:.0f} ms",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
        color=t.text_secondary,
        zorder=4,
    )

    # --- the knee ------------------------------------------------------------
    knee = sat.knee
    if knee.rps is not None:
        ax.axvline(knee.rps, color=t.muted, linewidth=HAIRLINE, zorder=1)
        ax.plot(
            [knee.rps],
            [knee.p99_ms],
            marker="o",
            markersize=MARKER_SIZE + 3,
            markerfacecolor=t.series[0],
            markeredgecolor=t.surface,
            markeredgewidth=RING_W + 0.5,
            zorder=5,
        )
        note = (
            f"C1 ≥ {knee.rps:g} rps — no SLO breach in this sweep"
            if knee.beyond_sweep
            else f"knee  C1 = {knee.rps:g} rps  ({knee.p99_ms:.0f} ms p99)"
        )
        ax.annotate(
            note,
            xy=(knee.rps, 1.0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(-9, -8),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9.5,
            color=t.text_primary,
            zorder=6,
        )
    else:
        ax.annotate(
            "no swept rate held the SLO — C1 is below the smallest offered RPS",
            xy=(0.5, 0.94),
            xycoords="axes fraction",
            ha="center",
            fontsize=9.5,
            color=t.text_secondary,
        )

    # --- direct end labels (dropped, not stacked, if they would collide) -----
    lo, hi = ax.get_ylim()
    ends = [(name, ys[-1]) for name, ys, _ in series]
    fractions = sorted(_axes_fraction(v, lo, hi, log=log_y) for _, v in ends)
    if all(b - a >= 0.045 for a, b in zip(fractions, fractions[1:], strict=False)):
        for name, value in ends:
            ax.annotate(
                name,
                xy=(x[-1], value),
                xytext=(9, -3),
                textcoords="offset points",
                fontsize=9.5,
                color=t.text_secondary,
                zorder=4,
            )
    _legend(ax, t, [_line_key(color, name) for name, _ys, color in series])

    p = points[0]
    ratios = [r.p99_p50_ratio for r in sat.rows if r.p99_p50_ratio != float("inf")]
    tail = f" · worst p99/p50 = {max(ratios):.1f}×" if ratios else ""
    left = 0.088
    fig.subplots_adjust(left=left, right=0.945, top=0.795, bottom=0.155)
    _chrome(
        fig,
        t,
        "Latency vs offered load",
        subtitle
        or (
            f"{sat.workers} worker{'s' if sat.workers != 1 else ''} · "
            f"L0 = {sat.l0_ms:.0f} ms unloaded p50{tail}"
        ),
        f"client-side (loadgen) · {p.duration_s:.0f}s per point · open loop, "
        f"{p.concurrency} max in flight · table view: {stem}.csv",
        x=left,
    )
    paths = _save(fig, out_dir, stem)
    paths.append(_write_table(analyze.saturation_table(sat), out_dir, stem))
    return paths


# --------------------------------------------------------------------------- #
# E5 — scaling sweep
# --------------------------------------------------------------------------- #
def plot_scaling(
    sc: analyze.Scaling,
    out_dir: str,
    *,
    theme: str | Theme = "light",
    stem: str = "e5-scaling",
    subtitle: str = "",
) -> list[str]:
    """Sustained tokens/s vs N, against the ideal-linear reference."""
    t = _theme(theme)
    ns = [r.workers for r in sc.rows]
    measured = [r.tokens_per_second for r in sc.rows]
    ideal = [r.ideal_tokens_per_second for r in sc.rows]

    fig, ax = _new_figure(t)
    per_worker = sc.baseline_tokens_per_second / sc.baseline_workers
    x_max = max(ns) * 1.12
    ax.plot(
        [0, x_max],
        [0, per_worker * x_max],
        color=t.muted,
        linewidth=1.6,
        linestyle=(0, (6, 4)),
        label=f"ideal linear ({per_worker:.0f} tok/s × N)",
        zorder=2,
    )
    ax.fill_between(
        ns,
        measured,
        ideal,
        color=t.series[0],
        alpha=0.10,
        linewidth=0,
        label="efficiency gap",
        zorder=1,
    )
    _line(ax, ns, measured, t.series[0], t, "measured", zorder=3)

    # Efficiency is the story, and there are only a handful of points — so
    # every marker is labelled with it (the baseline says what it is).
    for r in sc.rows:
        text = "baseline" if r.workers == sc.baseline_workers else f"{r.efficiency * 100:.0f}%"
        ax.annotate(
            text,
            xy=(r.workers, r.tokens_per_second),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=t.text_primary if r.meets_target is False else t.text_secondary,
            zorder=5,
        )

    ax.set_xlim(0, x_max)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, max(max(ideal), max(measured)) * 1.18)
    ax.set_xlabel("workers (N)", fontsize=10, color=t.text_secondary, labelpad=8)
    ax.set_ylabel(
        "sustained throughput (completion tokens/s)",
        fontsize=10,
        color=t.text_secondary,
        labelpad=8,
    )
    _style_axes(ax, t)
    _legend(
        ax,
        t,
        [
            _line_key(t.series[0], "measured"),
            _line_key(t.muted, f"ideal linear ({per_worker:.0f} tok/s × N)", dashed=True),
            Patch(facecolor=t.series[0], alpha=0.10, linewidth=0, label="efficiency gap"),
        ],
    )

    # The measured efficiency is already on every marker — the footnote only
    # has to say what the bar was and whether it was cleared.
    targets = ", ".join(
        f"≥{r.target * 100:.0f}% @ N={r.workers} ({'met' if r.meets_target else 'MISSED'})"
        for r in sc.rows
        if r.meets_target is not None
    )
    p = sc.rows[0].point
    left = 0.10
    fig.subplots_adjust(left=left, right=0.955, top=0.795, bottom=0.155)
    _chrome(
        fig,
        t,
        "Throughput scaling with worker count",
        subtitle
        or (
            f"efficiency = (tput_N / N) ÷ tput_1 · baseline N={sc.baseline_workers} "
            f"at {sc.baseline_tokens_per_second:.0f} tok/s"
        ),
        (f"targets {targets} · " if targets else "")
        + f"client-side (loadgen) · {p.duration_s:.0f}s/point at near-saturation load · "
        f"table view: {stem}.csv",
        x=left,
    )
    paths = _save(fig, out_dir, stem)
    paths.append(_write_table(analyze.scaling_table(sc), out_dir, stem))
    return paths


# --------------------------------------------------------------------------- #
# Manifest -> figures
# --------------------------------------------------------------------------- #
def render_sweep(manifest: dict, out_dir: str, *, theme: str = "light") -> list[str]:
    """Render whichever figure this ``sweep.json`` is for."""
    points = analyze.points_from_manifest(manifest)
    if not points:
        raise SystemExit("[bench] manifest has no analyzable points — nothing to plot")
    kind = manifest.get("kind", "")
    if kind == "scaling":
        return plot_scaling(analyze.analyze_scaling(points), out_dir, theme=theme)
    if kind == "saturation":
        multiplier = float(manifest.get("slo_multiplier", analyze.SLO_MULTIPLIER))
        sat = analyze.analyze_saturation(points, slo_multiplier=multiplier)
        return plot_saturation(sat, out_dir, theme=theme)
    raise SystemExit(f"[bench] unknown sweep kind {kind!r} (expected saturation|scaling)")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bench.plots", description="Render the benchmark figures."
    )
    parser.add_argument("--sweep", required=True, action="append", help="sweep.json (repeatable)")
    parser.add_argument("--out", default="", help="output directory (default: beside sweep.json)")
    parser.add_argument("--theme", default="light", choices=sorted(THEMES))
    args = parser.parse_args(argv)

    for sweep_path in args.sweep:
        with open(sweep_path) as f:
            manifest = json.load(f)
        out_dir = args.out or os.path.dirname(os.path.abspath(sweep_path))
        for path in render_sweep(manifest, out_dir, theme=args.theme):
            print(f"[bench] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
