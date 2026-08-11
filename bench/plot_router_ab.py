"""Render the X3 router A/B figure from raw loadgen reports.

Reads the per-point reports written by ``.fleet/x2x3/{py,go,base}-<rps>.json``
and draws p99 latency against offered load for both routers, with each router's
knee (its highest zero-error point) marked.

Chart conventions follow ``bench/plots.py`` so the whole set reads as one
system: same palette, log-log axes named in the axis label, threshold in the
reserved status red, direct labels at the series ends, and a CSV table view
beside the figure for anyone who cannot use colour.

Usage: python -m bench.plot_router_ab [--indir .fleet/x2x3] [--out docs/experiments]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Validated pair (scripts/validate_palette.js, light mode): worst adjacent
# CVD dE 24.7, normal-vision 33.6 -- comfortably above the >=8 target.
SERIES = {"python": "#eb6834", "go": "#2a78d6"}
STATUS = "#d03b3b"

LABELS = {"python": "Python router", "go": "Go router (router-go)"}


def load_points(indir: str, prefix: str) -> list[dict]:
    """Every <prefix>-<rps>.json in indir, ascending by offered RPS."""
    pts = []
    for name in os.listdir(indir):
        if not (name.startswith(prefix + "-") and name.endswith(".json")):
            continue
        with open(os.path.join(indir, name)) as f:
            d = json.load(f)
        dur = d["duration_s"] or 1
        pts.append(
            {
                "offered": d["rps"],
                "achieved": d["succeeded"] / dur,
                "p50": d["p50_ms"],
                "p99": d["p99_ms"],
                "errors": d["failed"],
                "dropped": d["dropped"],
                # A point only counts as healthy if the fleet neither failed a
                # request nor made the client give up finding a free slot.
                "clean": d["failed"] == 0 and d["dropped"] == 0,
            }
        )
    return sorted(pts, key=lambda p: p["offered"])


def knee(points: list[dict]) -> dict | None:
    """Top of the LEADING run of clean points.

    Walk from the lowest offered load and stop at the first unhealthy point, so
    a lucky sample above a breakdown cannot report capacity the router does not
    have. Same rule as bench.analyze.
    """
    best = None
    for p in points:
        if not p["clean"]:
            break
        best = p
    return best


def render(series: dict[str, list[dict]], ceiling: list[dict], out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    knees = {}
    for key, pts in series.items():
        clean = [p for p in pts if p["clean"]]
        colour = SERIES[key]
        ax.plot(
            [p["offered"] for p in clean],
            [p["p99"] for p in clean],
            color=colour,
            linewidth=2,
            zorder=3,
        )
        ax.scatter(
            [p["offered"] for p in clean],
            [p["p99"] for p in clean],
            s=64,
            color=colour,
            edgecolors=SURFACE,
            linewidths=2,
            zorder=4,
        )
        # Points where the router failed requests: hollow, so a broken point is
        # never mistaken for a measurement.
        bad = [p for p in pts if not p["clean"]]
        if bad:
            ax.scatter(
                [p["offered"] for p in bad],
                [p["p99"] for p in bad],
                s=64,
                facecolors="none",
                edgecolors=colour,
                linewidths=2,
                zorder=4,
            )
        k = knee(pts)
        knees[key] = k
        if k:
            # Nudge the two knee labels apart: the Python knee sits on a steeply
            # rising segment, so centring it puts the text on the line.
            dx, ha = (-12, "right") if key == "python" else (12, "left")
            ax.annotate(
                f"knee {k['offered']:,} rps",
                xy=(k["offered"], k["p99"]),
                xytext=(dx, -18),
                textcoords="offset points",
                ha=ha,
                color=TEXT_SECONDARY,
                fontsize=10,
            )
        if clean:
            last = clean[-1]
            ax.annotate(
                LABELS[key],
                xy=(last["offered"], last["p99"]),
                xytext=(10, 4),
                textcoords="offset points",
                color=TEXT_SECONDARY,
                fontsize=11,
                va="center",
            )

    # Harness ceiling: the highest load the client itself sustained cleanly with
    # no router in the path. Any knee at or above this line would be measuring
    # loadgen, not the router.
    ceil_clean = [p for p in ceiling if p["clean"]]
    if ceil_clean:
        top = max(p["offered"] for p in ceil_clean)
        ax.axvline(top, color=STATUS, linestyle="--", linewidth=2, zorder=2)
        # Anchored at the bottom of the plot and rotated along the rule: the
        # top-right corner is where the blown-up points land, so a horizontal
        # label there collides with the data it is meant to contextualise.
        ax.annotate(
            f"harness ceiling {top:,} rps — no router in path",
            xy=(top, 0.02),
            # Blended transform: x in data coords so the label tracks the rule,
            # y as an axes fraction so it sits just above the x-axis regardless
            # of how the log y-limits autoscale.
            xycoords=("data", "axes fraction"),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            rotation=90,
            color=STATUS,
            fontsize=9.5,
            annotation_clip=False,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offered load (requests/s, log scale)", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylabel("p99 latency (ms, log scale)", color=TEXT_SECONDARY, fontsize=11)
    fig.text(
        0.5,
        0.955,
        "Router overhead against zero-latency upstreams",
        ha="center",
        color=TEXT_PRIMARY,
        fontsize=17,
    )
    sub = " · ".join(
        f"{LABELS[k]} knee {knees[k]['offered']:,} rps" for k in ("python", "go") if knees.get(k)
    )
    fig.text(0.5, 0.9, sub, ha="center", color=TEXT_SECONDARY, fontsize=11.5)
    fig.text(
        0.5,
        0.025,
        "client-side (loadgen) · 4 stub workers, no model · hollow = router failed requests "
        "· single 8-core box, all processes contending · table view: router-ab.csv",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_SECONDARY)
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.08, right=0.97)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"router-ab.{ext}")
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)

    csv_path = os.path.join(out_dir, "router-ab.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["router", "offered_rps", "achieved_rps", "p50_ms", "p99_ms", "errors", "dropped"]
        )
        for key, pts in series.items():
            for p in pts:
                w.writerow(
                    [
                        key,
                        p["offered"],
                        f"{p['achieved']:.0f}",
                        f"{p['p50']:.2f}",
                        f"{p['p99']:.2f}",
                        p["errors"],
                        p["dropped"],
                    ]
                )
    paths.append(csv_path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", default=".fleet/x2x3")
    ap.add_argument("--out", default="docs/experiments")
    a = ap.parse_args()
    series = {"python": load_points(a.indir, "py"), "go": load_points(a.indir, "go")}
    ceiling = load_points(a.indir, "base")
    for p in render(series, ceiling, a.out):
        print(p)


if __name__ == "__main__":
    main()
