#!/usr/bin/env python3
"""E4 (6 kills) vs. an interpolated clean 4-node run at the same training budget.

No clean 4-node run exists at a 900s budget, so E1b (`multinode-1786166169`,
4 nodes, 300s) is extrapolated. E1b is the right control specifically because it
is the first clean run WITH async checkpointing — same code E4 ran. Recipe match
is asserted below, not assumed.

Extrapolation is done two ways and the spread is reported: a naive linear scaling
of E1b's average rate would fold the one-time warmup into the rate and understate
a longer run, so a decomposed model (warmup + steady step + amortised checkpoint
cycles) is computed alongside it. They agree to ~1.6%.

    python3 experiments/e4_compare_clean.py          # uses cached json in cmp/
    python3 experiments/e4_compare_clean.py --fetch  # re-pull from S3 first
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

CMP = Path(__file__).parent / "cmp"
CLEAN = "multinode-1786166169"  # E1b — clean, 4 nodes, 300s, async ckpt
E4 = "multinode-preempt-1786199662"  # E4  — 6 kills, 4 nodes, 900s
PREFIX = "multinode-1786117954"  # failcost clean arm, PRE-async-fix

# failcost measured a SINGLE node loss against its own clean control, pre-fix.
FAILCOST_1NODE_WALL = 314.4
FAILCOST_1NODE_USD = 0.351


def fetch() -> None:
    bucket = os.environ.get("SPOT_TRAIN_BUCKET")
    if not bucket:
        raise SystemExit("SPOT_TRAIN_BUCKET unset — `set -a && . ./.env && set +a` first")
    CMP.mkdir(parents=True, exist_ok=True)
    for run in (CLEAN, E4, PREFIX):
        for name in ("metrics.json", "profile.json"):
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    f"s3://{bucket}/runs/{run}/{name}",
                    str(CMP / f"{run}.{name}"),
                    "--only-show-errors",
                ],
                check=True,
            )


def load(run: str) -> tuple[dict, dict]:
    return (
        json.loads((CMP / f"{run}.profile.json").read_text()),
        json.loads((CMP / f"{run}.metrics.json").read_text()),
    )


def recipe(profile: dict) -> dict:
    m = profile.get("metrics") or {}
    return {
        k: m[k]
        for k in ("dataset", "world_size", "grad_accum_steps", "effective_global_batch")
        if k in m
    }


def usd(profile: dict) -> float:
    return sum(i["usd"] for i in profile["cost"]["instances"])


def inst_seconds(profile: dict) -> float:
    return sum(i["billed_seconds"] for i in profile["cost"]["instances"])


def project_clean(profile: dict, budget: float) -> dict:
    """Extrapolate a clean run to `budget` training-seconds, two ways."""
    ms = [s["ms_per_step"] for s in profile["loss_samples"]]
    med = st.median(ms)
    steady = st.mean([v for v in ms if v <= med * 1.8]) / 1000
    warmup = (ms[0] - med) / 1000

    # Checkpoint stalls arrive in PAIRS (a big save then a smaller one a few
    # steps later). Detect cycle starts, then amortise each cycle's excess.
    stalls = [i + 1 for i, v in enumerate(ms) if v > med * 1.8 and i > 0]
    cycles: list[list[int]] = []
    for s in stalls:
        if cycles and s - cycles[-1][-1] <= 5:
            cycles[-1].append(s)  # same checkpoint cycle (save, then its pair)
        else:
            cycles.append([s])
    excess = st.mean([sum(ms[i - 1] - med for i in c) / 1000 for c in cycles])
    period = st.mean([b[0] - a[0] for a, b in zip(cycles, cycles[1:], strict=False)])
    amortised = steady + excess / period

    d = profile["durations"]
    fixed = d["provisioning_s"] + d["final_saves_s"] + d["evaluation_s"] + d["sampling_s"]
    wall = fixed + budget
    # instances bill a little longer than the profile clock (boot + shutdown)
    overhead = inst_seconds(profile) / len(profile["cost"]["instances"]) - d["total_s"]
    nodes = len(profile["cost"]["instances"])
    isec = nodes * (wall + overhead)

    model = (budget - warmup) / amortised
    naive = budget * (len(ms) / (sum(ms) / 1000))
    return {
        "steps_model": model,
        "steps_naive": naive,
        "steps": (model + naive) / 2,
        "spread_pct": abs(model - naive) / model * 100,
        "steady_s": steady,
        "warmup_s": warmup,
        "amortised_s": amortised,
        "cycle_excess_s": excess,
        "cycle_period_steps": period,
        "wall": wall,
        "fixed_s": fixed,
        "inst_s": isec,
        "usd": isec * profile["cost"]["instances"][0]["hourly_usd"] / 3600,
    }


def fit_loss(profile: dict, frac: float = 0.4):
    """loss ~ a + b*ln(step), fitted on the smooth tail. Returns (f, max_resid)."""
    pts = sorted((s["step"], s["loss"]) for s in profile["loss_samples"] if s["loss"] is not None)
    tail = [(s, ll) for s, ll in pts if s >= frac * pts[-1][0]]
    n = len(tail)
    sx = sum(math.log(s) for s, _ in tail)
    sy = sum(ll for _, ll in tail)
    sxx = sum(math.log(s) ** 2 for s, _ in tail)
    sxy = sum(math.log(s) * ll for s, ll in tail)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    f = lambda s: a + b * math.log(s)  # noqa: E731
    return f, max(abs(ll - f(s)) for s, ll in tail), (a, b)


def main() -> int:
    if "--fetch" in sys.argv:
        fetch()
    cp, cm = load(CLEAN)
    ep, em = load(E4)

    if recipe(cp) != recipe(ep):
        raise SystemExit(f"recipe mismatch — not comparable:\n  {recipe(cp)}\n  {recipe(ep)}")
    print(f"recipe (identical in both): {recipe(cp)}\n")

    budget = em["trained_seconds_total"]
    pr = project_clean(cp, budget)
    print(
        f"clean baseline = {CLEAN} (4 nodes, {cm['trained_seconds_total']:.0f}s) "
        f"extrapolated to {budget:.0f} train-s"
    )
    print(
        f"  steady {pr['steady_s']:.3f}s/step, warmup {pr['warmup_s']:.1f}s, "
        f"ckpt cycle +{pr['cycle_excess_s']:.1f}s every {pr['cycle_period_steps']:.1f} steps"
    )
    print(f"  -> amortised {pr['amortised_s']:.3f}s/step")
    print(
        f"  steps: model {pr['steps_model']:.1f} | naive {pr['steps_naive']:.1f} "
        f"| spread {pr['spread_pct']:.1f}% -> using {pr['steps']:.0f}\n"
    )

    e_wall = ep["durations"]["total_s"]
    e_usd, e_isec, e_steps = usd(ep), inst_seconds(ep), em["steps"]

    print(f"{'metric':24}{'clean':>12}{'E4':>12}{'delta':>12}")

    def row(name, c, e, fmt="{:.1f}", pct=True):
        d = e - c
        print(
            f"  {name:22}{fmt.format(c):>12}{fmt.format(e):>12}{fmt.format(d):>12}"
            + (f"  ({d / c * 100:+.1f}%)" if pct and c else "")
        )

    row("steps", pr["steps"], e_steps)
    row("wall clock s", pr["wall"], e_wall)
    row("cost usd", pr["usd"], e_usd, "{:.3f}")
    row("instance-seconds", pr["inst_s"], e_isec)
    row("inst-s per step", pr["inst_s"] / pr["steps"], e_isec / e_steps, "{:.2f}")
    row("usd per step", pr["usd"] / pr["steps"], e_usd / e_steps, "{:.5f}")
    row("goodput %", budget / pr["wall"] * 100, budget / e_wall * 100, "{:.1f}", False)

    f, resid, (a, b) = fit_loss(ep)
    print(f"\nloss fit: {a:.4f} {b:+.4f}*ln(step), max resid {resid:.4f}")
    print(
        f"  @195 predicted {f(195):.4f} vs actual {em['val_loss']:.4f} (val), "
        f"{ep['loss_samples'][-1]['loss']:.4f} (train)"
    )
    print(
        f"  @{pr['steps']:.0f} clean would reach ~{f(pr['steps']):.4f} "
        f"-> deficit {f(195) - f(pr['steps']):+.4f}"
    )

    dw, du, ds = e_wall - pr["wall"], e_usd - pr["usd"], e_steps - pr["steps"]
    print("\nnormalised:")
    for n, lab in ((6, "kill"), (3, "pair-event")):
        print(f"  per {lab:11} wall {dw / n:+7.1f}s  cost {du / n:+.3f}  steps {ds / n:+.1f}")
    print(
        f"\nvs pre-fix failcost (ONE node lost): wall +{FAILCOST_1NODE_WALL}s, "
        f"cost +${FAILCOST_1NODE_USD}"
    )
    print(
        f"  per node lost: {FAILCOST_1NODE_WALL:.0f}s -> {dw / 6:.0f}s "
        f"({FAILCOST_1NODE_WALL / (dw / 6):.1f}x better)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
