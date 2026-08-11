"""Sweep runner — drives the Go loadgen once per sweep point.

    python -m bench.sweep saturation --url http://localhost:8000 \
        --rps 1,2,4,8,16,32 --workers 1 --duration 60 --dry-run
    python -m bench.sweep scaling    --url http://localhost:8000 \
        --workers 1,2,4,8 --rps-per-worker 8 --duration 60 --dry-run
    python -m bench.sweep report --sweep .fleet/bench/e2-saturation/sweep.json --plot

Shape follows the rest of the repo: the *planning* is pure and unit-tested
(:func:`plan_saturation`, :func:`plan_scaling`, :func:`loadgen_command`,
:func:`planned_commands`), and only a few thin functions actually run
anything. ``--dry-run`` prints the exact command list and executes nothing —
the same courtesy ``spot-orchestrate`` extends to anything that spends
resources.

Each sweep point writes the loadgen's own JSON report into the output
directory; ``sweep.json`` is the manifest that ties them together, recording
the sweep parameters plus every point's report inline so the analysis and the
charts need exactly one input file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

from . import analyze

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADGEN_SRC_DIR = os.path.join(REPO_ROOT, "loadgen")
DEFAULT_BINARY = os.path.join(REPO_ROOT, ".fleet", "loadgen-bin")
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, ".fleet", "bench")

DEFAULT_MAX_TOKENS = 64  # the plan measures L0 at max_tokens=64
DEFAULT_TIMEOUT_S = 60
DEFAULT_SETTLE_S = 20.0  # let a resized fleet's heartbeats land before loading it

SATURATION = "saturation"
SCALING = "scaling"


# --------------------------------------------------------------------------- #
# Plan (pure)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweepPoint:
    """One loadgen invocation."""

    label: str
    offered_rps: float
    workers: int
    duration_s: int
    concurrency: int
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_s: int = DEFAULT_TIMEOUT_S
    ramp_s: int = 0
    warmup: bool = False


@dataclass(frozen=True)
class SweepPlan:
    """A whole sweep: what to run, where the artifacts land."""

    kind: str
    url: str
    out_dir: str
    binary: str
    points: tuple[SweepPoint, ...]
    slo_multiplier: float = analyze.SLO_MULTIPLIER
    scale: bool = False
    scale_template: str = ""
    local: bool = True
    settle_s: float = DEFAULT_SETTLE_S
    notes: dict = field(default_factory=dict)


def concurrency_for(
    rps: float, override: int = 0, *, floor: int = 64, headroom: float = 4.0
) -> int:
    """In-flight cap for an offered rate.

    The generator is open-loop: it dispatches on schedule and counts a
    *dropped* request whenever no client worker is free. That number is only
    meaningful as a fleet signal if the client pool is comfortably larger than
    ``rps x expected latency`` — otherwise the client throttles itself and the
    saturation curve measures the benchmark, not the fleet.
    """
    if override > 0:
        return int(override)
    return max(floor, int(math.ceil(rps * headroom)))


def _label(prefix: str, value: float) -> str:
    return f"{prefix}{value:g}".replace(".", "p")


def plan_saturation(
    *,
    url: str,
    rps_list: list[float],
    workers: int = 1,
    duration_s: int = 60,
    concurrency: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    ramp_s: int = 0,
    out_dir: str = "",
    binary: str = DEFAULT_BINARY,
    warmup_s: int = 0,
    slo_multiplier: float = analyze.SLO_MULTIPLIER,
) -> SweepPlan:
    """E2 — fixed worker count, sweep offered RPS.

    ``L0`` is read off the lowest-RPS point, so keep RPS=1 in the list. An
    optional warmup point (``warmup_s`` > 0) runs first and is excluded from
    the analysis: the first generate on a cold worker pays one-time init (CUDA
    context, kernel autotune) and would otherwise inflate L0 and drag the whole
    SLO up with it.
    """
    if not rps_list:
        raise ValueError("plan_saturation needs at least one RPS")
    points: list[SweepPoint] = []
    if warmup_s > 0:
        points.append(
            SweepPoint(
                label="warmup",
                offered_rps=min(rps_list),
                workers=workers,
                duration_s=warmup_s,
                concurrency=concurrency_for(min(rps_list), concurrency),
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                warmup=True,
            )
        )
    for rps in sorted(rps_list):
        points.append(
            SweepPoint(
                label=_label("rps", rps),
                offered_rps=float(rps),
                workers=workers,
                duration_s=duration_s,
                concurrency=concurrency_for(rps, concurrency),
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                ramp_s=ramp_s,
            )
        )
    return SweepPlan(
        kind=SATURATION,
        url=url,
        out_dir=out_dir or os.path.join(DEFAULT_OUT_ROOT, "e2-saturation"),
        binary=binary,
        points=tuple(points),
        slo_multiplier=slo_multiplier,
        scale=False,
        notes={"workers": workers, "rps_list": [float(r) for r in sorted(rps_list)]},
    )


def plan_scaling(
    *,
    url: str,
    worker_list: list[int],
    rps_per_worker: float,
    duration_s: int = 60,
    concurrency: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    ramp_s: int = 0,
    out_dir: str = "",
    binary: str = DEFAULT_BINARY,
    scale: bool = True,
    scale_template: str = "",
    local: bool = True,
    settle_s: float = DEFAULT_SETTLE_S,
    slo_multiplier: float = analyze.SLO_MULTIPLIER,
) -> SweepPlan:
    """E5 — near-saturation load, sweep worker count.

    Offered load scales with the fleet (``rps_per_worker x N``) so every point
    sits at the same *per-worker* pressure; set ``rps_per_worker`` from E2's
    knee (``C1``, ~80-90% of it) so the fleet is busy but not collapsing —
    efficiency measured on an idle fleet is meaningless.
    """
    if not worker_list:
        raise ValueError("plan_scaling needs at least one worker count")
    points = []
    for n in sorted(worker_list):
        rps = rps_per_worker * n
        points.append(
            SweepPoint(
                label=f"n{n}",
                offered_rps=float(rps),
                workers=int(n),
                duration_s=duration_s,
                concurrency=concurrency_for(rps, concurrency),
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                ramp_s=ramp_s,
            )
        )
    return SweepPlan(
        kind=SCALING,
        url=url,
        out_dir=out_dir or os.path.join(DEFAULT_OUT_ROOT, "e5-scaling"),
        binary=binary,
        points=tuple(points),
        slo_multiplier=slo_multiplier,
        scale=scale,
        scale_template=scale_template,
        local=local,
        settle_s=settle_s,
        notes={
            "worker_list": [int(n) for n in sorted(worker_list)],
            "rps_per_worker": float(rps_per_worker),
        },
    )


def report_path(plan: SweepPlan, point: SweepPoint) -> str:
    return os.path.join(plan.out_dir, f"{point.label}.json")


def manifest_path(plan: SweepPlan) -> str:
    return os.path.join(plan.out_dir, "sweep.json")


def loadgen_command(plan: SweepPlan, point: SweepPoint) -> list[str]:
    """The exact argv for one sweep point.

    Flag names are the loadgen's own (``loadgen/main.go``): ``-url -rps
    -concurrency -duration -ramp -max-tokens -timeout -out``. Durations are Go
    duration strings.
    """
    cmd = [
        plan.binary,
        "-url",
        plan.url,
        "-rps",
        f"{point.offered_rps:g}",
        "-concurrency",
        str(point.concurrency),
        "-duration",
        f"{point.duration_s}s",
    ]
    if point.ramp_s > 0:
        cmd += ["-ramp", f"{point.ramp_s}s"]
    cmd += [
        "-max-tokens",
        str(point.max_tokens),
        "-timeout",
        f"{point.timeout_s}s",
        "-out",
        report_path(plan, point),
    ]
    return cmd


def scale_commands(plan: SweepPlan, workers: int) -> list[list[str]]:
    """How to resize the fleet to ``workers`` before the next point.

    Default is the repo's own fleet CLI (down, then up at the new size —
    the local fleet has no in-place resize). ``--scale-cmd`` swaps in anything
    else that takes a replica count, e.g.
    ``kubectl scale deploy/worker --replicas={n}``.
    """
    if not plan.scale:
        return []
    if plan.scale_template:
        return [shlex.split(plan.scale_template.format(n=workers))]
    where = ["--local"] if plan.local else []
    return [
        ["spot-orchestrate", "fleet", "down", *where],
        ["spot-orchestrate", "fleet", "up", *where, "--workers", str(workers)],
    ]


def planned_commands(plan: SweepPlan) -> list[list[str]]:
    """Every command the sweep would run, in order — what ``--dry-run`` prints.

    The fleet is only resized when the worker count actually changes, so a
    fixed-N sweep costs zero restarts.
    """
    cmds: list[list[str]] = []
    current_workers: int | None = None
    for point in plan.points:
        if plan.scale and point.workers != current_workers:
            cmds.extend(scale_commands(plan, point.workers))
            current_workers = point.workers
        cmds.append(loadgen_command(plan, point))
    return cmds


def build_manifest(plan: SweepPlan, entries: list[dict]) -> dict:
    """The ``sweep.json`` artifact: parameters + every point's raw report."""
    return {
        "kind": plan.kind,
        "url": plan.url,
        "created_unix": entries[0].get("started_unix", 0.0) if entries else 0.0,
        "slo_multiplier": plan.slo_multiplier,
        "params": {
            "out_dir": plan.out_dir,
            "scale": plan.scale,
            "scale_template": plan.scale_template,
            "local": plan.local,
            "settle_s": plan.settle_s,
            **plan.notes,
        },
        "points": entries,
    }


def manifest_entry(
    plan: SweepPlan,
    point: SweepPoint,
    report: dict | None,
    *,
    exit_code: int = 0,
    started_unix: float = 0.0,
) -> dict:
    return {
        "label": point.label,
        "offered_rps": point.offered_rps,
        "workers": point.workers,
        "duration_s": point.duration_s,
        "concurrency": point.concurrency,
        "max_tokens": point.max_tokens,
        "warmup": point.warmup,
        "report_path": report_path(plan, point),
        "exit_code": exit_code,
        "started_unix": started_unix,
        "report": report,
    }


# --------------------------------------------------------------------------- #
# Effects (thin — every subprocess in this repo lives in a function this small)
# --------------------------------------------------------------------------- #
def build_loadgen(binary: str = DEFAULT_BINARY) -> str:
    """``go build`` the loadgen once and return the binary path."""
    if shutil.which("go") is None:
        raise SystemExit("bench: the Go toolchain must be on PATH (bench builds loadgen/)")
    os.makedirs(os.path.dirname(binary), exist_ok=True)
    subprocess.run(["go", "build", "-o", binary, "."], cwd=LOADGEN_SRC_DIR, check=True)
    return binary


def locate_loadgen(binary: str = DEFAULT_BINARY, *, build: bool = True) -> str:
    """Existing binary if it is there, otherwise build it."""
    if os.path.exists(binary) and not build:
        return binary
    return build_loadgen(binary)


def _run(cmd: list[str]) -> int:
    print(f"[bench] $ {shlex.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def _read_report(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"[bench] WARNING: no usable report at {path}: {e}", file=sys.stderr)
        return None


def run_sweep(plan: SweepPlan, *, dry_run: bool = False, settle: bool = True) -> dict | None:
    """Execute the plan and write ``sweep.json``. ``dry_run`` prints only."""
    if dry_run:
        print(f"[bench] dry-run: {plan.kind} sweep, {len(plan.points)} points -> {plan.out_dir}")
        for cmd in planned_commands(plan):
            print(f"  {shlex.join(cmd)}")
        if plan.scale and plan.settle_s > 0:
            print(f"  (plus a {plan.settle_s:g}s settle after each resize)")
        print("[bench] dry-run: nothing executed")
        return None

    os.makedirs(plan.out_dir, exist_ok=True)
    entries: list[dict] = []
    current_workers: int | None = None
    for i, point in enumerate(plan.points, start=1):
        if plan.scale and point.workers != current_workers:
            for cmd in scale_commands(plan, point.workers):
                if _run(cmd) != 0:
                    raise SystemExit(f"[bench] fleet resize failed: {shlex.join(cmd)}")
            current_workers = point.workers
            if settle and plan.settle_s > 0:
                print(f"[bench] settling {plan.settle_s:g}s after resize to N={point.workers}")
                time.sleep(plan.settle_s)
        tag = " (warmup, excluded from analysis)" if point.warmup else ""
        print(
            f"[bench] point {i}/{len(plan.points)}: {point.label} — "
            f"{point.offered_rps:g} rps, N={point.workers}, {point.duration_s}s{tag}"
        )
        started = time.time()
        # The loadgen exits 1 when any request failed (scripts assert the
        # zero-visible-errors criterion) — that is data, not a reason to abort
        # the sweep, so the code is recorded and the report is still read.
        code = _run(loadgen_command(plan, point))
        report = _read_report(report_path(plan, point))
        entries.append(manifest_entry(plan, point, report, exit_code=code, started_unix=started))

    manifest = build_manifest(plan, entries)
    with open(manifest_path(plan), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[bench] manifest: {manifest_path(plan)}")
    return manifest


def summarize(manifest: dict) -> str:
    """Terminal summary for a finished sweep (pure-ish: dict in, text out)."""
    points = analyze.points_from_manifest(manifest)
    if not points:
        return "[bench] no analyzable points in this sweep"
    multiplier = float(manifest.get("slo_multiplier", analyze.SLO_MULTIPLIER))
    if manifest.get("kind") == SCALING:
        return analyze.render_scaling(analyze.analyze_scaling(points))
    return analyze.render_saturation(analyze.analyze_saturation(points, slo_multiplier=multiplier))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(" ", "").split(",") if x]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def _plot(manifest: dict, out_dir: str, theme: str) -> None:
    try:
        from . import plots
    except ImportError as e:  # matplotlib is the bench extra
        print(
            f"[bench] plotting needs matplotlib ({e}) — install with: pip install -e '.[bench]'",
            file=sys.stderr,
        )
        return
    for path in plots.render_sweep(manifest, out_dir, theme=theme):
        print(f"[bench] figure: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.sweep",
        description="Benchmark sweeps for the inference fleet (E2 saturation, E5 scaling).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default="http://localhost:8000", help="router base URL")
    common.add_argument("--duration", type=int, default=60, help="seconds per sweep point")
    common.add_argument(
        "--concurrency", type=int, default=0, help="in-flight cap (0 = auto from RPS)"
    )
    common.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    common.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="per-request, s")
    common.add_argument("--ramp", type=int, default=0, help="ramp seconds per point (0 = off)")
    common.add_argument("--out", default="", help="artifact directory")
    common.add_argument("--binary", default=DEFAULT_BINARY, help="loadgen binary path")
    common.add_argument(
        "--no-build", action="store_true", help="use an existing loadgen binary as-is"
    )
    common.add_argument(
        "--slo-multiplier", type=float, default=analyze.SLO_MULTIPLIER, help="SLO = k x L0"
    )
    common.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    common.add_argument("--plot", action="store_true", help="render the figures when done")
    common.add_argument("--theme", default="light", choices=["light", "dark"])

    sat = sub.add_parser(SATURATION, parents=[common], help="E2 — sweep offered RPS")
    sat.add_argument("--rps", default="1,2,4,8,16,32", help="comma-separated offered rates")
    sat.add_argument("--workers", type=int, default=1, help="fixed fleet size for this sweep")
    sat.add_argument("--warmup", type=int, default=0, help="warmup seconds before point 1")

    sca = sub.add_parser(SCALING, parents=[common], help="E5 — sweep worker count")
    sca.add_argument("--workers", default="1,2,4,8", help="comma-separated worker counts")
    sca.add_argument(
        "--rps-per-worker", type=float, required=True, help="offered rps per worker (~0.8 x C1)"
    )
    sca.add_argument(
        "--no-scale", action="store_true", help="do not resize the fleet between points"
    )
    sca.add_argument(
        "--scale-cmd", default="", help="resize command template, '{n}' = worker count"
    )
    sca.add_argument("--cloud", action="store_true", help="resize the CLOUD fleet (spends money)")
    sca.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S, help="pause after resize")

    rep = sub.add_parser("report", help="re-analyze (and re-plot) an existing sweep.json")
    rep.add_argument("--sweep", required=True, help="path to sweep.json")
    rep.add_argument("--out", default="", help="figure directory (default: beside sweep.json)")
    rep.add_argument("--plot", action="store_true")
    rep.add_argument("--theme", default="light", choices=["light", "dark"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "report":
        with open(args.sweep) as f:
            manifest = json.load(f)
        print(summarize(manifest))
        if args.plot:
            _plot(manifest, args.out or os.path.dirname(os.path.abspath(args.sweep)), args.theme)
        return 0

    binary = args.binary if args.dry_run else locate_loadgen(args.binary, build=not args.no_build)

    if args.command == SATURATION:
        plan = plan_saturation(
            url=args.url,
            rps_list=_floats(args.rps),
            workers=args.workers,
            duration_s=args.duration,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            ramp_s=args.ramp,
            out_dir=args.out,
            binary=binary,
            warmup_s=args.warmup,
            slo_multiplier=args.slo_multiplier,
        )
    else:
        plan = plan_scaling(
            url=args.url,
            worker_list=_ints(args.workers),
            rps_per_worker=args.rps_per_worker,
            duration_s=args.duration,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            ramp_s=args.ramp,
            out_dir=args.out,
            binary=binary,
            scale=not args.no_scale,
            scale_template=args.scale_cmd,
            local=not args.cloud,
            settle_s=args.settle,
            slo_multiplier=args.slo_multiplier,
        )
        if args.cloud and not args.dry_run:
            print(
                "[bench] NOTE: --cloud resizes the EC2 fleet — this launches and "
                "terminates real instances."
            )

    manifest = run_sweep(plan, dry_run=args.dry_run)
    if manifest is None:
        return 0
    print()
    print(summarize(manifest))
    if args.plot:
        _plot(manifest, plan.out_dir, args.theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
