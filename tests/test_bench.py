"""bench/ — pure analysis of loadgen reports + sweep planning.

CPU-only and offline: the fixtures below are hand-built dicts shaped exactly
like the JSON ``loadgen/main.go`` writes, so nothing here builds Go, launches a
subprocess, or touches the network.
"""

from __future__ import annotations

import os
import sys

# bench/ is a repo-root package, not an installed one — and it has to come FIRST
# on sys.path: conftest.py prepends third_party/nanoGPT, which ships its own
# bench.py (nanoGPT's benchmarking script) that would otherwise win the import.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import pytest  # noqa: E402

from bench import analyze, sweep  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures shaped like loadgen's report struct
# --------------------------------------------------------------------------- #
def make_report(
    *,
    rps: float,
    p50: float,
    p99: float | None = None,
    p90: float | None = None,
    p95: float | None = None,
    duration_s: float = 60.0,
    requests: int | None = None,
    failed: int = 0,
    dropped: int = 0,
    tokens_per_second: float = 100.0,
    concurrency: int = 64,
    kill_at_s: float | None = None,
) -> dict:
    """A loadgen report. Keys are verbatim from loadgen/main.go's json tags."""
    requests = int(rps * duration_s) if requests is None else requests
    succeeded = requests - failed
    p99 = 2.0 * p50 if p99 is None else p99
    p95 = p50 + 0.7 * (p99 - p50) if p95 is None else p95
    p90 = p50 + 0.4 * (p99 - p50) if p90 is None else p90
    report = {
        "url": "http://localhost:8000",
        "start_unix": 1_700_000_000.0,
        "rps": rps,
        "concurrency": concurrency,
        "duration_s": duration_s,
        "requests": requests,
        "succeeded": succeeded,
        "failed": failed,
        "dropped": dropped,
        "error_rate": (failed / requests) if requests else 0.0,
        "completion_tokens": int(tokens_per_second * duration_s),
        "tokens_per_second": tokens_per_second,
        "mean_ms": p50 * 1.1,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "per_second": [
            {
                "t": t,
                "sent": int(rps),
                "ok": int(rps),
                "errors": 0,
                "dropped": 0,
                "p99_ms": p99,
                "mean_ms": p50 * 1.1,
            }
            for t in range(int(duration_s))
        ],
    }
    if kill_at_s is not None:
        report["kill_at_s"] = kill_at_s
        report["kill_cmd"] = "spot-orchestrate fleet kill-worker --local"
    return report


def point(rps: float, p50: float, p99: float, *, workers: int = 1, **kw) -> analyze.Point:
    return analyze.parse_report(
        make_report(rps=rps, p50=p50, p99=p99, **kw), workers=workers, label=f"rps{rps:g}"
    )


def saturation_points() -> list[analyze.Point]:
    """A textbook curve: flat until 8 rps, blowing past 3xL0 (=300ms) at 16."""
    return [
        point(1, 100.0, 120.0),
        point(2, 102.0, 140.0),
        point(4, 110.0, 190.0),
        point(8, 130.0, 280.0),  # last point under the 300ms SLO
        point(16, 260.0, 900.0),
        point(32, 800.0, 4000.0),
    ]


# --------------------------------------------------------------------------- #
# Report parsing — the real schema field names
# --------------------------------------------------------------------------- #
def test_parse_report_reads_the_loadgen_schema():
    report = make_report(rps=8, p50=130.0, p99=280.0, failed=3, dropped=5, kill_at_s=30.0)
    p = analyze.parse_report(report, workers=2, label="n2")

    assert p.label == "n2" and p.workers == 2
    assert p.offered_rps == 8.0  # "rps"
    assert p.concurrency == 64  # "concurrency"
    assert p.duration_s == 60.0  # "duration_s"
    assert p.requests == 480  # "requests"
    assert p.succeeded == 477  # "succeeded"
    assert p.failed == 3  # "failed"
    assert p.dropped == 5  # "dropped"
    assert p.error_rate == pytest.approx(3 / 480)  # "error_rate"
    assert p.completion_tokens == 6000  # "completion_tokens"
    assert p.tokens_per_second == 100.0  # "tokens_per_second"
    assert (p.p50_ms, p.p90_ms, p.p95_ms, p.p99_ms) == (130.0, 190.0, 235.0, 280.0)
    assert p.mean_ms == pytest.approx(143.0)
    assert p.start_unix == 1_700_000_000.0  # "start_unix"
    assert p.kill_at_s == 30.0  # "kill_at_s"
    assert p.url == "http://localhost:8000"
    assert p.achieved_rps == pytest.approx(477 / 60)


def test_parse_report_survives_a_run_with_no_successes():
    """Go leaves every latency at 0 when nothing succeeded — parse, don't crash."""
    report = make_report(rps=32, p50=0.0, p99=0.0, requests=100, failed=100)
    p = analyze.parse_report(report)
    assert p.succeeded == 0 and p.error_rate == 1.0
    assert p.p99_p50_ratio == float("inf")
    assert not analyze.holds_slo(p, slo=300.0)  # zero p99 is not a passing point


def test_offered_rps_override_wins_over_the_report():
    """The manifest is the source of truth for what we *asked* for."""
    p = analyze.parse_report(make_report(rps=8, p50=100.0), offered_rps=8.5)
    assert p.offered_rps == 8.5


def test_points_from_manifest_skips_warmup():
    manifest = {
        "kind": "saturation",
        "points": [
            {
                "label": "warmup",
                "offered_rps": 1,
                "workers": 1,
                "warmup": True,
                "report": make_report(rps=1, p50=900.0),
            },
            {
                "label": "rps1",
                "offered_rps": 1,
                "workers": 1,
                "warmup": False,
                "report": make_report(rps=1, p50=100.0),
            },
            {"label": "rps2", "offered_rps": 2, "workers": 1, "report": None},  # crashed point
        ],
    }
    points = analyze.points_from_manifest(manifest)
    assert [p.label for p in points] == ["rps1"]
    assert analyze.points_from_manifest(manifest, include_warmup=True)[0].p50_ms == 900.0


# --------------------------------------------------------------------------- #
# L0 and the SLO
# --------------------------------------------------------------------------- #
def test_l0_is_the_lowest_rps_single_worker_p50():
    assert analyze.l0_ms(saturation_points()) == 100.0


def test_l0_ignores_multi_worker_points():
    points = [point(1, 40.0, 60.0, workers=4), point(1, 100.0, 120.0, workers=1)]
    assert analyze.l0_ms(points) == 100.0


def test_l0_needs_a_single_worker_point_with_successes():
    with pytest.raises(ValueError, match="cannot derive L0"):
        analyze.l0_ms([point(1, 40.0, 60.0, workers=4)])
    dead = analyze.parse_report(make_report(rps=1, p50=0.0, requests=10, failed=10))
    with pytest.raises(ValueError, match="cannot derive L0"):
        analyze.l0_ms([dead])


def test_slo_is_three_times_l0():
    assert analyze.slo_ms(100.0) == 300.0
    assert analyze.slo_ms(100.0, multiplier=2.0) == 200.0
    assert analyze.SLO_MULTIPLIER == 3.0


def test_holds_slo_rejects_an_error_heavy_point():
    fast_but_broken = analyze.parse_report(
        make_report(rps=8, p50=100.0, p99=150.0, requests=100, failed=40)
    )
    assert fast_but_broken.p99_ms <= 300.0
    assert not analyze.holds_slo(fast_but_broken, 300.0)


# --------------------------------------------------------------------------- #
# The knee (C1)
# --------------------------------------------------------------------------- #
def test_knee_is_the_last_rps_under_the_slo():
    sat = analyze.analyze_saturation(saturation_points())
    assert sat.l0_ms == 100.0
    assert sat.slo_ms == 300.0
    assert sat.knee.rps == 8.0
    assert sat.knee.p99_ms == 280.0
    assert sat.knee.beyond_sweep is False
    assert [r.within_slo for r in sat.rows] == [True, True, True, True, False, False]
    assert [r.is_knee for r in sat.rows] == [False, False, False, True, False, False]


def test_knee_stops_at_the_first_violation_not_a_later_lucky_point():
    points = [
        point(1, 100.0, 120.0),
        point(2, 110.0, 200.0),
        point(4, 400.0, 1200.0),  # breakdown
        point(8, 120.0, 250.0),  # a fluke sample above the breakdown
    ]
    knee = analyze.find_knee(points, analyze.slo_ms(100.0))
    assert knee.rps == 2.0


def test_knee_when_every_point_holds_the_slo():
    points = [point(1, 100.0, 120.0), point(2, 101.0, 130.0), point(4, 105.0, 180.0)]
    knee = analyze.find_knee(points, analyze.slo_ms(100.0))
    assert knee.rps == 4.0
    assert knee.beyond_sweep is True  # the real C1 is >= 4; the sweep did not find it
    assert "sweep higher" in analyze.render_saturation(analyze.analyze_saturation(points))


def test_knee_when_no_point_holds_the_slo():
    points = [point(1, 100.0, 120.0), point(2, 500.0, 2000.0), point(4, 900.0, 5000.0)]
    knee = analyze.find_knee(points, slo=110.0)  # a punishing SLO: even RPS=1 breaks it
    assert knee.rps is None and knee.p99_ms is None and knee.point is None
    assert knee.beyond_sweep is False
    assert knee.found is False


def test_find_knee_rejects_an_empty_sweep():
    with pytest.raises(ValueError, match="empty sweep"):
        analyze.find_knee([], slo=300.0)


# --------------------------------------------------------------------------- #
# p99 / p50 ratio
# --------------------------------------------------------------------------- #
def test_p99_p50_ratio():
    assert point(4, 100.0, 250.0).p99_p50_ratio == pytest.approx(2.5)
    assert point(4, 100.0, 100.0).p99_p50_ratio == pytest.approx(1.0)


def test_p99_p50_ratio_surfaces_in_the_saturation_rows():
    sat = analyze.analyze_saturation(saturation_points())
    assert sat.rows[0].p99_p50_ratio == pytest.approx(1.2)
    assert sat.rows[-1].p99_p50_ratio == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# Scaling efficiency
# --------------------------------------------------------------------------- #
def scaling_points(tps_by_n: dict[int, float]) -> list[analyze.Point]:
    return [
        point(8 * n, 120.0, 300.0, workers=n, tokens_per_second=tps) for n, tps in tps_by_n.items()
    ]


def test_scaling_efficiency_is_one_for_the_baseline():
    sc = analyze.analyze_scaling(scaling_points({1: 500.0}))
    assert sc.baseline_workers == 1
    assert sc.rows[0].efficiency == pytest.approx(1.0)
    assert sc.rows[0].ideal_tokens_per_second == pytest.approx(500.0)
    assert sc.rows[0].target is None and sc.rows[0].meets_target is None


def test_scaling_efficiency_math():
    sc = analyze.analyze_scaling(scaling_points({1: 500.0, 2: 950.0, 4: 1850.0, 8: 3200.0}))
    eff = {r.workers: r.efficiency for r in sc.rows}
    assert eff[1] == pytest.approx(1.0)
    assert eff[2] == pytest.approx(0.95)  # (950/2)/500
    assert eff[4] == pytest.approx(0.925)  # (1850/4)/500
    assert eff[8] == pytest.approx(0.80)  # (3200/8)/500
    ideal = {r.workers: r.ideal_tokens_per_second for r in sc.rows}
    assert ideal == {1: 500.0, 2: 1000.0, 4: 2000.0, 8: 4000.0}
    verdicts = {r.workers: r.meets_target for r in sc.rows}
    assert verdicts == {1: None, 2: None, 4: True, 8: True}  # >=90% @4, >=80% @8
    assert sc.passed


def test_scaling_targets_are_the_plan_numbers_and_can_fail():
    assert analyze.EFFICIENCY_TARGETS == {4: 0.90, 8: 0.80}
    sc = analyze.analyze_scaling(scaling_points({1: 500.0, 4: 1700.0, 8: 2800.0}))
    verdicts = {r.workers: r.meets_target for r in sc.rows}
    assert verdicts == {1: None, 4: False, 8: False}  # 85% and 70%
    assert not sc.passed
    assert "FAIL" in analyze.render_scaling(sc)


def test_scaling_baseline_can_be_a_non_unit_worker_count():
    """A sweep that starts at N=2 normalizes per worker, so N=2 reads 100%."""
    sc = analyze.analyze_scaling(scaling_points({2: 1000.0, 4: 1800.0}))
    assert sc.baseline_workers == 2
    eff = {r.workers: r.efficiency for r in sc.rows}
    assert eff[2] == pytest.approx(1.0)
    assert eff[4] == pytest.approx(0.9)  # (1800/4) / (1000/2)


def test_scaling_keeps_the_best_run_per_worker_count():
    points = scaling_points({1: 500.0}) + [
        point(32, 120.0, 300.0, workers=4, tokens_per_second=1700.0),
        point(32, 120.0, 300.0, workers=4, tokens_per_second=1900.0),
    ]
    sc = analyze.analyze_scaling(points)
    assert [r.tokens_per_second for r in sc.rows] == [500.0, 1900.0]


def test_scaling_needs_a_baseline_with_throughput():
    with pytest.raises(ValueError, match="no tokens/s"):
        analyze.analyze_scaling(scaling_points({1: 0.0, 2: 900.0}))
    with pytest.raises(ValueError, match="at least one point"):
        analyze.analyze_scaling([])


# --------------------------------------------------------------------------- #
# Table views + terminal rendering (pure)
# --------------------------------------------------------------------------- #
def test_saturation_table_has_a_header_and_one_row_per_point():
    sat = analyze.analyze_saturation(saturation_points())
    table = analyze.saturation_table(sat)
    assert table[0][:5] == ["offered_rps", "achieved_rps", "p50_ms", "p95_ms", "p99_ms"]
    assert len(table) == 1 + len(sat.rows)
    knee_row = next(row for row in table[1:] if row[-1] == "yes")
    assert knee_row[0] == "8"


def test_render_saturation_states_l0_slo_and_the_knee():
    text = analyze.render_saturation(analyze.analyze_saturation(saturation_points()))
    assert "L0 100.0 ms" in text
    assert "SLO p99 <= 3 x L0 = 300.0 ms" in text
    assert "C1 (knee): 8 rps" in text


def test_render_scaling_lists_every_worker_count():
    sc = analyze.analyze_scaling(scaling_points({1: 500.0, 2: 950.0, 4: 1850.0}))
    text = analyze.render_scaling(sc)
    assert "baseline N=1: 500.0 tok/s" in text
    assert "95.0%" in text and "92.5%" in text
    assert "RESULT: PASS" in text


# --------------------------------------------------------------------------- #
# Sweep planning — pure, and --dry-run executes nothing
# --------------------------------------------------------------------------- #
def test_plan_saturation_builds_one_point_per_rps_sorted():
    plan = sweep.plan_saturation(
        url="http://localhost:8000",
        rps_list=[8, 1, 4],
        workers=1,
        duration_s=30,
        out_dir="/tmp/e2",
        binary="/tmp/loadgen-bin",
    )
    assert plan.kind == "saturation"
    assert [p.offered_rps for p in plan.points] == [1.0, 4.0, 8.0]
    assert [p.label for p in plan.points] == ["rps1", "rps4", "rps8"]
    assert all(p.workers == 1 and p.duration_s == 30 for p in plan.points)
    assert plan.scale is False  # a fixed-N sweep never restarts the fleet


def test_plan_saturation_warmup_is_first_and_excluded_from_analysis():
    plan = sweep.plan_saturation(
        url="http://x", rps_list=[1, 2], warmup_s=15, out_dir="/tmp/e2", binary="/b"
    )
    assert plan.points[0].warmup is True and plan.points[0].duration_s == 15
    assert [p.warmup for p in plan.points[1:]] == [False, False]


def test_loadgen_command_uses_the_real_flag_names():
    plan = sweep.plan_saturation(
        url="http://localhost:8000",
        rps_list=[4],
        duration_s=45,
        concurrency=32,
        max_tokens=64,
        timeout_s=30,
        out_dir="/tmp/e2",
        binary="/tmp/loadgen-bin",
    )
    # fmt: off
    assert sweep.loadgen_command(plan, plan.points[0]) == [
        "/tmp/loadgen-bin",
        "-url", "http://localhost:8000",
        "-rps", "4",
        "-concurrency", "32",
        "-duration", "45s",
        "-max-tokens", "64",
        "-timeout", "30s",
        "-out", os.path.join("/tmp/e2", "rps4.json"),
    ]
    # fmt: on


def test_loadgen_command_adds_ramp_only_when_asked():
    plan = sweep.plan_saturation(
        url="http://x", rps_list=[4], ramp_s=10, out_dir="/tmp/e2", binary="/b"
    )
    cmd = sweep.loadgen_command(plan, plan.points[0])
    assert cmd[cmd.index("-ramp") + 1] == "10s"
    assert cmd.index("-ramp") > cmd.index("-duration")


def test_concurrency_is_auto_sized_above_the_offered_rate():
    assert sweep.concurrency_for(4) == 64  # floor
    assert sweep.concurrency_for(40) == 160  # 4x headroom
    assert sweep.concurrency_for(40, override=8) == 8


def test_dry_run_plan_for_a_scaling_sweep_is_the_exact_command_list(capsys, tmp_path):
    out_dir = str(tmp_path / "e5")  # deliberately not created
    plan = sweep.plan_scaling(
        url="http://localhost:8000",
        worker_list=[1, 2, 4],
        rps_per_worker=6,
        duration_s=60,
        concurrency=64,
        out_dir=out_dir,
        binary="/tmp/loadgen-bin",
    )

    # fmt: off
    def load(rps, name):
        return [
            "/tmp/loadgen-bin", "-url", "http://localhost:8000", "-rps", rps,
            "-concurrency", "64", "-duration", "60s", "-max-tokens", "64",
            "-timeout", "60s", "-out", os.path.join(out_dir, name),
        ]
    # fmt: on

    assert sweep.planned_commands(plan) == [
        ["spot-orchestrate", "fleet", "down", "--local"],
        ["spot-orchestrate", "fleet", "up", "--local", "--workers", "1"],
        load("6", "n1.json"),
        ["spot-orchestrate", "fleet", "down", "--local"],
        ["spot-orchestrate", "fleet", "up", "--local", "--workers", "2"],
        load("12", "n2.json"),
        ["spot-orchestrate", "fleet", "down", "--local"],
        ["spot-orchestrate", "fleet", "up", "--local", "--workers", "4"],
        load("24", "n4.json"),
    ]

    # --dry-run prints that list and runs nothing: no output dir, no manifest.
    assert sweep.run_sweep(plan, dry_run=True) is None
    printed = capsys.readouterr().out
    assert "nothing executed" in printed
    assert "spot-orchestrate fleet up --local --workers 4" in printed
    assert not os.path.exists(out_dir)


def test_fixed_worker_count_sweep_never_resizes_the_fleet():
    plan = sweep.plan_saturation(url="http://x", rps_list=[1, 2], out_dir="/tmp/e2", binary="/b")
    assert all(cmd[0] == "/b" for cmd in sweep.planned_commands(plan))


def test_scale_command_template_takes_the_worker_count():
    plan = sweep.plan_scaling(
        url="http://x",
        worker_list=[2],
        rps_per_worker=5,
        scale_template="kubectl scale deploy/worker --replicas={n}",
        out_dir="/tmp/e5",
        binary="/b",
    )
    assert sweep.scale_commands(plan, 4) == [["kubectl", "scale", "deploy/worker", "--replicas=4"]]


def test_cloud_scaling_plan_omits_the_local_flag():
    plan = sweep.plan_scaling(
        url="http://x", worker_list=[2], rps_per_worker=5, local=False, out_dir="/o", binary="/b"
    )
    assert sweep.scale_commands(plan, 2) == [
        ["spot-orchestrate", "fleet", "down"],
        ["spot-orchestrate", "fleet", "up", "--workers", "2"],
    ]


def test_plans_reject_empty_sweeps():
    with pytest.raises(ValueError):
        sweep.plan_saturation(url="http://x", rps_list=[])
    with pytest.raises(ValueError):
        sweep.plan_scaling(url="http://x", worker_list=[], rps_per_worker=4)


def test_cli_dry_run_does_not_build_or_run_anything(capsys, tmp_path):
    out_dir = str(tmp_path / "cli")  # deliberately not created
    # fmt: off
    code = sweep.main(
        [
            "saturation",
            "--url", "http://localhost:8000",
            "--rps", "1,2,4",
            "--duration", "20",
            "--out", out_dir,
            "--binary", "/tmp/loadgen-bin",
            "--dry-run",
        ]
    )
    # fmt: on
    printed = capsys.readouterr().out
    assert code == 0
    assert "dry-run" in printed and os.path.join(out_dir, "rps4.json") in printed
    assert not os.path.exists(out_dir)  # no build, no run, no artifacts


# --------------------------------------------------------------------------- #
# Manifest round trip
# --------------------------------------------------------------------------- #
def test_manifest_carries_parameters_and_every_point_report():
    plan = sweep.plan_saturation(
        url="http://x", rps_list=[1, 2], workers=1, out_dir="/tmp/e2", binary="/b"
    )
    entries = [
        sweep.manifest_entry(plan, p, make_report(rps=p.offered_rps, p50=100.0 * p.offered_rps))
        for p in plan.points
    ]
    manifest = sweep.build_manifest(plan, entries)
    assert manifest["kind"] == "saturation"
    assert manifest["params"]["rps_list"] == [1.0, 2.0]
    assert manifest["slo_multiplier"] == 3.0
    assert [e["report_path"] for e in manifest["points"]] == [
        "/tmp/e2/rps1.json",
        "/tmp/e2/rps2.json",
    ]
    points = analyze.points_from_manifest(manifest)
    assert [p.offered_rps for p in points] == [1.0, 2.0]
    assert sweep.summarize(manifest).startswith("=== E2 saturation curve ===")


# --------------------------------------------------------------------------- #
# Plotting — only that the files get written (no pixel assertions)
# --------------------------------------------------------------------------- #
matplotlib = pytest.importorskip("matplotlib", reason="charts need the bench extra")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_plots_write_png_svg_and_a_table_view(tmp_path, theme):
    from bench import plots

    sat = analyze.analyze_saturation(saturation_points())
    sc = analyze.analyze_scaling(scaling_points({1: 500.0, 2: 950.0, 4: 1850.0, 8: 3200.0}))
    written = plots.plot_saturation(sat, str(tmp_path), theme=theme, stem="e2")
    written += plots.plot_scaling(sc, str(tmp_path), theme=theme, stem="e5")
    assert [os.path.basename(p) for p in written] == [
        "e2.png",
        "e2.svg",
        "e2.csv",
        "e5.png",
        "e5.svg",
        "e5.csv",
    ]
    assert all(os.path.getsize(p) > 0 for p in written)


def test_render_sweep_dispatches_on_the_manifest_kind(tmp_path):
    from bench import plots

    plan = sweep.plan_scaling(
        url="http://x", worker_list=[1, 2], rps_per_worker=4, out_dir=str(tmp_path), binary="/b"
    )
    entries = [
        sweep.manifest_entry(
            plan, p, make_report(rps=p.offered_rps, p50=120.0, tokens_per_second=500.0 * p.workers)
        )
        for p in plan.points
    ]
    written = plots.render_sweep(sweep.build_manifest(plan, entries), str(tmp_path))
    assert any(p.endswith("e5-scaling.png") for p in written)


def test_unknown_theme_is_rejected():
    from bench import plots

    with pytest.raises(ValueError, match="unknown theme"):
        plots._theme("solarized")
