"""The Grafana dashboard while a run is IN FLIGHT.

Every defect these cover looked identical from the outside -- a panel reading
"No data" or a line that stopped mid-chart -- and every one of them was invisible
to a replay of a FINISHED run, because a finished run has profile.json,
metrics.json and a step log that runs to the end. Mid-run it has none of those,
so the fixtures here deliberately omit them.

The fixture is synthetic rather than a captured run: it has to stay small, and
what is being asserted is the shape of the output at a moment in time, not any
particular training curve.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

GRAFANA = Path(__file__).resolve().parents[1] / "deploy" / "grafana"

LAUNCH = 1_700_000_000  # run id suffix; ten digits, like a real one
RUN_ID = f"multinode-preempt-{LAUNCH}"
FIRST_STEP = LAUNCH + 200
NOW = LAUNCH + 400
NODES = 2


@contextmanager
def _load(module: str, argv: list[str], data_dir: Path):
    """Import export.py / build_dashboard.py with argv and DATA under our control.

    Both are scripts: they read sys.argv at import time AND inside main(), and
    they resolve their output directory from a module constant. A fresh module
    object with argv held in place for the whole call is the only way to drive
    them in-process.
    """
    old = sys.argv
    sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location(f"_{module}", GRAFANA / f"{module}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.DATA = data_dir
        yield mod
    finally:
        sys.argv = old


def _status(t: float, epoch: int, members: list[int], ckpt: int, nodes: list[dict]) -> str:
    doc = {
        "version": 1,
        "run_id": RUN_ID,
        "updated_at": t,
        "epoch": epoch,
        "members": members,
        "ckpt_step": ckpt,
        "done": False,
        "nodes": nodes,
    }
    return json.dumps({"t": t, "doc": doc})


def _node(idx: int, iid: str, aws_state: str) -> dict:
    return {"node": idx, "attempt": 0, "instance_id": iid, "aws_state": aws_state}


@pytest.fixture
def midrun(tmp_path: Path) -> Path:
    """A run 400s in: booted, trained, lost node 0, replacement still booting.

    No profile.json and no metrics.json -- both are written when the run ENDS,
    and assuming them is what made the fleet and cost panels read zero through
    the middle of a live run.
    """
    src = tmp_path / "live"
    (src / "logs").mkdir(parents=True)

    up = [_node(0, "i-aaa", "running"), _node(1, "i-bbb", "running")]
    replaced = [
        _node(0, "i-aaa", "shutting-down"),
        _node(1, "i-bbb", "running"),
        _node(0, "i-ccc", "pending"),
    ]
    lines = [
        # boot: boxes billing, nothing training yet
        _status(LAUNCH + 20, 0, [], -1, up),
        _status(LAUNCH + 60, 0, [], -1, up),
        # full world
        _status(LAUNCH + 190, 1, [0, 1], -1, up),
        _status(LAUNCH + 250, 1, [0, 1], 4, up),
        # node 0 killed; its replacement is already pending
        _status(LAUNCH + 330, 2, [1], 8, replaced),
        _status(LAUNCH + 390, 2, [1], 8, replaced),
    ]
    (src / "status_hist.jsonl").write_text("\n".join(lines) + "\n")
    (src / "run.log").write_text(
        "[supervisor] published epoch 1: members [0, 1] master=node0 (10.0.0.1:29401)\n"
        "[supervisor] terminated node 0 (i-aaa)\n"
        "[supervisor] published epoch 2: members [1] master=node1 (10.0.0.2:29402)\n"
    )
    # Steps stop at +300: after that the world is regrowing and nothing is
    # logged. That gap is the whole point of the trailing row.
    steps = [
        f"step {i}: loss {10.0 - i * 0.1:.4f}, 5000ms/step, 90000 tok/s, ws 2, "
        f"t {FIRST_STEP + i * 10}.0"
        for i in range(1, 11)
    ]
    (src / "logs" / "boot-node0.log").write_text("\n".join(steps) + "\n")
    (src / "logs" / "boot-node1.log").write_text("")
    return src


def _export(src: Path, out: Path, **kw) -> dict:
    argv = [
        "export.py",
        RUN_ID,
        "--live",
        f"--nodes={NODES}",
        f"--src={src}",
        f"--now={kw.get('now', NOW)}",
    ]
    with _load("export", argv, out) as mod:
        mod.main()
    d = out / RUN_ID
    return {
        "dir": d,
        "timeseries": _rows(d / "timeseries.csv"),
        "world": _rows(d / "world.csv"),
        "occupancy": _rows(d / "occupancy.csv"),
        "val": _rows(d / "val.csv"),
        "meta": _rows(d / "meta.csv"),
        "summary": json.loads((d / "summary.json").read_text()),
    }


def _rows(p: Path) -> list[dict]:
    import csv

    with p.open() as fh:
        return list(csv.DictReader(fh))


def test_series_reach_now_not_the_last_step(midrun: Path, tmp_path: Path):
    """Steps stop during a regrow; the wall-clock series must not stop with them."""
    r = _export(midrun, tmp_path / "data")
    last = r["timeseries"][-1]
    assert int(last["time"]) == NOW * 1000
    assert int(r["world"][-1]["time"]) == NOW * 1000
    # ...and the values on that row are the ones the fleet panels read.
    assert float(last["usd"]) > 0
    assert int(last["nodes_training"]) == 1
    assert int(last["nodes_lost"]) == 1
    assert int(last["furthest_step"]) == 10
    assert int(last["steps_at_risk"]) == 2  # last step 10 - durable 8


def test_trailing_row_invents_nothing(midrun: Path, tmp_path: Path):
    """Only what the status poll still asserts. Loss and step time measure a
    step, and no step is happening -- carrying them forward would be fiction."""
    last = _export(midrun, tmp_path / "data")["timeseries"][-1]
    assert last["train_loss"] == ""
    assert last["ms_per_step"] == ""
    assert last["current_step"] == ""


def test_cost_accrues_from_launch_not_from_the_first_step(midrun: Path, tmp_path: Path):
    """Boot is billed. The old estimate started the clock at the first step, so
    the cost tile read $0.00 through three minutes of paid-for boot."""
    ts = _export(midrun, tmp_path / "data")["timeseries"]
    boot = [r for r in ts if int(r["time"]) < FIRST_STEP * 1000]
    assert boot, "no rows before the first step: the boot window is invisible"
    assert float(boot[-1]["usd"]) > 0
    assert [float(r["usd"]) for r in ts] == sorted(float(r["usd"]) for r in ts)


def test_window_opens_on_the_launch(midrun: Path, tmp_path: Path):
    """The run id carries the launch timestamp, so the default window is
    knowable before any artifact exists."""
    s = _export(midrun, tmp_path / "data")["summary"]
    assert s["run_start_ms"] == LAUNCH * 1000
    assert s["active"] is True


def test_world_size_starts_at_launch(midrun: Path, tmp_path: Path):
    """Nothing is training during boot -- but "no row" and "zero" render very
    differently, and the panel used to have no rows at all until the first
    epoch was published."""
    world = _export(midrun, tmp_path / "data")["world"]
    assert int(world[0]["time"]) == LAUNCH * 1000
    assert int(world[0]["nodes_training"]) == 0
    assert [int(r["nodes_training"]) for r in world] == [0, 2, 1, 1]


def test_occupancy_covers_boot_and_replacement(midrun: Path, tmp_path: Path):
    """The Gantt is the panel you watch to see boxes come and go; it was blank
    for the boot, and a slot whose replacement was booting read "down"."""
    occ = _export(midrun, tmp_path / "data")["occupancy"]
    assert set(occ[0].keys()) == {"time", "slot0", "slot1"}
    assert occ[0]["slot0"] == "provisioning" and occ[0]["slot1"] == "provisioning"
    assert [r["slot0"] for r in occ] == ["provisioning", "leader", "provisioning"]
    assert [r["slot1"] for r in occ] == ["provisioning", "training", "leader"]


def test_a_run_with_no_steps_yet_still_exports(tmp_path: Path):
    """The first ~3 minutes of every run. Returning early here left the run with
    no summary.json, so the dashboard fell back to a rolling 6h window and the
    run itself had nothing to select."""
    src = tmp_path / "live"
    (src / "logs").mkdir(parents=True)
    (src / "status_hist.jsonl").write_text(
        _status(LAUNCH + 20, 0, [], -1, [_node(0, "i-aaa", "pending")]) + "\n"
    )
    r = _export(src, tmp_path / "data", now=LAUNCH + 30)
    assert r["summary"]["run_start_ms"] == LAUNCH * 1000
    assert int(r["world"][-1]["time"]) == (LAUNCH + 30) * 1000
    index = (tmp_path / "data" / "runs.csv").read_text()
    assert RUN_ID in index
    # Every file the dashboard fetches must EXIST even when it is empty: nginx
    # 404s an absent CSV and the panel shows a query error, which reads as
    # broken rather than as "this run has not started yet".
    for name in ("timeseries.csv", "world.csv", "occupancy.csv", "summary.csv"):
        assert (r["dir"] / name).exists(), name


def test_a_finished_run_stops_following_the_clock(midrun: Path, tmp_path: Path):
    """The LAST tick of every run is a --live tick, so `--live` alone cannot mean
    "active". A run that has reported done must pin its window to its own end,
    or a finished run grows an ever-widening empty margin on the right."""
    lines = (midrun / "status_hist.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])
    last["doc"]["done"] = True
    (midrun / "status_hist.jsonl").write_text("\n".join([*lines[:-1], json.dumps(last)]) + "\n")

    # No --now=: the exporter must decide the right edge for itself.
    argv = ["export.py", RUN_ID, "--live", f"--nodes={NODES}", f"--src={midrun}"]
    with _load("export", argv, tmp_path / "data") as mod:
        mod.main()
    s = json.loads((tmp_path / "data" / RUN_ID / "summary.json").read_text())
    assert s["active"] is False
    # Pinned to the last training step, not to wall-clock now.
    assert s["run_end_ms"] == int((FIRST_STEP + 100) * 1000)


def test_val_loss_is_placed_between_the_logged_steps(midrun: Path, tmp_path: Path):
    """The eval line carries no wall clock, and its step is usually not a logged
    one: eval every 25 with LOG_INTERVAL_STEPS=10 never coincides. The point has
    to be interpolated onto the step->time map or it lands at the wrong instant."""
    log = midrun / "logs" / "boot-node0.log"
    # step 5 IS logged; step 7 is between logged steps 7 and 8 -> interpolated.
    log.write_text(
        log.read_text() + "eval step 5: val_loss 8.4000\n" + "eval step 7: val_loss 8.1000\n"
    )
    got = _export(midrun, tmp_path / "data")["val"]
    assert [r["step"] for r in got] == ["5", "7"]
    assert [r["val_loss"] for r in got] == ["8.4", "8.1"]
    # step N is logged at FIRST_STEP + N*10
    assert int(got[0]["time"]) == int((FIRST_STEP + 50) * 1000)
    assert int(got[1]["time"]) == int((FIRST_STEP + 70) * 1000)


def test_val_loss_keeps_a_re_eval_after_rollback_but_drops_replayed_lines(
    midrun: Path, tmp_path: Path
):
    """A replacement box replays its predecessor's log, so an identical eval line
    appears twice and must collapse. A step re-trained after a rollback is
    genuinely evaluated twice with a DIFFERENT loss -- that one must survive, or
    the chart hides the rollback."""
    log = midrun / "logs" / "boot-node0.log"
    log.write_text(log.read_text() + "eval step 5: val_loss 8.4000\n")
    # same line replayed verbatim on the replacement, plus a real re-eval
    (midrun / "logs" / "boot-node0-r1.log").write_text(
        "eval step 5: val_loss 8.4000\neval step 5: val_loss 8.2500\n"
    )
    got = _export(midrun, tmp_path / "data")["val"]
    assert [(r["step"], r["val_loss"]) for r in got] == [("5", "8.25"), ("5", "8.4")]


def test_model_facts_come_from_the_log_and_are_not_multiplied_by_node_count(
    midrun: Path, tmp_path: Path
):
    """Every rank prints the parameter census, so summing the matches scaled one
    model's size by the world size -- 8 nodes turned 124M into 996M."""
    for n in (0, 1):
        p = midrun / "logs" / f"boot-node{n}.log"
        p.write_text(
            "+ export DATASET=openwebtext\n+ export N_LAYER=12\n+ export N_HEAD=12\n"
            "+ export N_EMBD=768\n+ export BLOCK_SIZE=1024\n+ export GLOBAL_BATCH_SIZE=480\n"
            "number of parameters: 123.69M\n"
            "num decayed parameter tensors: 50, with 124,354,560 parameters\n"
            "num non-decayed parameter tensors: 98, with 121,344 parameters\n" + p.read_text()
        )
    meta = _export(midrun, tmp_path / "data")["meta"]
    assert len(meta) == 2, "one row comes back numeric-long and renders as No data"
    assert meta[0]["model"] == "GPT-2 · 12L/12H/768d"
    assert meta[0]["dataset"] == "openwebtext"
    assert meta[0]["params_exact"] == "124475904"  # NOT x2 for the two nodes
    assert meta[0]["ctx"] == "1024" and meta[0]["global_batch"] == "480"


def test_model_tiles_select_the_string_field_explicitly(tmp_path: Path):
    """A stat panel's `fields: ""` means NUMERIC fields only, so a text tile
    renders "No data" while the backend returns the value."""
    with _load("build_dashboard", ["build_dashboard.py"], tmp_path / "data") as mod:
        panels = {p["title"]: p for p in mod.build()["panels"] if p["type"] == "stat"}
    assert panels["Model"]["options"]["reduceOptions"]["fields"] == "Model"
    assert panels["Dataset"]["options"]["reduceOptions"]["fields"] == "Dataset"
    # numeric tiles must keep the default, or they stop reducing
    assert panels["Parameters"]["options"]["reduceOptions"]["fields"] == ""


def test_loss_panel_queries_both_files_without_refid_prefixes(tmp_path: Path):
    """Two queries on one panel make Grafana label series "A train loss"."""
    with _load("build_dashboard", ["build_dashboard.py"], tmp_path / "data") as mod:
        panel = next(p for p in mod.build()["panels"] if p["title"].startswith("Loss"))
    assert [t["url"].split("/")[-1] for t in panel["targets"]] == ["timeseries.csv", "val.csv"]
    assert [t["refId"] for t in panel["targets"]] == ["A", "B"]
    named = {
        o["matcher"]["options"]
        for o in panel["fieldConfig"]["overrides"]
        for pr in o["properties"]
        if pr["id"] == "displayName"
    }
    assert named == {"train loss", "val loss"}


def test_gantt_target_carries_url_options(tmp_path: Path):
    """Without url_options the BACKEND query is fine and the browser's is not:
    /api/ds/query returns correct rows while the panel reads "No data" with an
    error corner. That is why the Gantt never rendered once."""
    data = tmp_path / "data"
    (data / RUN_ID).mkdir(parents=True)
    with _load("build_dashboard", ["build_dashboard.py", f"--run={RUN_ID}"], data) as mod:
        dash = mod.build()
    gantt = next(p for p in dash["panels"] if p["type"] == "state-timeline")
    assert gantt["targets"][0]["url_options"] == {"method": "GET"}
    # Requesting more slots than the CSV holds is fine -- Infinity drops the
    # absent ones -- and it is what keeps the panel node-count agnostic.
    slots = [c["selector"] for c in gantt["targets"][0]["columns"] if c["selector"] != "time"]
    assert len(slots) >= 8


def test_empty_run_is_selectable_and_sets_the_window(tmp_path: Path):
    """A launched run with nothing exported yet must still appear, and must set
    the window -- otherwise a fresh dashboard opens on a rolling tail."""
    data = tmp_path / "data"
    (data / RUN_ID).mkdir(parents=True)
    with _load("build_dashboard", ["build_dashboard.py", f"--run={RUN_ID}"], data) as mod:
        dash = mod.build()
    var = dash["templating"]["list"][0]
    assert var["current"]["value"] == RUN_ID
    assert RUN_ID in var["query"]
    assert dash["time"]["to"] == "now"
    assert dash["time"]["from"].startswith("2023-11-14")  # LAUNCH, minus the pad
