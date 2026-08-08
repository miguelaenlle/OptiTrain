#!/usr/bin/env python3
"""Export a run to the two CSVs the Grafana dashboard reads.

Deliberately the SAME shape a live writer would append to, so pass 2 (live) is a
change of writer, not of schema or dashboard:

  timeseries.csv   one row per logged step  — every metric panel
  occupancy.csv    one row per membership TRANSITION — the State Timeline Gantt

Timestamps are absolute epoch milliseconds. Grafana is time-native: its picker,
shared crosshair and synced zoom all key off real time, and a relative
"seconds since start" column would make the whole dashboard inert.

    python3 deploy/grafana/export.py <run_id>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE.parent.parent / ".context" / "e5"
DATA = HERE / "data"

STEP_RE = re.compile(
    r"step (\d+): loss ([0-9.]+), (\d+)ms/step, (\d+) tok/s, ws (\d+), t ([0-9.]+)"
)
RANK_RE = re.compile(r"\[rank (\d+)\] step (\d+) .*?\| ws (\d+)(?:\s*\| t ([0-9.]+))?")
NODE_RE = re.compile(r"boot-node(\d+)(?:-r(\d+))?\.log")
TOKENS_PER_STEP = 480 * 1024
TARGET_WORLD = 8

TS_COLUMNS = [
    "time",
    "durable_step",
    "current_step",
    "furthest_step",
    "steps_at_risk",
    "train_loss",
    "tokens_b",
    "nodes_training",
    "nodes_lost",
    "replacements",
    "below_full_world",
    "goodput",
    "ms_per_step",
    "whole_group_restarts",
    "usd",
    "usd_per_1k",
]


def load_steps(logs: Path) -> list[dict]:
    rows = []
    for p in sorted(logs.glob("boot-node*.log")):
        for line in p.read_text(errors="replace").splitlines():
            if m := STEP_RE.search(line):
                rows.append(
                    {
                        "step": int(m.group(1)),
                        "loss": float(m.group(2)),
                        "ms": int(m.group(3)),
                        "ws": int(m.group(5)),
                        "t": float(m.group(6)),
                    }
                )
    rows.sort(key=lambda r: r["t"])
    return rows


def load_status(path: Path) -> tuple[list[float], list[tuple[int, int]]]:
    ts, vals = [], []
    if not path.exists():
        return ts, vals
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = r["doc"]
        ts.append(float(r["t"]))
        vals.append((len(d.get("members") or []), int(d.get("ckpt_step", -1))))
    return ts, vals


def node_spans(logs: Path, step_time: dict[int, float]) -> list[dict]:
    """Every box's training window, keyed (slot, generation)."""
    known = sorted(step_time)
    spans: dict[tuple[int, int], dict] = {}
    for p in sorted(logs.glob("boot-node*.log")):
        m = NODE_RE.match(p.name)
        if not m:
            continue
        slot, gen = int(m.group(1)), int(m.group(2) or 0)
        times: list[float] = []
        for line in p.read_text(errors="replace").splitlines():
            if s := STEP_RE.search(line):
                times.append(float(s.group(6)))
            elif r := RANK_RE.search(line):
                if r.group(4):
                    times.append(float(r.group(4)))
                elif known:
                    step = int(r.group(2))
                    j = bisect_right(known, step)
                    cand = [k for k in (j - 1, j) if 0 <= k < len(known)]
                    if cand:
                        best = min(cand, key=lambda k: abs(known[k] - step))
                        times.append(step_time[known[best]])
        if times:
            spans[(slot, gen)] = {
                "slot": slot,
                "gen": gen,
                "name": f"node{slot}" + (f"-r{gen}" if gen else ""),
                "start": min(times),
                "end": max(times),
            }
    # a slot holds one occupant at a time; clamp so generations cannot overlap
    out, prev_end = [], {}
    for k in sorted(spans):
        e = spans[k]
        e["start"] = max(e["start"], prev_end.get(e["slot"], e["start"]))
        e["end"] = max(e["start"], e["end"])
        prev_end[e["slot"]] = e["end"]
        out.append(e)
    return out


def write_timeseries(steps, st_ts, st_vals, prof, met, path: Path) -> int:
    inst = prof["cost"]["instances"]
    kills = sorted(e["t_wall"] for e in prof["events"] if e["event"] == "kill")
    rel = sorted(e["t_wall"] for e in prof["events"] if e["event"] == "relaunch")
    t0 = steps[0]["t"]

    def cost_at(t: float) -> float:
        return sum(
            i["hourly_usd"] * max(0.0, min(t, i["stopped_at"]) - i["started_at"]) / 3600
            for i in inst
            if t > i["started_at"]
        )

    furthest = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TS_COLUMNS)
        for r in steps:
            furthest = max(furthest, r["step"])
            t, elapsed = r["t"], r["t"] - t0
            j = bisect_right(st_ts, t) - 1
            world, ckpt = st_vals[j] if 0 <= j < len(st_vals) else (r["ws"], -1)
            usd = cost_at(t)
            w.writerow(
                [
                    int(t * 1000),
                    ckpt if ckpt >= 0 else "",
                    r["step"],
                    furthest,
                    max(0, r["step"] - ckpt) if ckpt >= 0 else "",
                    round(r["loss"], 4),
                    round(r["step"] * TOKENS_PER_STEP / 1e9, 6),
                    r["ws"] or world,
                    bisect_right(kills, t),
                    bisect_right(rel, t),
                    1 if (r["ws"] or world) < TARGET_WORLD else 0,
                    round(min(elapsed, met["trained_seconds_total"]) / elapsed, 4)
                    if elapsed > 0
                    else "",
                    r["ms"],
                    0,
                    round(usd, 4),
                    round(usd / r["step"] * 1000, 4) if r["step"] else "",
                ]
            )
    return len(steps)


def epoch_timeline(status_path: Path, log_path: Path) -> list[dict]:
    """[(t, epoch, members, master)] — the AUTHORITATIVE occupancy record.

    Epoch publications are the supervisor's own decisions: `members` is exactly
    who was in the world and `master` is the elected leader. Timestamps come
    from the status poll, which observed the epoch counter change.

    This replaces inferring occupancy from step timestamps. That inference was
    wrong twice over: it could not see world sizes that existed between
    restarts (no steps are logged during a regrow, so 4->5->6->7->8 collapsed
    to 4->8), and de-overlapping its output ERASED 8 of 14 real down periods.
    """
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    meta: dict[int, tuple[list[int], int]] = {}
    for e, mem, mas in re.findall(
        r"published epoch (\d+): members \[([^\]]*)\] master=node(\d+)", log
    ):
        ep = int(e)
        if ep not in meta:
            members = [int(x) for x in mem.replace(" ", "").split(",") if x != ""]
            meta[ep] = (members, int(mas))

    out, seen = [], None
    for line in status_path.read_text().splitlines() if status_path.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = int(r["doc"].get("epoch") or 0)
        if ep == seen or ep not in meta:
            continue
        seen = ep
        members, master = meta[ep]
        out.append({"t": float(r["t"]), "epoch": ep, "members": members, "master": master})
    return out


def kill_times(prof: dict, log_path: Path) -> dict[str, float]:
    """instance_id -> kill wall clock.

    The profile's kill events carry timestamps but not which node they hit; the
    supervisor log names the node and instance but has no clock. Both are
    strictly ordered, so zipping them recovers the pair. Dedup by instance id
    because the streamed log re-appends the same lines (208 lines, 14 kills).
    """
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    pairs = list(dict.fromkeys(re.findall(r"terminated node (\d+) \((i-[0-9a-f]+)\)", log)))
    ts = sorted(e["t_wall"] for e in prof["events"] if e["event"] == "kill")
    return {inst: t for (_, inst), t in zip(pairs, ts, strict=False)}


def write_occupancy(
    epochs: list[dict], steps_by_slot: dict, prof: dict, path: Path, target: int
) -> int:
    """State per slot at every epoch boundary: training / provisioning / down.

    A slot in `members` is training. A slot absent from members but whose
    replacement has been launched is provisioning. Otherwise it is down. The
    leader is flagged separately so it can be styled without stealing a state.
    """
    relaunch = sorted(e["t_wall"] for e in prof["events"] if e["event"] == "relaunch")
    cols = [f"slot{s}" for s in range(target)]
    rows = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", *cols])
        for i, ep in enumerate(epochs):
            t = ep["t"]
            vals = []
            for s in range(target):
                if s in ep["members"]:
                    tag = "leader" if s == ep["master"] else "training"
                else:
                    # provisioning once a replacement has been launched for it
                    launched = any(t - 1 <= r <= t + 240 for r in relaunch)
                    tag = "provisioning" if launched else "down"
                vals.append(tag)
            w.writerow([int(t * 1000), *vals])
            rows += 1
            # hold the state until the next epoch; State Timeline needs no
            # intermediate rows, but a trailing row pins the final span
            if i == len(epochs) - 1:
                w.writerow([int((t + 1) * 1000), *vals])
    return rows


def write_world(epochs: list[dict], path: Path, target: int) -> int:
    """World size on the EPOCH clock, not the step clock.

    Worlds that exist between restarts carry no training steps, so sampling at
    step times drops them: E5's regrow 1->2->3->4->5->6->7->8 rendered as a
    single 1->8 jump. Epoch publications capture every one.
    """
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "nodes_training", "below_full_world"])
        for ep in epochs:
            n = len(ep["members"])
            w.writerow([int(ep["t"] * 1000), n, 1 if n < target else 0])
    return len(epochs)


def write_degraded(epochs: list[dict], path: Path, target: int) -> int:
    """Degraded intervals as Grafana region annotations.

    Dashboard annotations are drawn on EVERY time-series panel, which is how the
    static figures shade a recovery across progress and world size at once.
    """
    regions, open_at = [], None
    for ep in epochs:
        short = len(ep["members"]) < target
        if short and open_at is None:
            open_at = ep["t"]
        elif not short and open_at is not None:
            regions.append((open_at, ep["t"]))
            open_at = None
    if open_at is not None:
        regions.append((open_at, epochs[-1]["t"]))
    path.write_text(
        json.dumps(
            [
                {
                    "time": int(a * 1000),
                    "timeEnd": int(b * 1000),
                    "title": "degraded",
                    "text": f"world below {target}",
                    "tags": ["degraded"],
                }
                for a, b in regions
            ],
            indent=1,
        )
    )
    return len(regions)


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "multinode-preempt-1786207072"
    DATA.mkdir(parents=True, exist_ok=True)
    prof = json.loads((SRC / f"{run_id}.profile.json").read_text())
    met = json.loads((SRC / f"{run_id}.metrics.json").read_text())
    steps = load_steps(SRC / "logs")
    if not steps:
        raise SystemExit("no timestamped step lines found")
    st_ts, st_vals = load_status(SRC / "status_hist.jsonl")

    epochs = epoch_timeline(SRC / "status_hist.jsonl", SRC / "run.log")
    n1 = write_timeseries(steps, st_ts, st_vals, prof, met, DATA / "timeseries.csv")
    n2 = write_occupancy(epochs, {}, prof, DATA / "occupancy.csv", TARGET_WORLD)
    n3 = write_world(epochs, DATA / "world.csv", TARGET_WORLD)
    n4 = write_degraded(epochs, DATA / "degraded.json", TARGET_WORLD)

    t_start_ms = int(steps[0]["t"] * 1000)
    t_end_ms = int(steps[-1]["t"] * 1000)
    summary = {
        "run_id": run_id,
        # The dashboard's default window is derived from these. A finished run
        # pins to its own end so there is no dead trailing space; a live one
        # runs to "now" so the window follows the fleet.
        "run_start_ms": t_start_ms,
        "run_end_ms": t_end_ms,
        "active": False,
        "goodput": round(met["trained_seconds_total"] / prof["durations"]["total_s"], 4),
        "nodes_lost": sum(1 for e in prof["events"] if e["event"] == "kill"),
        "replacements": sum(1 for e in prof["events"] if e["event"] == "relaunch"),
        "whole_group_restarts": 0,
        "steps": met["steps"],
        "val_loss": round(met["val_loss"], 4),
        "usd": round(sum(i["usd"] for i in prof["cost"]["instances"]), 2),
        "wall_hours": round(prof["durations"]["total_s"] / 3600, 3),
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        f"timeseries.csv {n1} · occupancy.csv {n2} epochs · world.csv {n3} · "
        f"degraded.json {n4} regions"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
