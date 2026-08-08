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

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TS_COLUMNS)
        for r in steps:
            t, elapsed = r["t"], r["t"] - t0
            j = bisect_right(st_ts, t) - 1
            world, ckpt = st_vals[j] if 0 <= j < len(st_vals) else (r["ws"], -1)
            usd = cost_at(t)
            w.writerow(
                [
                    int(t * 1000),
                    ckpt if ckpt >= 0 else "",
                    r["step"],
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


def write_occupancy(spans, path: Path) -> int:
    """Rows at each TRANSITION; State Timeline holds a value until it changes,
    so per-second sampling would be pure waste."""
    changes: dict[float, dict[int, str]] = {}
    for e in spans:
        changes.setdefault(e["start"], {})[e["slot"]] = e["name"]
        changes.setdefault(e["end"], {}).setdefault(e["slot"], "down")
    current = {s: "down" for s in range(TARGET_WORLD)}
    cols = [f"slot{s}" for s in range(TARGET_WORLD)]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", *cols])
        for t in sorted(changes):
            current.update(changes[t])
            w.writerow([int(t * 1000), *[current[s] for s in range(TARGET_WORLD)]])
    return len(changes)


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "multinode-preempt-1786207072"
    DATA.mkdir(parents=True, exist_ok=True)
    prof = json.loads((SRC / f"{run_id}.profile.json").read_text())
    met = json.loads((SRC / f"{run_id}.metrics.json").read_text())
    steps = load_steps(SRC / "logs")
    if not steps:
        raise SystemExit("no timestamped step lines found")
    st_ts, st_vals = load_status(SRC / "status_hist.jsonl")
    spans = node_spans(SRC / "logs", {r["step"]: r["t"] for r in steps})

    n1 = write_timeseries(steps, st_ts, st_vals, prof, met, DATA / "timeseries.csv")
    n2 = write_occupancy(spans, DATA / "occupancy.csv")

    summary = {
        "run_id": run_id,
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
    print(f"timeseries.csv {n1} rows · occupancy.csv {n2} transitions · summary.json")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
