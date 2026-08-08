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
# --live points SRC at the polling working dir instead of a finished run's
# artifacts, so the same transforms serve both a replay and a run in flight.
SRC = HERE / ".live" if "--live" in sys.argv else HERE.parent.parent / ".context" / "e5"
DATA = HERE / "data"


def run_dir(run_id: str) -> Path:
    """Per-run output directory.

    A single shared data/ directory means the last export wins, so opening the
    dashboard during a new run shows the PREVIOUS run's numbers -- which is
    worse than showing nothing, because it looks like live data. Namespacing by
    run_id fixes that and gives run switching at the same time.
    """
    return DATA / run_id


STEP_RE = re.compile(
    r"step (\d+): loss ([0-9.]+), (\d+)ms/step, (\d+) tok/s, ws (\d+), t ([0-9.]+)"
)
RANK_RE = re.compile(r"\[rank (\d+)\] step (\d+) .*?\| ws (\d+)(?:\s*\| t ([0-9.]+))?")
NODE_RE = re.compile(r"boot-node(\d+)(?:-r(\d+))?\.log")
TOKENS_PER_STEP = 480 * 1024
TARGET_WORLD = next((int(a.split("=")[-1]) for a in sys.argv if a.startswith("--nodes")), 8)

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
    "degraded_band",
    "goodput",
    "ms_per_step",
    "ms_per_step_typical",
    "whole_group_restarts",
    "usd",
    "usd_per_1k",
]


def _load_json(*paths: Path):
    for p in paths:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
    return None


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


def write_timeseries(steps, st_ts, st_vals, prof, met, path: Path, log_counts=None) -> int:
    inst = prof["cost"]["instances"]
    kills = sorted(e["t_wall"] for e in prof["events"] if e["event"] == "kill")
    rel = sorted(e["t_wall"] for e in prof["events"] if e["event"] == "relaunch")
    # profile.json does not exist until the run ENDS, so mid-run every one of
    # these is empty and the fleet/cost panels read zero while the run is
    # visibly losing nodes. Fall back to the supervisor log, which is live.
    if not kills and log_counts:
        kills = log_counts.get("kills", [])
    if not rel and log_counts:
        rel = log_counts.get("relaunches", [])
    t0 = steps[0]["t"]

    def cost_at(t: float) -> float:
        if inst:
            return sum(
                i["hourly_usd"] * max(0.0, min(t, i["stopped_at"]) - i["started_at"]) / 3600
                for i in inst
                if t > i["started_at"]
            )
        # Live estimate: no instance ledger until the run ends. N nodes billed
        # from the first step is an underestimate (it ignores boot), but a cost
        # panel reading 0.00 through a run that is spending money is worse.
        rate = float(__import__("os").environ.get("HOURLY_USD") or 1.006)
        return TARGET_WORLD * rate * max(0.0, t - steps[0]["t"]) / 3600

    # Rolling MEDIAN, not mean: checkpoint steps run ~10x the steady step, and a
    # mean would drag the line toward them -- which is the spike problem, not a
    # fix for it. A median of 15 rejects them outright while still tracking the
    # real shift when the world shrinks. The raw series is kept alongside it, so
    # the spikes remain visible rather than being hidden.
    WIN = 31
    ms_all = [r["ms"] for r in steps]

    def typical(i: int) -> int:
        lo, hi = max(0, i - WIN // 2), min(len(ms_all), i + WIN // 2 + 1)
        w = sorted(ms_all[lo:hi])
        return w[len(w) // 2]

    furthest = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TS_COLUMNS)
        for i, r in enumerate(steps):
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
                    # Shades the degraded x-ranges INSIDE the progress panel by
                    # riding its own y-axis; a second axis would be worse than
                    # no shading at all.
                    furthest if (r["ws"] or world) < TARGET_WORLD else "",
                    round(min(elapsed, met["trained_seconds_total"]) / elapsed, 4)
                    if elapsed > 0
                    else "",
                    r["ms"],
                    typical(i),
                    0,
                    round(usd, 4),
                    round(usd / r["step"] * 1000, 4) if r["step"] else "",
                ]
            )
    return len(steps)


def log_events(epochs: list[dict], target: int) -> dict:
    """Kill/relaunch timestamps inferred from the EPOCH timeline.

    Used only when profile.json is absent (i.e. the run is still going). A slot
    leaving `members` is a loss; the epoch at which it happens is the timestamp.
    Coarser than the profile's own marks -- epoch-poll resolution rather than the
    exact API call -- but it makes the fleet counters move during a run instead
    of sitting at zero.
    """
    kills, rel = [], []
    prev: set[int] = set()
    for i, ep in enumerate(epochs):
        cur = set(ep["members"])
        if i:
            for _ in prev - cur:
                kills.append(ep["t"])
            for _ in cur - prev:
                rel.append(ep["t"])
        prev = cur
    return {"kills": sorted(kills), "relaunches": sorted(rel)}


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
    """Per-slot state at every transition: provisioning / training / leader / down.

    Provisioning is derived from the INSTANCE LEDGER rather than from relaunch
    marks. When a slot drops out of `members` at t_out and rejoins at t_in, its
    replacement is the instance whose `started_at` falls in that gap; the slot is
    therefore DOWN from t_out until that box starts, and PROVISIONING from then
    until it rejoins the world.

    The previous version asked "did any relaunch happen near this epoch?", which
    is slot-blind -- it fired only when an epoch boundary happened to coincide
    with some relaunch, so provisioning rendered as a sliver instead of the
    ~150-250s boot it actually is.
    """
    inst = sorted(prof["cost"]["instances"], key=lambda i: i["started_at"])
    starts = [i["started_at"] for i in inst]
    used: set[int] = set()
    # Originals all start before the first epoch; only replacements matter here.
    run_start = epochs[0]["t"] - 60 if epochs else 0.0

    # base state at each epoch boundary
    base: list[tuple[float, list[str]]] = []
    for ep in epochs:
        vals = []
        for sl in range(target):
            if sl in ep["members"]:
                vals.append("leader" if sl == ep["master"] else "training")
            else:
                vals.append("down")
        base.append((ep["t"], vals))

    # Per-slot provisioning WINDOWS, not point marks. The replacement's start
    # precedes the shrink epoch, so a point mark applied at that instant lands
    # while the slot still reads "training" and gets discarded. An interval is
    # applied whenever the base state says "down", which is the correct overlay.
    prov: dict[int, list[tuple[float, float]]] = {}
    for sl in range(target):
        i = 0
        while i < len(base):
            if base[i][1][sl] != "down":
                i += 1
                continue
            t_out = base[i][0]
            j = i
            while j < len(base) and base[j][1][sl] == "down":
                j += 1
            t_in = base[j][0] if j < len(base) else float("inf")
            cand = [
                (k, st_)
                for k, st_ in enumerate(starts)
                if t_out - 180 <= st_ < t_in and k not in used and st_ > run_start
            ]
            if cand:
                k, st_ = cand[0]
                used.add(k)
                prov.setdefault(sl, []).append((max(st_, t_out), t_in))
            i = j

    # transition times: epoch boundaries plus each provisioning start
    times = sorted({t for t, _ in base} | {w[0] for ws in prov.values() for w in ws})
    cols = [f"slot{s}" for s in range(target)]
    cur = ["down"] * target
    bi = 0
    rows = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", *cols])
        for t in times:
            while bi < len(base) and base[bi][0] <= t:
                cur = list(base[bi][1])
                bi += 1
            out = list(cur)
            for sl, windows in prov.items():
                if out[sl] == "down" and any(a_ <= t < b_ for a_, b_ in windows):
                    out[sl] = "provisioning"
            w.writerow([int(t * 1000), *out])
            rows += 1
    return rows


def write_world(epochs: list[dict], path: Path, target: int, end_t: float = 0.0) -> int:
    """World size on the EPOCH clock, not the step clock.

    Worlds that exist between restarts carry no training steps, so sampling at
    step times drops them: E5's regrow 1->2->3->4->5->6->7->8 rendered as a
    single 1->8 jump. Epoch publications capture every one.
    """
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "nodes_training", "below_full_world"])
        n = target
        for ep in epochs:
            n = len(ep["members"])
            w.writerow([int(ep["t"] * 1000), n, 1 if n < target else 0])
        # Hold the final world to the end of the run. Epochs stop being published
        # once membership settles, so without this the line simply ends at the
        # last epoch -- E5's stopped at 10:18 while the run went to 10:35.
        if end_t and epochs and end_t > epochs[-1]["t"]:
            w.writerow([int(end_t * 1000), n, 1 if n < target else 0])
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
    OUT_DIR = run_dir(run_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Mid-run neither profile.json nor metrics.json exists yet; both are written
    # at the end. Everything the live panels need comes from status + logs, so
    # missing files degrade the cost/summary tiles rather than failing the tick.
    prof = _load_json(SRC / f"{run_id}.profile.json", SRC / "profile.json") or {
        "cost": {"instances": []},
        "events": [],
        "durations": {},
    }
    met = _load_json(SRC / f"{run_id}.metrics.json", SRC / "metrics.json") or {}
    met.setdefault("trained_seconds_total", 0.0)
    met.setdefault("steps", 0)
    met.setdefault("val_loss", 0.0)

    # The run_id argument names the OUTPUT directory, while SRC is chosen
    # independently (--live vs the replay default). A mismatch therefore files
    # one run's artifacts under another run's name silently -- it happened, and
    # an 8-node trace landed in a 2-node run's directory looking plausible.
    src_id = prof.get("run_id")
    if not src_id and "--live" not in sys.argv:
        raise SystemExit(
            f"refusing to export: no profile for {run_id!r} in {SRC}. A replay "
            f"needs the run's own artifacts; without this check the exporter "
            f"happily reads whatever run SRC holds and files it under the id you "
            f"typed (an 8-node trace once landed in a 2-node run's directory)."
        )
    if src_id and src_id != run_id:
        raise SystemExit(
            f"refusing to export: artifacts in {SRC} belong to {src_id!r}, "
            f"not {run_id!r}. Use --live for a run in flight, or pass the id "
            f"whose artifacts SRC actually holds."
        )
    steps = load_steps(SRC / "logs")
    if not steps:
        # Normal for the first ~3 minutes of a run: boxes are still booting.
        print("waiting: no timestamped step lines yet")
        return 0
    st_ts, st_vals = load_status(SRC / "status_hist.jsonl")

    epochs = epoch_timeline(SRC / "status_hist.jsonl", SRC / "run.log")
    n1 = write_timeseries(
        steps,
        st_ts,
        st_vals,
        prof,
        met,
        OUT_DIR / "timeseries.csv",
        log_counts=log_events(epochs, TARGET_WORLD),
    )
    n2 = write_occupancy(epochs, {}, prof, OUT_DIR / "occupancy.csv", TARGET_WORLD)
    n3 = write_world(epochs, OUT_DIR / "world.csv", TARGET_WORLD, end_t=steps[-1]["t"])
    n4 = write_degraded(epochs, OUT_DIR / "degraded.json", TARGET_WORLD)

    t_start_ms = int(steps[0]["t"] * 1000)
    t_end_ms = int(steps[-1]["t"] * 1000)
    summary = {
        "run_id": run_id,
        # The dashboard's default window is derived from these. A finished run
        # pins to its own end so there is no dead trailing space; a live one
        # runs to "now" so the window follows the fleet.
        "run_start_ms": t_start_ms,
        "run_end_ms": t_end_ms,
        "active": "--live" in sys.argv,
        "goodput": round(met["trained_seconds_total"] / (prof["durations"].get("total_s") or 1), 4),
        "nodes_lost": sum(1 for e in prof["events"] if e["event"] == "kill"),
        "replacements": sum(1 for e in prof["events"] if e["event"] == "relaunch"),
        "whole_group_restarts": 0,
        "steps": met["steps"],
        "val_loss": round(met["val_loss"], 4),
        "usd": round(sum(i.get("usd") or 0 for i in prof["cost"]["instances"]), 2),
        "wall_hours": round((prof["durations"].get("total_s") or 0) / 3600, 3),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    # Stat tiles read this, not summary.json. Infinity's JSON parser returned no
    # usable frame for scalar selectors while its CSV parser works everywhere
    # else on this dashboard -- one row keeps every panel on the proven path.
    with (OUT_DIR / "summary.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        # A `time` column matters: without it Infinity tags the frame
        # "numeric-long", which Grafana tries to reinterpret as label/value pairs
        # and the stat panel renders "No data" even though the backend returned
        # the right number. With time present it is an ordinary series and
        # lastNotNull works, matching every other panel on the dashboard.
        keys = [k for k, v in summary.items() if isinstance(v, int | float)]
        wr.writerow(["time", *keys])
        wr.writerow([t_end_ms, *[summary[k] for k in keys]])
    index = sorted(d.name for d in DATA.iterdir() if d.is_dir())
    (DATA / "runs.csv").write_text("run\n" + "\n".join(index) + "\n")

    print(
        f"[{run_id}] timeseries.csv {n1} · occupancy.csv {n2} epochs · world.csv {n3} · "
        f"degraded.json {n4} regions"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
