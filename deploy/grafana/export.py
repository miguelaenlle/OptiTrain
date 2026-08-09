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
import os
import re
import sys
import time
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).parent


def _flag(name: str, default: str | None = None) -> str | None:
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


LIVE = "--live" in sys.argv
# --live points SRC at the polling working dir instead of a finished run's
# artifacts, so the same transforms serve both a replay and a run in flight.
# --src= names it explicitly, which is what makes a mid-run state replayable in
# a test: the live loop's working dir is per-run, and the old code had a
# SRC_OVERRIDE env var that nothing ever read.
_SRC_FLAG = _flag("src")
SRC = (
    Path(_SRC_FLAG)
    if _SRC_FLAG
    else (HERE / ".live" if LIVE else HERE.parent.parent / ".context" / "e5")
)
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
EVAL_RE = re.compile(r"eval step (\d+): val_loss ([0-9.]+)")
NODE_RE = re.compile(r"boot-node(\d+)(?:-r(\d+))?\.log")
TOKENS_PER_STEP = 480 * 1024
TARGET_WORLD = int(_flag("nodes") or 8)
# aws_state values AWS bills for. "shutting-down" and "terminated" are free, so
# a node stops accruing the moment the supervisor kills it.
BILLABLE_STATES = {"pending", "running"}

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


def load_evals(logs: Path, steps: list[dict]) -> list[dict]:
    """Periodic validation loss: [{t, step, loss}].

    `eval step S: val_loss X` is printed by rank 0 to stderr, so it rides the
    node log to S3 exactly like the step lines and is available LIVE -- unlike
    metrics.json's single end-of-run figure, which is also a DIFFERENT estimator
    (`estimate_loss` over eval_iters batches, vs the deterministic full pass over
    val.bin here). The two are not points on one curve and are never mixed.

    The line carries no wall clock, so it is placed on the step->time map built
    from the step lines. Eval and log intervals are independent (eval every 25
    with LOG_INTERVAL_STEPS=10 is a real configuration), so the eval step is
    usually BETWEEN two logged steps and its time is interpolated.

    A step re-trained after a rollback is evaluated again, giving two points at
    the same step with different times. Both are kept: on a wall-clock axis they
    are two genuine observations, and collapsing them would hide the rollback.
    """
    known = sorted({r["step"]: r["t"] for r in steps}.items())
    if not known:
        return []
    xs = [s for s, _ in known]
    ys = [t for _, t in known]

    def at(step: int) -> float:
        j = bisect_right(xs, step)
        if j == 0:
            return ys[0]
        if j >= len(xs):
            return ys[-1]
        # linear between the bracketing logged steps
        x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (step - x0) / (x1 - x0)

    out = []
    for p in sorted(logs.glob("boot-node*.log")):
        for line in p.read_text(errors="replace").splitlines():
            if m := EVAL_RE.search(line):
                step = int(m.group(1))
                out.append({"t": at(step), "step": step, "loss": float(m.group(2))})
    # One rank prints, but a replacement box replays its predecessor's log, so
    # the same (step, loss) can appear twice. Dedup on the pair, not on step:
    # a genuine re-eval after a rollback has a different loss and must survive.
    seen, uniq = set(), []
    for e in sorted(out, key=lambda e: (e["step"], e["loss"])):
        key = (e["step"], e["loss"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return sorted(uniq, key=lambda e: e["t"])


def write_val(evals: list[dict], path: Path) -> int:
    """Its own file, not a column of timeseries.csv.

    Val loss is sampled on a different clock from the step rows (every
    eval_interval_steps, which need not be a multiple of the log interval), so as
    a column it would be empty on almost every row and would need a row invented
    at each eval time. A separate series is what world.csv already does for the
    same reason.
    """
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "step", "val_loss"])
        for e in evals:
            w.writerow([int(e["t"] * 1000), e["step"], round(e["loss"], 4)])
    return len(evals)


def load_run_meta(logs: Path) -> dict:
    """Model, size and dataset as the RUN ITSELF reported them.

    Every field here is parsed from the box's own log: the trainer's parameter
    count is printed by nanoGPT's model.py, and the shape/dataset come from the
    env the boot script echoed under `set -x`. Nothing is hardcoded -- the
    dashboard description used to be a typed-in string that still claimed
    "8 x g5.xlarge" while a 2-node run was on screen.

    Blank until the box gets there: DATASET/N_LAYER land seconds into boot, the
    parameter count only when the trainer builds the model (~3 min). An empty
    tile is honest; a guessed one is not.
    """
    meta: dict[str, str] = {}
    text = ""
    for p in sorted(logs.glob("boot-node*.log")):
        text += p.read_text(errors="replace")
    for key in ("DATASET", "N_LAYER", "N_HEAD", "N_EMBD", "BLOCK_SIZE", "GLOBAL_BATCH_SIZE"):
        if m := re.search(rf"\b{key}=(\S+)", text):
            meta[key] = m.group(1)
    if m := re.search(r"number of parameters:\s*([0-9.]+)M", text):
        meta["params_m"] = m.group(1)
    # nanoGPT's headline count excludes position embeddings; the optimizer's
    # tensor census is the exact total. Keep both rather than reconciling them.
    #
    # ONE occurrence of each, not a sum over the file: every rank on every node
    # prints this census, so summing the matches multiplied one model's parameter
    # count by the world size -- 8 nodes turned 124M into 996M.
    dec = re.search(r"num decayed parameter tensors: \d+, with ([\d,]+) parameters", text)
    non = re.search(r"num non-decayed parameter tensors: \d+, with ([\d,]+) parameters", text)
    if dec and non:
        meta["params_exact"] = str(
            int(dec.group(1).replace(",", "")) + int(non.group(1).replace(",", ""))
        )
    n_layer, n_head, n_embd = meta.get("N_LAYER"), meta.get("N_HEAD"), meta.get("N_EMBD")
    if n_layer and n_head and n_embd:
        # Architecture as measured, not a marketing name: this repo trains
        # nanoGPT's GPT (GPT-2 architecture) and the dims are what identify it.
        meta["model"] = f"GPT-2 · {n_layer}L/{n_head}H/{n_embd}d"
    return meta


def write_meta(meta: dict, path: Path, t_start: float, t_end: float) -> int:
    """Two rows, deliberately.

    A ONE-row CSV comes back from Infinity tagged "numeric-long", which Grafana
    reinterprets as label/value pairs and renders as "No data" while the backend
    returns the right values -- the trap that made summary.csv unusable. Two rows
    spanning the run is an ordinary series, so lastNotNull works like everywhere
    else on this dashboard.
    """
    cols = ["model", "params_m", "params_exact", "DATASET", "BLOCK_SIZE", "GLOBAL_BATCH_SIZE"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "model", "params_m", "params_exact", "dataset", "ctx", "global_batch"])
        for t in (t_start, t_end):
            w.writerow([int(t * 1000), *[meta.get(c, "") for c in cols]])
    return len(meta)


def load_hist(path: Path) -> list[tuple[float, dict]]:
    """Every polled status document, in order: [(poll_time, doc)].

    This is the only source that exists from the moment the fleet is LAUNCHED --
    step logs start ~3 minutes later and profile.json only at the end -- so
    world size, checkpoint step, membership and the cost ledger are all derived
    from it while a run is in flight.
    """
    out: list[tuple[float, dict]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append((float(r["t"]), r["doc"]))
    return out


def status_series(hist: list[tuple[float, dict]]) -> tuple[list[float], list[tuple[int, int]]]:
    ts = [t for t, _ in hist]
    vals = [(len(d.get("members") or []), int(d.get("ckpt_step", -1))) for _, d in hist]
    return ts, vals


def hourly_rate() -> float:
    """$/hr per box. HOURLY_USD wins; otherwise look the instance type up in the
    orchestrator's own table so the dashboard and the cost ledger cannot drift."""
    env = os.environ.get("HOURLY_USD")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    itype = os.environ.get("INSTANCE_TYPE", "")
    try:
        sys.path.insert(0, str(HERE.parent.parent / "src"))
        from orchestrator.config import ON_DEMAND_HOURLY_USD  # type: ignore

        if itype in ON_DEMAND_HOURLY_USD:
            return float(ON_DEMAND_HOURLY_USD[itype])
    except Exception:
        pass
    return 1.006  # g5.xlarge on-demand, us-east-1


def observed_instances(hist: list[tuple[float, dict]], now: float) -> list[dict]:
    """Billing spans reconstructed from the status polls: [{start, end}].

    profile.json carries the real ledger but does not exist until the run ENDS,
    and the fallback it replaced (TARGET_WORLD x elapsed-since-the-first-step)
    was wrong in both directions: it billed a fixed node count regardless of who
    was actually alive, and it charged BOOT as free -- so the cost tile sat at
    $0.00 for the first three minutes of a run that was already spending money.

    Every status doc names each instance and its aws_state, from launch onward.
    An instance accrues from the first poll that sees it until the last poll
    that still finds it billable; one that is billable at the newest poll keeps
    accruing to `now`.
    """
    spans: dict[str, dict] = {}
    for t, doc in hist:
        for n in doc.get("nodes") or []:
            iid = n.get("instance_id")
            if not iid:
                continue
            s = spans.setdefault(iid, {"start": t, "end": t, "billable": False})
            s["billable"] = (n.get("aws_state") or "running") in BILLABLE_STATES
            if s["billable"]:
                s["end"] = t
    for s in spans.values():
        if s["billable"]:
            s["end"] = max(s["end"], now)
    return list(spans.values())


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


def write_timeseries(
    steps,
    hist,
    prof,
    met,
    path: Path,
    *,
    log_counts=None,
    t_launch: float,
    now: float,
    live: bool,
) -> int:
    st_ts, st_vals = status_series(hist)
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
    t0 = steps[0]["t"] if steps else t_launch
    rate = hourly_rate()
    observed = observed_instances(hist, now)

    def cost_at(t: float) -> float:
        if inst:
            return sum(
                i["hourly_usd"] * max(0.0, min(t, i["stopped_at"]) - i["started_at"]) / 3600
                for i in inst
                if t > i["started_at"]
            )
        return sum(
            rate * (min(t, s["end"]) - s["start"]) / 3600 for s in observed if t > s["start"]
        )

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

    furthest, last_world, last_ckpt = 0, 0, -1
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TS_COLUMNS)
        # The BOOT window, sampled at the status polls. Steps are the only other
        # source of rows and the first one lands ~3 minutes after launch, so
        # without these the cost curve, the world-size line and the fleet
        # counters all began mid-panel -- on a run whose boxes were already
        # billing. Same rule as the trailing row: only poll-asserted values.
        first_step_t = steps[0]["t"] if steps else float("inf")
        for t, doc in hist:
            if t >= first_step_t:
                break
            w.writerow(_fleet_row(t, doc, kills, rel, cost_at, furthest=0))
        for i, r in enumerate(steps):
            furthest = max(furthest, r["step"])
            t, elapsed = r["t"], r["t"] - t0
            j = bisect_right(st_ts, t) - 1
            world, ckpt = st_vals[j] if 0 <= j < len(st_vals) else (r["ws"], -1)
            world = r["ws"] or world
            last_world, last_ckpt = world, ckpt
            usd = cost_at(t)
            w.writerow(
                [
                    int(t * 1000),
                    # Before the first checkpoint the durable step is genuinely
                    # 0, not unknown -- and "at risk" is then the whole run.
                    # Writing "" instead left both panels blank for the first
                    # ~90s, which reads as broken rather than as "nothing
                    # durable yet".
                    max(ckpt, 0),
                    r["step"],
                    furthest,
                    max(0, r["step"] - max(ckpt, 0)),
                    round(r["loss"], 4),
                    round(r["step"] * TOKENS_PER_STEP / 1e9, 6),
                    world,
                    bisect_right(kills, t),
                    bisect_right(rel, t),
                    1 if world < TARGET_WORLD else 0,
                    # Shades the degraded x-ranges INSIDE the progress panel by
                    # riding its own y-axis; a second axis would be worse than
                    # no shading at all.
                    furthest if world < TARGET_WORLD else "",
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
        if live:
            doc = hist[-1][1] if hist else {}
            w.writerow(
                _fleet_row(
                    now,
                    doc,
                    kills,
                    rel,
                    cost_at,
                    furthest=furthest,
                    last_step=steps[-1]["step"] if steps else 0,
                    fallback_world=last_world,
                    fallback_ckpt=last_ckpt,
                )
            )
    return len(steps)


def _fleet_row(
    t: float,
    doc: dict,
    kills,
    rel,
    cost_at,
    *,
    furthest: int,
    last_step: int = 0,
    fallback_world: int = 0,
    fallback_ckpt: int = -1,
) -> list:
    """A row sourced from a STATUS POLL rather than from a training step.

    Steps are logged only while training, so the series had rows for the trained
    stretches and nothing else: nothing during the ~3 minute boot, and nothing
    between a kill and the next epoch. On a window that ends at "now" those gaps
    are not stale lines, they are empty margins -- and the right-hand one is
    exactly the moment you are watching the dashboard for.

    The rule for what may appear here: only what the status poll still asserts.
    Cost keeps accruing because the boxes are still billed; membership,
    checkpoint step and the fleet counters come straight from the document.
    train_loss, ms_per_step and current_step are left EMPTY -- they measure a
    step, and no step is happening. Carrying those forward would be inventing
    data, which is worse than a gap.

    `last_step` and `furthest` are NOT interchangeable, and after a rollback they
    differ. Anything about where training actually IS -- tokens processed, work
    at risk, unit cost -- continues from the last step, matching what the step
    rows carry; only `furthest_step` and the degraded band, which are about the
    frontier ever reached, use `furthest`. Using `furthest` for all of them made
    tokens and work-at-risk JUMP on the trailing row of a rolled-back run.
    """
    world = len(doc.get("members") or []) or fallback_world
    ckpt = max(int(doc.get("ckpt_step", fallback_ckpt)), 0)
    usd = cost_at(t)
    degraded = world < TARGET_WORLD
    return [
        int(t * 1000),
        ckpt,
        "",  # current_step: no step is in flight
        furthest,
        max(0, last_step - ckpt),
        "",  # train_loss: measured per step
        round(last_step * TOKENS_PER_STEP / 1e9, 6),
        world,
        bisect_right(kills, t),
        bisect_right(rel, t),
        1 if degraded else 0,
        furthest if degraded and furthest else "",
        "",  # goodput
        "",  # ms_per_step: measured per step
        "",  # ms_per_step_typical
        0,
        round(usd, 4),
        round(usd / last_step * 1000, 4) if last_step else "",
    ]


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


def epoch_timeline(hist: list[tuple[float, dict]], log_path: Path) -> list[dict]:
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
    for t, doc in hist:
        ep = int(doc.get("epoch") or 0)
        if ep == seen or ep not in meta:
            continue
        seen = ep
        members, master = meta[ep]
        out.append({"t": t, "epoch": ep, "members": members, "master": master})
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


def boot_row(hist: list[tuple[float, dict]], target: int) -> tuple[float, list[str]] | None:
    """Every slot's state before the first epoch is published: booting.

    The epoch timeline starts at the first `published epoch`, which on a cold
    fleet is ~3 minutes after launch. Until then occupancy.csv had no rows at
    all, so the Gantt -- the panel you watch precisely to see boxes come up --
    was blank for the whole boot. The status poll knows each box's aws_state
    from launch, so the boot phase is observed, not assumed.
    """
    if not hist:
        return None
    t, doc = hist[0]
    vals = ["down"] * target
    for n in doc.get("nodes") or []:
        sl = int(n.get("node", -1))
        if 0 <= sl < target and (n.get("aws_state") or "") in BILLABLE_STATES:
            vals[sl] = "provisioning"
    return (t, vals)


def write_occupancy(
    epochs: list[dict],
    hist: list[tuple[float, dict]],
    prof: dict,
    path: Path,
    target: int,
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

    # base state at each epoch boundary, preceded by the observed boot
    base: list[tuple[float, list[str]]] = []
    boot = boot_row(hist, target)
    if boot and (not epochs or boot[0] < epochs[0]["t"]):
        base.append(boot)
    for ep in epochs:
        vals = []
        for sl in range(target):
            if sl in ep["members"]:
                vals.append("leader" if sl == ep["master"] else "training")
            else:
                vals.append("down")
        base.append((ep["t"], vals))
    if not base:
        # Header only, but the FILE must exist: nginx 404s an absent CSV and the
        # panel renders a query error rather than an empty chart. On the first
        # tick of a run there is nothing to say yet, and "nothing yet" should
        # look like an empty panel, not like a broken one.
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerow(["time", *[f"slot{s}" for s in range(target)]])
        return 0

    # Slots that have a BILLABLE box but are not in the world are provisioning,
    # not down -- a replacement boots for ~150-250s before it can rejoin. The
    # status poll reports the replacement's instance and aws_state the moment it
    # is launched, so this is observable LIVE; the profile-ledger version below
    # only exists once the run has ended.
    poll_ts = [t for t, _ in hist]
    poll_boxed: list[set[int]] = []
    for _, doc in hist:
        s = set()
        for n in doc.get("nodes") or []:
            if (n.get("aws_state") or "") in BILLABLE_STATES:
                s.add(int(n.get("node", -1)))
        poll_boxed.append(s)

    def boxed_at(t: float) -> set[int]:
        j = bisect_right(poll_ts, t) - 1
        return poll_boxed[j] if 0 <= j < len(poll_boxed) else set()

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

    # transition times: epoch boundaries, each provisioning start, and every
    # poll at which the set of live boxes changed
    changes = {poll_ts[i] for i in range(1, len(poll_boxed)) if poll_boxed[i] != poll_boxed[i - 1]}
    t_first = base[0][0]
    times = sorted(
        {t for t, _ in base}
        | {w[0] for ws in prov.values() for w in ws}
        | {t for t in changes if t >= t_first}
    )
    cols = [f"slot{s}" for s in range(target)]
    cur = ["down"] * target
    bi = 0
    rows = 0
    prev_out = None
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", *cols])
        for t in times:
            while bi < len(base) and base[bi][0] <= t:
                cur = list(base[bi][1])
                bi += 1
            out = list(cur)
            live_boxes = boxed_at(t)
            for sl in range(target):
                if out[sl] != "down":
                    continue
                if sl in live_boxes or any(a_ <= t < b_ for a_, b_ in prov.get(sl, ())):
                    out[sl] = "provisioning"
            # State Timeline holds a value until it changes, so an unchanged row
            # is pure noise -- and on a 36h run the poll-derived times would add
            # thousands of them.
            if out == prev_out:
                continue
            prev_out = out
            w.writerow([int(t * 1000), *out])
            rows += 1
    return rows


def write_world(
    epochs: list[dict],
    path: Path,
    target: int,
    *,
    t_launch: float,
    end_t: float = 0.0,
) -> int:
    """World size on the EPOCH clock, not the step clock.

    Worlds that exist between restarts carry no training steps, so sampling at
    step times drops them: E5's regrow 1->2->3->4->5->6->7->8 rendered as a
    single 1->8 jump. Epoch publications capture every one.

    Two anchors bracket the epochs. The line starts at LAUNCH with world 0 --
    during the boot nothing is training, and without that row the panel had no
    rows at all for the first ~3 minutes of every run. It ends at `end_t`
    (`now`, live), because epochs stop being published once membership settles:
    E5's line stopped at 10:18 while the run ran to 10:35, and mid-regrow the
    line stopped wherever the last epoch was.
    """
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "nodes_training", "below_full_world"])
        n = 0
        if t_launch and (not epochs or t_launch < epochs[0]["t"]):
            w.writerow([int(t_launch * 1000), 0, 1])
        for ep in epochs:
            n = len(ep["members"])
            w.writerow([int(ep["t"] * 1000), n, 1 if n < target else 0])
        if end_t and end_t > (epochs[-1]["t"] if epochs else t_launch):
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


def launch_time(run_id: str, hist: list[tuple[float, dict]], steps: list[dict]) -> float:
    """When the RUN started -- which is the launch, not the first training step.

    The dashboard's default window opens here, so it has to be knowable on the
    first tick, three minutes before any step exists. Run ids end in the unix
    timestamp the orchestrator stamped at launch, which is exactly that. The
    fallbacks matter only for hand-made replay fixtures.

    Anchoring on the first step instead would hide the boot -- the most
    expensive idle minutes of a spot run, and the ones the fleet panels exist
    to show.
    """
    tail = run_id.rsplit("-", 1)[-1]
    if tail.isdigit() and len(tail) == 10:
        return float(tail)
    if hist:
        return hist[0][0]
    if steps:
        return steps[0]["t"]
    return time.time()


def write_index() -> None:
    index = sorted(d.name for d in DATA.iterdir() if d.is_dir())
    (DATA / "runs.csv").write_text("run\n" + "\n".join(index) + "\n")


def write_summary(out_dir: Path, summary: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    # Stat tiles read timeseries.csv, not this. Infinity's JSON parser returned
    # no usable frame for scalar selectors while its CSV parser works everywhere
    # else on this dashboard -- one row keeps every panel on the proven path.
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        # bool is a subclass of int, so `active` would otherwise land in a
        # numeric column as the string "True".
        keys = [
            k for k, v in summary.items() if isinstance(v, int | float) and not isinstance(v, bool)
        ]
        wr.writerow(["time", *keys])
        wr.writerow([summary["run_end_ms"], *[summary[k] for k in keys]])


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
    hist = load_hist(SRC / "status_hist.jsonl")
    epochs = epoch_timeline(hist, SRC / "run.log")
    t_launch = launch_time(run_id, hist, steps)
    # A run that has reported `done` is NOT live any more, even on a --live
    # tick: the last tick of every run is a --live tick. Leaving it marked
    # active pins the window to a receding "now", so a finished run grows an
    # ever-widening empty margin on the right -- the drift the window logic
    # exists to prevent.
    done = bool(hist[-1][1].get("done")) if hist else False
    live = LIVE and not done

    # Live, the right edge is NOW: that is what makes a window ending at "now"
    # show the fleet rather than an empty margin. A finished run pins to its
    # last activity so there is no dead trailing space. --now= freezes it, which
    # is what makes a truncated mid-run fixture replayable in a test.
    frozen = _flag("now")
    settled = steps[-1]["t"] if steps else (hist[-1][0] if hist else t_launch)
    now = float(frozen) if frozen else (time.time() if live else settled)

    n1 = write_timeseries(
        steps,
        hist,
        prof,
        met,
        OUT_DIR / "timeseries.csv",
        log_counts=log_events(epochs, TARGET_WORLD),
        t_launch=t_launch,
        now=now,
        live=live,
    )
    n2 = write_occupancy(epochs, hist, prof, OUT_DIR / "occupancy.csv", TARGET_WORLD)
    n3 = write_world(epochs, OUT_DIR / "world.csv", TARGET_WORLD, t_launch=t_launch, end_t=now)
    n4 = write_degraded(epochs, OUT_DIR / "degraded.json", TARGET_WORLD)
    evals = load_evals(SRC / "logs", steps)
    n5 = write_val(evals, OUT_DIR / "val.csv")
    meta = load_run_meta(SRC / "logs")
    write_meta(meta, OUT_DIR / "meta.csv", t_launch, now)

    t_start_ms = int(t_launch * 1000)
    t_end_ms = int(now * 1000)
    summary = {
        "run_id": run_id,
        # The dashboard's default window is derived from these. A finished run
        # pins to its own end so there is no dead trailing space; a live one
        # runs to "now" so the window follows the fleet.
        "run_start_ms": t_start_ms,
        "run_end_ms": t_end_ms,
        "active": live,
        "goodput": round(met["trained_seconds_total"] / (prof["durations"].get("total_s") or 1), 4),
        "nodes_lost": len(log_events(epochs, TARGET_WORLD)["kills"])
        or sum(1 for e in prof["events"] if e["event"] == "kill"),
        "replacements": len(log_events(epochs, TARGET_WORLD)["relaunches"])
        or sum(1 for e in prof["events"] if e["event"] == "relaunch"),
        "whole_group_restarts": 0,
        "steps": met["steps"] or (steps[-1]["step"] if steps else 0),
        "val_loss": round(met["val_loss"], 4),
        "usd": round(sum(i.get("usd") or 0 for i in prof["cost"]["instances"]), 2)
        or round(
            sum(
                hourly_rate() * (s["end"] - s["start"]) / 3600
                for s in observed_instances(hist, now)
            ),
            2,
        ),
        "wall_hours": round((prof["durations"].get("total_s") or (now - t_launch)) / 3600, 3),
    }
    write_summary(OUT_DIR, summary)
    write_index()

    print(
        f"[{run_id}] timeseries.csv {n1} · occupancy.csv {n2} epochs · world.csv {n3} · "
        f"degraded.json {n4} regions · val.csv {n5} · {meta.get('model') or 'model ?'}"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
