#!/usr/bin/env python3
"""Live dashboard loop: poll a running run, regenerate the CSVs + dashboard.

WHY THIS AND NOT A DATASOURCE PLUGIN
The intended live path was the Infinity datasource fetching CSVs over HTTP. Its
browser module fails to load in this environment (`react/jsx-runtime` 404) on
both Grafana 11.3 and 12.1.0, and the failure is silent -- backend queries
succeed, panels read "No data". Rather than keep guessing plugin/Grafana pairs
before a billed run, this uses the refresh path that is already proven: Grafana
re-reads `dashboards/*.json` from disk every 10s (provisioning
updateIntervalSeconds), so regenerating that file IS a live update.

It costs a 200KB rewrite every 10s and needs no plugin, no HTTP origin, and no
new failure mode on the day of the run.

Sources while a run is IN FLIGHT (profile.json only exists at the end):
  status.json      polled from S3 -> world size, ckpt_step, epoch, members
  logs/*.log       streamed to S3 -> per-step loss/ms/ws with wall clock
  supervisor log   local run log  -> epoch membership + master

    python3 deploy/grafana/live.py <run_id> [--interval 10] [--once]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
LIVE_ROOT = HERE / ".live"


def work_dir(run_id: str) -> Path:
    """Per-RUN working dir.

    A single shared .live/ was never cleared between runs, and `aws s3 cp` of an
    object that does not exist yet leaves the previous file in place. So a new
    run started life holding the LAST run's profile.json, metrics.json and
    boot-node logs. Two consequences, both silent: export.py's run-id guard
    matched the stale profile and refused every tick until the new run's own
    profile landed at the very end, and the old run's step lines were parsed as
    if they belonged to the new one.
    """
    return LIVE_ROOT / run_id


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def sync(bucket: str, run_id: str, WORK: Path) -> None:
    """Pull whatever exists so far. Missing objects are normal mid-run."""
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    base = f"s3://{bucket}/runs/{run_id}"
    for name in ("status.json", "profile.json", "metrics.json"):
        sh(
            "aws",
            "s3",
            "cp",
            f"{base}/{name}",
            str(WORK / name),
            "--region",
            "us-east-1",
            "--only-show-errors",
        )
    sh(
        "aws",
        "s3",
        "sync",
        f"{base}/logs/",
        str(WORK / "logs"),
        "--region",
        "us-east-1",
        "--only-show-errors",
    )
    # The supervisor's own per-tick status objects. This is what makes fleet
    # history survive a closed laptop: sync only fetches what is new, so a poll
    # loop that was offline for an hour backfills that hour instead of leaving a
    # permanent hole in world size and the Gantt.
    (WORK / "status").mkdir(parents=True, exist_ok=True)
    sh(
        "aws",
        "s3",
        "sync",
        f"{base}/status/",
        str(WORK / "status"),
        "--region",
        "us-east-1",
        "--only-show-errors",
    )


def copy_driver_log(log_path: str | None, WORK: Path) -> None:
    """Epoch publications live in the DRIVER's log, not in S3.

    export.py reads epoch membership + master from SRC/run.log, and without it
    world.csv and occupancy.csv come out empty -- the Gantt and world-size panels
    silently render nothing while every other panel works.
    """
    if not log_path:
        return
    src = Path(log_path)
    if src.exists():
        (WORK / "run.log").write_text(src.read_text(errors="replace"))


def append_status(WORK: Path) -> None:
    """Build status_hist.jsonl, preferring the SUPERVISOR's durable per-tick
    objects over what this poll loop happened to observe.

    status.json is overwritten in place, so polling it is inherently lossy: an
    hour with the laptop lid closed used to be an hour of world-size and Gantt
    history that no longer existed anywhere. The supervisor now publishes each
    tick under runs/<id>/status/, which sync backfills, so the history is
    complete regardless of when this loop was running.

    The poll path is kept as a fallback for runs launched before that existed
    (and for a supervisor too old to publish them), which is why this merges the
    two sources rather than replacing one with the other.
    """
    hist = WORK / "status_hist.jsonl"
    by_updated: dict[str, tuple[float, dict]] = {}
    for line in hist.read_text().splitlines() if hist.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(r.get("doc", {}).get("updated_at"))
        by_updated.setdefault(key, (float(r["t"]), r["doc"]))

    # Durable ticks. The filename is the supervisor's own millisecond clock, so
    # it dates the observation correctly even when we fetch it an hour late --
    # using fetch time here would smear a backfilled hour onto one instant.
    for p in sorted((WORK / "status").glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            t = int(p.stem) / 1000.0
        except ValueError:
            t = float(doc.get("updated_at") or 0.0)
        by_updated[str(doc.get("updated_at"))] = (t, doc)

    src = WORK / "status.json"
    if src.exists():
        try:
            doc = json.loads(src.read_text())
            by_updated.setdefault(str(doc.get("updated_at")), (time.time(), doc))
        except json.JSONDecodeError:
            pass

    rows = sorted(by_updated.values(), key=lambda r: r[0])
    hist.write_text("".join(json.dumps({"t": t, "doc": d}) + "\n" for t, d in rows))


def regenerate(run_id: str, nodes: int, WORK: Path) -> str:
    """Run the existing exporter + builder against this run's working dir."""
    r = subprocess.run(
        [
            sys.executable,
            str(HERE / "export.py"),
            run_id,
            "--live",
            f"--nodes={nodes}",
            f"--src={WORK}",
        ],
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    # --run pins the dashboard's default selection to the LIVE run, so replaying
    # an older run in another shell cannot steal it mid-flight.
    subprocess.run(
        [sys.executable, str(HERE / "build_dashboard.py"), f"--run={run_id}"],
        capture_output=True,
        text=True,
    )
    return out.strip().splitlines()[-1] if out.strip() else "(no output)"


def main() -> int:
    import os

    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run_id = sys.argv[1]
    interval = 10.0
    nodes = 4
    for a in sys.argv[2:]:
        if a.startswith("--interval"):
            interval = float(a.split("=")[-1])
        if a.startswith("--nodes"):
            nodes = int(a.split("=")[-1])
    once = "--once" in sys.argv
    log_path = next((a.split("=", 1)[-1] for a in sys.argv if a.startswith("--log=")), None)
    bucket = os.environ.get("SPOT_TRAIN_BUCKET")
    if not bucket:
        raise SystemExit("SPOT_TRAIN_BUCKET unset")

    WORK = work_dir(run_id)
    WORK.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[live] {run_id} every {interval:.0f}s -> {DATA}")
    # Seed BEFORE the first S3 poll. The run id carries its launch timestamp, so
    # this is enough to put the run in the selector and open the default window
    # on it -- the dashboard is live from the moment the fleet is launched
    # rather than from the first training step ~3 minutes later.
    regenerate(run_id, nodes, WORK)
    n = 0
    while True:
        sync(bucket, run_id, WORK)
        copy_driver_log(log_path, WORK)
        append_status(WORK)
        line = regenerate(run_id, nodes, WORK)
        n += 1
        print(f"[live] tick {n}: {line}", flush=True)
        if once:
            return 0
        done = False
        s = WORK / "status.json"
        if s.exists():
            try:
                done = bool(json.loads(s.read_text()).get("done"))
            except json.JSONDecodeError:
                done = False
        if done:
            print("[live] run reports done; final regenerate complete")
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
