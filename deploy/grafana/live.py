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
WORK = HERE / ".live"


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def sync(bucket: str, run_id: str) -> None:
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


def copy_driver_log(log_path: str | None) -> None:
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


def append_status(run_id: str) -> None:
    """status.json is OVERWRITTEN in place, so its history exists only if we
    keep every poll. Without this the world-size staircase and ckpt_step series
    are unrecoverable -- the same gap that had to be patched by hand for E5."""
    src = WORK / "status.json"
    if not src.exists():
        return
    try:
        doc = json.loads(src.read_text())
    except json.JSONDecodeError:
        return
    hist = WORK / "status_hist.jsonl"
    prev = None
    if hist.exists():
        lines = hist.read_text().splitlines()
        if lines:
            try:
                prev = json.loads(lines[-1])["doc"].get("updated_at")
            except Exception:
                prev = None
    if doc.get("updated_at") != prev:
        with hist.open("a") as fh:
            fh.write(json.dumps({"t": time.time(), "doc": doc}) + "\n")


def regenerate(run_id: str, nodes: int) -> str:
    """Run the existing exporter + builder against the live working dir."""
    env_src = f"SRC_OVERRIDE={WORK}"
    r = subprocess.run(
        [sys.executable, str(HERE / "export.py"), run_id, "--live", f"--nodes={nodes}"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "X": env_src},
    )
    out = (r.stdout or "") + (r.stderr or "")
    subprocess.run(
        [sys.executable, str(HERE / "build_dashboard.py")], capture_output=True, text=True
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

    WORK.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[live] {run_id} every {interval:.0f}s -> {DATA}")
    n = 0
    while True:
        sync(bucket, run_id)
        copy_driver_log(log_path)
        append_status(run_id)
        line = regenerate(run_id, nodes)
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
