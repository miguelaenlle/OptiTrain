#!/usr/bin/env python3
"""E6 — 4 nodes, three STAGGERED single-node failures. Live-dashboard proof.

    kill 1 of 4 at t+30s   -> world 3 -> recover to 4
    kill 1 of 4 at t+330s  -> world 3 -> recover to 4
    kill 1 of 4 at t+630s  -> world 3 -> recover to 4

Staggered, not simultaneous, and spaced ~300s because E5 measured recovery at
~230-260s: each failure must fully resolve before the next, or the dashboard
shows overlapping recoveries and proves nothing about a single one.

The FIRST kill lands 30s into training deliberately -- early enough that a
viewer watching the dashboard sees a dip almost immediately rather than waiting
out a quiet prologue.

Victims are drawn round-robin rather than randomly: this run exists to verify
the live pipeline end to end, so a reader should be able to predict what the
Gantt ought to look like and check it.

    python3 experiments/e6_staggered.py --dry-run
    python3 experiments/e6_staggered.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

from orchestrator import aws, experiments  # noqa: E402
from orchestrator.config import OrchestratorConfig  # noqa: E402

BUDGET = int(os.environ.get("TRAIN_TOTAL_SECONDS", "900"))
FIRST = float(os.environ.get("FIRST_KILL", "30"))
SPACING = float(os.environ.get("ROUND_SPACING", "300"))
VICTIMS = [1, 2, 3]  # never node 0 twice; keeps one original alive throughout


def build_schedule() -> list[tuple[float, int]]:
    return [(FIRST + i * SPACING, v) for i, v in enumerate(VICTIMS)]


def main() -> int:
    dry = "--dry-run" in sys.argv
    if dry:
        aws.set_dry_run(True)
    cfg = OrchestratorConfig()
    cfg.require_bucket()
    schedule = build_schedule()
    peak = (cfg.node_count + 1) * cfg.instance_vcpu_count()
    print(
        f"\n\033[1m⚠️  BILLABLE: {cfg.node_count}x {cfg.instance_type} "
        f"({cfg.spot_market}), {BUDGET}s training budget, {len(VICTIMS)} staggered "
        f"single-node kills.\033[0m\n"
        f"[e6] schedule: {schedule}\n"
        f"[e6] peak vCPU: {peak} of {cfg.vcpu_quota}\n"
        f"[e6] instances expected: {cfg.node_count + len(VICTIMS)}",
        file=sys.stderr,
    )
    if peak > cfg.vcpu_quota:
        raise SystemExit(f"peak {peak} vCPU exceeds VCPU_QUOTA={cfg.vcpu_quota}")
    if dry:
        print("[e6] dry-run: not launching", file=sys.stderr)
        return 0

    profile, metrics = experiments._run_supervised(
        cfg,
        kind="multinode-preempt",
        budget=BUDGET,
        replace_on_loss=True,
        kill_schedule=schedule,
        return_profile=True,
    )
    print(f"\n[e6] run_id={profile.run_id}", file=sys.stderr)
    if metrics is None:
        print("[e6] FAILED: no metrics.json", file=sys.stderr)
        return 1
    print(
        f"[e6] steps={metrics.get('steps')} val={metrics.get('val_loss')} "
        f"trained={metrics.get('trained_seconds_total')} "
        f"restarts={metrics.get('restart_count')} stop={metrics.get('stop_reason')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
