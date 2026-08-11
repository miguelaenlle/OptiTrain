#!/usr/bin/env python3
"""E7 — 1 node, one kill, resume from checkpoint. Live-dashboard proof.

The smallest end-to-end exercise of the live pipeline: train, lose the only
node, boot a replacement, resume from the S3 checkpoint, finish the budget.

Note what this does and does NOT test. At N=1 a kill takes the world to ZERO --
there are no survivors to train through the gap -- so the dashboard will show
world 1 -> 0 -> 1 with a flat progress line during the replacement's boot. That
is correct behaviour, not a stall: this run exercises resume-from-checkpoint,
not degraded-world training.

    python3 experiments/e7_one_node.py --dry-run
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

from orchestrator import aws, experiments  # noqa: E402
from orchestrator.config import OrchestratorConfig  # noqa: E402

BUDGET = int(os.environ.get("TRAIN_TOTAL_SECONDS", "300"))
KILL_AT = float(os.environ.get("FIRST_KILL", "60"))


def main() -> int:
    dry = "--dry-run" in sys.argv
    if dry:
        aws.set_dry_run(True)
    cfg = OrchestratorConfig()
    cfg.require_bucket()
    schedule = [(KILL_AT, 0)]
    print(
        f"\n\033[1m⚠️  BILLABLE: {cfg.node_count}x {cfg.instance_type} "
        f"({cfg.spot_market}), {BUDGET}s budget, 1 kill at t+{KILL_AT:.0f}s.\033[0m\n"
        f"[e7] schedule: {schedule}\n"
        f"[e7] instances expected: {cfg.node_count + 1}",
        file=sys.stderr,
    )
    if dry:
        print("[e7] dry-run: not launching", file=sys.stderr)
        return 0
    profile, metrics = experiments._run_supervised(
        cfg,
        kind="multinode-preempt",
        budget=BUDGET,
        replace_on_loss=True,
        kill_schedule=schedule,
        return_profile=True,
    )
    print(f"\n[e7] run_id={profile.run_id}", file=sys.stderr)
    if metrics is None:
        print("[e7] FAILED: no metrics.json", file=sys.stderr)
        return 1
    print(
        f"[e7] steps={metrics.get('steps')} trained={metrics.get('trained_seconds_total')} "
        f"resumed={metrics.get('resumed')} stop={metrics.get('stop_reason')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
