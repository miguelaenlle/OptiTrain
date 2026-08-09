#!/usr/bin/env python3
"""E4 — rolling PAIR failures on a 4-node world.

Three rounds. Each kills TWO nodes at the same instant, lets the world recover to
full, then kills the OTHER pair:

    round 1  t+120s  kill 2,3  -> survivors 0,1  -> recover to 4
    round 2  t+420s  kill 0,1  -> survivors 2,3  -> recover to 4   (kills the MASTER)
    round 3  t+720s  kill 2,3  -> survivors 0,1  -> recover to 4   (kills REPLACEMENTS)

Why a bespoke driver: ``multinode-preempt`` builds its schedule as
``[((k+1)*interval, victims[k]) ...]`` — one victim per entry at strictly
increasing times — so it cannot express a simultaneous pair. Simultaneity is the
whole point; a stagger would leave three survivors and quietly become the
already-measured single-node case.

What this tests that no previous run could: every earlier measurement was a
SINGLE failure, so nothing shows whether the Nth recovery costs the same as the
first. Three comparable rounds inside one run is a stronger comparison than
across runs, because everything else is held identical.

    python3 scripts/e4_rolling_pairs.py            # real
    python3 scripts/e4_rolling_pairs.py --dry-run  # prints the plan, launches nothing
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

from orchestrator import aws, experiments  # noqa: E402
from orchestrator.config import OrchestratorConfig  # noqa: E402

BUDGET = int(os.environ.get("TRAIN_TOTAL_SECONDS", "900"))
# 300s between rounds: E2b took ~210s from kill to full world, so this leaves
# ~90s of full-world training in between. Tighter and the rounds overlap, which
# measures cascading failure — a different (and worthwhile) experiment.
SPACING = float(os.environ.get("ROUND_SPACING", "300"))
FIRST = float(os.environ.get("FIRST_KILL", "120"))
ROUNDS = [(2, 3), (0, 1), (2, 3)]


def build_schedule() -> list[tuple[float, int]]:
    """(seconds_after_train_start, node) — both members of a pair share a time."""
    return [(FIRST + r * SPACING, victim) for r, pair in enumerate(ROUNDS) for victim in pair]


def main() -> int:
    dry = "--dry-run" in sys.argv
    if dry:
        aws.set_dry_run(True)
    cfg = OrchestratorConfig()
    cfg.require_bucket()
    schedule = build_schedule()

    per_round = [f"t+{FIRST + r * SPACING:.0f}s kill {list(pair)}" for r, pair in enumerate(ROUNDS)]
    peak_vcpu = (cfg.node_count + 2) * cfg.instance_vcpu_count()
    print(
        f"\n\033[1m⚠️  BILLABLE: {cfg.node_count}x {cfg.instance_type} ({cfg.spot_market}), "
        f"{BUDGET}s training budget, {len(ROUNDS)} rounds of PAIR kills.\033[0m\n"
        f"[e4] rounds: {' | '.join(per_round)}\n"
        f"[e4] schedule: {schedule}\n"
        f"[e4] peak vCPU (fleet + 2 simultaneous replacements): {peak_vcpu} of {cfg.vcpu_quota}\n"
        f"[e4] NOTE: {len(ROUNDS)} rounds x 2 epochs = {len(ROUNDS) * 2} epochs, and "
        f"max_epochs_without_progress=6 — safe only because checkpoints land between "
        f"rounds and reset the counter. A whole-group restart here IS the finding.",
        file=sys.stderr,
    )
    if peak_vcpu > cfg.vcpu_quota:
        raise SystemExit(f"peak {peak_vcpu} vCPU exceeds VCPU_QUOTA={cfg.vcpu_quota}")
    if dry:
        print("[e4] dry-run: not launching", file=sys.stderr)
        return 0

    profile, metrics = experiments._run_supervised(
        cfg,
        kind="multinode-preempt",
        budget=BUDGET,
        replace_on_loss=True,
        kill_schedule=schedule,
        return_profile=True,
    )
    print(f"\n[e4] run_id={profile.run_id}", file=sys.stderr)
    if metrics is None:
        print("[e4] FAILED: no metrics.json", file=sys.stderr)
        return 1
    print(
        f"[e4] steps={metrics.get('steps')} val={metrics.get('val_loss')} "
        f"trained={metrics.get('trained_seconds_total')} resumed={metrics.get('resumed')} "
        f"restarts={metrics.get('restart_count')} stop={metrics.get('stop_reason')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
