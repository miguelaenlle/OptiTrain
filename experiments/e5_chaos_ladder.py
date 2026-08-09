#!/usr/bin/env python3
"""E5 — 8-node chaos ladder: four escalating failures in one run.

    round 1  t+180s (+-jitter)  kill 1 of 8   -> world 7 -> recover to 8
    round 2  t+680s (+-jitter)  kill 2        -> world 6 -> recover to 8
    round 3  t+1180s(+-jitter)  kill 4        -> world 4 -> recover to 8
    round 4  t+1680s(+-jitter)  kill 7        -> world 1 -> recover to 8

`k` is the independent variable, so it is FIXED and ascending — one round per
value is what makes per-k cost attributable, and ascending means the two values
the 36h model actually needs (k=1, k=2) are banked before the risky k=7 round.

Everything else is RANDOM and seeded: which nodes die, the kill time (+-60s),
and which node survives round 4. The jitter matters more than it looks — it
randomises the phase between kills and the checkpoint cycle, so rollback is
SAMPLED across four rounds instead of being a single lucky draw. E4's rollback
numbers were, in its own words, "phase luck".

Victims are drawn from the LIVE set, so the master and previously-launched
replacements are both eligible, exactly as in a real reclaim.

    python3 experiments/e5_chaos_ladder.py --dry-run   # print plan, launch nothing
    python3 experiments/e5_chaos_ladder.py             # real, ~$8
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, "src")

from orchestrator import aws, experiments  # noqa: E402
from orchestrator.config import OrchestratorConfig  # noqa: E402

BUDGET = int(os.environ.get("TRAIN_TOTAL_SECONDS", "2100"))
FIRST = float(os.environ.get("FIRST_KILL", "180"))
SPACING = float(os.environ.get("ROUND_SPACING", "500"))
JITTER = float(os.environ.get("KILL_JITTER", "60"))
SEED = int(os.environ.get("CHAOS_SEED", "20260808"))
LADDER = [1, 2, 4, 7]  # nodes killed per round — the independent variable


def build_schedule(node_count: int, seed: int) -> list[tuple[float, int]]:
    """[(seconds_after_train_start, node_index)]; a round shares one timestamp.

    Simultaneity is expressed exactly as E4 did it — several entries with the
    same time. Victims are sampled without replacement within a round so one
    round never double-kills a node.
    """
    rng = random.Random(seed)
    sched: list[tuple[float, int]] = []
    for r, k in enumerate(LADDER):
        t = FIRST + r * SPACING + rng.uniform(-JITTER, JITTER)
        for victim in rng.sample(range(node_count), k):
            sched.append((round(t, 1), victim))
    return sched


def rounds_for_print(node_count: int, seed: int) -> list[tuple[float, list[int]]]:
    out: dict[float, list[int]] = {}
    for t, n in build_schedule(node_count, seed):
        out.setdefault(t, []).append(n)
    return sorted((t, sorted(v)) for t, v in out.items())


def main() -> int:
    dry = "--dry-run" in sys.argv
    if dry:
        aws.set_dry_run(True)
    cfg = OrchestratorConfig()
    cfg.require_bucket()

    n = cfg.node_count
    if n != 8:
        print(f"[e5] WARNING: NODES={n}, ladder designed for 8", file=sys.stderr)
    if max(LADDER) >= n:
        raise SystemExit(f"ladder kills {max(LADDER)} of {n} — need at least one survivor")

    schedule = build_schedule(n, SEED)
    per_round = [f"t+{t:.0f}s kill {v} (k={len(v)})" for t, v in rounds_for_print(n, SEED)]

    # Peak vCPU is at round 4: the 7 victims are still terminating (EC2 holds
    # quota until fully released) while their 7 replacements launch.
    peak_vcpu = (n + max(LADDER)) * cfg.instance_vcpu_count()
    print(
        f"\n\033[1m⚠️  BILLABLE: {n}x {cfg.instance_type} ({cfg.spot_market}), "
        f"{BUDGET}s training budget, {len(LADDER)} escalating rounds "
        f"({sum(LADDER)} kills total).\033[0m\n"
        f"[e5] seed={SEED} (rerun with CHAOS_SEED={SEED} to reproduce exactly)\n"
        f"[e5] rounds: {' | '.join(per_round)}\n"
        f"[e5] schedule: {schedule}\n"
        f"[e5] peak vCPU (fleet + {max(LADDER)} simultaneous replacements): "
        f"{peak_vcpu} of {cfg.vcpu_quota}\n"
        f"[e5] instances expected: {n + sum(LADDER)} = {n} + {sum(LADDER)}\n"
        f"[e5] MAX_EPOCHS_WITHOUT_PROGRESS={cfg.max_epochs_without_progress} "
        f"(need > {2 * len(LADDER)}: each round publishes shrink+grow)",
        file=sys.stderr,
    )
    if peak_vcpu > cfg.vcpu_quota:
        raise SystemExit(f"peak {peak_vcpu} vCPU exceeds VCPU_QUOTA={cfg.vcpu_quota}")
    if cfg.max_epochs_without_progress <= 2 * len(LADDER):
        raise SystemExit(
            f"MAX_EPOCHS_WITHOUT_PROGRESS={cfg.max_epochs_without_progress} is too low: "
            f"{len(LADDER)} rounds publish {2 * len(LADDER)} epochs and the floor would "
            f"fire on normal recoveries, discarding healthy survivors. Export 12."
        )
    if dry:
        print("[e5] dry-run: not launching", file=sys.stderr)
        return 0

    profile, metrics = experiments._run_supervised(
        cfg,
        kind="multinode-preempt",
        budget=BUDGET,
        replace_on_loss=True,
        kill_schedule=schedule,
        return_profile=True,
    )
    print(f"\n[e5] run_id={profile.run_id}", file=sys.stderr)
    if metrics is None:
        print("[e5] FAILED: no metrics.json", file=sys.stderr)
        return 1
    print(
        f"[e5] steps={metrics.get('steps')} val={metrics.get('val_loss')} "
        f"trained={metrics.get('trained_seconds_total')} resumed={metrics.get('resumed')} "
        f"restarts={metrics.get('restart_count')} stop={metrics.get('stop_reason')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
