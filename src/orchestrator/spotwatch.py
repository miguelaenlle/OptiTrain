"""Spotwatch — a 72-hour unattended answer to "can we even GET spot GPUs?".

``spot-orchestrate spotwatch deploy`` provisions a Lambda + a 10-minute
EventBridge rule that collects, per tick: spot placement scores over a fixed
PER-REGION request matrix (a response is capped at 10 rows, so only a
single-region ask gives every AZ a continuous series), spot prices, (daily) pool
offerings + AWS Spot Advisor interruption rates, and — at most once every 6
hours — a real 1-instance spot launch that is immediately given back. Everything
lands as append-only JSONL in ``s3://<bucket>/spotwatch/<date>/``.

``report`` turns days of that into the availability picture, in this order:
can we get an 8-node world in ONE AZ (the query that actually decides whether
this project can run on spot), how each AZ/GPU did in ABSOLUTE terms (best
score, longest good window — a ranked table of 3s must never read as success),
what switching AZ / GPU / both / region would buy, and only then prices,
interruption rates and the SPS-vs-probe calibration.

The collector itself is ``lambda_spotwatch.py`` (runs in AWS, single-file zip,
zero third-party deps). This module is the laptop side: provision, tear down,
report. It costs ~$1-3/month — Lambda's free tier covers 144 ticks/day and the
JSONL is ~1 GB/month; the truth probes are the only real spend, ~1 minute of
g5.xlarge spot four times a day.

``deploy`` and ``down`` are idempotent and honour ``--dry-run``; ``report`` is
pure S3 reads (no mutation, no instances) so it is always safe to run.
"""

from __future__ import annotations

import io
import json
import os
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from . import aws
from .config import OrchestratorConfig

# Names of everything `deploy` creates — `down` deletes exactly this set.
FUNCTION_NAME = "spotwatch"
ROLE_NAME = "spotwatch-lambda-role"
POLICY_NAME = "spotwatch-inline"
RULE_NAME = "spotwatch-tick"
TARGET_ID = "spotwatch-lambda"
PERMISSION_SID = "spotwatch-events-invoke"
RUNTIME = "python3.12"
HANDLER = "lambda_spotwatch.handler"
# The matrix is per-region (a placement-score response is capped at 10 rows, so
# only a single-region ask covers every AZ): ~456 requests at 10/s pacing is
# ~45-120s, plus a price call per region and up to a 45s probe wait. 600s leaves
# 3x headroom on the worst observed tick; the Lambda is billed per ms, so a
# generous ceiling costs nothing.
TIMEOUT_SECONDS = 600
MEMORY_MB = 256

_MODULE = "lambda_spotwatch.py"


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #
def build_package() -> bytes:
    """Zip the handler in memory — no build step, no container image, nothing to
    keep in sync. The timestamp is pinned so the same source always produces the
    same bytes; ``deploy`` compares the zip's SHA-256 against the deployed one
    and skips the code update when nothing changed."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODULE)
    with open(src, "rb") as f:
        data = f.read()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(_MODULE, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        z.writestr(info, data)
    return buf.getvalue()


def lambda_policy(bucket: str, prefix: str) -> dict:
    """Least privilege for the collector. The two interesting lines:

    * ``ec2:RunInstances``/``TerminateInstances`` exist only for the truth probe
      — the collector must be able to give capacity back, or a failed probe
      would bill forever.
    * S3 is scoped to the spotwatch prefix, so a bug in the collector can't
      touch checkpoints or run profiles.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "Observe",
                "Effect": "Allow",
                "Action": [
                    "ec2:GetSpotPlacementScores",
                    "ec2:DescribeSpotPriceHistory",
                    "ec2:DescribeInstanceTypeOfferings",
                    "ec2:DescribeAvailabilityZones",
                    "ec2:DescribeRegions",
                    "ec2:DescribeInstances",
                    "ec2:DescribeImages",
                    # DescribeVpcs: the probe checks for a default VPC first, so
                    # "no VPC here" is never recorded as "no capacity here".
                    "ec2:DescribeVpcs",
                ],
                "Resource": "*",
            },
            {
                "Sid": "TruthProbe",
                "Effect": "Allow",
                "Action": ["ec2:RunInstances", "ec2:TerminateInstances", "ec2:CreateTags"],
                "Resource": "*",
            },
            {
                "Sid": "WriteCollectedData",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject"],
                "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*",
            },
            {
                "Sid": "ListOwnPrefix",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": {"StringLike": {"s3:prefix": [f"{prefix}/*"]}},
            },
            {
                "Sid": "Logs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": "*",
            },
        ],
    }


def lambda_env(cfg: OrchestratorConfig) -> dict[str, str]:
    """Environment the collector reads. Note AWS_REGION is injected by Lambda —
    we never set it (reserved key)."""
    return {
        "SPOTWATCH_BUCKET": cfg.bucket,
        "SPOTWATCH_PREFIX": cfg.spotwatch_prefix,
        "SPOTWATCH_PROBE_REGION": cfg.region,
        "SPOTWATCH_PROBE_TYPE": cfg.spotwatch_probe_type,
        "SPOTWATCH_PROBE_ENABLED": "1" if cfg.spotwatch_probe_enabled else "0",
        "SPOTWATCH_PROBE_MIN_HOURS": str(cfg.spotwatch_probe_min_hours),
        "SPOTWATCH_DAILY_HOUR": str(cfg.spotwatch_daily_hour),
    }


# --------------------------------------------------------------------------- #
# deploy / down
# --------------------------------------------------------------------------- #
def deploy(cfg: OrchestratorConfig) -> None:
    cfg.require_bucket()
    aws.set_region(cfg.region)

    zip_bytes = build_package()
    role = aws.ensure_service_role(
        ROLE_NAME,
        "lambda.amazonaws.com",
        POLICY_NAME,
        lambda_policy(cfg.bucket, cfg.spotwatch_prefix),
    )
    fn_arn = aws.ensure_lambda_function(
        function_name=FUNCTION_NAME,
        role=role,
        handler=HANDLER,
        runtime=RUNTIME,
        zip_bytes=zip_bytes,
        env=lambda_env(cfg),
        timeout=TIMEOUT_SECONDS,
        memory_mb=MEMORY_MB,
        description="spotwatch: GPU spot availability collector (orchestrator/spotwatch.py)",
    )
    schedule = f"rate({cfg.spotwatch_interval_minutes} minutes)"
    rule_arn = aws.ensure_schedule_rule(
        RULE_NAME, schedule, "spotwatch collector tick (spot-orchestrate spotwatch)"
    )
    # Permission before target: EventBridge validates nothing at PutTargets, so
    # the wrong order just silently drops invocations.
    aws.ensure_lambda_permission(FUNCTION_NAME, PERMISSION_SID, "events.amazonaws.com", rule_arn)
    aws.put_rule_target(RULE_NAME, TARGET_ID, fn_arn)

    print(
        f"[spotwatch] deployed {FUNCTION_NAME} ({len(zip_bytes)} B) on {schedule} in "
        f"{cfg.region} -> s3://{cfg.bucket}/{cfg.spotwatch_prefix}/",
        file=sys.stderr,
    )
    from .lambda_spotwatch import sps_requests  # matrix size, without duplicating it

    per_region = len(sps_requests(["us-east-1"]))
    print(
        f"[spotwatch] placement-score matrix: {per_region - 14} requests per region + 14 "
        "worldwide leaderboard, paced at 10/s (measured bucket: 100 tokens, 20/s refill)",
        file=sys.stderr,
    )
    probes = (
        "disabled"
        if not cfg.spotwatch_probe_enabled
        else (
            f"{cfg.spotwatch_probe_type} at most 1 per {cfg.spotwatch_probe_min_hours}h, "
            f"skipped whenever a project=spot-train box is up in {cfg.region}"
        )
    )
    print(f"[spotwatch] truth probes: {probes}", file=sys.stderr)
    if cfg.spotwatch_probe_enabled:
        _check_probe_vpc(cfg)
    print(
        "[spotwatch] first tick lands within "
        f"{cfg.spotwatch_interval_minutes} min; then: spot-orchestrate spotwatch report",
        file=sys.stderr,
    )


def _check_probe_vpc(cfg: OrchestratorConfig) -> None:
    """Warn at deploy time if the probe region has no default VPC.

    The probe launches with no SubnetId, which requires one. Reporting it here
    (loudly, but without failing the deploy — the scores are the main dish)
    means a VPCIdNotSpecified error can never be mistaken later for a real
    InsufficientInstanceCapacity signal; the collector skips the probe outright
    and says why in the data."""
    vpc = aws.default_vpc_id(cfg.region)
    if vpc:
        print(f"[spotwatch] probe region {cfg.region} has default VPC {vpc}", file=sys.stderr)
        return
    print(
        f"[spotwatch] WARNING: no default VPC in {cfg.region} — truth probes will be "
        "SKIPPED (recorded as probe_skipped, never as 'no capacity'). Create one with "
        f"`aws ec2 create-default-vpc --region {cfg.region}` to enable them.",
        file=sys.stderr,
    )


def down(cfg: OrchestratorConfig) -> None:
    """Remove everything ``deploy`` created. Collected data in S3 is left alone
    — it is the experiment's result, and it costs cents to keep."""
    aws.set_region(cfg.region)
    aws.delete_schedule_rule(RULE_NAME, [TARGET_ID])
    aws.remove_lambda_permission(FUNCTION_NAME, PERMISSION_SID)
    aws.delete_lambda_function(FUNCTION_NAME)
    aws.delete_role(ROLE_NAME)
    print(
        f"[spotwatch] removed rule/function/role; collected data kept under "
        f"s3://{cfg.bucket}/{cfg.spotwatch_prefix}/",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Loading (pure parsing, so the analysis below is testable on synthetic data)
# --------------------------------------------------------------------------- #
def parse_shard(text: str) -> list[dict]:
    """One JSON object per line; a truncated final line (a Lambda killed
    mid-write) is skipped rather than failing the whole report."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def shard_epoch(key: str) -> float | None:
    """Epoch encoded in ``<prefix>/<date>/<epoch>-<tick>.jsonl``. Lets the report
    pick a time window from the key names alone — no GetObject on shards that
    fall outside it."""
    base = key.rsplit("/", 1)[-1]
    if not base.endswith(".jsonl"):
        return None
    head = base[: -len(".jsonl")].split("-")[0]
    try:
        return float(head)
    except ValueError:
        return None


def keys_in_window(keys: list[str], cutoff_t: float) -> list[str]:
    return sorted(k for k in keys if (e := shard_epoch(k)) is not None and e >= cutoff_t)


def load_records(cfg: OrchestratorConfig, since_hours: float) -> list[dict]:
    cfg.require_bucket()
    aws.set_region(cfg.region)
    cutoff = time.time() - since_hours * 3600
    keys = keys_in_window(aws.list_keys(cfg.bucket, f"{cfg.spotwatch_prefix}/"), cutoff)
    if not keys:
        raise SystemExit(
            f"no spotwatch shards in the last {since_hours:g}h under "
            f"s3://{cfg.bucket}/{cfg.spotwatch_prefix}/ — deployed? (spotwatch deploy)"
        )
    records: list[dict] = []
    for text in aws.get_texts(cfg.bucket, keys).values():
        records.extend(parse_shard(text))
    return [r for r in records if r.get("t", 0) >= cutoff]


# --------------------------------------------------------------------------- #
# Analysis (pure functions over records — no AWS, no clock)
# --------------------------------------------------------------------------- #
def _of(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r.get("type") == kind]


def _hour(rec: dict) -> int:
    return datetime.fromtimestamp(rec["t"], timezone.utc).hour


def _weekday(rec: dict) -> str:
    return datetime.fromtimestamp(rec["t"], timezone.utc).strftime("%a")


def sps_rows(
    records: list[dict],
    *,
    capacity: int,
    single_az: bool | None = None,
    scope: str | None = "region",
) -> list[dict]:
    """Scored rows, defaulting to the per-region (complete-coverage) scope.

    ``scope="global_top10"`` rows are the all-regions leaderboard: AWS truncates
    every response to 10 rows, so an AZ missing from them means "not in today's
    top ten", NOT "no capacity". Mixing the two would corrupt every rate we
    compute, so nothing but the leaderboard section may ask for that scope.
    """
    rows = [
        r
        for r in _of(records, "sps")
        if r.get("capacity") == capacity and isinstance(r.get("score"), int | float)
    ]
    if scope is not None:
        # Shards written before scope existed are per-region by construction.
        rows = [r for r in rows if r.get("scope", "region") == scope]
    if single_az is not None:
        rows = [r for r in rows if bool(r.get("single_az")) is single_az]
    return rows


def coverage(records: list[dict]) -> dict[str, Any]:
    """How much of what we asked for actually came back.

    Printed before any rate, because a rate computed over a half-collected
    window is worse than no rate at all."""
    ticks = _of(records, "tick")
    gaps = _of(records, "sps_gap")
    asked = sum(int(t.get("sps_requests", 0)) for t in ticks)
    return {
        "ticks": len(ticks),
        "requested": asked,
        "gaps": len(gaps),
        "throttled": sum(1 for g in gaps if g.get("throttled")),
        "gap_pct": (len(gaps) / asked) if asked else None,
        "span_hours": (
            (max(t["t"] for t in ticks) - min(t["t"] for t in ticks)) / 3600 if ticks else 0.0
        ),
    }


def per_tick_best(rows: list[dict], predicate: Callable[[dict], bool]) -> dict[str, float]:
    """Best score available at each tick among the pools the caller would accept.

    This is the core of every scenario: "at this moment, was ANY acceptable pool
    good?" — grouped by tick so a scenario that watches 40 pools isn't credited
    40 times for one lucky moment."""
    best: dict[str, float] = {}
    for r in rows:
        if not predicate(r):
            continue
        tick = r.get("tick_id", "")
        score = float(r["score"])
        if score > best.get(tick, -1.0):
            best[tick] = score
    return best


def odds(best_by_tick: dict[str, float], threshold: float) -> dict[str, Any]:
    scores = list(best_by_tick.values())
    if not scores:
        return {"ticks": 0, "p_good": None, "mean_best": None, "max_best": None}
    good = sum(1 for s in scores if s >= threshold)
    return {
        "ticks": len(scores),
        "p_good": good / len(scores),
        "mean_best": statistics.fmean(scores),
        "max_best": max(scores),
    }


def scenarios(
    records: list[dict],
    *,
    threshold: float,
    home_region: str,
    home_type: str,
    capacity: int,
) -> list[dict]:
    """The five questions, answered as P(a pool scores >= threshold at a tick).

    Each row widens what we are willing to change: nothing, when we ask, the GPU
    type, the AZ, or both — and finally the region, which tells you whether the
    home region is the binding constraint.
    """
    region_rows = sps_rows(records, capacity=capacity, single_az=False)
    az_rows = sps_rows(records, capacity=capacity, single_az=True)

    def _row(name: str, rows: list[dict], pred: Callable[[dict], bool]) -> dict:
        return {"scenario": name, **odds(per_tick_best(rows, pred), threshold)}

    def _home(r: dict) -> bool:
        return r.get("region") == home_region and r.get("instance_type") == home_type

    # The first two rows are region-level scores: "8 instances somewhere in this
    # region", which AWS is free to answer by spreading them across AZs. A
    # training world cannot use that (NCCL bandwidth, cross-AZ transfer charges),
    # so they are marked * and read as an UPPER BOUND, never as a plan.
    out = [
        _row("never switch (same region, same GPU) *", region_rows, _home),
        _row(
            "switch GPU only (same region, any of our GPUs) *",
            region_rows,
            lambda r: r.get("region") == home_region and r.get("instance_type") != "any",
        ),
        _row(
            "switch AZ only (same region+GPU, best AZ)",
            az_rows,
            lambda r: r.get("region") == home_region and r.get("instance_type") == home_type,
        ),
        _row(
            "switch both (same region, any GPU, best AZ)",
            az_rows,
            lambda r: r.get("region") == home_region and r.get("instance_type") != "any",
        ),
        _row(
            "switch region too (anywhere, any GPU, best AZ)",
            az_rows,
            lambda r: r.get("instance_type") != "any",
        ),
    ]
    # The basket request asks AWS the honest question ("any of these six") and is
    # not subject to the "one instance type always scores low" caveat, so it is
    # the sanity check on the per-type rows above.
    out.append(
        _row(
            "  [basket score, same region, all 6 GPUs in one ask]",
            az_rows,
            lambda r: r.get("region") == home_region and r.get("instance_type") == "any",
        )
    )
    return out


def hourly_odds(
    records: list[dict], *, threshold: float, home_region: str, home_type: str, capacity: int
) -> dict[int, dict]:
    """Scenario 1 (wait for a good time), resolved by hour of day."""
    rows = [
        r
        for r in sps_rows(records, capacity=capacity, single_az=True)
        if r.get("region") == home_region and r.get("instance_type") == home_type
    ]
    by_hour: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_hour[_hour(r)].append(r)
    return {
        h: odds(per_tick_best(rs, lambda _r: True), threshold) for h, rs in sorted(by_hour.items())
    }


def dow_hour_odds(
    records: list[dict], *, threshold: float, home_region: str, home_type: str, capacity: int
) -> dict[tuple[str, int], dict]:
    """Same, split by weekday too — this is where "3 AM Saturday is fine" shows
    up. Buckets are thin at 72h (~6 ticks each); the sample count rides along."""
    rows = [
        r
        for r in sps_rows(records, capacity=capacity, single_az=True)
        if r.get("region") == home_region and r.get("instance_type") == home_type
    ]
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(_weekday(r), _hour(r))].append(r)
    return {k: odds(per_tick_best(rs, lambda _r: True), threshold) for k, rs in buckets.items()}


def heatmap(rows: list[dict], row_key: str) -> dict[str, dict[int, float]]:
    """mean score per (row_key value, hour of day)."""
    cells: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[str(r.get(row_key, ""))][_hour(r)].append(float(r["score"]))
    return {
        name: {h: statistics.fmean(v) for h, v in sorted(hours.items())}
        for name, hours in sorted(cells.items())
    }


def availability(
    records: list[dict],
    *,
    capacity: int,
    threshold: float,
    region: str | None = None,
    single_az: bool = True,
) -> list[dict]:
    """ABSOLUTE availability per (region, AZ, type): best score ever seen, how
    many samples cleared the bar, and the longest unbroken stretch that did.

    Deliberately not a ranking. In a drought every pool is a 3 and a ranked
    table of 3s reads as success at a glance; ``best`` and ``longest_good_s``
    make "nothing here was ever usable" impossible to misread. AWS advises a
    score of >= 7 before attempting a launch, which is the default threshold.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in sps_rows(records, capacity=capacity, single_az=single_az):
        if region is not None and r.get("region") != region:
            continue
        groups[(r.get("region", ""), r.get("az_id", ""), r.get("instance_type", ""))].append(r)

    out = []
    for (reg, az_id, itype), rows in groups.items():
        rows.sort(key=lambda r: r["t"])
        scores = [float(r["score"]) for r in rows]
        best_run_s, best_run_n, run_start, run_n = 0.0, 0, None, 0
        for r in rows:
            if float(r["score"]) >= threshold:
                run_start = r["t"] if run_n == 0 else run_start
                run_n += 1
                if r["t"] - run_start >= best_run_s:
                    best_run_s, best_run_n = r["t"] - run_start, run_n
            else:
                run_start, run_n = None, 0
        out.append(
            {
                "region": reg,
                "az_id": az_id,
                "instance_type": itype,
                "best": max(scores),
                "mean": statistics.fmean(scores),
                "good_samples": sum(1 for s in scores if s >= threshold),
                "samples": len(scores),
                # A single good sample is a 0-second window: it proves a moment,
                # not a window you could actually launch a 4-node job into.
                "longest_good_s": best_run_s,
                "longest_good_samples": best_run_n,
            }
        )
    out.sort(key=lambda d: (-d["best"], -d["mean"], d["region"], d["az_id"]))
    return out


def headline(records: list[dict], *, threshold: float, home_region: str) -> dict[str, Any]:
    """The one-paragraph answer, as numbers: at the capacity we actually need
    (8 in one AZ), was anything ever good — at home, or anywhere?"""
    home = availability(records, capacity=8, threshold=threshold, region=home_region)
    world = availability(records, capacity=8, threshold=threshold)
    single = availability(records, capacity=1, threshold=threshold, region=home_region)
    return {
        "home_best": max((p["best"] for p in home), default=None),
        "home_good_pools": sum(1 for p in home if p["good_samples"]),
        "home_longest_good_s": max((p["longest_good_s"] for p in home), default=0.0),
        "world_best": max((p["best"] for p in world), default=None),
        "world_good_pools": sum(1 for p in world if p["good_samples"]),
        "home_cap1_best": max((p["best"] for p in single), default=None),
        "pools": len(home),
        "threshold": threshold,
    }


def rank_pools(
    records: list[dict], *, capacity: int, threshold: float, top: int = 15
) -> list[dict]:
    """(region, AZ, type) pools ranked by *sustained* score — mean over the whole
    window, with P(>=threshold) alongside so a pool that is briefly excellent
    doesn't outrank one that is reliably good. Read it only after the absolute
    table above: a ranking says which pool is least bad, not whether any is good."""
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in sps_rows(records, capacity=capacity, single_az=True):
        if r.get("instance_type") == "any":
            continue
        groups[(r.get("region", ""), r.get("az_id", ""), r.get("instance_type", ""))].append(
            float(r["score"])
        )
    out = [
        {
            "region": k[0],
            "az_id": k[1],
            "instance_type": k[2],
            "mean": statistics.fmean(v),
            "p_good": sum(1 for s in v if s >= threshold) / len(v),
            "samples": len(v),
        }
        for k, v in groups.items()
    ]
    out.sort(key=lambda d: (-d["mean"], -d["p_good"], d["region"]))
    return out[:top]


def az_names(records: list[dict]) -> dict[str, str]:
    """az-id -> az-name, from the daily ``az_map`` records. Placement scores are
    per AZ *id* and prices per AZ *name*; nothing joins without this."""
    return {r["az_id"]: r["az"] for r in _of(records, "az_map") if r.get("az_id")}


def calibration(records: list[dict], *, window_s: float = 900.0) -> list[dict]:
    """What did SPS say at the moment a probe succeeded or failed?

    The probe is ground truth; the score is a prediction. Joining them is the
    only way to learn what a score actually means for *this* account.
    """
    scored = sps_rows(records, capacity=1, single_az=False)
    ids_to_names = az_names(records)
    az_scored = sps_rows(records, capacity=1, single_az=True)
    out = []
    for p in _of(records, "probe"):
        near = [
            r
            for r in scored
            if r.get("region") == p.get("region")
            and r.get("instance_type") == p.get("instance_type")
            and abs(r["t"] - p["t"]) <= window_s
        ]
        az_near = [
            r
            for r in az_scored
            if r.get("region") == p.get("region")
            and r.get("instance_type") == p.get("instance_type")
            and abs(r["t"] - p["t"]) <= window_s
            and (p.get("az") or "?") == ids_to_names.get(r.get("az_id", ""), "")
        ]
        out.append(
            {
                "ts": p.get("ts"),
                "region": p.get("region"),
                "az": p.get("az", ""),
                "instance_type": p.get("instance_type"),
                "capacity_available": bool(p.get("capacity_available")),
                "error_code": p.get("error_code", ""),
                "region_score": max((r["score"] for r in near), default=None),
                "az_score": max((r["score"] for r in az_near), default=None),
            }
        )
    out.sort(key=lambda d: d["ts"] or "")
    return out


def price_summary(records: list[dict], *, top: int = 12) -> list[dict]:
    """Cheapest pools over the window (median of the sampled prices)."""
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in _of(records, "price"):
        groups[(r.get("region", ""), r.get("az", ""), r.get("instance_type", ""))].append(
            float(r["price_usd"])
        )
    out = [
        {
            "region": k[0],
            "az": k[1],
            "instance_type": k[2],
            "median_usd": statistics.median(v),
            "min_usd": min(v),
            "max_usd": max(v),
            "samples": len(v),
        }
        for k, v in groups.items()
    ]
    out.sort(key=lambda d: d["median_usd"])
    return out[:top]


def interruption_summary(records: list[dict]) -> list[dict]:
    """Newest Spot Advisor row per (region, type) — the reclaim-rate context that
    turns "I can get it" into "I can keep it"."""
    newest: dict[tuple[str, str], dict] = {}
    for r in _of(records, "interruption"):
        k = (r.get("region", ""), r.get("instance_type", ""))
        if k not in newest or r["t"] > newest[k]["t"]:
            newest[k] = r
    return sorted(newest.values(), key=lambda d: (d.get("region", ""), d.get("instance_type", "")))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
WIDTH = 100  # every rendered line fits a standard terminal


def _head(title: str) -> str:
    return f"--- {title} ".ljust(WIDTH, "-")[:WIDTH]


def _wrap(text: str, indent: str = "") -> str:
    import textwrap

    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)


def _cell(score: float | None) -> str:
    """One character per cell: 1-9 as digits, 10 as X, no data as '.'."""
    if score is None:
        return "."
    v = int(round(score))
    if v >= 10:
        return "X"
    return str(max(v, 0)) if v > 0 else "0"


def render_heatmap(title: str, grid: dict[str, dict[int, float]], label_width: int = 22) -> str:
    lines = [title, " " * label_width + "".join(f"{h % 10}" for h in range(24)) + "   mean"]
    for name, hours in grid.items():
        row = "".join(_cell(hours.get(h)) for h in range(24))
        mean = statistics.fmean(hours.values()) if hours else 0.0
        lines.append(f"{name[:label_width - 1]:<{label_width}}{row}   {mean:4.1f}")
    lines.append(" " * label_width + "^ hour of day (UTC), score 1-9, X=10, .=no data")
    return "\n".join(lines)


def _pct(v: float | None) -> str:
    return "  —  " if v is None else f"{100 * v:5.1f}%"


def _ticks(n: int) -> str:
    return f"{n} tick" + ("" if n == 1 else "s")


def _num(v: float | None, fmt: str = "4.1f") -> str:
    return "  — " if v is None else format(v, fmt)


def _fmt_dur(seconds: float) -> str:
    """Windows are quoted in the unit that makes them judgeable: a 20-minute
    window is not a place to start an 8-node job."""
    if seconds <= 0:
        return "0 (no 2 in a row)"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def render_headline(records: list[dict], *, threshold: float, home_region: str) -> str:
    """The answer, first, in absolute terms — before any ranking or rate.

    A ranked table of 3-out-of-10 scores reads as success at a glance, so the
    report opens by saying plainly whether anything was EVER good enough to
    launch into, at the capacity we actually need."""
    h = headline(records, threshold=threshold, home_region=home_region)
    lines = [_head(f"HEADLINE — CAN WE GET AN 8-NODE WORLD IN ONE AZ? ({home_region})")]
    if h["home_best"] is None:
        lines.append(_wrap(f"No capacity-8 single-AZ samples for {home_region} in this window."))
        return "\n".join(lines)
    if h["home_good_pools"] == 0:
        lines.append(
            _wrap(
                f"NO. Across {h['pools']} (AZ, GPU) pools in {home_region}, the best spot "
                f"placement score ever seen for 8 instances in one AZ was "
                f"{h['home_best']:.0f}/10 — never the {threshold:g} AWS advises before "
                "attempting a launch. Not once in the whole window."
            )
        )
    else:
        lines.append(
            _wrap(
                f"YES, sometimes: {h['home_good_pools']} of {h['pools']} (AZ, GPU) pools in "
                f"{home_region} reached >= {threshold:g} for 8 instances in one AZ; best "
                f"score {h['home_best']:.0f}/10, longest unbroken good stretch "
                f"{_fmt_dur(h['home_longest_good_s'])}."
            )
        )
    if h["world_best"] is not None:
        anywhere = (
            f"{h['world_good_pools']} pool(s) worldwide cleared the bar"
            if h["world_good_pools"]
            else "no pool in ANY monitored region cleared it either"
        )
        lines.append(
            _wrap(f"Elsewhere: best score anywhere was {h['world_best']:.0f}/10 — {anywhere}.")
        )
    if h["home_cap1_best"] is not None:
        lines.append(
            _wrap(
                f"For comparison, a SINGLE instance in {home_region} peaked at "
                f"{h['home_cap1_best']:.0f}/10. Capacity-1 scores are systematically "
                "optimistic relative to what a training world needs — read them as context, "
                "never as the answer."
            )
        )
    return "\n".join(lines)


def render_availability(
    records: list[dict], *, threshold: float, home_region: str, capacity: int, names: dict
) -> str:
    rows = availability(records, capacity=capacity, threshold=threshold, region=home_region)
    if not rows:
        return ""
    lines = [
        _head(f"ABSOLUTE AVAILABILITY — {home_region}, {capacity} instance(s), single AZ"),
        f"{'az':<16}{'gpu':<16}{'best':>6}{'mean':>7}{'>=thr':>8}{'samples':>9}"
        f"{'longest good window':>24}",
    ]
    for p in rows:
        az = names.get(p["az_id"], p["az_id"])
        lines.append(
            f"{az:<16}{p['instance_type']:<16}{p['best']:>6.0f}{p['mean']:>7.1f}"
            f"{p['good_samples']:>8}{p['samples']:>9}{_fmt_dur(p['longest_good_s']):>24}"
        )
    return "\n".join(lines)


def leaderboard(records: list[dict], *, capacity: int) -> list[dict]:
    """Rows from the all-regions query — a TOP-10 view of the world, never
    coverage (see sps_rows). Useful only to answer "is there capacity ANYWHERE
    right now", which is exactly how the report labels it."""
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in sps_rows(records, capacity=capacity, single_az=True, scope="global_top10"):
        groups[(r.get("region", ""), r.get("az_id", ""), r.get("instance_type", ""))].append(
            float(r["score"])
        )
    out = [
        {
            "region": k[0],
            "az_id": k[1],
            "instance_type": k[2],
            "best": max(v),
            "mean": statistics.fmean(v),
            "samples": len(v),
        }
        for k, v in groups.items()
    ]
    out.sort(key=lambda d: (-d["best"], -d["samples"]))
    return out[:10]


def render_relocation(
    records: list[dict], *, threshold: float, home_region: str, names: dict
) -> str:
    """The separate question: is moving out of the home region worth it?

    Kept apart from the home-region analysis on purpose. The S3 bucket and the
    ~17 GB dataset live in the home region, and a training world has to sit in
    ONE AZ (NCCL bandwidth; cross-AZ traffic is billed), so another region is
    not a free substitution — it means replicating the dataset, paying egress
    and re-homing checkpoints. A score elsewhere has to beat home by enough to
    pay for that, which is a judgement the number alone cannot make.
    """
    rows = [
        p
        for p in availability(records, capacity=8, threshold=threshold)
        if p["region"] != home_region
    ]
    lines = [_head("WOULD RELOCATING HELP? (a different question — data gravity applies)")]
    lines.append(
        _wrap(
            f"The bucket and the ~17 GB dataset are in {home_region}, and all 8 nodes must "
            "share one AZ. Relocating means replicating the dataset and paying cross-region "
            "egress, so treat these as candidates to evaluate, not drop-in alternatives."
        )
    )
    if not rows:
        lines.append("  (no capacity-8 samples outside the home region in this window)")
        return "\n".join(lines)
    good = [p for p in rows if p["good_samples"]]
    lines.append(
        _wrap(
            f"{len(good)} of {len(rows)} non-home pools ever reached >= {threshold:g}."
            if good
            else f"NO non-home pool ever reached >= {threshold:g} either — relocating "
            "would not fix this."
        )
    )
    lines.append(f"{'region':<16}{'az':<16}{'gpu':<16}{'best':>6}{'mean':>7}{'>=thr':>8}")
    for p in sorted(rows, key=lambda p: (-p["best"], -p["mean"]))[:10]:
        az = names.get(p["az_id"], p["az_id"])
        lines.append(
            f"{p['region']:<16}{az:<16}{p['instance_type']:<16}{p['best']:>6.0f}"
            f"{p['mean']:>7.1f}{p['good_samples']:>8}"
        )
    return "\n".join(lines)


def render_report(
    records: list[dict],
    *,
    threshold: float,
    home_region: str,
    home_type: str,
    since_hours: float,
) -> str:
    lines: list[str] = []
    a = lines.append
    names = az_names(records)
    cov = coverage(records)

    a("=" * WIDTH)
    a(f"SPOTWATCH — GPU spot availability, last {since_hours:g}h")
    a(
        _wrap(
            f"home: {home_type} in {home_region} · 'good' = spot placement score "
            f">= {threshold:g} (1-10; AWS advises >= 7 before attempting a launch)"
        )
    )
    a("=" * WIDTH)
    a(
        f"{cov['ticks']} ticks over {cov['span_hours']:.1f}h · {len(records)} records · "
        f"{cov['requested']} placement-score requests"
    )
    # Coverage before conclusions: a gap is a hole in the series, not a low score.
    a(
        _wrap(
            f"{cov['gaps']} request(s) returned nothing ({_pct(cov['gap_pct']).strip()}), "
            f"{cov['throttled']} of them throttled — those are GAPS, not zeros"
            if cov["gaps"]
            else "no gaps: every placement-score request in the window returned data"
        )
    )
    probes = _of(records, "probe")
    ok = sum(1 for p in probes if p.get("capacity_available"))
    a(
        f"truth probes: {ok}/{len(probes)} acquired capacity · "
        f"{len(_of(records, 'probe_skipped'))} skipped (rate limit / training / no default VPC)"
    )
    a("")

    a(render_headline(records, threshold=threshold, home_region=home_region))
    a("")

    # The decision query first (8 in one AZ); capacity 1 follows as context.
    for capacity in (8, 1):
        block = render_availability(
            records, threshold=threshold, home_region=home_region, capacity=capacity, names=names
        )
        if block:
            a(block)
            a("")

    for capacity in (8, 1):
        rows = scenarios(
            records,
            threshold=threshold,
            home_region=home_region,
            home_type=home_type,
            capacity=capacity,
        )
        if not any(r["ticks"] for r in rows):
            continue
        label = "THE DECISION QUERY" if capacity == 8 else "context only — optimistic"
        a(_head(f"IF WE SWITCH SOMETHING — {capacity} instance(s) ({label})"))
        a(f"{'scenario':<52}{'P(good)':>9}{'mean best':>11}{'peak':>6}{'ticks':>7}")
        for r in rows:
            a(
                f"{r['scenario']:<52}{_pct(r['p_good']):>9}{_num(r['mean_best']):>11}"
                f"{_num(r['max_best'], '4.0f'):>6}{r['ticks']:>7}"
            )
        a(
            _wrap(
                "* region-level score: AWS may satisfy it by spreading instances across AZs. "
                "A training world has to sit in ONE AZ, so those rows are an upper bound, not "
                "a plan — compare them with the single-AZ rows below them."
            )
        )
        a("")

    # Scenario 1, resolved: when is the home pool actually available?
    hours = hourly_odds(
        records, threshold=threshold, home_region=home_region, home_type=home_type, capacity=8
    )
    if hours:
        a(_head("WAIT FOR A GOOD TIME — best hours for the home pool (8 in one AZ)"))
        if not any((o["p_good"] or 0) > 0 for o in hours.values()):
            # Say it once, plainly: otherwise this reads as a list of options.
            a(_wrap("NO hour of day cleared the bar — waiting does not help in this window."))
        for h, o in sorted(hours.items(), key=lambda kv: -(kv[1]["p_good"] or 0))[:6]:
            a(
                f"  {h:02d}:00 UTC   P(good)={_pct(o['p_good'])}  mean={_num(o['mean_best'])}  "
                f"({_ticks(o['ticks'])})"
            )
        dow = dow_hour_odds(
            records, threshold=threshold, home_region=home_region, home_type=home_type, capacity=8
        )
        top_dow = sorted(dow.items(), key=lambda kv: (-(kv[1]["p_good"] or 0), -kv[1]["ticks"]))[:6]
        if top_dow:
            a("  best (weekday, hour) buckets — thin samples, read as a hint:")
            for (day, h), o in top_dow:
                a(f"    {day} {h:02d}:00   P(good)={_pct(o['p_good'])}  ({_ticks(o['ticks'])})")
        a("")

    az_rows = [
        r
        for r in sps_rows(records, capacity=8, single_az=True)
        if r.get("region") == home_region and r.get("instance_type") == home_type
    ]
    for r in az_rows:
        r["az_label"] = f"{names.get(r.get('az_id', ''), r.get('az_id', ''))}"
    if az_rows:
        a(
            render_heatmap(
                _head(f"AVAILABILITY BY AZ x HOUR ({home_type}, {home_region}, 8 in one AZ)"),
                heatmap(az_rows, "az_label"),
            )
        )
        a("")

    type_rows = [
        r for r in sps_rows(records, capacity=8, single_az=True) if r.get("region") == home_region
    ]
    if type_rows:
        a(
            render_heatmap(
                _head(f"AVAILABILITY BY GPU x HOUR ({home_region}, best AZ, 8 in one AZ)"),
                heatmap(type_rows, "instance_type"),
            )
        )
        a("")

    a(render_relocation(records, threshold=threshold, home_region=home_region, names=names))
    a("")

    board = leaderboard(records, capacity=8)
    if board:
        a(_head("GLOBAL TOP-10 LEADERBOARD (truncated by AWS — NOT coverage)"))
        a(
            _wrap(
                "One all-regions query per type: AWS returns only its 10 best rows, so an AZ "
                "missing here means 'not in the top 10', never 'no capacity'. Every rate and "
                "heatmap above deliberately ignores these rows."
            )
        )
        a(f"{'region':<16}{'az':<16}{'gpu':<16}{'best':>6}{'appearances':>13}")
        for p in board:
            az = names.get(p["az_id"], p["az_id"])
            a(
                f"{p['region']:<16}{az:<16}{p['instance_type']:<16}{p['best']:>6.0f}"
                f"{p['samples']:>13}"
            )
        a("")

    cal = calibration(records)
    if cal:
        a(_head("SPS vs TRUTH PROBE (did the score predict the launch?)"))
        a(
            f"{'when (UTC)':<22}{'region':<12}{'az':<14}{'gpu':<14}{'got it?':<9}"
            f"{'region SPS':>11}{'az SPS':>8}"
        )
        for c in cal:
            got = "YES" if c["capacity_available"] else f"no({c['error_code'] or '?'})"
            a(
                f"{(c['ts'] or ''):<22}{c['region']:<12}{(c['az'] or '-'):<14}"
                f"{c['instance_type']:<14}{got:<9}{_num(c['region_score'], '11.0f')}"
                f"{_num(c['az_score'], '8.0f')}"
            )
        by_band: dict[str, list[bool]] = defaultdict(list)
        for c in cal:
            s = c["region_score"]
            band = "no score" if s is None else f"score {int(s)}"
            by_band[band].append(c["capacity_available"])
        a(
            "  success rate by score: "
            + " · ".join(f"{b}: {sum(v)}/{len(v)}" for b, v in sorted(by_band.items()))
        )
        a("")

    prices = price_summary(records)
    if prices:
        a(_head("CHEAPEST POOLS (spot $/hr over the window)"))
        a(f"{'region':<16}{'az':<16}{'gpu':<16}{'median':>9}{'min':>9}{'max':>9}")
        for p in prices:
            a(
                f"{p['region']:<16}{p['az']:<16}{p['instance_type']:<16}"
                f"{p['median_usd']:>9.3f}{p['min_usd']:>9.3f}{p['max_usd']:>9.3f}"
            )
        a("")

    interrupts = [r for r in interruption_summary(records) if r.get("region") == home_region]
    if interrupts:
        a(_head(f"INTERRUPTION RATE + SAVINGS ({home_region}, AWS Spot Advisor)"))
        a(f"{'gpu':<16}{'interruption':<16}{'savings vs on-demand':>22}")
        for r in interrupts:
            a(
                f"{r['instance_type']:<16}{r.get('interruption_range', '?'):<16}"
                f"{str(r.get('savings_pct', '?')) + '%':>22}"
            )
        a("")

    a(verdict(records, threshold=threshold, home_region=home_region, home_type=home_type))
    return "\n".join(lines)


def verdict(records: list[dict], *, threshold: float, home_region: str, home_type: str) -> str:
    """The closing paragraph: the cheapest strategy that clears the bar at the
    capacity we actually need, or a plain statement that none does."""
    rows = scenarios(
        records, threshold=threshold, home_region=home_region, home_type=home_type, capacity=8
    )
    if not any(r["ticks"] for r in rows):  # no capacity-8 data in window: fall back
        rows = scenarios(
            records, threshold=threshold, home_region=home_region, home_type=home_type, capacity=1
        )
    named = [r for r in rows if not r["scenario"].startswith("  [")]
    lines = [_head("VERDICT")]
    winner = next((r for r in named if (r["p_good"] or 0) >= 0.8), None)
    if winner is None:
        winner = next((r for r in named if (r["p_good"] or 0) >= 0.5), None)
        if winner is None:
            best_case = max((r["p_good"] or 0) for r in named) if named else 0
            lines.append(
                _wrap(
                    "NEVER ENOUGH in this window: even the widest strategy (any GPU, any AZ, "
                    f"any region) scored >= {threshold:g} in only {100 * best_case:.0f}% of "
                    "ticks. The drought is the finding — record it and re-run the window later."
                )
            )
            lines.append(
                _wrap(
                    "Options: run the on-demand baseline now and keep spot opportunistic, "
                    "shrink the world (fewer nodes, smaller GPU), or gate the launcher on a "
                    "live placement score so it fires the moment capacity appears."
                )
            )
            return "\n".join(lines)
        lines.append(
            _wrap(
                f"MARGINAL: '{winner['scenario']}' is good in {_pct(winner['p_good']).strip()} of "
                "ticks — workable with a launcher that waits and retries."
            )
        )
    else:
        lines.append(
            _wrap(
                f"'{winner['scenario']}' is good in {_pct(winner['p_good']).strip()} of ticks "
                "— that is the cheapest strategy (of the five) that clears the bar."
            )
        )
    if winner["scenario"].startswith("switch region"):
        # Relocation is the one "switch" that isn't free: see render_relocation.
        lines.append(
            _wrap(
                f"Note: that answer moves us out of {home_region}, where the bucket and the "
                "~17 GB dataset live. Price the dataset copy and the cross-region egress "
                "before treating it as the plan."
            )
        )
    if winner["scenario"].endswith("*"):
        lines.append(
            _wrap(
                "Note: that row is a region-level score (instances may be spread across AZs). "
                "A single-AZ training world may still be unattainable — check the single-AZ "
                "rows before committing."
            )
        )
    hours = hourly_odds(
        records, threshold=threshold, home_region=home_region, home_type=home_type, capacity=8
    )
    best = max(hours.items(), key=lambda kv: (kv[1]["p_good"] or 0), default=None)
    # A "best hour" of 0% is noise dressed as advice — only print it if waiting
    # for that hour would actually change the outcome.
    if best and (best[1]["p_good"] or 0) > 0:
        average = statistics.fmean([o["p_good"] or 0 for o in hours.values()])
        lines.append(
            _wrap(
                f"Best hour to launch: {best[0]:02d}:00 UTC "
                f"(P(good)={_pct(best[1]['p_good']).strip()} vs {_pct(average).strip()} average)."
            )
        )
    return "\n".join(lines)


def render_png(
    records: list[dict], path: str, *, home_region: str, home_type: str, capacity: int = 8
) -> bool:
    """AZ x hour heatmap as a PNG (same optional-matplotlib pattern as
    compare/profile: absent matplotlib just means text-only).

    Defaults to capacity 8 — the whole-world query — and the colour scale is
    fixed at 0-10 with 7 marked, so a chart of 3s cannot look like success just
    because 3 is the best value present."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows = [
        r
        for r in sps_rows(records, capacity=capacity, single_az=True)
        if r.get("region") == home_region and r.get("instance_type") == home_type
    ]
    if not rows:
        return False
    names = az_names(records)
    for r in rows:
        r["az_label"] = names.get(r.get("az_id", ""), r.get("az_id", ""))
    grid = heatmap(rows, "az_label")
    labels = list(grid)
    data = [[grid[label].get(h, float("nan")) for h in range(24)] for label in labels]

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(labels) + 2.2))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=10, cmap="RdYlGn")
    ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)], fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("hour of day (UTC)")
    ax.set_title(
        f"Spot placement score — {capacity}x {home_type} in one AZ, {home_region} "
        "(10 = very likely)"
    )
    bar = fig.colorbar(im, ax=ax, shrink=0.8)
    # The launch bar, drawn on the scale: without it a wall of 3s in the middle
    # of a red-yellow-green ramp still reads as "amber, so probably fine".
    bar.ax.axhline(7, color="black", linewidth=1.5)
    bar.ax.set_ylabel("7 = AWS's launch threshold", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def report(
    cfg: OrchestratorConfig,
    *,
    since_hours: float = 72.0,
    threshold: float = 7.0,
    home_region: str = "",
    home_type: str = "",
) -> str:
    records = load_records(cfg, since_hours)
    home_region = home_region or cfg.region
    home_type = home_type or cfg.instance_type
    text = render_report(
        records,
        threshold=threshold,
        home_region=home_region,
        home_type=home_type,
        since_hours=since_hours,
    )
    print(text)
    out_dir = os.path.join("reports", f"spotwatch-{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(text + "\n")
    if render_png(
        records,
        os.path.join(out_dir, "availability.png"),
        home_region=home_region,
        home_type=home_type,
    ):
        print(f"\n[spotwatch] wrote {out_dir}/report.txt + availability.png", file=sys.stderr)
    else:
        print(
            f"\n[spotwatch] wrote {out_dir}/report.txt " "(no PNG — matplotlib missing or no data)",
            file=sys.stderr,
        )
    return text
