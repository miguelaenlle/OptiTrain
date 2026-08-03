"""Spotwatch collector — the code that runs IN AWS on a 10-minute schedule.

It answers one question, unattended, over days: **to what extent can we actually
acquire GPU spot capacity?** Every tick appends a JSONL shard to
``s3://<bucket>/spotwatch/<YYYY-MM-DD>/<epoch>-<id>.jsonl``:

  * ``sps``          — Spot placement scores (1-10) for a FIXED request matrix
  * ``sps_gap``      — a request that failed/threw after retries: a HOLE in the
                       series, recorded so it can never be read as a low score
  * ``price``        — newest spot $/hr per (AZ, type) pool
  * ``offering`` / ``az_map`` / ``interruption`` — daily context (pools that
                       exist, AZ-name↔AZ-id join key, Spot Advisor reclaim rates)
  * ``probe``        — a real 1-instance spot launch: the ground truth that
                       calibrates the scores
  * ``tick``         — self-report of the tick (counts, errors, duration)

Three measured facts about GetSpotPlacementScores drive the request matrix:

  1. **There is a hard cap on DISTINCT CONFIGURATIONS per rolling 24h, and it is
     the binding constraint.** It is not published and does not appear in
     service-quotas; it surfaces only as ``MaxConfigLimitExceeded``. Measured on
     this account: 28 distinct configurations succeeded and every further one
     failed — a 456-request matrix collected 9% of what it asked for, and *which*
     9% was decided by request ordering. Re-querying a configuration already
     spent costs nothing, so the budget is spent on the FIRST use of each
     configuration and a tick can poll them forever. Hence ``MAX_CONFIGURATIONS``
     below, a pinned priority-ordered matrix well under the observed cap, and a
     hard assert that the generated matrix fits. **Changing the matrix
     mid-experiment spends NEW configurations against the same daily cap** — so
     churn is expensive and the matrix should be settled once, then left alone.
  2. **Every response hard-caps at 10 scored rows.** MaxResults is ignored above
     10 and NextToken does not extend it. So an all-regions query is a global
     *top-10 leaderboard*, not a per-AZ series: a 17-region single-AZ query for
     g5.xlarge came back with 10 rows covering 7 regions and silently dropped
     ~70 AZs, and a dropped AZ is indistinguishable from "the score fell" —
     which destroys the only question the study exists to answer ("does
     use1-az4 recover at 3 AM Saturday?"). Only a SINGLE-region request returns
     that region's complete AZ set. Given (1) we cannot afford one request per
     region, so per-AZ coverage is bought where it matters (the home region and
     two relocation candidates) and the all-regions query is kept purely as a
     cheap leaderboard, tagged ``scope="global_top10"`` and excluded from every
     per-AZ analysis.
  3. **Throttling is data, not failure.** The token bucket is real too (bucket
     100, refill 20/s), so requests are paced at 10/s; a throttled request is
     retried with backoff and, if it still fails — or if it is refused by the
     configuration cap — it is written as an ``sps_gap`` record. A gap in a time
     series must never look like a bad score.

And one rule that outranks all of them:

  * **Never interfere with a training run.** The truth probe is the only part
    that spends money or takes capacity, so it is rate-limited to one per 6h
    AND skipped outright if any instance tagged ``project=spot-train`` is
    pending/running in the probe region — a probe must never lose a race for
    the last g5 against the thing we are actually trying to run.

This file is uploaded to Lambda as a single-file zip. Unlike the rest of
``src/orchestrator`` it therefore imports boto3 directly instead of going
through ``aws.py``: it must be self-contained (zero third-party deps — boto3
ships in the Lambda runtime) and it never executes on your laptop. Every
credentialed call made from your machine still lives in ``aws.py``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, NamedTuple

import boto3

# --------------------------------------------------------------------------- #
# The FIXED request matrix (see the module docstring for the measured facts)
# --------------------------------------------------------------------------- #
# The GPU types this project can actually train on. Ordered, never sorted at
# call time, never derived from anything observed at runtime.
INSTANCE_TYPES: tuple[str, ...] = (
    "g5.xlarge",
    "g5.2xlarge",
    "g4dn.xlarge",
    "g4dn.2xlarge",
    "g6.xlarge",
    "g6.2xlarge",
)
# Types worth a configuration of their own. The .2xlarge variants are dropped
# from the per-type expansion — they score 1 consistently and are not what we
# would train on — but they stay inside the basket, where they cost nothing
# extra and still count towards "any GPU I could use".
PER_TYPE_TYPES: tuple[str, ...] = ("g5.xlarge", "g4dn.xlarge", "g6.xlarge")
# 8 is THE decision query for this project: one training world, one AZ (NCCL
# bandwidth + cross-AZ transfer charges make a split world pointless). 1 is the
# optimistic lower bound, kept only as context — a good score at 1 says nothing
# about getting eight at once.
TARGET_CAPACITIES: tuple[int, ...] = (1, 8)
# AWS warns that a score for one or two instance types is always pessimistic
# ("specify at least three"). We still want per-type resolution — it is the only
# way to answer "would switching GPU type help?" — so we ask BOTH: one request
# per type (comparable to each other, conservative in absolute terms) plus one
# "basket" request naming all six (the realistic "I'll take whatever runs" ask).
BASKET_LABEL = "any"
# Where we would go if leaving the home region were worth it: same continent, so
# the dataset copy and the egress bill are as small as relocation gets. Basket
# only, capacity 8 — enough to answer "is relocating even on the table?".
CANDIDATE_REGIONS: tuple[str, ...] = ("us-east-2", "us-west-2")
# The self-imposed configuration budget. The real cap is unpublished and may
# vary by account; 28 was observed on ours, so we sit well under it and leave
# room for a future addition without a redesign. sps_requests() asserts against
# this, and the report surfaces the EFFECTIVE budget (configurations that
# actually succeeded) so an account with a smaller cap shows up as data.
MAX_CONFIGURATIONS = 20
# Every response is truncated to this many rows regardless of MaxResults/
# NextToken (measured). Per-region requests stay under it by construction; a
# response that hits it exactly is flagged ``truncated`` so the report knows the
# view is partial.
SPS_MAX_ROWS = 10
# Requests per second we allow ourselves. The measured bucket holds 100 tokens
# and refills at 20/s, so pacing at half the refill rate makes draining it
# impossible no matter how large the matrix grows.
SPS_RATE_PER_S = 10.0
# Error codes that mean "slow down" rather than "this request is wrong".
THROTTLE_CODES = (
    "RequestLimitExceeded",
    "Throttling",
    "ThrottlingException",
    "TooManyRequestsException",
)
# "You have already used your distinct-configuration budget for the last 24h."
# NOT retried: unlike a throttle it will not clear within a tick, and retrying
# would only burn time. It is recorded, because an unpublished limit can only be
# observed through its own error.
CONFIG_CAP_CODES = ("MaxConfigLimitExceeded",)


class SpsRequest(NamedTuple):
    """One GetSpotPlacementScores call plus the labels its records carry.

    ``scope`` is the important one: ``region`` means "complete AZ coverage for
    this region" (safe to build a per-AZ time series from), ``global_top10``
    means "the 10 best rows AWS chose to show us, worldwide" (a leaderboard —
    absence of an AZ proves nothing).
    """

    kwargs: dict[str, Any]
    label: str
    scope: str


# Regions used if the one-time enumeration fails. Deliberately a literal: a
# fallback that queried AWS would make the matrix vary with the failure.
FALLBACK_REGIONS: tuple[str, ...] = (
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-central-1",
    "eu-west-1",
    "ap-northeast-1",
    "ap-southeast-1",
)

SPOT_ADVISOR_URL = "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"

# Tags. The probe is tagged with a project of its own so no sweep, quota check
# or human ever confuses it with a training box; TRAINING_TAG is what training
# boxes carry (aws.launch stamps project=spot-train on every instance).
PROBE_TAG_KEY = "Purpose"
PROBE_TAG_VALUE = "spotwatch-probe"
TRAINING_TAG_KEY = "project"
TRAINING_TAG_VALUE = "spot-train"

_clients: dict[tuple[str, str], Any] = {}


def _client(service: str, region: str):
    """Cached boto3 client. Lambda reuses the container between ticks, so this
    also reuses the TLS connections — most ticks make ~50 API calls."""
    key = (service, region)
    if key not in _clients:
        _clients[key] = boto3.client(service, region_name=region)
    return _clients[key]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no AWS, no clock of their own)
# --------------------------------------------------------------------------- #
def _req(types: list[str], capacity: int, regions: list[str], scope: str) -> SpsRequest:
    """One configuration. SingleAvailabilityZone is always true: a training world
    has to land in ONE AZ (NCCL bandwidth, cross-AZ transfer charges), so a
    region-level score answers a question we cannot use — and at ~20 available
    configurations, we cannot afford to ask it."""
    return SpsRequest(
        kwargs={
            "InstanceTypes": list(types),
            "TargetCapacity": capacity,
            "TargetCapacityUnitType": "units",
            "SingleAvailabilityZone": True,
            "RegionNames": list(regions),
        },
        label=types[0] if len(types) == 1 else BASKET_LABEL,
        scope=scope,
    )


def sps_requests(regions: list[str], home_region: str = "us-east-1") -> list[SpsRequest]:
    """The complete, deterministic request matrix, in PRIORITY ORDER.

    Same inputs => byte-identical requests in the same order, forever (the
    leaderboard's region list is sorted so an unordered pin can't change it).

    The budget, not the API's speed, is what shapes this. ~20 configurations is
    all we get per 24h, so they are spent where they answer the actual question,
    highest value first — if a configuration is ever refused, we lose the tail
    (the worldwide leaderboard), never the head (us-east-1 at capacity 8):

      1. home region, per AZ, capacity 8 — basket + 3 trainable types (4)
         The decision query: can we get a whole 8-node world in one AZ, where
         the bucket and the 17 GB dataset already live?
      2. home region, per AZ, capacity 1 — same four asks (4)
         Context: how much of the drought is about *size* rather than supply.
      3. worldwide leaderboard, capacity 8 — same four asks (4)
         "Is there capacity anywhere?" for 4 configurations instead of 4 per
         region; its 10 rows are a top-10, never coverage.
      4. relocation candidates, per AZ, capacity 8, basket only (2)

    14 configurations, 14 requests per tick — seconds of wall clock, and it
    re-polls the SAME configurations forever, which is free once they are spent.
    """
    board_regions = sorted(regions)
    out: list[SpsRequest] = []
    seen: set[tuple] = set()

    def _add(types: list[str], capacity: int, request_regions: list[str], scope: str) -> None:
        # First use of a configuration is what costs budget, so an accidental
        # repeat is pure waste. It happens for real: on a single-region account
        # the "worldwide" leaderboard IS the home-region ask.
        req = _req(types, capacity, request_regions, scope)
        key = config_key(req)
        if key in seen:
            return
        seen.add(key)
        out.append(req)

    for capacity in (8, 1):  # 8 first: the query that decides the project
        _add(list(INSTANCE_TYPES), capacity, [home_region], "region")
        for instance_type in PER_TYPE_TYPES:
            _add([instance_type], capacity, [home_region], "region")
    _add(list(INSTANCE_TYPES), 8, board_regions, "global_top10")
    for instance_type in PER_TYPE_TYPES:
        _add([instance_type], 8, board_regions, "global_top10")
    for region in CANDIDATE_REGIONS:
        if region != home_region:
            _add(list(INSTANCE_TYPES), 8, [region], "region")

    # Self-enforcing budget: the matrix cannot grow past the cap by accident,
    # and anyone adding to it has to consciously decide what to spend it on.
    assert len(seen) <= MAX_CONFIGURATIONS, (
        f"matrix asks for {len(seen)} distinct configurations but the budget is "
        f"{MAX_CONFIGURATIONS}; AWS refuses the excess with MaxConfigLimitExceeded"
    )
    return out


def config_key(req: SpsRequest) -> tuple:
    """What AWS counts against the 24h budget: the request shape itself. Used to
    police the matrix here and, in the report, to measure the EFFECTIVE budget
    from what actually came back."""
    k = req.kwargs
    return (
        tuple(k["InstanceTypes"]),
        k["TargetCapacity"],
        k["TargetCapacityUnitType"],
        k["SingleAvailabilityZone"],
        tuple(k["RegionNames"]),
    )


def is_daily_tick(now: datetime, daily_hour: int = 3, tick_minutes: int = 10) -> bool:
    """True for the first tick of ``daily_hour`` UTC only.

    The pool enumeration and the Spot Advisor fetch are slow and change on the
    order of days, so they ride one tick out of 144. Gating on the clock (rather
    than on a marker object) keeps the tick stateless; EventBridge's at-least-
    once delivery can double-fire, which at worst writes the daily records
    twice — cheap, and the report de-duplicates by taking the newest.
    """
    return now.hour == daily_hour and now.minute < tick_minutes


def should_probe(
    *,
    now_t: float,
    last_probe_t: float | None,
    training_instances: list[str],
    enabled: bool = True,
    min_interval_s: float = 6 * 3600,
) -> tuple[bool, str]:
    """(do_it, reason). The rate limit and the non-interference rule in one
    place so both are testable without touching EC2.

    Non-interference is absolute: if ANY box tagged project=spot-train is
    pending/running in the probe region, we do not compete with it for capacity,
    however long it has been since the last probe.
    """
    if not enabled:
        return False, "disabled"
    if training_instances:
        return False, f"training instances present ({','.join(sorted(training_instances))})"
    if last_probe_t is not None and now_t - last_probe_t < min_interval_s:
        wait = min_interval_s - (now_t - last_probe_t)
        return False, f"rate-limited ({wait / 60:.0f} min to go)"
    return True, "ok"


def shard_key(prefix: str, now: datetime, tick_id: str) -> str:
    """Every tick writes its OWN key — never read-modify-write. Two invocations
    can share an epoch second (at-least-once delivery), hence the tick suffix:
    a collision would silently drop a tick's data."""
    return f"{prefix}/{now:%Y-%m-%d}/{int(now.timestamp())}-{tick_id}.jsonl"


def make_record(kind: str, tick_id: str, t: float, **fields: Any) -> dict[str, Any]:
    """Every record carries when it was taken, which tick it belongs to, and its
    type — the report groups by tick to ask "at this moment, was ANY pool good?"."""
    return {
        "type": kind,
        "tick_id": tick_id,
        "ts": datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "t": round(t, 3),
        **fields,
    }


def advisor_records(doc: dict, tick_id: str, t: float) -> list[dict[str, Any]]:
    """Flatten the public Spot Advisor JSON into one record per (region, type).

    ``r`` is an index into ``ranges`` (interruption frequency bucket), ``s`` is
    the savings percent vs on-demand — the two numbers that turn a placement
    score into an expected cost."""
    labels = {rng["index"]: rng.get("label", "") for rng in doc.get("ranges", [])}
    out: list[dict[str, Any]] = []
    for region, by_os in (doc.get("spot_advisor") or {}).items():
        for instance_type, vals in (by_os.get("Linux") or {}).items():
            if instance_type not in INSTANCE_TYPES:
                continue
            out.append(
                make_record(
                    "interruption",
                    tick_id,
                    t,
                    region=region,
                    instance_type=instance_type,
                    interruption_range=labels.get(vals.get("r"), ""),
                    interruption_index=vals.get("r"),
                    savings_pct=vals.get("s"),
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _config() -> dict[str, Any]:
    home = os.environ.get("AWS_REGION", "us-east-1")
    return {
        "bucket": os.environ["SPOTWATCH_BUCKET"],
        "prefix": os.environ.get("SPOTWATCH_PREFIX", "spotwatch"),
        "home_region": home,
        "probe_region": os.environ.get("SPOTWATCH_PROBE_REGION", home),
        "probe_type": os.environ.get("SPOTWATCH_PROBE_TYPE", "g5.xlarge"),
        "probe_enabled": os.environ.get("SPOTWATCH_PROBE_ENABLED", "1") not in ("0", "false", ""),
        "probe_min_hours": float(os.environ.get("SPOTWATCH_PROBE_MIN_HOURS", "6")),
        "probe_wait_s": float(os.environ.get("SPOTWATCH_PROBE_WAIT_SECONDS", "45")),
        "daily_hour": int(os.environ.get("SPOTWATCH_DAILY_HOUR", "3")),
    }


# --------------------------------------------------------------------------- #
# S3 helpers (small control docs; the shards themselves are write-only)
# --------------------------------------------------------------------------- #
def _get_json(bucket: str, key: str) -> dict | None:
    try:
        body = _client("s3", os.environ.get("AWS_REGION", "us-east-1")).get_object(
            Bucket=bucket, Key=key
        )["Body"]
        return json.loads(body.read().decode())
    except Exception:  # noqa: BLE001 — absent/unreadable control doc == not set yet
        return None


def _put_json(bucket: str, key: str, doc: dict) -> None:
    _client("s3", os.environ.get("AWS_REGION", "us-east-1")).put_object(
        Bucket=bucket, Key=key, Body=json.dumps(doc, indent=2).encode()
    )


# --------------------------------------------------------------------------- #
# (a) Spot placement scores
# --------------------------------------------------------------------------- #
def _pinned_regions(cfg: dict) -> tuple[list[str], list[dict]]:
    """The region list the SPS matrix is built from, enumerated ONCE.

    Written to S3 on first use and never rewritten: the region list defines the
    matrix, and a matrix that grows a region mid-experiment makes "score dropped"
    and "we started asking about somewhere new" impossible to tell apart in the
    series. Region drift shows up in the daily ``offering`` records instead,
    where it is data rather than a behaviour change.
    """
    key = f"{cfg['prefix']}/meta/regions.json"
    doc = _get_json(cfg["bucket"], key)
    if doc and doc.get("regions"):
        return list(doc["regions"]), []

    errors: list[dict] = []
    regions: list[str] = []
    try:
        ec2 = _client("ec2", cfg["home_region"])
        all_regions = [r["RegionName"] for r in ec2.describe_regions()["Regions"]]
    except Exception as e:  # noqa: BLE001 — fall back to the literal list
        errors.append({"stage": "describe_regions", "error": str(e)})
        all_regions = []
    for region in sorted(all_regions):
        try:
            r = _client("ec2", region).describe_instance_type_offerings(
                LocationType="region",
                Filters=[{"Name": "instance-type", "Values": list(INSTANCE_TYPES)}],
            )
            if r.get("InstanceTypeOfferings"):
                regions.append(region)
        except Exception as e:  # noqa: BLE001 — opt-in region, no creds, throttle
            errors.append({"stage": f"offerings:{region}", "error": str(e)})
    if not regions:
        regions = list(FALLBACK_REGIONS)
        errors.append({"stage": "enumerate", "error": "no regions found; using fallback list"})
    regions = sorted(regions)
    try:
        _put_json(
            cfg["bucket"],
            key,
            {
                "regions": regions,
                "instance_types": list(INSTANCE_TYPES),
                "pinned_at": datetime.now(timezone.utc).isoformat(),
                "note": "pinned once — changing this changes the SPS request configuration",
            },
        )
    except Exception as e:  # noqa: BLE001 — worst case we re-enumerate next tick
        errors.append({"stage": "pin_regions", "error": str(e)})
    return regions, errors


def error_code(exc: BaseException) -> str:
    """botocore's error code, or the exception class for anything else."""
    return getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)


def is_throttle(code: str) -> bool:
    return code in THROTTLE_CODES


def is_config_capped(code: str) -> bool:
    return code in CONFIG_CAP_CODES


_last_sps_call = 0.0


def _pace(rate_per_s: float = SPS_RATE_PER_S) -> None:
    """Keep at most ``rate_per_s`` requests per second across the whole tick.
    At half the measured refill rate the token bucket can never drain, so a
    bigger matrix costs time and nothing else."""
    global _last_sps_call
    gap = (1.0 / rate_per_s) - (time.time() - _last_sps_call)
    if gap > 0:
        time.sleep(gap)
    _last_sps_call = time.time()


def _sps_call(ec2, kwargs: dict, attempts: int = 4) -> tuple[dict, int]:
    """One scored request, retrying only on throttles. Returns (response,
    retries) — the retry count rides in the tick record so a tick that had to
    fight the bucket is visible before the data looks wrong."""
    retries = 0
    while True:
        _pace()
        try:
            return ec2.get_spot_placement_scores(**kwargs), retries
        except Exception as e:  # noqa: BLE001 — re-raised below unless throttled
            if not is_throttle(error_code(e)) or retries >= attempts - 1:
                raise
            time.sleep(2**retries)
            retries += 1


def _collect_sps(cfg: dict, regions: list[str], tick_id: str, t: float) -> tuple[list, list]:
    ec2 = _client("ec2", cfg["home_region"])
    records: list[dict] = []
    errors: list[dict] = []
    for req in sps_requests(regions, cfg["home_region"]):
        asked_region = req.kwargs["RegionNames"][0] if req.scope == "region" else ""
        try:
            resp, retries = _sps_call(ec2, req.kwargs)
            rows = resp.get("SpotPlacementScores", [])
            # NextToken is deliberately NOT followed: responses hard-cap at 10
            # rows either way, so paging would only re-serve the same top-10.
            for s in rows:
                records.append(
                    make_record(
                        "sps",
                        tick_id,
                        t,
                        region=s.get("Region", ""),
                        az_id=s.get("AvailabilityZoneId", ""),
                        instance_type=req.label,
                        capacity=req.kwargs["TargetCapacity"],
                        single_az=req.kwargs["SingleAvailabilityZone"],
                        score=s.get("Score"),
                        scope=req.scope,
                        truncated=len(rows) >= SPS_MAX_ROWS,
                        retries=retries,
                    )
                )
        except Exception as e:  # noqa: BLE001 — one bad request must not sink the tick
            code = error_code(e)
            # A HOLE, written down as such. Without this record the report
            # cannot tell "AWS refused to answer" from "the score was low".
            records.append(
                make_record(
                    "sps_gap",
                    tick_id,
                    t,
                    region=asked_region,
                    instance_type=req.label,
                    capacity=req.kwargs["TargetCapacity"],
                    single_az=req.kwargs["SingleAvailabilityZone"],
                    scope=req.scope,
                    error_code=code,
                    throttled=is_throttle(code),
                    # The unpublished 24h configuration cap. Counted separately
                    # from throttles because the fix is different: a throttle
                    # means wait, this means the matrix is too wide.
                    config_capped=is_config_capped(code),
                )
            )
            errors.append({"stage": "sps", "region": asked_region, "error": str(e)[:200]})
    return records, errors


# --------------------------------------------------------------------------- #
# (b) Spot prices
# --------------------------------------------------------------------------- #
def _collect_prices(regions: list[str], tick_id: str, t: float) -> tuple[list, list]:
    """Newest price point per pool. StartTime=now makes DescribeSpotPriceHistory
    return the current price for each (AZ, type) rather than a time series."""
    records: list[dict] = []
    errors: list[dict] = []
    now = datetime.now(timezone.utc)
    for region in regions:
        try:
            seen: set[tuple[str, str]] = set()
            paginator = _client("ec2", region).get_paginator("describe_spot_price_history")
            for page in paginator.paginate(
                InstanceTypes=list(INSTANCE_TYPES),
                ProductDescriptions=["Linux/UNIX"],
                StartTime=now,
            ):
                for p in page.get("SpotPriceHistory", []):
                    pool = (p.get("AvailabilityZone", ""), p.get("InstanceType", ""))
                    if pool in seen:  # newest first; later pages are older points
                        continue
                    seen.add(pool)
                    records.append(
                        make_record(
                            "price",
                            tick_id,
                            t,
                            region=region,
                            az=pool[0],
                            instance_type=pool[1],
                            price_usd=float(p["SpotPrice"]),
                            quoted_at=p["Timestamp"].isoformat(),
                        )
                    )
        except Exception as e:  # noqa: BLE001 — per-region isolation
            errors.append({"stage": f"price:{region}", "error": str(e)})
    return records, errors


# --------------------------------------------------------------------------- #
# (c) Daily-only context
# --------------------------------------------------------------------------- #
def _collect_daily(regions: list[str], tick_id: str, t: float) -> tuple[list, list]:
    records: list[dict] = []
    errors: list[dict] = []
    for region in regions:
        try:
            ec2 = _client("ec2", region)
            # ZoneId is the join key: placement scores are reported per AZ *id*
            # (use1-az4), prices per AZ *name* (us-east-1a), and the mapping is
            # per-account. Without this record the two data sets can't be joined.
            for z in ec2.describe_availability_zones().get("AvailabilityZones", []):
                records.append(
                    make_record(
                        "az_map",
                        tick_id,
                        t,
                        region=region,
                        az=z.get("ZoneName", ""),
                        az_id=z.get("ZoneId", ""),
                        state=z.get("State", ""),
                    )
                )
            paginator = ec2.get_paginator("describe_instance_type_offerings")
            for page in paginator.paginate(
                LocationType="availability-zone",
                Filters=[{"Name": "instance-type", "Values": list(INSTANCE_TYPES)}],
            ):
                for off in page.get("InstanceTypeOfferings", []):
                    records.append(
                        make_record(
                            "offering",
                            tick_id,
                            t,
                            region=region,
                            az=off.get("Location", ""),
                            instance_type=off.get("InstanceType", ""),
                        )
                    )
        except Exception as e:  # noqa: BLE001 — per-region isolation
            errors.append({"stage": f"daily:{region}", "error": str(e)})
    try:
        with urllib.request.urlopen(SPOT_ADVISOR_URL, timeout=20) as resp:  # noqa: S310
            doc = json.loads(resp.read().decode())
        records.extend(advisor_records(doc, tick_id, t))
    except Exception as e:  # noqa: BLE001 — public JSON; absence is not fatal
        errors.append({"stage": "spot_advisor", "error": str(e)})
    return records, errors


# --------------------------------------------------------------------------- #
# (d) Truth probe — the only part that spends money or takes capacity
# --------------------------------------------------------------------------- #
def _tagged_instances(region: str, key: str, value: str) -> list[str]:
    ec2 = _client("ec2", region)
    r = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{key}", "Values": [value]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
        ]
    )
    return [i["InstanceId"] for res in r["Reservations"] for i in res["Instances"]]


def _sweep_probes(region: str, tick_id: str, t: float) -> tuple[list, list]:
    """Terminate any probe instance still alive from an earlier tick.

    Belt and braces for the ``finally`` in ``_probe``: if the Lambda is killed
    mid-probe (timeout, deploy race), the leftover is caught here within 10
    minutes — and the instance's own ``shutdown -h +2`` kills it even if this
    never runs. Three independent stops, because an orphaned GPU box is the one
    failure mode that turns a $0 experiment into a $200 one."""
    records: list[dict] = []
    errors: list[dict] = []
    try:
        stray = _tagged_instances(region, PROBE_TAG_KEY, PROBE_TAG_VALUE)
        if stray:
            _client("ec2", region).terminate_instances(InstanceIds=stray)
            records.append(make_record("probe_sweep", tick_id, t, region=region, terminated=stray))
    except Exception as e:  # noqa: BLE001
        errors.append({"stage": "probe_sweep", "error": str(e)})
    return records, errors


def _probe_ami(region: str) -> str:
    """Newest Amazon Linux 2023 x86_64 image. Any bootable AMI works — the probe
    measures whether AWS *hands over the capacity*, not what runs on it."""
    r = _client("ec2", region).describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["al2023-ami-2023.*-kernel-6.1-x86_64"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(r.get("Images", []), key=lambda i: i["CreationDate"])
    if not images:
        raise RuntimeError(f"no Amazon Linux 2023 AMI found in {region}")
    return images[-1]["ImageId"]


def default_vpc_id(region: str) -> str:
    """The region's default VPC, or "" if it has none.

    RunInstances with no SubnetId only works inside a default VPC. Without one
    the probe fails with VPCIdNotSpecified — an infrastructure gap that would
    otherwise be logged next to InsufficientInstanceCapacity and read as "no
    capacity". Checking first turns it into an explicit skip instead.
    """
    r = _client("ec2", region).describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpcs = r.get("Vpcs", [])
    return vpcs[0]["VpcId"] if vpcs else ""


def _probe(cfg: dict, tick_id: str, t: float) -> tuple[list, list]:
    region = cfg["probe_region"]
    instance_type = cfg["probe_type"]
    records: list[dict] = []
    errors: list[dict] = []

    last = _get_json(cfg["bucket"], f"{cfg['prefix']}/meta/last-probe.json") or {}
    try:
        training = _tagged_instances(region, TRAINING_TAG_KEY, TRAINING_TAG_VALUE)
    except Exception as e:  # noqa: BLE001 — can't prove non-interference => don't probe
        return [], [{"stage": "probe_guard", "error": str(e)}]

    go, reason = should_probe(
        now_t=t,
        last_probe_t=last.get("t"),
        training_instances=training,
        enabled=cfg["probe_enabled"],
        min_interval_s=cfg["probe_min_hours"] * 3600,
    )
    if not go:
        return [make_record("probe_skipped", tick_id, t, region=region, reason=reason)], []

    try:
        vpc = default_vpc_id(region)
    except Exception as e:  # noqa: BLE001 — can't tell => don't guess "no capacity"
        vpc, errors = "", [{"stage": "probe_vpc", "error": str(e)}]
    if not vpc:
        # NOT a capacity signal — recorded as its own skip reason so the
        # calibration table never counts it as a failed launch.
        return [
            make_record(
                "probe_skipped",
                tick_id,
                t,
                region=region,
                reason=f"no default VPC in {region} — probe cannot launch without a subnet",
            )
        ], errors

    # Record the attempt BEFORE launching: if this invocation dies mid-probe the
    # rate limiter still counts it, so a crash loop can't launch every 10 min.
    _put_json(cfg["bucket"], f"{cfg['prefix']}/meta/last-probe.json", {"t": t, "region": region})

    instance_id = ""
    try:
        ami = _probe_ami(region)
        started = time.time()
        r = _client("ec2", region).run_instances(
            ImageId=ami,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            InstanceInitiatedShutdownBehavior="terminate",
            # Dead-man's switch: if this Lambda never gets to terminate it, the
            # box powers itself off in ~2 minutes and (shutdown-behavior=
            # terminate) that ends the billing.
            UserData="#!/bin/bash\nshutdown -h +2\n",
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            },
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": "spotwatch-probe"},
                        {"Key": PROBE_TAG_KEY, "Value": PROBE_TAG_VALUE},
                        # NOT project=spot-train: nothing must mistake a probe
                        # for a training box (or vice versa).
                        {"Key": "project", "Value": "spotwatch"},
                    ],
                }
            ],
        )
        inst = r["Instances"][0]
        instance_id = inst["InstanceId"]
        az = inst.get("Placement", {}).get("AvailabilityZone", "")
        state = inst["State"]["Name"]
        # RunInstances with one-time spot options is synchronous: an id means AWS
        # committed the capacity. Wait briefly for `running` to prove it really
        # materialised, then give it straight back.
        deadline = time.time() + cfg["probe_wait_s"]
        while state == "pending" and time.time() < deadline:
            time.sleep(5)
            d = _client("ec2", region).describe_instances(InstanceIds=[instance_id])
            got = d["Reservations"][0]["Instances"][0]
            state = got["State"]["Name"]
            az = got.get("Placement", {}).get("AvailabilityZone", az)
        records.append(
            make_record(
                "probe",
                tick_id,
                t,
                region=region,
                az=az,
                instance_type=instance_type,
                capacity_available=state in ("pending", "running"),
                fulfilled=state == "running",
                final_state=state,
                seconds_to_state=round(time.time() - started, 1),
                instance_id=instance_id,
                error_code="",
            )
        )
    except Exception as e:  # noqa: BLE001 — the failure IS the measurement
        code = error_code(e)
        records.append(
            make_record(
                "probe",
                tick_id,
                t,
                region=region,
                az="",
                instance_type=instance_type,
                capacity_available=False,
                fulfilled=False,
                final_state="failed",
                seconds_to_state=None,
                instance_id=instance_id,
                error_code=code,
            )
        )
    finally:
        if instance_id:
            try:
                _client("ec2", region).terminate_instances(InstanceIds=[instance_id])
            except Exception as e:  # noqa: BLE001 — the sweep retries next tick
                errors.append({"stage": "probe_terminate", "error": str(e)})
    return records, errors


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def handler(event, context):  # noqa: ARG001 — Lambda signature
    t = time.time()
    now = datetime.fromtimestamp(t, timezone.utc)
    tick_id = uuid.uuid4().hex[:8]
    cfg = _config()
    records: list[dict] = []
    errors: list[dict] = []

    regions, region_errors = _pinned_regions(cfg)
    errors.extend(region_errors)

    for collect in (
        lambda: _collect_sps(cfg, regions, tick_id, t),
        lambda: _collect_prices(regions, tick_id, t),
        lambda: _sweep_probes(cfg["probe_region"], tick_id, t),
        lambda: _probe(cfg, tick_id, t),
    ):
        try:
            recs, errs = collect()
        except Exception as e:  # noqa: BLE001 — a whole stage may fail; tick goes on
            recs, errs = [], [{"stage": "collect", "error": str(e)}]
        records.extend(recs)
        errors.extend(errs)

    daily = is_daily_tick(now, cfg["daily_hour"])
    if daily:
        try:
            recs, errs = _collect_daily(regions, tick_id, t)
        except Exception as e:  # noqa: BLE001
            recs, errs = [], [{"stage": "daily", "error": str(e)}]
        records.extend(recs)
        errors.extend(errs)

    gaps = [r for r in records if r["type"] == "sps_gap"]
    records.append(
        make_record(
            "tick",
            tick_id,
            t,
            daily=daily,
            regions=regions,
            sps_requests=len(sps_requests(regions, cfg["home_region"])),
            sps_rows=sum(1 for r in records if r["type"] == "sps"),
            # Coverage, not just health: the report divides by what we ASKED for,
            # so a tick that lost half its requests can't look like a bad market.
            sps_gaps=len(gaps),
            sps_throttled=sum(1 for r in gaps if r.get("throttled")),
            # The observable form of an unpublished limit. Non-zero here means
            # the matrix is wider than this account's 24h configuration budget.
            sps_config_capped=sum(1 for r in gaps if r.get("config_capped")),
            config_budget=MAX_CONFIGURATIONS,
            sps_retries=sum(int(r.get("retries", 0)) for r in records if r["type"] == "sps"),
            record_count=len(records) + 1,
            errors=errors[:20],  # cap: a total outage would otherwise write a novel
            error_count=len(errors),
            duration_s=round(time.time() - t, 2),
        )
    )

    key = shard_key(cfg["prefix"], now, tick_id)
    body = "".join(json.dumps(r) + "\n" for r in records)
    _client("s3", cfg["home_region"]).put_object(Bucket=cfg["bucket"], Key=key, Body=body.encode())
    print(
        f"[spotwatch] wrote s3://{cfg['bucket']}/{key} records={len(records)} errors={len(errors)}"
    )
    return {"key": key, "records": len(records), "errors": len(errors), "daily": daily}
