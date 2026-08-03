"""Spotwatch tests — hermetic: no AWS, no network, no clock dependence.

Pins the invariants that make a 72-hour unattended collector safe:

  - the SPS request matrix is FIXED (identical every tick, regardless of input
    ordering) — a matrix that drifts burns the 24h per-configuration throttle;
  - the expensive daily work runs on exactly one tick a day;
  - the truth probe is rate-limited AND never runs while a training box is up;
  - every record carries when/which-tick/what-type;
  - a whole tick runs end to end against a fake AWS, and always gives back the
    instance it borrowed;
  - the report's aggregation math and scenario ranking are right on synthetic
    records (this is the analysis the whole experiment exists to produce).
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

import pytest

from orchestrator import lambda_spotwatch as lam
from orchestrator import spotwatch
from orchestrator.config import OrchestratorConfig

REGIONS = ["us-east-1", "us-west-2", "eu-central-1"]


# --------------------------------------------------------------------------- #
# The fixed SPS request matrix
# --------------------------------------------------------------------------- #
def test_sps_matrix_is_identical_across_ticks():
    assert lam.sps_requests(REGIONS) == lam.sps_requests(REGIONS)
    # ... and independent of how the pinned region list happens to be ordered.
    assert lam.sps_requests(REGIONS) == lam.sps_requests(list(reversed(REGIONS)))


def test_sps_matrix_size_and_shape():
    reqs = lam.sps_requests(REGIONS)
    # 6 types x 2 capacities x 2 AZ modes, plus the 6-type basket x 2 x 2.
    assert len(reqs) == (len(lam.INSTANCE_TYPES) + 1) * 2 * 2 == 28
    configs = {
        (
            tuple(r["InstanceTypes"]),
            r["TargetCapacity"],
            r["TargetCapacityUnitType"],
            r["SingleAvailabilityZone"],
            tuple(r["RegionNames"]),
        )
        for r in reqs
    }
    assert len(configs) == len(reqs)  # every request is a DISTINCT configuration
    assert {r["TargetCapacity"] for r in reqs} == {1, 8}
    assert {r["SingleAvailabilityZone"] for r in reqs} == {True, False}
    assert all(r["RegionNames"] == sorted(REGIONS) for r in reqs)


def test_basket_request_is_labelled_any():
    reqs = lam.sps_requests(REGIONS)
    labels = [lam.request_label(r) for r in reqs]
    assert labels.count(lam.BASKET_LABEL) == 4  # one per (capacity, AZ mode)
    assert set(labels) == {*lam.INSTANCE_TYPES, "any"}


def test_matrix_does_not_depend_on_module_level_mutation():
    # Callers must not be able to mutate the constants through a returned request.
    reqs = lam.sps_requests(REGIONS)
    basket = next(r for r in reqs if lam.request_label(r) == "any")
    basket["InstanceTypes"].append("p5.48xlarge")
    assert "p5.48xlarge" not in lam.INSTANCE_TYPES
    assert lam.sps_requests(REGIONS) == lam.sps_requests(REGIONS)


# --------------------------------------------------------------------------- #
# Cheap tick vs full tick
# --------------------------------------------------------------------------- #
def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 8, 2, hour, minute, tzinfo=timezone.utc)


def test_daily_gate_fires_once_per_day():
    fires = [
        _at(h, m)
        for h in range(24)
        for m in range(0, 60, 10)
        if lam.is_daily_tick(_at(h, m), daily_hour=3)
    ]
    assert fires == [_at(3, 0)]


def test_daily_gate_respects_configured_hour():
    assert lam.is_daily_tick(_at(11, 5), daily_hour=11)
    assert not lam.is_daily_tick(_at(11, 15), daily_hour=11)
    assert not lam.is_daily_tick(_at(3, 0), daily_hour=11)


# --------------------------------------------------------------------------- #
# Truth probe: rate limit + non-interference
# --------------------------------------------------------------------------- #
NOW = 1_800_000_000.0


def test_probe_rate_limited_to_six_hours():
    go, why = lam.should_probe(now_t=NOW, last_probe_t=NOW - 5 * 3600, training_instances=[])
    assert not go and "rate-limited" in why
    go, why = lam.should_probe(now_t=NOW, last_probe_t=NOW - 6 * 3600 - 1, training_instances=[])
    assert go and why == "ok"
    go, _ = lam.should_probe(now_t=NOW, last_probe_t=None, training_instances=[])
    assert go  # first ever probe


def test_probe_skipped_while_training_runs_even_if_long_overdue():
    # NON-INTERFERENCE IS ABSOLUTE: a probe must never race a training launch
    # for the last GPU, no matter how stale the data is.
    go, why = lam.should_probe(
        now_t=NOW, last_probe_t=NOW - 30 * 24 * 3600, training_instances=["i-abc"]
    )
    assert not go
    assert "training instances present" in why and "i-abc" in why


def test_probe_can_be_disabled_outright():
    go, why = lam.should_probe(now_t=NOW, last_probe_t=None, training_instances=[], enabled=False)
    assert not go and why == "disabled"


def test_probe_guard_uses_the_tag_training_boxes_actually_carry():
    # aws.launch stamps project=spot-train on every training/fleet instance.
    assert (lam.TRAINING_TAG_KEY, lam.TRAINING_TAG_VALUE) == ("project", "spot-train")
    # ... and probes are tagged distinctly so nothing confuses the two.
    assert (lam.PROBE_TAG_KEY, lam.PROBE_TAG_VALUE) == ("Purpose", "spotwatch-probe")


# --------------------------------------------------------------------------- #
# Record schema + shard keys
# --------------------------------------------------------------------------- #
def test_record_schema_has_time_tick_and_type():
    r = lam.make_record("sps", "abc12345", NOW, region="us-east-1", score=7)
    assert r["type"] == "sps" and r["tick_id"] == "abc12345"
    assert r["t"] == NOW and r["ts"].endswith("Z")
    assert datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ")
    assert json.loads(json.dumps(r)) == r  # must survive a JSONL round trip


def test_shard_keys_are_distinct_per_tick():
    now = _at(4, 20)
    k1 = lam.shard_key("spotwatch", now, "aaaaaaaa")
    k2 = lam.shard_key("spotwatch", now, "bbbbbbbb")
    assert k1 != k2  # append-only via distinct keys, never read-modify-write
    assert k1.startswith("spotwatch/2026-08-02/") and k1.endswith(".jsonl")
    assert spotwatch.shard_epoch(k1) == pytest.approx(now.timestamp())


def test_advisor_records_flatten_only_our_types():
    doc = {
        "ranges": [{"index": 1, "label": "<5%"}, {"index": 3, "label": "10-15%"}],
        "spot_advisor": {
            "us-east-1": {
                "Linux": {"g5.xlarge": {"s": 65, "r": 1}, "m5.large": {"s": 70, "r": 3}},
                "Windows": {"g5.xlarge": {"s": 10, "r": 3}},
            }
        },
    }
    recs = lam.advisor_records(doc, "tick1", NOW)
    assert [r["instance_type"] for r in recs] == ["g5.xlarge"]  # m5/Windows dropped
    assert recs[0]["interruption_range"] == "<5%" and recs[0]["savings_pct"] == 65


# --------------------------------------------------------------------------- #
# Packaging + deploy plumbing (no AWS: dry-run only)
# --------------------------------------------------------------------------- #
def test_package_is_a_deterministic_zip_of_the_handler():
    a = spotwatch.build_package()
    assert a == spotwatch.build_package()  # same source => same bytes => no churn
    with zipfile.ZipFile(__import__("io").BytesIO(a)) as z:
        assert z.namelist() == ["lambda_spotwatch.py"]
        source = z.read("lambda_spotwatch.py").decode()
    assert "def handler(" in source
    # The handler must be dependency-free: boto3 (in the runtime) + stdlib only.
    for banned in ("import requests", "from orchestrator", "import numpy", "import torch"):
        assert banned not in source


def test_policy_is_scoped_to_the_spotwatch_prefix():
    pol = spotwatch.lambda_policy("my-bucket", "spotwatch")
    objects = next(s for s in pol["Statement"] if s["Sid"] == "WriteCollectedData")
    assert objects["Resource"] == "arn:aws:s3:::my-bucket/spotwatch/*"
    actions = {a for s in pol["Statement"] for a in s["Action"]}
    assert "ec2:GetSpotPlacementScores" in actions
    assert "ec2:TerminateInstances" in actions  # the probe must be able to clean up
    assert "s3:DeleteObject" not in actions  # collected data is append-only


def test_deploy_and_down_are_dry_runnable(monkeypatch, capsys):
    from orchestrator import aws

    monkeypatch.setattr(aws, "_DRY_RUN", True, raising=False)
    aws.set_dry_run(True)
    try:
        cfg = OrchestratorConfig(bucket="test-bucket")
        spotwatch.deploy(cfg)
        spotwatch.down(cfg)
    finally:
        aws.set_dry_run(False)
    err = capsys.readouterr().err
    assert "would deploy lambda spotwatch" in err
    assert "rate(10 minutes)" in err
    assert "would delete EventBridge rule spotwatch-tick" in err


def test_lambda_env_matches_what_the_handler_reads():
    cfg = OrchestratorConfig(bucket="test-bucket")
    env = spotwatch.lambda_env(cfg)
    assert env["SPOTWATCH_BUCKET"] == "test-bucket"
    assert "AWS_REGION" not in env  # reserved by Lambda; setting it fails deploy
    source = open(  # noqa: SIM115
        lam.__file__
    ).read()
    for key in env:
        assert key in source, f"{key} is deployed but the handler never reads it"


# --------------------------------------------------------------------------- #
# A whole tick against a fake AWS (still hermetic: boto3 is never constructed)
# --------------------------------------------------------------------------- #
class FakeAws:
    """One object standing in for every boto3 client the handler uses."""

    def __init__(self, *, objects=None, training=(), stray=(), run_error=None, state="running"):
        self.objects = dict(objects or {})
        self.training = list(training)
        self.stray = list(stray)
        self.run_error = run_error
        self.state = state
        self.sps_calls: list[dict] = []
        self.launched: list[dict] = []
        self.terminated: list[str] = []

    def __call__(self, service, region):  # replaces lambda_spotwatch._client
        return self

    # --- s3 ---------------------------------------------------------------- #
    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg names
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key].encode())}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body.decode()
        return {}

    # --- ec2 --------------------------------------------------------------- #
    def get_spot_placement_scores(self, **kwargs):
        self.sps_calls.append(kwargs)
        return {
            "SpotPlacementScores": [
                {"Region": "us-east-1", "AvailabilityZoneId": "use1-az4", "Score": 7}
            ]
        }

    def get_paginator(self, name):
        pages = {
            "describe_spot_price_history": [
                {
                    "SpotPriceHistory": [
                        {
                            "AvailabilityZone": "us-east-1d",
                            "InstanceType": "g5.xlarge",
                            "SpotPrice": "0.42",
                            "Timestamp": datetime(2026, 8, 1, tzinfo=timezone.utc),
                        }
                    ]
                }
            ],
            "describe_instance_type_offerings": [
                {"InstanceTypeOfferings": [{"Location": "us-east-1d", "InstanceType": "g5.xlarge"}]}
            ],
        }[name]

        class _P:
            def paginate(self, **kwargs):
                return iter(pages)

        return _P()

    def describe_instances(self, Filters=None, InstanceIds=None):  # noqa: N803
        if InstanceIds:
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": InstanceIds[0],
                                "State": {"Name": self.state},
                                "Placement": {"AvailabilityZone": "us-east-1d"},
                            }
                        ]
                    }
                ]
            }
        wanted = {f["Name"]: f["Values"] for f in Filters or []}
        ids = self.training if "tag:project" in wanted else self.stray
        return {"Reservations": [{"Instances": [{"InstanceId": i} for i in ids]}]}

    def describe_availability_zones(self):
        return {
            "AvailabilityZones": [
                {"ZoneName": "us-east-1d", "ZoneId": "use1-az4", "State": "available"}
            ]
        }

    def describe_images(self, **kwargs):
        return {"Images": [{"ImageId": "ami-fake", "CreationDate": "2026-07-01T00:00:00Z"}]}

    def run_instances(self, **kwargs):
        if self.run_error:
            raise self.run_error
        self.launched.append(kwargs)
        return {
            "Instances": [
                {
                    "InstanceId": "i-probe",
                    "State": {"Name": self.state},
                    "Placement": {"AvailabilityZone": "us-east-1d"},
                }
            ]
        }

    def terminate_instances(self, InstanceIds):  # noqa: N803
        self.terminated.extend(InstanceIds)
        return {}


PINNED = json.dumps({"regions": ["us-east-1"]})


@pytest.fixture
def tick_env(monkeypatch):
    monkeypatch.setenv("SPOTWATCH_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr(lam.time, "sleep", lambda _s: None)


def _run_tick(monkeypatch, fake, at_hour=12):
    """Invoke the handler with the clock pinned to a known UTC hour."""
    when = datetime(2026, 8, 1, at_hour, 5, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(lam.time, "time", lambda: when)
    monkeypatch.setattr(lam, "_client", fake)
    result = fake_result = lam.handler({}, None)
    shard = fake.objects[fake_result["key"]]
    return result, [json.loads(line) for line in shard.splitlines()], fake_result


def test_tick_writes_one_shard_with_every_record_type(tick_env, monkeypatch):
    fake = FakeAws(objects={"spotwatch/meta/regions.json": PINNED})
    result, records, _ = _run_tick(monkeypatch, fake)
    kinds = {r["type"] for r in records}
    assert {"sps", "price", "probe", "tick"} <= kinds
    assert result["records"] == len(records)
    assert result["key"].startswith("spotwatch/2026-08-01/")
    # Exactly the fixed matrix went out — no more, no fewer, no ad-hoc extras.
    assert len(fake.sps_calls) == 28
    assert all(r["tick_id"] == records[0]["tick_id"] for r in records)
    # Non-daily tick: no pool enumeration, no advisor fetch.
    assert "offering" not in kinds and "interruption" not in kinds and "az_map" not in kinds


def test_daily_tick_adds_the_expensive_context(tick_env, monkeypatch):
    fake = FakeAws(objects={"spotwatch/meta/regions.json": PINNED})
    monkeypatch.setattr(lam.urllib.request, "urlopen", _boom)  # advisor unreachable
    _, records, _ = _run_tick(monkeypatch, fake, at_hour=3)
    kinds = {r["type"] for r in records}
    assert {"offering", "az_map"} <= kinds
    tick = next(r for r in records if r["type"] == "tick")
    assert tick["daily"] is True
    # A dead advisor URL is logged, not fatal — the rest of the tick still landed.
    assert any("spot_advisor" in e.get("stage", "") for e in tick["errors"])


def _boom(*args, **kwargs):
    raise OSError("no network in tests")


def test_probe_launches_once_and_always_gives_the_instance_back(tick_env, monkeypatch):
    fake = FakeAws(objects={"spotwatch/meta/regions.json": PINNED})
    _, records, _ = _run_tick(monkeypatch, fake)
    probe = next(r for r in records if r["type"] == "probe")
    assert probe["capacity_available"] is True and probe["fulfilled"] is True
    assert fake.terminated == ["i-probe"]  # the finally block, not best effort
    tags = {t["Key"]: t["Value"] for t in fake.launched[0]["TagSpecifications"][0]["Tags"]}
    assert tags["Purpose"] == "spotwatch-probe" and tags["project"] == "spotwatch"
    assert "shutdown -h" in fake.launched[0]["UserData"]  # dead-man's switch
    spot_opts = fake.launched[0]["InstanceMarketOptions"]["SpotOptions"]
    assert spot_opts["SpotInstanceType"] == "one-time"
    # The attempt is recorded, so the next tick is rate-limited even if we crash.
    assert json.loads(fake.objects["spotwatch/meta/last-probe.json"])["t"] > 0


def test_probe_records_insufficient_capacity_as_the_measurement(tick_env, monkeypatch):
    err = RuntimeError("no capacity")
    err.response = {"Error": {"Code": "InsufficientInstanceCapacity"}}
    fake = FakeAws(objects={"spotwatch/meta/regions.json": PINNED}, run_error=err)
    _, records, _ = _run_tick(monkeypatch, fake)
    probe = next(r for r in records if r["type"] == "probe")
    assert probe["capacity_available"] is False
    assert probe["error_code"] == "InsufficientInstanceCapacity"
    assert fake.terminated == []  # nothing was ever launched


def test_tick_never_probes_while_training_is_running(tick_env, monkeypatch):
    fake = FakeAws(objects={"spotwatch/meta/regions.json": PINNED}, training=["i-train"])
    _, records, _ = _run_tick(monkeypatch, fake)
    assert not fake.launched  # NON-INTERFERENCE: not one RunInstances
    skipped = next(r for r in records if r["type"] == "probe_skipped")
    assert "i-train" in skipped["reason"]
    assert any(r["type"] == "sps" for r in records)  # observation continues


def test_tick_sweeps_leftover_probes(tick_env, monkeypatch):
    fake = FakeAws(objects={"spotwatch/meta/regions.json": PINNED}, stray=["i-orphan"])
    _, records, _ = _run_tick(monkeypatch, fake)
    assert "i-orphan" in fake.terminated
    sweep = next(r for r in records if r["type"] == "probe_sweep")
    assert sweep["terminated"] == ["i-orphan"]


def test_pinned_region_list_is_never_rewritten(tick_env, monkeypatch):
    # Rewriting it would silently change the SPS request configuration and reset
    # the 24h throttle budget mid-experiment.
    fake = FakeAws(objects={"spotwatch/meta/regions.json": PINNED})
    _run_tick(monkeypatch, fake)
    assert json.loads(fake.objects["spotwatch/meta/regions.json"]) == {"regions": ["us-east-1"]}
    first = [r["RegionNames"] for r in fake.sps_calls]
    _run_tick(monkeypatch, fake, at_hour=13)
    assert [r["RegionNames"] for r in fake.sps_calls][: len(first)] == first


def test_first_ever_tick_enumerates_and_pins_regions(tick_env, monkeypatch):
    fake = FakeAws()  # no pinned doc yet
    fake.describe_regions = lambda: {"Regions": [{"RegionName": "us-east-1"}]}
    fake.describe_instance_type_offerings = lambda **kw: {
        "InstanceTypeOfferings": [{"InstanceType": "g5.xlarge"}]
    }
    _run_tick(monkeypatch, fake)
    pinned = json.loads(fake.objects["spotwatch/meta/regions.json"])
    assert pinned["regions"] == ["us-east-1"]
    assert "pinned_at" in pinned


# --------------------------------------------------------------------------- #
# Report: window selection + aggregation math on synthetic records
# --------------------------------------------------------------------------- #
def test_keys_in_window_filters_by_epoch_in_the_key():
    keys = [
        "spotwatch/2026-08-01/1000-aaaa.jsonl",
        "spotwatch/2026-08-01/2000-bbbb.jsonl",
        "spotwatch/meta/regions.json",
        "spotwatch/2026-08-01/notes.txt",
    ]
    assert spotwatch.keys_in_window(keys, 1500) == ["spotwatch/2026-08-01/2000-bbbb.jsonl"]


def test_parse_shard_skips_a_truncated_final_line():
    text = '{"type": "sps", "t": 1}\n\n{"type": "price", "t": 2}\n{"type": "tru'
    recs = spotwatch.parse_shard(text)
    assert [r["type"] for r in recs] == ["sps", "price"]


BASE = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc).timestamp()


def _sps(tick, hour, region, az_id, itype, score, *, capacity=1, single_az=True, day=0):
    t = BASE + day * 86400 + hour * 3600
    return lam.make_record(
        "sps",
        tick,
        t,
        region=region,
        az_id=az_id,
        instance_type=itype,
        capacity=capacity,
        single_az=single_az,
        score=score,
    )


def _synthetic() -> list[dict]:
    """Two ticks an hour apart in a world where:
    - us-east-1 / g5.xlarge is hopeless at az1 (score 2) but fine at az4 (8);
    - g4dn.xlarge is good region-wide at hour 3 only;
    - us-west-2 is uniformly excellent.
    """
    recs: list[dict] = []
    for tick, hour in (("t0", 1), ("t1", 3)):
        recs += [
            _sps(tick, hour, "us-east-1", "use1-az1", "g5.xlarge", 2),
            _sps(tick, hour, "us-east-1", "use1-az4", "g5.xlarge", 8 if hour == 3 else 3),
            _sps(tick, hour, "us-east-1", "use1-az1", "g4dn.xlarge", 9 if hour == 3 else 4),
            _sps(tick, hour, "us-west-2", "usw2-az1", "g5.xlarge", 10),
            _sps(tick, hour, "us-east-1", "", "g5.xlarge", 3, single_az=False),
            _sps(
                tick, hour, "us-east-1", "", "g4dn.xlarge", 9 if hour == 3 else 4, single_az=False
            ),
            _sps(tick, hour, "us-east-1", "use1-az4", "any", 9 if hour == 3 else 5),
        ]
        recs.append(
            lam.make_record(
                "tick", tick, BASE + hour * 3600, daily=False, error_count=0, record_count=7
            )
        )
    recs += [
        lam.make_record(
            "az_map", "t0", BASE, region="us-east-1", az="us-east-1a", az_id="use1-az1"
        ),
        lam.make_record(
            "az_map", "t0", BASE, region="us-east-1", az="us-east-1d", az_id="use1-az4"
        ),
    ]
    return recs


def test_per_tick_best_takes_the_best_pool_not_the_count():
    recs = _synthetic()
    rows = spotwatch.sps_rows(recs, capacity=1, single_az=True)
    best = spotwatch.per_tick_best(rows, lambda r: r["region"] == "us-east-1")
    assert set(best) == {"t0", "t1"}
    assert best["t0"] == 5.0  # the 'any' basket record is the best at hour 1
    assert best["t1"] == 9.0


def test_odds_counts_ticks_not_records():
    o = spotwatch.odds({"t0": 3.0, "t1": 9.0, "t2": 7.0}, threshold=7)
    assert o["ticks"] == 3
    assert o["p_good"] == pytest.approx(2 / 3)
    assert o["mean_best"] == pytest.approx(19 / 3)
    assert o["max_best"] == 9.0
    empty = spotwatch.odds({}, threshold=7)
    assert empty == {"ticks": 0, "p_good": None, "mean_best": None, "max_best": None}


def test_scenario_ranking_widens_monotonically():
    rows = spotwatch.scenarios(
        _synthetic(), threshold=7, home_region="us-east-1", home_type="g5.xlarge", capacity=1
    )
    by_name = {r["scenario"]: r for r in rows}
    never = by_name["never switch (same region, same GPU)"]
    gpu = by_name["switch GPU only (same region, any of our GPUs)"]
    az = by_name["switch AZ only (same region+GPU, best AZ)"]
    both = by_name["switch both (same region, any GPU, best AZ)"]
    anywhere = by_name["switch region too (anywhere, any GPU, best AZ)"]

    # home pool is never good; switching GPU or AZ each rescue one of two ticks;
    # widening to both can only ever help; another region is always available.
    assert never["p_good"] == 0.0
    assert gpu["p_good"] == pytest.approx(0.5)
    assert az["p_good"] == pytest.approx(0.5)
    assert both["p_good"] == pytest.approx(0.5)
    assert anywhere["p_good"] == 1.0
    assert never["p_good"] <= gpu["p_good"] <= both["p_good"] <= anywhere["p_good"]
    assert never["p_good"] <= az["p_good"] <= both["p_good"]
    # every scenario saw both ticks (a scenario with no data reports ticks=0)
    assert {r["ticks"] for r in rows} == {2}


def test_hourly_odds_finds_the_good_hour():
    hours = spotwatch.hourly_odds(
        _synthetic(), threshold=7, home_region="us-east-1", home_type="g5.xlarge", capacity=1
    )
    assert set(hours) == {1, 3}
    assert hours[1]["p_good"] == 0.0
    assert hours[3]["p_good"] == 1.0  # az4 hits 8 at 03:00


def test_rank_pools_prefers_sustained_scores_and_drops_the_basket():
    pools = spotwatch.rank_pools(_synthetic(), capacity=1, threshold=7)
    assert (pools[0]["region"], pools[0]["az_id"]) == ("us-west-2", "usw2-az1")
    assert pools[0]["mean"] == 10.0 and pools[0]["p_good"] == 1.0
    assert all(p["instance_type"] != "any" for p in pools)  # basket isn't a pool
    assert {p["samples"] for p in pools} == {2}


def test_heatmap_averages_per_hour():
    grid = spotwatch.heatmap(
        [
            _sps("t0", 3, "us-east-1", "use1-az1", "g5.xlarge", 4),
            _sps("t1", 3, "us-east-1", "use1-az1", "g5.xlarge", 8),
            _sps("t2", 4, "us-east-1", "use1-az1", "g5.xlarge", 2),
        ],
        "az_id",
    )
    assert grid["use1-az1"][3] == 6.0
    assert grid["use1-az1"][4] == 2.0


def test_calibration_joins_probes_to_the_score_at_that_moment():
    recs = _synthetic()
    probe_t = BASE + 3 * 3600 + 60  # one minute after the 03:00 tick
    recs.append(
        lam.make_record(
            "probe",
            "t1",
            probe_t,
            region="us-east-1",
            az="us-east-1d",
            instance_type="g5.xlarge",
            capacity_available=True,
            fulfilled=True,
            error_code="",
        )
    )
    recs.append(
        lam.make_record(
            "probe",
            "t0",
            BASE + 3600 + 60,
            region="us-east-1",
            az="",
            instance_type="g5.xlarge",
            capacity_available=False,
            fulfilled=False,
            error_code="InsufficientInstanceCapacity",
        )
    )
    cal = spotwatch.calibration(recs)
    assert len(cal) == 2
    failed, ok = cal[0], cal[1]
    assert failed["capacity_available"] is False
    assert failed["region_score"] == 3  # SPS said 3 and the launch indeed failed
    assert ok["capacity_available"] is True
    assert ok["region_score"] == 3  # region-level ask stays pessimistic...
    assert ok["az_score"] == 8  # ...but the AZ we actually landed in scored 8


def test_calibration_ignores_scores_outside_the_join_window():
    recs = _synthetic()
    recs.append(
        lam.make_record(
            "probe",
            "tX",
            BASE + 12 * 3600,  # hours away from any tick
            region="us-east-1",
            az="us-east-1d",
            instance_type="g5.xlarge",
            capacity_available=True,
            error_code="",
        )
    )
    assert spotwatch.calibration(recs)[-1]["region_score"] is None


def test_price_and_interruption_summaries():
    recs = [
        lam.make_record(
            "price",
            "t0",
            BASE,
            region="us-east-1",
            az="us-east-1a",
            instance_type="g5.xlarge",
            price_usd=0.40,
        ),
        lam.make_record(
            "price",
            "t1",
            BASE + 600,
            region="us-east-1",
            az="us-east-1a",
            instance_type="g5.xlarge",
            price_usd=0.60,
        ),
        lam.make_record(
            "price",
            "t0",
            BASE,
            region="us-east-1",
            az="us-east-1b",
            instance_type="g5.xlarge",
            price_usd=0.30,
        ),
        lam.make_record(
            "interruption",
            "t0",
            BASE,
            region="us-east-1",
            instance_type="g5.xlarge",
            interruption_range="<5%",
            savings_pct=60,
        ),
        lam.make_record(
            "interruption",
            "t1",
            BASE + 86400,
            region="us-east-1",
            instance_type="g5.xlarge",
            interruption_range="10-15%",
            savings_pct=64,
        ),
    ]
    prices = spotwatch.price_summary(recs)
    assert prices[0]["az"] == "us-east-1b" and prices[0]["median_usd"] == 0.30
    assert prices[1]["median_usd"] == pytest.approx(0.50)  # median of 0.40/0.60
    assert prices[1]["min_usd"] == 0.40 and prices[1]["max_usd"] == 0.60
    interrupts = spotwatch.interruption_summary(recs)
    assert len(interrupts) == 1  # newest row per (region, type)
    assert interrupts[0]["interruption_range"] == "10-15%"


def test_render_report_is_plain_text_and_answers_the_question():
    text = spotwatch.render_report(
        _synthetic(), threshold=7, home_region="us-east-1", home_type="g5.xlarge", since_hours=72
    )
    assert "SPOTWATCH" in text
    assert "never switch (same region, same GPU)" in text
    assert "WAIT FOR A GOOD TIME" in text
    assert "AVAILABILITY BY AZ x HOUR" in text
    assert "us-east-1d" in text  # az-id resolved to a name via the az_map records
    assert "VERDICT" in text
    assert max(len(line) for line in text.splitlines()) <= spotwatch.WIDTH


def test_verdict_calls_out_a_hopeless_window():
    dead = [
        _sps(f"t{i}", i, "us-east-1", "use1-az1", "g5.xlarge", 2, single_az=False) for i in range(4)
    ]
    text = spotwatch.verdict(dead, threshold=7, home_region="us-east-1", home_type="g5.xlarge")
    assert "NEVER ENOUGH" in text


def test_verdict_names_the_cheapest_working_strategy():
    always = []
    for i in range(10):
        always.append(_sps(f"t{i}", i % 24, "us-east-1", "", "g5.xlarge", 9, single_az=False))
    text = spotwatch.verdict(always, threshold=7, home_region="us-east-1", home_type="g5.xlarge")
    assert "never switch (same region, same GPU)" in text and "100.0%" in text


def test_heatmap_render_marks_missing_hours():
    grid = {"use1-az1": {3: 8.0, 4: 10.0}}
    out = spotwatch.render_heatmap("title", grid, label_width=22)
    row = next(line for line in out.splitlines() if line.startswith("use1-az1"))
    cells = row[22:46]
    assert cells.count(".") == 22  # 24 hours, 2 with data
    assert cells[3] == "8" and cells[4] == "X"  # 10 renders as X


def test_render_png_is_optional(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "availability.png"
    assert spotwatch.render_png(
        _synthetic(), str(out), home_region="us-east-1", home_type="g5.xlarge"
    )
    assert out.stat().st_size > 0
    # no data for that pool => no chart, no crash
    assert not spotwatch.render_png(
        _synthetic(), str(out), home_region="ap-south-1", home_type="g5.xlarge"
    )


def test_record_window_boundaries_are_inclusive_of_the_cutoff():
    keys = [f"spotwatch/2026-08-01/{int(BASE)}-aaaa.jsonl"]
    assert spotwatch.keys_in_window(keys, BASE) == keys
    assert spotwatch.keys_in_window(keys, BASE + 1) == []


def test_report_reads_shards_and_renders(monkeypatch, tmp_path, capsys):
    """The S3 side of `report`, with the two aws.py calls stubbed — proves the
    key filtering, shard parsing and rendering line up end to end."""
    from orchestrator import aws

    recs = _synthetic()
    now = __import__("time").time()
    for i, r in enumerate(recs):  # slide the synthetic window into "recently"
        r["t"] = now - 3600 + i
    key = f"spotwatch/2026-08-01/{int(now - 3600)}-aaaa.jsonl"
    monkeypatch.setattr(aws, "list_keys", lambda bucket, prefix: [key, "spotwatch/meta/x.json"])
    monkeypatch.setattr(
        aws,
        "get_texts",
        lambda bucket, keys: {key: "".join(json.dumps(r) + "\n" for r in recs)},
    )
    monkeypatch.chdir(tmp_path)
    cfg = OrchestratorConfig(bucket="test-bucket")
    text = spotwatch.report(cfg, since_hours=72, threshold=7, home_type="g5.xlarge")
    assert "SPOTWATCH" in text and "VERDICT" in text
    assert text in capsys.readouterr().out
    written = list(tmp_path.glob("reports/spotwatch-*/report.txt"))
    assert written and written[0].read_text().startswith("=")


def test_report_refuses_an_empty_window(monkeypatch):
    from orchestrator import aws

    monkeypatch.setattr(aws, "list_keys", lambda bucket, prefix: [])
    with pytest.raises(SystemExit, match="no spotwatch shards"):
        spotwatch.load_records(OrchestratorConfig(bucket="test-bucket"), 72)


def test_seventy_two_hour_window_is_the_documented_default():
    # The deliverable is a 72h unattended collection; the report defaults to it.
    import inspect

    assert inspect.signature(spotwatch.report).parameters["since_hours"].default == 72.0
    # ... and 10-minute ticks over 3 days is what the sample counts assume.
    assert OrchestratorConfig().spotwatch_interval_minutes == 10
    assert timedelta(hours=72) / timedelta(minutes=10) == 432
