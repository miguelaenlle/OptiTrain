"""A8 — `spot-orchestrate extend`: continue a COMPLETED supervised run.

Two uses: patch-and-continue after a bug kills a long run (otherwise the whole
run is written off), and buying a longer result later.

`resume` cannot do this — it derives `kind = run_id.split("-", 1)[0]`, mapping
`multinode-preempt-…` to `"multinode"`, then refuses as a single-box salvage
tool. These tests pin the three things that would silently cost money or data.
"""

from __future__ import annotations

import json

import pytest

from orchestrator import experiments
from orchestrator.config import OrchestratorConfig

RUN = "multinode-preempt-1786231481"


class FakeAws:
    def __init__(self, docs=None, has_ckpt=True):
        self.docs = docs or {}
        self.has_ckpt = has_ckpt
        self.deleted: list[str] = []
        self.puts: dict[str, str] = {}

    def is_dry_run(self):
        return False

    def any_object_under(self, bucket, prefix):
        return self.has_ckpt

    def object_exists(self, bucket, key):
        return key in self.docs or key in self.puts

    def get_json(self, bucket, key):
        return self.docs.get(key)

    def put_text(self, bucket, key, body, **kw):
        self.puts[key] = body

    def delete_object(self, bucket, key):
        self.deleted.append(key)
        self.docs.pop(key, None)


@pytest.fixture
def wired(monkeypatch):
    cfg = OrchestratorConfig(bucket="b", node_count=8)
    fake = FakeAws(
        docs={
            cfg.run_metrics_key(RUN): {"steps": 19600, "trained_seconds_total": 82800.0},
            cfg.run_profile_key(RUN): {"run_id": RUN, "cost": {"instances": []}},
        }
    )
    calls: dict = {}

    def spy(cfg_, **kw):
        calls.update(kw)
        return {"ok": True}

    monkeypatch.setattr(experiments, "aws", fake)
    monkeypatch.setattr(experiments, "_run_supervised", spy)
    return cfg, fake, calls


def test_kind_is_taken_off_the_right_not_the_left(wired):
    """The exact bug that makes `resume` unusable here: splitting on the FIRST
    '-' turns multinode-preempt into multinode."""
    cfg, _fake, calls = wired
    experiments.run_extend(cfg, RUN, budget=125000)
    assert calls["kind"] == "multinode-preempt"
    assert calls["run_id"] == RUN


def test_budget_is_a_total_not_an_increment(wired):
    """Passing an increment would silently stop the run almost immediately,
    because budget-in-checkpoint computes budget minus trained_seconds."""
    cfg, _fake, _calls = wired
    with pytest.raises(SystemExit, match="new TOTAL, not an increment"):
        experiments.run_extend(cfg, RUN, budget=36000)  # less than the 82800 trained


def test_completed_artifacts_are_archived_before_re_entry(wired):
    """A re-run overwrites metrics.json and profile.json. Without archiving,
    extending destroys the very result that was just paid for -- and orch's agent
    treats an existing metrics.json as 'nothing to do'."""
    cfg, fake, _calls = wired
    experiments.run_extend(cfg, RUN, budget=125000)
    archived = sorted(fake.puts)
    assert any(k.endswith("metrics-seg1.json") for k in archived), archived
    assert any(k.endswith("profile-seg1.json") for k in archived), archived
    assert cfg.run_metrics_key(RUN) in fake.deleted
    # the archived copy must be the real document, not a stub
    body = json.loads(fake.puts[[k for k in archived if k.endswith("metrics-seg1.json")][0]])
    assert body["steps"] == 19600


def test_refuses_a_run_with_no_checkpoints(monkeypatch):
    cfg = OrchestratorConfig(bucket="b", node_count=8)
    monkeypatch.setattr(experiments, "aws", FakeAws(has_ckpt=False))
    monkeypatch.setattr(experiments, "_run_supervised", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="nothing to extend from"):
        experiments.run_extend(cfg, RUN, budget=125000)


def test_refuses_a_single_box_kind(monkeypatch):
    cfg = OrchestratorConfig(bucket="b", node_count=8)
    monkeypatch.setattr(experiments, "aws", FakeAws())
    monkeypatch.setattr(experiments, "_run_supervised", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="supervised multi-node runs"):
        experiments.run_extend(cfg, "baseline-123", budget=100)
