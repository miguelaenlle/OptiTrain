"""CH1 — the main driver must be able to express a SIMULTANEOUS multi-node loss.

`PREEMPT_VICTIMS` gives one victim per round at evenly spaced times, so a
mass-loss event needed a bespoke driver (`scripts/e4_rolling_pairs.py`). E5
survived killing 7 of 8 nodes at once and that is the strongest result this
system has, so expressing it belongs in `multinode-preempt`, not a one-off.

The supervisor already collected due kills into a SET, so simultaneity needed
nothing there beyond a schedule that can say it.
"""

from __future__ import annotations

import pytest

from orchestrator.config import LEADER_VICTIM, OrchestratorConfig


def _cfg(spec: str, nodes: int = 8) -> OrchestratorConfig:
    return OrchestratorConfig(bucket="b", node_count=nodes, preempt_schedule_spec=spec)


def test_empty_spec_falls_back_to_the_old_behaviour():
    assert _cfg("").preempt_schedule() == []


def test_single_victim():
    assert _cfg("480:3").preempt_schedule() == [(480.0, 3)]


def test_simultaneous_group_shares_one_timestamp():
    """The whole point: several victims at the SAME second, which the supervisor
    fires in one tick."""
    assert _cfg("1440:1,4").preempt_schedule() == [(1440.0, 1), (1440.0, 4)]


def test_mass_loss_six_of_eight():
    got = _cfg("2400:0,1,2,3,4,5").preempt_schedule()
    assert len(got) == 6
    assert {v for _t, v in got} == {0, 1, 2, 3, 4, 5}
    assert {t for t, _v in got} == {2400.0}


def test_leader_token_defers_resolution():
    """L must stay a sentinel here -- the master moves, so resolving it at
    schedule-build time would pin it to whoever was master at launch."""
    assert _cfg("960:L").preempt_schedule() == [(960.0, LEADER_VICTIM)]


def test_full_rehearsal_schedule_parses_and_sorts():
    spec = "480:3;960:L;1440:1,4;1920:0;2400:0,1,2,3,4,5;3000:7"
    got = _cfg(spec).preempt_schedule()
    assert [t for t, _v in got] == sorted(t for t, _v in got)
    assert len(got) == 12  # 1 + 1(L) + 2 + 1 + 6 + 1
    assert sum(1 for _t, v in got if v == LEADER_VICTIM) == 1
    assert sum(1 for t, _v in got if t == 2400.0) == 6  # the mass-loss group


def test_total_loss_is_refused():
    """Killing every node is a whole-group restart, not a preemption -- and
    finding that out three minutes into a billed 8-node run is expensive."""
    with pytest.raises(SystemExit, match="total loss with no survivors"):
        _cfg("100:0,1,2,3", nodes=4).preempt_schedule()


def test_out_of_range_index_is_refused():
    with pytest.raises(SystemExit, match="outside"):
        _cfg("100:9", nodes=8).preempt_schedule()


def test_malformed_spec_is_refused():
    with pytest.raises(SystemExit, match="expected"):
        _cfg("100").preempt_schedule()
    with pytest.raises(SystemExit, match="not a number"):
        _cfg("soon:1").preempt_schedule()
    with pytest.raises(SystemExit, match="must be a node index"):
        _cfg("100:x").preempt_schedule()
