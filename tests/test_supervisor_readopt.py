"""A restarted supervisor must RE-ADOPT the running fleet, not re-form it.

WHAT HAPPENED (rehearsal multinode-preempt-1786239456, 2026-08-08)

The supervisor was SIGKILLed at t+32m as a deliberate test. systemd restarted it
and the fleet grew 8 -> 10 -> 11 -> 13 -> 14 before it was stopped by hand.
Training itself never broke (steps advanced 370 -> 400 across the restart), and
_reap_orphans correctly declined to terminate the members -- so the re-adoption
work on the ORCH side held. The failure is one layer down, in the supervisor's
own memory.

THE CHAIN

``SupervisorState`` is "cross-tick memory the pure reducer deliberately doesn't
hold" -- and it is rebuilt from defaults on every process start:

    self.st = SupervisorState()      # epoch=0, members=frozenset()

``decide`` then takes its STARTUP branch, because that branch keys off
``obs.epoch == 0`` meaning "no epoch published yet" -- which after a restart is
false in the world but true in memory:

    if obs.epoch == 0:
        if len(healthy) >= obs.node_count:
            return [PublishEpoch(1, ...)]

So a supervisor that restarts onto a healthy 8-node world at epoch 8 republishes
**epoch 1**, clobbering the live epoch document. Sidecars kill and relaunch
torchrun on an epoch change, all 8 boxes churn at once, their heartbeats go
quiet, ``members - healthy`` becomes non-empty, and ``replace_on_loss`` launches
a replacement for each. That is the fleet growth -- a second-order effect. The
primary defect is the epoch reset.

THE FIX THIS FILE PINS

On startup, restore ``epoch`` / ``members`` / ``master`` / ``ips`` from the
durable epoch document (``runs/<run_id>/epoch.json``) before the first tick. The
authority already lives in S3 -- the supervisor simply never reads it back.

These tests are PURE (no AWS, no clock, no I/O) because ``decide`` is a pure
reducer. That is what makes verifying this cost seconds instead of $10 and an
hour of fleet time.
"""

from __future__ import annotations

import pytest

from orchestrator.supervisor import (
    LaunchReplacement,
    NodeObs,
    Observation,
    Policy,
    PublishEpoch,
    SupervisorState,
    decide,
)

POLICY = Policy(replace_on_loss=True, recovery_timeout_s=600.0)


def _nodes(count: int = 8, *, unhealthy: set[int] | None = None) -> tuple[NodeObs, ...]:
    unhealthy = unhealthy or set()
    return tuple(
        NodeObs(
            node=i,
            aws_state="terminated" if i in unhealthy else "running",
            registered=True,
            log_age_s=2.0,
        )
        for i in range(count)
    )


def _obs(**kw) -> Observation:
    base = dict(
        node_count=8,
        nodes=_nodes(),
        epoch=8,
        members=frozenset(range(8)),
        metrics_exists=False,
        no_progress_s=10.0,
    )
    base.update(kw)
    return Observation(**base)


# --------------------------------------------------------------------------- #
# The defect, stated as the reducer sees it
# --------------------------------------------------------------------------- #
def test_a_live_supervisor_at_steady_state_does_nothing():
    """Control: healthy world, membership already correct -> no actions."""
    assert decide(_obs(), POLICY) == []


def test_restarted_supervisor_must_not_republish_epoch_1_over_a_live_world():
    """THE BUG. With state reset to defaults the reducer cannot tell "no epoch
    has ever been published" from "I forgot which epoch is live", and picks the
    most destructive reading -- it republishes epoch 1 over a live epoch 8.

    Passing this test is exactly what state restoration buys: obs.epoch reflects
    the DURABLE epoch, so the startup branch is not taken at all.
    """
    restarted = _obs(epoch=0, members=frozenset(), no_progress_s=None)
    actions = decide(restarted, POLICY)
    republished = [a for a in actions if isinstance(a, PublishEpoch)]
    assert not republished, (
        f"restarted supervisor republished {republished} over a live epoch-8 world; "
        f"sidecars restart torchrun on an epoch change, which is what cascaded the "
        f"fleet from 8 to 14 boxes in the rehearsal"
    )


def test_restored_state_makes_the_restart_a_no_op():
    """The fix, expressed as a property: a supervisor whose state was restored
    from epoch.json sees exactly what the live one saw, so it decides nothing."""
    restored = _obs(epoch=8, members=frozenset(range(8)))
    assert decide(restored, POLICY) == []


# --------------------------------------------------------------------------- #
# The fix must not blunt real recovery
# --------------------------------------------------------------------------- #
def test_after_restore_a_genuinely_dead_node_is_still_replaced_exactly_once():
    """Re-adoption must not turn into "never launch anything". A member observed
    gone still shrinks the world and gets one replacement."""
    obs = _obs(nodes=_nodes(unhealthy={5}))
    actions = decide(obs, POLICY)
    assert LaunchReplacement(5) in actions
    assert sum(isinstance(a, LaunchReplacement) for a in actions) == 1
    published = [a for a in actions if isinstance(a, PublishEpoch)]
    assert published == [PublishEpoch(9, (0, 1, 2, 3, 4, 5, 6, 7)[:5] + (6, 7))]


def test_genuine_cold_start_still_publishes_epoch_1():
    """The startup branch is correct when it is actually a cold start: no epoch
    document exists, so there is nothing to restore and epoch 0 is the truth."""
    cold = _obs(epoch=0, members=frozenset(), no_progress_s=None)
    # A cold start is distinguished by the ABSENCE of a durable epoch doc, which
    # the shell resolves before building the Observation. Here we assert only
    # that the reducer still forms the group when it is genuinely told epoch 0
    # AND that is the truth -- see restore_state tests for who decides that.
    assert decide(cold, POLICY) == [PublishEpoch(1, tuple(range(8)))]


# --------------------------------------------------------------------------- #
# The restore itself
# --------------------------------------------------------------------------- #
def _restore():
    """The function under test, once it exists. Skips (rather than fails) until
    then, so this file is useful to whoever is mid-fix."""
    from orchestrator import supervisor

    fn = getattr(supervisor, "restore_state", None)
    if fn is None:
        pytest.skip("supervisor.restore_state not implemented yet")
    return fn


EPOCH_DOC = {
    "run_id": "multinode-preempt-1",
    "epoch": 8,
    "members": [
        {"node": 0, "ip": "172.31.25.128", "rank": 0},
        {"node": 2, "ip": "172.31.22.188", "rank": 1},
        {"node": 7, "ip": "172.31.27.193", "rank": 2},
    ],
    "master_addr": "172.31.25.128",
}


def test_restore_reads_epoch_members_and_master_from_the_doc():
    st = _restore()(SupervisorState(), EPOCH_DOC)
    assert st.epoch == 8
    assert st.members == frozenset({0, 2, 7})
    # master is the doc's rank-0 member -- epoch_doc() puts it there by
    # construction, so the ordering is the authority, not a separate field.
    assert st.master == 0
    assert st.ips[0] == "172.31.25.128"
    assert st.ips[7] == "172.31.27.193"


def test_restore_with_no_doc_leaves_a_cold_start_untouched():
    """No epoch document => genuinely nothing has been published => epoch 0 is
    correct and the startup branch SHOULD run."""
    st = _restore()(SupervisorState(), None)
    assert st.epoch == 0
    assert st.members == frozenset()


def test_restore_tolerates_a_malformed_doc_without_crashing_the_run():
    """A truncated or half-written epoch doc must not take the supervisor down on
    startup -- that would turn a recoverable restart into a dead control plane."""
    st = _restore()(SupervisorState(), {"epoch": "not-a-number", "members": None})
    assert st.epoch == 0
    assert st.members == frozenset()
