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

from orchestrator.supervisor import (
    LaunchReplacement,
    NodeObs,
    Observation,
    Policy,
    PublishEpoch,
    SupervisorState,
    decide,
    restore_state,
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
    """THE BUG, stated where it can actually be fixed.

    An earlier version of this test asserted that ``decide`` itself must not
    republish when handed ``epoch=0``. That was wrong, and provably so: it fed
    the reducer exactly the same Observation as
    ``test_genuine_cold_start_still_publishes_epoch_1`` and demanded the opposite
    answer. At the reducer's interface "restarted onto a live world" and "cold
    start" are the SAME observation -- which is precisely why the fix cannot live
    there.

    So the property is about the SHELL: a supervisor that restarts while
    epoch.json says a world is live must rebuild its memory from that document
    BEFORE observing, and then decide nothing. Restoration is what makes the
    restart invisible; the reducer stays honest and unchanged.
    """
    live_doc = {
        "epoch": 8,
        "members": [{"node": i, "ip": f"172.31.0.{i}", "rank": i} for i in range(8)],
    }
    # A freshly-started process: SupervisorState() is all defaults.
    st = restore_state(SupervisorState(), live_doc)
    assert st.epoch == 8, "restart did not re-adopt the live epoch"

    # The first Observation is built from that restored memory.
    first_tick = _obs(epoch=st.epoch, members=st.members)
    actions = decide(first_tick, POLICY)
    republished = [a for a in actions if isinstance(a, PublishEpoch)]
    assert not republished, (
        f"restarted supervisor republished {republished} over a live epoch-8 world; "
        f"sidecars kill and relaunch torchrun on an epoch change, which is what "
        f"cascaded the fleet from 8 to 14 boxes in the rehearsal"
    )
    assert actions == [], f"a restart onto a healthy world must be a no-op, got {actions}"


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
    return restore_state


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


# --------------------------------------------------------------------------- #
# Fleet identity — the half the first version of this file missed
# --------------------------------------------------------------------------- #
# Restoring epoch/members alone was NOT enough, and the 2-node cloud check is
# what proved it: the supervisor re-adopted epoch 3 correctly, then published
# epoch 4 with members [0] and the fleet went 2 -> 5.
#
# Two dicts are ALSO process memory, and _observe needs both:
#   node_ids[node] -> instance id : without it aws_state cannot be resolved, so
#                                   the node is never observed at all, drops out
#                                   of `healthy`, and gets replaced;
#   logs[node]["key"]             : read every tick for the heartbeat, so a
#                                   missing entry is a KeyError on tick one.
#
# And _run_supervised launched node_count boxes unconditionally, which doubles a
# fleet that never stopped training. These tests pin the adoption that fixes it.
class _FakeAws:
    def __init__(self, status_doc, states):
        self._doc, self._states = status_doc, states
        self.described = []

    def get_json(self, _bucket, _key):
        return self._doc

    def instance_state(self, iid):
        self.described.append(iid)
        return self._states.get(iid, "terminated")


def _status(*rows):
    return {
        "nodes": [
            dict(zip(("node", "instance_id", "state", "attempt"), r, strict=False)) for r in rows
        ]
    }


def _adopt(monkeypatch, status_doc, states):
    from orchestrator import experiments

    fake = _FakeAws(status_doc, states)
    monkeypatch.setattr(experiments, "aws", fake)
    cfg = type(
        "Cfg",
        (),
        {
            "bucket": "b",
            "run_status_key": lambda self, r: "k",
            "run_logs_key": lambda self, r, node=0, attempt=0: f"logs/{node}",
        },
    )()
    logs: dict[int, dict] = {}
    return experiments.adopt_fleet(cfg, "run-1", logs), logs


def test_adopt_returns_live_boxes_so_they_are_not_relaunched(monkeypatch):
    doc = _status((0, "i-aaa", "alive", 0), (1, "i-bbb", "alive", 0))
    adopted, logs = _adopt(monkeypatch, doc, {"i-aaa": "running", "i-bbb": "running"})
    assert adopted == {0: "i-aaa", 1: "i-bbb"}
    # logs must be populated too, or _observe raises KeyError on the first tick.
    assert logs[0]["key"] and logs[1]["key"]


def test_adopt_skips_dead_attempts(monkeypatch):
    """A replaced node keeps its dead attempt in status.json. Adopting that row
    would resurrect a terminated box's id and make the slot look occupied."""
    doc = _status((0, "i-dead", "dead", 0), (0, "i-live", "alive", 1))
    adopted, logs = _adopt(monkeypatch, doc, {"i-live": "running", "i-dead": "terminated"})
    assert adopted == {0: "i-live"}
    assert logs[0]["attempt"] == 1


def test_adopt_rejects_a_box_aws_no_longer_runs(monkeypatch):
    """status.json can be a tick stale. Believing it over EC2 would adopt a
    ghost, and the slot would never be refilled."""
    doc = _status((0, "i-gone", "alive", 0))
    adopted, _ = _adopt(monkeypatch, doc, {"i-gone": "shutting-down"})
    assert adopted == {}


def test_cold_start_adopts_nothing(monkeypatch):
    """No status document => nothing to adopt => launch the whole group."""
    adopted, logs = _adopt(monkeypatch, None, {})
    assert adopted == {} and logs == {}
