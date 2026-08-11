"""Epoch-supervisor tests — the pure reducer (no AWS, no clock) plus the small
storage/schema helpers. The reducer owns all membership logic, so these tables
are the specification of how the fleet reacts to loss/join/kill/stall.
"""

from __future__ import annotations

from orchestrator.config import OrchestratorConfig
from orchestrator.supervisor import (
    Done,
    LaunchReplacement,
    NodeObs,
    Observation,
    Policy,
    PublishEpoch,
    TerminateNode,
    WholeGroupRestart,
    decide,
    epoch_doc,
    status_doc,
)
from spot_train import s3_store

SHRINK = Policy(replace_on_loss=False, recovery_timeout_s=600)
PREEMPT = Policy(replace_on_loss=True, recovery_timeout_s=600)


def _node(i, state="running", registered=True, log_age=None):
    return NodeObs(node=i, aws_state=state, registered=registered, log_age_s=log_age)


def _obs(
    nodes,
    *,
    epoch,
    members,
    node_count=None,
    metrics=False,
    no_progress=None,
    due=(),
    epochs_without_progress=0,
):
    return Observation(
        node_count=node_count if node_count is not None else len(nodes),
        nodes=tuple(nodes),
        epoch=epoch,
        members=frozenset(members),
        metrics_exists=metrics,
        no_progress_s=no_progress,
        due_kills=frozenset(due),
        epochs_without_progress=epochs_without_progress,
    )


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
def test_startup_waits_for_all_nodes():
    # Two nodes desired, only one running/registered -> publish nothing yet.
    obs = _obs([_node(0), _node(1, state="pending")], epoch=0, members=[], node_count=2)
    assert decide(obs, SHRINK) == []


def test_startup_publishes_epoch_1_when_all_healthy():
    obs = _obs([_node(0), _node(1)], epoch=0, members=[], node_count=2)
    assert decide(obs, SHRINK) == [PublishEpoch(1, (0, 1))]


# --------------------------------------------------------------------------- #
# Loss -> shrink
# --------------------------------------------------------------------------- #
def test_lost_member_shrinks_without_replacement_under_shrink_policy():
    # node 1 terminated; shrink policy republishes survivors only, no relaunch.
    obs = _obs(
        [_node(0), _node(1, state="terminated")],
        epoch=1,
        members=[0, 1],
        node_count=2,
    )
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]


def test_lost_member_shrinks_and_relaunches_under_preempt_policy():
    obs = _obs(
        [_node(0), _node(1, state="shutting-down")],
        epoch=1,
        members=[0, 1],
        node_count=2,
    )
    assert decide(obs, PREEMPT) == [PublishEpoch(2, (0,)), LaunchReplacement(1)]


def test_scheduled_kill_only_terminates_membership_unchanged():
    # due_kills=1: TERMINATE the box, but do NOT shrink yet — node 1 still reads
    # healthy (AWS lag), so membership is untouched. The shrink is observation-
    # driven and comes a tick or two later (next test).
    obs = _obs([_node(0), _node(1)], epoch=1, members=[0, 1], due=[1])
    assert decide(obs, SHRINK) == [TerminateNode(1)]
    assert decide(obs, PREEMPT) == [TerminateNode(1)]


def test_shrink_happens_when_kill_is_observed_next_tick():
    # After the terminate, once AWS shows node 1 gone (shutting-down), the same
    # reducer that handles a real reclaim shrinks — and, under preempt, replaces.
    # due is empty now (the shell dedups the already-issued kill).
    obs = _obs([_node(0), _node(1, state="shutting-down")], epoch=1, members=[0, 1])
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]
    assert decide(obs, PREEMPT) == [PublishEpoch(2, (0,)), LaunchReplacement(1)]


# --------------------------------------------------------------------------- #
# Join -> grow
# --------------------------------------------------------------------------- #
def test_replacement_registered_grows_group():
    # Running at world 1 (members={0}); node 1's replacement is now healthy and
    # not yet a member -> republish including it.
    obs = _obs([_node(0), _node(1)], epoch=2, members=[0], node_count=2)
    assert decide(obs, PREEMPT) == [PublishEpoch(3, (0, 1))]


def test_no_op_when_membership_matches_healthy():
    obs = _obs([_node(0), _node(1)], epoch=3, members=[0, 1])
    assert decide(obs, PREEMPT) == []


def test_pending_replacement_not_yet_admitted():
    # The replacement box is booting (pending) — not healthy, so no grow yet.
    obs = _obs([_node(0), _node(1, state="pending")], epoch=2, members=[0], node_count=2)
    assert decide(obs, PREEMPT) == []


# --------------------------------------------------------------------------- #
# Floors
# --------------------------------------------------------------------------- #
def test_metrics_exists_is_done():
    obs = _obs([_node(0)], epoch=5, members=[0], metrics=True)
    assert decide(obs, SHRINK) == [Done()]


def test_stall_triggers_whole_group_restart():
    obs = _obs([_node(0), _node(1)], epoch=2, members=[0, 1], no_progress=700)
    (act,) = decide(obs, PREEMPT)
    assert isinstance(act, WholeGroupRestart)
    # The reason must name the condition — this is the most destructive action
    # the supervisor takes, and a bare "floor" left a real incident undiagnosable.
    assert "no checkpoint progress" in act.reason and "700" in act.reason


def test_all_gone_triggers_whole_group_restart():
    obs = _obs(
        [_node(0, state="terminated"), _node(1, state="terminated")],
        epoch=2,
        members=[0, 1],
    )
    (act,) = decide(obs, PREEMPT)
    assert isinstance(act, WholeGroupRestart)
    assert "no healthy members" in act.reason


def test_stale_heartbeat_counts_as_lost():
    # node 1 still "running" per AWS but its log went silent > timeout -> wedged.
    obs = _obs(
        [_node(0, log_age=1.0), _node(1, log_age=999.0)],
        epoch=1,
        members=[0, 1],
        node_count=2,
    )
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]


def test_unregistered_node_is_not_healthy():
    obs = _obs([_node(0), _node(1, registered=False)], epoch=0, members=[], node_count=2)
    assert decide(obs, SHRINK) == []  # only 1 of 2 registered -> keep waiting


# --------------------------------------------------------------------------- #
# epoch_doc schema + config keys + storage
# --------------------------------------------------------------------------- #
def test_epoch_doc_ranks_and_master():
    doc = epoch_doc("r", 3, (0, 2), {0: "10.0.0.1", 2: "10.0.0.2"}, port_base=29400, master=0)
    assert doc == {
        "epoch": 3,
        "members": [
            {"node": 0, "ip": "10.0.0.1", "rank": 0},
            {"node": 2, "ip": "10.0.0.2", "rank": 1},
        ],
        "node_count": 2,
        "master_addr": "10.0.0.1",  # the elected master
        "master_port": 29403,  # base + epoch
    }


def test_epoch_doc_puts_master_at_rank_0():
    # A non-lowest master (a survivor) is placed at rank 0 — torchrun needs
    # master_addr to be rank 0's host; the rest keep sorted order after it.
    doc = epoch_doc(
        "r", 5, (0, 1, 2), {0: "10.0.0.0", 1: "10.0.0.1", 2: "10.0.0.2"}, port_base=29400, master=1
    )
    assert doc["members"] == [
        {"node": 1, "ip": "10.0.0.1", "rank": 0},  # master -> rank 0
        {"node": 0, "ip": "10.0.0.0", "rank": 1},
        {"node": 2, "ip": "10.0.0.2", "rank": 2},
    ]
    assert doc["master_addr"] == "10.0.0.1"


def test_elect_master_sticky_survivor():
    from orchestrator.supervisor import elect_master

    # Startup: no prior epoch -> lowest index.
    assert elect_master((0, 1, 2, 3), frozenset(), None) == 0
    # Master (0) dies, shrink to survivors -> lowest-index survivor becomes master.
    assert elect_master((1, 2, 3), frozenset({0, 1, 2, 3}), 0) == 1
    # Grow back: node0's replacement re-takes index 0, but the master is STICKY to
    # the survivor (1) — it does NOT migrate to the fresh box. (The bug fixed.)
    assert elect_master((0, 1, 2, 3), frozenset({1, 2, 3}), 1) == 1
    # A worker dies, master (0) survives -> master unchanged.
    assert elect_master((0, 1, 2), frozenset({0, 1, 2, 3}), 0) == 0
    # The sticky master itself dies -> migrate to the lowest-index survivor.
    assert elect_master((2, 3), frozenset({1, 2, 3}), 1) == 2
    # Whole group replaced (no survivor of the prior epoch) -> lowest index.
    assert elect_master((0, 1), frozenset({5, 6}), 5) == 0


def test_publish_epoch_keeps_master_sticky_across_grow_back(monkeypatch):
    # The end-to-end fix for the observed hang: master node0 killed, group shrinks
    # to [1,2,3] (master -> node1), then grows back to 4 with node0's replacement.
    # The rendezvous master must STAY node1 (rank 0) — never migrate to the fresh
    # node0·r1 — so the grow-back rendezvous doesn't deadlock on a not-yet-ready box.
    import json as _json

    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 4
    puts: dict[str, str] = {}
    monkeypatch.setattr(sup_mod.aws, "put_text", lambda b, k, t, **kw: puts.__setitem__(k, t))

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={i: f"i-{i}" for i in range(4)},
        logs={i: {"key": f"k{i}", "attempt": 0} for i in range(4)},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )
    s.st.ips = {0: "10.0.0.0", 1: "10.0.0.1", 2: "10.0.0.2", 3: "10.0.0.3"}

    def master_of() -> dict:
        return _json.loads(puts[cfg.run_epoch_key("r")])

    s._publish_epoch(1, (0, 1, 2, 3))  # startup
    assert s.st.master == 0 and master_of()["master_addr"] == "10.0.0.0"

    s._publish_epoch(2, (1, 2, 3))  # node0 (master) killed -> survivor node1 takes over
    assert s.st.master == 1 and master_of()["master_addr"] == "10.0.0.1"

    s._publish_epoch(3, (0, 1, 2, 3))  # node0·r1 rejoins -> master STAYS node1 (sticky)
    doc = master_of()
    assert s.st.master == 1 and doc["master_addr"] == "10.0.0.1"
    ranks = {m["node"]: m["rank"] for m in doc["members"]}
    assert ranks[1] == 0 and ranks[0] == 1  # survivor is rank 0; replacement a worker


def test_config_epoch_keys():
    cfg = OrchestratorConfig(bucket="b")
    assert cfg.run_epoch_key("r") == "runs/r/epoch.json"
    assert cfg.run_node_key("r", 2) == "runs/r/nodes/node2.json"
    assert cfg.run_nodes_prefix("r") == "runs/r/nodes/"
    assert cfg.run_uri("r") == "s3://b/runs/r"


def test_read_bytes_roundtrip_and_absent(tmp_path):
    uri = str(tmp_path / "doc.json")
    assert s3_store.read_bytes(uri) is None  # absent
    s3_store.put_bytes(b'{"epoch": 1}', uri)
    assert s3_store.read_bytes(uri) == b'{"epoch": 1}'


def test_run_node_uri_is_full_s3_uri():
    # Regression: the supervisor reads registrations via read_bytes, which treats
    # a prefix-less string as a LOCAL path — so it MUST be a full s3:// URI, or
    # every node reads as unregistered and epoch 1 never publishes.
    cfg = OrchestratorConfig(bucket="b")
    assert cfg.run_node_uri("r", 1) == "s3://b/runs/r/nodes/node1.json"


def test_observe_sees_registration_and_publishes_epoch_1(monkeypatch):
    # End-to-end shell test through the real Supervisor._observe: two running,
    # registered nodes must yield PublishEpoch(1). This exercises _node_ip's URI
    # construction (the layer the bare-key bug lived in) that the pure-reducer
    # tables above can't reach.
    from orchestrator import supervisor as sup_mod

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: "running")
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: -1)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    # Registrations present ONLY at the full s3:// node URIs — a bare key returns
    # None, which is exactly the bug this guards against.
    node_docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: node_docs.get(uri))

    from orchestrator.profile import RunProfile

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )
    obs = s._observe(now=0.0, wall=0.0)
    assert all(n.registered for n in obs.nodes)
    assert decide(obs, PREEMPT) == [PublishEpoch(1, (0, 1))]


def test_terminate_does_not_shortcut_membership(monkeypatch):
    # The heart of "observation-driven": after the supervisor terminates a node,
    # its health follows AWS state, NOT the fact that we killed it. While AWS
    # still lags at "running" the node stays healthy (no shrink); only once AWS
    # reports it gone does membership react.
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    state = {"i-0": "running", "i-1": "running"}
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: state[iid])
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: 5)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    monkeypatch.setattr(sup_mod.aws, "terminate", lambda iid: None)  # AWS lag: state unchanged
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-shrink", market="spot"),
        run_id="r",
        policy=SHRINK,
        node_ids={0: "i-0", 1: "i-1"},
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )
    s.st.epoch, s.st.members = 1, frozenset({0, 1})

    s._terminate(1)  # kill the box; AWS still shows it running (lag)
    obs = s._observe(now=1.0, wall=1.0)
    assert decide(obs, SHRINK) == []  # NOT shrunk — node 1 still observed healthy

    state["i-1"] = "shutting-down"  # AWS finally reflects the death
    obs = s._observe(now=2.0, wall=2.0)
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]  # now it reacts


# --------------------------------------------------------------------------- #
# status_doc — the observability document the `logs` viewer reads
# --------------------------------------------------------------------------- #
_ORCH_KEY = "runs/r/logs/orchestrator.log"


def _status(obs, *, members, logs, prev=None, now=42.0, done=False, ips=None, node_ids=None):
    return status_doc(
        "r",
        obs,
        SHRINK,
        epoch=obs.epoch,
        members=frozenset(members),
        ips=ips or {},
        node_ids=node_ids or {},
        logs=logs,
        orch_log_key=_ORCH_KEY,
        prev=prev,
        now=now,
        done=done,
    )


def _states(doc):
    return {(e["node"], e["attempt"]): e["state"] for e in doc["nodes"]}


_LOGS2 = {0: {"key": "k0", "attempt": 0}, 1: {"key": "k1", "attempt": 0}}


def test_status_doc_alive_dead_pending():
    # node 0 healthy, node 1 terminated, node 2 still booting (unregistered).
    obs = _obs(
        [_node(0), _node(1, state="terminated"), _node(2, registered=False)],
        epoch=2,
        members=[0, 1],
        node_count=3,
    )
    logs = {**_LOGS2, 2: {"key": "k2", "attempt": 0}}
    doc = _status(obs, members=[0], logs=logs, node_ids={0: "i-0", 1: "i-1", 2: "i-2"})
    assert _states(doc) == {(0, 0): "alive", (1, 0): "dead", (2, 0): "pending"}
    assert doc["updated_at"] == 42.0
    assert doc["members"] == [0]
    assert doc["orchestrator"] == {"log_key": _ORCH_KEY}
    assert doc["done"] is False
    by_node = {e["node"]: e for e in doc["nodes"]}
    assert by_node[1]["aws_state"] == "terminated" and by_node[1]["instance_id"] == "i-1"


def test_status_doc_stale_heartbeat_kills_previously_alive():
    # (1,0) was alive last tick; now AWS still says running but the log went
    # silent past the timeout -> dead. A never-alive stale entry stays pending.
    prev = _status(_obs([_node(0), _node(1)], epoch=1, members=[0, 1]), members=[0, 1], logs=_LOGS2)
    obs = _obs([_node(0, log_age=1.0), _node(1, log_age=999.0)], epoch=1, members=[0, 1])
    doc = _status(obs, members=[0, 1], logs=_LOGS2, prev=prev)
    assert _states(doc) == {(0, 0): "alive", (1, 0): "dead"}


def test_status_doc_dead_is_sticky():
    prev = _status(
        _obs([_node(0), _node(1, state="terminated")], epoch=2, members=[0, 1]),
        members=[0],
        logs=_LOGS2,
    )
    assert _states(prev)[(1, 0)] == "dead"
    # Even if the observation flips healthy again, (1,0) never resurrects.
    obs = _obs([_node(0), _node(1)], epoch=3, members=[0, 1])
    doc = _status(obs, members=[0, 1], logs=_LOGS2, prev=prev)
    assert _states(doc)[(1, 0)] == "dead"


def test_status_doc_replacement_carries_dead_attempt_forward():
    prev = _status(
        _obs([_node(0), _node(1, state="terminated")], epoch=2, members=[0, 1]),
        members=[0],
        logs=_LOGS2,
    )
    # The replacement booted: node 1 now maps to attempt 1 with a fresh log key.
    logs = {0: {"key": "k0", "attempt": 0}, 1: {"key": "k1-r1", "attempt": 1}}
    obs = _obs([_node(0), _node(1)], epoch=3, members=[0])
    doc = _status(obs, members=[0], logs=logs, prev=prev)
    assert _states(doc) == {(0, 0): "alive", (1, 0): "dead", (1, 1): "alive"}
    by_key = {(e["node"], e["attempt"]): e for e in doc["nodes"]}
    assert by_key[(1, 0)]["log_key"] == "k1"  # frozen entry keeps its own log
    assert by_key[(1, 1)]["log_key"] == "k1-r1"


def test_status_doc_done_flag():
    obs = _obs([_node(0)], epoch=5, members=[0], metrics=True)
    doc = _status(obs, members=[0], logs={0: {"key": "k0", "attempt": 0}}, done=True)
    assert doc["done"] is True


def test_supervisor_writes_status_each_tick_and_survives_failure(monkeypatch):
    # The shell hook: _write_status uploads status.json (+ orchestrator.log once
    # events accrued), and a failing put_text is swallowed — observability must
    # never kill the run.
    import json as _json

    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: "running")
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: -1)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs={
            0: {"key": "k0", "attempt": 0, "state": {"printed": 0}},
            1: {"key": "k1", "attempt": 0, "state": {"printed": 0}},
        },
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )

    # Tick 1: put_text raises -> swallowed, nothing cached.
    def boom(b, k, t):
        raise RuntimeError("s3 down")

    monkeypatch.setattr(sup_mod.aws, "put_text", boom)
    s._write_status(s._observe(now=0.0, wall=100.0), 100.0)
    assert s._last_status is None

    # Tick 2: healthy write; an _event makes orchestrator.log upload too.
    puts: dict[str, str] = {}
    monkeypatch.setattr(sup_mod.aws, "put_text", lambda b, k, t, **kw: puts.__setitem__(k, t))
    s._event("terminated node 1 (i-1)")
    s._write_status(s._observe(now=1.0, wall=101.0), 101.0)
    doc = _json.loads(puts[cfg.run_status_key("r")])
    assert doc["updated_at"] == 101.0
    assert _states(doc) == {(0, 0): "alive", (1, 0): "alive"}
    assert "terminated node 1 (i-1)" in puts[cfg.run_orch_log_key("r")]
    assert s._last_status == doc

    # Tick 3: no new events -> status re-uploaded, orchestrator.log NOT.
    puts.clear()
    s._write_status(s._observe(now=2.0, wall=102.0), 102.0)
    assert cfg.run_status_key("r") in puts and cfg.run_orch_log_key("r") not in puts


def test_replacement_attempt_is_not_born_dead(monkeypatch):
    # Regression: on the tick that launches a replacement, _execute bumps node 1
    # to attempt 1 with a fresh instance, but the tick's `obs` still holds the
    # OLD terminated instance. If status is written from that stale obs paired
    # with the new logs, node1·r1 is stamped dead the instant it appears and
    # sticky-dead locks it forever (the box shows [DEAD] while training at ws 2).
    # Writing status BEFORE _execute keeps obs and logs consistent.
    import json as _json

    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    aws_state = {"i-0": "running", "i-1": "terminated"}  # node 1 just reclaimed
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: aws_state.get(iid, "running"))
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: 5)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))
    puts: dict[str, str] = {}
    monkeypatch.setattr(sup_mod.aws, "put_text", lambda b, k, t, **kw: puts.__setitem__(k, t))

    logs = {
        0: {"key": "k0", "attempt": 0, "state": {"printed": 0}},
        1: {"key": "k1", "attempt": 0, "state": {"printed": 0}},
    }

    def launch(node):  # a replacement: bump attempt + fresh instance, as the real one does
        logs[node] = {"key": f"k{node}-r1", "attempt": 1, "state": {"printed": 0}}
        aws_state["i-1r"] = "running"
        return "i-1r"

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs=logs,
        launch_node=launch,
        pull_logs=lambda: None,
    )
    s.st.epoch, s.st.members = 1, frozenset({0, 1})
    s.st.ips = {0: "10.0.0.0", 1: "10.0.0.1"}

    # Reproduce ONE loop iteration in the fixed order: observe -> write -> execute.
    obs = s._observe(now=1.0, wall=1.0)
    actions = decide(obs, PREEMPT)
    assert LaunchReplacement(1) in actions  # node 1 observed gone -> replace
    s._write_status(obs, 1.0)  # status written from the consistent (obs, logs) pair
    s._execute(actions)  # only now does logs[1] become attempt 1 + i-1r

    doc = _json.loads(puts[cfg.run_status_key("r")])
    st = {(e["node"], e["attempt"]): e["state"] for e in doc["nodes"]}
    assert st[(1, 0)] == "dead"  # the reclaimed attempt: correctly dead
    assert (1, 1) not in st  # the replacement is NOT yet present, so never born dead

    # Next tick: the replacement is observed running -> it surfaces alive.
    s.st.ips[1] = "10.0.0.1"  # replacement re-registers
    obs2 = s._observe(now=2.0, wall=2.0)
    s._write_status(obs2, 2.0)
    doc2 = _json.loads(puts[cfg.run_status_key("r")])
    st2 = {(e["node"], e["attempt"]): e["state"] for e in doc2["nodes"]}
    assert st2[(1, 1)] == "alive"  # born alive, not dead
    assert st2[(1, 0)] == "dead"  # predecessor carried forward, frozen


def test_supervisor_emits_parseable_lifecycle_events():
    # The orchestrator half of the event-sourced timeline: killed/down/epoch land
    # in orchestrator.log as structured [event] records the viewer parses.
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile
    from spot_train import events as ev

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs={0: {"key": "k0", "attempt": 0}, 1: {"key": "k1", "attempt": 0}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )
    s._emit_event("epoch", epoch=2, world=1)
    s._emit_event("killed", node=1, cause="scheduled-kill")
    s._emit_event("down", node=0, cause="reclaimed")

    recs = ev.parse("\n".join(s._orch_lines))
    by_state = {r["state"]: r for r in recs}
    assert set(by_state) == {"epoch", "killed", "down"}
    assert by_state["epoch"]["world"] == 1 and by_state["epoch"]["by"] == "orch"
    assert by_state["killed"]["node"] == 1 and by_state["killed"]["cause"] == "scheduled-kill"
    assert by_state["down"]["node"] == 0 and by_state["down"]["cause"] == "reclaimed"
    assert s._orch_dirty is True  # flagged for the next status upload


def test_scheduled_kill_fires_exactly_once_even_after_replacement(monkeypatch):
    # Regression: the kill schedule is level-triggered on "elapsed >= secs", so
    # without per-ENTRY edge-triggering it re-fires every tick after the due
    # time — and re-kills each replacement the instant it rejoins (an infinite
    # kill loop, observed on AWS as epochs 5,6,7,8,... churning).
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: "running")
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: 5)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
        kill_schedule=[(100.0, 1)],  # one kill, at 100s after train start
    )
    s._train_start = 0.0

    # Well past the due time: the kill fires on the first observe...
    assert 1 in s._observe(now=200.0, wall=0.0).due_kills
    # ...and NEVER again, even though elapsed is still >> 100 and node 1 has been
    # "replaced" (fresh instance id, as _launch_replacement would set).
    s.node_ids[1] = "i-1-replacement"
    for t in (210.0, 220.0, 300.0):
        assert s._observe(now=t, wall=0.0).due_kills == frozenset()


# --------------------------------------------------------------------------- #
# Stall detection must not fire DURING a legitimate recovery
# --------------------------------------------------------------------------- #
def test_recovery_in_progress_is_not_a_stall():
    """The bug this pins cost a live 8-node run.

    A replacement has to boot AND pull the dataset (~4-5 min for a 17 GB
    train.bin) before it can take one step, so a freshly published epoch shows
    no checkpoint progress for minutes. That is recovery working, not a wedged
    group — and the shell restarts the clock on every publication so the new
    world is judged on its OWN elapsed time. Below the timeout, keep waiting.
    """
    obs = _obs(
        [_node(0), _node(1)],
        epoch=2,
        members=[0, 1],
        no_progress=120.0,  # well past the OLD 150s-era margin, still recovering
        epochs_without_progress=1,
    )
    assert decide(obs, PREEMPT) == []


def test_genuine_stall_still_breaks_the_deadlock():
    # Past the timeout with the world nominally healthy: nothing is coming back.
    obs = _obs([_node(0), _node(1)], epoch=2, members=[0, 1], no_progress=601.0)
    (act,) = decide(obs, PREEMPT)
    assert isinstance(act, WholeGroupRestart)
    assert "no checkpoint progress" in act.reason


def test_flapping_world_restarts_even_though_each_epoch_resets_the_clock():
    """Resetting the stall clock per epoch would make the deadlock-breaker
    unreachable for a world that re-forms endlessly without ever training —
    each publication would hand it a fresh budget forever. The
    epochs-without-progress counter is what closes that hole."""
    obs = _obs(
        [_node(0), _node(1)],
        epoch=7,
        members=[0, 1],
        no_progress=5.0,  # clock just reset by the newest epoch
        epochs_without_progress=6,  # ...but six worlds in a row never trained
    )
    (act,) = decide(obs, PREEMPT)
    assert isinstance(act, WholeGroupRestart)
    assert "epochs published with no checkpoint" in act.reason


def test_progress_clears_the_flap_counter_path():
    # Same epoch count but progress happened (counter cleared by the shell).
    obs = _obs(
        [_node(0), _node(1)],
        epoch=7,
        members=[0, 1],
        no_progress=5.0,
        epochs_without_progress=0,
    )
    assert decide(obs, PREEMPT) == []


def test_two_normal_preemptions_do_not_exhaust_the_restart_budget():
    """One preemption publishes TWO epochs — shrink onto survivors, then grow
    when the replacement joins. With the budget at 3, ~1.5 textbook recoveries
    spent it, so the mechanism working correctly could trigger the most
    destructive action available (discard every healthy survivor and relaunch).
    The budget is counted in preemptions now, not epochs."""
    p = Policy(replace_on_loss=True, recovery_timeout_s=600)
    # Two preemptions = 4 epoch publications with no checkpoint in between.
    obs = _obs(
        [_node(0), _node(1)],
        epoch=5,
        members=[0, 1],
        no_progress=30.0,
        epochs_without_progress=4,
    )
    assert decide(obs, p) == [], "two normal recoveries must not trip the floor"
    # A third full preemption (6 epochs) is genuine evidence the world is stuck.
    stuck = _obs(
        [_node(0), _node(1)],
        epoch=7,
        members=[0, 1],
        no_progress=30.0,
        epochs_without_progress=6,
    )
    (act,) = decide(stuck, p)
    assert isinstance(act, WholeGroupRestart)
    assert "epochs published with no checkpoint" in act.reason


def test_terminated_node_stops_looking_healthy_immediately():
    """The exact race that wasted 642 node-seconds per failure.

    A node terminated ~26s earlier still passed every check in `_healthy`: its
    cached IP kept `registered` true (st.ips is keyed by node INDEX, so the slot
    kept its dead occupant's address), EC2 briefly still reported `running`, and
    its log age was under the 90s heartbeat timeout. So the reducer regrew the
    world onto a corpse at t+448s and then idled until the replacement could
    actually train at t+662s.
    """
    # A corpse that still looks alive on every axis EXCEPT registration.
    corpse = _node(1, state="running", registered=False, log_age=26.0)
    obs = _obs([_node(0), corpse], epoch=1, members=[0, 1], node_count=2)
    acts = decide(obs, PREEMPT)
    # Shrinks onto the survivor and asks for a replacement — it must NOT keep
    # node 1 in the membership just because AWS and the log still look fine.
    assert PublishEpoch(2, (0,)) in acts
    assert LaunchReplacement(1) in acts


def test_regrow_waits_for_the_replacement_to_announce_itself():
    # World shrunk to {0}; node 1's replacement is booting but has not
    # registered (it is still pulling the corpus). No regrow yet.
    booting = _node(1, state="running", registered=False)
    assert decide(_obs([_node(0), booting], epoch=2, members=[0], node_count=2), PREEMPT) == []
    # Once it registers — which now happens only AFTER its dataset is local —
    # the world grows back.
    ready = _node(1, state="running", registered=True)
    assert decide(_obs([_node(0), ready], epoch=2, members=[0], node_count=2), PREEMPT) == [
        PublishEpoch(3, (0, 1))
    ]


def test_dead_nodes_registration_does_not_outlive_it(monkeypatch):
    """The bug that cost 155s of survivor idle time per failure.

    nodes/node<i>.json is keyed by node INDEX and persists in S3, so a killed
    node's registration is still sitting there after it dies. _node_ip falls
    through to S3 on a cache miss, so the slot read as registered again within
    one tick — and _healthy passed it (EC2 still said running, log seconds old).
    The reducer then published a 4-member epoch while the replacement had not
    begun booting, and every survivor blocked in init_process_group waiting for
    a rank that did not exist.

    A registration only counts when the box that wrote it is the box now in the
    slot. The doc carries instance_id; the supervisor knows node_ids[node].
    """
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    # node 1's doc was written by i-OLD, which we have since replaced with i-NEW.
    node_docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0", "instance_id": "i-0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1", "instance_id": "i-OLD"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: node_docs.get(uri))
    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-NEW"},  # slot 1 now belongs to the replacement
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-NEW",
        pull_logs=lambda: None,
    )
    assert s._node_ip(0) == "10.0.0.0", "the live node stays registered"
    assert s._node_ip(1) is None, "the DEAD occupant's registration must not count"

    # ...and once the replacement writes its OWN doc, the slot registers again.
    node_docs[cfg.run_node_uri("r", 1)] = b'{"ip": "10.0.0.9", "instance_id": "i-NEW"}'
    assert s._node_ip(1) == "10.0.0.9"


def test_registration_trusted_when_instance_id_is_unavailable(monkeypatch):
    """IMDS is absent in the localhost E2E, where register() writes "unknown".
    Refusing those would make the harness never form a world at all."""
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    monkeypatch.setattr(
        sup_mod.s3_store,
        "read_bytes",
        lambda uri: b'{"ip": "127.0.0.1", "instance_id": "unknown"}',
    )
    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0"},
        logs={0: {"key": "k0", "state": {"printed": 0}}},
        launch_node=lambda n: "i-x",
        pull_logs=lambda: None,
    )
    assert s._node_ip(0) == "127.0.0.1"


def test_replacement_launches_without_waiting_for_the_corpse(monkeypatch):
    """A replacement must not queue behind the dead instance's quota release.

    _launch_replacement used to call wait_quota_released first: terminate, then
    poll every 5s until AWS moved the box out of shutting-down (tens of seconds),
    THEN launch — unconditionally, even with a mostly-idle quota. Measured on a
    4-node run: 20 of a 64 vCPU quota in use, so it never had to wait at all.

    wait_vcpu_headroom already covers the case it guarded — it returns at once
    when used+needed fits and blocks only when it does not, which is precisely
    when the corpse releasing is what creates room.
    """
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    calls = []
    monkeypatch.setattr(sup_mod.aws, "wait_quota_released", lambda i: calls.append(("quota", i)))
    monkeypatch.setattr(sup_mod.aws, "wait_vcpu_headroom", lambda n, q: calls.append(("head", n)))
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: None)

    s = sup_mod.Supervisor(
        OrchestratorConfig(bucket="b"),
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-dead"},
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )
    s._launch_replacement(1)
    assert ("quota", "i-dead") not in calls, "must not serialize on the dead instance"
    assert any(c[0] == "head" for c in calls), "headroom check must still gate the launch"
    assert s.node_ids[1] == "i-new"
