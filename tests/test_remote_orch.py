"""Remote-orchestrator tests — hermetic (no AWS: every call is monkeypatched or
dry-run, and the generated shell is only parsed with `bash -n`).

Pins the properties a 36-hour unattended run depends on:

  - the control-plane user-data clones the RIGHT branch, carries the run env,
    runs under systemd with restart-on-failure, relays its log to S3, and is
    EXEMPT from the training boxes' dead-man's switch;
  - the boot-progress state machine walks pending -> running -> markers ->
    heartbeat -> attach and never rewinds;
  - `orch status` extracts its fields from the published documents (pure);
  - `orch down` is a safe no-op with nothing up, and warns before it kills;
  - attach DELEGATES to logview.run_logs rather than duplicating the viewer.

Also covers the dataset-directory resolution shared by every training/fleet
boot: instance-store NVMe when the box has one, root volume when it doesn't.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from orchestrator import bootstrap, orch
from orchestrator.config import OrchestratorConfig


def _cfg(**kw) -> OrchestratorConfig:
    return OrchestratorConfig(bucket="test-bucket", **kw)


def _orch_ud(cfg: OrchestratorConfig | None = None, **kw) -> str:
    cfg = cfg or _cfg(repo_branch="phase1/remote-orch")
    env = kw.pop("env", {"NODES": "8", "BASELINE_SECONDS": "129600"})
    return bootstrap.build_orchestrator_user_data(
        cfg, orch_id="orch-20260802-101500", experiment="multinode", env=env, **kw
    )


# --------------------------------------------------------------------------- #
# user-data for the control plane
# --------------------------------------------------------------------------- #
def test_orchestrator_user_data_bash_syntax(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    path = tmp_path / "orch.sh"
    path.write_text(_orch_ud())
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed:\n{r.stderr}"


def test_orchestrator_user_data_clones_the_configured_branch():
    # The branch matters: boxes this orchestrator launches clone the same one,
    # so a feature-branch run must not silently boot main's code.
    ud = _orch_ud()
    assert "git clone --depth 1 -b phase1/remote-orch" in ud
    assert "-b main " not in ud


def test_orchestrator_user_data_carries_the_run_env():
    ud = _orch_ud(env={"NODES": "8", "BASELINE_SECONDS": "129600", "DATASET": "openwebtext"})
    assert 'NODES="8"' in ud
    assert 'BASELINE_SECONDS="129600"' in ud
    assert 'DATASET="openwebtext"' in ud
    assert "EnvironmentFile=/home/ec2-user/orch.env" in ud


def test_orchestrator_user_data_runs_under_systemd_with_restart():
    ud = _orch_ud()
    assert "/etc/systemd/system/spot-orch.service" in ud
    assert "systemctl enable --now spot-orch.service" in ud
    # Survives SSH/user-data exit AND restarts on crash — the agent then resumes
    # the same run_id rather than starting a fresh run.
    assert "Restart=on-failure" in ud
    assert "RestartSec=30" in ud
    assert "orch _agent --orch-id orch-20260802-101500 --experiment multinode" in ud


def test_orchestrator_user_data_relays_logs_with_a_size_cap():
    cfg = _cfg(orch_log_max_bytes=1234, orch_log_upload_seconds=7)
    ud = _orch_ud(cfg)
    # Same relay mechanism as the training boxes (boto3 upload_file on a timer)…
    assert "upload_file" in ud
    assert "orchestrators/orch-20260802-101500/orchestrator.log" in ud
    assert "orchestrators/orch-20260802-101500/boot.log" in ud
    assert "spot-orch-relay.service" in ud
    # …plus the cap that keeps 36h of stdout off an 8GB root volume.
    assert "CAP = 1234" in ud
    assert "time.sleep(7)" in ud


def test_orchestrator_user_data_writes_boot_progress_markers():
    ud = _orch_ud()
    assert "orchestrators/orch-20260802-101500/progress.json" in ud
    for phase in ("provision", "clone", "install", "start"):
        assert f"mark {phase} " in ud, f"missing boot marker for {phase}"


def test_orchestrator_is_exempt_from_the_training_dead_mans_switch():
    # MAX_INSTANCE_LIFETIME_SECONDS exists to reap orphaned TRAINING boxes when
    # the orchestrator dies; applying it here would kill the reaper mid-run.
    cfg = _cfg(max_instance_lifetime_seconds=3600)
    ud = _orch_ud(cfg)
    assert "autokill" not in ud
    assert "poweroff" not in ud
    # Training boxes with the same cfg DO get it — the exemption is deliberate,
    # not an accident of the knob being unset.
    train = bootstrap.build_user_data(cfg, run_id="r", market="spot", max_seconds=60)
    assert "spot-autokill" in train


def test_orchestrator_lifetime_is_its_own_opt_in_knob():
    cfg = _cfg(orch_max_lifetime_seconds=200000)  # comfortably over a 36h run
    ud = _orch_ud(cfg)
    assert "systemd-run --on-active=200000s --unit=orch-autokill" in ud


def test_orchestrator_user_data_carries_no_credentials():
    # The box must use its instance-profile role: copied keys/session tokens
    # expire long before a 36h run ends (and user-data is readable via IMDS).
    ud = _orch_ud(env={"AWS_ACCESS_KEY_ID": "AKIA-nope", "WANDB_API_KEY": "secret"})
    for marker in ("AKIA-nope", "secret", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        assert marker not in ud


def test_orchestrator_user_data_installs_no_torch():
    # A control plane never trains; pulling the real dependency set would drag
    # ~2GB of CUDA torch onto an 8GB root volume.
    ud = _orch_ud()
    assert "--no-deps -e /home/ec2-user/app" in ud
    assert "pip install torch" not in ud


# --------------------------------------------------------------------------- #
# env relay (allowlist + secret filter + budget alias)
# --------------------------------------------------------------------------- #
def test_relay_env_inherits_allowlisted_knobs_and_drops_secrets(monkeypatch):
    monkeypatch.setenv("NODES", "4")
    monkeypatch.setenv("MAX_STEPS", "5000")  # trainer passthrough
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "nope")
    monkeypatch.setenv("WANDB_API_KEY", "nope")
    monkeypatch.setenv("SOME_RANDOM_VAR", "nope")
    env = _cfg().orch_relay_env({"NODES": "8"})  # --env wins over the shell
    assert env["NODES"] == "8"
    assert env["MAX_STEPS"] == "5000"
    for k in ("AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY", "SOME_RANDOM_VAR"):
        assert k not in env
    assert _cfg().orch_secretish({"WANDB_API_KEY": "x", "NODES": "8"}) == ["WANDB_API_KEY"]


def test_budget_alias_maps_to_the_knob_each_experiment_reads():
    assert orch._budget_env_key("multinode") == "BASELINE_SECONDS"
    assert orch._budget_env_key("multinode-preempt") == "TRAIN_TOTAL_SECONDS"
    cfg = _cfg(baseline_seconds=129600, train_total_seconds=180)
    assert orch._budget_seconds(cfg, "multinode") == 129600
    assert orch._budget_seconds(cfg, "multinode-preempt") == 180
    # The budget recorded for the run is the one the CONTROL PLANE will boot
    # with, not this laptop's config.
    assert orch._budget_seconds(cfg, "multinode", {"BASELINE_SECONDS": "7200"}) == 7200


# --------------------------------------------------------------------------- #
# boot progress state machine
# --------------------------------------------------------------------------- #
def test_boot_progress_walks_the_whole_sequence():
    p = orch.BootProgress()
    assert p.observe(state="pending", progress=None, heartbeat=None, run_ready=False)
    assert p.phase == "pending" and not p.ready
    # Same observation twice emits nothing new (no line spam while polling).
    assert p.observe(state="pending", progress=None, heartbeat=None, run_ready=False) == []
    assert p.observe(state="running", progress=None, heartbeat=None, run_ready=False)
    assert p.phase == "running"
    for phase in ("provision", "clone", "install", "start"):
        lines = p.observe(
            state="running",
            progress={"phase": phase, "detail": "d"},
            heartbeat=None,
            run_ready=False,
        )
        assert lines and phase in lines[0]
        assert p.phase == phase
    # Heartbeat with a run_id, but nothing to view yet -> still not attaching.
    p.observe(
        state="running",
        progress={"phase": "start", "detail": ""},
        heartbeat={"run_id": "multinode-1"},
        run_ready=False,
    )
    assert p.phase == "live" and p.run_id == "multinode-1" and not p.ready
    p.observe(
        state="running",
        progress={"phase": "start", "detail": ""},
        heartbeat={"run_id": "multinode-1"},
        run_ready=True,
    )
    assert p.ready and not p.failed


def test_boot_progress_never_rewinds():
    p = orch.BootProgress()
    p.observe(state="running", progress={"phase": "install"}, heartbeat=None, run_ready=False)
    assert p.phase == "install"
    # A stale/again-earlier marker must not move the display backwards.
    assert (
        p.observe(state="running", progress={"phase": "clone"}, heartbeat=None, run_ready=False)
        == []
    )
    assert p.phase == "install"


def test_boot_progress_detects_a_dead_box():
    p = orch.BootProgress()
    p.observe(state="pending", progress=None, heartbeat=None, run_ready=False)
    p.observe(state="terminated", progress=None, heartbeat=None, run_ready=False)
    assert p.failed and "terminated" in p.failed
    assert not p.ready


def test_boot_progress_surfaces_a_provisioning_failure():
    # user-data writes an `error` marker (e.g. the clone failed) so the laptop
    # stops waiting instead of timing out 30 minutes later.
    p = orch.BootProgress()
    p.observe(state="running", progress={"phase": "provision"}, heartbeat=None, run_ready=False)
    p.observe(
        state="running",
        progress={"phase": "error", "detail": "clone of phase1/x failed"},
        heartbeat=None,
        run_ready=False,
    )
    assert "clone of phase1/x failed" in p.failed
    assert not p.ready


def test_boot_progress_tolerates_unknown_markers():
    p = orch.BootProgress()
    p.observe(state="running", progress={"phase": "who-knows"}, heartbeat=None, run_ready=False)
    assert p.phase == "running"  # unknown phase ignored, not crashed on


# --------------------------------------------------------------------------- #
# status field extraction (pure)
# --------------------------------------------------------------------------- #
def _status_docs(now: float):
    state = {
        "orch_id": "orch-1",
        "experiment": "multinode",
        "instance_id": "i-abc",
        "instance_type": "t3.micro",
        "hourly_usd": 0.0104,
        "created_at": now - 3600,
        "budget_s": 129600,
        "run_id": "multinode-1",
    }
    heartbeat = {
        "at": now - 4,
        "attempt": 1,
        "pid": 3412,
        "experiment": "multinode",
        "run_id": "multinode-1",
        "step": 12480,
        "loss": 3.214,
        "world_size": 7,
        "elapsed_s": 3600,
        "budget_s": 129600,
        "cost_usd": 18.42,
        "cost_control_plane_usd": 0.0104,
        "error": "",
    }
    run_status = {"epoch": 3, "members": [0, 1, 2, 3, 4, 5, 6], "ckpt_step": 12400, "done": False}
    return state, heartbeat, run_status


def test_format_status_reports_every_field():
    now = 1_800_000_000.0
    state, heartbeat, run_status = _status_docs(now)
    out = "\n".join(
        orch.format_status(
            orch_id="orch-1",
            instance={"id": "i-abc", "type": "t3.micro", "state": "running"},
            state=state,
            heartbeat=heartbeat,
            run_status=run_status,
            now=now,
        )
    )
    assert "i-abc" in out and "t3.micro" in out and "[running]" in out
    assert "heartbeat 4s ago (healthy)" in out
    assert "attempt 1" in out and "pid 3412" in out
    assert "experiment=multinode" in out and "run=multinode-1" in out
    assert "epoch 3" in out and "world 7" in out and "ckpt step 12400" in out
    assert "step 12480" in out and "3.214" in out
    assert "1h00m of 36h00m budget" in out
    assert "$18.42" in out  # includes the control plane's own row
    assert "orch logs" in out


def test_format_status_says_when_the_run_is_over():
    now = 1_800_000_000.0
    state, heartbeat, run_status = _status_docs(now)
    heartbeat["done"] = True
    out = "\n".join(
        orch.format_status(
            orch_id="orch-1",
            instance={"id": "i-abc", "type": "t3.micro", "state": "running"},
            state=state,
            heartbeat=heartbeat,
            run_status=run_status,
            now=now,
        )
    )
    # The box outlives the run on purpose; nothing else stops the meter.
    assert "RUN FINISHED" in out and "orch down --all" in out


def test_format_status_flags_a_stale_heartbeat():
    now = 1_800_000_000.0
    state, heartbeat, run_status = _status_docs(now)
    heartbeat["at"] = now - 600
    out = "\n".join(
        orch.format_status(
            orch_id="orch-1",
            instance={"id": "i-abc", "type": "t3.micro", "state": "running"},
            state=state,
            heartbeat=heartbeat,
            run_status=run_status,
            now=now,
            stale_after=60,
        )
    )
    assert "STALE" in out


def test_format_status_with_nothing_up_and_before_the_first_heartbeat():
    now = 1_800_000_000.0
    assert (
        "no control plane found"
        in orch.format_status(
            orch_id="", instance=None, state=None, heartbeat=None, run_status=None, now=now
        )[0]
    )
    state, _, _ = _status_docs(now)
    state["run_id"] = ""
    out = "\n".join(
        orch.format_status(
            orch_id="orch-1",
            instance={"id": "i-abc", "type": "t3.micro", "state": "pending"},
            state=state,
            heartbeat=None,
            run_status=None,
            now=now,
        )
    )
    assert "heartbeat: none yet" in out
    assert "run=(not started)" in out
    # Cost still accrues before the agent starts — bill the box from launch.
    assert "control plane only" in out


# --------------------------------------------------------------------------- #
# down / logs / attach — driven with fake AWS responses
# --------------------------------------------------------------------------- #
class FakeAws:
    """Stand-in for orchestrator.aws: records mutations, serves canned docs."""

    def __init__(self, instances=None, docs=None):
        self.instances = instances or {}
        self.docs = docs or {}
        self.terminated: list[str] = []
        self.launched: list[dict] = []
        self.puts: dict[str, str] = {}
        self.dry = False

    def set_region(self, region):  # noqa: D102
        pass

    def is_dry_run(self):
        return self.dry

    def instances_by_tag(self, key, value):
        return list(self.instances.get((key, value), []))

    def get_json(self, bucket, key):
        return self.docs.get(key)

    def object_exists(self, bucket, key):
        return key in self.docs

    def list_keys(self, bucket, prefix):
        return [k for k in self.docs if k.startswith(prefix)]

    def terminate(self, iid):
        self.terminated.append(iid)

    def put_text(self, bucket, key, body, quiet=False):
        self.puts[key] = body

    def resolve_ami(self, ami_id, name_filter):
        return ami_id or "ami-FAKE"

    def ensure_security_group(self, name, region):
        return "sg-FAKE"

    def launch(self, **kw):
        self.launched.append(kw)
        return "i-NEW"

    def instance_state(self, iid):
        return "running"

    def wait_quota_released(self, iid):
        pass


def _controller(orch_id="orch-1", iid="i-orch", state="running"):
    return {
        "id": iid,
        "state": state,
        "type": "t3.micro",
        "public_ip": "",
        "private_ip": "",
        "tags": {orch.ORCH_TAG: orch_id, orch.ORCH_ROLE_TAG: orch.CONTROLLER},
    }


def _node(iid, run_id):
    return {
        "id": iid,
        "state": "running",
        "type": "g5.xlarge",
        "public_ip": "",
        "private_ip": "",
        "tags": {"Name": f"spot-train-{run_id}"},
    }


def test_down_with_nothing_up_is_a_safe_noop(monkeypatch, capsys):
    fake = FakeAws()
    monkeypatch.setattr(orch, "aws", fake)
    orch.down(_cfg(), yes=True)
    assert fake.terminated == []
    assert "nothing to terminate" in capsys.readouterr().out


def test_down_prints_the_plan_before_killing(monkeypatch, capsys):
    cfg = _cfg()
    fake = FakeAws(
        instances={
            (orch.ORCH_ROLE_TAG, orch.CONTROLLER): [_controller()],
            ("Name", "spot-train-multinode-1"): [_node("i-n0", "multinode-1")],
        },
        docs={cfg.orch_heartbeat_key("orch-1"): {"run_id": "multinode-1"}},
    )
    monkeypatch.setattr(orch, "aws", fake)
    orch.down(cfg, yes=True)
    out = capsys.readouterr()
    # The plan is printed, and the control plane dies…
    assert "plan:" in out.out
    assert out.out.index("plan:") < out.out.index("terminated 1")
    assert fake.terminated == ["i-orch"]
    # …but an orphaned fleet is a loud warning, not a silent bill.
    assert "NOT terminating 1 training box" in out.err


def test_down_all_takes_the_fleet_too(monkeypatch, capsys):
    cfg = _cfg()
    fake = FakeAws(
        instances={
            (orch.ORCH_ROLE_TAG, orch.CONTROLLER): [_controller()],
            ("Name", "spot-train-multinode-1"): [
                _node("i-n0", "multinode-1"),
                _node("i-n1", "multinode-1"),
            ],
        },
        docs={cfg.orch_heartbeat_key("orch-1"): {"run_id": "multinode-1"}},
    )
    monkeypatch.setattr(orch, "aws", fake)
    orch.down(cfg, all_=True, yes=True)
    assert fake.terminated == ["i-orch", "i-n0", "i-n1"]
    assert "training box  i-n0" in capsys.readouterr().out


def test_logs_resolves_the_active_run_and_delegates_to_the_viewer(monkeypatch):
    cfg = _cfg()
    fake = FakeAws(
        instances={(orch.ORCH_ROLE_TAG, orch.CONTROLLER): [_controller()]},
        docs={cfg.orch_heartbeat_key("orch-1"): {"run_id": "multinode-1"}},
    )
    monkeypatch.setattr(orch, "aws", fake)
    seen = {}

    def fake_run_logs(cfg_, run_id, **kw):
        seen["run_id"] = run_id
        seen["kw"] = kw

    from orchestrator import logview

    monkeypatch.setattr(logview, "run_logs", fake_run_logs)
    orch.logs(cfg, node=2, grid=True)
    # The viewer is the SHARED entry point (`spot-orchestrate logs`), not a copy.
    assert seen["run_id"] == "multinode-1"
    assert seen["kw"]["node"] == 2 and seen["kw"]["grid"] is True


def test_logs_without_an_active_run_explains_itself(monkeypatch):
    fake = FakeAws()
    monkeypatch.setattr(orch, "aws", fake)
    with pytest.raises(SystemExit, match="no active run"):
        orch.logs(_cfg())


def test_attach_ctrl_c_only_detaches(monkeypatch, capsys):
    from orchestrator import logview

    def boom(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(logview, "run_logs", boom)
    orch.attach(_cfg(), "multinode-1", orch_id="orch-1", instance_id="i-orch")
    err = capsys.readouterr().err
    # No exception escapes, and the user is told nothing was killed + how back in.
    assert "detached" in err
    assert "orch logs" in err
    assert "orch down --all" in err
    assert "only reads S3" in err


def test_up_dry_run_launches_nothing_and_skips_attach(monkeypatch, capsys):
    cfg = _cfg(repo_branch="phase1/remote-orch")
    fake = FakeAws()
    fake.dry = True
    monkeypatch.setattr(orch, "aws", fake)
    monkeypatch.setattr(orch, "attach", lambda *a, **kw: pytest.fail("dry-run must not attach"))
    orch_id = orch.up(cfg, experiment="multinode", env_overrides={"NODES": "8"})
    err = capsys.readouterr().err
    assert orch_id.startswith("orch-")
    assert "env NODES=8" in err
    assert "dry-run: skipping boot wait + attach" in err
    # The launch went through the (dry-run) aws module with the control plane's
    # own profile, on-demand, never spot.
    assert len(fake.launched) == 1
    assert fake.launched[0]["market"] == "on-demand"
    assert fake.launched[0]["profile_name"] == cfg.orch_instance_profile
    assert fake.launched[0]["instance_type"] == cfg.orch_instance_type


def test_up_rejects_an_unknown_experiment(monkeypatch):
    monkeypatch.setattr(orch, "aws", FakeAws())
    with pytest.raises(SystemExit, match="unknown experiment"):
        orch.up(_cfg(), experiment="train-a-dragon")


def test_up_translates_the_budget_knob(monkeypatch, capsys):
    fake = FakeAws()
    fake.dry = True
    monkeypatch.setattr(orch, "aws", fake)
    orch.up(
        _cfg(),
        experiment="multinode",
        env_overrides={"TRAIN_BUDGET_SECONDS": "129600"},
        no_attach=True,
    )
    err = capsys.readouterr().err
    assert "TRAIN_BUDGET_SECONDS=129600 -> BASELINE_SECONDS" in err
    assert "env BASELINE_SECONDS=129600" in err


def test_state_seed_lets_status_work_before_the_agent_starts(monkeypatch):
    cfg = _cfg()
    # Staged dataset present: `up` preflights it so the box doesn't fail 3
    # minutes into a boot.
    fake = FakeAws(docs={f"{cfg.data_prefix}/{cfg.dataset}/train.bin": {}})
    monkeypatch.setattr(orch, "aws", fake)
    monkeypatch.setattr(orch, "_wait_for_boot", lambda *a, **kw: orch.BootProgress())
    monkeypatch.setattr(orch, "attach", lambda *a, **kw: None)
    orch_id = orch.up(cfg, experiment="multinode", no_attach=True)
    doc = json.loads(fake.puts[cfg.orch_state_key(orch_id)])
    assert doc["experiment"] == "multinode"
    assert doc["instance_id"] == "i-NEW"
    assert doc["budget_s"] == cfg.baseline_seconds
    assert doc["hourly_usd"] == cfg.orch_hourly_usd()  # in the ledger from t=0


# --------------------------------------------------------------------------- #
# the on-box agent: restart semantics
# --------------------------------------------------------------------------- #
def test_agent_resumes_the_same_run_id_after_a_restart(monkeypatch):
    cfg = _cfg()
    fake = FakeAws(
        instances={("Name", "spot-train-multinode-7"): [_node("i-old", "multinode-7")]},
        docs={
            cfg.orch_state_key("orch-1"): {
                "orch_id": "orch-1",
                "experiment": "multinode",
                "run_id": "multinode-7",
                "attempts": 1,
                "created_at": 1.0,
                "instance_id": "i-orch",
            }
        },
    )
    monkeypatch.setattr(orch, "aws", fake)
    monkeypatch.setattr(orch, "_imds", lambda path: "")
    seen = {}

    def fake_invoke(cfg_, experiment, *, run_id, on_profile):
        seen["run_id"] = run_id
        seen["experiment"] = experiment

    monkeypatch.setattr(orch, "_invoke_experiment", fake_invoke)
    rc = orch.run_agent(cfg, orch_id="orch-1", experiment="multinode")
    assert rc == 0
    # Same run prefix => the trainer's one resume path picks up its checkpoints.
    assert seen["run_id"] == "multinode-7"
    # The previous attempt's boxes are reaped first: they poll an epoch nobody
    # advances any more and hold the vCPU quota the new fleet needs.
    assert fake.terminated == ["i-old"]
    assert json.loads(fake.puts[cfg.orch_state_key("orch-1")])["attempts"] == 2


def test_agent_exits_zero_when_the_run_already_finished(monkeypatch):
    cfg = _cfg()
    fake = FakeAws(
        docs={
            cfg.orch_state_key("orch-1"): {"run_id": "multinode-7", "attempts": 1},
            cfg.run_metrics_key("multinode-7"): {"val_loss": 3.0},
        }
    )
    monkeypatch.setattr(orch, "aws", fake)
    monkeypatch.setattr(orch, "_imds", lambda path: "")
    monkeypatch.setattr(
        orch, "_invoke_experiment", lambda *a, **kw: pytest.fail("must not re-run a done run")
    )
    assert orch.run_agent(cfg, orch_id="orch-1", experiment="multinode") == 0


def test_agent_refuses_to_silently_redo_a_sweep(monkeypatch):
    cfg = _cfg()
    fake = FakeAws(docs={cfg.orch_state_key("orch-1"): {"attempts": 1, "run_id": ""}})
    monkeypatch.setattr(orch, "aws", fake)
    monkeypatch.setattr(orch, "_imds", lambda path: "")
    monkeypatch.setattr(
        orch, "_invoke_experiment", lambda *a, **kw: pytest.fail("sweeps are not resumable")
    )
    # rc 0 so systemd stops retrying; hours of GPU time are not redone blindly.
    assert orch.run_agent(cfg, orch_id="orch-1", experiment="scaling-clean") == 0


def test_agent_registers_the_control_plane_in_the_cost_ledger(monkeypatch):
    cfg = _cfg()
    fake = FakeAws()
    monkeypatch.setattr(orch, "aws", fake)
    monkeypatch.setattr(
        orch, "_imds", lambda path: "i-orch" if path == "instance-id" else "us-e-1a"
    )
    from orchestrator.profile import RunProfile

    profile = RunProfile("multinode-9", kind="multinode", market="spot")

    def fake_invoke(cfg_, experiment, *, run_id, on_profile):
        on_profile(profile)

    monkeypatch.setattr(orch, "_invoke_experiment", fake_invoke)
    assert orch.run_agent(cfg, orch_id="orch-1", experiment="multinode") == 0
    row = profile.instances[0]
    assert row.instance_id == "i-orch" and row.market == "on-demand"
    assert row.hourly_usd == cfg.orch_hourly_usd()
    hb = json.loads(fake.puts[cfg.orch_heartbeat_key("orch-1")])
    assert hb["run_id"].startswith("multinode-") and hb["done"] is True


def test_agent_returns_nonzero_so_systemd_restarts_it(monkeypatch):
    cfg = _cfg()
    fake = FakeAws()
    monkeypatch.setattr(orch, "aws", fake)
    monkeypatch.setattr(orch, "_imds", lambda path: "")

    def boom(*a, **kw):
        raise RuntimeError("supervisor exploded")

    monkeypatch.setattr(orch, "_invoke_experiment", boom)
    assert orch.run_agent(cfg, orch_id="orch-1", experiment="multinode") == 1
    hb = json.loads(fake.puts[cfg.orch_heartbeat_key("orch-1")])
    assert "supervisor exploded" in hb["error"]


# --------------------------------------------------------------------------- #
# dataset directory: instance-store NVMe when present, root volume otherwise
# --------------------------------------------------------------------------- #
def _probe_script(ud: str, tmp_path, *, nvme: str, dataset: str = "openwebtext") -> str:
    """Extract the dataset-location block from a generated user-data script and
    make it runnable here: fake home for the env file, fake mount for the NVMe."""
    start = ud.index("# --- dataset location")
    end = ud.index('df -h "$DATA_LOCAL_DIR"', start) + len('df -h "$DATA_LOCAL_DIR" || true')
    body = ud[start:end]
    body = body.replace("/opt/dlami/nvme", nvme)
    body = body.replace("/home/ubuntu/spot-train.env", str(tmp_path / "spot-train.env"))
    body = body.replace("/home/ubuntu/app/third_party/nanoGPT/data", str(tmp_path / "repo-data"))
    return "set -euo pipefail\n" + body


def _run_probe(ud: str, tmp_path, *, nvme: str, min_free_kb: str = "1") -> str:
    script = tmp_path / "probe.sh"
    script.write_text(_probe_script(ud, tmp_path, nvme=nvme))
    # The real threshold (50GB free) is what proves a candidate is the instance
    # store and not the 30GB root; a test box has no such filesystem, so lower it.
    r = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "SPOT_TRAIN_MIN_FREE_KB": min_free_kb,
            "SPOT_TRAIN_MOUNT_WAIT": "0",  # no need to wait for a mount that will never come
        },
    )
    assert r.returncode == 0, r.stderr
    return (tmp_path / "spot-train.env").read_text()


def test_dataset_dir_uses_the_instance_store_when_present(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    ud = bootstrap.build_user_data(
        _cfg(dataset="openwebtext"), run_id="r", market="spot", max_seconds=60
    )
    env = _run_probe(ud, tmp_path, nvme=str(nvme))
    assert f'export DATA_LOCAL_DIR="{nvme}/spot-train-data/openwebtext"' in env


def test_dataset_dir_falls_back_to_the_root_volume(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    ud = bootstrap.build_user_data(
        _cfg(dataset="openwebtext"), run_id="r", market="spot", max_seconds=60
    )
    missing = tmp_path / "no-such-mount"
    env = _run_probe(ud, tmp_path, nvme=str(missing))
    assert f'export DATA_LOCAL_DIR="{tmp_path / "repo-data"}/openwebtext"' in env
    # Crucially it must NOT have manufactured the mount point on the root volume.
    assert not missing.exists()


def test_dataset_dir_ignores_a_mount_point_that_is_too_small_to_be_a_disk(tmp_path):
    # The failure this guards: the DLAMI ships /opt/dlami/nvme as a directory but
    # the instance store never mounted (wrong instance type, unit not run). Free
    # space is then the 30GB root's — reject it and keep the fallback.
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    ud = bootstrap.build_user_data(
        _cfg(dataset="openwebtext"), run_id="r", market="spot", max_seconds=60
    )
    env = _run_probe(ud, tmp_path, nvme=str(nvme), min_free_kb="999999999999")
    assert f'export DATA_LOCAL_DIR="{tmp_path / "repo-data"}/openwebtext"' in env


def test_dataset_dir_is_resolved_identically_for_every_box(tmp_path):
    # One resolution point, so the box that DOWNLOADS the data and the process
    # that READS it can never disagree (17GB of OpenWebText in the wrong place
    # is a disk-full boot failure on every node at once).
    cfg = _cfg(dataset="openwebtext")
    # The dataset-resolution shell is identical everywhere; multi-node ADDS the
    # node-local checkpoint dir on top of the same $DATA_PARENT decision, so
    # compare against the variant each builder actually uses.
    block = bootstrap._data_dir_block(cfg)
    mn_block = bootstrap._data_dir_block(cfg, local_ckpt=True)
    for ud in (
        bootstrap.build_user_data(cfg, run_id="r", market="spot", max_seconds=60),
        bootstrap.build_user_data(
            cfg, run_id="r", market="spot", max_seconds=60, ddp=True, nodes=2, node_index=1
        ),
        bootstrap.build_provisioning_user_data(cfg, run_id="r", market="spot", max_seconds=60),
        bootstrap.build_fleet_user_data(
            cfg, fleet_id="f", role="worker", worker_id="w0", run_id="r", logs_key="k", port=8001
        ),
    ):
        assert block in ud or mn_block in ud
        # And never as a build-time constant (it varies by instance type).
        assert 'export DATA_LOCAL_DIR="/home/ubuntu/app' not in ud
    # The shared prefix — everything up to the DATA_LOCAL_DIR export — must be
    # byte-identical between the two variants: one resolution point, no drift.
    shared = block.split('echo "export DATA_LOCAL_DIR')[0]
    assert shared and mn_block.startswith(shared)


def test_dataset_dir_logs_path_and_free_space(tmp_path):
    ud = bootstrap.build_user_data(_cfg(), run_id="r", market="spot", max_seconds=60)
    assert 'echo "[data] DATA_LOCAL_DIR=$DATA_LOCAL_DIR"' in ud
    assert 'df -h "$DATA_LOCAL_DIR"' in ud


def test_node_local_checkpoints_ride_the_fast_disk():
    # REVISED: the node-local tier now rides the same disk decision as the
    # dataset, i.e. the instance-store NVMe when the box has one.
    #
    # The original reasoning was "instance store is wiped on stop/terminate, so
    # keep checkpoints on the root volume". But this tier is NOT durable by
    # design — it exists solely so a survivor can restart IN PLACE, and a box
    # that dies takes /tmp with it just as surely as it takes the instance
    # store. S3 remains the durable tier. What the old placement did buy was a
    # 1.5 GB write at ~125 MB/s (gp3 root) instead of ~1 GB/s (NVMe): ~12s on the
    # training critical path, every checkpoint.
    #
    # Still falls back to /tmp when there is no instance store, and is emitted
    # ONLY for multi-node runs (a single box has no peers to re-form with).
    ud = bootstrap.build_user_data(
        _cfg(), run_id="r", market="spot", max_seconds=60, ddp=True, nodes=2, node_index=0
    )
    assert 'LOCAL_CHECKPOINT_DIR="$DATA_PARENT/spot-ckpt"' in ud
    assert 'LOCAL_CHECKPOINT_DIR="/tmp/spot-ckpt"' in ud  # fallback branch present
    single = bootstrap.build_user_data(_cfg(), run_id="r", market="spot", max_seconds=60)
    assert "LOCAL_CHECKPOINT_DIR" not in single


# --------------------------------------------------------------------------- #
# G4-lite: a supervisor restart must RE-ADOPT a healthy fleet, not rebuild it
# --------------------------------------------------------------------------- #
def _member_node(iid, run_id, ip):
    n = _node(iid, run_id)
    n["private_ip"] = ip
    return n


def _epoch_doc(*ips):
    """What supervisor.epoch_doc actually writes: members are dicts carrying an
    ip, NOT node indices, and there is no instance id anywhere in the document."""
    return {
        "epoch": 4,
        "members": [{"node": i, "ip": ip, "rank": i} for i, ip in enumerate(ips)],
        "node_count": len(ips),
        "master_addr": ips[0],
        "master_port": 29404,
    }


def test_reap_readopts_boxes_still_in_the_published_epoch(monkeypatch):
    """The fleet keeps TRAINING while the supervisor is gone -- sidecars run
    static torchrun against the last published epoch doc. Terminating members on
    restart threw away a healthy 8-node fleet plus every step since the last
    checkpoint, on the most likely failure a long run has."""
    cfg = _cfg()
    run_id = "multinode-preempt-7"
    fake = FakeAws(
        instances={
            ("Name", f"spot-train-{run_id}"): [
                _member_node("i-a", run_id, "10.0.0.1"),
                _member_node("i-b", run_id, "10.0.0.2"),
            ]
        },
        docs={cfg.run_epoch_key(run_id): _epoch_doc("10.0.0.1", "10.0.0.2")},
    )
    monkeypatch.setattr(orch, "aws", fake)
    orch._reap_orphans(cfg, run_id)
    assert fake.terminated == [], "re-adoptable members were terminated"


def test_reap_still_kills_boxes_outside_the_epoch(monkeypatch):
    """A box that is genuinely orphaned polls an epoch nobody advances and holds
    vCPU quota the replacement fleet needs. Those must still go."""
    cfg = _cfg()
    run_id = "multinode-preempt-7"
    fake = FakeAws(
        instances={
            ("Name", f"spot-train-{run_id}"): [
                _member_node("i-member", run_id, "10.0.0.1"),
                _member_node("i-orphan", run_id, "10.0.9.9"),
            ]
        },
        docs={cfg.run_epoch_key(run_id): _epoch_doc("10.0.0.1")},
    )
    monkeypatch.setattr(orch, "aws", fake)
    orch._reap_orphans(cfg, run_id)
    assert fake.terminated == ["i-orphan"]


def test_reap_takes_everything_when_no_epoch_was_ever_published(monkeypatch):
    """Nothing was adopted, so the old whole-fleet behaviour is exactly right --
    failing toward reaping keeps an un-placeable box from holding quota."""
    cfg = _cfg()
    run_id = "multinode-preempt-7"
    fake = FakeAws(
        instances={("Name", f"spot-train-{run_id}"): [_member_node("i-a", run_id, "10.0.0.1")]},
        docs={},
    )
    monkeypatch.setattr(orch, "aws", fake)
    orch._reap_orphans(cfg, run_id)
    assert fake.terminated == ["i-a"]


# --------------------------------------------------------------------------- #
# `orch up --run-id` — adopting a run whose control-plane BOX died
# --------------------------------------------------------------------------- #
# systemd resurrects a supervisor PROCESS; nothing resurrects the box. Without
# adoption, losing the control-plane instance ends the run outright even though
# the fleet is still training and every checkpoint is durable in S3.
def _up_args(**kw):
    base = dict(experiment="multinode-preempt", no_attach=True, run_id="", force=False)
    base.update(kw)
    return base


def test_up_seeds_the_run_id_so_the_agent_adopts_instead_of_minting(monkeypatch):
    """run_agent mints a run_id ONLY when the state doc's field is empty, so
    seeding it is the entire adoption mechanism."""
    cfg = _cfg()
    fake = FakeAws()
    fake.dry = True
    monkeypatch.setattr(orch, "aws", fake)
    orch.up(cfg, **_up_args(run_id="multinode-preempt-123"))
    state = [json.loads(body) for key, body in fake.puts.items() if "orchestrators/" in key]
    assert state and state[0]["run_id"] == "multinode-preempt-123"


def test_up_without_run_id_still_mints_a_fresh_one(monkeypatch):
    """Cold start is unchanged: an empty field is what tells the agent to mint."""
    cfg = _cfg()
    fake = FakeAws()
    fake.dry = True
    monkeypatch.setattr(orch, "aws", fake)
    orch.up(cfg, **_up_args())
    state = [json.loads(body) for key, body in fake.puts.items() if "orchestrators/" in key]
    assert state and state[0]["run_id"] == ""


def test_up_rejects_a_run_id_from_a_different_experiment(monkeypatch):
    """Adopting the wrong run would point a fresh control plane at another run's
    checkpoints -- silent, and unrecoverable once it resumes from them."""
    cfg = _cfg()
    fake = FakeAws()
    fake.dry = True
    monkeypatch.setattr(orch, "aws", fake)
    with pytest.raises(SystemExit, match="does not look like"):
        orch.up(cfg, **_up_args(experiment="baseline", run_id="multinode-preempt-123"))


def test_up_refuses_to_adopt_a_non_resumable_experiment(monkeypatch):
    """A sweep drives several runs and has no single resume point, so silently
    redoing GPU hours is the failure mode to prevent."""
    cfg = _cfg()
    fake = FakeAws()
    fake.dry = True
    monkeypatch.setattr(orch, "aws", fake)
    with pytest.raises(SystemExit, match="cannot adopt"):
        orch.up(cfg, **_up_args(experiment="scaling-experiment", run_id="scaling-123"))


# --------------------------------------------------------------------------- #
# `orch down` must VERIFY, not assume
# --------------------------------------------------------------------------- #
# It used to print "terminated N instance(s)" straight after issuing the calls,
# which is a statement of intent. An API error, a throttle or a permission gap
# all still printed success while the boxes kept billing. This is the operator's
# stop button on a multi-hundred-dollar run.
class _StuckAws(FakeAws):
    """Terminate 'succeeds' but the instance never leaves running."""

    def __init__(self, *a, stuck=(), **kw):
        super().__init__(*a, **kw)
        self.stuck = set(stuck)
        self.waited: list[str] = []

    def wait_quota_released(self, iid):
        self.waited.append(iid)
        if iid in self.stuck:
            raise TimeoutError(f"{iid} still running")


def _one_controller():
    return {(orch.ORCH_ROLE_TAG, orch.CONTROLLER): [_controller()]}


def test_down_waits_for_each_victim_to_stop_billing(monkeypatch, capsys):
    from orchestrator import orch

    fake = _StuckAws(instances=_one_controller())
    monkeypatch.setattr(orch, "aws", fake)
    orch.down(_cfg(), orch_id="orch-1", all_=True, yes=True)
    assert fake.terminated, "nothing was terminated"
    assert fake.waited == fake.terminated, "down() did not verify each instance stopped"
    assert "verified" in capsys.readouterr().out


def test_down_shouts_when_an_instance_refuses_to_die(monkeypatch, capsys):
    """Silence about a box that kept billing is the worst possible outcome."""
    from orchestrator import orch

    stuck_id = _controller()["id"]
    fake = _StuckAws(instances=_one_controller(), stuck={stuck_id})
    monkeypatch.setattr(orch, "aws", fake)
    orch.down(_cfg(), orch_id="orch-1", all_=True, yes=True)
    err = capsys.readouterr().err
    assert "did NOT stop" in err and "still" in err
    assert "terminate-instances" in err, "must hand the operator the exact recovery command"
