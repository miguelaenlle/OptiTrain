"""Remote dataset-prep tests — hermetic (no AWS: user-data is a string, the EC2
client is faked, S3 lookups are monkeypatched).

``stage-data --remote`` launches a box that runs unattended for an hour, so the
properties pinned here are the ones that make that safe:

  - it TERMINATES ITSELF on every path — an unconditional lifetime timer armed
    before anything fallible, a trap that powers off on any exit, and an explicit
    shutdown after the status doc;
  - it clones the CONFIGURED branch (not main) and redirects the 54 GB HF cache
    onto the volume that was sized for it;
  - the launch sizes its own gp3 root (size + throughput) and existing launches
    are left byte-for-byte unchanged;
  - an already-staged corpus is refused and a truncated upload is caught;
  - Ctrl-C detaches instead of killing, and ``--dry-run`` touches no AWS at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from orchestrator import aws, bootstrap, prep
from orchestrator.config import OrchestratorConfig

_BRANCH = "phase1/gpt2-owt-baseline"


def _cfg(**kw) -> OrchestratorConfig:
    cfg = OrchestratorConfig(bucket="test-bucket")
    cfg.dataset = "openwebtext"
    cfg.repo_branch = _BRANCH
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _ud(cfg: OrchestratorConfig | None = None) -> str:
    return bootstrap.build_prep_user_data(cfg or _cfg(), prep_id="prep-1")


@pytest.fixture(autouse=True)
def _no_dry_run():
    """Every test starts from a known dry-run state and restores it — the flag is
    module-global, and a leaked True would make other suites silently no-op."""
    aws.set_dry_run(False)
    yield
    aws.set_dry_run(False)


# --------------------------------------------------------------------------- #
# user-data
# --------------------------------------------------------------------------- #
def test_bash_syntax(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    path = tmp_path / "prep.sh"
    path.write_text(_ud())
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed:\n{r.stderr}"


def test_clones_the_configured_branch():
    ud = _ud()
    assert f"git clone --depth 1 -b {_BRANCH} " in ud
    # The whole point of relaying the branch: a prep launched from a feature
    # branch must not silently run main's prepare.py / staging code.
    assert "-b main " not in ud


def test_hf_home_redirected_onto_the_big_volume():
    ud = _ud()
    assert f'export HF_HOME="{bootstrap._PREP_HF_HOME}"' in ud
    assert 'mkdir -p "$HF_HOME"' in ud
    # ...and it must be set before the download starts.
    assert ud.index("HF_HOME") < ud.index("prepare.py")


def test_prepares_then_stages_through_the_ordinary_path():
    ud = _ud()
    assert 'PREPARE="data/openwebtext/prepare.py"' in ud
    assert '"$PREP_PY" -u prepare.py' in ud
    assert '"$PREP_PY" -u -m orchestrator stage-data' in ud
    # Order matters: upload what prepare.py produced, not what it hasn't yet.
    assert ud.index("-u prepare.py") < ud.index("-m orchestrator stage-data")
    # The box needs the bucket/dataset/region the laptop was configured with.
    assert 'export SPOT_TRAIN_BUCKET="test-bucket"' in ud
    assert 'export DATASET="openwebtext"' in ud
    assert 'export AWS_REGION="us-east-1"' in ud


def test_installs_lazy_deps_but_not_torch():
    ud = _ud()
    assert "pip install datasets tiktoken tqdm numpy boto3" in ud
    # --no-deps on the package: its dependency set would drag ~2 GB of CUDA
    # torch onto a box that only tokenizes text.
    assert "pip install --no-deps -e /home/ec2-user/app" in ud
    assert "pip install -e" not in ud  # a plain editable install would pull torch in
    assert "pip install torch" not in ud


def test_terminates_on_success_and_on_failure():
    ud = _ud()
    # 1. trap: fires on a set -e abort, a traceback, any early exit.
    assert "trap terminate_self EXIT" in ud
    assert "shutdown -h now || systemctl poweroff || poweroff -f" in ud
    # 2. explicit shutdown on the normal path, after the status doc is written.
    assert ud.rstrip().endswith('exit "$RC"')
    assert ud.index("prep/prep-1/status.json") < ud.rindex("shutdown -h now")
    # 3. the status doc records BOTH outcomes (it is never conditional on success).
    assert '"ok": $RC == 0' in ud
    assert '"rc": $RC' in ud
    # The job runs in a subshell so a failure becomes an RC instead of an early
    # exit that would skip the status doc entirely.
    assert "RC=$?" in ud


def test_lifetime_backstop_is_unconditional_and_armed_first():
    cfg = _cfg(prep_max_lifetime_seconds=7200)
    ud = _ud(cfg)
    assert "systemd-run --on-active=7200s --unit=spot-prep-autokill" in ud
    assert "shutdown -h +120" in ud  # ceil(7200/60) fallback when systemd-run is absent
    # Armed BEFORE anything that can fail — a script that dies on the next line
    # must still leave a box that terminates itself.
    for later in ("dnf install", "git clone", "trap terminate_self"):
        assert ud.index("spot-prep-autokill") < ud.index(later), later
    # And it cannot be switched off: even 0 gets clamped to a real timer.
    assert "--on-active=60s" in _ud(_cfg(prep_max_lifetime_seconds=0))


def test_streams_its_log_to_the_prep_key():
    ud = _ud()
    cfg = _cfg()
    assert cfg.prep_log_key("prep-1") == "prep/prep-1/prep.log"
    assert cfg.prep_status_key("prep-1") == "prep/prep-1/status.json"
    assert cfg.prep_log_uri("prep-1") == "s3://test-bucket/prep/prep-1/prep.log"
    # Same relay mechanism as the training boxes: boto3 upload_file on a timer,
    # started before the slow work so the whole hour is watchable.
    assert 'c.upload_file("/var/log/spot-prep.log", "test-bucket", "prep/prep-1/prep.log")' in ud
    assert ud.index("upload_file") < ud.index("git clone")


def test_region_is_exported_before_the_relay_starts():
    # The relay is its own boto3 process: without a region in the environment it
    # would fall back to an IMDS lookup we don't need.
    ud = _ud()
    assert 'export AWS_DEFAULT_REGION="us-east-1"' in ud
    assert ud.index("AWS_DEFAULT_REGION") < ud.index("upload_file")


def test_no_credentials_in_user_data(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shhh")
    monkeypatch.setenv("HF_TOKEN", "hf_shhh")
    ud = _ud()
    for secret in ("shhh", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"):
        assert secret not in ud


def test_prep_passthrough_only_when_set(monkeypatch):
    monkeypatch.delenv("OWT_NUM_PROC", raising=False)
    assert "OWT_NUM_PROC" not in _ud()
    monkeypatch.setenv("OWT_NUM_PROC", "12")
    assert 'export OWT_NUM_PROC="12"' in _ud()


# --------------------------------------------------------------------------- #
# Block device mapping
# --------------------------------------------------------------------------- #
class _FakeEC2:
    """Records RunInstances kwargs so we can assert on the launch shape."""

    def __init__(self):
        self.calls: list[dict] = []

    def describe_images(self, ImageIds):  # noqa: N803 — boto3's parameter name
        return {"Images": [{"RootDeviceName": "/dev/xvda"}]}

    def run_instances(self, **kwargs):
        self.calls.append(kwargs)
        return {"Instances": [{"InstanceId": "i-fake"}]}


@pytest.fixture
def fake_ec2(monkeypatch) -> _FakeEC2:
    fake = _FakeEC2()
    monkeypatch.setattr(aws, "_client", lambda service: fake)
    return fake


def test_root_volume_mapping_shape():
    mapping = aws.root_volume_mapping("/dev/xvda", 200, iops=6000, throughput=500)
    assert mapping == [
        {
            "DeviceName": "/dev/xvda",
            "Ebs": {
                "VolumeSize": 200,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
                "Iops": 6000,
                "Throughput": 500,
            },
        }
    ]
    # Unset knobs are omitted entirely (EC2 then applies the gp3 defaults)
    # instead of being sent as zeros, which RunInstances rejects.
    ebs = aws.root_volume_mapping("/dev/sda1", 30)[0]["Ebs"]
    assert "Iops" not in ebs and "Throughput" not in ebs


def _launch(**kw) -> dict:
    aws.launch(
        ami_id="ami-1",
        instance_type="g4dn.xlarge",
        profile_name="p",
        security_group_id="sg-1",
        user_data="#!/bin/bash",
        market="on-demand",
        run_id="r1",
        **kw,
    )


def test_existing_launches_are_unaffected(fake_ec2):
    _launch()
    kwargs = fake_ec2.calls[-1]
    # The additive parameter defaults to off, so proven training/fleet/control
    # -plane launches keep inheriting the AMI's own mapping.
    assert "BlockDeviceMappings" not in kwargs
    assert kwargs["InstanceInitiatedShutdownBehavior"] == "terminate"


def test_launch_sizes_the_root_volume_when_asked(fake_ec2):
    _launch(root_volume_gb=200, root_volume_iops=6000, root_volume_throughput=500)
    bdm = fake_ec2.calls[-1]["BlockDeviceMappings"]
    # The device name is READ from the AMI: a mismatched name would silently add
    # a second volume and leave the box on its tiny default root.
    assert bdm[0]["DeviceName"] == "/dev/xvda"
    assert bdm[0]["Ebs"]["VolumeSize"] == 200
    assert bdm[0]["Ebs"]["Throughput"] == 500


def test_prep_launch_carries_volume_and_tags(monkeypatch, fake_ec2):
    cfg = _cfg()
    monkeypatch.setattr(aws, "object_size", lambda bucket, key: None)  # nothing staged
    monkeypatch.setattr(aws, "ensure_security_group", lambda name, region: "sg-1")
    monkeypatch.setattr(aws, "resolve_ami", lambda ami, filt: "ami-al2023")
    prep_id = prep.run_remote_prep(cfg, attach=False)
    kwargs = fake_ec2.calls[-1]
    assert kwargs["InstanceType"] == "c6i.4xlarge"
    assert kwargs["IamInstanceProfile"] == {"Name": cfg.instance_profile}
    assert kwargs["BlockDeviceMappings"][0]["Ebs"] == {
        "VolumeSize": 200,
        "VolumeType": "gp3",
        "DeleteOnTermination": True,
        "Iops": 6000,
        "Throughput": 500,
    }
    tags = {t["Key"]: t["Value"] for t in kwargs["TagSpecifications"][0]["Tags"]}
    assert tags[prep.PREP_TAG] == prep_id  # tag-based reattach, no local state
    assert tags["market"] == "on-demand"  # never spot: a reclaim wastes the hour


# --------------------------------------------------------------------------- #
# Idempotence + verification
# --------------------------------------------------------------------------- #
def test_branch_mismatch_is_flagged(monkeypatch, capsys):
    """REPO_BRANCH defaults to main and the box runs the CLONED prepare.py, so a
    forgotten override would quietly spend an hour on the wrong script."""
    import subprocess as sp

    class _R:
        stdout = "phase1/gpt2-owt-baseline\n"

    monkeypatch.setattr(sp, "run", lambda *a, **kw: _R())
    prep._warn_branch_mismatch(_cfg(repo_branch="main"))
    assert "REPO_BRANCH=main" in capsys.readouterr().err
    prep._warn_branch_mismatch(_cfg())  # matching branch => silent
    assert capsys.readouterr().err == ""


def test_check_sizes_accepts_a_plausible_corpus():
    ok, lines = prep.check_sizes("openwebtext", {"train.bin": 18_100_000_000, "val.bin": 8_800_000})
    assert ok
    assert any("16.9 GB" in line for line in lines)


def test_check_sizes_rejects_a_truncated_bin():
    ok, lines = prep.check_sizes("openwebtext", {"train.bin": 2_000_000_000, "val.bin": 8_800_000})
    assert not ok
    assert any("TRUNCATED" in line for line in lines)


def test_check_sizes_rejects_a_missing_bin():
    ok, lines = prep.check_sizes("openwebtext", {"train.bin": 18_100_000_000, "val.bin": None})
    assert not ok
    assert any("MISSING" in line for line in lines)


def test_check_sizes_unknown_dataset_only_rejects_empties():
    # No measured floor exists for a capped/parameterised recipe, so we refuse to
    # invent one — but a sub-KB bin is a failed write in any recipe.
    ok, _ = prep.check_sizes("shakespeare_char", {"train.bin": 1_003_854, "val.bin": 111_540})
    assert ok
    ok, _ = prep.check_sizes("shakespeare_char", {"train.bin": 12, "val.bin": 111_540})
    assert not ok


def test_already_staged_is_refused(monkeypatch, capsys):
    sizes = {"train.bin": 18_100_000_000, "val.bin": 8_800_000}
    monkeypatch.setattr(aws, "object_size", lambda bucket, key: sizes[key.rsplit("/", 1)[-1]])

    def must_not_launch(**kwargs):
        raise AssertionError("an already-staged dataset must never launch a box")

    monkeypatch.setattr(aws, "launch", must_not_launch)
    with pytest.raises(SystemExit, match="refusing to redo"):
        prep.run_remote_prep(_cfg(), attach=False)
    assert "already staged" in capsys.readouterr().err


def test_verify_reports_sizes(monkeypatch, capsys):
    monkeypatch.setattr(aws, "object_size", lambda bucket, key: 18_100_000_000)
    assert prep.verify(_cfg()) is True
    assert "train.bin" in capsys.readouterr().err


def test_finish_fails_loudly_on_a_truncated_upload(monkeypatch):
    monkeypatch.setattr(prep, "watch", lambda cfg, prep_id, iid="": {"ok": True, "rc": 0})
    monkeypatch.setattr(aws, "object_size", lambda bucket, key: 5_000)  # truncated
    with pytest.raises(SystemExit, match="not usable"):
        prep._finish(_cfg(), "prep-1", "i-1", attach=True)


def test_finish_fails_when_the_box_reports_failure(monkeypatch):
    monkeypatch.setattr(prep, "watch", lambda cfg, prep_id, iid="": {"ok": False, "rc": 1})
    with pytest.raises(SystemExit, match="FAILED"):
        prep._finish(_cfg(), "prep-1", "i-1", attach=True)


# --------------------------------------------------------------------------- #
# Attach / detach
# --------------------------------------------------------------------------- #
def test_ctrl_c_detaches_without_killing_anything(monkeypatch, capsys):
    def interrupted(cfg, prep_id, iid=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(prep, "watch", interrupted)

    def must_not_terminate(instance_id):
        raise AssertionError("Ctrl-C must DETACH, never terminate the prep box")

    monkeypatch.setattr(aws, "terminate", must_not_terminate)
    assert prep._finish(_cfg(), "prep-1", "i-42", attach=True) == "prep-1"
    err = capsys.readouterr().err
    assert "detached" in err
    assert "--attach prep-1" in err  # how to get back
    assert "terminate-instances" in err and "i-42" in err  # how to stop it


def test_no_attach_prints_how_to_watch(monkeypatch, capsys):
    def must_not_watch(*a, **kw):
        raise AssertionError("--no-attach must not stream")

    monkeypatch.setattr(prep, "watch", must_not_watch)
    prep._finish(_cfg(), "prep-1", "i-42", attach=False)
    assert "--attach prep-1" in capsys.readouterr().err


def test_watch_streams_new_bytes_then_returns_status(monkeypatch, capsys):
    cfg = _cfg()
    chunks = ["[prep] cloning\n", "[prep] cloning\n[prep] tokenizing\n"]
    seen = {"n": 0}

    monkeypatch.setattr(aws, "object_exists", lambda bucket, key: True)
    monkeypatch.setattr(aws, "get_text", lambda bucket, key: chunks[min(seen["n"], 1)])

    def status(bucket, key):
        seen["n"] += 1
        return {"ok": True, "rc": 0} if seen["n"] > 2 else None

    monkeypatch.setattr(aws, "get_json", status)
    monkeypatch.setattr(prep.time, "sleep", lambda s: None)
    assert prep.watch(cfg, "prep-1") == {"ok": True, "rc": 0}
    # Each poll prints only what is NEW — an hour-long log is never re-echoed.
    assert capsys.readouterr().out == "[prep] cloning\n[prep] tokenizing\n"


def test_watch_gives_up_when_the_box_is_gone(monkeypatch, capsys):
    cfg = _cfg(log_stream_seconds=0)
    monkeypatch.setattr(aws, "object_exists", lambda bucket, key: False)
    monkeypatch.setattr(aws, "get_json", lambda bucket, key: None)
    monkeypatch.setattr(aws, "instance_state", lambda iid: "terminated")
    monkeypatch.setattr(prep.time, "sleep", lambda s: None)
    # monotonic jumps a minute per poll so the liveness check runs immediately.
    ticks = iter([0.0] + [61.0 * i for i in range(1, 200)])
    monkeypatch.setattr(prep.time, "monotonic", lambda: next(ticks))
    assert prep.watch(cfg, "prep-1", "i-42") is None
    assert "terminated" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Cost notice + dry run
# --------------------------------------------------------------------------- #
def test_cost_estimate_includes_the_volume():
    cfg = _cfg()
    instance_only = cfg.prep_hourly_usd()
    est = prep.estimate_usd(cfg, 1.0)
    assert est is not None and instance_only is not None
    assert est > instance_only  # storage + provisioned IOPS/throughput are billed
    assert est < instance_only + 0.15  # ...and they are cents, not dollars
    # An unpriced instance type yields no number rather than a wrong one.
    assert prep.estimate_usd(_cfg(prep_instance_type="c9.enormous"), 1.0) is None


def test_dry_run_end_to_end(tmp_path):
    """`stage-data --remote --dry-run` from the real CLI: prints the plan and the
    billable notice, and provably calls no AWS API (dry-run never builds a
    client, so this needs no creds and no boto3)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        **os.environ,
        "PYTHONPATH": os.path.join(root, "src"),
        "SPOT_TRAIN_BUCKET": "test-bucket",
        "DATASET": "openwebtext",
        "AWS_REGION": "us-east-1",
    }
    r = subprocess.run(
        [sys.executable, "-m", "orchestrator", "stage-data", "--remote", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,  # away from the repo so no local .env can colour the run
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "BILLABLE" in out
    assert "c6i.4xlarge" in out
    assert "[aws:dry-run] would RunInstances" in out
    assert "root=200GB gp3/6000iops/500MBps" in out
    assert "no AWS calls were made" in out
    # Local staging must be untouched by the flag: no prepare.py ran here.
    assert not list(tmp_path.iterdir())


def test_local_stage_data_still_default(monkeypatch, tmp_path):
    """Without --remote the command is the old local path, unchanged."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        **os.environ,
        "PYTHONPATH": os.path.join(root, "src"),
        "SPOT_TRAIN_BUCKET": "test-bucket",
        "DATASET": "openwebtext",
    }
    r = subprocess.run(
        [sys.executable, "-m", "orchestrator", "stage-data", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    out = r.stdout + r.stderr
    assert "RunInstances" not in out  # no box is ever launched by local staging
    assert "[aws:dry-run] would head s3://test-bucket/data/openwebtext/train.bin" in out
