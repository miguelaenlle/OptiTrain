"""Build the EC2 user-data script that runs on the box at boot.

The box is an AWS Deep Learning AMI (Ubuntu) with CUDA + PyTorch preinstalled.
Two DLAMI realities shape these scripts:

  1. The PyTorch environment only auto-activates for the ``ubuntu`` **login**
     shell — root user-data would otherwise get a system python without torch.
     So we run everything via ``sudo -u ubuntu -i`` (login shell) and preflight
     ``import torch`` to fail loudly if the env is wrong.
  2. We run from source with ``PYTHONPATH`` instead of ``pip install -e .`` so we
     never try to write into a possibly root-owned framework env. (nanoGPT is
     found by the trainer relative to the package.)

Three builders:

  * ``build_provisioning_user_data`` — clone the repo + nanoGPT submodule and
    install deps, write a ``source``-able env file, then STOP and leave the box
    UP. You ssh in, confirm the code is there, and run training by hand. Used
    while bringing the box up.
  * ``build_user_data`` — the full run: provision, then ``spot_train.train``
    under a wall-clock budget, then self-terminate on success. Used once the box
    is proven. (Parked while we validate provisioning over SSH.)
  * ``build_bake_user_data`` — provision only (no training, no env file), then
    write a status marker to S3; ``spot-orchestrate bake-ami`` images the box.

The trainer pulls the dataset and reads/writes S3 via the instance profile role
— no credentials are passed in user-data.
"""

from __future__ import annotations

import base64
import json
import os

from .config import OrchestratorConfig


def _trainer_env(
    cfg: OrchestratorConfig, *, run_id: str, market: str, max_seconds: int
) -> dict[str, str]:
    """Environment the trainer reads via ``TrainConfig.from_env`` (spot_train
    config). Written to a ``source``-able file so you can run training by hand
    after sshing in, and exported inline for the full-run builder."""
    env = {
        "PYTHONPATH": "/home/ubuntu/app/src",
        "CHECKPOINT_URI": cfg.run_checkpoint_uri(run_id),
        "METRICS_URI": cfg.run_metrics_uri(run_id),
        "SAMPLES_URI": cfg.run_samples_uri(run_id),
        "SAMPLES_PREFIX_URI": cfg.run_samples_prefix_uri(run_id),
        # base64(JSON array): the env file is `export K="v"` lines, and prompt
        # text (quotes, spaces, newlines) must survive that quoting untouched.
        # The loads/dumps round-trip fails fast here on a malformed .env value —
        # before any instance is launched.
        "SAMPLE_PROMPTS": base64.b64encode(
            json.dumps(json.loads(cfg.sample_prompts)).encode()
        ).decode(),
        "DATA_URI": cfg.data_uri(),
        "DATASET": cfg.dataset,
        # NOTE: DATA_LOCAL_DIR is deliberately NOT here — it is resolved at boot
        # by _data_dir_block (instance-store NVMe when the box has one, root
        # volume otherwise) and appended to this same env file.
        "MAX_SECONDS": str(max_seconds),
        "CHECKPOINT_INTERVAL_SECONDS": str(cfg.checkpoint_interval_seconds),
        "SMOKE_TEST_EVERY": str(cfg.smoke_test_every),
        "EVAL_ITERS": str(cfg.eval_iters),
        "BATCH_SIZE": str(cfg.batch_size),
        "RUN_ID": run_id,
        "MARKET": market,
        "DEVICE": "auto",  # trainer auto-detects cuda vs cpu on the box
        "DDP_DATA_MODE": cfg.ddp_data_mode,  # only used when launched via torchrun
        "PYTHONUNBUFFERED": "1",  # unbuffered so `tail -f` shows per-step lines live
    }
    # Recipe/cadence knobs (MAX_STEPS, LR schedule, EVAL/SAMPLE intervals, …)
    # relay verbatim, and only when set — unset keeps the trainer's defaults.
    env.update(cfg.trainer_passthrough())
    return env


def _export_block(env: dict[str, str]) -> str:
    return "\n".join(f'export {k}="{v}"' for k, v in env.items())


# Where a box puts the dataset. Resolved AT BOOT, in shell, because it depends on
# the instance type: the DLAMI root volume is only 30 GB (mostly CUDA + PyTorch
# already), while OpenWebText's train.bin alone is ~17 GB — downloading that into
# the repo checkout fills the disk and kills the boot, on every node at once.
# g5/g4dn ship a large EPHEMERAL NVMe instance store (250 GB on g5.xlarge) that
# is included in the price, unused today, and ~4x faster to write than the
# default 125 MB/s gp3 root — which also shortens every preemption recovery.
# t3 (and the fleet's c7i) have no instance store at all, so this must be probed,
# never baked in.
#
# Ephemeral is CORRECT here: the dataset is re-pulled from S3 on every boot
# anyway. Nothing that must outlive the instance moves — checkpoints still go to
# S3 (durable) plus LOCAL_CHECKPOINT_DIR=/tmp on the root volume (node-local by
# design, only ever used by a survivor restarting in place).
_DATA_PARENT_FALLBACK = "/home/ubuntu/app/third_party/nanoGPT/data"
_DATA_SUBDIR = "spot-train-data"
# A candidate must have at least this much free (KB) to be believed. It doubles
# as the "is this really the instance store?" test: the DLAMI root volume is
# 30 GB TOTAL, so anything with 50 GB free cannot be it — which is what saves us
# if the mount point directory exists but the instance store isn't mounted on it.
# Overridable per boot (SPOT_TRAIN_MIN_FREE_KB) for odd instance types.
_DATA_MIN_FREE_KB = 50_000_000


def _data_dir_block(cfg: OrchestratorConfig) -> str:
    """Shell that picks the dataset directory and appends the resolved
    ``DATA_LOCAL_DIR`` export to the env file. The ONE place the path is decided,
    so the download (trainer) and the read (trainer/fleet worker) can never
    disagree. Runs under ``set -e``, hence the `if`/`|| true` discipline."""
    return f"""
# --- dataset location: prefer the ephemeral NVMe instance store ------------- #
# Candidates: the DLAMI's conventional mount first, then ANY non-root filesystem
# mounted off a local disk (so this keeps working if AWS moves the mount point).
DATA_PARENT="{_DATA_PARENT_FALLBACK}"
MIN_FREE_KB="${{SPOT_TRAIN_MIN_FREE_KB:-{_DATA_MIN_FREE_KB}}}"
for attempt in 1 2 3; do
  CANDS="/opt/dlami/nvme $(lsblk -ln -o NAME,MOUNTPOINT 2>/dev/null \\
    | awk '$1 ~ /^(nvme|xvd|sd)/ && $2 != "" && $2 != "/" && $2 !~ /^\\/boot/ {{print $2}}' \\
    | tr '\\n' ' ' || true)"
  for cand in $CANDS; do
    # MUST already exist: the mount unit creates it. Never `mkdir -p` the mount
    # point itself — that would silently manufacture a directory ON THE ROOT
    # VOLUME and then happily fill it with 17 GB of OpenWebText.
    [ -d "$cand" ] || continue
    FREE_KB=$(df -Pk "$cand" 2>/dev/null | awk 'NR==2 {{print $4}}' || echo 0)
    [ "${{FREE_KB:-0}}" -ge "$MIN_FREE_KB" ] || continue
    # The mount is usually root-owned; we run as ubuntu, so take it with sudo
    # and hand it over. No sudo => try the next candidate.
    if {{ mkdir -p "$cand/{_DATA_SUBDIR}" 2>/dev/null \\
         || {{ sudo -n mkdir -p "$cand/{_DATA_SUBDIR}" 2>/dev/null \\
              && sudo -n chown "$(id -un)" "$cand/{_DATA_SUBDIR}" 2>/dev/null; }}; }} \\
       && [ -w "$cand/{_DATA_SUBDIR}" ]; then
      DATA_PARENT="$cand/{_DATA_SUBDIR}"
      break
    fi
  done
  # The DLAMI mounts the instance store from a boot unit that can land just
  # after user-data starts, so retry briefly before settling for the root
  # volume — but never sleep after the last attempt (a box with no instance
  # store must not pay for this on every preemption recovery).
  [ "$DATA_PARENT" = "{_DATA_PARENT_FALLBACK}" ] || break
  [ "$attempt" = 3 ] || sleep "${{SPOT_TRAIN_MOUNT_WAIT:-3}}"
done
DATA_LOCAL_DIR="$DATA_PARENT/{cfg.dataset}"
mkdir -p "$DATA_LOCAL_DIR"
echo "export DATA_LOCAL_DIR=\\"$DATA_LOCAL_DIR\\"" >> /home/ubuntu/spot-train.env
# Log the resolved path AND free space: a future disk-full failure should be
# diagnosable straight from this log instead of needing a repro.
echo "[data] DATA_LOCAL_DIR=$DATA_LOCAL_DIR"
df -h "$DATA_LOCAL_DIR" || true
"""


def _provision_steps(cfg: OrchestratorConfig, exports: str) -> str:
    """Setup run as the ubuntu login shell (torch env active): clone the repo,
    populate the nanoGPT submodule, install deps, write a source-able env file,
    and preflight. Shared by both builders."""
    return f"""set -euxo pipefail
cd /home/ubuntu

# Interpreter: the DLAMI base shell only ships `python3`; plain `python` exists
# only inside the pytorch venv, which the non-interactive login shell does NOT
# reliably auto-activate. Use the venv python by absolute path (it has torch +
# boto3 and always exists on this AMI); fall back to python3 on other images.
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# Code: clone the repo, or — on a pre-baked AMI where the clone already exists —
# fast-forward it to the branch tip so baked boxes still track {cfg.repo_branch}
# (a bare `[ -d app ] || clone` would pin baked boxes at bake-time code forever).
# Then populate the nanoGPT submodule: a plain clone leaves third_party/nanoGPT
# EMPTY, and the trainer imports GPT from there, so the init is not optional.
if [ -d app/.git ]; then
  git -C app fetch --depth 1 origin {cfg.repo_branch}
  git -C app reset --hard FETCH_HEAD
else
  git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
fi
cd app
git submodule update --init --depth 1

# Deps: the DLAMI ships torch + numpy; boto3 is usually missing. Best-effort so a
# broken pip doesn't wedge the whole boot.
"$VENV_PY" -c "import boto3" 2>/dev/null \\
  || "$VENV_PY" -m pip install boto3 \\
  || "$VENV_PY" -m pip install --user boto3 || true
# tiktoken: the GPT-2 BPE codec for end-of-run/interval samples on BPE datasets
# (OpenWebText). Missing => sampling skips gracefully, training is unaffected.
"$VENV_PY" -c "import tiktoken" 2>/dev/null \\
  || "$VENV_PY" -m pip install tiktoken \\
  || "$VENV_PY" -m pip install --user tiktoken || true

# Env: drop the trainer's config into a file you can `source` before a manual run.
cat > /home/ubuntu/spot-train.env <<'ENV'
{exports}
ENV
{_data_dir_block(cfg)}
# Preflight: fail loudly IN THIS LOG if the torch/boto3 env is wrong.
"$VENV_PY" - <<'PY'
import torch, boto3, numpy  # noqa: F401
print("preflight ok: torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
PY
"""


def _fleet_env(
    cfg: OrchestratorConfig, *, fleet_id: str, role: str, worker_id: str, run_id: str, port: int
) -> dict[str, str]:
    """Environment for a fleet box (inference worker or router). Workers need
    the checkpoint + codec locations; the router only needs the heartbeat
    prefix. ADVERTISE_ADDR is exported at boot from IMDS (private IP).

    ``fleet_id=""`` = standalone serve box: no heartbeat prefix, so the worker
    skips registration entirely (there is no router to find it)."""
    env = {
        "PYTHONPATH": "/home/ubuntu/app/src",
        "FLEET_WORKERS_URI": cfg.fleet_workers_uri(fleet_id) if fleet_id else "",
        "PORT": str(port),
        "HOST": "0.0.0.0",
        "MARKET": cfg.fleet_market if role == "worker" else "on-demand",
        "PYTHONUNBUFFERED": "1",
    }
    if role == "worker":
        env.update(
            {
                "WORKER_ID": worker_id,
                "CHECKPOINT_URI": cfg.run_checkpoint_uri(run_id),
                "DATA_URI": cfg.data_uri(),
                "DATASET": cfg.dataset,
                # DATA_LOCAL_DIR comes from _data_dir_block (same resolution as
                # the training boxes) — a worker only needs meta.pkl, but the two
                # call sites must never disagree about where the data lives.
                "RUN_ID": run_id,
                "DEVICE": "auto",
            }
        )
    return env


def build_fleet_user_data(
    cfg: OrchestratorConfig,
    *,
    fleet_id: str,
    role: str,  # "worker" | "router"
    worker_id: str = "",
    run_id: str = "",
    logs_key: str,
    port: int,
) -> str:
    """Boot one fleet box: provision code (same clone/fast-forward + submodule
    steps as the trainer), pip-install the fleet deps, then run the worker or
    router service in the foreground while streaming the boot log to S3.

    Unlike training boxes there is no self-terminate: serving runs until
    `fleet down` (or a spot reclaim) terminates the instance. A nonzero service
    exit leaves the box up for debugging — the heartbeat TTL removes a wedged
    worker from rotation either way."""
    module = "inference.worker" if role == "worker" else "inference.router"
    env = _fleet_env(
        cfg, fleet_id=fleet_id, role=role, worker_id=worker_id, run_id=run_id, port=port
    )
    steps = _provision_steps(cfg, _export_block(env))
    bucket = cfg.bucket
    interval = cfg.log_stream_seconds
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
chmod 644 /var/log/spot-train-boot.log

sudo -u ubuntu -i bash <<'BOOT'
set -x
cd /home/ubuntu

VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# S3 log uploader first, so the whole boot streams to the orchestrator.
"$VENV_PY" -c "import boto3" 2>/dev/null || "$VENV_PY" -m pip install boto3 2>/dev/null \
  || "$VENV_PY" -m pip install --user boto3 2>/dev/null || true
"$VENV_PY" - <<'PY' &
import time, boto3
c = boto3.client("s3")
while True:
    try:
        c.upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
    except Exception:
        pass
    time.sleep({interval})
PY
UPLOADER_PID=$!

{steps}

# Fleet deps (not in the DLAMI/baked image): FastAPI + uvicorn.
"$VENV_PY" -c "import fastapi, uvicorn" 2>/dev/null \\
  || "$VENV_PY" -m pip install fastapi uvicorn \\
  || "$VENV_PY" -m pip install --user fastapi uvicorn

set +e
source /home/ubuntu/spot-train.env

# Advertise the private IP so the router (same SG) can dial this box directly.
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \\
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
PRIVATE_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
  http://169.254.169.254/latest/meta-data/local-ipv4)
export ADVERTISE_ADDR="${{PRIVATE_IP}}:{port}"

"$VENV_PY" -u -m {module} --port {port}
RC=$?

kill "$UPLOADER_PID" 2>/dev/null || true
"$VENV_PY" - <<'PY'
import boto3
try:
    boto3.client("s3").upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
except Exception:
    pass
PY
exit "$RC"
BOOT
echo "fleet {role} exited rc=$? — box left up; run 'fleet down' to terminate it"
"""


def _orch_env_file(env: dict[str, str]) -> str:
    """systemd ``EnvironmentFile`` lines (``K="v"``). systemd strips the quotes
    and does NO shell expansion, so a value with spaces (SAMPLE_PROMPTS) is safe;
    a value containing a double quote would not be, hence the strip."""
    return "\n".join(f'{k}="{v.replace(chr(34), "")}"' for k, v in sorted(env.items()))


def build_orchestrator_user_data(
    cfg: OrchestratorConfig,
    *,
    orch_id: str,
    experiment: str,
    env: dict[str, str],
) -> str:
    """Boot the DURABLE CONTROL PLANE (``spot-orchestrate orch up``): a small
    on-demand box that runs the epoch supervisor for the whole run so the laptop
    can go to sleep.

    Deliberately different from the training boxes in four ways:

      1. **Amazon Linux 2023, not the DLAMI, and no torch.** The control plane
         never trains — it makes AWS calls and reads/writes S3. We install boto3
         into a venv and run the package from source (``PYTHONPATH``), the same
         discipline the training boxes use; ``pip install -e . --no-deps`` is
         attempted too, purely so ``spot-orchestrate`` exists for SSM debugging.
         Installing the real dependency set would drag ~2GB of CUDA torch onto an
         8GB root volume for nothing.
      2. **systemd, not a foreground boot script.** ``spot-orch.service`` owns
         the run, so it survives user-data exiting, restarts on failure, and is
         inspectable with ``systemctl status`` / ``journalctl`` over SSM.
      3. **No dead-man's switch by default.** ``MAX_INSTANCE_LIFETIME_SECONDS``
         reaps orphaned *training* boxes when the orchestrator dies; applying it
         to the orchestrator would kill the reaper mid-run. ``orch_max_lifetime_
         seconds`` is a separate, opt-in ceiling.
      4. **Progress markers.** Each provisioning phase writes
         ``orchestrators/<id>/progress.json`` so ``orch up`` on the laptop can
         show real progress (clone → install → start) rather than a blind wait.

    No credentials are passed in: the box uses its instance-profile role, which
    (unlike a copied session token) never expires — the requirement that makes an
    unattended 36-hour run possible at all.
    """
    bucket = cfg.bucket
    progress_key = cfg.orch_progress_key(orch_id)
    log_key = cfg.orch_log_key(orch_id)
    boot_log_key = cfg.orch_boot_log_key(orch_id)
    interval = cfg.orch_log_upload_seconds
    cap = cfg.orch_log_max_bytes
    # Last gate before the value is baked into user-data (which anything on the
    # box can read from IMDS): drop credential-shaped names here too, not only in
    # cfg.orch_relay_env. The box authenticates with its instance-profile role.
    secretish = set(cfg.orch_secretish(env))
    env_file = _orch_env_file({k: v for k, v in env.items() if k not in secretish})
    # Opt-in ceiling only (see the docstring): normally absent entirely.
    autokill = ""
    if cfg.orch_max_lifetime_seconds > 0:
        n = cfg.orch_max_lifetime_seconds
        autokill = (
            f'echo "[autokill] control plane self-terminates in {n}s (ORCH_MAX_LIFETIME_SECONDS)"\n'
            f"systemd-run --on-active={n}s --unit=orch-autokill /usr/bin/systemctl poweroff || true"
        )
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-orch-boot.log 2>&1
chmod 644 /var/log/spot-orch-boot.log
{autokill}
# --- interpreter + boto3 ---------------------------------------------------- #
# AL2023 ships python3.9; the package targets 3.10+, so prefer 3.11 when the
# image has it. Installed as SEPARATE commands so a missing python3.11 package
# can't take git down with it. venv keeps us off the system site-packages
# (PEP 668); if venv creation fails on some other image we fall back to the
# interpreter itself + --user.
(dnf install -y git || yum install -y git) 2>/dev/null || true
dnf install -y python3.11 python3.11-pip 2>/dev/null \\
  || dnf install -y python3-pip 2>/dev/null \\
  || yum install -y python3-pip 2>/dev/null || true
BASE_PY="$(command -v python3.11 || command -v python3)"
if "$BASE_PY" -m venv /home/ec2-user/venv 2>/dev/null; then
  ORCH_PY=/home/ec2-user/venv/bin/python
else
  ORCH_PY="$BASE_PY"
fi
"$ORCH_PY" -m pip install --upgrade pip 2>/dev/null || true
"$ORCH_PY" -m pip install boto3 || "$ORCH_PY" -m pip install --user boto3 || true

# Phase markers: one tiny S3 doc the laptop polls so `orch up` shows real
# progress. Defined only after boto3 exists (nothing to report before that).
mark() {{
  "$ORCH_PY" - "$1" "$2" <<'PY' || true
import json, sys, time, boto3
boto3.client("s3").put_object(
    Bucket="{bucket}",
    Key="{progress_key}",
    Body=json.dumps({{"phase": sys.argv[1], "detail": sys.argv[2], "at": time.time()}}).encode(),
)
PY
}}
mark provision "git + python + boto3"

# --- log relay -------------------------------------------------------------- #
# Same mechanism the training boxes use (boto3 upload_file on a timer), with one
# addition the training boxes don't need: a SIZE CAP. A 36h run's stdout would
# otherwise fill an 8GB root volume, so past orch_log_max_bytes the local file is
# trimmed to its newest half (systemd appends with O_APPEND, so the writer just
# continues at the new end).
cat > /home/ec2-user/orch-relay.py <<'PY'
import os, time
import boto3

CAP = {cap}
FILES = [
    ("/var/log/spot-orch.log", "{log_key}"),
    ("/var/log/spot-orch-boot.log", "{boot_log_key}"),
]
c = boto3.client("s3")
while True:
    for path, key in FILES:
        try:
            if os.path.getsize(path) > CAP:
                with open(path, "rb") as f:
                    f.seek(-CAP // 2, os.SEEK_END)
                    tail = b"[log trimmed: kept the newest bytes]\\n" + f.read()
                with open(path, "wb") as f:
                    f.write(tail)
            c.upload_file(path, "{bucket}", key)
        except Exception:
            pass
    time.sleep({interval})
PY
cat > /etc/systemd/system/spot-orch-relay.service <<'UNIT'
[Unit]
Description=spot-train orchestrator log relay ({orch_id})

[Service]
Type=simple
ExecStart=ORCH_PY_PATH /home/ec2-user/orch-relay.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
sed -i "s|ORCH_PY_PATH|$ORCH_PY|" /etc/systemd/system/spot-orch-relay.service
touch /var/log/spot-orch.log
systemctl daemon-reload
systemctl enable --now spot-orch-relay.service

# --- code ------------------------------------------------------------------- #
# Branch matters: the boxes this orchestrator launches clone the SAME branch, so
# `orch up` from a feature branch must not silently run main's code.
git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} /home/ec2-user/app
# No nanoGPT submodule: the control plane never imports the model (the training
# boxes init it themselves).
if [ ! -d /home/ec2-user/app/src/orchestrator ]; then
  # Fail LOUDLY and immediately: without this the box would spin through pip and
  # systemd and only surface as "no heartbeat" 30 minutes later on the laptop.
  mark error "clone of {cfg.repo_branch} failed — see boot.log"
  exit 1
fi
mark clone "$(git -C /home/ec2-user/app rev-parse --short HEAD) ({cfg.repo_branch})"

# Editable install WITHOUT dependencies (see the docstring — torch is 2GB of
# nothing on a control plane); matplotlib is optional and only renders cost.png.
"$ORCH_PY" -m pip install --no-deps -e /home/ec2-user/app || true
"$ORCH_PY" -m pip install matplotlib || true
mark install "boto3 + package (no torch)"

# --- run -------------------------------------------------------------------- #
cat > /home/ec2-user/orch.env <<'ENV'
{env_file}
ENV
cat > /etc/systemd/system/spot-orch.service <<'UNIT'
[Unit]
Description=spot-train remote orchestrator ({orch_id}: {experiment})
After=network-online.target
Wants=network-online.target
# Restart forever-ish, but stop flapping after 10 crashes in 10 minutes: at that
# point the config is broken, and a box parked for post-mortem beats a hot loop.
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/app
EnvironmentFile=/home/ec2-user/orch.env
Environment=PYTHONPATH=/home/ec2-user/app/src
Environment=PYTHONUNBUFFERED=1
ExecStart=ORCH_PY_PATH -m orchestrator orch _agent --orch-id {orch_id} --experiment {experiment}
# The agent is idempotent across restarts: it reloads orch.json, reaps the
# orphaned fleet, and RESUMES the same run_id from its S3 checkpoints.
Restart=on-failure
RestartSec=30
StandardOutput=append:/var/log/spot-orch.log
StandardError=append:/var/log/spot-orch.log

[Install]
WantedBy=multi-user.target
UNIT
sed -i "s|ORCH_PY_PATH|$ORCH_PY|" /etc/systemd/system/spot-orch.service
chown -R ec2-user:ec2-user /home/ec2-user
systemctl daemon-reload
systemctl enable --now spot-orch.service
mark start "spot-orch.service"
echo "[orch] user-data done — the run is systemd's now; this box outlives your laptop"
"""


def build_provisioning_user_data(
    cfg: OrchestratorConfig, *, run_id: str, market: str, max_seconds: int
) -> str:
    """Provision only: get the code + deps onto the box, then leave it UP for SSH.
    Does NOT run training and does NOT shut the box down."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    steps = _provision_steps(cfg, _export_block(env))
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
echo "[provision] start"

sudo -u ubuntu -i bash <<'BOOT'
{steps}
echo "[provision] DONE"
echo "[provision] code: /home/ubuntu/app   env: /home/ubuntu/spot-train.env"
echo "[provision] to train by hand:"
echo "    cd /home/ubuntu/app && source /home/ubuntu/spot-train.env"
echo "    /opt/pytorch/bin/python -m spot_train.train"
BOOT
echo "[provision] user-data exited rc=$? — box left UP for SSH (no training, no shutdown)"
"""


def build_bake_user_data(cfg: OrchestratorConfig, *, bake_id: str, base_ami: str) -> str:
    """Provision a box for IMAGING (``spot-orchestrate bake-ami``): clone the repo
    + nanoGPT submodule and install boto3 — the same steps every training boot
    performs, minus anything run-specific — then write a status marker to S3 so
    the orchestrator knows to stop the box and call CreateImage.

    Deliberately bakes NO run state: no spot-train.env (each training boot writes
    its own), no dataset (stays in S3 — EBS lazy-restore would make baked bins
    first-touch-slow anyway), no credentials (the instance profile is IMDS-only,
    nothing lands on disk). The clone IS baked, and stays current because
    ``_provision_steps`` fast-forwards an existing clone at every boot."""
    bucket = cfg.bucket
    status_key = cfg.bake_status_key(bake_id)
    log_key = cfg.bake_log_key(bake_id)
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-bake.log 2>&1
chmod 644 /var/log/spot-train-bake.log   # let the ubuntu log-uploader read it

sudo -u ubuntu -i bash <<'BOOT'
set -x
cd /home/ubuntu

# Interpreter: venv python by absolute path (same DLAMI reality as training boots).
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# boto3 FIRST — the log uploader and the status marker need it even if the
# provisioning below fails. (Also the one pip dep the bake exists to pre-install.)
"$VENV_PY" -c "import boto3" 2>/dev/null || "$VENV_PY" -m pip install boto3 2>/dev/null \\
  || "$VENV_PY" -m pip install --user boto3 2>/dev/null || true
"$VENV_PY" - <<'PY' &
import time, boto3
c = boto3.client("s3")
while True:
    try:
        c.upload_file("/var/log/spot-train-bake.log", "{bucket}", "{log_key}")
    except Exception:
        pass
    time.sleep(10)
PY
UPLOADER_PID=$!

# Provision in a subshell so a failure is captured as RC, not a silent exit —
# the status marker below must ALWAYS be written.
(
  set -euxo pipefail
  if [ -d app/.git ]; then
    git -C app fetch --depth 1 origin {cfg.repo_branch}
    git -C app reset --hard FETCH_HEAD
  else
    git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
  fi
  cd app
  git submodule update --init --depth 1
  git rev-parse HEAD > /home/ubuntu/bake-commit
  "$VENV_PY" - <<'PY'
import torch, boto3, numpy  # noqa: F401
print("bake preflight ok: torch", torch.__version__, flush=True)
PY
)
RC=$?

kill "$UPLOADER_PID" 2>/dev/null || true
"$VENV_PY" - <<PY
import json
import boto3
c = boto3.client("s3")
try:
    c.upload_file("/var/log/spot-train-bake.log", "{bucket}", "{log_key}")
except Exception:
    pass
try:
    commit = open("/home/ubuntu/bake-commit").read().strip()
except Exception:
    commit = ""
c.put_object(
    Bucket="{bucket}",
    Key="{status_key}",
    Body=json.dumps(
        {{"ok": $RC == 0, "rc": $RC, "commit": commit, "base_ami": "{base_ami}"}}
    ).encode(),
)
PY
exit "$RC"
BOOT
echo "bake user-data exited rc=$? — box left UP for the orchestrator (stop + CreateImage on ok)"
"""


def _multinode_loop(
    cfg: OrchestratorConfig, *, run_id: str, nodes: int, node_index: int, nproc: str
) -> str:
    """Bash for the multi-node EPOCH loop: hand control to the Python sidecar.

    The box no longer negotiates a rendezvous with its peers. It registers its
    IP, then the sidecar (``orchestrator.sidecar``) polls the orchestrator's
    ``epoch.json`` and runs STATIC torchrun for whatever membership the
    orchestrator published — killing and relaunching torchrun when the epoch
    advances (a peer died, or a replacement joined). All the membership logic
    lives in the supervisor + sidecar Python (tested, our logs), not in
    torchrun's dynamic rendezvous (a version-dependent black box that hung on
    the DLAMI's torch). ``nproc`` is passed through via NPROC_PER_NODE so the
    sidecar's static torchrun uses one rank per GPU. The sidecar exits 0 on
    metrics.json (run done) and nonzero if its idle budget lapses (box left up
    for the orchestrator's whole-group-restart watchdog)."""
    run_uri = cfg.run_uri(run_id)
    return f"""
# --- multi-node: epoch protocol (orchestrator owns membership) ---------------
# Register this box's IP, then let the sidecar obey epoch.json: static torchrun
# per epoch, relaunched on every membership change. NPROC_PER_NODE picks the
# per-box rank count (gpu = one per GPU). metrics.json is the group-wide done
# signal; the sidecar's exit code is this script's exit code.
export NPROC_PER_NODE="{nproc}"
"$VENV_PY" -m orchestrator.sidecar --run-uri "{run_uri}" --node-index {node_index}
MN_RC=$?
(exit "$MN_RC")
"""


def build_user_data(
    cfg: OrchestratorConfig,
    *,
    run_id: str,
    market: str,
    max_seconds: int,
    logs_key: str | None = None,
    ddp: bool = False,
    nproc_per_node: int = 0,
    nodes: int = 1,
    node_index: int = 0,
) -> str:
    """Full run: provision, then train under the wall-clock budget while syncing
    the boot log to S3 every ``log_stream_seconds`` so the orchestrator can stream
    it live without SSH. On success the box self-terminates as a cost backstop (the
    orchestrator also terminates it once it sees metrics.json). On failure the box
    stays up for debugging; the orchestrator reaps it on the metrics timeout.

    ``ddp=True`` runs the trainer under torchrun (single-node DDP); the trainer
    detects RANK and joins the process group. ``nproc_per_node <= 0`` auto-detects
    one rank per GPU on the box (torchrun --nproc_per_node=gpu); a positive value
    forces that count. ``ddp=False`` is the plain single-process launch
    (baseline/preempt) — byte-identical to before.

    ``nodes > 1`` (implies ddp) hands the box to the epoch sidecar (see
    ``_multinode_loop``): it registers its IP and runs STATIC torchrun for
    whatever membership the orchestrator publishes in epoch.json, relaunching on
    every epoch change. A peer's death crashes this box's torchrun (NCCL_TIMEOUT)
    as a backstop, but the primary signal is the orchestrator publishing the
    next epoch — the sidecar kills and relaunches within one ~3s poll. Every
    (re)start resumes via the one proven resume path — the node-local checkpoint
    tier when this box has the agreed step, else S3. No node hosts a rendezvous
    store, so any node (including epoch rank 0) is freely killable."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    if nodes > 1:
        # Short collective timeout so survivors of a node kill abort fast (see
        # distributed.init) — the in-band backstop to the supervisor's epoch bump.
        env["NCCL_TIMEOUT"] = str(cfg.nccl_timeout_seconds)
        # NCCL network hygiene (why 4-node hung where 2-node worked): with no
        # interface pinned, NCCL auto-selects one — and the DLAMI ships Docker, so
        # every box has a docker0 on the SAME non-routable 172.17.0.0/16. NCCL can
        # advertise those addresses; inter-node connects then hang (worse the more
        # nodes) and the first collective never completes -> NCCL_TIMEOUT abort.
        # Exclude docker0/lo so NCCL uses the real VPC ENI; disable IB (none on
        # g4dn/g5); WARN-level debug so any remaining net issue is IN the log.
        env["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "^docker0,lo")
        env["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
        env["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")
        # Force the built-in TCP socket transport and DON'T load the aws-ofi-nccl
        # (EFA/libfabric) net plugin. The newer DLAMI auto-loads that plugin, but
        # g4dn/g5.xlarge have NO EFA, so its init fails ("NET/OFI Unable to find a
        # protocol that worked") and NCCL >=2.29 does not cleanly fall back — the
        # first inter-node all-reduce then hangs to NCCL_TIMEOUT and the run never
        # trains. Sockets work (the c10d rendezvous already proves inter-node TCP).
        # Operator-overridable for a real EFA fleet (NCCL_NET=OFI).
        env["NCCL_NET"] = os.environ.get("NCCL_NET", "Socket")
        env["NCCL_NET_PLUGIN"] = os.environ.get("NCCL_NET_PLUGIN", "none")
        # Relaunch-after-kill can find the GPU fragmented or a just-reaped worker's
        # allocation still settling; expandable segments let the allocator grow into
        # freed space instead of OOM-ing on a large contiguous request. Cheap and
        # safe; operator-overridable.
        env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        # No EFA on g4dn/g5.xlarge, so the gradient all-reduce rides bare TCP,
        # whose NCCL defaults (1 thread x 1 socket) leave bandwidth on the table.
        # Parallelize the socket transport: ~2-4x all-reduce throughput, the
        # cheapest fix for the comms-bound multi-node step. Operator-overridable.
        env["NCCL_SOCKET_NTHREADS"] = os.environ.get("NCCL_SOCKET_NTHREADS", "4")
        env["NCCL_NSOCKS_PERTHREAD"] = os.environ.get("NCCL_NSOCKS_PERTHREAD", "2")
        # Skip torch's post-timeout debug-info dump: it added ~2 minutes to every
        # peer-death crash (observed 18:26:51 -> 18:29:10), delaying the relaunch.
        env["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"
        # Run-level budget: rides in the checkpoint (trained_seconds), so every
        # epoch's torchrun computes its own remaining time — no budget.json, and
        # boot/NCCL-stall/teardown are never billed. MAX_SECONDS still gets a
        # value but the trainer overrides it from this once resumed.
        env["TRAIN_BUDGET_SECONDS"] = str(max_seconds)
        # Node-local checkpoint tier: survivors of an epoch change resume from
        # their own disk instead of re-downloading from S3.
        env["LOCAL_CHECKPOINT_DIR"] = "/tmp/spot-ckpt"
    steps = _provision_steps(cfg, _export_block(env))
    bucket = cfg.bucket
    logs_key = logs_key or cfg.run_logs_key(run_id)
    interval = cfg.log_stream_seconds
    if nodes > 1:
        nproc = "gpu" if nproc_per_node <= 0 else str(nproc_per_node)
        run_cmd = _multinode_loop(
            cfg, run_id=run_id, nodes=nodes, node_index=node_index, nproc=nproc
        )
    elif ddp:
        nproc = "gpu" if nproc_per_node <= 0 else str(nproc_per_node)
        # nproc=gpu => torchrun uses torch.cuda.device_count() (one rank per GPU).
        # OMP_NUM_THREADS=1: N gloo ranks on one CPU box otherwise spawn N×cores
        # threads and thrash. torchrun --standalone sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*.
        run_cmd = (
            f'OMP_NUM_THREADS=1 "$VENV_PY" -m torch.distributed.run '
            f"--standalone --nproc_per_node={nproc} -m spot_train.train"
        )
    else:
        run_cmd = '"$VENV_PY" -u -m spot_train.train'
    # Dead-man's switch (runs as root, at boot): a systemd timer that OS-poweroffs
    # the box after max_instance_lifetime_seconds no matter what — the timer is
    # systemd-owned, so it survives the boot script exiting and a dead orchestrator.
    # poweroff + InstanceInitiatedShutdownBehavior=terminate => the instance
    # terminates (billing stops). Falls back to `shutdown` if systemd-run is absent.
    autokill = ""
    if cfg.max_instance_lifetime_seconds > 0:
        n = cfg.max_instance_lifetime_seconds
        mins = max(1, -(-n // 60))  # ceil(n/60) for the shutdown fallback (minutes)
        autokill = (
            f'echo "[autokill] self-terminate scheduled in {n}s (dead-man switch)"\n'
            f"systemd-run --on-active={n}s --unit=spot-autokill /usr/bin/systemctl poweroff "
            f'|| shutdown -h +{mins} "spot-train autokill" || true'
        )
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
chmod 644 /var/log/spot-train-boot.log   # let the ubuntu log-uploader read it
{autokill}
sudo -u ubuntu -i bash <<'BOOT'
set -x
cd /home/ubuntu

# Interpreter: use the pytorch venv python by absolute path — the base shell only
# has `python3`, and the login shell doesn't reliably auto-activate the venv, so
# a bare `python` fails (command not found). Fall back to python3 on other AMIs.
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# Start the S3 log uploader FIRST — before clone/pip — so the *entire* boot
# (clone, submodule, deps, preflight, then training) streams to the orchestrator,
# not just the training tail. boto3 is required for it, so ensure it up front
# (the provision steps below install it too; this is idempotent). Uses the
# instance-profile role via IMDS — no creds in user-data.
"$VENV_PY" -c "import boto3" 2>/dev/null || "$VENV_PY" -m pip install boto3 2>/dev/null \
  || "$VENV_PY" -m pip install --user boto3 2>/dev/null || true
"$VENV_PY" - <<'PY' &
import time, boto3
c = boto3.client("s3")
while True:
    try:
        c.upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
    except Exception:
        pass
    time.sleep({interval})
PY
UPLOADER_PID=$!

# Provision (clone repo + nanoGPT submodule, deps, env file, preflight) — now streamed.
{steps}

# From here we manage exit codes by hand (final log flush + teardown), so drop -e.
set +e
# Load the run config the trainer reads (PYTHONPATH so `spot_train` imports, plus
# the S3 URIs, MAX_SECONDS, etc.). Provisioning wrote this file but doesn't source it.
source /home/ubuntu/spot-train.env
{run_cmd}
RC=$?

# Stop the uploader and push one final copy so the tail of the run is in S3.
kill "$UPLOADER_PID" 2>/dev/null || true
"$VENV_PY" - <<'PY'
import boto3
try:
    boto3.client("s3").upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
except Exception:
    pass
PY
exit "$RC"
BOOT
RC=$?

if [ "$RC" -eq 0 ]; then
  shutdown -h now   # cost backstop; orchestrator also terminates on metrics.json
else
  echo "training exited $RC — leaving instance up for debugging; orchestrator reaps on timeout"
fi
"""
