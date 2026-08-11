"""Orchestrator configuration.

All values have defaults except the S3 bucket, which you must set (it's globally
unique). Everything is overridable via environment variables so you can keep the
concrete names in your git-ignored ``.env`` rather than in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


# Sentinel victim meaning "whichever node is master when this kill fires".
# Resolved by the supervisor at fire time, not at schedule-build time: the
# master moves (elect_master is sticky to a SURVIVOR), so a hardcoded index
# would quietly stop testing re-election after the first kill.
LEADER_VICTIM = -1

# Trainer knobs the orchestrator relays verbatim (only when set in ITS
# environment): the convergence recipe + periodic eval/sample cadence. The
# orchestrator never branches on these values, so they stay untyped strings —
# TrainConfig.from_env parses them on the box.
_TRAINER_PASSTHROUGH = (
    "N_LAYER",
    "N_HEAD",
    "N_EMBD",
    "BLOCK_SIZE",
    "TARGET_LOSS",
    "MAX_STEPS",
    "GLOBAL_BATCH_SIZE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "DROPOUT",
    "DTYPE",
    "DDP_COMM_HOOK",
    "WARMUP_STEPS",
    "LR_DECAY_STEPS",
    "MIN_LR",
    "GRAD_CLIP",
    "CHECKPOINT_ASYNC",
    # Without this the durable-tier prune depth is stuck at TrainConfig's default
    # on the box: pruning still happens, but the recipe cannot tune it and
    # CHECKPOINT_KEEP=0 cannot turn it off. A knob that silently does nothing is
    # worse than no knob.
    "CHECKPOINT_KEEP",
    "LOG_INTERVAL_STEPS",
    "EVAL_INTERVAL_STEPS",
    "SAMPLE_INTERVAL_STEPS",
    "SAMPLE_INTERVAL_PROMPTS",
    "SAMPLE_INTERVAL_TOKENS",
    "SAMPLE_MAX_NEW_TOKENS",
    "SAMPLE_TEMPERATURE",
    "SAMPLE_TOP_K",
    "SAMPLES_PER_PROMPT",
)

# Orchestrator knobs the REMOTE control plane inherits from your shell/.env when
# you run `orch up`. An ALLOWLIST, never a blanket copy of os.environ: the env
# lands in EC2 user-data (readable via IMDS by anything on the box), so
# credentials and API keys must never ride along — boto3 on the box resolves its
# creds from the attached instance-profile role instead.
_ORCH_RELAY_ENV = (
    "SPOT_TRAIN_BUCKET",
    "AWS_REGION",
    "REPO_URL",
    "REPO_BRANCH",
    "INSTANCE_TYPE",
    "AMI_ID",
    "AMI_NAME_FILTER",
    "SSH_KEY_NAME",
    "IAM_ROLE",
    "IAM_PROFILE",
    "SECURITY_GROUP",
    "MAX_INSTANCE_LIFETIME_SECONDS",
    "DATASET",
    "HOURLY_USD",
    "BASELINE_SECONDS",
    "SPOT_SEG1_SECONDS",
    "SPOT_SEG2_SECONDS",
    "CHECKPOINT_INTERVAL_SECONDS",
    "EVAL_ITERS",
    "BATCH_SIZE",
    "MARKET",
    "TRAIN_TOTAL_SECONDS",
    "PREEMPT_COUNT",
    "PREEMPT_GRACE",
    "PREEMPT_AFTER",
    "PREEMPT_CHECKPOINT_SECONDS",
    "PREEMPT_VICTIMS",
    # Without these the REMOTE control plane silently falls back to the
    # evenly-spaced single-victim schedule -- i.e. the mass-loss event just
    # would not happen, and the run would look like it worked.
    "PREEMPT_SCHEDULE",
    "INSTANCE_LIFETIME_SLACK_SECONDS",
    "SMOKE_TEST_EVERY",
    "SAMPLE_PROMPTS",
    "DDP_NPROC_PER_NODE",
    "DDP_DATA_MODE",
    "NODES",
    "RDZV_PORT",
    "NCCL_TIMEOUT",
    "NCCL_INIT_TIMEOUT",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_DEBUG",
    "NCCL_NET",
    "NCCL_NET_PLUGIN",
    "NCCL_SOCKET_NTHREADS",
    "NCCL_NSOCKS_PERTHREAD",
    "RECOVERY_TIMEOUT",
    "VCPU_QUOTA",
    "INSTANCE_VCPUS",
    "METRICS_TIMEOUT",
    "METRICS_OVERHEAD",
    "S3_MAX_CONCURRENCY",
    "S3_CHUNK_MB",
    "LOG_STREAM_SECONDS",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_GROUP",
    "WANDB_DISABLED",
)

# Belt-and-braces: any name that smells like a credential is dropped from the
# relayed env even if it was explicitly passed with `--env`. user-data is not a
# secret store.
_SECRETISH = ("SECRET", "PASSWORD", "TOKEN", "API_KEY", "ACCESS_KEY", "CREDENTIAL")

# vCPUs per instance type, for the quota-headroom gate. Only the types this
# project plausibly launches; anything else needs INSTANCE_VCPUS set explicitly.
_INSTANCE_VCPUS = {
    "g4dn.xlarge": 4,
    "g4dn.2xlarge": 8,
    "g4dn.4xlarge": 16,
    "g4dn.12xlarge": 48,
    "g5.xlarge": 4,
    "g5.2xlarge": 8,
    "g5.12xlarge": 48,
    "g6.xlarge": 4,
    "g6.12xlarge": 48,
}


# On-demand $/hr (us-east-1, Linux) for the cost ledger. Spot rates are NOT
# listed here — they move hourly and vary per AZ, so they're queried live at
# launch (aws.spot_hourly_rate). Types missing from this table need HOURLY_USD.
ON_DEMAND_HOURLY_USD = {
    "g4dn.xlarge": 0.526,
    "g4dn.2xlarge": 0.752,
    "g4dn.12xlarge": 3.912,
    "g5.xlarge": 1.006,
    "g6.xlarge": 0.805,
    # Control-plane boxes (`orch up`): tiny, on-demand, and long-lived, so their
    # cost must show up in the ledger for a multi-day run to be honest.
    "t3.micro": 0.0104,
    "t3.small": 0.0208,
    "t3.medium": 0.0416,
    # Dataset-prep boxes (`stage-data --remote`): CPU-only, one hour, one job.
    "c6i.2xlarge": 0.34,
    "c6i.4xlarge": 0.68,
    "c6i.8xlarge": 1.36,
}


@dataclass
class OrchestratorConfig:
    # --- AWS placement -------------------------------------------------------
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    instance_type: str = field(default_factory=lambda: _env("INSTANCE_TYPE", "g4dn.xlarge"))
    # Dead-man's switch: seconds after boot a training box self-terminates even if
    # the orchestrator never sends TerminateInstances (laptop crash, network drop,
    # kill -9). 0 = off. The box OS-poweroffs and InstanceInitiatedShutdownBehavior
    # (=terminate) turns that into a real termination, so billing stops. A normal
    # run ends far sooner, so this only ever catches orphans.
    max_instance_lifetime_seconds: int = field(
        default_factory=lambda: _env_int("MAX_INSTANCE_LIFETIME_SECONDS", 0)
    )
    # Slack added to a run's own budget when DERIVING the dead-man timer: boot,
    # the dataset pull and the post-budget eval/sample tail all happen outside
    # TRAIN_BUDGET_SECONDS. 1h is generous against a measured ~4min boot + ~2min
    # dataset, because a timer that fires DURING a healthy run is far worse than
    # one that fires late.
    instance_lifetime_slack_seconds: int = field(
        default_factory=lambda: _env_int("INSTANCE_LIFETIME_SLACK_SECONDS", 3600)
    )
    # Deep Learning AMI. If AMI_ID is set we use it verbatim; otherwise we resolve
    # the newest Amazon-owned image matching this name filter via DescribeImages.
    # Default targets the PyTorch DLAMI (Ubuntu 22.04) so CUDA + PyTorch are
    # preinstalled and user-data does no GPU/torch setup.
    ami_id: str = field(default_factory=lambda: _env("AMI_ID", ""))
    ami_name_filter: str = field(
        default_factory=lambda: _env(
            "AMI_NAME_FILTER",
            "Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*",
        )
    )

    # SSH-verification mode: name of an EXISTING EC2 key pair in `region` to
    # attach so you can ssh into the box. Blank = launch without SSH access.
    key_name: str = field(default_factory=lambda: _env("SSH_KEY_NAME", ""))

    # --- names created by `setup` (you own these) ---------------------------
    # These are TRAINING's values. The inference platform never reads them —
    # see `for_inference()`, which sources INFERENCE_*-prefixed vars instead.
    bucket: str = field(default_factory=lambda: _env("SPOT_TRAIN_BUCKET", ""))
    # Stamped on every instance as tag:project. Teardown scripts scope on it, so
    # the two projects must never share a value — an inference box tagged
    # spot-train would be invisible to the inference reaper and bill unnoticed.
    project_tag: str = "spot-train"
    role_name: str = field(default_factory=lambda: _env("IAM_ROLE", "spot-train-role"))
    instance_profile: str = field(default_factory=lambda: _env("IAM_PROFILE", "spot-train-profile"))
    security_group: str = field(default_factory=lambda: _env("SECURITY_GROUP", "spot-train-sg"))

    # --- S3 key layout -------------------------------------------------------
    run_prefix: str = "runs"
    data_prefix: str = "data"

    # --- code delivery -------------------------------------------------------
    repo_url: str = field(
        default_factory=lambda: _env("REPO_URL", "https://github.com/miguelaenlle/OptiTrain.git")
    )
    repo_branch: str = field(default_factory=lambda: _env("REPO_BRANCH", "main"))

    # --- AMI baking (spot-orchestrate bake-ami) ------------------------------
    # Instance type for the throwaway bake box. pip installs are arch-independent
    # and the DLAMI boots fine without a GPU, so a cheap CPU box is the default —
    # it also keeps the bake entirely off the G-vCPU quota.
    bake_instance_type: str = field(default_factory=lambda: _env("BAKE_INSTANCE_TYPE", "t3.xlarge"))
    # Seconds to wait for the bake box's provisioning to write its status marker.
    bake_timeout_seconds: int = field(default_factory=lambda: _env_int("BAKE_TIMEOUT", 1200))
    # Baked AMIs to retain (newest first); older ones are deregistered and their
    # snapshots deleted after a successful bake. Snapshots bill monthly.
    bake_keep_images: int = field(default_factory=lambda: _env_int("BAKE_KEEP_IMAGES", 2))

    # --- experiment knobs ----------------------------------------------------
    dataset: str = field(default_factory=lambda: _env("DATASET", "shakespeare_char"))
    # $/hr override for the cost ledger: pins the on-demand rate when the
    # instance type isn't in ON_DEMAND_HOURLY_USD (or to correct it for another
    # region). 0 = use the table. Spot rows always use the live queried price.
    hourly_usd: float = field(default_factory=lambda: _env_float("HOURLY_USD", 0.0))
    baseline_seconds: int = field(default_factory=lambda: _env_int("BASELINE_SECONDS", 300))
    spot_seg1_seconds: int = field(default_factory=lambda: _env_int("SPOT_SEG1_SECONDS", 120))
    spot_seg2_seconds: int = field(default_factory=lambda: _env_int("SPOT_SEG2_SECONDS", 180))
    checkpoint_interval_seconds: int = field(
        default_factory=lambda: _env_int("CHECKPOINT_INTERVAL_SECONDS", 30)
    )
    eval_iters: int = field(default_factory=lambda: _env_int("EVAL_ITERS", 200))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 12))

    # Market the spot-style experiments (spot/preempt/ddp-preempt) launch in.
    # MARKET=on-demand runs the same kill/resume mechanics on on-demand capacity —
    # useful when the spot vCPU quota is exhausted. baseline/ddp are always on-demand.
    spot_market: str = field(default_factory=lambda: _env("MARKET", "spot"))

    # --- preemption experiment ----------------------------------------------
    # Total TRAINING seconds to accumulate across all segments (kills don't count).
    train_total_seconds: int = field(default_factory=lambda: _env_int("TRAIN_TOTAL_SECONDS", 180))
    # Number of preemptions to perform. The total training is split evenly across
    # (preempt_count + 1) segments — so 1 => train, kill once, reboot, finish. The
    # node is NOT told the schedule; it only gets its remaining budget as MAX_SECONDS.
    preempt_count: int = field(default_factory=lambda: _env_int("PREEMPT_COUNT", 1))
    # Seconds to wait for the trainer's SIGTERM checkpoint to land before terminating.
    preempt_grace_seconds: int = field(default_factory=lambda: _env_int("PREEMPT_GRACE", 90))
    # Seconds of training before each kill. 0 (default) = split train_total_seconds
    # evenly across segments. Set small (e.g. PREEMPT_AFTER=15) to exercise the
    # kill/resume path fast while debugging; the number of kills stays preempt_count.
    preempt_after_seconds: int = field(default_factory=lambda: _env_int("PREEMPT_AFTER", 0))
    # Checkpoint interval while a kill schedule is armed. Bounds lost work per
    # hard kill (no warning) — but it is paid on EVERY step of the whole run, so
    # it must be priced against the checkpoint's real size. 5s was right when a
    # Shakespeare checkpoint was a few MB; at GPT-2 124M a checkpoint is 1.5 GB,
    # and 5s meant serializing + pushing 1.5 GB toward S3 continuously on the
    # same NIC the gradient all-reduce rides. Measured on the 2-kill 1h run:
    # 6330 ms/step vs 4013 clean — a +58% tax on every step, dwarfing the
    # downtime it was meant to bound. At 60s the worst case loses ~15 steps
    # (~1 min) per kill, and the tax drops to ~2-3%.
    preempt_checkpoint_seconds: int = field(
        default_factory=lambda: _env_int("PREEMPT_CHECKPOINT_SECONDS", 60)
    )
    # How often the trainer runs the (noisy) checkpoint verify+smoke test. Set per
    # experiment so frequent preemption checkpoints don't flood the loss output.
    smoke_test_every: int = field(default_factory=lambda: _env_int("SMOKE_TEST_EVERY", 1))

    # --- end-of-run + periodic text samples ----------------------------------
    # JSON array of prompts the trainer samples from at the end of a run (and at
    # SAMPLE_INTERVAL_STEPS snapshots). Relayed to the box base64-encoded so the
    # env file's export K="v" quoting can't be broken by quotes/newlines.
    sample_prompts: str = field(
        default_factory=lambda: _env("SAMPLE_PROMPTS", '["ROMEO:", "JULIET:", "First Citizen:"]')
    )

    # --- multinode-preempt victim schedule ------------------------------------
    # Comma-separated node index to hard-kill per preemption round, e.g. "1,0"
    # (kill node 1 first, then node 0). Any node is killable — the epoch after a
    # kill just names a new lowest-index master. Empty = always the last node.
    preempt_victims: str = field(default_factory=lambda: _env("PREEMPT_VICTIMS", ""))
    # Explicit chaos schedule; when set it WINS over preempt_victims/count. See
    # preempt_schedule() for the grammar. This is what lets a single event kill
    # several nodes at once.
    preempt_schedule_spec: str = field(default_factory=lambda: _env("PREEMPT_SCHEDULE", ""))

    # --- DDP experiment (spot-orchestrate ddp) ------------------------------
    # Ranks torchrun launches on the box. 0 (default) = auto: one rank per GPU on
    # the machine (torchrun --nproc_per_node=gpu). Set a positive value to force a
    # fixed count — needed to exercise multi-rank DDP on a CPU-only box.
    ddp_nproc_per_node: int = field(default_factory=lambda: _env_int("DDP_NPROC_PER_NODE", 0))
    # "shard" (real data-parallel) | "replicate" (identical data, determinism check).
    ddp_data_mode: str = field(default_factory=lambda: _env("DDP_DATA_MODE", "shard"))

    # --- multi-node experiment (spot-orchestrate multinode) ------------------
    # Nodes in the training group; each runs torchrun with one rank per GPU.
    # The orchestrator owns membership: it publishes runs/<run_id>/epoch.json
    # (who is in the group, their ranks, the master addr/port) and every box's
    # sidecar polls it and runs STATIC torchrun for the current epoch. Node 0 of
    # an epoch is just the lowest live node index — no node hosts a rendezvous
    # store, so any node is killable.
    node_count: int = field(default_factory=lambda: _env_int("NODES", 2))
    # Base for the per-epoch master port: master_port = rdzv_port + epoch, so a
    # relaunched master never fights TIME_WAIT on its own previous socket.
    rdzv_port: int = field(default_factory=lambda: _env_int("RDZV_PORT", 29400))
    # Collective timeout exported to multi-node boxes (torch's default is 10
    # minutes). Under the epoch supervisor a survivor's torchrun is normally
    # killed by its sidecar the moment the shrink epoch lands (~3s), so this is
    # the IN-BAND BACKSTOP: if the supervisor is slow, the survivor's collective
    # still aborts here rather than hanging. 20s keeps >10x margin over the worst
    # legitimate stall at this model size (an async-checkpoint snapshot or a slow
    # TCP allreduce is well under 2s).
    nccl_timeout_seconds: int = field(default_factory=lambda: _env_int("NCCL_TIMEOUT", 20))
    # No-checkpoint-progress this long => the whole group is wedged (e.g. a torchrun
    # rendezvous that can't converge) => whole-group restart. The deadlock-breaker
    # of last resort. Must sit ABOVE the worst-case LEGITIMATE no-progress window
    # so it never false-fires mid-recovery, yet well under METRICS_TIMEOUT so a
    # genuine hang is broken within a run instead of stalling to the deadline.
    #
    # 150s was that floor when the dataset was Shakespeare (a whole-group reboot
    # was ~45-70s boot + restore + first checkpoint ~= 90s). Full OpenWebText
    # invalidated it: a REPLACEMENT must boot (~2 min) AND pull a 17 GB train.bin
    # (measured 117s) before it can take a step, so legitimate recovery is ~4-5
    # min. At 150s the supervisor gave up mid-recovery and restarted the whole
    # group — discarding healthy survivors — on every single preemption, which is
    # the exact opposite of what the shrink-and-continue design exists to do.
    # Observed live on an 8-node run: 3 whole-group restarts, zero real progress.
    #
    # Sized for boot + dataset pull + collective re-init, with headroom. If the
    # corpus grows again, this has to grow with it — it is a function of how long
    # a replacement takes to become useful, not a free-floating constant.
    recovery_timeout_seconds: int = field(default_factory=lambda: _env_int("RECOVERY_TIMEOUT", 600))

    # --- remote orchestrator (spot-orchestrate orch up) ----------------------
    # The durable control plane: one ALWAYS-ON-DEMAND box that runs the epoch
    # supervisor so a 36h run doesn't need your laptop awake. Never spot — if the
    # control plane is reclaimed mid-run there is nobody left to replace it.
    orch_instance_type: str = field(default_factory=lambda: _env("ORCH_INSTANCE_TYPE", "t3.micro"))
    # A plain Amazon Linux 2023 image, NOT the DLAMI: the control plane never
    # trains, so it needs no CUDA/torch and a small root volume is a feature (a
    # 100GB DLAMI volume would bill for the whole run). Kept separate from
    # AMI_ID/AMI_NAME_FILTER so a DLAMI pin in your .env can't leak in here.
    orch_ami_id: str = field(default_factory=lambda: _env("ORCH_AMI_ID", ""))
    orch_ami_name_filter: str = field(
        default_factory=lambda: _env("ORCH_AMI_NAME_FILTER", "al2023-ami-2023.*-x86_64")
    )
    # The control plane's OWN instance profile: it launches/terminates training
    # boxes, so it needs more than the worker role (see docs/iam/). Separate role
    # = the training boxes never inherit EC2 lifecycle rights.
    orch_role_name: str = field(
        default_factory=lambda: _env("ORCH_IAM_ROLE", "spot-train-orch-role")
    )
    orch_instance_profile: str = field(
        default_factory=lambda: _env("ORCH_IAM_PROFILE", "spot-train-orch-profile")
    )
    # How often the on-box agent republishes heartbeat.json (liveness + live
    # step/loss/cost for `orch status`).
    orch_heartbeat_seconds: int = field(default_factory=lambda: _env_int("ORCH_HEARTBEAT", 10))
    # Heartbeat older than this => the control plane is presumed wedged/gone.
    orch_stale_seconds: int = field(default_factory=lambda: _env_int("ORCH_STALE_SECONDS", 60))
    # Log relay cadence, and the cap that keeps 36h of stdout from filling an 8GB
    # root volume — the local file is trimmed to its newest half past this size.
    orch_log_upload_seconds: int = field(default_factory=lambda: _env_int("ORCH_LOG_UPLOAD", 15))
    orch_log_max_bytes: int = field(
        default_factory=lambda: _env_int("ORCH_LOG_MAX_BYTES", 32 * 1024 * 1024)
    )
    # Dead-man's switch for the CONTROL PLANE, deliberately its own knob and
    # deliberately 0 (off) by default: MAX_INSTANCE_LIFETIME_SECONDS exists to
    # reap orphaned *training* boxes when the orchestrator dies, and applying it
    # here would kill the very process that does the reaping mid-run. Set it only
    # if you want a hard ceiling comfortably above the run budget.
    orch_max_lifetime_seconds: int = field(
        default_factory=lambda: _env_int("ORCH_MAX_LIFETIME_SECONDS", 0)
    )
    # `orch up` gives up waiting for the box to boot + provision after this.
    orch_boot_timeout_seconds: int = field(
        default_factory=lambda: _env_int("ORCH_BOOT_TIMEOUT", 1800)
    )

    # --- remote dataset prep (spot-orchestrate stage-data --remote) ----------
    # One throwaway on-demand box prepares the corpus IN AWS and uploads the bins
    # same-region (free, minutes) instead of pushing 17 GB up a home uplink.
    # CPU is NOT the bottleneck — tiktoken does ~29 MB/s/core, so even 4 cores
    # tokenize OpenWebText in minutes; the HF download, the ~52 GB of cache
    # writes and the S3 upload are. 16 vCPU keeps that pipeline saturated for
    # ~$0.68/hr on a job that runs about an hour.
    prep_instance_type: str = field(
        default_factory=lambda: _env("PREP_INSTANCE_TYPE", "c6i.4xlarge")
    )
    # Root volume, sized EXPLICITLY: OpenWebText needs ~110 GB transient (54 GB
    # HF cache + ~35 GB tokenized arrow + 17 GB bins), and every AMI default is
    # far below that (AL2023 8 GB, DLAMI 30 GB). Inheriting the AMI's mapping —
    # which is what every other launch in this repo does — would fill the disk
    # ~40 minutes in.
    prep_volume_gb: int = field(default_factory=lambda: _env_int("PREP_VOLUME_GB", 200))
    # gp3 defaults to 125 MB/s and 3000 IOPS. ~52 GB of cache writes at 125 MB/s
    # is ~7 minutes of pure disk wait on the critical path; 500 MB/s costs cents
    # for a one-hour volume. gp3 caps throughput at 0.25 MB/s per provisioned
    # IOPS, so 500 MB/s needs >=2000 IOPS — 6000 leaves headroom for the many
    # small random writes the arrow cache makes.
    prep_volume_throughput: int = field(
        default_factory=lambda: _env_int("PREP_VOLUME_THROUGHPUT", 500)
    )
    prep_volume_iops: int = field(default_factory=lambda: _env_int("PREP_VOLUME_IOPS", 6000))
    # Dead-man's switch — MANDATORY here (unlike the control plane's opt-in
    # ceiling): this box exists to run ONE unattended job, and the worst possible
    # outcome is a wedged prep billing overnight. 4h is ~4x the expected runtime,
    # so it only ever fires on a genuinely stuck job.
    prep_max_lifetime_seconds: int = field(
        default_factory=lambda: _env_int("PREP_MAX_LIFETIME_SECONDS", 4 * 3600)
    )
    # Roughly how long the job takes, for the billable notice's cost estimate.
    prep_expected_minutes: int = field(
        default_factory=lambda: _env_int("PREP_EXPECTED_MINUTES", 70)
    )

    # --- inference fleet (ROADMAP Part 1) ------------------------------------
    # CPU instances by default: the 10M-param model serves fine on CPU, and
    # C/T-family spot draws on the "standard" spot quota, not the G quota.
    fleet_worker_count: int = field(default_factory=lambda: _env_int("FLEET_WORKERS", 4))
    fleet_worker_instance_type: str = field(
        default_factory=lambda: _env("FLEET_WORKER_INSTANCE_TYPE", "c7i.large")
    )
    fleet_router_instance_type: str = field(
        default_factory=lambda: _env("FLEET_ROUTER_INSTANCE_TYPE", "t3.small")
    )
    fleet_market: str = field(default_factory=lambda: _env("FLEET_MARKET", "spot"))
    fleet_router_port: int = field(default_factory=lambda: _env_int("FLEET_ROUTER_PORT", 8000))
    fleet_worker_port: int = field(default_factory=lambda: _env_int("FLEET_WORKER_PORT", 8001))
    # Who may reach the router's public port. Default is open (toy model, short
    # experiments); set FLEET_INGRESS_CIDR=<your-ip>/32 to tighten.
    fleet_ingress_cidr: str = field(default_factory=lambda: _env("FLEET_INGRESS_CIDR", "0.0.0.0/0"))

    # --- spotwatch (unattended spot-availability collector) -------------------
    # S3 prefix the Lambda writes JSONL shards under; also the only prefix its
    # IAM role can touch, so a bug in the collector can't reach checkpoints.
    spotwatch_prefix: str = field(default_factory=lambda: _env("SPOTWATCH_PREFIX", "spotwatch"))
    # Tick cadence. 10 minutes = 144 samples/day, comfortably inside Lambda's
    # free tier and fine-grained enough to see an hour-of-day pattern.
    spotwatch_interval_minutes: int = field(
        default_factory=lambda: _env_int("SPOTWATCH_INTERVAL_MINUTES", 10)
    )
    # UTC hour whose first tick also does the daily-only work (pool enumeration
    # + Spot Advisor fetch). 3 = quiet hour, away from business-hours throttling.
    spotwatch_daily_hour: int = field(default_factory=lambda: _env_int("SPOTWATCH_DAILY_HOUR", 3))
    # Truth probe: a real 1-instance spot launch, immediately returned. This is
    # the only part that spends money or competes for capacity — hence the hard
    # rate limit and the non-interference skip (see lambda_spotwatch.should_probe).
    spotwatch_probe_enabled: bool = field(
        default_factory=lambda: _env("SPOTWATCH_PROBE_ENABLED", "1") not in ("0", "false", "False")
    )
    spotwatch_probe_type: str = field(
        default_factory=lambda: _env("SPOTWATCH_PROBE_TYPE", "g5.xlarge")
    )
    spotwatch_probe_min_hours: float = field(
        default_factory=lambda: _env_float("SPOTWATCH_PROBE_MIN_HOURS", 6.0)
    )

    # --- vCPU quota gate ------------------------------------------------------
    # The account's "Running On-Demand G and VT instances" vCPU quota. Launches
    # wait until running+pending G/VT usage leaves headroom under this before
    # calling RunInstances (no Service Quotas API — update this if AWS raises
    # your quota).
    vcpu_quota: int = field(default_factory=lambda: _env_int("VCPU_QUOTA", 8))
    # vCPUs of `instance_type`. 0 (default) = look up the builtin table; set
    # explicitly for instance types the table doesn't know.
    instance_vcpus: int = field(default_factory=lambda: _env_int("INSTANCE_VCPUS", 0))

    # --- polling -------------------------------------------------------------
    metrics_poll_seconds: int = 15
    # How long the orchestrator waits for metrics.json before declaring the run
    # dead. This is a FLOOR, not the whole story — see metrics_deadline_for(),
    # which every supervised run must use instead of reading this directly. A
    # fixed 1800s silently capped a 1h run at 30 minutes: the fleet was healthy
    # (epoch 1, world 8, step 400, loss 4.86, zero crashes) and got terminated
    # anyway, because the watchdog's patience was shorter than the work it was
    # watching. At 36h a fixed default would kill the run before the first eval.
    metrics_timeout_seconds: int = field(default_factory=lambda: _env_int("METRICS_TIMEOUT", 1800))
    # Slack added on top of the training budget: instance launch, clone/pip, the
    # dataset pull (~2 min for a 17 GB train.bin), plus the final eval + sample +
    # checkpoint tail after the budget expires. Generous on purpose — this
    # deadline exists to catch a WEDGED run, and being late to notice one costs
    # far less than killing a healthy fleet mid-run.
    metrics_overhead_seconds: int = field(
        default_factory=lambda: _env_int("METRICS_OVERHEAD", 1200)
    )

    def metrics_deadline_for(self, budget_seconds: float | None) -> int:
        """Watchdog deadline for a run with this training budget.

        The deadline has to outlast the work: budget + boot + dataset + the
        post-budget eval/checkpoint tail. An explicit METRICS_TIMEOUT still wins
        when it is larger, so operators can extend but never accidentally
        shorten a run below its own budget.
        """
        if not budget_seconds or budget_seconds <= 0:
            return self.metrics_timeout_seconds
        return max(
            self.metrics_timeout_seconds, int(budget_seconds) + self.metrics_overhead_seconds
        )

    # How often the orchestrator pulls the box's boot log from S3 to print new
    # lines. Smaller than the metrics poll — this drives the live view latency.
    log_stream_seconds: int = field(default_factory=lambda: _env_int("LOG_STREAM_SECONDS", 3))

    # --- visualization (optional, Weights & Biases) -------------------------
    # Logging happens on the ORCHESTRATOR only; spot boxes never see the key.
    wandb_project: str = field(default_factory=lambda: _env("WANDB_PROJECT", "spot-train"))
    wandb_entity: str = field(default_factory=lambda: _env("WANDB_ENTITY", ""))
    # Optional W&B group for a comparison suite (e.g. shakespeare-convergence);
    # empty keeps the historical group-by-market behavior.
    wandb_group: str = field(default_factory=lambda: _env("WANDB_GROUP", ""))

    # -- derived S3 locations ------------------------------------------------ #
    def data_uri(self) -> str:
        return f"s3://{self.bucket}/{self.data_prefix}/{self.dataset}/"

    def run_checkpoint_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/checkpoints/"

    def run_metrics_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/metrics.json"

    def run_metrics_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/metrics.json"

    # End-of-run consolidated text samples (trainer writes it just before
    # metrics.json, so it's always present when the done-signal appears).
    def run_samples_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/samples.json"

    def run_samples_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/samples.json"

    # Mid-training inference snapshots (samples/step-<12-digit>.json), written
    # immediately at each SAMPLE_INTERVAL_STEPS gate so they survive preemption.
    def run_samples_prefix_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/samples/"

    def run_samples_prefix(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/samples/"

    # The box's boot/training log, synced here every few seconds so the orchestrator
    # can stream it back without SSH. Preemption uses a per-segment key (seg-N.log)
    # so a fresh instance doesn't overwrite the previous segment's log; multi-node
    # adds a per-node suffix so the boxes don't clobber each other, and replacement
    # launches an attempt suffix (-rK) so they don't clobber the dead node's log.
    def run_logs_key(
        self,
        run_id: str,
        segment: int | None = None,
        node: int | None = None,
        attempt: int = 0,
    ) -> str:
        name = "boot" if segment is None else f"seg-{segment}"
        if node is not None:
            name += f"-node{node}"
        if attempt:
            name += f"-r{attempt}"
        return f"{self.run_prefix}/{run_id}/logs/{name}.log"

    def run_logs_uri(self, run_id: str, segment: int | None = None) -> str:
        return f"s3://{self.bucket}/{self.run_logs_key(run_id, segment)}"

    # Epoch protocol (see supervisor.py / sidecar.py). The orchestrator is the
    # ONLY writer of epoch.json — the membership document every box's sidecar
    # polls: {epoch, members:[{node,ip,rank}], node_count, master_addr,
    # master_port}. Each box registers node<i>.json {ip, instance_id} once at
    # boot; that registration is both the ready-marker and the join request
    # (admission = the orchestrator including the node in a published epoch).
    def run_epoch_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/epoch.json"

    def run_epoch_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_epoch_key(run_id)}"

    # Observability doc the supervisor rewrites every tick: per-(node, attempt)
    # liveness + log keys, so ANY process (the `logs` viewer) can discover which
    # log belongs to whom and whether that box is alive — without the driver's
    # in-memory state. Same single writer as epoch.json; sidecars never read it.
    def run_status_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/status.json"

    def run_schedule_key(self, run_id: str) -> str:
        """Durable chaos-schedule progress: which entries have fired, and the
        WALL clock at which training started.

        Both were process memory, so any supervisor restart replayed the whole
        PREEMPT_SCHEDULE from zero -- at hour 8 of a 24h run that re-fires the
        mass loss of 6 of 8, with no live knob to stop it because the schedule is
        baked into user-data.
        """
        return f"{self.run_prefix}/{run_id}/schedule.json"

    def run_status_prefix(self, run_id: str) -> str:
        """Per-tick status objects — the DURABLE history status.json cannot be.

        status.json is overwritten in place, so fleet history existed only while
        a laptop was awake to poll it. These are written by the supervisor, so an
        offline laptop loses nothing: `aws s3 sync` on this prefix rebuilds the
        whole world-size / occupancy timeline after the fact.
        """
        return f"{self.run_prefix}/{run_id}/status/"

    def run_status_tick_key(self, run_id: str, when: float) -> str:
        # Millisecond, zero-padded: lexicographic order is chronological order,
        # which is what makes `s3 sync` + a sorted read reconstruct the timeline.
        return f"{self.run_status_prefix(run_id)}{int(when * 1000):015d}.json"

    def run_status_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_status_key(run_id)}"

    # The supervisor's OWN decision log (published epoch / terminated / relaunch),
    # uploaded next to the box logs so the viewer can show the control plane's
    # narrative as a tab too.
    def run_orch_log_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/logs/orchestrator.log"

    def run_logs_prefix(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/logs/"

    def run_nodes_prefix(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/nodes/"

    def run_node_key(self, run_id: str, node: int) -> str:
        return f"{self.run_prefix}/{run_id}/nodes/node{node}.json"

    def run_node_uri(self, run_id: str, node: int) -> str:
        return f"s3://{self.bucket}/{self.run_node_key(run_id, node)}"

    def run_uri(self, run_id: str) -> str:
        """s3://bucket/runs/<run_id> — the base the box sidecar is pointed at."""
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}"

    # Inference-fleet keys: heartbeat docs the router polls, per-box boot logs,
    # and a state doc recording which instances belong to the fleet.
    def fleet_workers_uri(self, fleet_id: str) -> str:
        return f"s3://{self.bucket}/fleet/{fleet_id}/workers/"

    def fleet_logs_key(self, fleet_id: str, name: str) -> str:
        return f"fleet/{fleet_id}/logs/{name}.log"

    def fleet_state_key(self, fleet_id: str) -> str:
        return f"fleet/{fleet_id}/fleet.json"

    # Remote-orchestrator control keys. Everything the laptop needs to find,
    # watch, and cost a detached control plane lives under this one prefix:
    #   orch.json      — what it was asked to run (experiment/env) + its run_id,
    #                    written by the on-box agent; survives a systemd restart,
    #                    which is how a restarted agent RESUMES the same run.
    #   progress.json  — boot phase markers written by user-data, so `orch up`
    #                    can show real provisioning progress instead of a spinner.
    #   heartbeat.json — the agent's liveness + live step/loss/cost.
    #   orchestrator.log / boot.log — the streamed process and user-data logs.
    def orch_prefix(self, orch_id: str) -> str:
        return f"orchestrators/{orch_id}/"

    def orch_state_key(self, orch_id: str) -> str:
        return f"orchestrators/{orch_id}/orch.json"

    def orch_progress_key(self, orch_id: str) -> str:
        return f"orchestrators/{orch_id}/progress.json"

    def orch_heartbeat_key(self, orch_id: str) -> str:
        return f"orchestrators/{orch_id}/heartbeat.json"

    def orch_log_key(self, orch_id: str) -> str:
        return f"orchestrators/{orch_id}/orchestrator.log"

    def orch_boot_log_key(self, orch_id: str) -> str:
        return f"orchestrators/{orch_id}/boot.log"

    def orch_hourly_usd(self) -> float | None:
        """$/hr for the control-plane box's cost-ledger row (None = unknown)."""
        return ON_DEMAND_HOURLY_USD.get(self.orch_instance_type)

    def orch_relay_env(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        """The env the remote control plane boots with: this shell's values for
        the allowlisted knobs (so your .env recipe carries over verbatim), then
        the explicit ``--env K=V`` overrides on top. Credential-shaped names are
        dropped from BOTH sources — user-data is world-readable on the box."""
        env = {k: os.environ[k] for k in _ORCH_RELAY_ENV if os.environ.get(k)}
        env.update(self.trainer_passthrough())  # the recipe knobs (MAX_STEPS, LR, …)
        env.update(overrides or {})  # explicit --env wins over the inherited shell
        return {k: v for k, v in env.items() if not any(s in k.upper() for s in _SECRETISH)}

    def orch_secretish(self, env: dict[str, str]) -> list[str]:
        """Names in ``env`` that look like credentials (dropped by orch_relay_env
        — reported so the operator sees WHY their value didn't make it across)."""
        return sorted(k for k in env if any(s in k.upper() for s in _SECRETISH))

    # AMI-bake control keys: the bake box writes status.json (ok/rc/commit) when
    # provisioning finishes and streams its boot log next to it.
    def bake_status_key(self, bake_id: str) -> str:
        return f"bake/{bake_id}/status.json"

    def bake_log_key(self, bake_id: str) -> str:
        return f"bake/{bake_id}/bake.log"

    # Remote dataset prep (`stage-data --remote`): the box streams its whole log
    # here so the laptop can watch an hour-long job, and writes status.json last
    # (the done signal — same shape as the bake marker).
    def prep_log_key(self, prep_id: str) -> str:
        return f"prep/{prep_id}/prep.log"

    def prep_status_key(self, prep_id: str) -> str:
        return f"prep/{prep_id}/status.json"

    def prep_log_uri(self, prep_id: str) -> str:
        return f"s3://{self.bucket}/{self.prep_log_key(prep_id)}"

    def prep_hourly_usd(self) -> float | None:
        """$/hr for the prep box (None = unknown type, so no cost estimate)."""
        return ON_DEMAND_HOURLY_USD.get(self.prep_instance_type)

    def on_demand_hourly_usd(self) -> float | None:
        """$/hr for on-demand ledger rows: HOURLY_USD override, else the table.
        None => unknown (the ledger row is kept but flagged, cost sums skip it)."""
        if self.hourly_usd:
            return self.hourly_usd
        return ON_DEMAND_HOURLY_USD.get(self.instance_type)

    # The tool-agnostic run profile (timeline + loss + merged metrics) the
    # orchestrator writes at end of run. W&B is just a mirror of this.
    def run_profile_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/profile.json"

    # The cost graph (cumulative $ + loss-per-dollar) rendered at finalize.
    def run_cost_png_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/cost.png"

    def run_profile_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/profile.json"

    def wandb_enabled(self) -> bool:
        """W&B mirror is on iff an API key is present (loaded from .env) and not
        explicitly disabled. Absent key => S3 profile.json only, no third party."""
        if os.environ.get("WANDB_DISABLED", "") in ("1", "true", "True"):
            return False
        return bool(os.environ.get("WANDB_API_KEY"))

    def instance_vcpu_count(self) -> int:
        """vCPUs one `instance_type` box consumes against the G/VT quota."""
        if self.instance_vcpus > 0:
            return self.instance_vcpus
        try:
            return _INSTANCE_VCPUS[self.instance_type]
        except KeyError:
            raise SystemExit(
                f"Unknown vCPU count for instance type {self.instance_type!r} — "
                "set INSTANCE_VCPUS=<n> in your .env so the quota gate can count it."
            ) from None

    def trainer_passthrough(self) -> dict[str, str]:
        """Recipe/cadence env vars relayed to the box verbatim — only the ones
        actually set here, so an unset knob keeps the trainer's own default."""
        return {k: os.environ[k] for k in _TRAINER_PASSTHROUGH if os.environ.get(k)}

    def preempt_victim_schedule(self) -> list[int]:
        """Node index to kill per preemption round. Empty PREEMPT_VICTIMS keeps
        the proven default (always the last node); otherwise one index per round,
        each in [0, node_count) — 0 (the master) is allowed."""
        raw = self.preempt_victims.strip()
        if not raw:
            return [self.node_count - 1] * self.preempt_count
        try:
            victims = [int(v) for v in raw.split(",")]
        except ValueError:
            raise SystemExit(
                f"PREEMPT_VICTIMS={raw!r} — must be comma-separated node indices"
            ) from None
        if len(victims) != self.preempt_count:
            raise SystemExit(
                f"PREEMPT_VICTIMS has {len(victims)} entries but PREEMPT_COUNT is "
                f"{self.preempt_count} — one victim per kill round"
            )
        bad = [v for v in victims if not 0 <= v < self.node_count]
        if bad:
            raise SystemExit(
                f"PREEMPT_VICTIMS contains {bad} — node indices must be in [0, {self.node_count})"
            )
        return victims

    def instance_lifetime_for(self, max_seconds: int) -> int:
        """Seconds after boot at which a training box self-terminates.

        Both dead-man switches shipped defaulting to 0 (OFF), which is the wrong
        default for a 24h run: if the control plane dies and a node then fails,
        survivors' torchrun crash-loops, sidecars exhaust MAX_EPOCH_CRASHES and
        exit "leaving the box up for the watchdog" -- a watchdog that no longer
        exists. The fleet then idles at full GPU rate until a human notices.

        An explicit MAX_INSTANCE_LIFETIME_SECONDS still wins; otherwise derive
        from the run's own budget plus slack.

        KNOWN LIMITATION, do not paper over it: the timer starts at THIS box's
        boot. A replacement launched at hour 20 of a 24h run gets a full-length
        timer from its own boot, so this is a backstop for the
        fleet-abandoned-early case, not a tight bound late in a run. The proper
        fix is a renewable lease in status.json, deliberately out of scope.
        """
        if self.max_instance_lifetime_seconds > 0:
            return self.max_instance_lifetime_seconds
        if max_seconds <= 0:
            return 0
        return max_seconds + self.instance_lifetime_slack_seconds

    def preempt_schedule(self) -> list[tuple[float, int]]:
        """Explicit chaos schedule: [(seconds_after_train_start, victim), ...].

        PREEMPT_VICTIMS gives ONE victim per round at evenly spaced times, so it
        cannot express a simultaneous multi-node loss -- which is why
        scripts/e4_rolling_pairs.py exists as a bespoke driver. The 24h run needs
        a simultaneous loss of 6 of 8 (E5 survived 7 of 8), so that capability
        belongs in the main driver, not in a one-off script.

        Format -- semicolon-separated events, each ``<seconds>:<victims>``:

            PREEMPT_SCHEDULE="480:3;960:L;1440:1,4;2400:0,1,2,3,4,5"

        Victims are node indices (stable slots -- a replacement re-takes its
        index), or ``L`` for whichever node is master WHEN THE KILL FIRES. L
        matters because the master moves: elect_master is sticky to a survivor,
        so after any kill the leader may no longer be node 0, and a hardcoded
        index would quietly stop testing re-election.

        Several victims at the SAME timestamp fire in one supervisor tick -- the
        supervisor already collects due kills into a set, so simultaneity needs
        nothing beyond expressing it here.

        Empty => fall back to the PREEMPT_VICTIMS/PREEMPT_COUNT behaviour.
        """
        raw = (self.preempt_schedule_spec or "").strip()
        if not raw:
            return []
        out: list[tuple[float, int]] = []
        for event in raw.split(";"):
            event = event.strip()
            if not event:
                continue
            if ":" not in event:
                raise SystemExit(
                    f"PREEMPT_SCHEDULE event {event!r} — expected '<seconds>:<victims>', "
                    f"e.g. '1440:1,4' or '960:L'"
                )
            when, victims = event.split(":", 1)
            try:
                secs = float(when)
            except ValueError:
                raise SystemExit(
                    f"PREEMPT_SCHEDULE event {event!r} — {when!r} is not a number"
                ) from None
            for v in victims.split(","):
                v = v.strip()
                if v.upper() == "L":
                    out.append((secs, LEADER_VICTIM))
                    continue
                try:
                    idx = int(v)
                except ValueError:
                    raise SystemExit(
                        f"PREEMPT_SCHEDULE event {event!r} — victim {v!r} must be a "
                        f"node index or 'L'"
                    ) from None
                if not 0 <= idx < self.node_count:
                    raise SystemExit(
                        f"PREEMPT_SCHEDULE event {event!r} — node index {idx} outside "
                        f"[0, {self.node_count})"
                    )
                out.append((secs, idx))
        # A kill group that removes EVERY node is a total loss, not a chaos
        # event: there is no survivor to keep training and the run becomes a
        # whole-group restart. Catch it here rather than three minutes into a
        # billed run.
        by_time: dict[float, int] = {}
        for secs, _v in out:
            by_time[secs] = by_time.get(secs, 0) + 1
        for secs, n in sorted(by_time.items()):
            if n >= self.node_count:
                raise SystemExit(
                    f"PREEMPT_SCHEDULE kills {n} of {self.node_count} nodes at t+{secs:.0f}s — "
                    f"that is a total loss with no survivors, not a preemption. Leave at least one."
                )
        return sorted(out)

    # Whole-group-restart floor, in EPOCHS with no checkpoint progress. One
    # preemption publishes two epochs (shrink, then grow), so the default 6 is
    # three full recoveries. A chaos run with many scheduled rounds needs this
    # raised or the floor fires on the mechanism working as designed — E4 sat
    # exactly at 6 with three rounds and survived only because checkpoints
    # landed between them and reset the counter.
    max_epochs_without_progress: int = field(
        default_factory=lambda: _env_int("MAX_EPOCHS_WITHOUT_PROGRESS", 6)
    )

    @classmethod
    def for_inference(cls) -> OrchestratorConfig:
        """Config for the inference platform — its own region, bucket and names.

        Training and inference share this class *in the same process*, so
        precedence tricks on the generic vars are not safe: putting
        ``AWS_REGION=us-east-2`` in a shared ``.env`` to move inference would
        silently move **training** too, into a region whose GPU quota it does
        not own.

        So inference reads ``INFERENCE_*``-prefixed vars **only**, with its own
        defaults, and ignores ``AWS_REGION`` / ``SPOT_TRAIN_BUCKET`` /
        ``IAM_ROLE`` / ``IAM_PROFILE`` / ``SECURITY_GROUP`` entirely. Those keep
        their training meaning. The two configurations cannot collide because
        they never read the same variable.

        The IAM names matter as much as the region: **IAM is global, not
        regional**, so the us-east-1/us-east-2 split does not protect it. An
        ``inference-*`` prefix is what keeps us off ``spot-train-role`` and
        friends.
        """
        cfg = cls()
        cfg.region = _env("INFERENCE_REGION", "us-east-2")
        cfg.bucket = _env("INFERENCE_BUCKET", "")
        cfg.project_tag = _env("INFERENCE_PROJECT_TAG", "inference")
        cfg.role_name = _env("INFERENCE_IAM_ROLE", "inference-role")
        cfg.instance_profile = _env("INFERENCE_IAM_PROFILE", "inference-profile")
        cfg.security_group = _env("INFERENCE_SECURITY_GROUP", "inference-sg")

        # Hard guard, not a convention. Training runs 8-node g5.xlarge fleets in
        # us-east-1 and replaces preempted nodes automatically; landing there
        # would race it for GPU capacity, and the failures look like AWS
        # capacity errors rather than contention.
        if cfg.region == "us-east-1" and _env("INFERENCE_ALLOW_US_EAST_1", "") != "yes":
            raise SystemExit(
                "Refusing to run the inference fleet in us-east-1: that region belongs "
                "to the spot-training project (see docs/prompts/inference-agent-region.md). "
                "Set INFERENCE_REGION=us-east-2, or INFERENCE_ALLOW_US_EAST_1=yes if you "
                "genuinely mean it."
            )
        return cfg

    def require_bucket(self) -> None:
        if not self.bucket:
            raise SystemExit(
                "No S3 bucket set. Put SPOT_TRAIN_BUCKET=<name> in your .env for "
                "training, or INFERENCE_BUCKET=<name> for the inference fleet "
                "(see .env.example), and run `setup` first."
            )
