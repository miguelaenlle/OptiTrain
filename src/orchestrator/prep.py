"""Prepare a dataset INSIDE AWS — ``spot-orchestrate stage-data --remote``.

Local ``stage-data`` tokenizes on your machine and uploads the bins. That stops
working at OpenWebText scale: the job needs ~110 GB of transient disk (54 GB
HuggingFace cache + ~35 GB tokenized arrow + 17 GB of bins) and then pushes a
17 GB ``train.bin`` up a home uplink. Run the identical job on one throwaway
EC2 box instead and the S3 upload is same-region (free, minutes) and the whole
thing is ~45-70 minutes for well under a dollar.

Deliberately a THROWAWAY, not a service:

  * **one on-demand box** (``PREP_INSTANCE_TYPE``, default ``c6i.4xlarge``) with
    an explicitly sized 200 GB gp3 root at 500 MB/s — the AMI defaults (8 GB on
    AL2023, 30 GB on the DLAMI) cannot hold this job, and 125 MB/s of default
    gp3 throughput puts ~7 minutes of pure disk wait on the critical path;
  * **it runs the ordinary code paths**: ``data/<dataset>/prepare.py`` then
    ``spot-orchestrate stage-data`` — the same staging that runs on a laptop, so
    there is exactly one uploader to keep correct;
  * **it terminates itself** on success and on failure, with a systemd lifetime
    timer as an independent backstop (see ``bootstrap.build_prep_user_data``);
  * **S3 is the whole interface**: the box streams its log to
    ``prep/<id>/prep.log`` and writes ``prep/<id>/status.json`` last. The laptop
    only ever GETs them, so Ctrl-C detaches and can never hurt the job.

Credentials: none are passed. The box uses the ordinary worker instance profile
(``IAM_PROFILE``), whose policy is exactly the S3 read/write on the bucket this
job needs — see docs/iam/worker-policy.json.
"""

from __future__ import annotations

import sys
import time

from . import aws
from .config import OrchestratorConfig

# The bins a staged dataset must have. meta.pkl is char-level only (BPE corpora
# ship none), so it is never required — same rule as orchestrator.dataset.
_REQUIRED = ("train.bin", "val.bin")

# Smallest byte size a staged bin can plausibly have, per dataset. Two jobs:
# refuse to redo an hour of work when the corpus is already staged, and catch a
# TRUNCATED upload here rather than at 8-node launch time. Only recipes with a
# FIXED output are listed — these numbers are measured from the prep script's
# own token counts (OpenWebText: 9.04B train x 2 bytes ~= 18.1 GB, 4.4M val x 2
# ~= 8.8 MB), never guessed. openwebtext_300m is absent on purpose: its size
# follows OWT_TARGET_TOKENS, so any floor would be a lie.
_MIN_BYTES: dict[str, dict[str, int]] = {
    "openwebtext": {"train.bin": 16_000_000_000, "val.bin": 4_000_000},
}
# Everything else only has to be non-trivially non-empty: a bin under 1 KB is a
# failed write in any recipe, but we refuse to invent a tighter bound.
_DEFAULT_MIN_BYTES = 1024

# gp3 pricing (us-east-1) for the billable notice. Storage is per GB-month;
# IOPS above the free 3000 and throughput above the free 125 MB/s bill on top.
_GP3_GB_MONTH_USD = 0.08
_GP3_IOPS_MONTH_USD = 0.005
_GP3_MBPS_MONTH_USD = 0.040
_GP3_FREE_IOPS = 3000
_GP3_FREE_MBPS = 125
_HOURS_PER_MONTH = 730.0

# EC2 tag that marks a prep box, so a detached session can find it again.
PREP_TAG = "prep"


def _human(n: int | None) -> str:
    if n is None:
        return "absent"
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:,.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= 1024
    return f"{val:,.1f} TB"  # pragma: no cover — loop always returns


# --------------------------------------------------------------------------- #
# Verification (pure) — "did we actually stage a usable corpus?"
# --------------------------------------------------------------------------- #
def check_sizes(dataset: str, sizes: dict[str, int | None]) -> tuple[bool, list[str]]:
    """PURE verdict on the staged objects' sizes: ``(ok, lines to print)``.

    A prep that "succeeded" but uploaded a truncated bin has to be caught here —
    the alternative is discovering it when eight nodes boot, download it, and
    train on half a corpus."""
    floors = _MIN_BYTES.get(dataset, {})
    ok = True
    lines: list[str] = []
    for name in _REQUIRED:
        size = sizes.get(name)
        floor = floors.get(name, _DEFAULT_MIN_BYTES)
        if size is None:
            ok = False
            lines.append(f"  {name}: MISSING")
        elif size < floor:
            ok = False
            lines.append(f"  {name}: {_human(size)} — TRUNCATED (expected >= {_human(floor)})")
        else:
            lines.append(f"  {name}: {_human(size)}")
    return ok, lines


def staged_sizes(cfg: OrchestratorConfig) -> dict[str, int | None]:
    """Live sizes of the dataset's bins in S3 (None where absent)."""
    prefix = f"{cfg.data_prefix}/{cfg.dataset}"
    return {name: aws.object_size(cfg.bucket, f"{prefix}/{name}") for name in _REQUIRED}


def verify(cfg: OrchestratorConfig) -> bool:
    """Print what actually landed in S3 and return whether it is usable."""
    ok, lines = check_sizes(cfg.dataset, staged_sizes(cfg))
    print(f"[prep] staged at {cfg.data_uri()}", file=sys.stderr)
    for line in lines:
        print(f"[prep] {line}", file=sys.stderr)
    return ok


# --------------------------------------------------------------------------- #
# Cost notice
# --------------------------------------------------------------------------- #
def estimate_usd(cfg: OrchestratorConfig, hours: float) -> float | None:
    """Estimated dollars for a prep of ``hours``: the instance plus the volume it
    provisions (storage + the IOPS/throughput above gp3's free tier). None when
    the instance type isn't in the rate table — better no number than a wrong one."""
    rate = cfg.prep_hourly_usd()
    if rate is None:
        return None
    share = hours / _HOURS_PER_MONTH
    ebs = cfg.prep_volume_gb * _GP3_GB_MONTH_USD * share
    ebs += max(0, cfg.prep_volume_iops - _GP3_FREE_IOPS) * _GP3_IOPS_MONTH_USD * share
    ebs += max(0, cfg.prep_volume_throughput - _GP3_FREE_MBPS) * _GP3_MBPS_MONTH_USD * share
    return rate * hours + ebs


def _billable_notice(cfg: OrchestratorConfig) -> str:
    mins = cfg.prep_expected_minutes
    est = estimate_usd(cfg, mins / 60.0)
    cost = f"about ${est:.2f}" if est is not None else "cost unknown (type not in the rate table)"
    return (
        f"\n\033[1m⚠️  BILLABLE: one on-demand {cfg.prep_instance_type} + "
        f"{cfg.prep_volume_gb} GB gp3 ({cfg.prep_volume_throughput} MB/s) in {cfg.region}, "
        f"~{max(20, mins - 25)}-{mins} min, {cost}.\n"
        f"   It terminates itself when done, and unconditionally after "
        f"{cfg.prep_max_lifetime_seconds // 3600}h no matter what.\033[0m\n"
    )


# --------------------------------------------------------------------------- #
# Watching a prep (read-only: Ctrl-C here cannot touch the box)
# --------------------------------------------------------------------------- #
def _warn_branch_mismatch(cfg: OrchestratorConfig) -> None:
    """Warn when the box would clone a different branch than the one you are
    sitting on. REPO_BRANCH defaults to ``main``, and the box runs the *cloned*
    ``data/<dataset>/prepare.py`` — so a forgotten REPO_BRANCH silently spends an
    hour running someone else's prep script. Best-effort: not in a git checkout
    (or no git) simply skips the check."""
    import subprocess

    try:
        local = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — no git / not a checkout => nothing to compare
        return
    if local and local != "HEAD" and local != cfg.repo_branch:
        print(
            f"\n\033[1m⚠️  The box will clone REPO_BRANCH={cfg.repo_branch}, but you are on "
            f"{local}. Set REPO_BRANCH={local} in .env if that is the code you mean "
            f"to run.\033[0m\n",
            file=sys.stderr,
        )


def _detach_banner(cfg: OrchestratorConfig, prep_id: str, iid: str) -> str:
    return "\n".join(
        [
            "",
            f"[prep] detached — the job keeps going on {iid} ({prep_id}).",
            f"[prep]   reattach:   spot-orchestrate stage-data --remote --attach {prep_id}",
            f"[prep]   log:        {cfg.prep_log_uri(prep_id)}",
            f"[prep]   terminate:  aws ec2 terminate-instances --region {cfg.region} "
            f"--instance-ids {iid}",
            "[prep] Nothing was killed: this view only reads S3.",
        ]
    )


def find_instance(cfg: OrchestratorConfig, prep_id: str) -> str:
    """Instance id of a (still-running) prep box, or "". Tag-based discovery, so
    a reattach from another terminal needs no local state."""
    found = aws.instances_by_tag(PREP_TAG, prep_id)
    return found[0]["id"] if found else ""


def watch(cfg: OrchestratorConfig, prep_id: str, iid: str = "") -> dict | None:
    """Stream the box's prep.log until it publishes status.json (the done
    signal), printing new bytes as they land. Returns the status document, or
    None if the box vanished / the wait timed out.

    Raises KeyboardInterrupt to the caller so Ctrl-C reads as DETACH."""
    log_key = cfg.prep_log_key(prep_id)
    printed = 0
    started = time.monotonic()
    # Generous: the box's own dead-man switch fires first, so this only bounds
    # the laptop's patience, never the job.
    deadline = started + cfg.prep_max_lifetime_seconds + 600
    last_note = started
    while True:
        if aws.object_exists(cfg.bucket, log_key):
            text = aws.get_text(cfg.bucket, log_key)
            if len(text) > printed:
                sys.stdout.write(text[printed:])
                sys.stdout.flush()
                printed = len(text)
        status = aws.get_json(cfg.bucket, cfg.prep_status_key(prep_id))
        if status is not None:
            return status
        now = time.monotonic()
        if now > deadline:
            print("\n[prep] timed out waiting for the box's status document", file=sys.stderr)
            return None
        if iid and now - last_note >= 60:
            # A box that died without writing status (autokill, capacity pull)
            # would otherwise leave this loop printing nothing for hours.
            state = aws.instance_state(iid)
            if state not in ("pending", "running"):
                print(
                    f"\n[prep] instance {iid} is {state} with no status document",
                    file=sys.stderr,
                )
                return None
            if printed == 0:
                print(f"[prep] waiting for the box to start logging… ({int(now - started)}s)")
            last_note = now
        time.sleep(cfg.log_stream_seconds)


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def run_remote_prep(cfg: OrchestratorConfig, *, attach: bool = True, prep_id: str = "") -> str:
    """Launch (or reattach to) a remote dataset prep. Returns the prep id.

    ``prep_id`` set = pure reattach: no launch, no billable notice, just the
    viewer again. This is what makes Ctrl-C safe to press."""
    from . import bootstrap

    cfg.require_bucket()
    aws.set_region(cfg.region)

    if prep_id:
        iid = find_instance(cfg, prep_id)
        print(f"[prep] reattaching to {prep_id} ({iid or 'instance gone'})", file=sys.stderr)
        return _finish(cfg, prep_id, iid, attach=True)

    # Idempotence: an hour of work and ~$1 is far too much to spend re-deriving
    # bytes that are already in the bucket.
    if not aws.is_dry_run():
        ok, lines = check_sizes(cfg.dataset, staged_sizes(cfg))
        if ok:
            print(f"[prep] {cfg.data_uri()} is already staged:", file=sys.stderr)
            for line in lines:
                print(f"[prep] {line}", file=sys.stderr)
            raise SystemExit(
                "[prep] refusing to redo the prep. Delete those objects (or set "
                "DATASET=<other>) if you really want to rebuild them."
            )

    prep_id = time.strftime("prep-%Y%m%d-%H%M%S")
    print(
        f"[prep] {prep_id}: preparing {cfg.dataset} on 1x{cfg.prep_instance_type} "
        f"(on-demand) in {cfg.region}, branch {cfg.repo_branch}",
        file=sys.stderr,
    )
    _warn_branch_mismatch(cfg)
    print(_billable_notice(cfg), file=sys.stderr)

    ud = bootstrap.build_prep_user_data(cfg, prep_id=prep_id)
    # AL2023, not the DLAMI: this job needs python + pip, not CUDA. Reuses the
    # control plane's AMI resolution so there is one lightweight-image pattern.
    ami = aws.resolve_ami(cfg.orch_ami_id, cfg.orch_ami_name_filter)
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)
    iid = aws.launch(
        ami_id=ami,
        instance_type=cfg.prep_instance_type,
        # The ordinary worker profile: its policy is exactly S3 read/write on the
        # bucket (docs/iam/worker-policy.json), which is all this job needs. No
        # keys are copied anywhere.
        profile_name=cfg.instance_profile,
        security_group_id=sg_id,
        user_data=ud,
        market="on-demand",  # NEVER spot: a reclaim mid-tokenize wastes the hour
        run_id=prep_id,
        key_name=cfg.key_name,
        extra_tags={PREP_TAG: prep_id, "prep_dataset": cfg.dataset},
        root_volume_gb=cfg.prep_volume_gb,
        root_volume_iops=cfg.prep_volume_iops,
        root_volume_throughput=cfg.prep_volume_throughput,
    )
    print(f"[prep] instance {iid}", file=sys.stderr)
    print(f"[prep]   log:        {cfg.prep_log_uri(prep_id)}", file=sys.stderr)
    print(
        f"[prep]   reattach:   spot-orchestrate stage-data --remote --attach {prep_id}",
        file=sys.stderr,
    )
    print(
        f"[prep]   terminate:  aws ec2 terminate-instances --region {cfg.region} "
        f"--instance-ids {iid}",
        file=sys.stderr,
    )
    if aws.is_dry_run():
        print("[prep] dry-run: skipping the log stream", file=sys.stderr)
        return prep_id
    return _finish(cfg, prep_id, iid, attach=attach)


def _finish(cfg: OrchestratorConfig, prep_id: str, iid: str, *, attach: bool) -> str:
    """Watch (if asked), then verify what landed in S3."""
    if not attach:
        print(
            f"[prep] not attaching — watch with `spot-orchestrate stage-data --remote "
            f"--attach {prep_id}`",
            file=sys.stderr,
        )
        return prep_id
    print(f"[prep] streaming {cfg.prep_log_uri(prep_id)} — Ctrl-C DETACHES", file=sys.stderr)
    try:
        status = watch(cfg, prep_id, iid)
    except KeyboardInterrupt:
        print(_detach_banner(cfg, prep_id, iid or "the prep box"), file=sys.stderr)
        return prep_id
    if status is None:
        raise SystemExit(
            f"[prep] {prep_id} did not report a result — see {cfg.prep_log_uri(prep_id)}"
        )
    if not status.get("ok"):
        raise SystemExit(
            f"[prep] {prep_id} FAILED (rc={status.get('rc')}) — the box terminated itself; "
            f"log: {cfg.prep_log_uri(prep_id)}"
        )
    # The box says it uploaded; check what is actually there before anyone
    # launches an 8-node run against it.
    if not verify(cfg):
        raise SystemExit(f"[prep] {prep_id} reported success but the staged bins are not usable")
    print(f"\033[1m[prep] {prep_id} DONE — {cfg.data_uri()} is ready\033[0m", file=sys.stderr)
    return prep_id
