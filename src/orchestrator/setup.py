"""One-time, idempotent AWS infra setup: bucket + instance profile + SG.

Run once (``spot-orchestrate setup``). Needs your local creds to have S3, IAM,
and EC2 permissions. Everything here is safe to re-run. Use ``--dry-run`` to see
exactly what it would create before granting it those permissions.
"""

from __future__ import annotations

import sys

from . import aws
from .config import OrchestratorConfig


def ensure_infra(cfg: OrchestratorConfig) -> None:
    cfg.require_bucket()
    aws.set_region(cfg.region)

    aws.ensure_bucket(cfg.bucket, cfg.region)
    aws.ensure_instance_profile(cfg.role_name, cfg.instance_profile, cfg.bucket)
    # The control plane's own role (`orch up`): EC2 lifecycle + S3 + PassRole for
    # the worker role. Created here because it needs the same one-time IAM rights
    # as the worker role — the ongoing controller never creates roles.
    aws.ensure_orchestrator_profile(
        cfg.orch_role_name, cfg.orch_instance_profile, cfg.bucket, cfg.role_name
    )
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)

    print(
        f"[setup] ready: bucket={cfg.bucket} profile={cfg.instance_profile} "
        f"orch_profile={cfg.orch_instance_profile} sg={sg_id} region={cfg.region}",
        file=sys.stderr,
    )
    print(
        "[setup] note: a freshly-created IAM instance profile can take ~10s to "
        "propagate before the first launch succeeds.",
        file=sys.stderr,
    )
