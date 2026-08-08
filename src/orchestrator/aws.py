"""The ONLY module that talks to AWS.

Every credentialed call lives here so the surface is auditable in one place.
Design rules:

  - Credentials are never referenced in code — boto3 resolves them from the
    ambient environment/profile at call time.
  - Every *mutating* call logs a plain-English line before it fires.
  - ``set_dry_run(True)`` makes every function log what it *would* do and call
    nothing — so ``--dry-run`` provably touches no AWS API and needs no creds.

The orchestrator's other modules (setup, experiments, dataset) call these
functions; they never import boto3 themselves.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_DRY_RUN = False
_clients: dict[str, Any] = {}

# All training infrastructure lives in ONE region, and this is it.
#
# We share an AWS account with a separate inference platform, which owns
# us-east-2. The split is what keeps the two independent: G/VT vCPU quota is
# PER-REGION, so neither project can starve the other, and an over-broad
# `describe-instances` in one region cannot even see — let alone terminate —
# the other project's boxes. See docs/region-split.md.
#
# Enforced rather than defaulted: a stray AWS_REGION in the environment would
# otherwise silently launch GPUs into the inference platform's region and eat
# their quota. Set ALLOW_REGION_OVERRIDE=1 to deliberately step outside.
TRAINING_REGION = "us-east-1"

_region = TRAINING_REGION


def set_dry_run(flag: bool) -> None:
    global _DRY_RUN
    _DRY_RUN = flag


def is_dry_run() -> bool:
    return _DRY_RUN


def set_region(region: str) -> None:
    """Point every AWS client at ``region``, refusing to leave TRAINING_REGION.

    Single choke point: every entry point (orch, setup, spotwatch, prep,
    experiments) calls this before touching AWS, so the guard covers all of them.
    """
    if region != TRAINING_REGION and not os.environ.get("ALLOW_REGION_OVERRIDE"):
        raise SystemExit(
            f"Refusing to use region {region!r}: all training infrastructure is "
            f"pinned to {TRAINING_REGION} (docs/region-split.md). us-east-2 belongs "
            f"to the inference platform sharing this AWS account — launching there "
            f"would consume its GPU quota.\n"
            f"Fix AWS_REGION in your .env, or set ALLOW_REGION_OVERRIDE=1 if you "
            f"really mean it."
        )
    global _region
    _region = region
    _clients.clear()


def _client(service: str):
    import boto3  # lazy: only imported when a real call is made

    if service not in _clients:
        _clients[service] = boto3.client(service, region_name=_region)
    return _clients[service]


def _log(msg: str) -> None:
    prefix = "[aws:dry-run] would" if _DRY_RUN else "[aws]"
    print(f"{prefix} {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Read-only lookups
# --------------------------------------------------------------------------- #
def resolve_ami(ami_id: str, name_filter: str) -> str:
    """Return an explicit AMI id, or the newest Amazon-owned image whose name
    matches ``name_filter`` (via DescribeImages — no SSM public parameters)."""
    if ami_id:
        return ami_id
    if _DRY_RUN:
        _log(f"resolve AMI via DescribeImages name~={name_filter!r}")
        return "ami-DRYRUN"
    r = _client("ec2").describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": [name_filter]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = sorted(r.get("Images", []), key=lambda i: i["CreationDate"])
    if not images:
        raise SystemExit(
            f"No AMI matched {name_filter!r} in this region. Set AMI_ID explicitly "
            f"(see README) or adjust AMI_NAME_FILTER."
        )
    chosen = images[-1]
    _log(f"resolved AMI {chosen['ImageId']} ({chosen['Name']})")
    return chosen["ImageId"]


def image_root_device(ami_id: str) -> str:
    """The AMI's root device name ("/dev/xvda" on Amazon Linux 2023, "/dev/sda1"
    on the Ubuntu DLAMI).

    Asked, never guessed: a BlockDeviceMapping only RESIZES the root volume when
    its DeviceName matches the AMI's exactly. A wrong name is not an error — EC2
    silently attaches a SECOND, unmounted volume and the box boots with the tiny
    default root anyway, which for a 110 GB dataset prep means a disk-full crash
    40 minutes in instead of a clear failure at launch."""
    if _DRY_RUN:
        _log(f"describe-images {ami_id} (root device name)")
        return "/dev/xvda"
    try:
        r = _client("ec2").describe_images(ImageIds=[ami_id])
        return r["Images"][0].get("RootDeviceName") or "/dev/xvda"
    except Exception:  # noqa: BLE001 — unreadable image => fall back to the common name
        return "/dev/xvda"


def root_volume_mapping(
    device_name: str, size_gb: int, *, iops: int = 0, throughput: int = 0
) -> list[dict[str, Any]]:
    """The ``BlockDeviceMappings`` entry that resizes/retunes an instance's root
    volume. PURE (no API call) so the shape is unit-testable.

    gp3 because it is the only volume type where IOPS and throughput are dialed
    independently of size; DeleteOnTermination so a self-terminating box can
    never leave a 200 GB volume billing behind it."""
    ebs: dict[str, Any] = {
        "VolumeSize": int(size_gb),
        "VolumeType": "gp3",
        "DeleteOnTermination": True,
    }
    if iops > 0:
        ebs["Iops"] = int(iops)
    if throughput > 0:
        ebs["Throughput"] = int(throughput)
    return [{"DeviceName": device_name, "Ebs": ebs}]


def object_exists(bucket: str, key: str) -> bool:
    if _DRY_RUN:
        _log(f"head s3://{bucket}/{key}")
        return False
    try:
        _client("s3").head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def object_size(bucket: str, key: str) -> int | None:
    """Size in bytes of an S3 object, or None if it is absent. Used by
    ``stage-data`` to tell "already uploaded" from "uploaded something else" —
    a 17 GB re-upload is an hour, so we want more than existence before skipping."""
    if _DRY_RUN:
        _log(f"head s3://{bucket}/{key} (size)")
        return None
    try:
        return int(_client("s3").head_object(Bucket=bucket, Key=key)["ContentLength"])
    except Exception:  # noqa: BLE001 — NoSuchKey / 403 => treat as absent
        return None


def any_object_under(bucket: str, prefix: str) -> bool:
    """True if at least one object exists under ``prefix`` (e.g. first checkpoint)."""
    if _DRY_RUN:
        _log(f"list s3://{bucket}/{prefix} (MaxKeys=1)")
        return False
    r = _client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return r.get("KeyCount", 0) > 0


def get_text(bucket: str, key: str) -> str:
    if _DRY_RUN:
        _log(f"get s3://{bucket}/{key}")
        return "{}"
    return _client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()


def get_texts(bucket: str, keys: list[str]) -> dict[str, str]:
    """Fetch many small objects at once. The spotwatch report reads one shard per
    10-minute tick (432 for a 72h window) — serially that is minutes of pure
    round-trip latency, so they go out on a small thread pool. Unreadable keys
    are skipped rather than failing a report over one bad object."""
    if _DRY_RUN:
        _log(f"get {len(keys)} objects from s3://{bucket}/")
        return {}
    from concurrent.futures import ThreadPoolExecutor

    def _one(key: str) -> tuple[str, str]:
        try:
            return key, _client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        except Exception:  # noqa: BLE001 — one shard missing must not sink the report
            return key, ""

    with ThreadPoolExecutor(max_workers=16) as pool:
        return {k: v for k, v in pool.map(_one, keys) if v}


def get_json(bucket: str, key: str) -> dict | None:
    """Small control document as a dict, or None if it's absent/unreadable. One
    API call instead of head+get, and never raises — the remote-orchestrator
    views poll documents that legitimately don't exist yet."""
    if _DRY_RUN:
        _log(f"get s3://{bucket}/{key} (json)")
        return None
    import json

    try:
        return json.loads(_client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode())
    except Exception:  # noqa: BLE001 — absent/partial/malformed => "no document yet"
        return None


def max_checkpoint_step(bucket: str, prefix: str) -> int:
    """Highest checkpoint step under ``prefix`` (ckpt-<step>.pt), or -1 if none.
    Used to detect training-start (step advances past the resume point) and to
    confirm the graceful SIGTERM checkpoint landed before we terminate the box."""
    if _DRY_RUN:
        _log(f"list checkpoints s3://{bucket}/{prefix}")
        return -1
    import contextlib

    best = -1
    paginator = _client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            base = obj["Key"].rsplit("/", 1)[-1]
            if base.startswith("ckpt-") and base.endswith(".pt"):
                with contextlib.suppress(ValueError):
                    best = max(best, int(base[len("ckpt-") : -len(".pt")]))
    return best


def object_last_modified(bucket: str, key: str) -> float | None:
    """POSIX timestamp of ``key``'s last write, or None if absent. The boxes
    re-upload their boot log every few seconds, so the age of the log key is a
    free liveness heartbeat — no box-side heartbeat machinery needed."""
    if _DRY_RUN:
        _log(f"head s3://{bucket}/{key} (last-modified)")
        return None
    try:
        r = _client("s3").head_object(Bucket=bucket, Key=key)
        return r["LastModified"].timestamp()
    except Exception:  # noqa: BLE001 — absent or transient => no heartbeat yet
        return None


def list_keys(bucket: str, prefix: str) -> list[str]:
    """All object keys under ``prefix``, sorted. Used to collect the trainer's
    per-step sample snapshots (runs/<run_id>/samples/step-*.json)."""
    if _DRY_RUN:
        _log(f"list s3://{bucket}/{prefix}")
        return []
    keys: list[str] = []
    paginator = _client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return sorted(keys)


def default_vpc_id(region: str) -> str:
    """The region's default VPC id, or "" if it has none. The spotwatch probe
    launches with no SubnetId, which only works inside a default VPC — checking
    at deploy time keeps a missing VPC from later masquerading as "no capacity"."""
    if _DRY_RUN:
        _log(f"describe-vpcs isDefault=true in {region}")
        return "vpc-DRYRUN"
    import boto3  # noqa: PLC0415 — per-region client, outside the cached default

    r = boto3.client("ec2", region_name=region).describe_vpcs(
        Filters=[{"Name": "isDefault", "Values": ["true"]}]
    )
    vpcs = r.get("Vpcs", [])
    return vpcs[0]["VpcId"] if vpcs else ""


def ssm_online(instance_id: str) -> bool:
    """True if the SSM agent on the instance is registered and online (so we can
    send it a command). Boxes get AmazonSSMManagedInstanceCore via the instance
    profile and outbound HTTPS via the public IP."""
    if _DRY_RUN:
        _log(f"ssm describe-instance-information {instance_id}")
        return True
    r = _client("ssm").describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
    )
    info = r.get("InstanceInformationList", [])
    return bool(info) and info[0].get("PingStatus") == "Online"


def ssm_send(instance_id: str, commands: list[str]) -> str:
    """Run shell commands on the instance via SSM RunCommand; returns command id.
    This is how the orchestrator delivers the 'Spot' shutdown signal (SIGTERM to
    the trainer) without SSH."""
    _log(f"ssm send-command {instance_id}: {' && '.join(commands)}")
    if _DRY_RUN:
        return "cmd-DRYRUN"
    r = _client("ssm").send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
    )
    return r["Command"]["CommandId"]


def instance_state(instance_id: str) -> str:
    if _DRY_RUN:
        _log(f"describe {instance_id}")
        return "running"
    r = _client("ec2").describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0]["State"]["Name"]


def public_ip(instance_id: str) -> str:
    """Public IPv4 of the instance, or "" if it has none. (SSH-verification mode.)"""
    if _DRY_RUN:
        _log(f"describe {instance_id} (public ip)")
        return "203.0.113.10"
    r = _client("ec2").describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")


def private_ip(instance_id: str) -> str:
    """Private IPv4 of the instance (intra-SG traffic, e.g. router -> workers)."""
    if _DRY_RUN:
        _log(f"describe {instance_id} (private ip)")
        return "10.0.0.10"
    r = _client("ec2").describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0].get("PrivateIpAddress", "")


def instances_by_tag(key: str, value: str) -> list[dict[str, str]]:
    """Non-terminated instances carrying tag ``key=value`` — fleet discovery.
    Returns [{id, state, type, public_ip, private_ip, tags:{...}}, ...]."""
    if _DRY_RUN:
        _log(f"describe-instances tag:{key}={value}")
        return []
    r = _client("ec2").describe_instances(
        Filters=[
            {"Name": f"tag:{key}", "Values": [value]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
        ]
    )
    out = []
    for res in r["Reservations"]:
        for inst in res["Instances"]:
            out.append(
                {
                    "id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "type": inst["InstanceType"],
                    "public_ip": inst.get("PublicIpAddress", ""),
                    "private_ip": inst.get("PrivateIpAddress", ""),
                    "tags": {t["Key"]: t["Value"] for t in inst.get("Tags", [])},
                }
            )
    return out


def instance_az(instance_id: str) -> str:
    """Availability zone the instance landed in (spot prices are per-AZ)."""
    if _DRY_RUN:
        _log(f"describe {instance_id} (az)")
        return f"{_region}a"
    r = _client("ec2").describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0]["Placement"]["AvailabilityZone"]


def spot_hourly_rate(instance_type: str, az: str) -> float | None:
    """Current spot $/hr for ``instance_type`` in ``az`` — the newest point in
    DescribeSpotPriceHistory, which is what a fresh spot launch starts billing
    at. Returns None if AWS returns no price point."""
    if _DRY_RUN:
        _log(f"describe-spot-price-history {instance_type} in {az}")
        return 0.0
    from datetime import datetime, timezone

    r = _client("ec2").describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        AvailabilityZone=az,
        StartTime=datetime.now(timezone.utc),
    )
    hist = r.get("SpotPriceHistory", [])
    return float(hist[0]["SpotPrice"]) if hist else None


# --------------------------------------------------------------------------- #
# Mutating: S3
# --------------------------------------------------------------------------- #
def ensure_bucket(bucket: str, region: str) -> None:
    _log(f"create S3 bucket {bucket} in {region} (idempotent)")
    if _DRY_RUN:
        return
    s3 = _client("s3")
    if object_exists_bucket(s3, bucket):
        return
    kwargs: dict[str, Any] = {"Bucket": bucket}
    if region != "us-east-1":  # us-east-1 rejects an explicit LocationConstraint
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)


ABORT_INCOMPLETE_UPLOAD_DAYS = 7


def ensure_bucket_lifecycle(bucket: str, days: int = ABORT_INCOMPLETE_UPLOAD_DAYS) -> None:
    """Expire incomplete multipart uploads server-side.

    The worker/orchestrator roles can now abort their own failed uploads, but
    that only helps a process that is still ALIVE to do it. Preemption kills the
    box mid-write by definition — that is the whole experiment — so the parts of
    a half-written 17 GB checkpoint have nobody left to clean them up. They then
    bill indefinitely and do not show up in `s3 ls`, which is how 4.86 GB
    accumulated here unnoticed over a month. S3 expiring them is the only
    cleanup that cannot be skipped by the failure it is cleaning up after.

    Idempotent: the rule is written by id, so re-running replaces it in place.
    """
    _log(f"put S3 lifecycle on {bucket}: abort incomplete multipart uploads after {days}d")
    if _DRY_RUN:
        return
    _client("s3").put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "abort-incomplete-multipart-uploads",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},  # whole bucket
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": days},
                }
            ]
        },
    )


def abort_incomplete_uploads(bucket: str) -> tuple[int, int]:
    """Abort every in-progress multipart upload now. Returns (count, bytes).

    For clearing the backlog that accrued before the lifecycle rule existed;
    the rule handles everything after. Safe to run when no upload is in flight,
    but NOT while a real upload is running — it would abort that too.
    """
    s3 = _client("s3")
    ups = s3.list_multipart_uploads(Bucket=bucket).get("Uploads", [])
    total = 0
    for u in ups:
        try:
            parts = s3.list_parts(Bucket=bucket, Key=u["Key"], UploadId=u["UploadId"])
            total += sum(p["Size"] for p in parts.get("Parts", []))
        except Exception:  # noqa: BLE001 — size is for reporting only
            pass
    _log(f"abort {len(ups)} incomplete multipart upload(s) in {bucket} ({total / 1e9:.2f} GB)")
    if _DRY_RUN:
        return len(ups), total
    for u in ups:
        s3.abort_multipart_upload(Bucket=bucket, Key=u["Key"], UploadId=u["UploadId"])
    return len(ups), total


def object_exists_bucket(s3, bucket: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except Exception:
        return False


class _UploadProgress:
    """Callback that logs an upload's progress every ~10%.

    Only attached to big objects: a 17 GB ``train.bin`` is an hour of complete
    silence otherwise, which is indistinguishable from a hang."""

    STEP = 0.10

    def __init__(self, label: str, total: int):
        self._label = label
        self._total = max(1, total)
        self._sent = 0
        self._next = self.STEP
        self._started = time.time()

    def __call__(self, chunk: int) -> None:
        self._sent += chunk
        frac = self._sent / self._total
        if frac < self._next and self._sent < self._total:
            return
        self._next = frac + self.STEP
        elapsed = max(time.time() - self._started, 1e-6)
        rate = self._sent / elapsed / (1 << 20)
        _log(
            f"upload {self._label}: {frac * 100:.0f}% "
            f"({self._sent / (1 << 30):.1f} GB, {rate:.0f} MB/s, {elapsed:.0f}s)"
        )


# Log progress for anything above this; below it, uploads finish before a first
# progress line would be useful (shakespeare's bins are ~1 MB).
_PROGRESS_THRESHOLD_BYTES = 256 << 20


def upload_file(local_path: str, bucket: str, key: str) -> None:
    """Upload one file. boto3's ``upload_file`` is the *managed* transfer — it
    switches to a threaded multipart upload above 8 MB — so this is the correct
    entrypoint for a 17 GB dataset bin (a raw ``put_object`` would fail: a single
    S3 PUT caps at 5 GB). The SHA-256 is computed per part as the bytes stream,
    so memory stays flat regardless of file size."""
    size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    _log(f"upload {local_path} ({size / (1 << 20):.1f} MB) -> s3://{bucket}/{key}")
    if _DRY_RUN:
        return
    callback = None
    if size >= _PROGRESS_THRESHOLD_BYTES:
        callback = _UploadProgress(key.rsplit("/", 1)[-1], size)
    _client("s3").upload_file(
        local_path,
        bucket,
        key,
        ExtraArgs={"ChecksumAlgorithm": "SHA256"},
        Callback=callback,
    )


def put_text(bucket: str, key: str, body: str, *, quiet: bool = False) -> None:
    """Write a small control document the boxes poll. ``quiet`` logs the key but
    not the body — for documents rewritten on a timer (the remote orchestrator's
    heartbeat), where echoing the payload every few seconds for 36 hours would
    swamp the orchestrator's own log."""
    _log(f"put s3://{bucket}/{key}" + (f" ({len(body)} bytes)" if quiet else f": {body}"))
    if _DRY_RUN:
        return
    _client("s3").put_object(Bucket=bucket, Key=key, Body=body.encode())


def delete_object(bucket: str, key: str) -> None:
    """Delete a control document (e.g. a stale rdzv.json before a whole-group
    restart, so fresh boxes can't dial a dead node 0's address)."""
    _log(f"delete s3://{bucket}/{key}")
    if _DRY_RUN:
        return
    _client("s3").delete_object(Bucket=bucket, Key=key)


# --------------------------------------------------------------------------- #
# Mutating: IAM (instance profile granting the box S3 access)
# --------------------------------------------------------------------------- #
def ensure_instance_profile(role_name: str, profile_name: str, bucket: str) -> None:
    """Create a role the EC2 box assumes, scoped to read/write ``bucket``, and an
    instance profile wrapping it. Idempotent."""
    import json

    _log(f"create IAM role {role_name} + instance profile {profile_name} for s3://{bucket}")
    if _DRY_RUN:
        return
    iam = _client("iam")
    assume = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    _ignore_exists(
        lambda: iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume))
    )
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                # DeleteObject is required: the atomic checkpoint writes a .tmp
                # key, copies it to the final key, then DELETES the .tmp
                # (s3_store._s3_save). Without it, checkpointing fails AccessDenied.
                #
                # AbortMultipartUpload is required for a DIFFERENT reason, and its
                # absence leaks money silently. Checkpoints and the 17 GB dataset
                # go through boto3's managed transfer, i.e. multipart. When an
                # upload fails, boto3 tries to abort it — and abort is NOT covered
                # by PutObject. Denied, the parts stay, billing forever and
                # invisible to `s3 ls`. Preemption testing kills boxes mid-write by
                # design, so this is the normal path, not an edge case: 35 orphans
                # / 4.86 GB accumulated before this was noticed. The List* actions
                # let a box see its own stragglers.
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                    "s3:ListBucketMultipartUploads",
                ],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name, PolicyName="spot-train-s3", PolicyDocument=json.dumps(policy)
    )
    # SSM Session Manager: lets you attach a shell to the box (no inbound ports)
    # to `tail -f /var/log/spot-train-boot.log` and run nvidia-smi.
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )
    _ignore_exists(lambda: iam.create_instance_profile(InstanceProfileName=profile_name))
    # An instance profile holds at most one role; on a re-run the role is already
    # attached and AddRoleToInstanceProfile raises LimitExceeded. Add only if the
    # role isn't already in the profile (idempotent).
    attached = [
        r["RoleName"]
        for r in iam.get_instance_profile(InstanceProfileName=profile_name)["InstanceProfile"][
            "Roles"
        ]
    ]
    if role_name not in attached:
        iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)


def ensure_orchestrator_profile(
    role_name: str, profile_name: str, bucket: str, worker_role_name: str
) -> None:
    """Create the CONTROL PLANE's role + instance profile (``orch up``).

    Strictly more than the worker role and strictly less than your laptop: the
    orchestrator box launches/terminates training instances and reads/writes the
    run bucket, so it gets EC2 lifecycle + S3 + a ``PassRole`` scoped to the
    single worker role and to EC2 only. No IAM writes, no bucket creation.

    A role (not copied keys) is the whole point: the box's credentials are
    refreshed by IMDS for as long as it lives, so a 36-hour run never dies of an
    expired session token. Mirrors docs/iam/orchestrator-policy.json. Idempotent.
    """
    import json

    _log(
        f"create IAM role {role_name} + instance profile {profile_name} "
        f"(EC2 lifecycle + s3://{bucket} + PassRole {worker_role_name})"
    )
    if _DRY_RUN:
        return
    iam = _client("iam")
    assume = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    _ignore_exists(
        lambda: iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume))
    )
    account = _client("sts").get_caller_identity()["Account"]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ec2:RunInstances",
                    "ec2:TerminateInstances",
                    "ec2:CreateTags",
                    "ec2:DescribeInstances",
                    "ec2:DescribeInstanceStatus",
                    "ec2:DescribeImages",
                    "ec2:DescribeSecurityGroups",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeVpcs",
                    "ec2:DescribeSpotPriceHistory",
                    "ec2:AuthorizeSecurityGroupIngress",
                    "ec2:CreateSecurityGroup",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                # DeleteObject: the supervisor clears stale control documents.
                # Multipart actions for the same reason as the worker role: the
                # orchestrator streams logs and profiles through boto3's managed
                # transfer, and an upload it cannot abort leaks billable parts.
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                    "s3:ListBucketMultipartUploads",
                ],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
            {
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": f"arn:aws:iam::{account}:role/{worker_role_name}",
                "Condition": {"StringEquals": {"iam:PassedToService": "ec2.amazonaws.com"}},
            },
            {
                "Effect": "Allow",
                "Action": ["ssm:SendCommand", "ssm:DescribeInstanceInformation"],
                "Resource": "*",
            },
        ],
    }
    iam.put_role_policy(
        RoleName=role_name, PolicyName="spot-train-orch", PolicyDocument=json.dumps(policy)
    )
    # SSM Session Manager on the control plane too: `journalctl -u spot-orch -f`
    # over a 36h run without opening SSH.
    iam.attach_role_policy(
        RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )
    _ignore_exists(lambda: iam.create_instance_profile(InstanceProfileName=profile_name))
    attached = [
        r["RoleName"]
        for r in iam.get_instance_profile(InstanceProfileName=profile_name)["InstanceProfile"][
            "Roles"
        ]
    ]
    if role_name not in attached:
        iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)


def ensure_security_group(name: str, region: str) -> str:
    """Create the security group and ensure it allows inbound SSH (port 22).
    Returns the group id.

    SSH-verification mode: the group used to be egress-only (user-data mode needs
    no inbound). We now open TCP 22 so you can ssh into a bare box. Idempotent —
    AWS raises InvalidPermission.Duplicate if the rule already exists.
    """
    _log(f"ensure security group {name} (inbound SSH :22) in {region}")
    if _DRY_RUN:
        return "sg-DRYRUN"
    ec2 = _client("ec2")
    existing = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]}])[
        "SecurityGroups"
    ]
    gid = (
        existing[0]["GroupId"]
        if existing
        else ec2.create_security_group(GroupName=name, Description="spot-train (SSH verify)")[
            "GroupId"
        ]
    )
    try:
        ec2.authorize_security_group_ingress(
            GroupId=gid,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    # TEMP: open to the world for a quick SSH test. Tighten to your
                    # own IP (e.g. "<your-ip>/32") if the box stays up any length of
                    # time, and revert this whole block when done verifying.
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH (temp verify)"}],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001 — boto ClientError; duplicate rule is fine
        if "InvalidPermission.Duplicate" not in str(e):
            raise
    # Multi-node DDP: allow ALL TCP between instances in this group (the c10d
    # rendezvous TCPStore on node 0 plus the NCCL/gloo data-plane sockets, which
    # use ephemeral ports). Self-referencing, so nothing new is exposed publicly.
    try:
        ec2.authorize_security_group_ingress(
            GroupId=gid,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 0,
                    "ToPort": 65535,
                    "UserIdGroupPairs": [
                        {"GroupId": gid, "Description": "intra-group DDP (rendezvous + NCCL)"}
                    ],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001 — boto ClientError; duplicate rule is fine
        if "InvalidPermission.Duplicate" not in str(e):
            raise
    return gid


def authorize_port(group_id: str, port: int, cidr: str, description: str) -> None:
    """Idempotently open one TCP port on the group (e.g. the fleet router's
    public :8000). Same duplicate-tolerant pattern as ensure_security_group."""
    _log(f"authorize ingress tcp :{port} from {cidr} on {group_id} ({description})")
    if _DRY_RUN:
        return
    try:
        _client("ec2").authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": port,
                    "ToPort": port,
                    "IpRanges": [{"CidrIp": cidr, "Description": description}],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001 — boto ClientError; duplicate rule is fine
        if "InvalidPermission.Duplicate" not in str(e):
            raise


# --------------------------------------------------------------------------- #
# Mutating: EC2 lifecycle
# --------------------------------------------------------------------------- #
def launch(
    *,
    ami_id: str,
    instance_type: str,
    profile_name: str,
    security_group_id: str,
    user_data: str,
    market: str,
    run_id: str,
    key_name: str = "",
    extra_tags: dict[str, str] | None = None,
    root_volume_gb: int = 0,
    root_volume_iops: int = 0,
    root_volume_throughput: int = 0,
) -> str:
    """Launch one instance (on-demand or spot). Returns the instance id.

    ``extra_tags`` are added to the standard Name/project/market tags (the fleet
    uses them for discovery); None keeps the original tag set exactly.

    ``root_volume_gb <= 0`` (the default, and what every training/fleet/control
    -plane launch passes) sends NO BlockDeviceMappings at all, so those launches
    keep inheriting the AMI's own mapping byte-for-byte. A positive value opts in
    to an explicitly sized gp3 root — needed by ``stage-data --remote``, whose
    110 GB of transient dataset caches do not fit any AMI default."""
    volume = (
        f" root={root_volume_gb}GB gp3"
        f"{f'/{root_volume_iops}iops' if root_volume_iops else ''}"
        f"{f'/{root_volume_throughput}MBps' if root_volume_throughput else ''}"
        if root_volume_gb > 0
        else ""
    )
    _log(
        f"RunInstances type={instance_type} market={market} ami={ami_id} "
        f"run_id={run_id} key={key_name or '<none>'} "
        f"user-data={'yes' if user_data else 'none'}{volume} (public IP + SSH ingress)"
    )
    if _DRY_RUN:
        return "i-DRYRUN"
    kwargs: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": profile_name},
        # SSH-verification mode: give the box a public IP so you can reach it, and
        # attach the SG via the interface. NOTE: when you pass NetworkInterfaces you
        # must NOT also set top-level "SecurityGroupIds" — the group goes in here.
        "NetworkInterfaces": [
            {
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "Groups": [security_group_id],
            }
        ],
        # --- ORIGINAL (SG without public IP) — restore when done SSH-testing ---
        # "SecurityGroupIds": [security_group_id],
        "InstanceInitiatedShutdownBehavior": "terminate",
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"spot-train-{run_id}"},
                    {"Key": "project", "Value": "spot-train"},
                    {"Key": "market", "Value": market},
                    *[{"Key": k, "Value": v} for k, v in (extra_tags or {}).items()],
                ],
            }
        ],
    }
    if market == "spot":
        kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }
    # Additive by design: absent unless a caller explicitly asks for a sized root
    # volume, so the proven launches are unchanged (see the docstring).
    if root_volume_gb > 0:
        kwargs["BlockDeviceMappings"] = root_volume_mapping(
            image_root_device(ami_id),
            root_volume_gb,
            iops=root_volume_iops,
            throughput=root_volume_throughput,
        )
    if key_name:  # SSH-verification mode: attach a key pair so you can ssh in
        kwargs["KeyName"] = key_name
    if user_data:  # the boot script (provisioning); empty => bare boot, no user-data
        kwargs["UserData"] = user_data
    r = _client("ec2").run_instances(**kwargs)
    return r["Instances"][0]["InstanceId"]


def wait_running(instance_id: str) -> None:
    _log(f"wait until running: {instance_id}")
    if _DRY_RUN:
        return
    _client("ec2").get_waiter("instance_running").wait(InstanceIds=[instance_id])


def stop_instance(instance_id: str) -> None:
    """Stop (not terminate) an instance — used before CreateImage so the AMI
    snapshots a quiesced filesystem."""
    _log(f"StopInstances {instance_id}")
    if _DRY_RUN:
        return
    _client("ec2").stop_instances(InstanceIds=[instance_id])


def wait_stopped(instance_id: str) -> None:
    _log(f"wait until stopped: {instance_id}")
    if _DRY_RUN:
        return
    _client("ec2").get_waiter("instance_stopped").wait(InstanceIds=[instance_id])


def create_image(instance_id: str, name: str, tags: dict[str, str]) -> str:
    """Register an AMI from the instance's root volume. The instance should be
    stopped (see ``stop_instance``); returns the new image id."""
    _log(f"CreateImage {instance_id} -> {name!r}")
    if _DRY_RUN:
        return "ami-DRYRUN"
    r = _client("ec2").create_image(
        InstanceId=instance_id,
        Name=name,
        TagSpecifications=[
            {
                "ResourceType": "image",
                "Tags": [{"Key": k, "Value": v} for k, v in tags.items()],
            }
        ],
    )
    return r["ImageId"]


def wait_image_available(image_id: str, timeout: int = 1800) -> None:
    """Poll until the AMI's snapshot finishes (state=available). The stock boto3
    waiter gives up after 10 minutes; a DLAMI-sized root volume can take longer,
    hence the hand-rolled loop."""
    _log(f"wait until AMI available: {image_id}")
    if _DRY_RUN:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = _client("ec2").describe_images(ImageIds=[image_id])
        state = r["Images"][0]["State"] if r.get("Images") else "pending"
        if state == "available":
            return
        if state in ("failed", "error"):
            raise SystemExit(f"AMI {image_id} entered state {state!r} — bake failed")
        time.sleep(15)
    raise TimeoutError(f"AMI {image_id} not available after {timeout}s")


def list_baked_images(name_prefix: str) -> list[dict[str, Any]]:
    """Our own AMIs whose name starts with ``name_prefix``, oldest first. Each:
    {id, name, created, snapshot_ids} — snapshot ids so pruning can delete the
    backing storage too (DeregisterImage alone leaves the snapshot billing)."""
    if _DRY_RUN:
        _log(f"DescribeImages self name~={name_prefix}*")
        return []
    r = _client("ec2").describe_images(
        Owners=["self"], Filters=[{"Name": "name", "Values": [f"{name_prefix}*"]}]
    )
    images = sorted(r.get("Images", []), key=lambda i: i["CreationDate"])
    return [
        {
            "id": img["ImageId"],
            "name": img["Name"],
            "created": img["CreationDate"],
            "snapshot_ids": [
                bdm["Ebs"]["SnapshotId"]
                for bdm in img.get("BlockDeviceMappings", [])
                if "Ebs" in bdm and "SnapshotId" in bdm["Ebs"]
            ],
        }
        for img in images
    ]


def deregister_image(image_id: str, snapshot_ids: list[str]) -> None:
    """Delete an old baked AMI and its backing snapshots."""
    _log(f"DeregisterImage {image_id} + delete snapshots {snapshot_ids}")
    if _DRY_RUN:
        return
    _client("ec2").deregister_image(ImageId=image_id)
    for sid in snapshot_ids:
        _client("ec2").delete_snapshot(SnapshotId=sid)


def terminate(instance_id: str) -> None:
    _log(f"TerminateInstances {instance_id}")
    if _DRY_RUN:
        return
    _client("ec2").terminate_instances(InstanceIds=[instance_id])


def wait_quota_released(instance_id: str) -> None:
    """Block until the instance leaves pending/running — the point at which it
    stops counting against the vCPU quota, so a replacement can launch. Do NOT
    wait for full 'terminated': shutting-down can linger for minutes (a hung OS
    shutdown holds it until AWS force-kills) and the quota is already free."""
    _log(f"wait until instance stops counting against quota: {instance_id}")
    if _DRY_RUN:
        return
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if instance_state(instance_id) not in ("pending", "running"):
            return
        time.sleep(5)
    raise TimeoutError(f"{instance_id} still running 300s after TerminateInstances")


def vcpus_in_use() -> int:
    """vCPUs currently counting against the "Running On-Demand G and VT
    instances" quota: every pending/running G- or VT-family instance in the
    region, whoever launched it (external instances eat the same quota, so
    counting only our own would overshoot)."""
    if _DRY_RUN:
        return 0
    total = 0
    paginator = _client("ec2").get_paginator("describe_instances")
    pages = paginator.paginate(
        Filters=[{"Name": "instance-state-name", "Values": ["pending", "running"]}]
    )
    for page in pages:
        for res in page.get("Reservations", []):
            for inst in res.get("Instances", []):
                family = inst.get("InstanceType", "").split(".")[0]
                if family.startswith("g") or family.startswith("vt"):
                    cpu = inst.get("CpuOptions", {})
                    total += cpu.get("CoreCount", 0) * cpu.get("ThreadsPerCore", 1)
    return total


def wait_vcpu_headroom(needed: int, quota: int, timeout: int = 900) -> None:
    """Block until `needed` vCPUs fit under `quota` alongside current usage, so
    RunInstances isn't fired into a quota wall. Polls DescribeInstances every
    15s (one API call per poll — no spam); logs once when it has to wait."""
    if needed > quota:
        raise SystemExit(
            f"Launch needs {needed} vCPUs but VCPU_QUOTA={quota} — it can never fit. "
            "Raise the quota (Service Quotas console) and update VCPU_QUOTA."
        )
    _log(f"wait for vCPU headroom: need {needed} of {quota} quota")
    if _DRY_RUN:
        return
    waiting_logged = False
    deadline = time.monotonic() + timeout
    while True:
        used = vcpus_in_use()
        if used + needed <= quota:
            if waiting_logged:
                _log(f"vCPU headroom available ({used} used + {needed} needed <= {quota})")
            return
        if not waiting_logged:
            _log(f"quota full ({used} used + {needed} needed > {quota}); polling every 15s")
            waiting_logged = True
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no vCPU headroom after {timeout}s ({used} used + {needed} needed > {quota})"
            )
        time.sleep(15)


# --------------------------------------------------------------------------- #
# Mutating: Lambda + EventBridge (the spotwatch collector's control plane)
#
# Everything here is ensure_*: safe to re-run, updating in place rather than
# creating a second copy. `spotwatch deploy` is expected to be run repeatedly.
# --------------------------------------------------------------------------- #
def role_arn(role_name: str) -> str:
    if _DRY_RUN:
        _log(f"get IAM role arn {role_name}")
        return f"arn:aws:iam::000000000000:role/{role_name}"
    return _client("iam").get_role(RoleName=role_name)["Role"]["Arn"]


def ensure_service_role(role_name: str, service: str, policy_name: str, policy: dict) -> str:
    """Create/refresh a role an AWS *service* (not EC2) assumes, with one inline
    policy. Returns the role ARN. Re-running rewrites the policy, so tightening
    permissions is just another `deploy`."""
    import json

    _log(f"create IAM role {role_name} assumable by {service} + inline policy {policy_name}")
    if _DRY_RUN:
        return f"arn:aws:iam::000000000000:role/{role_name}"
    iam = _client("iam")
    assume = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": service},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    _ignore_exists(
        lambda: iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume))
    )
    iam.put_role_policy(
        RoleName=role_name, PolicyName=policy_name, PolicyDocument=json.dumps(policy)
    )
    return iam.get_role(RoleName=role_name)["Role"]["Arn"]


def delete_role(role_name: str) -> None:
    """Delete a role and everything attached to it (IAM refuses otherwise)."""
    _log(f"delete IAM role {role_name} (+ its inline/attached policies)")
    if _DRY_RUN:
        return
    iam = _client("iam")
    try:
        for name in iam.list_role_policies(RoleName=role_name).get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=name)
        for pol in iam.list_attached_role_policies(RoleName=role_name).get("AttachedPolicies", []):
            iam.detach_role_policy(RoleName=role_name, PolicyArn=pol["PolicyArn"])
        iam.delete_role(RoleName=role_name)
    except Exception as e:  # noqa: BLE001 — already gone is a successful teardown
        if "NoSuchEntity" not in str(e):
            raise


def lambda_code_sha256(function_name: str) -> str | None:
    """Base64 SHA-256 of the deployed zip, or None if the function doesn't exist
    — lets `deploy` skip the code update when nothing changed."""
    if _DRY_RUN:
        _log(f"get lambda {function_name} code sha")
        return None
    try:
        return _client("lambda").get_function(FunctionName=function_name)["Configuration"][
            "CodeSha256"
        ]
    except Exception:  # noqa: BLE001 — ResourceNotFound == not deployed yet
        return None


def ensure_lambda_function(
    *,
    function_name: str,
    role: str,
    handler: str,
    runtime: str,
    zip_bytes: bytes,
    env: dict[str, str],
    timeout: int,
    memory_mb: int,
    description: str = "",
) -> str:
    """Create the function, or update code+config in place. Returns its ARN."""
    import base64
    import hashlib

    want_sha = base64.b64encode(hashlib.sha256(zip_bytes).digest()).decode()
    have_sha = lambda_code_sha256(function_name)
    _log(
        f"deploy lambda {function_name} ({len(zip_bytes)}B, {runtime}, {timeout}s, "
        f"{memory_mb}MB) — {'create' if have_sha is None else 'update'}"
    )
    if _DRY_RUN:
        return f"arn:aws:lambda:{_region}:000000000000:function:{function_name}"
    lam = _client("lambda")
    if have_sha is None:
        # A freshly created role isn't immediately assumable by Lambda (IAM is
        # eventually consistent); CreateFunction then fails with
        # InvalidParameterValueException. Retry for ~60s rather than making the
        # operator re-run deploy.
        deadline = time.monotonic() + 60
        while True:
            try:
                r = lam.create_function(
                    FunctionName=function_name,
                    Runtime=runtime,
                    Role=role,
                    Handler=handler,
                    Code={"ZipFile": zip_bytes},
                    Description=description,
                    Timeout=timeout,
                    MemorySize=memory_mb,
                    Environment={"Variables": env},
                    Publish=True,
                )
                return r["FunctionArn"]
            except Exception as e:  # noqa: BLE001
                if "cannot be assumed" not in str(e) or time.monotonic() > deadline:
                    raise
                _log("waiting for the new IAM role to become assumable by Lambda...")
                time.sleep(5)
    if have_sha != want_sha:
        lam.update_function_code(FunctionName=function_name, ZipFile=zip_bytes, Publish=True)
        _wait_lambda_updated(function_name)
    lam.update_function_configuration(
        FunctionName=function_name,
        Role=role,
        Handler=handler,
        Runtime=runtime,
        Timeout=timeout,
        MemorySize=memory_mb,
        Environment={"Variables": env},
        Description=description,
    )
    _wait_lambda_updated(function_name)
    return lam.get_function(FunctionName=function_name)["Configuration"]["FunctionArn"]


def _wait_lambda_updated(function_name: str, timeout: int = 120) -> None:
    """Code and config updates are asynchronous; a second update while the first
    is InProgress fails with ResourceConflictException."""
    lam = _client("lambda")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cfg = lam.get_function_configuration(FunctionName=function_name)
        if cfg.get("LastUpdateStatus") != "InProgress":
            return
        time.sleep(2)


def delete_lambda_function(function_name: str) -> None:
    _log(f"delete lambda {function_name}")
    if _DRY_RUN:
        return
    try:
        _client("lambda").delete_function(FunctionName=function_name)
    except Exception as e:  # noqa: BLE001 — already gone is a successful teardown
        if "ResourceNotFound" not in str(e):
            raise


def ensure_lambda_permission(
    function_name: str, statement_id: str, principal: str, source_arn: str
) -> None:
    """Let ``principal`` (e.g. events.amazonaws.com) invoke the function. The
    statement id makes it idempotent — a duplicate is a no-op, not a second
    grant."""
    _log(f"allow {principal} ({source_arn}) to invoke {function_name} [{statement_id}]")
    if _DRY_RUN:
        return
    try:
        _client("lambda").add_permission(
            FunctionName=function_name,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal=principal,
            SourceArn=source_arn,
        )
    except Exception as e:  # noqa: BLE001
        if "ResourceConflictException" not in str(e):
            raise


def remove_lambda_permission(function_name: str, statement_id: str) -> None:
    _log(f"revoke invoke permission {statement_id} on {function_name}")
    if _DRY_RUN:
        return
    try:
        _client("lambda").remove_permission(FunctionName=function_name, StatementId=statement_id)
    except Exception as e:  # noqa: BLE001 — already gone is a successful teardown
        if "ResourceNotFound" not in str(e):
            raise


def ensure_schedule_rule(rule_name: str, schedule: str, description: str = "") -> str:
    """EventBridge rule on a fixed cadence, e.g. ``rate(10 minutes)``. PutRule is
    an upsert, so changing the cadence is just another deploy."""
    _log(f"put EventBridge rule {rule_name} schedule={schedule!r}")
    if _DRY_RUN:
        return f"arn:aws:events:{_region}:000000000000:rule/{rule_name}"
    return _client("events").put_rule(
        Name=rule_name, ScheduleExpression=schedule, State="ENABLED", Description=description
    )["RuleArn"]


def put_rule_target(rule_name: str, target_id: str, target_arn: str) -> None:
    """Point the rule at the function. Same target id => replaced, not added."""
    _log(f"put target {target_id} -> {target_arn} on rule {rule_name}")
    if _DRY_RUN:
        return
    _client("events").put_targets(Rule=rule_name, Targets=[{"Id": target_id, "Arn": target_arn}])


def delete_schedule_rule(rule_name: str, target_ids: list[str]) -> None:
    """Remove the targets then the rule (EventBridge refuses to delete a rule
    that still has targets)."""
    _log(f"delete EventBridge rule {rule_name} (targets {target_ids})")
    if _DRY_RUN:
        return
    events = _client("events")
    try:
        events.remove_targets(Rule=rule_name, Ids=target_ids)
        events.delete_rule(Name=rule_name)
    except Exception as e:  # noqa: BLE001 — already gone is a successful teardown
        if "ResourceNotFound" not in str(e):
            raise


def _ignore_exists(fn) -> None:
    """Run an idempotent IAM create, swallowing 'already exists' errors."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — boto ClientError; treat EntityAlreadyExists as ok
        if "EntityAlreadyExists" not in str(e) and "already exists" not in str(e):
            raise
