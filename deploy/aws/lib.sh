#!/usr/bin/env bash
# Shared guards for every inference AWS script. Source, don't run.
#
# The whole isolation story is region + tag. Training owns us-east-1 and runs
# 8-node g5.xlarge fleets there that auto-replace preempted nodes; landing in
# that region would race it for GPU capacity and surface as "capacity errors"
# rather than as contention. So every script here pins the region explicitly
# instead of inheriting an ambient default, and scopes every mutation by
# tag:project.

set -euo pipefail

export AWS_PAGER=""
INF_REGION="${INFERENCE_REGION:-us-east-2}"
PROJECT_TAG="${INFERENCE_PROJECT_TAG:-inference}"
BUCKET="${INFERENCE_BUCKET:-optitrain-inference-us-east-2}"
SG_NAME="${INFERENCE_SECURITY_GROUP:-inference-sg}"
IAM_PROFILE="${INFERENCE_IAM_PROFILE:-inference-profile}"

die() { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '  \033[32m✓\033[0m %s\n' "$*"; }
note(){ printf '  \033[33m•\033[0m %s\n' "$*"; }

# Hard stop, not a convention. A typo in one variable must not be able to reach
# the other project's region.
[ "$INF_REGION" = "us-east-1" ] && \
  die "us-east-1 belongs to the training project — refusing (see docs/prompts/inference-agent-region.md)"
[ "$INF_REGION" = "us-east-2" ] || \
  note "region is $INF_REGION (expected us-east-2)"

aws2() { aws --region "$INF_REGION" "$@"; }

# Instance ids belonging to THIS project only. Every destructive call must be
# built from this, never from an unfiltered describe.
ours_ids() {
  aws2 ec2 describe-instances \
    --filters Name=instance-state-name,Values=running,pending,stopping,stopped \
              "Name=tag:project,Values=${PROJECT_TAG}" \
              ${1:+"Name=tag:k3s_cluster,Values=$1"} \
    --query 'Reservations[].Instances[].InstanceId' --output text | tr '\t' ' '
}

require_creds() {
  aws sts get-caller-identity >/dev/null 2>&1 || die "no working AWS credentials"
}
