#!/usr/bin/env bash
# Terminate the GPU experiment box. Scoped by region AND tag:project, like
# every destructive path here — there is no code path that builds an instance
# list without a tag filter.
cd "$(dirname "$0")/../.."
. deploy/aws/lib.sh
require_creds

IDS=$(aws2 ec2 describe-instances \
  --filters Name=instance-state-name,Values=running,pending,stopping,stopped \
            "Name=tag:project,Values=${PROJECT_TAG}" \
            "Name=tag-key,Values=gpu_run" \
  --query 'Reservations[].Instances[].InstanceId' --output text | tr '\t' ' ')

[ -z "${IDS// /}" ] && { ok "no GPU instances running"; exit 0; }

for id in $IDS; do
  p=$(aws2 ec2 describe-instances --instance-ids "$id" \
        --query 'Reservations[0].Instances[0].Tags[?Key==`project`]|[0].Value' --output text)
  [ "$p" = "$PROJECT_TAG" ] || die "refusing: $id has project=$p"
  t=$(aws2 ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].InstanceType' --output text)
  echo "  terminating $id ($t)"
done
aws2 ec2 terminate-instances --instance-ids $IDS --query 'TerminatingInstances[].InstanceId' --output text
aws2 ec2 wait instance-terminated --instance-ids $IDS 2>/dev/null || true
ok "GPU instances terminated — billing stopped"
