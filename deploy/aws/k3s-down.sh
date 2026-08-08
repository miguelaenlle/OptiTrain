#!/usr/bin/env bash
# Terminate an inference k3s cluster in us-east-2.
#
#   ./deploy/aws/k3s-down.sh <cluster-id>   one cluster
#   ./deploy/aws/k3s-down.sh --all          every project=inference instance
#
# Every selection is scoped by BOTH region and tag:project. There is
# deliberately no code path here that builds an instance list without a tag
# filter — an unfiltered describe piped into terminate-instances is an
# account-wide kill switch, and the training project shares this account.

cd "$(dirname "$0")/../.."
. deploy/aws/lib.sh
require_creds

TARGET="${1:-}"
[ -z "$TARGET" ] && die "usage: k3s-down.sh <cluster-id> | --all"

if [ "$TARGET" = "--all" ]; then
  IDS=$(ours_ids)
  SCOPE="every project=${PROJECT_TAG} instance in ${INF_REGION}"
else
  IDS=$(ours_ids "$TARGET")
  SCOPE="cluster ${TARGET} in ${INF_REGION}"
fi

if [ -z "${IDS// /}" ]; then
  ok "nothing to terminate — $SCOPE is already empty"
  exit 0
fi

echo "About to terminate ($SCOPE):"
for id in $IDS; do
  desc=$(aws2 ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].[InstanceType,Tags[?Key==`Name`]|[0].Value,Tags[?Key==`project`]|[0].Value]' \
    --output text)
  echo "   $id  $desc"
done
echo

# Belt and braces: re-verify every id carries our project tag before the
# destructive call, in case the filter above was ever edited badly.
for id in $IDS; do
  p=$(aws2 ec2 describe-instances --instance-ids "$id" \
        --query 'Reservations[0].Instances[0].Tags[?Key==`project`]|[0].Value' --output text)
  [ "$p" = "$PROJECT_TAG" ] || die "refusing: $id has project=$p, expected $PROJECT_TAG"
done

aws2 ec2 terminate-instances --instance-ids $IDS \
  --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
ok "terminate issued"

echo "  waiting for shutdown…"
aws2 ec2 wait instance-terminated --instance-ids $IDS 2>/dev/null || true
ok "all terminated — billing stopped"

REMAIN=$(ours_ids)
if [ -n "${REMAIN// /}" ]; then
  note "other project=${PROJECT_TAG} instances still running: $REMAIN"
else
  ok "no ${PROJECT_TAG} instances remain in ${INF_REGION}"
fi
