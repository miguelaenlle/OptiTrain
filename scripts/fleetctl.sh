#!/usr/bin/env bash
# Project-scoped instance queries. SOURCE THIS instead of writing bare
# `aws ec2 describe-instances` in an experiment driver.
#
# Every training/fleet box this repo launches is stamped project=spot-train
# (aws.py TagSpecifications) and lives in us-east-1. Another workload — an
# inference platform — runs in the SAME AWS account out of us-east-2, so an
# unfiltered query could return THEIR instances too, and the reap traps in our
# drivers pipe that list straight into terminate-instances. That is an
# account-wide kill switch, not a cleanup.
#
# TWO independent guards, deliberately redundant:
#   --region  keeps us out of us-east-2 entirely  (docs/region-split.md)
#   tag       keeps us off anything unlabelled inside us-east-1
# Either alone would do today; both mean a mistake in one is not fatal.
#
#   ours_ids           -> instance ids we own (running/pending)
#   ours_count         -> how many
#   reap_ours          -> terminate ONLY ours
#   others_count       -> how many instances we do NOT own (context, never touched)
PROJECT_TAG="${PROJECT_TAG:-spot-train}"
TRAIN_REGION="${TRAIN_REGION:-us-east-1}"
_LIVE="Name=instance-state-name,Values=running,pending"

ours_ids() {
  aws ec2 describe-instances --region "$TRAIN_REGION" \
    --filters "$_LIVE" "Name=tag:project,Values=${PROJECT_TAG}" \
    --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | tr '\t' ' '
}
ours_count() { local i; i=$(ours_ids); [ -z "$i" ] && echo 0 || echo "$i" | wc -w | tr -d ' '; }
others_count() {
  aws ec2 describe-instances --region "$TRAIN_REGION" --filters "$_LIVE" \
    --query "length(Reservations[].Instances[?!(Tags[?Key=='project' && Value=='${PROJECT_TAG}'])])" \
    --output text 2>/dev/null || echo 0
}
reap_ours() {
  local ids; ids=$(ours_ids)
  if [ -n "$ids" ]; then
    echo "[reap] terminating ${PROJECT_TAG} in ${TRAIN_REGION}: $ids"
    aws ec2 terminate-instances --region "$TRAIN_REGION" --instance-ids $ids >/dev/null 2>&1
  fi
  echo "[reap] left untouched (not ours, ${TRAIN_REGION}): $(others_count)"
}
