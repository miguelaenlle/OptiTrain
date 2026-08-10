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
  # FAIL LOUD, NOT EMPTY. This used to swallow stderr and yield "" on any API
  # error, which is indistinguishable from "nothing is running" -- so one expired
  # credential or a RequestLimitExceeded produced a fully green teardown report
  # while GPUs kept billing. `watch` polls describe-instances every 15s for
  # hours, so throttling is a realistic trigger, not a hypothetical.
  #
  # Callers MUST check the exit status: 0 = the list is authoritative (possibly
  # empty), non-zero = we do not know, assume something is running.
  local out rc
  out=$(aws ec2 describe-instances --region "$TRAIN_REGION" \
    --filters "$_LIVE" "Name=tag:project,Values=${PROJECT_TAG}" \
    --query 'Reservations[].Instances[].InstanceId' --output text 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    echo "[fleetctl] EC2 QUERY FAILED (rc=$rc): $out" >&2
    return 1
  fi
  printf '%s' "$out" | tr '\t' ' '
}
ours_count() {
  # Prints a number and returns 0, or prints nothing and returns 1 when the
  # query failed. `n=$(ours_count); rc=$?` works -- exit status survives command
  # substitution -- so callers can tell "zero" from "unknown".
  local i rc
  i=$(ours_ids); rc=$?
  [ $rc -eq 0 ] || return 1
  [ -z "$i" ] && echo 0 || echo "$i" | wc -w | tr -d ' '
}

others_count() {
  # Arithmetic, not a JMESPath negation. The negation form
  #   length(...Instances[?!(Tags[?Key=='project' && Value=='X'])])
  # UNDERCOUNTS (verified against a fixture: 3 instances, 1 ours, it answers 1
  # instead of 2) -- untagged instances and the [] flatten interact badly with
  # `!`. This line exists to reassure the operator that nothing else was
  # touched, so it has to be obviously right rather than clever.
  local total ours
  total=$(aws ec2 describe-instances --region "$TRAIN_REGION" --filters "$_LIVE" \
    --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0)
  ours=$(ours_count)
  echo $(( ${total:-0} - ${ours:-0} ))
}

reap_ours() {
  # CLOSED LOOP. The old version printed "terminating ..." BEFORE the call,
  # discarded stdout+stderr, never checked the exit code and never re-queried --
  # so it announced success whatever happened. This is the stop button on a
  # multi-hundred-dollar run; it has to report what is TRUE, not what was
  # attempted.
  local ids err rc
  ids=$(ours_ids); rc=$?
  if [ $rc -ne 0 ]; then
    # We could not read EC2. Refusing is the only honest answer: reporting
    # "nothing running" here is exactly how a teardown looks green while the
    # fleet bills.
    echo "[reap] CANNOT VERIFY -- EC2 query failed. Nothing was terminated." >&2
    echo "[reap] Retry, or terminate by hand in the console before walking away." >&2
    return 1
  fi
  if [ -z "$ids" ]; then
    echo "[reap] nothing of ours running in ${TRAIN_REGION}"
    echo "[reap] left untouched (not ours): $(others_count)"
    return 0
  fi
  echo "[reap] terminating ${PROJECT_TAG} in ${TRAIN_REGION}: $ids"
  err=$(aws ec2 terminate-instances --region "$TRAIN_REGION" --instance-ids $ids 2>&1 >/dev/null)
  rc=$?
  if [ "$rc" != "0" ]; then
    echo "[reap] ERROR: terminate-instances failed (rc=$rc): $err" >&2
  fi
  # Poll until they leave running/pending -- the point at which billing stops.
  # Do NOT wait for full 'terminated': shutting-down can linger for minutes.
  local left
  for _ in $(seq 1 30); do
    left=$(ours_count)
    [ "$left" = "0" ] && break
    sleep 2
  done
  if [ "${left:-1}" = "0" ]; then
    echo "[reap] VERIFIED: 0 of ours still running in ${TRAIN_REGION}"
  else
    echo "[reap] STILL RUNNING after 60s: $(ours_ids)" >&2
    echo "[reap] terminate them by hand before walking away:" >&2
    echo "       aws ec2 terminate-instances --region ${TRAIN_REGION} --instance-ids \$(ours_ids)" >&2
    return 1
  fi
  echo "[reap] left untouched (not ours): $(others_count)"
}
