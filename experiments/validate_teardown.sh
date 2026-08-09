#!/usr/bin/env bash
# Did the teardown work? Answer without trusting anything that printed earlier.
#
#   experiments/validate_teardown.sh            # check + report
#   experiments/validate_teardown.sh --fix      # ...and force-kill anything left
#
# Every teardown path in this repo used to be OPEN LOOP: issue terminate, print
# a success message, exit. This closes the loop from OUTSIDE all of them -- it
# asks EC2 directly, so it is unaffected by a swallowed prompt, a discarded
# stderr, or a message that describes intent rather than outcome.
#
# Exit 0 = nothing of ours is billing. Exit 1 = something is, and it says what.
set -uo pipefail
cd "$(dirname "$0")/.."

FIX=0
[ "${1:-}" = "--fix" ] && FIX=1

set -a && . ./.env && set +a
REGION="${TRAIN_REGION:-us-east-1}"
TAG="${PROJECT_TAG:-spot-train}"

hr() { printf '%s\n' "------------------------------------------------------------"; }
ok()   { printf '  \033[32mOK  \033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mBAD \033[0m %s\n' "$1"; }

echo; hr; echo "  TEARDOWN CHECK — region $REGION, tag project=$TAG"; hr

# --- 1. billing state, straight from EC2 ------------------------------------ #
# running + pending are the only states AWS bills for. shutting-down and
# terminated are free, so a box mid-shutdown is NOT a failure -- an earlier
# check of mine called that a bug because it re-queried with no delay and raced
# the transition.
LIVE=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:project,Values=$TAG" "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' \
  --output text 2>/dev/null)

SHUTTING=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:project,Values=$TAG" "Name=instance-state-name,Values=shutting-down" \
  --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0)

if [ -z "$LIVE" ]; then
  ok "0 instances billing"
  [ "${SHUTTING:-0}" != "0" ] && echo "       ($SHUTTING still shutting down — free, will finish on their own)"
else
  bad "$(echo "$LIVE" | wc -l | tr -d ' ') instance(s) STILL BILLING:"
  echo "$LIVE" | awk '{printf "         %s  %-12s %-10s %s\n", $1, $2, $3, $4}'
fi

# --- 2. the helpers agree ----------------------------------------------------- #
. scripts/fleetctl.sh
OURS=$(ours_count); OTHERS=$(others_count)
# grep -c exits 1 on no match, so `|| echo 0` would append a SECOND zero and
# make "0" != "0\n0". Count with awk, which exits 0 either way.
EC2N=$(printf '%s' "$LIVE" | awk 'NF{n++} END{print n+0}')
[ "${OURS:-0}" = "$EC2N" ] \
  && ok "ours_count agrees with EC2 ($OURS)" \
  || bad "ours_count says $OURS, EC2 says $EC2N"
[ "${OTHERS:-0}" -ge 0 ] 2>/dev/null \
  && ok "others_count = $OTHERS (instances we must never touch)" \
  || bad "others_count returned garbage: '$OTHERS'"

# --- 3. control planes -------------------------------------------------------- #
CP=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:orch_role,Values=controller" "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null)
[ -z "$CP" ] && ok "no control plane left running" || bad "control plane(s) still up: $CP"

# --- 4. is a run still writing? ----------------------------------------------- #
# A live supervisor writes a status tick every few seconds. A recent one means
# something is still orchestrating even if you thought you stopped it.
# Sort by the unix timestamp the run id EMBEDS, not lexically -- `aws s3 ls`
# orders by name, so "scaling1-..." beat "multinode-1786311446" and this checked
# a run from days ago.
NEWEST=$(aws s3 ls "s3://$SPOT_TRAIN_BUCKET/runs/" --region "$REGION" 2>/dev/null \
  | awk '{print $2}' | tr -d '/' \
  | awk -F- '{print $NF, $0}' | sort -k1,1n | tail -1 | awk '{print $2}')
if [ -n "$NEWEST" ]; then
  LASTTICK=$(aws s3 ls "s3://$SPOT_TRAIN_BUCKET/runs/$NEWEST/status/" --region "$REGION" 2>/dev/null \
    | tail -1 | awk '{print $4}' | sed 's/\.json//')
  if [ -n "$LASTTICK" ]; then
    # Tick filenames are zero-padded to 15 digits, and bash reads a leading-zero
    # number as OCTAL ("value too great for base"). 10# forces base 10.
    AGE=$(( $(date -u +%s) - 10#$LASTTICK/1000 ))
    # A recent tick only means something is RUNNING if instances are actually
    # alive. Right after a teardown the last tick is seconds old because the
    # supervisor wrote it just before dying -- flagging that is a false alarm,
    # and a stop button that cries wolf gets ignored.
    if [ -z "$LIVE" ]; then
      ok "newest run ($NEWEST) last wrote ${AGE}s ago, but nothing is running"
    elif [ "$AGE" -gt 120 ]; then
      ok "newest run ($NEWEST) went quiet ${AGE}s ago — no live supervisor"
    else
      bad "run $NEWEST wrote a status tick ${AGE}s ago AND instances are up — STILL RUNNING"
    fi
  else
    ok "newest run ($NEWEST) has no status ticks"
  fi
fi

# --- 5. optional force-kill ---------------------------------------------------- #
if [ -n "$LIVE" ] && [ "$FIX" = "1" ]; then
  hr; echo "  --fix: terminating everything of ours"
  IDS=$(echo "$LIVE" | awk '{print $1}' | tr '\n' ' ')
  aws ec2 terminate-instances --region "$REGION" --instance-ids $IDS \
    --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
  echo "  waiting for them to stop billing..."
  for _ in $(seq 1 30); do
    [ "$(ours_count)" = "0" ] && break
    sleep 2
  done
  [ "$(ours_count)" = "0" ] && ok "verified: 0 billing" || bad "STILL BILLING — kill by hand: $IDS"
fi

hr
if [ -z "$LIVE" ] && [ -z "$CP" ]; then
  printf '  \033[32mCLEAN — nothing is billing\033[0m\n\n'; exit 0
fi
printf '  \033[31mNOT CLEAN\033[0m — re-run with --fix, or:\n'
printf '    . scripts/fleetctl.sh && reap_ours\n\n'
exit 1
