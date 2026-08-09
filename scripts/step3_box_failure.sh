#!/usr/bin/env bash
# STEP 3 — control-plane BOX failure, then resume onto a fresh one.
#
#   scripts/step3_box_failure.sh [--nodes 2] [--keep] [--dry-run]
#
# systemd resurrects a supervisor PROCESS; nothing resurrects the BOX it runs
# on. This terminates the control plane outright and proves two things:
#
#   1. the fleet keeps TRAINING without it -- sidecars run static torchrun
#      against the last published epoch doc, so the supervisor was never in the
#      training data path;
#   2. `orch up --run-id` ADOPTS the still-running fleet instead of rebuilding
#      it, which is the only way back from a dead control plane.
#
# Everything is condition-driven, never `sleep N && hope`: an earlier attempt
# was lost because a fixed wait raced a run that had already finished.
#
# COST ~$1.50 (2x g5.xlarge for ~20 min + two t3.medium control planes).
set -uo pipefail
cd "$(dirname "$0")/.."

NODES=2
KEEP=0
DRY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --nodes) NODES="$2"; shift 2 ;;
    --keep)  KEEP=1; shift ;;          # leave the fleet up at the end
    --dry-run) DRY="--dry-run"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
. scripts/fleetctl.sh

# --- the budget knob -------------------------------------------------------- #
# Pass TRAIN_BUDGET_SECONDS, never the experiment's own key. `multinode` reads
# BASELINE_SECONDS while `multinode-preempt` reads TRAIN_TOTAL_SECONDS, and
# naming the wrong one is SILENT -- the run just uses the 300s default, trains
# ~200s and completes before the test can start. That cost two attempts.
# orch.up translates this into whichever key the experiment actually reads.
BUDGET=2400
OUTAGE=180          # seconds to leave the control plane dead
STEADY_STEPS=15     # kill only once training is unambiguously underway

COMMON_ENV=(
  --env MARKET=on-demand
  --env "NODES=$NODES"
  --env "TRAIN_BUDGET_SECONDS=$BUDGET"
  --env CHECKPOINT_INTERVAL_SECONDS=30
  --env LOG_INTERVAL_STEPS=1        # loss from step 1, and exact gap timing
  --env EVAL_INTERVAL_STEPS=25
  --env LR_DECAY_STEPS=20000
  --env MAX_STEPS=24000
  --env CHECKPOINT_KEEP=10
  --env VCPU_QUOTA=64
)

ORCH1=""; ORCH2=""; RID=""; LIVE_PID=""
cleanup() {
  [ -n "${LIVE_PID:-}" ] && kill "$LIVE_PID" 2>/dev/null
  if [ "$KEEP" = "1" ]; then
    echo; echo "[step3] --keep set: leaving everything up. Tear down with:"
    [ -n "$ORCH2" ] && echo "         spot-orchestrate orch down --id $ORCH2 --all"
    [ -n "$ORCH1" ] && echo "         spot-orchestrate orch down --id $ORCH1 --all"
    return
  fi
  echo; echo "[step3] tearing down"
  for o in "$ORCH2" "$ORCH1"; do
    [ -n "$o" ] && python3 -m orchestrator orch down --id "$o" --all >/dev/null 2>&1
  done
  reap_ours                          # region + tag scoped; belt and braces
  echo "[step3] done. Artifacts kept: deploy/grafana/.live/$RID/ (replayable)"
}
trap cleanup EXIT INT TERM

s3() { aws s3 cp "s3://$SPOT_TRAIN_BUCKET/$1" - --region us-east-1 2>/dev/null; }
fleet_size() {
  aws ec2 describe-instances --region us-east-1 \
    --filters "Name=tag:Name,Values=spot-train-$RID" \
              "Name=instance-state-name,Values=running,pending" \
    --query "length(Reservations[].Instances[])" --output text 2>/dev/null || echo 0
}
epoch_now() { s3 "runs/$RID/epoch.json" | python3 -c "import json,sys;print(json.load(sys.stdin).get('epoch',0))" 2>/dev/null || echo 0; }
steps_now() { grep -hcE '^step [0-9]+: loss' deploy/grafana/.live/"$RID"/logs/boot-node*.log 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0; }
say() { printf '\n\033[1m[step3 %s] %s\033[0m\n' "$(date -u +%H:%M:%S)" "$*"; }

# --- 0. preflight ----------------------------------------------------------- #
say "preflight"
aws sts get-caller-identity --region us-east-1 --query Arn --output text || exit 1
LEFTOVER=$(aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:project,Values=spot-train" "Name=instance-state-name,Values=running,pending" \
  --query "length(Reservations[].Instances[])" --output text)
if [ "$LEFTOVER" != "0" ]; then
  echo "[step3] $LEFTOVER instance(s) already running — reap them first:" >&2
  echo "        . scripts/fleetctl.sh && reap_ours" >&2
  exit 1
fi
docker ps --format '{{.Names}}' | grep -q spot-train-grafana-grafana \
  || echo "[step3] WARNING: Grafana not up. (cd deploy/grafana && docker compose up -d)"

# --- 1. launch -------------------------------------------------------------- #
say "launching $NODES-node clean run (budget ${BUDGET}s, no chaos)"
OUT=$(python3 -m orchestrator orch up --no-attach $DRY \
  --experiment multinode "${COMMON_ENV[@]}" 2>&1) || { echo "$OUT" >&2; exit 1; }
echo "$OUT" | tail -4
ORCH1=$(echo "$OUT" | grep -oE 'orch-[0-9]{8}-[0-9]{6}' | head -1)
[ -n "$ORCH1" ] || { echo "[step3] no orch id in output" >&2; exit 1; }
[ -n "$DRY" ] && { echo "[step3] dry-run complete"; exit 0; }

say "waiting for the control plane to mint a run id (orch=$ORCH1)"
for _ in $(seq 1 40); do
  RID=$(s3 "orchestrators/$ORCH1/orch.json" | python3 -c "import json,sys;print(json.load(sys.stdin).get('run_id',''))" 2>/dev/null)
  [ -n "$RID" ] && break; sleep 10
done
[ -n "$RID" ] || { echo "[step3] control plane never minted a run id" >&2; exit 1; }
CP1=$(s3 "orchestrators/$ORCH1/orch.json" | python3 -c "import json,sys;print(json.load(sys.stdin).get('instance_id',''))")
echo "[step3] run_id=$RID  control_plane=$CP1"

# --- 2. dashboard ----------------------------------------------------------- #
python3 deploy/grafana/live.py "$RID" --interval=10 --nodes="$NODES" \
  "--log=deploy/grafana/.live/$RID/logs/orchestrator.log" > deploy/grafana/live.log 2>&1 &
LIVE_PID=$!
START_MS=$(( $(date -u +%s) * 1000 - 60000 ))
say "DASHBOARD → http://localhost:3001/d/dist-training/?var-run=$RID&from=$START_MS&to=now&refresh=10s"

# --- 3. wait for steady training -------------------------------------------- #
say "waiting for >= $STEADY_STEPS logged steps (boot + dataset pull ~4 min)"
for _ in $(seq 1 120); do
  N=$(steps_now); E=$(epoch_now)
  printf '\r  steps=%-5s epoch=%-3s fleet=%-3s' "${N:-0}" "${E:-0}" "$(fleet_size)"
  [ "${N:-0}" -ge "$STEADY_STEPS" ] 2>/dev/null && break
  sleep 10
done; echo

# --- 4. record, then KILL THE BOX ------------------------------------------- #
PRE_EPOCH=$(epoch_now); PRE_FLEET=$(fleet_size); PRE_STEPS=$(steps_now)
PRE_TICK=$(aws s3 ls "s3://$SPOT_TRAIN_BUCKET/runs/$RID/status/" --region us-east-1 | tail -1 | awk '{print $4}')
say "PRE-KILL  epoch=$PRE_EPOCH fleet=$PRE_FLEET steps=$PRE_STEPS"
say "TERMINATING CONTROL PLANE $CP1 — this is permanent, nothing will restart it"
aws ec2 terminate-instances --region us-east-1 --instance-ids "$CP1" \
  --query "TerminatingInstances[].CurrentState.Name" --output text

# --- 5. observe the outage --------------------------------------------------- #
say "outage: ${OUTAGE}s. Training must CONTINUE; world size/Gantt must FREEZE."
for _ in $(seq 1 $((OUTAGE/15))); do
  printf '\r  steps=%-5s (was %s)  fleet=%-3s  <- steps must climb, fleet must hold' \
    "$(steps_now)" "$PRE_STEPS" "$(fleet_size)"
  sleep 15
done; echo
MID_STEPS=$(steps_now); MID_FLEET=$(fleet_size)

# --- 6. resume onto a fresh control plane ------------------------------------ #
say "resuming: orch up --run-id $RID"
OUT2=$(python3 -m orchestrator orch up --no-attach \
  --experiment multinode --run-id "$RID" "${COMMON_ENV[@]}" 2>&1) || { echo "$OUT2" >&2; exit 1; }
ORCH2=$(echo "$OUT2" | grep -oE 'orch-[0-9]{8}-[0-9]{6}' | head -1)
echo "$OUT2" | grep -E "adopting|instance i-" | head -3

say "waiting for the new control plane to adopt (up to 5 min)"
for _ in $(seq 1 30); do
  if s3 "runs/$RID/logs/orchestrator.log" | grep -q "re-adopted epoch"; then break; fi
  printf '\r  fleet=%-3s epoch=%-3s steps=%-5s' "$(fleet_size)" "$(epoch_now)" "$(steps_now)"
  sleep 10
done; echo

# --- 7. verdict -------------------------------------------------------------- #
POST_EPOCH=$(epoch_now); POST_FLEET=$(fleet_size); POST_STEPS=$(steps_now)
LOG=$(s3 "runs/$RID/logs/orchestrator.log")
say "RESULT"
printf '  %-42s %s\n' "steps advanced during outage"  "$PRE_STEPS -> $MID_STEPS"
printf '  %-42s %s\n' "fleet before / during / after" "$PRE_FLEET / $MID_FLEET / $POST_FLEET"
printf '  %-42s %s\n' "epoch before / after"          "$PRE_EPOCH / $POST_EPOCH"
echo
pass=0; fail=0
chk() { if [ "$1" = "1" ]; then printf '  \033[32mPASS\033[0m  %s\n' "$2"; pass=$((pass+1));
        else printf '  \033[31mFAIL\033[0m  %s\n' "$2"; fail=$((fail+1)); fi; }
chk "$([ "${MID_STEPS:-0}" -gt "${PRE_STEPS:-0}" ] && echo 1 || echo 0)" "training continued while the control plane was dead"
chk "$([ "$MID_FLEET" = "$PRE_FLEET" ] && echo 1 || echo 0)"             "fleet untouched during the outage ($PRE_FLEET)"
chk "$([ "$POST_FLEET" = "$PRE_FLEET" ] && echo 1 || echo 0)"            "fleet NOT rebuilt after resume (still $PRE_FLEET, not $((PRE_FLEET*2)))"
chk "$([ "${POST_EPOCH:-0}" -ge "${PRE_EPOCH:-0}" ] && echo 1 || echo 0)" "epoch never went backwards"
chk "$(echo "$LOG" | grep -q 're-adopted epoch' && echo 1 || echo 0)"    "supervisor re-adopted the live world"
chk "$(echo "$LOG" | grep -q 'adopted .* running box' && echo 1 || echo 0)" "driver adopted the running boxes"
chk "$(echo "$LOG" | grep -cq 'published epoch 1' && echo 0 || echo 1)"  "no epoch-1 republish (the rehearsal's cascade)"
chk "$([ "${POST_STEPS:-0}" -gt "${MID_STEPS:-0}" ] && echo 1 || echo 0)" "training still advancing after resume"

# The control-plane-down band should cover the outage, and only the outage.
python3 deploy/grafana/export.py "$RID" --live --nodes="$NODES" \
  --src="deploy/grafana/.live/$RID" >/dev/null 2>&1
BAND=$(python3 -c "
import json
try: r=json.load(open('deploy/grafana/data/$RID/control_plane_down.json'))
except Exception: r=[]
print(len(r), *(f\"{(x['timeEnd']-x['time'])/1000:.0f}s\" for x in r))" 2>/dev/null)
chk "$([ "${BAND%% *}" = "1" ] && echo 1 || echo 0)" "dashboard drew exactly one 'control plane down' band ($BAND)"

echo
if [ "$fail" = "0" ]; then printf '\033[32m  ALL %s CHECKS PASSED\033[0m\n' "$pass"
else printf '\033[31m  %s PASSED, %s FAILED\033[0m\n' "$pass" "$fail"; fi
echo
echo "  Dashboard (replayable after teardown):"
echo "  http://localhost:3001/d/dist-training/?var-run=$RID&from=$START_MS&to=now"
[ "$fail" = "0" ] || exit 1
