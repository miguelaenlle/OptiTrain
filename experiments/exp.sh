#!/usr/bin/env bash
# One idempotent entry point for every paid experiment.
#
#   experiments/exp.sh <name> up       start, or REJOIN if already running
#   experiments/exp.sh <name> status   one-shot snapshot
#   experiments/exp.sh <name> watch    follow until it ends (Ctrl-C is safe)
#   experiments/exp.sh <name> dash     ensure the Grafana loop + print the URL
#   experiments/exp.sh <name> verify   PASS/FAIL checks for this experiment
#   experiments/exp.sh <name> down     verified teardown
#   experiments/exp.sh list            what exists and what is live
#
# WHY IT IS SHAPED THIS WAY
#
# The operator and an agent both need to drive the same run, from different
# shells, at different times, without stepping on each other. So there is NO
# local state: the pointer lives in S3 at experiments/<name>/current.json, and
# every subcommand rediscovers from there. A fresh clone, a second terminal, or
# an agent that has never seen this run can all `status` it immediately.
#
# `up` is idempotent because a second launch is the expensive mistake: it would
# double an 8-node fleet. It reads the pointer, checks whether that control
# plane is STILL ALIVE in EC2 (the pointer alone is not proof -- the box may
# have died), and rejoins if so. Only a genuinely dead or absent run launches.
#
# Ctrl-C anywhere is safe. Nothing here owns the run; the control plane is on
# AWS and keeps going. That is the same property the whole system is built on.
set -uo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-}"; CMD="${2:-status}"
[ -n "$NAME" ] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

set -a && . ./.env && set +a
REGION="${TRAIN_REGION:-us-east-1}"
B="$SPOT_TRAIN_BUCKET"
PTR="experiments/$NAME/current.json"

# --------------------------------------------------------------------------- #
# Experiment definitions. Add a case; everything else is generic.
#
# BUDGET is passed as TRAIN_BUDGET_SECONDS and NEVER as an experiment's own key:
# `multinode` reads BASELINE_SECONDS, `multinode-preempt` reads
# TRAIN_TOTAL_SECONDS, and naming the wrong one is SILENT -- the run takes the
# 300s default, trains ~200s and finishes before the test starts. orch up
# translates TRAIN_BUDGET_SECONDS into whichever key applies.
# --------------------------------------------------------------------------- #
case "$NAME" in
  step4)   # laptop disconnect / reconnect. Clean job; the observer is the test.
    KIND=multinode; NODES=2; BUDGET=2400
    EXTRA=(--env LOG_INTERVAL_STEPS=1 --env EVAL_INTERVAL_STEPS=25
           --env CHECKPOINT_INTERVAL_SECONDS=30) ;;
  step5)   # 1h chaos rehearsal, all event types in order
    KIND=multinode-preempt; NODES=8; BUDGET=3300
    EXTRA=(--env LOG_INTERVAL_STEPS=1 --env EVAL_INTERVAL_STEPS=25
           --env CHECKPOINT_INTERVAL_SECONDS=120 --env SAMPLE_INTERVAL_STEPS=200
           --env MAX_EPOCHS_WITHOUT_PROGRESS=30
           --env "PREEMPT_SCHEDULE=480:3;960:L;1440:1,4;2400:0,1,2,3,4,5;3000:7") ;;
  final24h)
    KIND=multinode-preempt; NODES=8; BUDGET=82800
    EXTRA=(--env LOG_INTERVAL_STEPS=10 --env EVAL_INTERVAL_STEPS=300
           --env CHECKPOINT_INTERVAL_SECONDS=120 --env SAMPLE_INTERVAL_STEPS=1500
           --env MAX_EPOCHS_WITHOUT_PROGRESS=30) ;;
  list) : ;;
  *) echo "unknown experiment '$NAME' (step4 | step5 | final24h)" >&2; exit 2 ;;
esac

hr()  { printf '%s\n' "--------------------------------------------------------------"; }
say() { printf '\n\033[1m[%s %s] %s\033[0m\n' "$NAME" "$(date -u +%H:%M:%S)" "$*"; }
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }

s3get() { aws s3 cp "s3://$B/$1" - --region "$REGION" 2>/dev/null; }
jq_() { python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
print(d.get('$1','') if isinstance(d,dict) else '')"; }

pointer_orch() { s3get "$PTR" | jq_ orch_id; }
pointer_run()  { s3get "$PTR" | jq_ run_id; }

# A pointer is a CLAIM, not proof. The box it names may have died -- which is a
# whole experiment we ran. Always confirm against EC2.
orch_alive() {
  [ -n "${1:-}" ] || return 1
  local n
  n=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:orch,Values=$1" "Name=instance-state-name,Values=running,pending" \
    --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0)
  [ "${n:-0}" != "0" ]
}
fleet_size() {
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=spot-train-$1" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0
}
last_tick_age() {   # seconds since the supervisor last wrote — liveness, not a guess
  local t
  t=$(aws s3 ls "s3://$B/runs/$1/status/" --region "$REGION" 2>/dev/null \
      | tail -1 | awk '{print $4}' | sed 's/\.json//')
  [ -n "$t" ] || { echo ""; return; }
  echo $(( $(date -u +%s) - 10#$t/1000 ))
}
steps_of() { grep -hcE '^step [0-9]+: loss' deploy/grafana/.live/"$1"/logs/boot-node*.log 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0; }

# --------------------------------------------------------------------------- #
list_all() {
  hr; printf '  %-10s %-30s %-8s %s\n' EXPERIMENT RUN FLEET "LAST TICK"; hr
  for n in step4 step5 final24h; do
    local r f a
    r=$(s3get "experiments/$n/current.json" | jq_ run_id)
    [ -n "$r" ] || { printf '  %-10s %-30s\n' "$n" "(never run)"; continue; }
    f=$(fleet_size "$r"); a=$(last_tick_age "$r")
    printf '  %-10s %-30s %-8s %s\n' "$n" "$r" "$f" "${a:+${a}s ago}"
  done; hr
}
[ "$NAME" = "list" ] && { list_all; exit 0; }

# --------------------------------------------------------------------------- #
ensure_dash() {
  local rid="$1"
  pgrep -f "live.py $rid" >/dev/null 2>&1 || {
    nohup python3 deploy/grafana/live.py "$rid" --interval=10 --nodes="$NODES" \
      "--log=deploy/grafana/.live/$rid/logs/orchestrator.log" \
      > "deploy/grafana/live-$rid.log" 2>&1 &
    sleep 1
  }
  local from; from=$(( $(s3get "$PTR" | jq_ started_at) * 1000 - 60000 ))
  echo "  http://localhost:3001/d/dist-training/?var-run=$rid&from=$from&to=now&refresh=10s"
}

do_status() {
  local o r; o=$(pointer_orch); r=$(pointer_run)
  [ -n "$r" ] || { echo "[$NAME] never started. Run: experiments/exp.sh $NAME up"; return 1; }
  local f a s; f=$(fleet_size "$r"); a=$(last_tick_age "$r"); s=$(steps_of "$r")
  hr
  printf '  run          %s\n' "$r"
  printf '  orch         %s  (%s)\n' "$o" "$(orch_alive "$o" && echo alive || echo DEAD)"
  printf '  fleet        %s box(es)\n' "$f"
  printf '  supervisor   %s\n' "${a:+last tick ${a}s ago}"
  printf '  steps seen   %s   (local log copy; run `dash` to refresh)\n' "$s"
  s3get "runs/$r/metrics.json" >/dev/null 2>&1 \
    && printf '  \033[33mrun COMPLETE (metrics.json written)\033[0m\n'
  hr
}

do_up() {
  local o r; o=$(pointer_orch); r=$(pointer_run)
  if [ -n "$r" ] && orch_alive "$o"; then
    say "REJOINING $r (control plane $o is alive) — not launching"
    do_status; say "dashboard"; ensure_dash "$r"; return 0
  fi
  if [ -n "$r" ] && [ "$(fleet_size "$r")" != "0" ]; then
    # Fleet up, control plane gone: this is the box-failure case. Adopt it.
    say "control plane $o is GONE but $(fleet_size "$r") box(es) still train — adopting $r"
    OUT=$(python3 -m orchestrator orch up --no-attach --experiment "$KIND" --run-id "$r" \
      --env MARKET=on-demand --env "NODES=$NODES" --env "TRAIN_BUDGET_SECONDS=$BUDGET" \
      --env CHECKPOINT_KEEP=10 --env VCPU_QUOTA=64 "${EXTRA[@]}" 2>&1) || { echo "$OUT" >&2; return 1; }
    o=$(echo "$OUT" | grep -oE 'orch-[0-9]{8}-[0-9]{6}' | head -1)
  else
    say "starting a NEW $KIND run: $NODES nodes, budget ${BUDGET}s"
    OUT=$(python3 -m orchestrator orch up --no-attach --experiment "$KIND" \
      --env MARKET=on-demand --env "NODES=$NODES" --env "TRAIN_BUDGET_SECONDS=$BUDGET" \
      --env CHECKPOINT_KEEP=10 --env VCPU_QUOTA=64 "${EXTRA[@]}" 2>&1) || { echo "$OUT" >&2; return 1; }
    o=$(echo "$OUT" | grep -oE 'orch-[0-9]{8}-[0-9]{6}' | head -1)
    r=""
    for _ in $(seq 1 40); do
      r=$(s3get "orchestrators/$o/orch.json" | jq_ run_id); [ -n "$r" ] && break; sleep 10
    done
  fi
  [ -n "$r" ] || { echo "[$NAME] control plane never minted a run id" >&2; return 1; }
  # Publish the pointer so any other shell -- or agent -- finds this run.
  printf '{"experiment":"%s","orch_id":"%s","run_id":"%s","started_at":%s,"nodes":%s}' \
    "$NAME" "$o" "$r" "$(date -u +%s)" "$NODES" \
    | aws s3 cp - "s3://$B/$PTR" --region "$REGION" --only-show-errors
  say "run=$r orch=$o"; say "dashboard"; ensure_dash "$r"
  [ "$NAME" = "step4" ] && cat <<'STEP4'

  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 4 — the operator IS the test.                          │
  │                                                              │
  │  1. Wait until `status` shows steps climbing (~6 min).       │
  │  2. DISCONNECT wifi / close the lid for at least 8 minutes.  │
  │  3. Reconnect, then run:                                     │
  │         experiments/exp.sh step4 dash     # backfills        │
  │         experiments/exp.sh step4 verify                      │
  │                                                              │
  │  The run must be untouched, and the dashboard must have NO   │
  │  hole across the offline window.                             │
  └──────────────────────────────────────────────────────────────┘
STEP4
  return 0
}

do_watch() {
  local r; r=$(pointer_run); [ -n "$r" ] || { echo "not started"; return 1; }
  echo "[$NAME] watching $r — Ctrl-C detaches, the run keeps going"
  while true; do
    printf '\r  fleet=%-3s steps=%-6s tick=%-8s' \
      "$(fleet_size "$r")" "$(steps_of "$r")" "$(last_tick_age "$r")s"
    s3get "runs/$r/metrics.json" >/dev/null 2>&1 && { echo; echo "  run complete."; return 0; }
    sleep 15
  done
}

do_verify() {
  local r; r=$(pointer_run); [ -n "$r" ] || { echo "not started"; return 1; }
  say "verifying $r"
  python3 deploy/grafana/export.py "$r" --live --nodes="$NODES" \
    --src="deploy/grafana/.live/$r" >/dev/null 2>&1
  local D="deploy/grafana/data/$r"
  local pass=0 fail=0
  chk() { if [ "$1" = 1 ]; then ok "$2"; pass=$((pass+1)); else bad "$2"; fail=$((fail+1)); fi; }

  chk "$([ "$(steps_of "$r")" -gt 0 ] && echo 1 || echo 0)" "training produced steps"
  chk "$([ -s "$D/world.csv" ] && echo 1 || echo 0)"        "world.csv has rows"

  if [ "$NAME" = "step4" ]; then
    # THE test. The supervisor wrote a status object every tick throughout; the
    # laptop simply fetched them late. So after `dash` backfills, there must be
    # NO gap -- and the control-plane-down detector must agree by drawing
    # nothing. A band here means backfill is broken; it is the exact inverse of
    # the box-failure test, which is why the two check each other.
    local bands; bands=$(python3 -c "
import json
try: print(len(json.load(open('$D/control_plane_down.json'))))
except Exception: print(-1)")
    chk "$([ "$bands" = "0" ] && echo 1 || echo 0)" \
        "NO control-plane-down band (supervisor never stopped writing) [got $bands]"
    local gap; gap=$(python3 -c "
import csv
ts=[int(r['time'])/1000 for r in csv.DictReader(open('$D/timeseries.csv'))]
g=[b-a for a,b in zip(ts,ts[1:]) if b-a>120]
print(f'{max(g):.0f}' if g else '0')" 2>/dev/null || echo -1)
    chk "$([ "${gap%%.*}" -lt 120 ] 2>/dev/null && echo 1 || echo 0)" \
        "no >120s hole in the timeseries across the offline window [max ${gap}s]"
    chk "$(grep -qc 'supervisor_up' "$D/timeseries.csv" 2>/dev/null && echo 1 || echo 0)" \
        "supervisor_up column present"
  fi
  echo
  [ "$fail" = 0 ] && printf '  \033[32mALL %s PASSED\033[0m\n\n' "$pass" \
                  || printf '  \033[31m%s passed, %s FAILED\033[0m\n\n' "$pass" "$fail"
  [ "$fail" = 0 ]
}

do_down() {
  local o r; o=$(pointer_orch); r=$(pointer_run)
  say "tearing down $r"
  pkill -f "live.py $r" 2>/dev/null
  [ -n "$o" ] && python3 -m orchestrator orch down --id "$o" --all --yes 2>&1 | tail -3
  . scripts/fleetctl.sh && reap_ours
  aws s3 rm "s3://$B/$PTR" --region "$REGION" --only-show-errors 2>/dev/null
  experiments/validate_teardown.sh
}

case "$CMD" in
  up)     do_up ;;
  status) do_status ;;
  watch)  do_watch ;;
  dash)   r=$(pointer_run); [ -n "$r" ] && { say "dashboard"; ensure_dash "$r"; } || echo "not started" ;;
  verify) do_verify ;;
  down)   do_down ;;
  *) echo "unknown command '$CMD' (up|status|watch|dash|verify|down)" >&2; exit 2 ;;
esac
