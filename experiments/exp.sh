#!/usr/bin/env bash
# One entry point for every paid experiment.
#
#   experiments/exp.sh                 what is live, what it costs, what to type next
#   experiments/exp.sh up <name>       start, or REJOIN if already running
#   experiments/exp.sh watch           follow: events as they happen (Ctrl-C is safe)
#   experiments/exp.sh status          one-shot snapshot
#   experiments/exp.sh verify          PASS/FAIL checks
#   experiments/exp.sh down            verified teardown  (no name needed)
#   experiments/exp.sh list            every experiment
#
# The experiment NAME is only needed for `up`. Everything else resolves to
# whichever experiment is live, because when you are worried about a running
# job you should not have to remember what you called it. `exp.sh <name> <cmd>`
# still works, for scripts and agents that want to be explicit.
#
# WHY THERE IS NO LOCAL STATE
#
# The operator and an agent drive the same run from different shells at
# different times. The pointer lives in S3 at experiments/<name>/current.json,
# so a fresh clone, a second terminal, or an agent that has never seen this run
# can all act on it immediately -- the same S3-is-the-control-plane property the
# system already runs on.
#
# `up` is idempotent because a second launch is the expensive mistake: it would
# double an 8-node fleet. It reads the pointer, then CONFIRMS against EC2 that
# the control plane is still alive -- a pointer is a claim, not proof, and we
# have run the experiment where the box it named had died.
#
# Ctrl-C is safe everywhere. Nothing here owns the run.
set -uo pipefail
cd "$(dirname "$0")/.."

# The RECIPE is not optional. Sourcing only .env leaves DATASET/N_LAYER/N_EMBD/
# BLOCK_SIZE/GLOBAL_BATCH_SIZE at their defaults, which are the shakespeare_char
# toy settings -- step4 ran a 10.67M char model instead of GPT-2 124M on
# OpenWebText and nothing said so. Silent-wrong-config is the failure mode this
# repo keeps hitting; `up` now prints the resolved model before spending.
set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
REGION="${TRAIN_REGION:-us-east-1}"
B="$SPOT_TRAIN_BUCKET"
KNOWN="step4 step5 step6 final24h"

# --------------------------------------------------------------------------- #
# Arg parsing: verb-first for humans, noun-first for scripts.
# --------------------------------------------------------------------------- #
A1="${1:-}"; A2="${2:-}"
case " $KNOWN " in
  *" $A1 "*) NAME="$A1"; CMD="${A2:-status}" ;;        # exp.sh step5 up
  *)         CMD="${A1:-overview}"; NAME="${A2:-}" ;;  # exp.sh up step5 | exp.sh status
esac

hr()  { printf '%s\n' "----------------------------------------------------------------"; }
say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date -u +%H:%M:%S)" "$*"; }
ok()  { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
tip() { printf '\n  \033[36mnext:\033[0m %s\n\n' "$1"; }

s3get() { aws s3 cp "s3://$B/$1" - --region "$REGION" 2>/dev/null; }
jq_() { python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
print(d.get('$1','') if isinstance(d,dict) else '')"; }

fleet_size() {
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=spot-train-$1" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0
}
orch_alive() {
  [ -n "${1:-}" ] || return 1
  local n; n=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:orch,Values=$1" "Name=instance-state-name,Values=running,pending" \
    --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0)
  [ "${n:-0}" != "0" ]
}
last_tick_age() {
  local t; t=$(aws s3 ls "s3://$B/runs/$1/status/" --region "$REGION" 2>/dev/null \
      | tail -1 | awk '{print $4}' | sed 's/\.json//')
  [ -n "$t" ] || { echo ""; return; }
  echo $(( $(date -u +%s) - 10#$t/1000 ))
}
# Durable step from the SUPERVISOR's own status doc, not from a local log copy.
# The local copy only advances while live.py is running, so a healthy run with
# no dashboard loop would otherwise report 0 steps forever -- exactly the
# confusion that made a working run look dead.
durable_step() { s3get "runs/$1/status.json" | jq_ ckpt_step; }
run_done()     { s3get "runs/$1/metrics.json" >/dev/null 2>&1; }

# Cost so far. export.py derives it from the observed instance ledger in the
# status ticks, so it is right mid-run, not just at the end.
cost_of() {
  local rid="$1" n="${2:-2}"
  python3 deploy/grafana/export.py "$rid" --live --nodes="$n" \
    --src="deploy/grafana/.live/$rid" >/dev/null 2>&1
  python3 -c "
import csv,sys
try:
    rows=[r for r in csv.DictReader(open('deploy/grafana/data/$rid/timeseries.csv')) if r.get('usd')]
    print(f\"\$%.2f\" % float(rows[-1]['usd']) if rows else '')
except Exception: print('')" 2>/dev/null
}

pointer_of() { s3get "experiments/$1/current.json"; }

# Resolve the experiment when the operator did not name one: prefer a live
# control plane, then a live fleet, then the most recently started.
resolve_name() {
  [ -n "$NAME" ] && return 0
  local best="" best_t=0
  for n in $KNOWN; do
    local p o r t; p=$(pointer_of "$n"); [ -n "$p" ] || continue
    o=$(echo "$p" | jq_ orch_id); r=$(echo "$p" | jq_ run_id); t=$(echo "$p" | jq_ started_at)
    if orch_alive "$o" || [ "$(fleet_size "$r")" != "0" ]; then NAME="$n"; return 0; fi
    [ "${t:-0}" -gt "$best_t" ] 2>/dev/null && { best="$n"; best_t="$t"; }
  done
  NAME="$best"
  [ -n "$NAME" ]
}

# --------------------------------------------------------------------------- #
# Experiment definitions. Add a case; everything else is generic.
#
# BUDGET is passed as TRAIN_BUDGET_SECONDS and NEVER as an experiment's own key:
# `multinode` reads BASELINE_SECONDS, `multinode-preempt` reads
# TRAIN_TOTAL_SECONDS, and naming the wrong one is SILENT -- the run takes the
# 300s default, trains ~200s and finishes before the test starts. That cost two
# attempts. orch up translates TRAIN_BUDGET_SECONDS into whichever key applies.
# --------------------------------------------------------------------------- #
load_def() {
  case "$1" in
    step4)  KIND=multinode; NODES=2; BUDGET=2400
      EXTRA=(--env LOG_INTERVAL_STEPS=1 --env EVAL_INTERVAL_STEPS=25
             --env CHECKPOINT_INTERVAL_SECONDS=30) ;;
    step5)  # compressed replica of the 24h run: every event type, in order, in ~55 min
      KIND=multinode-preempt; NODES=8; BUDGET=3600
      EXTRA=(--env LOG_INTERVAL_STEPS=1 --env EVAL_INTERVAL_STEPS=25
             --env CHECKPOINT_INTERVAL_SECONDS=120 --env SAMPLE_INTERVAL_STEPS=200
             --env MAX_EPOCHS_WITHOUT_PROGRESS=30
             # METRICS_OVERHEAD is WALL-CLOCK SLACK, not schedule. The supervisor
             # waits for metrics.json (the trainer's done-signal) and gives up at
             # budget + overhead, measured on the WALL clock -- but the budget
             # counts TRAINING seconds only. Everything else spends wall without
             # spending budget: boot, the 17GB dataset pull, and every recovery.
             #
             # This killed step5. Default overhead is 1200s, so the deadline was
             # 3600+1200 = 80 min against a run that honestly needed ~2h. It was
             # terminated at step 984 with 1749s of budget still owed and no
             # metrics.json. The deadline was shorter than the run.
             #
             # Size it as a generous UPPER BOUND, never an estimate. Too small
             # kills a healthy fleet and loses the whole run; too large only
             # means noticing a genuinely wedged one later, which config.py's own
             # comment calls out as the far cheaper mistake.
             # TWO deadlines, and BOTH are sized from the TRAINING budget while
             # measured against the WALL clock. Fixing only one just changes which
             # guillotine lands first:
             #   watchdog     = budget + METRICS_OVERHEAD          (supervisor gives up)
             #   per-box kill = budget + INSTANCE_LIFETIME_SLACK   (box poweroffs itself)
             # step5 needs ~2h wall for 1h of training, so the stock 1h slack would
             # have self-terminated the fleet mid-run. Keep the per-box timer BEHIND
             # the watchdog, so the smarter guard (which is actually watching for
             # metrics.json) is the one that decides.
             --env METRICS_OVERHEAD=7200 --env INSTANCE_LIFETIME_SLACK_SECONDS=14400
             # Mass loss is the FINALE, deliberately. An earlier version fired one more
             # kill at t+3000, only 600s after it -- but six replacements launch
             # serially (each blocking in wait_running) and then each pays boot plus
             # the 118s dataset pull, so a full 2->8 regrow takes 5-8 min. That kill
             # landed mid-staircase and took out a survivor, turning the headline
             # event into a sawtooth. Now it gets a ~21 min tail: full regrow, then
             # sustained training back at world 8.
             --env "PREEMPT_SCHEDULE=480:3;960:L;1440:1,4;2400:0,1,2,3,4,5") ;;
    step6)  # SCHEDULE PERSISTENCE across a control-plane BOX failure.
      #
      # _save_schedule / _restore_schedule has 8 unit tests and ZERO cloud runs.
      # It is what stops a restarted supervisor replaying PREEMPT_SCHEDULE from
      # zero -- at hour 8 of the 24h run a replay would re-fire the mass loss of
      # 6 of 8, unscheduled, with no live knob to stop it (the schedule is baked
      # into user-data).
      #
      # Why this needs its own run rather than riding step5: the assertion is
      # about what happens BETWEEN two kills, so the control plane must die while
      # entries are still pending. step5 has no supervisor event at all.
      #
      # Two kills 8 min apart on 2 nodes. The operator terminates the control
      # plane between them; `up` then takes the adopt branch. Assertions:
      #   - the SPENT kill does not re-fire  (exactly one `terminated node 0`)
      #   - the PENDING kill still fires, on the ORIGINAL train-start clock, not
      #     restarted from the adoption moment
      # The second is the one that catches a wrong fix: restoring `fired` while
      # re-arming the clock from `now` would pass a naive check and still shift
      # every remaining kill by the outage duration.
      KIND=multinode-preempt; NODES=2; BUDGET=1800
      EXTRA=(--env LOG_INTERVAL_STEPS=1 --env EVAL_INTERVAL_STEPS=0
             --env CHECKPOINT_INTERVAL_SECONDS=30
             --env MAX_EPOCHS_WITHOUT_PROGRESS=30
             # The operator deliberately kills the control plane mid-run here, so
             # wall clock includes a full outage plus a second control-plane boot.
             # 1h of slack on a 30 min budget.
             --env METRICS_OVERHEAD=3600 --env INSTANCE_LIFETIME_SLACK_SECONDS=10800
             --env "PREEMPT_SCHEDULE=300:0;780:1") ;;
    final24h)
      # 23h of TRAINING time; wall lands ~24-25h once recovery downtime is added.
      # LR schedule stays at the recipe's 600000 (Karpathy's) -- deliberately NOT
      # tuned to the horizon, because the achievable step count is uncertain to
      # within 2x (step5 ran 1958ms/step late-run vs a 4013ms early baseline), and
      # tuning to a horizon you cannot predict is worse than not tuning. Keeping
      # it also makes step counts directly comparable to nanoGPT's published OWT
      # curve, since the 480x1024 global batch already matches exactly.
      KIND=multinode-preempt; NODES=8; BUDGET=82800
      EXTRA=(--env LOG_INTERVAL_STEPS=10 --env EVAL_INTERVAL_STEPS=300
             --env CHECKPOINT_INTERVAL_SECONDS=120 --env SAMPLE_INTERVAL_STEPS=1500
             --env MAX_EPOCHS_WITHOUT_PROGRESS=30
             # 6h of slack -> a 29h deadline against a ~25-26h expected wall
             # clock (23h training + ~23 recoveries at ~300s + boot). See the
             # step5 note above for why this is an upper bound, not an estimate.
             --env METRICS_OVERHEAD=21600 --env INSTANCE_LIFETIME_SLACK_SECONDS=28800
             # ~24 kill events over 23h, spaced ~1h. MEASURED in step5: recovery
             # to full world is ~213-314s and FLAT in the number of nodes lost --
             # losing 6 recovered in 259s, the same as losing 1, because
             # replacements boot in parallel. So 3600s spacing is ~12x recovery,
             # with ample room even if a replacement is slow to get capacity.
             #
             # TWO mass losses, deliberately. Hour 2 is the cheap abort point: it
             # proves 6-of-8 at production duration with only ~$16 sunk, instead
             # of discovering a problem at hour 16 with ~$130 gone. Hour 16 keeps
             # the narrative event, with long steady-state history either side.
             --env "PREEMPT_SCHEDULE=\
1800:3;5400:L;7200:0,1,2,3,4,5;10800:6;14400:L;18000:2,7;21600:1;25200:4;\
28800:L;32400:0,3;36000:5;39600:6;43200:L;46800:1,2;50400:7;54000:4;\
57600:0,1,2,3,4,5;61200:6;64800:L;68400:3,5;72000:2;75600:7;79200:1") ;;
    *) echo "unknown experiment '$1' ($KNOWN)" >&2; return 2 ;;
  esac
}

# --------------------------------------------------------------------------- #
ensure_grafana() {
  curl -sf -o /dev/null --max-time 3 http://localhost:3001/api/health 2>/dev/null && return 0
  echo "  starting Grafana..."
  (cd deploy/grafana && docker compose up -d >/dev/null 2>&1) || true
  for _ in $(seq 1 20); do
    curl -sf -o /dev/null --max-time 2 http://localhost:3001/api/health 2>/dev/null && return 0
    sleep 2
  done
  echo "  WARNING: Grafana not reachable on :3001 — is Docker running?" >&2
}
ensure_dash() {
  local rid="$1"
  ensure_grafana
  pgrep -f "live.py $rid" >/dev/null 2>&1 || {
    nohup python3 deploy/grafana/live.py "$rid" --interval=10 --nodes="$NODES" \
      "--log=deploy/grafana/.live/$rid/logs/orchestrator.log" \
      > "deploy/grafana/live-$rid.log" 2>&1 &
    sleep 1
  }
  local from; from=$(( $(pointer_of "$NAME" | jq_ started_at) * 1000 - 60000 ))
  echo "  http://localhost:3001/d/dist-training/?var-run=$rid&from=$from&to=now&refresh=10s"
}

# --------------------------------------------------------------------------- #
do_overview() {
  hr; printf '  %-9s %-28s %-6s %-9s %s\n' EXPERIMENT RUN FLEET COST "SUPERVISOR"; hr
  local live_name="" live_run=""
  for n in $KNOWN; do
    local p r f a c; p=$(pointer_of "$n")
    [ -n "$p" ] || { printf '  %-9s %s\n' "$n" "(never run)"; continue; }
    r=$(echo "$p" | jq_ run_id); f=$(fleet_size "$r"); a=$(last_tick_age "$r")
    c=$(cost_of "$r" "$(echo "$p" | jq_ nodes)")
    local ended; ended=$(echo "$p" | jq_ ended_at)
    printf '  %-9s %-28s %-6s %-9s %s\n' "$n" "$r" "$f" "${c:--}" \
      "$([ -n "$ended" ] && echo "ended" || echo "${a:+${a}s ago}")"
    [ "$f" != "0" ] && { live_name="$n"; live_run="$r"; }
  done
  hr
  if [ -n "$live_name" ]; then
    printf '\n  \033[1m%s is LIVE\033[0m (%s boxes)\n' "$live_name" "$(fleet_size "$live_run")"
    tip "experiments/exp.sh watch     ·  stop it: experiments/exp.sh down"
  else
    tip "experiments/exp.sh up step5     (nothing is running)"
  fi
}

do_status() {
  local p; p=$(pointer_of "$NAME")
  [ -n "$p" ] || { echo "[$NAME] never started."; tip "experiments/exp.sh up $NAME"; return 1; }
  local o r nodes; o=$(echo "$p" | jq_ orch_id); r=$(echo "$p" | jq_ run_id); nodes=$(echo "$p" | jq_ nodes)
  local f a d c; f=$(fleet_size "$r"); a=$(last_tick_age "$r"); d=$(durable_step "$r"); c=$(cost_of "$r" "$nodes")
  hr
  printf '  experiment   %s\n'         "$NAME"
  printf '  run          %s\n'         "$r"
  printf '  control      %s  (%s)\n'   "$o" "$(orch_alive "$o" && echo alive || echo DEAD)"
  printf '  fleet        %s box(es)\n' "$f"
  printf '  durable step %s\n'         "${d:--}"
  printf '  cost so far  %s\n'         "${c:--}"
  printf '  supervisor   %s\n'         "${a:+last tick ${a}s ago}"
  run_done "$r" && printf '  \033[33mrun COMPLETE (metrics.json written)\033[0m\n'
  hr
  if run_done "$r"; then       tip "experiments/exp.sh verify   then   experiments/exp.sh down"
  elif [ "$f" = "0" ]; then    tip "experiments/exp.sh up $NAME    (fleet is gone)"
  else                         tip "experiments/exp.sh watch     ·  stop it: experiments/exp.sh down"; fi
}

do_up() {
  [ -n "$NAME" ] || { echo "which experiment? ($KNOWN)" >&2; return 2; }
  load_def "$NAME" || return $?
  local p o r; p=$(pointer_of "$NAME"); o=$(echo "$p" | jq_ orch_id); r=$(echo "$p" | jq_ run_id)
  if [ -n "$r" ] && orch_alive "$o"; then
    say "REJOINING $r — control plane $o is alive, launching nothing"
    do_status; say "dashboard"; ensure_dash "$r"; return 0
  fi
  if [ -n "$r" ] && [ "$(fleet_size "$r")" != "0" ] && ! run_done "$r"; then
    say "control plane GONE, $(fleet_size "$r") box(es) still training — adopting $r"
    OUT=$(python3 -m orchestrator orch up --no-attach --experiment "$KIND" --run-id "$r" \
      --env MARKET=on-demand --env "NODES=$NODES" --env "TRAIN_BUDGET_SECONDS=$BUDGET" \
      --env CHECKPOINT_KEEP=10 --env VCPU_QUOTA=64 "${EXTRA[@]}" 2>&1) \
      || { echo "$OUT" >&2; return 1; }
    o=$(echo "$OUT" | grep -oE 'orch-[0-9]{8}-[0-9]{6}' | head -1)
  else
    say "starting $NAME: $KIND, $NODES nodes, budget ${BUDGET}s"
    printf '  model   %sL/%sH/%sd  ctx %s  global batch %s\n' \
      "${N_LAYER:-?}" "${N_HEAD:-?}" "${N_EMBD:-?}" "${BLOCK_SIZE:-?}" "${GLOBAL_BATCH_SIZE:-?}"
    printf '  dataset %s\n' "${DATASET:-?}"
    if [ "${DATASET:-}" != "openwebtext" ]; then
      echo "  REFUSING: DATASET=${DATASET:-unset}, expected openwebtext." >&2
      echo "  A paid chaos run on the wrong corpus produces numbers that look" >&2
      echo "  fine and compare to nothing. Check recipes/gpt2-owt.env." >&2
      return 1
    fi
    OUT=$(python3 -m orchestrator orch up --no-attach --experiment "$KIND" \
      --env MARKET=on-demand --env "NODES=$NODES" --env "TRAIN_BUDGET_SECONDS=$BUDGET" \
      --env CHECKPOINT_KEEP=10 --env VCPU_QUOTA=64 "${EXTRA[@]}" 2>&1) \
      || { echo "$OUT" >&2; return 1; }
    o=$(echo "$OUT" | grep -oE 'orch-[0-9]{8}-[0-9]{6}' | head -1)
    r=""; for _ in $(seq 1 40); do
      r=$(s3get "orchestrators/$o/orch.json" | jq_ run_id); [ -n "$r" ] && break; sleep 10
    done
  fi
  [ -n "$r" ] || { echo "control plane never minted a run id" >&2; return 1; }
  printf '{"experiment":"%s","orch_id":"%s","run_id":"%s","started_at":%s,"nodes":%s}' \
    "$NAME" "$o" "$r" "$(date -u +%s)" "$NODES" \
    | aws s3 cp - "s3://$B/experiments/$NAME/current.json" --region "$REGION" --only-show-errors
  say "run=$r  orch=$o"; say "dashboard"; ensure_dash "$r"
  tip "experiments/exp.sh watch     ·  stop it: experiments/exp.sh down"
}

# Follow: counters on one line, and EVENTS on their own lines as they land, so
# it reads like a record of what the system decided rather than three numbers.
do_watch() {
  local p r nodes; p=$(pointer_of "$NAME"); r=$(echo "$p" | jq_ run_id)
  [ -n "$r" ] || { echo "[$NAME] not started"; return 1; }
  nodes=$(echo "$p" | jq_ nodes)
  echo "[$NAME] watching $r — Ctrl-C detaches, the run keeps going"
  local seen; seen=$(mktemp); : > "$seen"
  while true; do
    s3get "runs/$r/logs/orchestrator.log" 2>/dev/null \
      | grep -oE "published epoch [0-9]+: members \[[^]]*\] master=node[0-9]+|terminated node [0-9]+|re-adopted epoch [0-9]+|adopted [0-9]+ running box|whole-group restart" \
      | sort -u > /tmp/ev.$$ 2>/dev/null || true
    if [ -s /tmp/ev.$$ ]; then
      comm -13 "$seen" /tmp/ev.$$ 2>/dev/null | while read -r line; do
        [ -n "$line" ] && printf '\r  \033[35m%s  %s\033[0m\n' "$(date -u +%H:%M:%S)" "$line"
      done
      mv /tmp/ev.$$ "$seen"
    fi
    printf '\r  fleet=%-3s durable=%-7s cost=%-8s tick=%-6s' \
      "$(fleet_size "$r")" "$(durable_step "$r")" "$(cost_of "$r" "$nodes")" "$(last_tick_age "$r")s"
    run_done "$r" && { echo; echo "  run complete."; rm -f "$seen"; tip "experiments/exp.sh verify"; return 0; }
    sleep 15
  done
}

do_verify() {
  local p r nodes; p=$(pointer_of "$NAME"); r=$(echo "$p" | jq_ run_id)
  [ -n "$r" ] || { echo "[$NAME] not started"; return 1; }
  nodes=$(echo "$p" | jq_ nodes)
  say "verifying $NAME / $r"
  python3 deploy/grafana/export.py "$r" --live --nodes="$nodes" \
    --src="deploy/grafana/.live/$r" >/dev/null 2>&1
  local D="deploy/grafana/data/$r" pass=0 fail=0
  chk() { if [ "$1" = 1 ]; then ok "$2"; pass=$((pass+1)); else bad "$2"; fail=$((fail+1)); fi; }
  local bands; bands=$(python3 -c "
import json
try: print(len(json.load(open('$D/control_plane_down.json'))))
except Exception: print(-1)")

  chk "$([ -s "$D/world.csv" ] && echo 1 || echo 0)" "world.csv has rows"
  chk "$([ "$(durable_step "$r")" -gt 0 ] 2>/dev/null && echo 1 || echo 0)" "checkpoints advanced"

  case "$NAME" in
    step4)
      # The INVERSE of the box-failure test: the supervisor never stopped
      # writing, the laptop just fetched late, so backfill must leave no trace.
      chk "$([ "$bands" = "0" ] && echo 1 || echo 0)" \
          "NO control-plane-down band — backfill closed the offline window [got $bands]" ;;
    step6)
      # Read the supervisor's own log: it records every termination with a
      # timestamp, and _restore_schedule announces what it re-armed.
      local LOG kills0 kills1 readopt
      LOG=$(s3get "runs/$r/logs/orchestrator.log")
      kills0=$(printf '%s' "$LOG" | grep -c "terminated node 0" || true)
      kills1=$(printf '%s' "$LOG" | grep -c "terminated node 1" || true)
      readopt=$(printf '%s' "$LOG" | grep -o "re-armed kill schedule:[^(]*" | head -1)

      chk "$([ "${kills0:-0}" -eq 1 ] 2>/dev/null && echo 1 || echo 0)" \
          "spent kill did NOT replay — exactly one 'terminated node 0' [got ${kills0:-?}]"
      chk "$([ "${kills1:-0}" -eq 1 ] 2>/dev/null && echo 1 || echo 0)" \
          "pending kill still fired after adoption [got ${kills1:-?}]"
      chk "$([ -n "$readopt" ] && echo 1 || echo 0)" \
          "supervisor re-armed rather than restarted: ${readopt:-<absent>}"

      # THE assertion that catches a wrong fix. The second kill is scheduled at
      # t+780s on the ORIGINAL train-start clock. If the restarted supervisor
      # re-armed its clock from the adoption moment instead of from the persisted
      # wall start, this kill lands late by roughly the outage duration -- and a
      # check that only counted kills would call that a pass.
      python3 - "$r" <<'PYEOF'
import json, re, subprocess, sys, os
rid = sys.argv[1]; b = os.environ["SPOT_TRAIN_BUCKET"]
def s3(k):
    return subprocess.run(["aws","s3","cp",f"s3://{b}/{k}","-","--region","us-east-1"],
                          capture_output=True, text=True).stdout
sched = json.loads(s3(f"runs/{rid}/schedule.json") or "{}")
ts = float(sched.get("train_start_wall") or 0)
log = s3(f"runs/{rid}/logs/orchestrator.log")
m = re.search(r"\[(\d+):(\d+):(\d+)\] \[supervisor\] terminated node 1", log)
if not (ts and m):
    print("  \033[31mFAIL\033[0m  cannot time the pending kill (missing schedule.json or log line)")
    sys.exit()
h, mi, se = (int(x) for x in m.groups())
# Log stamps are UTC wall-clock-of-day; fold onto the run's own day.
import datetime as dt
d0 = dt.datetime.fromtimestamp(ts, dt.UTC)
fired = d0.replace(hour=h, minute=mi, second=se)
if fired < d0: fired += dt.timedelta(days=1)
offset = (fired - d0).total_seconds()
drift = offset - 780
verdict = "\033[32mPASS\033[0m" if abs(drift) <= 90 else "\033[31mFAIL\033[0m"
print(f"  {verdict}  pending kill fired at t+{offset:.0f}s (scheduled 780s, drift {drift:+.0f}s)")
if abs(drift) > 90:
    print("        -> clock was re-armed from the RESTART, not from the persisted")
    print("           train start; every remaining kill shifts by the outage.")
PYEOF
      ;;
    step5|final24h)
      local kills epochs
      kills=$(s3get "runs/$r/logs/orchestrator.log" | grep -c "terminated node" || echo 0)
      epochs=$(s3get "runs/$r/logs/orchestrator.log" | grep -c "published epoch" || echo 0)
      chk "$([ "$kills" -gt 0 ] && echo 1 || echo 0)"  "chaos schedule fired [$kills kills]"
      chk "$([ "$epochs" -gt 2 ] && echo 1 || echo 0)" "world re-formed after losses [$epochs epochs]"
      chk "$(s3get "runs/$r/logs/orchestrator.log" | grep -qc "whole-group restart" && echo 0 || echo 1)" \
          "ZERO whole-group restarts" ;;
  esac
  echo
  [ "$fail" = 0 ] && printf '  \033[32mALL %s PASSED\033[0m\n' "$pass" \
                  || printf '  \033[31m%s passed, %s FAILED\033[0m\n' "$pass" "$fail"
  tip "experiments/exp.sh down"
  [ "$fail" = 0 ]
}

do_dash() {
  local p r; p=$(pointer_of "$NAME"); r=$(echo "$p" | jq_ run_id)
  [ -n "$r" ] || { echo "[$NAME] not started"; return 1; }
  load_def "$NAME" >/dev/null 2>&1
  say "dashboard"; ensure_dash "$r"
}

do_down() {
  local p o r; p=$(pointer_of "$NAME"); o=$(echo "$p" | jq_ orch_id); r=$(echo "$p" | jq_ run_id)
  say "tearing down $NAME / ${r:-?}"
  [ -n "$r" ] && pkill -f "live.py $r" 2>/dev/null
  [ -n "$o" ] && python3 -m orchestrator orch down --id "$o" --all --yes 2>&1 | tail -3
  . scripts/fleetctl.sh && reap_ours
  # Mark it ended rather than deleting the pointer. Deleting made the run
  # unfindable the moment it was torn down, so `verify` -- which reads S3 and
  # .live, both of which survive -- had nothing to resolve. You almost always
  # want to verify AFTER stopping the spend, not before.
  if [ -n "$NAME" ] && [ -n "$p" ]; then
    echo "$p" | python3 -c "
import json,sys,time
d=json.load(sys.stdin); d['ended_at']=int(time.time()); print(json.dumps(d))" \
      | aws s3 cp - "s3://$B/experiments/$NAME/current.json" --region "$REGION" --only-show-errors
  fi
  # Independent audit. Deliberately a separate script: its whole value is that
  # it asks EC2 directly and is unaffected by anything the teardown printed.
  experiments/validate_teardown.sh
}

case "$CMD" in
  overview) do_overview ;;
  list)     do_overview ;;
  up)       do_up ;;
  status)   resolve_name && do_status || { echo "nothing has been run yet"; tip "experiments/exp.sh up step5"; } ;;
  watch)    resolve_name && do_watch  || echo "nothing is running" ;;
  verify)   resolve_name && do_verify || echo "nothing to verify" ;;
  dash)     resolve_name && do_dash   || echo "nothing is running" ;;
  down)     resolve_name && do_down   || { echo "no experiment pointer — sweeping anyway"; . scripts/fleetctl.sh && reap_ours; experiments/validate_teardown.sh; } ;;
  *) echo "unknown command '$CMD' (up|status|watch|verify|dash|down|list)" >&2; exit 2 ;;
esac
