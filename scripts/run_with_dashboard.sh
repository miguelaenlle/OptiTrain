#!/usr/bin/env bash
# ONE COMMAND: launch an experiment and stream it to Grafana.
#
#   scripts/run_with_dashboard.sh <driver.py|driver.sh> [NODES]
#
# The live loop is a separate PROCESS but must not be a separate STEP -- having
# to remember it is how a run ends up with no dashboard, which already happened
# once. This starts the stack, starts the driver, waits for it to print its
# run_id, attaches the loop to that id, opens the browser on it, and tears
# everything down together.
#
# A .sh driver is run with bash, a .py driver with python3. That matters because
# an experiment's knobs (MARKET, NODES, TRAIN_TOTAL_SECONDS, FIRST_KILL, the
# kill schedule) live in the experiment's own run.sh next to the driver -- so
# pointing this at the .py alone silently runs it with default budgets.
set -uo pipefail
cd "$(dirname "$0")/.."
DRIVER="${1:?usage: run_with_dashboard.sh <driver.py|driver.sh> [NODES]}"
NODES="${2:-${NODES:-2}}"
LOG="$(dirname "$DRIVER")/run.log"
case "$DRIVER" in
  *.sh) RUNNER=(bash "$DRIVER") ;;
  *)    RUNNER=(python3 "$DRIVER") ;;
esac

set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
. scripts/fleetctl.sh

cleanup() {
  [ -n "${LIVE_PID:-}" ] && kill "$LIVE_PID" 2>/dev/null
  echo "[wd] reaping fleet"; reap_ours
}
trap cleanup EXIT INT TERM

# Bring the stack up here rather than assuming it. "one command" has to include
# the container that serves the CSVs -- forgetting it produces a dashboard that
# looks broken (every panel "No data") for a reason that has nothing to do with
# the run. Idempotent: already-running containers are left alone.
if ! docker compose -f deploy/grafana/docker-compose.yml ps --status running 2>/dev/null \
     | grep -q grafana; then
  echo "[wd] starting grafana stack"
  docker compose -f deploy/grafana/docker-compose.yml up -d >/dev/null 2>&1
fi
for _ in $(seq 1 40); do
  curl -sf http://localhost:3001/api/health >/dev/null 2>&1 && break
  sleep 1
done

echo "[wd] driver: ${RUNNER[*]}  nodes=$NODES  log=$LOG"
"${RUNNER[@]}" > "$LOG" 2>&1 &
DRIVER_PID=$!

# The run_id only exists once the driver prints it; poll rather than guess.
RUN_ID=""
for _ in $(seq 1 60); do
  RUN_ID=$(grep -oE "run_id=[a-z0-9-]+" "$LOG" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$RUN_ID" ] && break
  kill -0 "$DRIVER_PID" 2>/dev/null || break
  sleep 2
done
if [ -z "$RUN_ID" ]; then
  echo "[wd] driver never reported a run_id; see $LOG"; wait "$DRIVER_PID"; exit 1
fi

echo "[wd] run_id=$RUN_ID"
python3 deploy/grafana/live.py "$RUN_ID" --interval=10 "--nodes=$NODES" "--log=$LOG" \
  > deploy/grafana/live.log 2>&1 &
LIVE_PID=$!

# Open the dashboard with the run and the window in the URL. The provisioned
# JSON carries both, but Grafana only applies them to a tab that has no from/to
# of its own -- a tab left open from an earlier session keeps ITS range and run,
# so the new run renders as an empty window on somebody else's data. URL
# parameters win over everything, which is why they are spelled out here.
# from = launch (the run id's timestamp suffix), so the boot is in frame.
LAUNCH_MS="$(( $(echo "$RUN_ID" | grep -oE '[0-9]{10}$') * 1000 - 30000 ))"
URL="http://localhost:3001/d/dist-training/?var-run=${RUN_ID}&from=${LAUNCH_MS}&to=now&refresh=10s"
echo "[wd] dashboard: $URL"
command -v open >/dev/null 2>&1 && open "$URL" >/dev/null 2>&1

wait "$DRIVER_PID"
echo "[wd] driver finished rc=$?; final dashboard refresh"
sleep 12
