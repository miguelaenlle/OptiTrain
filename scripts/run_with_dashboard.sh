#!/usr/bin/env bash
# ONE COMMAND: launch an experiment and stream it to Grafana.
#
#   scripts/run_with_dashboard.sh <driver.py> [NODES]
#
# The live loop is a separate PROCESS but must not be a separate STEP -- having
# to remember it is how a run ends up with no dashboard, which already happened
# once. This starts the driver, waits for it to print its run_id, attaches the
# loop to that id, and tears both down together.
set -uo pipefail
cd "$(dirname "$0")/.."
DRIVER="${1:?usage: run_with_dashboard.sh <driver.py> [NODES]}"
NODES="${2:-${NODES:-2}}"
LOG="$(dirname "$DRIVER")/run.log"

set -a && . ./.env && . ./recipes/gpt2-owt.env && set +a
. scripts/fleetctl.sh

cleanup() {
  [ -n "${LIVE_PID:-}" ] && kill "$LIVE_PID" 2>/dev/null
  echo "[wd] reaping fleet"; reap_ours
}
trap cleanup EXIT INT TERM

echo "[wd] driver: $DRIVER  nodes=$NODES  log=$LOG"
python3 "$DRIVER" > "$LOG" 2>&1 &
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
echo "[wd] dashboard: http://localhost:3001  (Run = $RUN_ID)"
wait "$DRIVER_PID"
echo "[wd] driver finished rc=$?; final dashboard refresh"
sleep 12
