#!/usr/bin/env bash
# P3.3 — the experiments that need REAL machines, not k3d.
#
#   c  cross-node serving          workers on different EC2 instances actually serve
#   d  node kill under load        terminate a whole instance mid-traffic
#   f  BLACK-HOLE failure mode     the one k3d and localhost cannot produce
#
# (f) is why this runs in the cloud at all. Killing a local process yields an
# instant ECONNREFUSED; a terminating EC2 instance stops answering and its
# packets are silently DROPPED. The router then has no error to react to -- it
# waits. That is exactly what ROUTER_CONNECT_TIMEOUT_SECONDS and
# ROUTER_UPSTREAM_KEEPALIVE=false exist to bound, and it has never been tested.
#
# Usage: ./deploy/aws/p33-experiments.sh <cluster-id>

cd "$(dirname "$0")/../.."
. deploy/aws/lib.sh
require_creds

CLUSTER="${1:?usage: p33-experiments.sh <cluster-id>}"
export KUBECONFIG="$PWD/.fleet/kubeconfig-${CLUSTER}"
[ -f "$KUBECONFIG" ] || die "no kubeconfig for $CLUSTER"
OUT=".fleet/p33"; mkdir -p "$OUT"

RURL="http://127.0.0.1:8000"
pf() { kubectl port-forward svc/fleet-router 8000:8000 >/dev/null 2>&1 & echo $!; }
status() { curl -s -m 5 "$RURL/fleet/status"; }
live() { status | python3 -c 'import sys,json;print(json.load(sys.stdin)["live_workers"])' 2>/dev/null || echo 0; }

wait_live() {  # n timeout_s
  for _ in $(seq 1 "${2:-120}"); do [ "$(live)" = "$1" ] && return 0; sleep 1; done
  return 1
}

PF=$(pf); sleep 5
trap 'kill $PF 2>/dev/null' EXIT

echo "=== cluster ==="
kubectl get nodes -o wide --no-headers | awk '{print "  "$1"  "$2"  "$6}'
kubectl get pods -o wide --no-headers | awk '{print "  "$1"  "$3"  node="$7}'
echo

# --- (c) cross-node serving -------------------------------------------------
# Verified by worker identity, not assumed from "it returned 200". Two pods on
# ONE node would still answer happily and prove nothing about the network.
echo "=== (c) cross-node serving ==="
wait_live 2 180 || die "fleet never reached 2 live workers"
: > "$OUT/c-workers.txt"
for _ in $(seq 1 20); do
  curl -s -m 20 -X POST "$RURL/v1/completions" -H 'content-type: application/json' \
    -d '{"prompt":"hello","max_tokens":8}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("worker_id","?"))' >> "$OUT/c-workers.txt" 2>/dev/null || echo ERR >> "$OUT/c-workers.txt"
done
echo "  responses by worker:"; sort "$OUT/c-workers.txt" | uniq -c | sed 's/^/    /'
echo "  pod -> node placement:"
kubectl get pods -l app=fleet-worker -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName --no-headers | sed 's/^/    /'
NODES_USED=$(kubectl get pods -l app=fleet-worker -o jsonpath='{.items[*].spec.nodeName}' | tr ' ' '\n' | sort -u | wc -l | tr -d ' ')
echo "  distinct nodes hosting workers: $NODES_USED  (need >=2 for this to mean anything)"

# --- (d)+(f) terminate a whole instance under load --------------------------
echo
echo "=== (d)+(f) terminating a worker's INSTANCE under load ==="
VICTIM_POD=$(kubectl get pods -l app=fleet-worker -o jsonpath='{.items[0].metadata.name}')
VICTIM_NODE=$(kubectl get pod "$VICTIM_POD" -o jsonpath='{.spec.nodeName}')
VICTIM_IP=$(kubectl get node "$VICTIM_NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
VICTIM_ID=$(aws2 ec2 describe-instances \
  --filters "Name=private-ip-address,Values=$VICTIM_IP" "Name=tag:project,Values=${PROJECT_TAG}" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
[ -n "$VICTIM_ID" ] && [ "$VICTIM_ID" != "None" ] || die "could not map node $VICTIM_NODE to an instance"
echo "  victim pod=$VICTIM_POD node=$VICTIM_NODE ip=$VICTIM_IP instance=$VICTIM_ID"

# Safety: never terminate anything that is not ours.
P=$(aws2 ec2 describe-instances --instance-ids "$VICTIM_ID" \
     --query 'Reservations[0].Instances[0].Tags[?Key==`project`]|[0].Value' --output text)
[ "$P" = "$PROJECT_TAG" ] || die "refusing: $VICTIM_ID has project=$P"

./.fleet/loadgen-bin -url "$RURL" -rps 5 -duration 150s -concurrency 32 \
  -max-tokens 8 -prompts .fleet/x2x3/prompts.txt \
  -kill-after 40s \
  -kill-cmd "aws ec2 terminate-instances --region ${INF_REGION} --instance-ids ${VICTIM_ID}" \
  -out "$OUT/d-node-kill.json" 2>&1 | tail -10

echo
echo "  router counters after:"; status | python3 -m json.tool | sed -n '1,12p'
echo "  nodes after:"; kubectl get nodes --no-headers | sed 's/^/    /'
echo
echo "artifacts in $OUT/"
