#!/usr/bin/env bash
# Idempotent inference infrastructure in us-east-2: IAM instance profile,
# security group, and the k3s-specific ingress rules.
#
# Creates only. Never deletes. Never touches us-east-1 or any spot-train-*
# resource — IAM is GLOBAL, so the inference-* prefix is the only thing keeping
# these off training's role/profile/SG.
#
# Usage: ./deploy/aws/infra-up.sh

cd "$(dirname "$0")/../.."
. deploy/aws/lib.sh
require_creds

echo "Inference infra — region=$INF_REGION project=$PROJECT_TAG"
echo

# --- IAM + bucket + base SG, via the orchestrator's own idempotent setup -----
# Driven entirely by config, so for_inference() gives it the inference-* names.
PYTHONPATH=src python - <<'PY'
from orchestrator import aws, setup
from orchestrator.config import OrchestratorConfig

cfg = OrchestratorConfig.for_inference()
assert cfg.region != "us-east-1", "guard failed"
if not cfg.bucket:
    raise SystemExit("INFERENCE_BUCKET is unset — put it in .env")
print(f"  region={cfg.region} bucket={cfg.bucket}")
print(f"  role={cfg.role_name} profile={cfg.instance_profile} sg={cfg.security_group}")
setup.ensure_infra(cfg)
PY
ok "IAM profile + bucket + base security group"

# --- k3s ingress ------------------------------------------------------------
SG_ID=$(aws2 ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
          --query 'SecurityGroups[0].GroupId' --output text)
[ "$SG_ID" = "None" ] && die "security group $SG_NAME not found after setup"
ok "security group $SG_NAME = $SG_ID"

MYIP="$(curl -s -m 10 https://checkip.amazonaws.com || true)"
[ -n "$MYIP" ] && MYCIDR="${MYIP}/32" || { MYCIDR="0.0.0.0/0"; note "could not detect your IP; opening admin ports to 0.0.0.0/0"; }

# Adding a rule that already exists raises InvalidPermission.Duplicate, which is
# success for our purposes — hence the || true on each.
add_self() {  # port proto  — node-to-node, source is the SG itself
  aws2 ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --ip-permissions "IpProtocol=$2,FromPort=$1,ToPort=$1,UserIdGroupPairs=[{GroupId=$SG_ID}]" \
    >/dev/null 2>&1 && ok "self:$2/$1" || note "self:$2/$1 (already present)"
}
add_cidr() {  # port cidr desc
  aws2 ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --ip-permissions "IpProtocol=tcp,FromPort=$1,ToPort=$1,IpRanges=[{CidrIp=$2,Description=\"$3\"}]" \
    >/dev/null 2>&1 && ok "$2 tcp/$1 ($3)" || note "$2 tcp/$1 (already present)"
}

# THE one people miss: flannel's VXLAN overlay is UDP 8472. Without it nodes
# join fine, pods schedule fine, and cross-node traffic silently black-holes —
# which looks exactly like an application bug.
add_self 8472 udp    # flannel VXLAN
add_self 6443 tcp    # kube API (agents -> server)
add_self 10250 tcp   # kubelet (metrics/exec)
add_self 8000 tcp    # router -> anywhere in the SG
add_self 8001 tcp    # worker pods

add_cidr 6443 "$MYCIDR" "kubectl from operator"
add_cidr 8000 "$MYCIDR" "router from operator"
add_cidr 22   "$MYCIDR" "ssh from operator"

echo
echo "=============================================="
echo "  SG:      $SG_NAME ($SG_ID)"
echo "  REGION:  $INF_REGION"
echo "  ADMIN:   $MYCIDR"
echo "=============================================="
