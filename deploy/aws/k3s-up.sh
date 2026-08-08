#!/usr/bin/env bash
# Bring up a k3s cluster in us-east-2: 1 server + N agents.
#
#   ./deploy/aws/k3s-up.sh                 2 CPU agents  (~$0.10/hr)
#   ./deploy/aws/k3s-up.sh --gpu           2 g5.xlarge   (~$2.10/hr)
#   ./deploy/aws/k3s-up.sh --agents 1
#
# Join secret goes through S3 rather than user-data: user-data is readable by
# anyone who can call DescribeInstanceAttribute, and a k3s node-token is
# cluster-admin. The instance profile already grants S3, so this needs no new
# IAM.
#
# ⚠️ BILLABLE. Tear down with ./deploy/aws/k3s-down.sh when finished.

cd "$(dirname "$0")/../.."
. deploy/aws/lib.sh
require_creds

AGENTS=2
AGENT_TYPE="t3.medium"
SERVER_TYPE="t3.small"
GPU=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)    GPU=yes; AGENT_TYPE="g5.xlarge"; shift ;;
    --agents) AGENTS="${2:?}"; shift 2 ;;
    *) die "unknown arg: $1" ;;
  esac
done

CLUSTER="k3s-$(date +%Y%m%d-%H%M%S)"
S3="s3://${BUCKET}/k3s/${CLUSTER}"
IMAGES_S3="s3://${BUCKET}/images/fleet-images.tar"
KUBECONFIG_LOCAL=".fleet/kubeconfig-${CLUSTER}"

# Ubuntu 22.04 via SSM so we never pin a stale AMI id.
AMI=$(aws2 ssm get-parameters \
  --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
  --query 'Parameters[0].Value' --output text)
[ -n "$AMI" ] && [ "$AMI" != "None" ] || die "could not resolve Ubuntu AMI"

SG_ID=$(aws2 ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
          --query 'SecurityGroups[0].GroupId' --output text)
[ "$SG_ID" = "None" ] && die "run ./deploy/aws/infra-up.sh first"

# Subnet must be explicit. AWS's implicit selection only considers subnets
# flagged DefaultForAz, and this account's us-east-2 subnets are not — so
# RunInstances fails with "No subnets found for the default VPC" even though
# five perfectly good subnets exist. Pick a public one (MapPublicIpOnLaunch)
# and keep every node in ONE AZ: cross-AZ chatter is billed per GB, and a k3s
# control plane talks to its agents constantly.
SUBNET="${INFERENCE_SUBNET:-}"
if [ -z "$SUBNET" ]; then
  # Public + routable is NOT sufficient. Network ACLs are STATELESS, so a
  # subnet can accept inbound SSH while every outbound connection hangs: the
  # SYN leaves, and the reply — which arrives on an ephemeral port — is dropped
  # by the ingress rules. That failure looks like a broken mirror or a dead
  # internet gateway, and it cost us a cluster: apt, the AWS CLI download, SSM
  # and the k3s installer all timed out on a subnet whose NACL allowed only
  # tcp/22, tcp/5000 and tcp/8080 inbound.
  #
  # So require an ingress rule that allows all protocols from anywhere, and
  # skip subnets that do not have one rather than editing someone else's ACL.
  for cand in $(aws2 ec2 describe-subnets \
        --filters Name=map-public-ip-on-launch,Values=true Name=state,Values=available \
        --query 'Subnets[].SubnetId' --output text); do
    nacl=$(aws2 ec2 describe-network-acls \
             --filters "Name=association.subnet-id,Values=$cand" \
             --query 'NetworkAcls[0].NetworkAclId' --output text)
    permissive=$(aws2 ec2 describe-network-acls --network-acl-ids "$nacl" \
      --query "NetworkAcls[0].Entries[?Egress==\`false\` && RuleAction=='allow' && Protocol=='-1' && CidrBlock=='0.0.0.0/0'].RuleNumber" \
      --output text)
    if [ -n "$permissive" ]; then SUBNET="$cand"; break; fi
    note "skipping $cand — NACL $nacl blocks ephemeral inbound (outbound would hang)"
  done
fi
[ -n "$SUBNET" ] && [ "$SUBNET" != "None" ] || \
  die "no public subnet with a permissive NACL; set INFERENCE_SUBNET explicitly"
SUBNET_AZ=$(aws2 ec2 describe-subnets --subnet-ids "$SUBNET" \
  --query 'Subnets[0].AvailabilityZone' --output text)

# A subnet with a public IP but no internet gateway would let instances boot and
# then hang forever downloading the k3s installer — a slow, confusing failure.
IGW=$(aws2 ec2 describe-route-tables \
  --filters "Name=association.subnet-id,Values=$SUBNET" \
  --query 'RouteTables[].Routes[?starts_with(GatewayId,`igw-`)].GatewayId' --output text)
[ -z "$IGW" ] && IGW=$(aws2 ec2 describe-route-tables \
  --filters "Name=vpc-id,Values=$(aws2 ec2 describe-subnets --subnet-ids "$SUBNET" --query 'Subnets[0].VpcId' --output text)" \
            Name=association.main,Values=true \
  --query 'RouteTables[].Routes[?starts_with(GatewayId,`igw-`)].GatewayId' --output text)
[ -n "$IGW" ] || die "subnet $SUBNET has no internet gateway route — nodes could not install k3s"

echo "cluster=$CLUSTER  region=$INF_REGION  ami=$AMI"
echo "subnet=$SUBNET ($SUBNET_AZ) igw=$IGW"
echo "server=$SERVER_TYPE  agents=${AGENTS}x${AGENT_TYPE}${GPU:+ (GPU)}"
echo

tags() {  # role
  echo "ResourceType=instance,Tags=[{Key=Name,Value=${CLUSTER}-$1},{Key=project,Value=${PROJECT_TAG}},{Key=k3s_cluster,Value=${CLUSTER}},{Key=k3s_role,Value=$1}]"
}

# --- server -----------------------------------------------------------------
SERVER_UD=$(cat <<EOF
#!/bin/bash
set -x
exec > >(tee /var/log/k3s-boot.log) 2>&1
apt-get update -y && apt-get install -y unzip curl
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install

PRIVATE_IP=\$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)
PUBLIC_IP=\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)

# Preload our images. No ECR access on this IAM user, and the nodes cannot
# reach a private registry, so images ride in via S3: k3s imports any tarball
# found in /var/lib/rancher/k3s/agent/images/ at startup. This must happen
# BEFORE the k3s install, and it is the right pattern anyway -- nodes
# self-provision instead of depending on an operator to push to them.
mkdir -p /var/lib/rancher/k3s/agent/images
aws s3 cp ${IMAGES_S3} /var/lib/rancher/k3s/agent/images/fleet-images.tar --region ${INF_REGION} || true

# --tls-san lets kubectl connect from outside using the public IP; without it
# the served cert only covers the private address and every call fails on name
# mismatch.
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --tls-san \$PUBLIC_IP --node-external-ip \$PUBLIC_IP --write-kubeconfig-mode 644" sh -

until [ -f /var/lib/rancher/k3s/server/node-token ]; do sleep 2; done
until kubectl get nodes 2>/dev/null | grep -q Ready; do sleep 3; done

sed "s|127.0.0.1|\$PUBLIC_IP|" /etc/rancher/k3s/k3s.yaml > /tmp/kubeconfig
aws s3 cp /var/lib/rancher/k3s/server/node-token ${S3}/node-token --region ${INF_REGION}
aws s3 cp /tmp/kubeconfig ${S3}/kubeconfig --region ${INF_REGION}
echo "\$PRIVATE_IP" > /tmp/server-ip
aws s3 cp /tmp/server-ip ${S3}/server-ip --region ${INF_REGION}
echo K3S_SERVER_READY
EOF
)

SERVER_ID=$(aws2 ec2 run-instances \
  --image-id "$AMI" --instance-type "$SERVER_TYPE" --count 1 \
  --iam-instance-profile "Name=${IAM_PROFILE}" \
  --network-interfaces "DeviceIndex=0,AssociatePublicIpAddress=true,Groups=$SG_ID,SubnetId=$SUBNET" \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "$(tags server)" \
  --user-data "$SERVER_UD" \
  --query 'Instances[0].InstanceId' --output text)
ok "server launching: $SERVER_ID"

aws2 ec2 wait instance-running --instance-ids "$SERVER_ID"
SERVER_PRIVATE=$(aws2 ec2 describe-instances --instance-ids "$SERVER_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
SERVER_PUBLIC=$(aws2 ec2 describe-instances --instance-ids "$SERVER_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ok "server running  private=$SERVER_PRIVATE  public=$SERVER_PUBLIC"

echo "  waiting for k3s to publish its join token (a few minutes on first boot)…"
for i in $(seq 1 90); do
  aws2 s3 cp "${S3}/node-token" /tmp/node-token >/dev/null 2>&1 && break
  sleep 10
done
[ -f /tmp/node-token ] || die "server never published a token — check /var/log/k3s-boot.log on $SERVER_ID"
ok "join token published"

# --- agents -----------------------------------------------------------------
AGENT_UD=$(cat <<EOF
#!/bin/bash
set -x
exec > >(tee /var/log/k3s-boot.log) 2>&1
apt-get update -y && apt-get install -y unzip curl
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install

until aws s3 cp ${S3}/node-token /tmp/node-token --region ${INF_REGION}; do sleep 5; done
# Preload our images. No ECR access on this IAM user, and the nodes cannot
# reach a private registry, so images ride in via S3: k3s imports any tarball
# found in /var/lib/rancher/k3s/agent/images/ at startup. This must happen
# BEFORE the k3s install, and it is the right pattern anyway -- nodes
# self-provision instead of depending on an operator to push to them.
mkdir -p /var/lib/rancher/k3s/agent/images
aws s3 cp ${IMAGES_S3} /var/lib/rancher/k3s/agent/images/fleet-images.tar --region ${INF_REGION} || true

PUBLIC_IP=\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
curl -sfL https://get.k3s.io | K3S_URL=https://${SERVER_PRIVATE}:6443 \
  K3S_TOKEN=\$(cat /tmp/node-token) \
  INSTALL_K3S_EXEC="agent --node-external-ip \$PUBLIC_IP" sh -
echo K3S_AGENT_JOINED
EOF
)

for i in $(seq 1 "$AGENTS"); do
  AID=$(aws2 ec2 run-instances \
    --image-id "$AMI" --instance-type "$AGENT_TYPE" --count 1 \
    --iam-instance-profile "Name=${IAM_PROFILE}" \
    --network-interfaces "DeviceIndex=0,AssociatePublicIpAddress=true,Groups=$SG_ID,SubnetId=$SUBNET" \
    --instance-initiated-shutdown-behavior terminate \
    --tag-specifications "$(tags agent)" \
    --user-data "$AGENT_UD" \
    --query 'Instances[0].InstanceId' --output text)
  ok "agent $i launching: $AID"
done

# --- kubeconfig -------------------------------------------------------------
mkdir -p .fleet
aws2 s3 cp "${S3}/kubeconfig" "$KUBECONFIG_LOCAL" >/dev/null
chmod 600 "$KUBECONFIG_LOCAL"
ok "kubeconfig -> $KUBECONFIG_LOCAL"

cat <<EOF

==================================================
  CLUSTER: $CLUSTER
  SERVER:  $SERVER_ID  ($SERVER_PUBLIC)
  AGENTS:  ${AGENTS} x ${AGENT_TYPE}
  S3:      ${S3}
==================================================

  export KUBECONFIG=$PWD/$KUBECONFIG_LOCAL
  kubectl get nodes -w          # agents take ~2-3 min to join

  TEAR DOWN:  ./deploy/aws/k3s-down.sh $CLUSTER
EOF
