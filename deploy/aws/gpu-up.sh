#!/usr/bin/env bash
# E1/E2 rig: one g5.xlarge serving gpt2-xl as a BARE PROCESS in us-east-2.
#
# Bare process on purpose. P3.3 produced five separate infrastructure failures
# that each looked like an application bug; adding CUDA, containerd and a device
# plugin on top of a first GPU run would make an engine fault and a container
# fault indistinguishable at $1/hr. Kubernetes comes after these numbers exist.
#
# ⚠️ BILLABLE: g5.xlarge is ~$1.006/hr. ./deploy/aws/gpu-down.sh when done.

cd "$(dirname "$0")/../.."
. deploy/aws/lib.sh
require_creds

TYPE="${GPU_TYPE:-g5.xlarge}"
MODEL="${GPU_MODEL:-gpt2-xl}"
RUN="gpu-$(date +%Y%m%d-%H%M%S)"
S3="s3://${BUCKET}/gpu/${RUN}"

# Base DLAMI: Ubuntu 22.04 with the NVIDIA driver and CUDA already installed.
# Building the driver ourselves is ~10 minutes of billable boot for nothing.
AMI=$(aws2 ssm get-parameters \
  --names /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query 'Parameters[0].Value' --output text)
[ -n "$AMI" ] && [ "$AMI" != "None" ] || die "could not resolve the Deep Learning AMI"

SG_ID=$(aws2 ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
          --query 'SecurityGroups[0].GroupId' --output text)
[ "$SG_ID" = "None" ] && die "run ./deploy/aws/infra-up.sh first"

# loadgen runs on the laptop and hits the worker directly, so 8001 needs to be
# open to us specifically (the existing rule is SG-internal only).
MYIP="$(curl -s -m 10 https://checkip.amazonaws.com)/32"
aws2 ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=tcp,FromPort=8001,ToPort=8001,IpRanges=[{CidrIp=${MYIP},Description=\"loadgen -> worker\"}]" \
  >/dev/null 2>&1 && ok "opened 8001 to $MYIP" || note "8001 already open to $MYIP"

# Same NACL-aware selection as k3s-up: a subnet can accept inbound while every
# outbound connection hangs, because NACLs are stateless.
SUBNET="${INFERENCE_SUBNET:-}"
if [ -z "$SUBNET" ]; then
  for cand in $(aws2 ec2 describe-subnets \
        --filters Name=map-public-ip-on-launch,Values=true Name=state,Values=available \
        --query 'Subnets[].SubnetId' --output text); do
    az=$(aws2 ec2 describe-subnets --subnet-ids "$cand" --query 'Subnets[0].AvailabilityZone' --output text)
    aws2 ec2 describe-instance-type-offerings --location-type availability-zone \
      --filters Name=instance-type,Values=$TYPE "Name=location,Values=$az" \
      --query 'InstanceTypeOfferings[0].Location' --output text | grep -q "$az" || {
        note "skipping $cand — $TYPE not offered in $az"; continue; }
    nacl=$(aws2 ec2 describe-network-acls --filters "Name=association.subnet-id,Values=$cand" \
             --query 'NetworkAcls[0].NetworkAclId' --output text)
    permissive=$(aws2 ec2 describe-network-acls --network-acl-ids "$nacl" \
      --query "NetworkAcls[0].Entries[?Egress==\`false\` && RuleAction=='allow' && Protocol=='-1' && CidrBlock=='0.0.0.0/0'].RuleNumber" \
      --output text)
    [ -n "$permissive" ] && { SUBNET="$cand"; break; }
    note "skipping $cand — NACL $nacl blocks ephemeral inbound"
  done
fi
[ -n "$SUBNET" ] || die "no usable subnet for $TYPE"

# Ship the source rather than cloning: this branch is not pushed anywhere the
# box can reach, and a tarball in our own bucket needs no git credentials.
tar czf /tmp/src.tgz src pyproject.toml
aws2 s3 cp /tmp/src.tgz "${S3}/src.tgz" --only-show-errors
ok "source staged at ${S3}/src.tgz"

UD=$(cat <<EOF
#!/bin/bash
set -x
exec > >(tee /var/log/gpu-boot.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/opt/hf

nvidia-smi || echo "NO GPU DRIVER"

apt-get update -y && apt-get install -y python3-pip unzip
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install

# cu121 wheels match the driver on this AMI. Installing the default CPU wheel
# here would silently serve on CPU and quietly invalidate every number.
pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
pip3 install --no-cache-dir transformers accelerate tiktoken fastapi uvicorn numpy boto3 requests

mkdir -p /opt/fleet && cd /opt/fleet
aws s3 cp ${S3}/src.tgz /opt/fleet/src.tgz --region ${INF_REGION}
tar xzf src.tgz

python3 -c "import torch;print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))" \
  | tee /opt/fleet/cuda.txt
aws s3 cp /opt/fleet/cuda.txt ${S3}/cuda.txt --region ${INF_REGION}

cd /opt/fleet
PYTHONPATH=/opt/fleet/src \
PRETRAINED_MODEL=${MODEL} SERVE_ENGINE=hf DEVICE=cuda SERVE_DTYPE=auto \
FLEET_WORKERS_URI="" HOST=0.0.0.0 PORT=8001 \
nohup python3 -m inference.worker --port 8001 > /var/log/worker.log 2>&1 &

# Ready only once the model is actually loaded and answering.
for i in \$(seq 1 120); do
  curl -sf http://127.0.0.1:8001/healthz && break
  sleep 10
done
curl -s http://127.0.0.1:8001/healthz > /opt/fleet/ready.json
aws s3 cp /opt/fleet/ready.json ${S3}/ready.json --region ${INF_REGION}
aws s3 cp /var/log/worker.log ${S3}/worker.log --region ${INF_REGION}
echo GPU_WORKER_READY
EOF
)

IID=$(aws2 ec2 run-instances \
  --image-id "$AMI" --instance-type "$TYPE" --count 1 \
  --iam-instance-profile "Name=${IAM_PROFILE}" \
  --network-interfaces "DeviceIndex=0,AssociatePublicIpAddress=true,Groups=$SG_ID,SubnetId=$SUBNET" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${RUN}},{Key=project,Value=${PROJECT_TAG}},{Key=gpu_run,Value=${RUN}}]" \
  --user-data "$UD" \
  --query 'Instances[0].InstanceId' --output text)
ok "launched $IID ($TYPE)"
echo "$RUN" > .fleet/current-gpu-run
echo "$IID" > .fleet/current-gpu-instance

aws2 ec2 wait instance-running --instance-ids "$IID"
PUB=$(aws2 ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$PUB" > .fleet/current-gpu-ip
ok "running at $PUB"

echo "  waiting for torch + gpt2-xl (~6GB download on first boot)…"
for i in $(seq 1 120); do
  curl -sf -m 5 "http://${PUB}:8001/healthz" >/dev/null 2>&1 && { ok "worker READY"; break; }
  sleep 15
done
curl -s -m 10 "http://${PUB}:8001/healthz" || die "worker never came up — aws s3 cp ${S3}/worker.log -"
echo
aws2 s3 cp "${S3}/cuda.txt" - 2>/dev/null || true

cat <<EOF

==================================================
  RUN:      $RUN
  INSTANCE: $IID
  WORKER:   http://${PUB}:8001
  S3:       ${S3}
==================================================
  TEAR DOWN:  ./deploy/aws/gpu-down.sh
EOF
