#!/usr/bin/env bash
# Create the inference platform's own S3 bucket in us-east-2.
#
# Deliberately separate from the training bucket: that one holds a ~17GB corpus
# and live run checkpoints, and S3 buckets are regional -- reading a us-east-1
# bucket from us-east-2 instances would add cross-region charges and latency to
# a registry the router polls every few seconds.
#
# Creates only. Never deletes, never touches us-east-1, never touches the
# training bucket. Safe to re-run.
#
# Usage:  ./scripts/create-inference-bucket.sh [bucket-name]

set -euo pipefail

REGION="us-east-2"
BUCKET="${1:-optitrain-inference-us-east-2}"

die() { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '  \033[32m✓\033[0m %s\n' "$*"; }

# --- guards ---------------------------------------------------------------

[ "$REGION" = "us-east-2" ] || die "REGION must be us-east-2 (training owns us-east-1)"
command -v aws >/dev/null 2>&1 || die "aws CLI not found. brew install awscli"

echo "Checking credentials…"
CALLER=$(aws sts get-caller-identity --output json 2>/dev/null) \
  || die "no working AWS credentials (try: aws configure, or set AWS_PROFILE)"
ACCT=$(printf '%s' "$CALLER" | python3 -c 'import sys,json;print(json.load(sys.stdin)["Account"])')
ARN=$(printf '%s' "$CALLER" | python3 -c 'import sys,json;print(json.load(sys.stdin)["Arn"])')
ok "account $ACCT"
ok "identity $ARN"

echo
echo "Target: s3://$BUCKET  (region $REGION)"
echo

# --- create ---------------------------------------------------------------

if aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null 2>&1; then
  ok "bucket already exists and is yours — reusing"
else
  # LocationConstraint is REQUIRED for every region except us-east-1.
  if aws s3api create-bucket \
        --bucket "$BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" >/dev/null 2>&1; then
    ok "created s3://$BUCKET"
  else
    # Bucket names are globally unique across all of AWS; fall back to a
    # name that cannot collide.
    BUCKET="optitrain-inference-us-east-2-${ACCT}"
    echo "  … base name unavailable, retrying as $BUCKET"
    aws s3api create-bucket \
      --bucket "$BUCKET" \
      --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null \
      || die "could not create $BUCKET"
    ok "created s3://$BUCKET"
  fi
fi

# --- lock down ------------------------------------------------------------

aws s3api put-public-access-block \
  --bucket "$BUCKET" --region "$REGION" \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
ok "public access blocked"

aws s3api put-bucket-tagging \
  --bucket "$BUCKET" --region "$REGION" \
  --tagging 'TagSet=[{Key=project,Value=inference},{Key=managed-by,Value=optitrain}]'
ok "tagged project=inference"

# No versioning on purpose: workers rewrite heartbeat docs every ~5s, which
# would accumulate thousands of versions. Lifecycle handles cleanup instead.
aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET" --region "$REGION" \
  --lifecycle-configuration '{"Rules":[
    {"ID":"expire-fleet-state","Status":"Enabled","Filter":{"Prefix":"fleet/"},
     "Expiration":{"Days":30}},
    {"ID":"abort-incomplete-uploads","Status":"Enabled","Filter":{"Prefix":""},
     "AbortIncompleteMultipartUpload":{"DaysAfterInitiation":7}}
  ]}'
ok "lifecycle rules set (fleet/ expires 30d, failed uploads 7d)"

# --- verify ---------------------------------------------------------------

LOC=$(aws s3api get-bucket-location --bucket "$BUCKET" --query LocationConstraint --output text)
[ "$LOC" = "$REGION" ] || die "bucket is in '$LOC', expected $REGION"
ok "verified region = $LOC"

cat <<EOF

==================================================
  BUCKET:  $BUCKET
  REGION:  $LOC
==================================================

Add these lines to .env in the repo root:

  INFERENCE_BUCKET=$BUCKET
  INFERENCE_REGION=$REGION
  INFERENCE_IAM_ROLE=inference-role
  INFERENCE_IAM_PROFILE=inference-profile
  INFERENCE_SECURITY_GROUP=inference-sg

LEAVE the training vars (AWS_REGION, SPOT_TRAIN_BUCKET, IAM_ROLE, IAM_PROFILE,
SECURITY_GROUP) exactly as they are. The inference fleet reads only the
INFERENCE_*-prefixed names, so the two projects never touch the same variable
and one .env can safely hold both. Setting a bare AWS_REGION=us-east-2 would
have moved TRAINING to us-east-2 -- which is the collision this avoids.

The inference-* IAM names matter as much as the region: IAM is global and is
NOT split by region, so the us-east-1/us-east-2 boundary does not protect it.
EOF
