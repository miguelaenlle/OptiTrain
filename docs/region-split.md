# Region split — training and inference share an AWS account

Two projects, one AWS account, developed in parallel in separate worktrees.

| project | region | owns |
|---|---|---|
| **spot training** (this repo) | **us-east-1** | GPU fleets, supervisor `t3.micro`, spotwatch |
| **inference platform** | **us-east-2** | everything it launches |

**Neither project launches anything in the other's region.** That is the whole
contract.

## Why region, not tags

Tags require every query to opt *in* to being careful. A region split makes
isolation the default: an over-broad `describe-instances` in us-east-1 cannot
return an inference box, so it cannot terminate one either. The failure mode
becomes "sees nothing" instead of "reaps someone else's fleet".

Three concrete properties come free:

1. **GPU quota stops being shared.** G/VT vCPU limits are *per-region*. Training
   gets the full us-east-1 allowance (64 vCPU = 16× `g5.xlarge`) with no
   inference workload competing for it, which matters because an 8-node run
   plus replacements is already 24+ vCPU. Inference gets us-east-2's allowance
   to itself — smaller needs, entirely separate ceiling.
2. **Capacity errors are honest.** `InsufficientInstanceCapacity` in us-east-1
   now means AWS is actually out, not that the other project took the last box.
3. **Blast radius is bounded.** The worst mistake either agent can make with a
   careless query is confined to its own region.

## What this does NOT isolate

Region is not an account boundary. Three things stay shared, and both projects
must handle them by **naming**, not by region:

| resource | scope | rule |
|---|---|---|
| **IAM** roles, policies, instance profiles | **global** | distinct name prefixes. Training owns `spot-train-*` (`spot-train-role`, `spot-train-profile`, `spot-train-orch-role`). Inference must not create or modify anything under that prefix. |
| **S3 buckets** | global namespace, regional data | separate buckets. Training's lives in us-east-1; inference should create its own in us-east-2 so reads are local and free. |
| **Service quotas that are global** | account-wide | few matter here; EC2-classic-era limits and S3 bucket count are the notable ones. |
| **Billing** | account-wide | cost attribution relies on the `project` tag, which is why training still stamps `project=spot-train` on every instance. |

Both guards are kept in this repo — region *and* tag — so a mistake in one is
not fatal.

## How it is enforced on the training side

Not documentation alone; the region is pinned in code.

- **`aws.TRAINING_REGION = "us-east-1"`.** `aws.set_region()` raises if asked for
  anything else. It is the single choke point — `orch`, `setup`, `spotwatch`,
  `prep`, and `experiments` all call it before touching AWS — so a stray
  `AWS_REGION=us-east-2` in the environment fails loudly at startup instead of
  quietly launching GPUs into the inference platform's quota. Escape hatch:
  `ALLOW_REGION_OVERRIDE=1`.
- **`scripts/fleetctl.sh`** passes `--region us-east-1` on every
  `describe-instances` / `terminate-instances`, in addition to filtering on
  `tag:project=spot-train`. Experiment drivers source this instead of writing
  bare AWS calls; the `trap reap EXIT` in each driver goes through `reap_ours`.
- **`.env`** carries `AWS_REGION=us-east-1`, which is what the guard expects.

### One deliberate exception

`spotwatch` queries **Spot Placement Scores across many regions** — including
us-east-2. This is read-only (`GetSpotPlacementScores`) and launches nothing, so
it does not touch the contract. Its *truth probes*, which do launch a real
`g5.xlarge`, use `SPOTWATCH_PROBE_REGION = cfg.region` and are therefore
us-east-1 only.

## Verifying

```bash
. scripts/fleetctl.sh
ours_count      # training boxes running — expect 0 between experiments
others_count    # untagged boxes in us-east-1 — context only, never touched

# the inference platform's region, which we only ever READ:
aws ec2 describe-instances --region us-east-2 \
  --filters Name=instance-state-name,Values=running,pending \
  --query 'length(Reservations[].Instances[])'
```

A non-zero count in us-east-2 is expected and is **not** ours to clean up.
