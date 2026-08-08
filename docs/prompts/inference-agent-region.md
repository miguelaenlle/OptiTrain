# Prompt for the inference-platform agent

Paste the block below into that agent's session (or into its `CLAUDE.md`).

---

## AWS region contract — read before provisioning anything

You share one AWS account with a **spot-training project** running in a separate
worktree. We isolate by **region**, and it is the only thing keeping the two
workloads from interfering:

| project | region |
|---|---|
| spot training (not yours) | **us-east-1** |
| **inference platform (you)** | **us-east-2** |

**Launch everything in `us-east-2`. Never launch, terminate, modify, or stop
anything in `us-east-1`.**

### Why

GPU quota is *per-region*. The training project runs 8-node `g5.xlarge` fleets
that consume most of the us-east-1 G/VT vCPU allowance, and it launches
replacement nodes automatically when boxes are preempted — so a shared region
would mean the two projects racing for the same capacity, with failures that
look like AWS capacity errors rather than contention. Splitting by region gives
each project its own quota ceiling. Inference needs far fewer instances, so
us-east-2's allowance is comfortable.

### Rules

1. **Pin the region explicitly.** Set `AWS_REGION=us-east-2` in your config, and
   pass `--region us-east-2` on every AWS CLI call rather than relying on
   ambient config. Ambient defaults are how cross-region accidents happen.
2. **Never run an unfiltered `terminate-instances`.** This pattern is an
   account-wide kill switch:
   ```bash
   # DO NOT DO THIS
   ids=$(aws ec2 describe-instances --filters Name=instance-state-name,Values=running \
         --query 'Reservations[].Instances[].InstanceId' --output text)
   aws ec2 terminate-instances --instance-ids $ids
   ```
   Always scope by **both** region and your own tag.
3. **Tag every instance you launch** with a distinct project tag, e.g.
   `project=inference`. Training stamps `project=spot-train`. Tags are what makes
   billing attributable and gives a second guard behind the region split.
4. **IAM is global — it is NOT split by region.** Do not create or modify
   anything named `spot-train-*` (`spot-train-role`, `spot-train-profile`,
   `spot-train-orch-role`, the `spot-train-s3` inline policy, or the
   `spot-train-sg` security group). Use your own prefix, e.g. `inference-*`.
5. **Use your own S3 bucket**, created in us-east-2. Do not write to or delete
   from the training bucket (`SPOT_TRAIN_BUCKET` in the other project's `.env`);
   it holds a 17 GB prepared corpus and live run checkpoints, and deleting a
   checkpoint mid-run destroys a multi-hour training job.
6. **Quota increases:** request them for **us-east-2**. Filing against us-east-1
   is wasted effort for you and confusing for the other project.

### Reading across the boundary is fine

`describe-*`, `GetSpotPlacementScores`, price lookups — all read-only calls are
harmless in any region. The rule is about anything that **creates, mutates, or
destroys**.

### Your teardown snippet

Scoped both ways. Use this shape instead of a bare reap:

```bash
INF_REGION=us-east-2
PROJECT_TAG=inference

ours_ids() {
  aws ec2 describe-instances --region "$INF_REGION" \
    --filters Name=instance-state-name,Values=running,pending \
              "Name=tag:project,Values=${PROJECT_TAG}" \
    --query 'Reservations[].Instances[].InstanceId' --output text | tr '\t' ' '
}
reap_ours() {
  local ids; ids=$(ours_ids)
  [ -n "$ids" ] && aws ec2 terminate-instances --region "$INF_REGION" --instance-ids $ids
}
trap reap_ours EXIT INT TERM
```

### If you believe you need us-east-1

Stop and ask the human first. Do not launch there on your own judgement — the
training project may have a multi-hour run in flight whose quota headroom you
would consume.
