# Sharing one AWS account: training fleet vs inference platform

Two independent workloads, one account (`841031789519`), one region
(`us-east-1`), running **in parallel**. This is the contract between them.

The training side (this repo) already complies as of the change that introduced
`scripts/fleetctl.sh`. The inference side needs the mirror image.

---

## The rule

> **Never act on an instance you did not launch. Prove ownership by tag, on
> every query, in every script — not just the ones that terminate.**

Ownership is `project`:

| workload | `project` tag | Name prefix |
|---|---|---|
| training (this repo) | `spot-train` | `spot-train-<run_id>` |
| **inference platform** | **`inference`** (pick one, never `spot-train`) | `inference-*` |

Training boxes are stamped at launch in `aws.py`:

```python
Tags: [{"Key": "Name",    "Value": f"spot-train-{run_id}"},
       {"Key": "project", "Value": "spot-train"},
       {"Key": "market",  "Value": market}, ...]
```

---

## The hazard this exists to prevent

Every experiment driver here runs under a teardown trap. It used to be:

```bash
reap() {
  ids=$(aws ec2 describe-instances \
        --filters Name=instance-state-name,Values=running,pending \
        --query 'Reservations[].Instances[].InstanceId' --output text)
  aws ec2 terminate-instances --instance-ids $ids     # <-- EVERY instance in the account
}
trap reap EXIT INT TERM
```

That is not a cleanup, it is an **account-wide kill switch**. It fires on every
exit path — success, error, Ctrl-C. With an inference fleet running in parallel
it would have terminated it, silently, on the next experiment.

The same flaw sat in the verification habit: `aws ec2 describe-instances
--filters ...running,pending --query 'length(...)'` reports "0 instances, nothing
billing" only if nobody else is running anything. With a second workload it
reports *their* count and the check becomes meaningless in both directions.

**Both are fixed here.** All drivers now source `scripts/fleetctl.sh`:

```bash
ours_ids      # instance ids tagged project=spot-train, running/pending
ours_count    # how many of ours
others_count  # how many are NOT ours — reported, never touched
reap_ours     # terminate ONLY ours
```

Teardown now prints what it left alone, so a mistake is visible rather than
silent.

---

## Genuinely shared, and how to divide it

Tags separate ownership. They do **not** separate these:

### 1. G/VT vCPU quota — 64, shared, first-come-first-served

`aws.vcpus_in_use()` deliberately counts **every** pending/running G/VT instance
in the region, whoever launched it, because external instances consume the same
quota — counting only our own would overshoot and hit `InstanceLimitExceeded`.

So the two workloads compete for one 64-vCPU pool. Proposed split:

| workload | budget | at g5.xlarge (4 vCPU) |
|---|---|---|
| training | **40 vCPU** | 8 nodes + 2 replacements |
| inference | **24 vCPU** | 6 boxes |

Set `VCPU_QUOTA` to your budget, **not** to 64. It is a self-imposed ceiling: the
launcher blocks until `used + needed <= VCPU_QUOTA`, so an honest lower number
makes each side wait for its own budget instead of starving the other. Training
currently peaks at 24 vCPU (4 nodes + 2 simultaneous replacements) and will reach
36 at 8 nodes.

Note the asymmetry: training tolerates waiting (a launcher blocks, a run is
slower). Inference serving latency usually does not. **If contention shows up,
inference should get the larger budget** — the training side is the one that can
absorb a delay.

### 2. Region and AZ

Both in `us-east-1`. GPU spot capacity is currently poor there — measured across
all AZs, `g5.xlarge` at 8 instances scored **1–3 out of 10**, and 4 truth probes
failed to acquire. Expect on-demand to be the only reliable option for both, and
expect each other's launches to make a bad pool briefly worse.

### 3. Shared infra — do not reuse

| resource | training uses | inference should |
|---|---|---|
| S3 bucket | `your-unique-spot-train-bucket` | own bucket, or an unambiguous prefix |
| IAM role / profile | `spot-train-role` / `spot-train-profile` | its **own** role |
| security group | `spot-train-sg` | its **own** SG |

Reusing the training instance profile would grant inference write access to
training checkpoints. The bucket has a lifecycle rule that **aborts incomplete
multipart uploads after 7 days** — harmless for finished writes, fatal for a
long-lived resumable upload, which is another reason not to share it.

### 4. Service quotas that are not vCPUs

`GetSpotPlacementScores` has an **undocumented distinct-configuration cap per
rolling 24h** (~28–39 observed on this account) that is *not* in service-quotas
and surfaces only as `MaxConfigLimitExceeded`. A `spotwatch` Lambda already
consumes part of that budget every 10 minutes. **If inference calls SPS, expect
refusals** and coordinate first — the budget is account-wide.

---

## Checklist for the inference side

- [ ] Every launch stamps `project=<yours>` and a `Name` prefix
- [ ] Every `describe-instances` filters `Name=tag:project,Values=<yours>`
- [ ] Teardown terminates **only** tagged instances, and logs what it skipped
- [ ] `VCPU_QUOTA` set to your budget, not 64
- [ ] Own bucket / IAM role / security group
- [ ] Ctrl-C and error paths obey the same filter as the happy path
- [ ] Instances carry a self-termination backstop so a dead driver cannot leak

---

## Prompt for the inference-platform agent

> You are building an inference platform in AWS account `841031789519`,
> `us-east-1`. **A separate GPU training workload runs in this same account, in
> parallel, right now.** It is not yours. Terminating one of its instances
> destroys a paid multi-hour experiment.
>
> **The rule: never act on an instance you did not launch, and prove ownership by
> tag on every query — not only the ones that terminate.**
>
> **1. Tag everything you launch.** Add `project=inference` (and a `Name` prefix
> like `inference-*`) to the `TagSpecifications` of every `RunInstances` call. An
> untagged instance is indistinguishable from someone else's and will be treated
> as untouchable by the other side's tooling — and by yours.
>
> **2. Filter every query.** Use
> `--filters Name=instance-state-name,Values=running,pending Name=tag:project,Values=inference`.
> This applies to teardown, to health checks, and to any "is anything still
> running?" verification. An unfiltered `describe-instances` in a cleanup script
> is an account-wide kill switch — the training side shipped exactly that bug and
> had to retrofit a shared helper (`scripts/fleetctl.sh`) to fix it. Do not repeat
> it.
>
> **3. Make teardown say what it did not touch.** Print the count of instances you
> deliberately left alone. Silent cleanup hides the failure mode you care about.
> Ensure the trap covers error and Ctrl-C paths, not just success.
>
> **4. Budget your vCPUs; do not claim the pool.** The account has **64 G/VT
> on-demand vCPUs, shared**. Quota is consumed by *every* G/VT instance in the
> region regardless of owner, so the two workloads genuinely compete. Take a
> budget of **24 vCPU** (6 × g5.xlarge) and enforce it in your launcher: block
> until `used + needed <= your_budget` rather than until the account limit. If you
> need more, ask — training can absorb a delay more gracefully than serving
> latency can, so the split is negotiable in your favour.
>
> **5. Do not reuse training's infrastructure.** It owns the S3 bucket
> `your-unique-spot-train-bucket`, the IAM role/profile `spot-train-role` /
> `spot-train-profile`, and the security group `spot-train-sg`. Create your own.
> Sharing the instance profile would grant you write access to training
> checkpoints; the bucket also has a lifecycle rule that aborts incomplete
> multipart uploads after 7 days, which will silently break a long resumable
> upload.
>
> **6. Two account-wide limits that will surprise you.** GPU **spot** capacity in
> us-east-1 is currently very poor — `g5.xlarge` measured 1–3/10 across all AZs
> with 4 of 4 acquisition probes failing, so plan for on-demand. And
> `GetSpotPlacementScores` has an **undocumented cap on distinct configurations
> per rolling 24h** (~28–39 on this account) that is not in service-quotas and
> appears only as `MaxConfigLimitExceeded`; a training-side Lambda already spends
> part of it every 10 minutes. Coordinate before calling SPS.
>
> **7. Give every instance a self-termination backstop.** Training boxes set a
> systemd timer plus `InstanceInitiatedShutdownBehavior=terminate`, so a crashed
> driver cannot leak a fleet overnight. Do the same; a shared account means your
> leak becomes everyone's quota problem.
>
> Before your first launch, confirm you can answer: *which instances are mine, by
> tag?* — and that every destructive path in your code uses that answer.
