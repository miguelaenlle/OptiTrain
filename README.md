<h1 align="center">OptiTrain</h1>

<p align="center">
  <b>Train and serve LLMs on hardware that can disappear at any moment.</b><br>
  Fault-tolerant multi-node training on AWS spot instances — preemption is a
  normal, survivable event, not an outage.
</p>

<p align="center">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="Platform: AWS" src="https://img.shields.io/badge/platform-AWS-orange.svg">
</p>

<p align="center">
  <a href="#quickstart-no-aws-no-gpu">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#running-on-aws">Running on AWS</a> ·
  <a href="./ROADMAP.md">Roadmap</a>
</p>

---

## Why

Spot / interruptible GPUs cost 60–90% less than on-demand, and can be reclaimed
with little or no warning. The bet is simple: **if the training loop can resume
without losing correctness, and a control plane can replace lost nodes
automatically, the price difference more than pays for the recovery overhead.**

Most of the difficulty isn't the model — it's everything around it. This project
takes Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) as a pinned
submodule and owns only the fault-tolerance layer.

## What "survives preemption" means here

A run that is killed and resumed must continue along the **same loss trajectory**
as an uninterrupted one. That requires checkpointing everything that affects the
next step:

- model weights and optimizer state
- the step number and the remaining time budget
- **all RNG states**
- the **data-loader position**

...and restoring it through **one code path** — startup always tries to restore
the latest checkpoint and falls back to fresh, never two branches. Checkpoints are
written to a temp key and atomically renamed, so a mid-write kill cannot corrupt
the last good one.

## Status

| Area | State |
|---|---|
| Single node, 1 GPU · kill + resume on spot | ✅ proven on AWS |
| Single node, 4 GPUs · DDP | ✅ proven on AWS |
| Multi-node spot · epoch supervisor, node replacement | ✅ proven on AWS — [experiment writeups](#results) |
| Async two-tier checkpointing, cost ledger, run profiles | ✅ built |
| Spot-capacity collector (`spotwatch`) | ✅ built, collecting |
| Inference fleet (Python router + workers) + Go load generator | 🟡 implemented and tested locally; **no cloud results yet** |
| Headline benchmark: spot vs on-demand to the same loss | ⬜ not yet run |
| Go / Kubernetes control plane, RL finetuning, platform UI | ⬜ planned — see [ROADMAP.md](./ROADMAP.md) |

## Quickstart (no AWS, no GPU)

Everything below runs on a laptop CPU and costs nothing.

```bash
git clone --recurse-submodules https://github.com/miguelaenlle/Spot-Distributed-LLM-Training.git
cd Spot-Distributed-LLM-Training
pip install -e ".[dev,fleet]"        # add ",viz" for W&B + timeline plots
```

Already cloned without submodules? `git submodule update --init`.

**Run the test suite** — 420 passing, 2 skipped, no AWS calls, no GPU:

```bash
pytest -q
```

**Prove kill-and-resume locally.** This is the core invariant, and it is
deliberately testable before any instance is launched:

```bash
pytest tests/test_kill_resume.py -v
```

**Run the whole multi-node protocol on localhost** — real sidecars, real static
`torchrun`, real epoch documents, no cloud:

```bash
pytest tests/test_epoch_e2e.py -v
```

**Bring up an inference fleet as local processes** (router + 2 workers, no AWS):

```bash
spot-orchestrate fleet up --local --workers 2
spot-orchestrate fleet status --local
```

Then point the Go load generator at it and kill a worker mid-test:

```bash
cd loadgen
go run . -url http://localhost:8000 -rps 20 -duration 60s \
         -kill-after 20s -kill-cmd "spot-orchestrate fleet kill-worker --local"
```

## How it works

```
                 ┌──────────────────────────────┐
   epoch.json    │      EPOCH SUPERVISOR        │   monotonic epoch documents
   (S3, one  ◄───┤  decide(Observation,Policy)  │   are the ONLY membership
    writer)      │        -> [Action]           │   authority
                 └──────────────┬───────────────┘
                                │ publishes
        ┌───────────────┬───────┴───────┬───────────────┐
        ▼               ▼               ▼               ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ sidecar │     │ sidecar │     │ sidecar │     │ sidecar │   each box obeys
   │ static  │     │ static  │     │ static  │     │ static  │   the epoch by
   │torchrun │     │torchrun │     │torchrun │     │torchrun │   restarting a
   └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘   STATIC world
        └───────────────┴───────┬───────┴───────────────┘
                                ▼
                    S3: checkpoints · dataset · metrics · logs
```

**No node hosts a rendezvous store, so any node is killable — including the
master.** A single orchestrator publishes monotonic epoch documents; every box
runs a sidecar that obeys them by launching a *static* `torchrun` per epoch.

Key properties:

- **Survivors keep training.** When a node dies, the world shrinks to N−1 and
  training continues while a replacement boots — it does not block waiting.
- **Constant global batch.** Gradient accumulation holds the global batch fixed
  regardless of world size, so a membership change does not bend the loss curve.
- **Two-tier checkpoints.** Node-local NVMe for instant survivor restores;
  async rank-0 S3 for replacements. A group-MIN agreement picks the resume step.
- **Budget lives in the checkpoint.** The remaining training-second budget is
  checkpointed, so downtime is never billed against the run.
- **The decision core is a pure reducer** — `decide(Observation, Policy) ->
  [Action]` — table-tested and replayable without touching AWS.

This replaced an earlier `torchrun`-elastic design that passed every local test on
torch 2.4 and hung for >180s on the DLAMI's torch ≥2.8. Full rationale and the
protocol spec: [docs/multinode-design.md](./docs/multinode-design.md).

## Results

Per-experiment writeups, each with timelines, validity checks, and the runs that
produced them:

| Experiment | Question | Writeup |
|---|---|---|
| **E1** | Can checkpointing be made cheap enough to ignore? | [e1-results.md](./docs/e1-results.md) |
| **E2** | Why do survivors idle while a replacement boots? | [e2-results.md](./docs/e2-results.md) |
| **E2b** | Do survivors now train *through* a failure? | [e2b-results.md](./docs/e2b-results.md) |
| **E4** | Does recovery cost grow with the number of failures? | [e4-results.md](./docs/e4-results.md) |

<p align="center">
  <img src="./docs/img/e4-rolling-pairs.png" width="720"
       alt="Timeline of three rounds of paired node failures with automatic replacement">
</p>

<p align="center"><i>
E4 — half a 4-node world destroyed three times in a row, including the master and
including replacements. Survivor rows stay green (training) through every dip.
</i></p>

> **Headline benchmark pending.** The spot-vs-on-demand cost comparison at scale,
> and the inference fleet's stress-test numbers, are not yet run. They'll be added
> here when they exist rather than estimated.

## Running on AWS

> [!WARNING]
> **The commands in this section launch real EC2 instances and cost real money.**
> Every one of them accepts `--dry-run`, which logs each AWS call and launches
> nothing. Use it first. Individual experiments in `docs/` cost roughly $0.75–$2
> each; a multi-day run costs considerably more.

**Prerequisites**

1. AWS credentials resolvable by boto3 (an SSO profile is recommended).
2. `cp .env.example .env` and set `SPOT_TRAIN_BUCKET` to a real, globally-unique
   bucket name you own.
3. **A G-class quota increase** for both on-demand and spot. Fresh accounts sit at
   zero and approval can take days — [file it early](https://console.aws.amazon.com/servicequotas/).

**One-time infrastructure** (idempotent: S3 bucket + IAM instance profile +
security group):

```bash
spot-orchestrate setup --dry-run     # inspect first
spot-orchestrate setup
```

**Stage a dataset.** Small corpora prepare locally; large ones prepare *in* AWS on
a single throwaway self-terminating box, because ~110 GB of transient cache does
not fit a laptop:

```bash
DATASET=shakespeare_char spot-orchestrate stage-data
DATASET=openwebtext      spot-orchestrate stage-data --remote
```

**Run experiments:**

```bash
spot-orchestrate baseline            # on-demand reference run
spot-orchestrate spot                # spot + controlled kill + resume
spot-orchestrate multinode           # N-node elastic run
spot-orchestrate multinode-preempt   # ...with a scheduled kill schedule
spot-orchestrate compare <run_id>...
```

`spot-orchestrate --help` lists every verb, including the scaling sweeps
(`scaling-clean`, `scaling-preempt`), the inference fleet (`fleet ...`), and the
spot-capacity collector (`spotwatch ...`).

### Long runs: put the control plane in the cloud

A multi-day run shouldn't depend on your laptop staying awake. `orch up` launches
the control plane on an always-on (never spot) `t3.micro`, runs the experiment
there under systemd, then attaches your terminal to the live dashboard:

```bash
spot-orchestrate orch up --experiment multinode --env NODES=8 --env TRAIN_BUDGET_SECONDS=129600
# boot progress → full-screen dashboard. Ctrl-C DETACHES; it cannot stop the run.

spot-orchestrate orch status      # heartbeat age, epoch, world size, step/loss, cost
spot-orchestrate orch logs        # reattach (finds the active run for you)
spot-orchestrate orch down --all  # terminate the control plane + its training fleet
```

It authenticates with an instance-profile role (no keys are copied, so nothing
expires mid-run), streams its log to S3, restarts the run on crash **resuming the
same `run_id`**, and bills itself into the run's cost ledger.

### Watching a run

Training boxes have **no inbound ports** — you attach over SSM Session Manager,
and the orchestrator prints the exact command with the instance id at launch.

```bash
spot-orchestrate logs <run_id>            # live per-node dashboard; --grid tiles all nodes
aws ssm start-session --target <instance-id> --region us-east-1
```

## Repository layout

Deliberately minimal — directories appear when their roadmap part begins.

```
src/spot_train/     the trainer: one resume path, wall-clock budget, checkpoint/restore
  train.py            entrypoint — restore-or-fresh, time budget, eval, metrics.json
  checkpoint.py       full-state save/restore + verify() + post-save smoke test
  s3_store.py         local + S3 behind one interface; atomic rename, SHA-256
  rng.py / data.py    RNG capture-restore; memmap batches + loader position
src/orchestrator/   the control plane (boto3) — runs on your laptop or a t3.micro
  aws.py              the ONLY module that calls AWS; every call logs + honors --dry-run
  supervisor.py       epoch supervisor: pure decide() reducer + effects loop
  sidecar.py          per-box: obey epoch.json, run static torchrun per epoch
  orch.py             durable remote orchestrator (up/status/logs/down)
  spotwatch.py        GPU spot-availability collector (Lambda, ~$1-3/mo)
src/inference/      serving fleet: worker + router + heartbeat registry
loadgen/            Go load generator / chaos harness (own go.mod)
third_party/        nanoGPT as a pinned submodule — we import the model, not rewrite it
docs/               design notes, experiment writeups, IAM policies, runbooks
tests/              422 tests — checkpoint/resume, supervisor, sidecar, fleet, E2E
```

## Design principles

These are load-bearing; changes that violate them get rejected.

1. **Don't write the model.** We own the fault-tolerance layer, not the transformer.
2. **One resume code path.** Restore-latest-or-fresh, never two branches.
3. **Determinism first.** Kill-and-resume must pass locally on CPU before it goes
   near a spot instance. Determinism is miserable to debug on a vanishing machine.
4. **Checkpoint everything that affects the next step** — including RNG states and
   data-loader position. Those two are what keep resume from silently diverging.
5. **Atomic writes.** Temp key, then rename.
6. **Assume no warning.** Poll IMDS and handle SIGTERM, but checkpoint
   periodically regardless — some kills give no notice at all.
7. **Measure, don't predict.** Every claim in `docs/` cites a run id and a
   validity section. Predictions that missed are recorded alongside the ones that
   landed.

## Development

```bash
pip install -e ".[dev,fleet]"
pytest -q                          # 420 pass, 2 skip — CPU only, no AWS
ruff check --fix . && ruff format .
```

Enable formatting on commit with either mechanism:

```bash
git config core.hooksPath .githooks     # native hook, uses the ruff on PATH
# or
pip install pre-commit && pre-commit install
```

`third_party/` is excluded from linting — nanoGPT is upstream and read-only. Go
code lives in its own module with its own `go.mod`; ruff governs Python only.

## A note on regions

This AWS account is shared with a separate inference platform that owns
**us-east-2**. Training is pinned to **us-east-1** and `aws.set_region()` raises on
anything else, so GPU quota is isolated per region and a careless
`describe-instances` in one project cannot see the other's boxes. Full contract:
[docs/region-split.md](./docs/region-split.md).

## Documentation

| Document | What's in it |
|---|---|
| [ROADMAP.md](./ROADMAP.md) | The plan of record — inference fleet, RL finetuning, unified platform |
| [CLAUDE.md](./CLAUDE.md) | Working guidance, phase history, conventions |
| [docs/multinode-design.md](./docs/multinode-design.md) | The epoch-supervisor protocol |
| [docs/checkpoint-tiers.md](./docs/checkpoint-tiers.md) | How the two checkpoint tiers work |
| [docs/gpt2-reproduction.md](./docs/gpt2-reproduction.md) | Scaling to a GPT-2-class run; the streaming data plane |
| [docs/failure-cost-runbook.md](./docs/failure-cost-runbook.md) | Running the controlled failure-cost A/B |
| [docs/backlog.md](./docs/backlog.md) | Known gaps, each with a cost and a reason |
| [docs/iam/](./docs/iam/) | Least-privilege policies per principal |

## License

MIT — see [LICENSE](./LICENSE). nanoGPT is included as a submodule under its own
MIT license.
