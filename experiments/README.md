# experiments/

Every experiment **driver** that spends money, in one committable place.

These used to live in `.context/<name>/`, which is git-ignored — so the code that
produced each published result was never in the repo, only its logs. Artifacts
(logs, PNGs, profile/metrics JSON) still belong in `.context/`; this directory is
the reproducible half.

## Before and after ANY paid run

```bash
experiments/validate_teardown.sh          # is anything billing?
experiments/validate_teardown.sh --fix    # ...and kill it if so
```

It asks EC2 directly rather than trusting a teardown message. Every teardown path
in this repo was open-loop — issue terminate, print success, exit — so a swallowed
prompt or a discarded stderr looked identical to a clean stop. Run this before
walking away.

## The experiments

| file | what it does | cost |
|---|---|---|
| `step3_box_failure.sh` | Terminates the control-plane BOX, proves training continues without it, then `orch up --run-id` adopts the live fleet. 9 PASS/FAIL checks. | ~$1.50 |
| `validate_teardown.sh` | Closed-loop billing check (above). | $0 |
| `e7_run.sh` + `e7_one_node.py` | 2-node preemption, one scheduled kill. | ~$1 |
| `e6_staggered.py` | Staggered preemption. | ~$4 |
| `e5_chaos_ladder.py` | Chaos ladder — increasing kill density. | ~$8 |
| `e4_rolling_pairs.py` | **Simultaneous** pair kills. Predates `PREEMPT_SCHEDULE`, which can now express this in the main driver. | ~$4 |
| `e4_compare_clean.py` | Compares clean vs preempted runs from cached S3 profiles. | $0 |
| `e1_run.sh`, `e1b_run.sh`, `e2_run.sh`, `e2b_run.sh` | Early single/multi-node validations. | ~$1 each |

## Conventions worth keeping

- **`cd "$(dirname "$0")/.."`** — one level up, not two. These moved from
  `.context/<exp>/`, and every relative path was rewritten for the new depth.
- **Pass `TRAIN_BUDGET_SECONDS`, never an experiment's own budget key.**
  `multinode` reads `BASELINE_SECONDS`; `multinode-preempt` reads
  `TRAIN_TOTAL_SECONDS`. Naming the wrong one is silent — the run takes the 300s
  default, trains ~200s and finishes before your test starts. `orch up`
  translates `TRAIN_BUDGET_SECONDS` into whichever key applies.
- **`orch down` needs `--yes` in a script.** It prompts when stdin is a tty; with
  output redirected the prompt is invisible and teardown silently no-ops.
- **Wait on conditions, not clocks.** A fixed `sleep` races boot time, which
  varies with the ~118s dataset pull.
