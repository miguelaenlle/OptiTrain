# `bench/` — inference benchmark harness

Drives the Go load generator (`loadgen/`) across sweeps and turns its JSON
reports into the two portfolio figures for the inference platform
(`docs/inference-platform-plan.md` §3).

| Experiment | What it sweeps | What it answers |
|---|---|---|
| **E2 — saturation curve** | offered RPS, fixed worker count | p50 / p95 / p99 vs load; where the fleet breaks the SLO (`C1`, the knee) |
| **E5 — scaling sweep** | worker count `N = 1, 2, 4, 8 …` at near-saturation load | sustained tokens/s vs N and the **scaling efficiency** |

```
bench/
  sweep.py     plan + run sweeps, write sweep.json        (pure planning, thin effects)
  analyze.py   L0 / SLO / knee / efficiency               (PURE — no I/O, unit-tested)
  plots.py     the two figures (PNG + SVG) + CSV table    (rendering only)
```

`bench` is a plain package at the repo root — run it as `python -m bench.…`
from there. It is deliberately **not** installed by `pyproject.toml`; only its
matplotlib dependency is, under the `bench` extra.

---

## Metric definitions

Everything keys off one calibration, so no target is arbitrary:

| Symbol | Definition |
|---|---|
| **L0** | unloaded single-request p50 latency — 1 worker, RPS = 1, `max_tokens=64` |
| **SLO** | `p99 ≤ 3 × L0` |
| **C1** (the knee) | the highest offered RPS whose p99 still holds the SLO |
| **p99 / p50** | tail-to-median spread — target `< 3×` (`< 2×` excellent) |
| **scaling efficiency** | `(tput_N / N) ÷ tput_1` — targets **≥ 90 % @ N=4**, **≥ 80 % @ N=8** |

Two rules the analysis enforces so the numbers stay honest:

* A point only counts as holding the SLO if it actually **succeeded** and its
  error rate is under 1 % (`analyze.DEFAULT_MAX_ERROR_RATE`) — a run that
  failed 90 % of its requests must not "pass" on the fast 10 % that survived.
* The knee is the top of the **leading** run of healthy points: the walk stops
  at the first violation, so a lucky sample above a real breakdown cannot
  report capacity the fleet does not have. Two degenerate cases are labelled
  rather than smoothed over — every point held (`C1 ≥ max swept RPS`, "sweep
  higher") and no point held (`C1` is below the smallest swept RPS).

All numbers on these charts are **client-side** (loadgen). Server-side
(Prometheus) numbers are reported separately, per the plan's reporting rule.

---

## Running a sweep locally

Prerequisites: the Go toolchain on `PATH` (the harness builds `loadgen/` once
into `.fleet/loadgen-bin`) and `pip install -e '.[bench]'` for matplotlib.

```bash
# 0. a fleet to shoot at
spot-orchestrate fleet up --local --workers 1

# 1. see the plan before it runs anything (repo convention for anything that
#    spends resources — no build, no requests, no artifacts)
python -m bench.sweep saturation --url http://localhost:8000 \
  --rps 1,2,4,8,16,32 --workers 1 --duration 60 --dry-run

# 2. E2 — the saturation curve (keep RPS=1 in the list: that point IS L0)
python -m bench.sweep saturation --url http://localhost:8000 \
  --rps 1,2,4,8,16,32 --workers 1 --duration 60 --warmup 20 --plot

# 3. E5 — the scaling sweep, at ~80 % of the knee E2 just found
python -m bench.sweep scaling --url http://localhost:8000 \
  --workers 1,2,4,8 --rps-per-worker 6 --duration 60 --plot

# re-analyze / re-plot an existing sweep without re-running the load
python -m bench.sweep report --sweep .fleet/bench/e2-saturation/sweep.json --plot
python -m bench.plots --sweep .fleet/bench/e5-scaling/sweep.json --theme dark
```

Artifacts land under `.fleet/bench/<experiment>/` (already git-ignored, same
place `fleet preempt` writes):

| File | What it is |
|---|---|
| `rps8.json`, `n4.json`, … | the **loadgen's own report** for one sweep point, untouched |
| `sweep.json` | the manifest: sweep parameters + every point's report inline — the single input the analysis and the charts need |
| `e2-saturation.png` / `.svg` | the saturation figure |
| `e5-scaling.png` / `.svg` | the scaling figure |
| `*.csv` | the **table view** of each figure — every plotted value, readable without colour |

### Useful flags

| Flag | Why |
|---|---|
| `--dry-run` | print the exact command list, execute nothing |
| `--warmup N` | run and discard N seconds first, so one-time model init (CUDA context, kernel autotune) never lands in `L0` |
| `--concurrency` | in-flight cap; `0` (default) auto-sizes to `max(64, 4 × rps)`. The generator is open loop — it counts a *dropped* request whenever no client worker is free, which is only a fleet signal if the client pool is comfortably bigger than `rps × latency` |
| `--rps-per-worker` | E5 offers `rps_per_worker × N`, so every point sits at the same per-worker pressure. Efficiency measured on an idle fleet is meaningless |
| `--scale-cmd 'kubectl scale deploy/worker --replicas={n}'` | resize the fleet with something other than the local `spot-orchestrate fleet` CLI |
| `--no-scale` | you resize the fleet yourself between points |
| `--cloud` | resize the **EC2** fleet — launches and terminates real instances (costs money) |
| `--theme light\|dark` | both are selected palettes, not an inverted flip |

`--duration 60` is the smallest window that gives a stable p99 at low RPS
(60 requests at 1 rps). A local CPU fleet is for proving the harness, not for
the headline numbers — those come from the GPU run.

---

## Chart conventions

Both figures follow one system, so they read as a pair:

* **Palette** — the validated categorical slots 1–3 in fixed order, plus a
  dark set stepped for the dark surface. Threshold lines wear the reserved
  *status* red and never a series colour. Validated in both modes (all checks
  pass; light-mode slot 3 sits at 2.74:1 contrast, whose relief obligation is
  met by the direct labels and the CSV table view every figure ships with).
* **Marks** — 2 px lines, 8 px markers with a 2 px surface ring, hairline
  solid gridlines, no top/right spines, **one y-scale per chart** (never a
  second axis).
* **Text** — labels and legends wear ink tokens, never the series colour;
  identity comes from the coloured mark beside them. Labels are selective:
  series ends, the knee, the efficiency at each N.
* **Log axes** appear on doubling sweeps and are named in the axis label — on
  a linear scale one post-knee blow-up squashes the entire pre-knee region,
  which is exactly where the SLO argument lives.
* Nothing is smoothed, extrapolated or invented. A sweep that never crossed
  the SLO says so on the chart instead of drawing a knee that was not
  measured.

---

## Tests

`tests/test_bench.py` — CPU-only, offline, fast. It builds loadgen-shaped
report dicts inline and exercises the pure analysis (parsing against the real
schema, `L0`/SLO derivation, knee detection including the all-under and
all-over cases, the p99/p50 ratio, efficiency math) plus sweep planning and
`--dry-run` (asserting the generated command list without executing it). The
only plotting assertion is that files get written.

```bash
python -m pytest tests/test_bench.py -q
```
