# Handoff — Grafana training dashboard

Everything below is committed and working unless marked **OPEN**. Read the
"Traps" section before changing anything: most of it is failures that cost hours
and look like something else.

## Run it

```bash
# launch an experiment, stream it, open it — one command, nothing to start first
scripts/run_with_dashboard.sh .context/e7/one_node.py 2

# replay a finished run
python3 deploy/grafana/export.py <run_id> && python3 deploy/grafana/build_dashboard.py

# the stack alone (run_with_dashboard.sh brings it up itself)
cd deploy/grafana && docker compose up -d      # http://localhost:3001
```

`run_with_dashboard.sh` starts the compose stack if it is down, waits for
Grafana's health endpoint, seeds the run's data directory **before** the first
S3 poll, and opens the browser at
`?var-run=<id>&from=<launch>&to=now&refresh=10s`. Those URL parameters are not
decoration — see Traps #12.

Port **3001**, compose project `spot-train-grafana`. The inference platform owns
3000 — do not move onto it. Same isolation rule as us-east-1 / us-east-2
(`docs/region-split.md`).

## Architecture

```
AWS run ──► S3 (status.json, logs/*.log, profile.json at end)
                     │
              live.py (10s loop)
                     ├─ appends .live/<run_id>/status_hist.jsonl  ◄── status.json is
                     │                                     OVERWRITTEN; history exists
                     │                                     only if polled
                     ├─ export.py --live --nodes=N --src=.live/<run_id>
                     │     └─ data/<run_id>/{timeseries,occupancy,world,summary}.csv
                     └─ build_dashboard.py --run=<run_id>
                           └─ dashboards/distributed-training.json
                     │
        nginx serves ./data ──► Infinity datasource ──► Grafana (10s refresh)
```

The working dir is **per run** (`.live/<run_id>/`). A shared one was never
cleared, and `aws s3 cp` of an object that does not exist yet leaves the previous
file in place — so a new run started life holding the last run's
`profile.json`, `metrics.json` and `boot-node*.log`.

Three files own everything:

| file | role |
|---|---|
| `deploy/grafana/export.py` | run artifacts → per-run CSVs. All transforms live here. |
| `deploy/grafana/build_dashboard.py` | dashboard JSON as code. Panels, layout, run variable. |
| `deploy/grafana/live.py` | the 10s poll loop |

**Nothing is configured in the Grafana UI.** Both the dashboard and the
datasources are provisioned from disk. If you change a panel in the browser it
will be overwritten within 10s.

## Data model

`data/<run_id>/` — per-run, which is load-bearing. A single shared directory
meant the last export won, so opening the dashboard during a new run showed the
*previous* run's numbers. Stale data that looks live is worse than no data.

| file | shape | drives |
|---|---|---|
| `timeseries.csv` | boot rows + one row per logged step + a `now` row | every metric panel + all stat tiles |
| `occupancy.csv` | one row per transition, from launch | the Gantt (State Timeline) |
| `world.csv` | launch row + one row per epoch + a `now` row | world size |
| `val.csv` | one row per periodic eval | val loss, on the loss panel |
| `meta.csv` | two rows, identical | the Model tiles |
| `summary.csv` | one row | (currently unused — see Traps) |
| `runs.csv` | index | informational |

Timestamps are absolute epoch ms. Grafana is time-native; relative seconds make
the whole dashboard inert.

Every file exists from the first tick, header-only if there is nothing to say.
An absent CSV is a 404 from nginx and renders as a panel *error*; an empty one
renders as an empty panel, which is what "this run has not started yet" should
look like.

## What is authoritative

Occupancy and world size come from **supervisor epoch publications**
(`published epoch N: members [...] master=nodeM`), not from step timestamps.
This matters — see Traps #1.

**Steps are not a clock.** They only exist while training, so the boot (~3 min)
and every regrow are gaps in the step log. Everything on the wall-clock axis is
therefore anchored to the **status poll**, which exists from launch: world size,
checkpoint step, membership, the fleet counters and the cost ledger. The rule
for the boot rows and the trailing `now` row is that a value may appear only if
the status poll still asserts it — `train_loss`, `ms_per_step` and
`current_step` are left empty there, because they measure a step and no step is
happening. Carrying them forward would invent data, which is worse than a gap.

**Cost is billed from launch**, off the instance spans observed in
`status_hist.jsonl` (`nodes[].instance_id` + `aws_state`), at
`HOURLY_USD` → `ON_DEMAND_HOURLY_USD[INSTANCE_TYPE]` → 1.006. `profile.json`'s
real ledger wins once it exists.

## Traps

**1. Never infer fleet state from step timestamps.** Steps are only logged while
training, so no steps exist during a regrow: world size rendered `4→8` and
`1→8`, hiding the real `4→5→6→7→8` staircase. Worse, de-overlapping inferred
spans *erased 8 of 14 real down periods*. Both fixed by reading epoch
publications. If you add a fleet-state panel, source it the same way.

**2. `status.json` is overwritten in place.** There is no history unless
something polls and appends. `live.py` does this. If you rebuild the loop, keep
it — E5's world-size history had to be reconstructed by hand because it was
missing.

**3. The Infinity plugin has a silent browser failure.** Backend queries succeed
(`/api/ds/query` returns correct rows), health checks pass, and every panel reads
"No data" because `module.js` 404s on `react/jsx-runtime`. Seen on Grafana 11.3
*and* 12.1.0 despite 12.1.0 satisfying the plugin's declared `>=11.6`. It works
now on `grafana:latest`. **If panels go blank, check the browser console before
touching the queries.**

**4. Do not use a one-row CSV for stat tiles.** It returns tagged
`numeric-long`, which Grafana reinterprets as label/value pairs → "No data" with
a correct backend response. Adding a `time` column does not help. Stat tiles
therefore run `lastNotNull` over `timeseries.csv` columns. `summary.csv` is still
written but unused; either wire it properly or delete it.

**5. `export.py` reads SRC but writes to `data/<run_id>/`, and they are chosen
independently.** Without `--live`, SRC defaults to `.context/e5`. Running
`export.py <other_run_id>` therefore read E5's logs and filed them under the
other run's name — an 8-node trace in a 2-node run's directory, looking entirely
plausible. A guard now refuses a replay with no matching profile, but the shape
of the bug is worth remembering when adding sources.

**6. `--live` needs the DRIVER's log, not just S3.** Epoch publications
(`published epoch N: members [...] master=nodeM`) are printed by the supervisor
to the driver's log, which never reaches S3. Without `--log=` the world-size and
Gantt panels render **empty while every other panel works** — a partial failure
that is easy to miss. `run_with_dashboard.sh` passes it automatically.

**7. `url_options` is not optional on an Infinity target.** The Gantt carried
every other key its siblings do and omitted `{"method": "GET"}`. With it missing
the *backend* query is perfect — `/api/ds/query` returns the right rows, and so
does a server-side render of any other panel — while the browser's query yields
nothing and the panel reads "No data" with a pink error corner. That single
missing key is the entire reason slot occupancy never rendered, across every
run. It was misdiagnosed twice as a field-ordering problem; an `organize`
transformation pinning `time` to index 0 changes nothing, verified by rendering
with and without it. **If one panel is blank and its neighbours work, diff the
targets key by key before theorising.**

**7b. The Gantt requests `slot0..slot15` regardless of node count.** Infinity
silently drops selectors the CSV does not contain, so a 2-node run renders
exactly two rows. Reading the header at build time (the previous fix) coupled
the panel to whichever run happened to be newest, so switching `$run` to a
different node count blanked it.

**8. profile.json does not exist until the run ENDS.** Everything sourced from
it — kills, relaunches, the instance cost ledger — reads zero mid-run while the
fleet is visibly losing nodes. There are now fallbacks (fleet counts from the
epoch timeline, cost from `nodes x elapsed x rate`), but any NEW panel sourced
from `profile.*` will silently be blank live unless it gets one too.

**9. Panels below the fold do not render in a `fullPage` screenshot.** Grafana
lazy-renders. Use `?viewPanel=<id>` per panel, or the image-renderer service.

**10. `multinode-preempt` requires NODES ≥ 2.** The guard is not hit by
`--dry-run`, which skips supervision — a 1-node run fails only after launch.

**11. A series that ends at the last step looks identical to a broken panel.**
Steps stop during a regrow, so `timeseries.csv` and `world.csv` used to end two
or three minutes in the past — on a window ending at `now`, an empty right-hand
margin at exactly the moment you are watching. Both now carry a `now` row.
Anything new on the wall-clock axis needs one too.

**12. Grafana does not push a provisioned dashboard's time range or variable
into a tab that is already open.** The JSON's `time` and the run selector's
`current` apply only to a tab with no `from`/`to`/`var-run` of its own; an
existing tab keeps its own, so a new run renders as somebody else's window on
somebody else's run. Re-provisioning every 10s does not change this — `refresh`
re-runs the queries, it does not reload the dashboard. `run_with_dashboard.sh`
therefore spells the run and the window out in the URL, where they win.
Structural changes (new panels, changed columns) still need a browser reload.

**13. `export.py` returning early meant a launched run did not exist.** With no
step lines it wrote nothing at all — no `summary.json`, no `runs.csv` entry, no
CSVs — for the first ~3 minutes, so the window fell back to a rolling `now-6h`
and the run had nothing to select. The launch time is knowable with no artifacts
at all: **run ids end in the unix timestamp of the launch.**

**14. `palette-classic` assigns colour by field index, and Infinity does not
return fields in the requested order.** "durable" and "current step" came back
as two greens close enough to read as one line; nodes-lost and replacements were
identical. Series that mean different things get a pinned colour (`colors()`).

## Open items

**DONE — provisioning vs. down.** Resolved without needing slot-attributed kill
timestamps: the status doc names the replacement's instance and `aws_state` the
moment it is launched, so a slot outside `members` that has a billable box is
`provisioning`, and one that does not is genuinely `down`. Observable live, at
poll resolution, from a source that exists from launch.

**OPEN — goodput.** Removed deliberately. The old series was
`min(elapsed, budget)/elapsed`, which is pinned at 1.0 then decays
hyperbolically — the shape of the formula, not a measurement. Doing it properly
needs `trained_seconds` per tick from the supervisor (`profile.observe()` already
carries it for the W&B path). Note the useful distinction: Σ`ms_per_step` /
elapsed is **utilization** (0.853 on E5) while durable trained/elapsed is
**goodput** (0.629); the gap is redone work.

**OPEN — multi-run comparison.** The `run` variable switches between runs but
cannot overlay two. Grafana needs one query per run for that, or a repeated row.

**OPEN — S3 origin.** `live.py` polls from a laptop. For a 36h run the loop
should run beside the supervisor on the `t3.micro`, or nginx should proxy S3
directly so the dashboard has no laptop dependency.

**OPEN — annotations.** Degraded regions are drawn as a `degraded_band` series
riding the panel's own y-axis. `degraded.json` is written but not wired as real
Grafana annotations.

## Design rules worth keeping

- **Full width, stacked, one shared time axis.** Grafana syncs crosshair and
  zoom across panels, so a vertical line reads as one instant across progress,
  fleet, efficiency and cost. Side-by-side halves time resolution.
- **Two x-axes, deliberately.** Fleet/cost/progress on wall clock — *downtime is
  invisible on a step axis*, steps simply stop. Quality on `train_step`.
- **Never smooth a loss curve.** Smoothing across a rollback invents data.
- **Step time keeps raw + rolling median (window 31).** A mean is dragged by
  checkpoint spikes; the median rejects them. Residual ~11s spikes at world 1 are
  real (one node doing all 40 accumulation micro-batches), not artifacts.
- **Every series must be O(1) in node count.** Per-node state belongs in the
  Gantt, never in metric names — 30 replacements would otherwise be 30 panels.
- **No typed-in facts about the run.** Model, parameter count, dataset, context
  and global batch are parsed from the box's own log (`load_run_meta`). The
  dashboard description used to be a hand-written string and it drifted: it
  still read "8 x g5.xlarge · GPT-2 124M · OpenWebText" while a 2-node run was
  on screen. A caption cannot follow `$run`; a per-run query can.

## Val loss

`eval step S: val_loss X` is printed by rank 0 to stderr, so it rides the node
log to S3 and is live like the step lines. Requirements and traps:

- **`EVAL_INTERVAL_STEPS` must be non-zero.** `recipes/gpt2-owt.env` sets 1000;
  the short driver scripts under `.context/` override it to **0**, so a smoke
  run produces no curve at all. A 34-step test run will never reach step 1000
  either — set it to ~10 for anything you intend to eyeball.
- **It is not `metrics.json`'s `val_loss`.** That one is a different estimator
  (`estimate_loss` over `eval_iters` batches vs the deterministic full pass over
  `val.bin` here) and exists only at the end. The two are never mixed into one
  series.
- **The eval line carries no wall clock.** It is interpolated onto the
  step→time map, because the eval interval need not be a multiple of
  `LOG_INTERVAL_STEPS` (eval every 25 with logging every 10 is real).
- **Runs before the `t` field cannot be plotted.** Older logs
  (e.g. `multinode-1785795454`, which has a clean 35-point OWT curve) print step
  lines with no `t <epoch>`, so there is no step→time map and `val.csv` comes
  out empty. Those runs are readable but not replayable onto a wall clock.
- Eval is cheap: 9.25s against 3599.89s of training on that 897-step OWT run,
  **0.3%**. Cost is not a reason to keep the interval coarse.

## Verification

`435 passed, 2 skipped`. Run tests in a clean env:
`env -i PATH="$PATH" HOME="$HOME" python3 -m pytest tests/ -q` — sourcing `.env`
leaks `VCPU_QUOTA`/`NODES` into pytest and fails two unrelated tests.

`tests/test_dashboard_live.py` covers the MID-RUN state specifically: a
synthetic source with no `profile.json` and no `metrics.json`, steps that stop
before `now`, and a slot whose replacement is still booting. Every defect it
covers was invisible to a replay of a finished run. `export.py --now=<epoch>`
freezes the clock so a truncated fixture is deterministic; `--src=<dir>` points
it at one.

**Always look at the rendered dashboard before claiming a fix.** Structure
verifies programmatically; appearance does not. Several defects here were only
caught by opening the image — including the blank Gantt, whose backend query
was returning correct rows the whole time. Per-panel PNGs without a browser:

```bash
curl -o /tmp/p12.png "http://localhost:3001/render/d-solo/dist-training/x\
?panelId=12&var-run=<run_id>&from=<ms>&to=<ms>&width=1000&height=300&tz=UTC"
```
