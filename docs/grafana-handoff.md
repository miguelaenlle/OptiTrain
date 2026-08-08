# Handoff — Grafana training dashboard

Everything below is committed and working unless marked **OPEN**. Read the
"Traps" section before changing anything: most of it is failures that cost hours
and look like something else.

## Run it

```bash
cd deploy/grafana && docker compose up -d      # http://localhost:3001

# replay a finished run
python3 deploy/grafana/export.py <run_id> && python3 deploy/grafana/build_dashboard.py

# launch an experiment AND stream it, one command
scripts/run_with_dashboard.sh .context/e7/one_node.py 2
```

Port **3001**, compose project `spot-train-grafana`. The inference platform owns
3000 — do not move onto it. Same isolation rule as us-east-1 / us-east-2
(`docs/region-split.md`).

## Architecture

```
AWS run ──► S3 (status.json, logs/*.log, profile.json at end)
                     │
              live.py (10s loop)
                     ├─ appends .live/status_hist.jsonl   ◄── status.json is OVERWRITTEN;
                     │                                        history exists only if polled
                     ├─ export.py --live --nodes=N
                     │     └─ data/<run_id>/{timeseries,occupancy,world,summary}.csv
                     └─ build_dashboard.py
                           └─ dashboards/distributed-training.json
                     │
        nginx serves ./data ──► Infinity datasource ──► Grafana (10s refresh)
```

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
| `timeseries.csv` | one row per logged step | every metric panel + all stat tiles |
| `occupancy.csv` | one row per transition | the Gantt (State Timeline) |
| `world.csv` | one row per epoch + a tail row | world size |
| `summary.csv` | one row | (currently unused — see Traps) |
| `runs.csv` | index | informational |

Timestamps are absolute epoch ms. Grafana is time-native; relative seconds make
the whole dashboard inert.

## What is authoritative

Occupancy and world size come from **supervisor epoch publications**
(`published epoch N: members [...] master=nodeM`), not from step timestamps.
This matters — see Traps #1.

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

**7. The Gantt's slot columns are baked at BUILD time from the newest run's
`occupancy.csv`.** They used to be hardcoded `range(8)`, so a 2-node run
requested six nonexistent columns and the State Timeline rendered *nothing* — a
blank panel, not a partial one. It now reads the header, but switching to a run
with a DIFFERENT node count still needs `build_dashboard.py` re-run. Making the
columns follow `$run` at query time is the proper fix.

**8. profile.json does not exist until the run ENDS.** Everything sourced from
it — kills, relaunches, the instance cost ledger — reads zero mid-run while the
fleet is visibly losing nodes. There are now fallbacks (fleet counts from the
epoch timeline, cost from `nodes x elapsed x rate`), but any NEW panel sourced
from `profile.*` will silently be blank live unless it gets one too.

**9. Panels below the fold do not render in a `fullPage` screenshot.** Grafana
lazy-renders. Use `?viewPanel=<id>` per panel, or the image-renderer service.

**10. `multinode-preempt` requires NODES ≥ 2.** The guard is not hit by
`--dry-run`, which skips supervision — a 1-node run fails only after launch.

## Open items

**OPEN — provisioning vs. down.** `down` currently never renders. Replacements
launch ~10s before the shrink epoch appears in the polled status, so at epoch
resolution the whole gap is boot time. Showing a real `down` band needs
**slot-attributed kill timestamps**; `profile.json`'s `kill` events carry no node
index. The supervisor log has `terminated node N` lines in order — zipping those
with the kill event timestamps is the likely fix.

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

## Verification

`420 passed, 2 skipped`. Run tests in a clean env:
`env -i PATH="$PATH" HOME="$HOME" python3 -m pytest tests/ -q` — sourcing `.env`
leaks `VCPU_QUOTA`/`NODES` into pytest and fails two unrelated tests.

Always look at the rendered dashboard before claiming a fix. Structure verifies
programmatically; appearance does not. Two defects in this project were only
caught by opening the image.
