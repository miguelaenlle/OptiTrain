# Grafana dashboard — distributed training

Port **3001** (the inference platform's Grafana owns 3000), compose project
`spot-train-grafana` so neither stack can stop the other's containers.

```bash
python3 deploy/grafana/export.py <run_id>      # run -> data/*.csv
python3 deploy/grafana/build_dashboard.py      # -> dashboards/*.json
cd deploy/grafana && docker compose up -d      # http://localhost:3001
```

## Pass 1 (now): static, embedded

Panels carry their CSV inline (`scenarioId: csv_content`) against Grafana's
**built-in** testdata datasource, so the stack needs no plugin.

That is not the original design. The intended datasource was the Infinity
plugin reading CSVs over HTTP, and its backend works — `/api/ds/query` returns
all 1097 rows. But its browser module fails to load:

```
404 Not Found, loading react/jsx-runtime from
.../yesoreyeram-infinity-datasource/module.js?_cache=3.11.2
```

Grafana's SystemJS import map does not provide `react/jsx-runtime`, on 11.3 or
on 12.1.0 — even though 12.1.0 satisfies the plugin's own declared dependency
(`>=11.6`). The failure mode is nasty: the datasource tests green, the backend
query returns correct data, and every panel silently reads "No data".

`nginx` and the Infinity provisioning are kept in place because pass 2 needs
them; flip `EMBED = False` in `build_dashboard.py` once the plugin loads.

## Pass 2 (live)

The CSV schema is already the shape a live writer appends to, so going live is
a change of *writer*, not of schema or dashboard:

1. have the supervisor append to `timeseries.csv` / `occupancy.csv` each tick
   (or write them to S3 and point the datasource at that URL)
2. set `"active": true` in `summary.json` — the dashboard's default window then
   runs start-of-run → `now` instead of pinning to the run's end
3. resolve the Infinity plugin, or swap to any datasource that reads remote CSV

`refresh: 10s` is already set, and nginx sends `Cache-Control: no-store` so a
polled CSV is never served stale.

## Why these panels

Full width, stacked, one shared time axis: Grafana syncs the crosshair and zoom
across every panel, so a vertical line reads as one instant across progress,
fleet, efficiency and cost at once. Side-by-side would halve time resolution and
break that reading. Heights encode hierarchy — the progress hero is tall, its
supporting strips short.

The Gantt is Grafana's native **State Timeline**: rows are SLOTS (fixed at the
world size, so 30 replacements do not become 30 rows) and each band is labelled
with the instance that held the slot. It holds a value until it changes, so the
CSV stores transitions only — 36h costs no more rows than 36 minutes.
