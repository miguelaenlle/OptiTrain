# Grafana dashboard — distributed training

Local, read-only view of a training run. **Port 3001** — the inference
platform's Grafana owns 3000, and the compose project is namespaced
`spot-train-grafana` so `docker compose down` in either repo cannot stop the
other's containers.

```bash
python3 export.py <run_id>          # run artifacts -> data/*.csv
python3 build_dashboard.py          # dashboard JSON (code, not a UI artifact)
docker compose up -d                # http://localhost:3001
```

Everything is provisioned: datasource, dashboard and time range exist on first
boot with nothing to click.

## Layout

Every panel is full width and stacked, because Grafana shares one time picker,
one crosshair and one zoom across a dashboard — so a vertical line reads as a
single instant across progress, fleet, efficiency and cost at once. Panel
heights encode hierarchy: the progress hero is tall, its supporting strips short.

## Going live (pass 2)

The schema is already the live schema; only the *writer* changes.

| | pass 1 (now) | pass 2 (live) |
|---|---|---|
| who writes the CSVs | `export.py`, after the run | the supervisor, each tick |
| where they live | `./data`, served by nginx | same path on the orchestrator box, or S3 |
| dashboard time | pinned to the run's span | falls back to `now-6h` automatically |
| refresh | already `10s` | unchanged |

No dashboard, datasource or panel changes are required.

## Known issue

`grafana-image-renderer` cannot render this Grafana build — its browser 404s on
`/react/jsx-runtime`, so server-side PNG export returns blank panels. The
dashboard itself is fine; verify in a real browser. Panel queries were confirmed
directly against `/api/ds/query` (1097 rows, correct field types).
