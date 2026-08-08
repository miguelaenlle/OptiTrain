#!/usr/bin/env python3
"""Generate the Grafana dashboard JSON. The dashboard is code, not a UI artifact.

Layout rule: every panel is FULL WIDTH (w=24) and stacked. Grafana shares one
time picker, one crosshair and one zoom across every panel on a dashboard, so
stacking makes a vertical line read as a single instant across progress, fleet,
efficiency and cost simultaneously. Side-by-side panels would halve the time
resolution and break that reading.

Heights encode hierarchy: the progress hero is tall, its supporting strips are
short.

    python3 deploy/grafana/build_dashboard.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).parent / "dashboards" / "distributed-training.json"
DATA = Path(__file__).parent / "data"

# EMBED mode uses Grafana's BUILT-IN testdata datasource with the CSV inlined in
# the dashboard, so the stack needs no plugin at all. That matters because the
# Infinity plugin (3.11.2) fails to load in this environment -- its module.js
# imports `react/jsx-runtime`, which Grafana's SystemJS import map does not
# provide, even on 12.1.0 which satisfies the plugin's declared dependency
# (>=11.6). The backend query works perfectly; only the browser module fails.
#
# Embedding is right for a FINISHED run (the data is fixed and small). A live
# run needs a real datasource -- see README "Going live".
EMBED = False
DS_EMBED = {"type": "grafana-testdata-datasource", "uid": "testdata-embedded"}
DS_INFINITY = {"type": "yesoreyeram-infinity-datasource", "uid": "rundata"}
DS = DS_EMBED if EMBED else DS_INFINITY
TS_URL = "http://data/timeseries.csv"
OCC_URL = "http://data/occupancy.csv"
SUM_URL = "http://data/summary.csv"

# Colour-blind-safe pairs, and consistent with the static figures in docs/ so a
# reader moving between the dashboard and the writeups sees one visual language.
GREEN, BLUE, AMBER, RED, GREY = "green", "blue", "orange", "red", "text"

_y = 0  # running grid cursor
_ids = [0]


def _next_id() -> int:
    """Stable panel ids so a single panel can be deep-linked or PNG-rendered."""
    _ids[0] += 1
    return _ids[0]


def _pos(h: int, w: int = 24, x: int = 0) -> dict:
    global _y
    p = {"h": h, "w": w, "x": x, "y": _y}
    if x + w >= 24:
        _y += h
    return p


def _cols(names: list[tuple[str, str]]) -> list[dict]:
    """Infinity column spec. `time` must be typed as epoch ms or Grafana treats
    it as a plain number and the panel silently renders with no time axis."""
    out = [{"selector": "time", "text": "time", "type": "timestamp_epoch"}]
    for sel, text in names:
        out.append({"selector": sel, "text": text, "type": "number"})
    return out


def _slice_csv(src: Path, wanted: list[tuple[str, str]]) -> str:
    """time + only the requested columns, renamed to their legend labels.

    Slicing per panel keeps each embedded blob small; inlining all 15 columns in
    all 11 panels would bloat the dashboard for no benefit.
    """
    import csv as _csv
    import io

    rows = list(_csv.DictReader(src.open()))
    buf = io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(["time", *[label for _, label in wanted]])
    for r in rows:
        vals = [r.get(sel, "") for sel, _ in wanted]
        if all(v == "" for v in vals):
            continue
        w.writerow([r["time"], *vals])
    return buf.getvalue()


def embed_target(wanted: list[tuple[str, str]], src: str = "timeseries.csv") -> list[dict]:
    return [
        {
            "refId": "A",
            "datasource": DS_EMBED,
            "scenarioId": "csv_content",
            "csvContent": _slice_csv(DATA / src, wanted),
        }
    ]


def ts_target(
    cols: list[tuple[str, str]], url: str = TS_URL, src: str = "timeseries.csv"
) -> list[dict]:
    if EMBED:
        return embed_target(cols, src)
    return [
        {
            "refId": "A",
            "datasource": DS,
            "type": "csv",
            "source": "url",
            "format": "table",
            "url": url,
            "url_options": {"method": "GET"},
            "parser": "backend",
            "columns": _cols(cols),
        }
    ]


def timeseries(title: str, cols: list[tuple[str, str]], h: int, **opts) -> dict:
    field = {
        "custom": {
            "drawStyle": opts.get("draw", "line"),
            "lineWidth": opts.get("lw", 2),
            "fillOpacity": opts.get("fill", 0),
            "showPoints": "never",
            "spanNulls": True,
            "lineInterpolation": opts.get("interp", "linear"),
            "axisLabel": opts.get("unit_label", ""),
        },
        "color": {"mode": "palette-classic"},
    }
    if "min" in opts:
        field["min"] = opts["min"]
    if "max" in opts:
        field["max"] = opts["max"]
    if "decimals" in opts:
        field["decimals"] = opts["decimals"]
    return {
        "type": "timeseries",
        "id": _next_id(),
        "title": title,
        "datasource": DS,
        "gridPos": _pos(h),
        "targets": ts_target(cols, src=opts.get("src", "timeseries.csv")),
        "fieldConfig": {"defaults": field, "overrides": opts.get("overrides", [])},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            # Shared crosshair is the whole point of stacking: hovering one panel
            # marks the same instant on every other panel.
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }


def stat(title: str, selector: str, h: int, w: int, x: int, **opts) -> dict:
    if EMBED:
        meta = json.loads((DATA / "summary.json").read_text())
        targets = [
            {
                "refId": "A",
                "datasource": DS_EMBED,
                "scenarioId": "csv_content",
                "csvContent": f"{title}\n{meta.get(selector, '')}\n",
            }
        ]
    else:
        targets = [
            {
                "refId": "A",
                "datasource": DS,
                "type": "csv",
                "source": "url",
                "format": "table",
                "url": SUM_URL,
                "parser": "backend",
                "columns": [{"selector": selector, "text": title, "type": opts.get("t", "number")}],
            }
        ]
    return {
        "type": "stat",
        "id": _next_id(),
        "title": title,
        "datasource": DS,
        "gridPos": _pos(h, w, x),
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": opts.get("unit", "none"),
                "decimals": opts.get("decimals", 0),
                "color": {"mode": "fixed", "fixedColor": opts.get("color", GREY)},
                "mappings": [],
            },
            "overrides": [],
        },
        "options": {
            "graphMode": "none",
            "textMode": "value",
            "colorMode": "value",
            "justifyMode": "center",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
    }


def state_timeline(title: str, h: int) -> dict:
    """The Gantt. Grafana's State Timeline is exactly this chart, natively:
    it holds each series' value until it changes, so transitions alone suffice
    and 36h costs no more rows than 36 minutes."""
    return {
        "type": "state-timeline",
        "id": _next_id(),
        "title": title,
        "datasource": DS,
        "gridPos": _pos(h),
        "targets": embed_target([(f"slot{i}", f"slot{i}") for i in range(8)], "occupancy.csv")
        if EMBED
        else [
            {
                "refId": "A",
                "datasource": DS,
                "type": "csv",
                "source": "url",
                "format": "table",
                "url": OCC_URL,
                "parser": "backend",
                "columns": [
                    {"selector": "time", "text": "time", "type": "timestamp_epoch"},
                    *[
                        {"selector": f"slot{i}", "text": f"slot{i}", "type": "string"}
                        for i in range(8)
                    ],
                ],
            }
        ],
        "fieldConfig": {
            "defaults": {
                "custom": {"lineWidth": 0, "fillOpacity": 90},
                "color": {"mode": "palette-classic"},
                # Four states, documented by colour. Leader is a distinct state
                # rather than an overlay because State Timeline has no marker
                # layer -- it is "training, and holds rank 0".
                "mappings": [
                    {
                        "type": "value",
                        "options": {
                            "training": {"color": "#1b7f4b", "index": 0, "text": "training"},
                            "leader": {"color": "#0b5b34", "index": 1, "text": "training (leader)"},
                            "provisioning": {
                                "color": "#3b6fd4",
                                "index": 2,
                                "text": "provisioning",
                            },
                            "down": {"color": "#b9bec4", "index": 3, "text": "down"},
                        },
                    }
                ],
            },
            "overrides": [],
        },
        "options": {
            "mergeValues": True,
            "showValue": "auto",
            "alignValue": "left",
            "rowHeight": 0.9,
            # Legend off: with one entry per instance it became a wall of
            # swatches restating labels already drawn inside each band.
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": False},
            "tooltip": {"mode": "single"},
        },
    }


def row(title: str) -> dict:
    return {
        "type": "row",
        "id": _next_id(),
        "title": title,
        "collapsed": False,
        "gridPos": _pos(1),
        "panels": [],
    }


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _time_range() -> dict:
    """Default window: START OF RUN -> end of run, or -> now while it is live.

    "All time" for this dashboard means the run, not a rolling clock: a fixed
    `now-6h` hides the beginning of a 36h run, and a rolling window on a
    FINISHED run drifts until the whole thing scrolls off. So the start is
    always the first training step, and only the end is conditional.

    Must be ISO-8601 or a relative string. Grafana's dashboard JSON does NOT
    accept epoch milliseconds -- an epoch string renders the picker as
    "Invalid date" and every panel as "No data" while the datasource works
    perfectly. That cost a full debug cycle; hence this note.
    """
    summary = Path(__file__).parent / "data" / "summary.json"
    try:
        meta = json.loads(summary.read_text())
        start = int(meta["run_start_ms"])
        pad = 30_000
        if meta.get("active"):
            return {"from": _iso(start - pad), "to": "now"}
        return {"from": _iso(start - pad), "to": _iso(int(meta["run_end_ms"]) + pad)}
    except Exception:
        # No export yet (pass 2 builds before the run starts): follow the clock.
        return {"from": "now-6h", "to": "now"}


def _annotations() -> list[dict]:
    """Degraded windows, shaded on EVERY time-series panel.

    Grafana annotations are dashboard-scoped, so one definition shades progress,
    world size, goodput and cost identically -- which is exactly what the static
    figures in docs/ do by hand.
    """
    base = [
        {
            "builtIn": 1,
            "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True,
            "hide": True,
            "name": "Annotations & Alerts",
            "type": "dashboard",
        }
    ]
    try:
        regions = json.loads((DATA / "degraded.json").read_text())
    except Exception:
        return base
    csv_rows = "time,timeEnd,text\n" + "".join(
        f"{r['time']},{r['timeEnd']},degraded\n" for r in regions
    )
    base.append(
        {
            "datasource": DS_EMBED,
            "enable": True,
            "hide": False,
            "iconColor": "rgba(255, 96, 96, 0.35)",
            "name": "degraded world",
            "target": {"scenarioId": "csv_content", "csvContent": csv_rows},
        }
    )
    return base


def build() -> dict:
    panels: list[dict] = []

    panels.append(row("Headline"))
    for i, (t, sel, kw) in enumerate(
        [
            ("Nodes lost", "nodes_lost", {"color": AMBER}),
            ("Replacements", "replacements", {"color": BLUE}),
            ("Whole-group restarts", "whole_group_restarts", {"color": RED}),
            ("Steps", "steps", {}),
            ("Cost (USD)", "usd", {"unit": "currencyUSD", "decimals": 2}),
        ]
    ):
        panels.append(stat(t, sel, 4, 24 // 5 + (1 if i < 24 % 5 else 0), i * 5, **kw))

    panels.append(row("1 · The run"))
    panels.append(
        timeseries(
            "Durable progress vs. current frontier",
            [
                ("durable_step", "durable (checkpointed)"),
                ("furthest_step", "furthest reached"),
                ("current_step", "current step"),
            ],
            10,
            unit_label="training step",
        )
    )
    panels.append(
        timeseries(
            "World size",
            [("nodes_training", "nodes training")],
            5,
            src="world.csv",
            interp="stepAfter",
            fill=15,
            unit_label="nodes",
            min=0,
        )
    )
    panels.append(
        timeseries(
            "Work at risk — steps since the last durable checkpoint",
            [("steps_at_risk", "steps at risk")],
            5,
            fill=20,
            unit_label="steps",
            min=0,
        )
    )

    panels.append(row("2 · Fleet"))
    panels.append(state_timeline("Slot occupancy — which instance held each slot", 10))
    # World size repeated directly under the Gantt: the two are read together,
    # and scrolling between them to correlate a dip with a slot going down is
    # exactly the friction stacking is supposed to remove.
    panels.append(
        timeseries(
            "World size",
            [("nodes_training", "nodes training")],
            4,
            src="world.csv",
            interp="stepAfter",
            fill=15,
            unit_label="nodes",
            min=0,
        )
    )
    panels.append(
        timeseries(
            "Nodes lost and replacements launched (cumulative)",
            [("nodes_lost", "nodes lost"), ("replacements", "replacements launched")],
            6,
            interp="stepAfter",
            unit_label="count",
            min=0,
        )
    )

    panels.append(row("3 · Efficiency"))
    panels.append(
        timeseries("Step time", [("ms_per_step", "ms per step")], 6, unit_label="ms", min=0)
    )

    panels.append(row("4 · Model quality"))
    panels.append(timeseries("Training loss", [("train_loss", "train loss")], 6, unit_label="loss"))
    panels.append(
        timeseries("Tokens processed", [("tokens_b", "tokens (billions)")], 6, fill=10, min=0)
    )

    panels.append(row("5 · Cost"))
    panels.append(timeseries("Cumulative spend", [("usd", "USD")], 6, fill=10, min=0, decimals=2))
    panels.append(timeseries("Unit cost", [("usd_per_1k", "USD per 1k steps")], 6, decimals=2))

    panels.append(row("6 · Sentinels"))
    panels.append(
        timeseries(
            "Whole-group restarts (expected: flat at zero) and degraded intervals",
            [
                ("whole_group_restarts", "whole-group restarts"),
                ("below_full_world", "below full world"),
            ],
            5,
            interp="stepAfter",
            fill=15,
            min=0,
        )
    )

    return {
        "title": "Distributed training — fault tolerance under node loss",
        "description": (
            "Model: GPT-2 124M (nanoGPT, Karpathy) · Dataset: OpenWebText · "
            "8 x g5.xlarge · global batch 480 sequences x 1024 tokens"
        ),
        "uid": "dist-training",
        "tags": ["training", "fault-tolerance"],
        "timezone": "browser",
        "editable": True,
        "graphTooltip": 1,  # shared crosshair across every panel
        # Pass 1 (static): pin to the exported run's own span, so the dashboard
        # shows data whenever it is opened rather than only within 6h of the run.
        # Pass 2 (live): no CSV on disk yet at build time -> falls back to a
        # rolling window, which is what a run in flight wants.
        "time": _time_range(),
        "timepicker": {
            "refresh_intervals": ["5s", "10s", "30s", "1m", "5m"],
            "nowDelay": "",
            "hidden": False,
        },
        "refresh": "10s",  # live-ready: pass 2 only changes who writes the CSVs
        "annotations": {"list": _annotations()},
        "schemaVersion": 39,
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
