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
DS = {"type": "yesoreyeram-infinity-datasource", "uid": "rundata"}
TS_URL = "http://data/timeseries.csv"
OCC_URL = "http://data/occupancy.csv"
SUM_URL = "http://data/summary.json"

# Colour-blind-safe pairs, and consistent with the static figures in docs/ so a
# reader moving between the dashboard and the writeups sees one visual language.
GREEN, BLUE, AMBER, RED, GREY = "green", "blue", "orange", "red", "text"

_y = 0  # running grid cursor


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


def ts_target(cols: list[tuple[str, str]], url: str = TS_URL) -> list[dict]:
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
        "title": title,
        "datasource": DS,
        "gridPos": _pos(h),
        "targets": ts_target(cols),
        "fieldConfig": {"defaults": field, "overrides": opts.get("overrides", [])},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            # Shared crosshair is the whole point of stacking: hovering one panel
            # marks the same instant on every other panel.
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }


def stat(title: str, selector: str, h: int, w: int, x: int, **opts) -> dict:
    return {
        "type": "stat",
        "title": title,
        "datasource": DS,
        "gridPos": _pos(h, w, x),
        "targets": [
            {
                "refId": "A",
                "datasource": DS,
                "type": "json",
                "source": "url",
                "format": "table",
                "url": SUM_URL,
                "parser": "backend",
                "root_selector": "",
                "columns": [{"selector": selector, "text": title, "type": opts.get("t", "number")}],
            }
        ],
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
        "title": title,
        "datasource": DS,
        "gridPos": _pos(h),
        "targets": [
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
                "mappings": [
                    {
                        "type": "value",
                        "options": {"down": {"color": "#b9bec4", "index": 0, "text": "down"}},
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
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "single"},
        },
    }


def row(title: str) -> dict:
    return {
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": _pos(1),
        "panels": [],
    }


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _time_range() -> dict:
    """Absolute span of the exported run, with 2% padding; rolling if no export.

    Must be ISO-8601. Grafana's dashboard JSON accepts relative strings
    ("now-6h") or ISO timestamps, but NOT epoch milliseconds -- an epoch string
    renders the picker as "Invalid date" and every panel as "No data", with the
    datasource itself working perfectly. Cost an entire debug cycle; hence this
    note.
    """
    csv_path = Path(__file__).parent / "data" / "timeseries.csv"
    try:
        lines = csv_path.read_text().splitlines()
        first = int(lines[1].split(",")[0])
        last = int(lines[-1].split(",")[0])
        pad = max(30_000, int((last - first) * 0.02))
        return {"from": _iso(first - pad), "to": _iso(last + pad)}
    except Exception:
        return {"from": "now-6h", "to": "now"}


def build() -> dict:
    panels: list[dict] = []

    panels.append(row("Headline"))
    for i, (t, sel, kw) in enumerate(
        [
            ("Goodput", "goodput", {"unit": "percentunit", "decimals": 1, "color": GREEN}),
            ("Nodes lost", "nodes_lost", {"color": AMBER}),
            ("Replacements", "replacements", {"color": BLUE}),
            ("Whole-group restarts", "whole_group_restarts", {"color": RED}),
            ("Steps", "steps", {}),
            ("Cost (USD)", "usd", {"unit": "currencyUSD", "decimals": 2}),
        ]
    ):
        panels.append(stat(t, sel, 4, 4, i * 4, **kw))

    panels.append(row("1 · The run"))
    panels.append(
        timeseries(
            "Durable progress vs. current frontier",
            [("durable_step", "durable step (survives failure)"), ("current_step", "current step")],
            10,
            unit_label="training step",
        )
    )
    panels.append(
        timeseries(
            "World size",
            [("nodes_training", "nodes training")],
            5,
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
        timeseries(
            "Goodput — fraction of wall clock spent training",
            [("goodput", "goodput")],
            6,
            fill=10,
            min=0,
            max=1,
            decimals=3,
        )
    )
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
        "refresh": "10s",  # live-ready: pass 2 only changes who writes the CSVs
        "schemaVersion": 39,
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
