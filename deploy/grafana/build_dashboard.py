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
import sys
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
TS_URL = "http://data/${run}/timeseries.csv"
OCC_URL = "http://data/${run}/occupancy.csv"
SUM_URL = "http://data/${run}/summary.csv"

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


def _cols(names: list[tuple[str, str]], types: str = "number") -> list[dict]:
    """Infinity column spec. `time` must be typed as epoch ms or Grafana treats
    it as a plain number and the panel silently renders with no time axis."""
    out = [{"selector": "time", "text": "time", "type": "timestamp_epoch"}]
    for sel, text in names:
        out.append({"selector": sel, "text": text, "type": types})
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
    cols: list[tuple[str, str]],
    url: str = TS_URL,
    src: str = "timeseries.csv",
    ref: str = "A",
    types: str = "number",
) -> list[dict]:
    if EMBED:
        return embed_target(cols, src)
    url = f"http://data/${{run}}/{src}"
    return [
        {
            "refId": ref,
            "datasource": DS,
            "type": "csv",
            "source": "url",
            "format": "table",
            "url": url,
            "url_options": {"method": "GET"},
            "parser": "backend",
            "columns": _cols(cols, types),
        }
    ]


def colors(*pairs: tuple[str, str]) -> list[dict]:
    """Pin a colour per series by legend label.

    palette-classic assigns by field index, and Infinity does not return fields
    in the requested order -- so a three-series panel drew "durable" and
    "current step" in two greens close enough to be one line, and nodes-lost and
    replacements identically. Naming the colour also lets a series keep the same
    meaning across panels: replacements are blue in the Gantt, in the headline
    tile and in the fleet chart.
    """
    return [
        {
            "matcher": {"id": "byName", "options": label},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": colour}}],
        }
        for label, colour in pairs
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
        # `extra` adds a SECOND query against a different file. Series sampled on
        # different clocks cannot share one CSV without inventing rows for the
        # sparser one -- val loss is logged every eval_interval_steps, which need
        # not line up with the step rows at all.
        "targets": ts_target(cols, src=opts.get("src", "timeseries.csv"))
        + [
            t
            for i, (esrc, ecols) in enumerate(opts.get("extra", []))
            for t in ts_target(ecols, src=esrc, ref=chr(ord("B") + i))
        ],
        "fieldConfig": {"defaults": field, "overrides": opts.get("overrides", [])},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            # Shared crosshair is the whole point of stacking: hovering one panel
            # marks the same instant on every other panel.
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }


def stat(title: str, selector: str, h: int, w: int, x: int, **opts) -> dict:
    """Headline tile: lastNotNull over a column of timeseries.csv.

    Deliberately NOT a separate summary file. A one-row CSV comes back tagged
    "numeric-long", which Grafana reinterprets as label/value pairs and renders
    as "No data" even though the backend returns the right number -- adding a
    time column did not change that. Reusing the series every other panel
    already renders removes the whole failure mode.
    """
    return {
        "type": "stat",
        "id": _next_id(),
        "title": title,
        "datasource": DS,
        "gridPos": _pos(h, w, x),
        "targets": ts_target(
            [(selector, title)],
            src=opts.get("src", "timeseries.csv"),
            types=opts.get("type", "number"),
        ),
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
            # A string tile needs the smaller font: "GPT-2 · 12L/12H/768d" at the
            # numeric tile size overflows into an ellipsis.
            "textMode": "value",
            "colorMode": "value",
            "justifyMode": "center",
            # `fields: ""` means NUMERIC FIELDS ONLY -- a string column is dropped
            # and the tile reads "No data" while the backend returns the value.
            # Naming the field explicitly is what lets a text tile render.
            "reduceOptions": {
                "calcs": ["lastNotNull"],
                "fields": title if opts.get("type") == "string" else "",
                "values": False,
            },
        },
    }


# The Gantt asks for MAX_SLOTS slot columns regardless of the run's node count.
# Infinity silently drops selectors the CSV does not contain, so a 2-node run
# renders exactly two rows -- verified against 8 requested columns on a 2-column
# occupancy.csv. This is what makes the panel node-count agnostic: reading the
# header at BUILD time coupled the panel to whichever run happened to be newest,
# so switching $run to a run with a different node count blanked the chart.
MAX_SLOTS = 16


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
        "targets": embed_target(
            [(f"slot{i}", f"slot{i}") for i in range(MAX_SLOTS)], "occupancy.csv"
        )
        if EMBED
        else [
            {
                "refId": "A",
                "datasource": DS,
                "type": "csv",
                "source": "url",
                "format": "table",
                "url": OCC_URL,
                # url_options is NOT optional. Without it the BACKEND query still
                # returns correct rows (/api/ds/query is fine, and so is a
                # server-side render of any other panel), but the browser's query
                # yields nothing and the panel reads "No data" with an error
                # corner. Every timeseries target here carries it; this one did
                # not, which is the whole reason the Gantt never rendered once.
                "url_options": {"method": "GET"},
                "parser": "backend",
                "columns": [
                    {"selector": "time", "text": "time", "type": "timestamp_epoch"},
                    *[
                        {"selector": f"slot{i}", "text": f"slot{i}", "type": "string"}
                        for i in range(MAX_SLOTS)
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
        # Infinity returns fields as [slot0, slot1, ..., time], with time LAST.
        # That is fine: State Timeline finds the time field wherever it sits --
        # confirmed by rendering the panel with and without an `organize`
        # transformation pinning time to index 0, which changed nothing. Field
        # order was the wrong suspect for the blank Gantt; url_options above was
        # the cause.
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


def _launch_ms(run_id: str) -> int | None:
    """Launch time straight from the run id, which ends in a unix timestamp.

    Knowing this without reading any artifact is what lets the default window
    open on the run the moment it is launched, rather than after the first
    training step lands three minutes later.
    """
    tail = run_id.rsplit("-", 1)[-1]
    return int(tail) * 1000 if tail.isdigit() and len(tail) == 10 else None


def _runs() -> list[str]:
    """Run ids with exported data, newest first (ids embed a unix timestamp)."""
    if not DATA.exists():
        return []
    # Every run directory is listed, including ones with no data yet. A run that
    # is still booting SHOULD be selectable -- hiding it makes a launched run
    # look like it never started, which is exactly the confusion this variable
    # exists to prevent. An empty run renders empty panels, which is honest.
    return sorted(
        (d.name for d in DATA.iterdir() if d.is_dir()),
        key=lambda n: n.rsplit("-", 1)[-1],
        reverse=True,
    )


def _default_run() -> str:
    """The run the dashboard opens on. --run= pins it to the LIVE run; without
    it, the newest exported run wins. Replaying an old run mid-flight would
    otherwise steal the default out from under a run in progress."""
    runs = _runs()
    pinned = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--run=")), None)
    if pinned:
        return pinned
    return runs[0] if runs else ""


def _run_variable() -> dict:
    """The run selector.

    Without this every export overwrote one shared data/ directory, so opening
    the dashboard during a new run showed the PREVIOUS run's numbers -- worse
    than empty, because stale data looks live. It is also the thing that makes
    Grafana worth more than a static PNG: switching and comparing runs.

    A CUSTOM variable (baked at build time) rather than a query variable: the
    list changes only when a run is exported, and it avoids depending on a
    second datasource query path.
    """
    runs = _runs()
    cur = _default_run()
    if cur and cur not in runs:
        runs = [cur, *runs]
    return {
        "name": "run",
        "label": "Run",
        "type": "custom",
        "query": ",".join(runs),
        "options": [{"text": r, "value": r, "selected": r == cur} for r in runs],
        "current": {"text": cur, "value": cur, "selected": True},
        "includeAll": False,
        "multi": False,
    }


def _time_range() -> dict:
    """Default window: START OF RUN -> end of run, or -> now while it is live.

    "All time" for this dashboard means the run, not a rolling clock: a fixed
    `now-6h` hides the beginning of a 36h run, and a rolling window on a
    FINISHED run drifts until the whole thing scrolls off. So the start is
    always the run's LAUNCH, and only the end is conditional.

    The launch is read from the run id when summary.json is not there yet, which
    is the whole first stretch of a run. Falling back to `now-6h` in that window
    is what made a fresh dashboard open on a rolling tail instead of on the run.

    Must be ISO-8601 or a relative string. Grafana's dashboard JSON does NOT
    accept epoch milliseconds -- an epoch string renders the picker as
    "Invalid date" and every panel as "No data" while the datasource works
    perfectly. That cost a full debug cycle; hence this note.
    """
    pad = 30_000
    run = _default_run()
    meta = {}
    if run:
        try:
            meta = json.loads((DATA / run / "summary.json").read_text())
        except Exception:
            meta = {}
    start = meta.get("run_start_ms") or _launch_ms(run) if run else None
    if not start:
        return {"from": "now-6h", "to": "now"}
    # A live run runs to "now" so the window follows the fleet; a finished one
    # pins to its own end so there is no dead trailing space. Unknown (no
    # summary yet) means the run was just launched, which is live.
    if meta.get("active", True):
        return {"from": _iso(int(start) - pad), "to": "now"}
    return {"from": _iso(int(start) - pad), "to": _iso(int(meta["run_end_ms"]) + pad)}


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
    # Per-run, like every other artifact. This read was left pointing at the
    # flat path when exports were namespaced, so the degraded shading has been
    # silently absent ever since -- a missing file returns `base` and no panel
    # complains.
    runs = _runs()

    def _regions(name: str) -> list[dict]:
        try:
            return json.loads((DATA / runs[0] / name).read_text())
        except Exception:
            return []

    for fname, label, colour in (
        ("degraded.json", "degraded", "rgba(255, 96, 96, 0.35)"),
        # A DISTINCT colour, deliberately. "the world is short a node" and
        # "nobody is steering" are different failures, and a shared colour would
        # hide the more serious one inside the more common one.
        ("control_plane_down.json", "control plane down", "rgba(140, 90, 220, 0.45)"),
    ):
        regions = _regions(fname)
        if not regions:
            continue
        rows = "time,timeEnd,text\n" + "".join(
            f"{r['time']},{r['timeEnd']},{label}\n" for r in regions
        )
        base.append(
            {
                "datasource": DS_EMBED,
                "enable": True,
                "hide": False,
                "iconColor": colour,
                "name": label,
                "target": {"scenarioId": "csv_content", "csvContent": rows},
            }
        )
    return base


def build() -> dict:
    panels: list[dict] = []

    # What is being trained, as the RUN reported it. This used to be a typed-in
    # dashboard description that still read "8 x g5.xlarge" while a 2-node run
    # was on screen -- a caption cannot follow $run, so it drifts silently. These
    # tiles are per-run queries, so switching runs re-reads the facts.
    panels.append(row("Model"))
    for i, (t, sel, kw) in enumerate(
        [
            ("Model", "model", {"type": "string"}),
            ("Parameters", "params_exact", {"unit": "short", "decimals": 2}),
            ("Dataset", "dataset", {"type": "string"}),
            ("Context", "ctx", {"unit": "none"}),
            ("Global batch", "global_batch", {"unit": "none"}),
        ]
    ):
        panels.append(
            stat(t, sel, 3, 24 // 5 + (1 if i < 24 % 5 else 0), i * 5, src="meta.csv", **kw)
        )

    panels.append(row("Headline"))
    for i, (t, sel, kw) in enumerate(
        [
            ("Nodes lost", "nodes_lost", {"color": AMBER}),
            ("Replacements", "replacements", {"color": BLUE}),
            ("Whole-group restarts", "whole_group_restarts", {"color": RED}),
            ("Steps", "furthest_step", {}),
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
            overrides=colors(
                ("durable (checkpointed)", GREEN),
                ("furthest reached", BLUE),
                ("current step", AMBER),
            ),
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
            # Same colours as the two headline tiles above.
            overrides=colors(("nodes lost", AMBER), ("replacements launched", BLUE)),
        )
    )

    panels.append(row("3 · Efficiency"))
    panels.append(
        timeseries("Step time", [("ms_per_step", "ms per step")], 6, unit_label="ms", min=0)
    )

    panels.append(row("4 · Model quality"))
    panels.append(
        timeseries(
            "Loss — training and validation",
            [("train_loss", "train loss")],
            6,
            unit_label="loss",
            # Val loss rides the SAME panel: the two are the same quantity in the
            # same units, and the gap between them is the thing worth reading. It
            # is a separate query because it is sampled every
            # eval_interval_steps, not every step -- sparse points joined into a
            # line, which is exactly how the underlying series behaves.
            extra=[("val.csv", [("val_loss", "val loss")])],
            overrides=colors(("train loss", BLUE), ("val loss", AMBER))
            + [
                {
                    "matcher": {"id": "byName", "options": "val loss"},
                    "properties": [
                        # Sparse by nature -- every eval_interval_steps, not every
                        # step. Without markers a 3-point series reads as a
                        # straight line with no indication of where it was
                        # actually measured.
                        {"id": "custom.showPoints", "value": "always"},
                        {"id": "custom.pointSize", "value": 5},
                        # Grafana prefixes a series with its query refId once a
                        # panel has more than one query ("B val loss"). An
                        # explicit displayName suppresses that.
                        {"id": "displayName", "value": "val loss"},
                    ],
                },
                {
                    "matcher": {"id": "byName", "options": "train loss"},
                    "properties": [{"id": "displayName", "value": "train loss"}],
                },
            ],
        )
    )
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
            overrides=colors(("whole-group restarts", RED), ("below full world", AMBER)),
        )
    )
    # Control-plane liveness as a HARD series, not only a shaded band. During an
    # outage the fleet panels merely stop changing, which reads as "the fleet was
    # stable" -- the opposite of the truth. This drops to 0 and says so, and it
    # is greppable in the CSV rather than only visible by eye.
    panels.append(
        timeseries(
            "Supervisor up (0 = control plane not writing; training continues regardless)",
            [("supervisor_up", "supervisor up")],
            4,
            interp="stepAfter",
            fill=25,
            min=0,
            max=1,
            decimals=0,
            overrides=colors(("supervisor up", "purple")),
        )
    )

    return {
        "title": "Distributed training — fault tolerance under node loss",
        # No model/dataset caption here on purpose. A static string cannot follow
        # $run, and this one drifted: it claimed "8 x g5.xlarge, GPT-2 124M,
        # OpenWebText" regardless of what was actually selected. The Model row
        # carries those facts now, parsed from the run's own logs.
        "description": (
            "Fault tolerance under node loss. Model facts are per-run, read from the box's log."
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
        "templating": {"list": [_run_variable()]},
        "schemaVersion": 39,
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
