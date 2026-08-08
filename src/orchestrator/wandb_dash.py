"""The W&B dashboard schema — one definition, used live and for republishing.

DESIGN RULES (why this file exists rather than scattered `wandb.log` calls):

1. **Namespaced keys create the layout.** W&B groups panels by the `prefix/`
   in a metric name, so choosing names IS choosing the dashboard. Five sections,
   in the order a reader should ask the questions:
       progress/  is it training?
       fleet/     is the fleet intact?
       health/    is it efficient, and did anything go wrong?
       perf/      how fast?
       cost/      what is it costing?
   No un-prefixed metrics: those land in a default dumping ground.

2. **Every series must be O(1) in node count.** A `nodes/<id>/...` key looks
   harmless at 8 nodes and produces 38 panels on a 36h run with 30 replacements.
   Per-node state goes in the slot TABLE (one panel, fixed 8 rows), never in
   metric names.

3. **Nothing is logged that a reader cannot act on.** Removed from the previous
   schema:
       t_rel          - an axis logged as a metric; renders as a meaningless
                        ever-increasing line
       tok_s          - deterministic function of ms_per_step and batch shape
       cost/hourly_usd- constant for a homogeneous fleet; a flat line
       profile/segments, profile/cost, eval/val_table
                      - tables duplicating data already charted
   That is 5 fewer panels and no lost information.

4. **Derived metrics beat raw ones for long runs.** At 72k steps a raw
   `ms_per_step` chart is a checkpoint-spike comb. We log the raw value (cheap,
   and W&B downsamples) but the *headline* is `health/goodput`, which is the
   number the whole system is judged on.

The single most important series is `progress/ckpt_step`: the furthest step that
SURVIVES a failure. It is the only progress measure that cannot be inflated by
work that is about to be rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sections in reading order. W&B sorts panel groups alphabetically, so the
# numeric prefixes are load-bearing -- they pin the order without any UI config.
PROGRESS = "1_progress"
FLEET = "2_fleet"
HEALTH = "3_health"
PERF = "4_perf"
COST = "5_cost"

SLOT_COLUMNS = ["slot", "instance", "state", "t_start", "t_end", "generation"]

# Metric leaf names are the LEGEND TEXT. W&B renders a series as
# "<run name> <full metric key>", so an abbreviation like `ckpt_step` reaches the
# reader as "multinode-preempt-1786207072-dash 1_progress/ckpt_step" and explains
# nothing. Spelling the leaf out is the only lever that reliably improves the
# legend without depending on template placeholders we cannot preview.
K_DURABLE = f"{PROGRESS}/durable step (survives failure)"
K_CURRENT = f"{PROGRESS}/current step"
K_AT_RISK = f"{PROGRESS}/steps at risk"
K_LOSS = f"{PROGRESS}/train loss"
K_VAL = f"{PROGRESS}/val loss"
K_TOKENS = f"{PROGRESS}/tokens (billions)"
K_WORLD = f"{FLEET}/nodes training"
K_DEGRADED = f"{FLEET}/below full world"
K_LOST = f"{FLEET}/nodes lost (cumulative)"
K_REPL = f"{FLEET}/replacements launched (cumulative)"
K_GOODPUT = f"{HEALTH}/goodput"
K_WGR = f"{HEALTH}/whole-group restarts"
K_MS = f"{PERF}/ms per step"
K_USD = f"{COST}/spend (USD)"
K_USD_1K = f"{COST}/USD per 1k steps"
K_GANTT = f"{FLEET}/gantt"


@dataclass
class RunState:
    """Everything the dashboard needs at one tick. Populated by the supervisor
    live, or replayed from a finished run's artifacts."""

    t_rel: float = 0.0
    step: int = 0
    ckpt_step: int = -1
    loss: float | None = None
    val_loss: float | None = None
    ms_per_step: float | None = None
    world_size: int | None = None
    target_world: int | None = None
    nodes_lost: int = 0
    replacements: int = 0
    whole_group_restarts: int = 0
    trained_seconds: float = 0.0
    usd: float | None = None
    tokens: int | None = None
    slots: list[list] = field(default_factory=list)


def define_axes(wb) -> None:
    """Pin every chart's x-axis.

    Two axes matter and they answer different questions, so both are defined:
      train_step - progress against work done; resume overlaps are visible
      t_rel      - progress against wall clock; DOWNTIME GAPS are visible

    t_rel is an AXIS here, never a logged metric. Logging it as a metric (the
    previous behaviour) produced a rising diagonal line panel that no one reads.
    """
    wb.define_metric("train_step")
    wb.define_metric("t_rel")
    wb.define_metric("*", step_metric="train_step")


def tick_payload(s: RunState) -> dict:
    """The per-tick metrics. Keep this small; it runs tens of thousands of times."""
    out: dict[str, float] = {
        "train_step": s.step,
        "t_rel": s.t_rel,
        K_CURRENT: s.step,
    }
    if s.ckpt_step >= 0:
        out[K_DURABLE] = s.ckpt_step
        # How far ahead of durable progress we are running: this is exactly what
        # a failure would discard right now. Reads as a sawtooth, and each tooth
        # is the rollback that a kill at that instant would have cost.
        out[K_AT_RISK] = max(0, s.step - s.ckpt_step)
    if s.loss is not None:
        out[K_LOSS] = s.loss
    if s.tokens is not None:
        out[K_TOKENS] = round(s.tokens / 1e9, 6)

    if s.world_size is not None:
        out[K_WORLD] = s.world_size
        if s.target_world:
            out[K_DEGRADED] = 1 if s.world_size < s.target_world else 0
    out[K_LOST] = s.nodes_lost
    out[K_REPL] = s.replacements

    # Goodput is the headline: fraction of wall clock actually spent training.
    # trained_seconds is checkpoint-carried, so downtime can never inflate it.
    if s.t_rel > 0 and s.trained_seconds:
        out[K_GOODPUT] = round(s.trained_seconds / s.t_rel, 4)
    out[K_WGR] = s.whole_group_restarts

    if s.ms_per_step:
        out[K_MS] = s.ms_per_step

    if s.usd is not None:
        out[K_USD] = round(s.usd, 4)
        if s.step > 0:
            out[K_USD_1K] = round(s.usd / s.step * 1000, 4)
    return out


def val_payload(step: int, val_loss: float) -> dict:
    return {"train_step": step, K_VAL: val_loss}


def slot_rows(occupancy: list[dict]) -> list[list]:
    """Rows for the live Gantt: ONE ROW PER SLOT-OCCUPANCY, not per instance.

    A world of N has exactly N slots for the life of the run; instances are
    transient occupants. Keying the chart on the slot is what makes it O(N)
    instead of O(N + failures) -- 8 rows at 36h with 30 replacements, versus the
    38-row wall the per-instance version produces.
    """
    rows = []
    for o in occupancy:
        rows.append(
            [
                f"slot{o['slot']}",
                o.get("instance", "?"),
                o.get("state", "train"),
                round(float(o.get("t_start", 0.0)), 1),
                round(float(o.get("t_end", 0.0)), 1),
                int(o.get("generation", 0)),
            ]
        )
    return rows


_STATE_COLORS = {
    # Chosen for lightness separation, not hue alone, so the three states stay
    # distinguishable under deuteranopia/protanopia and in greyscale print.
    "prov": ("#3b6fd4", "provisioning"),
    "train": ("#1b7f4b", "training"),
    "down": ("#b9bec4", "down / replaced"),
}


def gantt_html(
    occupancy: list[dict],
    target_world: int,
    now_s: float,
    events: list[dict] | None = None,
) -> str:
    """A scrollable SVG Gantt, as a self-contained HTML string.

    ONE ROW PER INSTANCE (node0, node0-r1, ...), matching the matplotlib figure
    in the writeups: an operator debugging a recovery needs to know *which box*
    took over a slot, and collapsing to slots throws that away.

    Scale is handled by SCROLLING, not by compressing. Pixels-per-second is
    fixed, so a 250s recovery is the same width at hour 1 and hour 35, and the
    canvas simply grows; the container scrolls in BOTH axes, so a 36h run with
    forty instances loses no interval and no row. Compressing 36h into one
    viewport would make each recovery ~3px and effectively delete it.
    """
    px_per_s = 0.06
    row_h, pad, label_w, axis_h = 22, 10, 96, 22
    rows = sorted(occupancy, key=lambda o: (int(o["slot"]), int(o.get("generation", 0))))
    n = max(1, len(rows))
    width = max(760, int(now_s * px_per_s) + label_w + 60)
    height = n * row_h + 2 * pad + axis_h

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="11">'
    ]
    base_y = n * row_h + pad

    # time gridlines first so bars draw over them
    span = max(now_s, 1.0)
    tick = 3600 if span > 7200 else (600 if span > 1200 else 120)
    t = tick
    while t <= span:
        x = label_w + t * px_per_s
        lbl = f"{t / 3600:g}h" if tick == 3600 else f"{t / 60:g}m"
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{base_y}" '
            f'stroke="#dadce0" stroke-dasharray="2,3"/>'
            f'<text x="{x + 3:.1f}" y="{base_y + 15}" fill="#5f6368">{lbl}</text>'
        )
        t += tick

    for i, seg in enumerate(rows):
        y = pad + i * row_h
        gen = int(seg.get("generation", 0))
        name = seg.get("instance") or f"node{seg['slot']}"
        parts.append(
            f'<text x="4" y="{y + 14}" fill="#3c4043">{name}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{width - label_w - 12}" '
            f'height="{row_h - 6}" fill="#f6f7f8" rx="3"/>'
        )
        x0 = label_w + seg["t_start"] * px_per_s
        w = max(2.0, (seg["t_end"] - seg["t_start"]) * px_per_s)
        color, _ = _STATE_COLORS.get(seg.get("state", "train"), _STATE_COLORS["train"])
        dur = seg["t_end"] - seg["t_start"]
        parts.append(
            f'<rect x="{x0:.1f}" y="{y + 3}" width="{w:.1f}" height="{row_h - 6}" '
            f'fill="{color}" rx="2"><title>{name} (gen {gen}) · '
            f"{dur:.0f}s from {seg['t_start']:.0f}s to {seg['t_end']:.0f}s"
            f"</title></rect>"
        )
        if w > 34:
            parts.append(f'<text x="{x0 + 4:.1f}" y="{y + 14}" fill="#ffffff">{dur:.0f}s</text>')

    # control-plane markers, drawn on top of every row
    for ev in events or []:
        x = label_w + float(ev.get("t", 0.0)) * px_per_s
        kind = ev.get("kind")
        if kind == "kill":
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{base_y}" '
                f'stroke="#d93025" stroke-width="1.5" opacity="0.85">'
                f"<title>node lost at {ev.get('t', 0):.0f}s</title></line>"
            )
        elif kind == "relaunch":
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{base_y}" '
                f'stroke="#f9a825" stroke-width="1" stroke-dasharray="3,2" opacity="0.8">'
                f"<title>replacement launched at {ev.get('t', 0):.0f}s</title></line>"
            )
    parts.append("</svg>")

    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px">'
        f'<span style="width:11px;height:11px;background:{c};border-radius:2px;'
        f'display:inline-block;margin-right:6px"></span>{label}</span>'
        for c, label in _STATE_COLORS.values()
    )
    swatches += (
        '<span style="display:inline-flex;align-items:center;margin-right:16px">'
        '<span style="width:2px;height:12px;background:#d93025;display:inline-block;'
        'margin-right:6px"></span>node lost</span>'
        '<span style="display:inline-flex;align-items:center">'
        '<span style="width:2px;height:12px;background:#f9a825;display:inline-block;'
        'margin-right:6px"></span>replacement launched</span>'
    )
    return (
        '<div style="font:12px ui-sans-serif,system-ui,sans-serif;color:#3c4043">'
        f'<div style="margin:0 0 8px 2px">{swatches}</div>'
        '<div style="overflow:auto;max-height:560px;border:1px solid #e3e6e8;'
        f'border-radius:6px;background:#fff">{"".join(parts)}</div>'
        '<div style="margin-top:6px;color:#5f6368">'
        f"{n} instances across {target_world} slots · scroll horizontally for time, "
        "vertically for instances — the time scale is fixed, so a recovery is the "
        "same width anywhere in the run.</div></div>"
    )


def header_markdown(cfg: dict, s: RunState, wall_s: float) -> str:
    """The headline panel: what this run is, and the six numbers that judge it."""
    goodput = (s.trained_seconds / wall_s * 100) if wall_s else 0.0
    unit = "$%.2f" % (s.usd / s.step * 1000) if s.usd and s.step else "—"
    row = " | ".join(
        [
            "",
            f"**{goodput:.1f}%**",
            str(s.nodes_lost),
            str(s.replacements),
            f"**{s.whole_group_restarts}**",
            f"{s.step:,}",
            unit,
            "",
        ]
    ).strip()
    return f"""## Distributed training — fault tolerance under node loss

**Model:** GPT-2 124M (nanoGPT, Karpathy) &nbsp;·&nbsp; **Dataset:** OpenWebText
&nbsp;·&nbsp; **Nodes:** {cfg.get("nodes", "?")} × {cfg.get("instance_type", "?")}
&nbsp;·&nbsp; **Global batch:** {cfg.get("global_batch", "?")} sequences ×
{cfg.get("block_size", "?")} tokens

| goodput | nodes lost | replacements | whole-group restarts | steps | $ / 1k steps |
|---|---|---|---|---|---|
| {row} |

Goodput is the fraction of billed wall clock spent actually training; the training
budget is carried inside the checkpoint, so downtime can never inflate it.
Whole-group restarts must stay at zero — one means healthy survivors were discarded.
"""


def summary(s: RunState, wall_s: float, metrics: dict) -> dict:
    """End-of-run scalars. These populate the runs TABLE, which is the run-history
    and cross-run comparison view -- so only decision-grade numbers belong here."""
    out = {
        "goodput": round(s.trained_seconds / wall_s, 4) if wall_s else None,
        "nodes_lost": s.nodes_lost,
        "replacements": s.replacements,
        "whole_group_restarts": s.whole_group_restarts,
        "wall_hours": round(wall_s / 3600, 3),
        "trained_hours": round(s.trained_seconds / 3600, 3),
        "usd": round(s.usd, 2) if s.usd is not None else None,
    }
    for k in ("steps", "val_loss", "resumed", "restart_count", "effective_global_batch"):
        if k in metrics:
            out[k] = metrics[k]
    if s.usd and metrics.get("steps"):
        out["usd_per_1k_steps"] = round(s.usd / metrics["steps"] * 1000, 4)
    if s.tokens:
        out["tokens_billions"] = round(s.tokens / 1e9, 4)
    return {k: v for k, v in out.items() if v is not None}
