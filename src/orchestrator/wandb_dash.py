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
        f"{PROGRESS}/step": s.step,
    }
    if s.ckpt_step >= 0:
        out[f"{PROGRESS}/ckpt_step"] = s.ckpt_step
        # How far ahead of durable progress we are running: this is exactly what
        # a failure would discard right now. Reads as a sawtooth, and each tooth
        # is the rollback that a kill at that instant would have cost.
        out[f"{PROGRESS}/at_risk_steps"] = max(0, s.step - s.ckpt_step)
    if s.loss is not None:
        out[f"{PROGRESS}/loss"] = s.loss
    if s.tokens is not None:
        out[f"{PROGRESS}/tokens_billions"] = round(s.tokens / 1e9, 6)

    if s.world_size is not None:
        out[f"{FLEET}/world_size"] = s.world_size
        if s.target_world:
            out[f"{FLEET}/degraded"] = 1 if s.world_size < s.target_world else 0
    out[f"{FLEET}/nodes_lost"] = s.nodes_lost
    out[f"{FLEET}/replacements"] = s.replacements

    # Goodput is the headline: fraction of wall clock actually spent training.
    # trained_seconds is checkpoint-carried, so downtime can never inflate it.
    if s.t_rel > 0 and s.trained_seconds:
        out[f"{HEALTH}/goodput"] = round(s.trained_seconds / s.t_rel, 4)
    out[f"{HEALTH}/whole_group_restarts"] = s.whole_group_restarts

    if s.ms_per_step:
        out[f"{PERF}/ms_per_step"] = s.ms_per_step

    if s.usd is not None:
        out[f"{COST}/usd"] = round(s.usd, 4)
        if s.step > 0:
            out[f"{COST}/usd_per_1k_steps"] = round(s.usd / s.step * 1000, 4)
    return out


def val_payload(step: int, val_loss: float) -> dict:
    return {"train_step": step, f"{PROGRESS}/val_loss": val_loss}


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
    "train": ("#1b7f4b", "training"),
    "prov": ("#3b6fd4", "provisioning"),
    "down": ("#b9bec4", "down / replaced"),
}


def gantt_html(occupancy: list[dict], target_world: int, now_s: float) -> str:
    """A scrollable SVG Gantt of slot occupancy, as a self-contained HTML string.

    Horizontal scroll rather than time-compression is deliberate: squeezing 36h
    into one viewport makes a 250s recovery ~3px wide and effectively deletes it.
    Here the pixels-per-second rate is FIXED, the canvas grows with the run, and
    the container scrolls -- so a recovery is the same size at hour 1 and hour 35
    and no interval is ever aggregated away.

    Rows are SLOTS (always ``target_world`` of them), not instances, so the chart
    height is constant no matter how many replacements a run burns through.
    """
    px_per_s = 0.05  # 36h -> ~6500px of scrollable canvas
    row_h, pad, label_w = 26, 8, 78
    width = max(600, int(now_s * px_per_s) + label_w + 40)
    height = target_world * row_h + 2 * pad + 26

    by_slot: dict[int, list[dict]] = {}
    for o in occupancy:
        by_slot.setdefault(int(o["slot"]), []).append(o)

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="11">'
    ]
    for i in range(target_world):
        y = pad + i * row_h
        parts.append(
            f'<text x="4" y="{y + 16}" fill="#3c4043">slot{i}</text>'
            f'<rect x="{label_w}" y="{y + 4}" width="{width - label_w - 8}" '
            f'height="{row_h - 8}" fill="#f1f3f4" rx="3"/>'
        )
        for seg in sorted(by_slot.get(i, []), key=lambda s: s["t_start"]):
            x0 = label_w + seg["t_start"] * px_per_s
            w = max(2.0, (seg["t_end"] - seg["t_start"]) * px_per_s)
            color, _ = _STATE_COLORS.get(seg.get("state", "train"), _STATE_COLORS["train"])
            parts.append(
                f'<rect x="{x0:.1f}" y="{y + 4}" width="{w:.1f}" height="{row_h - 8}" '
                f'fill="{color}" rx="2"><title>{seg.get("instance", "")} · '
                f"{seg.get('state', 'train')} · "
                f"{seg['t_start']:.0f}-{seg['t_end']:.0f}s"
                f"</title></rect>"
            )
    # hour gridlines + legend
    base_y = target_world * row_h + pad
    for hour in range(1, int(now_s // 3600) + 1):
        x = label_w + hour * 3600 * px_per_s
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{base_y}" '
            f'stroke="#c8ccd0" stroke-dasharray="2,3"/>'
            f'<text x="{x + 3:.1f}" y="{base_y + 14}" fill="#5f6368">{hour}h</text>'
        )
    parts.append("</svg>")
    legend = " ".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px">'
        f'<span style="width:11px;height:11px;background:{c};border-radius:2px;'
        f'display:inline-block;margin-right:5px"></span>{label}</span>'
        for c, label in _STATE_COLORS.values()
    )
    return (
        '<div style="font:13px ui-sans-serif,system-ui,sans-serif;color:#3c4043">'
        f'<div style="margin-bottom:6px">{legend}</div>'
        f'<div style="overflow-x:auto;border:1px solid #e3e6e8;border-radius:6px">'
        f'{"".join(parts)}</div>'
        '<div style="margin-top:5px;color:#5f6368">Scroll horizontally — time scale is '
        "fixed, so a recovery occupies the same width at any point in the run.</div></div>"
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
