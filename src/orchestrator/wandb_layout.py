"""The W&B workspace layout, AS CODE — no UI configuration, ever.

`wandb_dash` decides *what* is logged; this decides *how it is arranged*. Both
are committed, so the dashboard is reproducible from a clean account with one
command instead of an afternoon of clicking.

Two rules drive every choice below.

**Stack panels that share an x-axis.** Section 1 puts progress, world size and
work-at-risk in three full-width panels on the same wall-clock axis, so reading
down a vertical line answers "how far along, how many nodes alive, how much
would a failure cost" at one instant. That is the `e5-progress.png` figure from
the writeups, made live.

**Use two x-axes deliberately.** Anything about the fleet, cost, or elapsed
progress is plotted against `t_rel` (wall clock), because DOWNTIME IS INVISIBLE
on a step axis — steps simply stop advancing. Anything about model quality is
plotted against `train_step`, because quality is a function of work done, not
time elapsed. Mixing the two is the easiest way to hide what a failure cost.

    python -m orchestrator.wandb_layout
"""

from __future__ import annotations

from . import wandb_dash as D


def _sections(entity: str = ""):
    """(title, caption, panels) in reading order.

    Each section carries a caption in full sentences saying what to look for and
    what would count as a failure — the discipline a paper figure caption uses.
    A panel needing no caption is usually a panel that should not exist.
    """
    import wandb_workspaces.reports.v2 as wr

    return [
        (
            "1 · The run",
            "**Training advances monotonically through every failure.** "
            "`ckpt_step` is durable progress — the furthest step that survives a "
            "node loss — while `step` is the optimistic frontier; the gap between "
            "them is exactly the work a failure would erase at that instant. "
            "Durable progress must never decrease. All three panels share a "
            "wall-clock axis, so a vertical line reads as one moment in the run.",
            [
                wr.LinePlot(
                    title="Durable progress vs. current frontier",
                    x="t_rel",
                    y=[D.K_DURABLE, D.K_CURRENT],
                    title_x="wall clock (s)",
                    title_y="training step",
                    layout=wr.Layout(w=24, h=10),
                ),
                wr.LinePlot(
                    title="World size",
                    x="t_rel",
                    y=[D.K_WORLD],
                    title_x="wall clock (s)",
                    title_y="nodes training",
                    layout=wr.Layout(w=24, h=5),
                ),
                wr.LinePlot(
                    title="Work at risk — steps since the last durable checkpoint",
                    x="t_rel",
                    y=[D.K_AT_RISK],
                    title_x="wall clock (s)",
                    title_y="steps",
                    layout=wr.Layout(w=24, h=5),
                ),
            ],
        ),
        (
            "2 · Slot occupancy",
            "**Each row is a slot, not a machine.** A world of N has exactly N "
            "slots for the life of the run and instances are transient occupants, "
            "so the chart stays N rows tall however many replacements the run "
            "burns through. The Gantt scrolls horizontally at a fixed time scale "
            "rather than compressing, so a recovery is the same width at hour 1 "
            "and hour 35; the table beneath it is the lossless record backing "
            "every segment. The cumulative counters must rise in lockstep — one "
            "replacement for every node lost.",
            [
                # Native Vega-Lite Gantt: W&B renders it, so drag-zoom on the
                # time axis and tooltips come from the same engine as every
                # other panel. The HTML version below is a fallback for
                # environments where the custom-chart preset is unavailable.
                wr.CustomChart(
                    query={"summaryTable": {"tableKey": D.GANTT_PANEL_TABLE_KEY}},
                    chart_name=f"{entity}/{D.GANTT_CHART_NAME}",
                    chart_fields={
                        "instance": "instance",
                        "state": "state",
                        "t_start": "t_start",
                        "t_end": "t_end",
                    },
                    chart_strings={"title": "Fleet slot occupancy over time"},
                    layout=wr.Layout(w=24, h=14),
                ),
                wr.MediaBrowser(
                    media_keys=[D.K_GANTT],
                    layout=wr.Layout(w=24, h=14),
                ),
                wr.LinePlot(
                    title="Nodes lost and replacements launched (cumulative)",
                    x="t_rel",
                    y=[D.K_LOST, D.K_REPL],
                    title_x="wall clock (s)",
                    title_y="count",
                    layout=wr.Layout(w=24, h=7),
                ),
            ],
        ),
        (
            "3 · Efficiency",
            "**Goodput is the run-level answer; step time is its per-iteration "
            "cause.** Goodput is the fraction of billed wall clock actually spent "
            "training — the training budget lives inside the checkpoint, so "
            "downtime can never inflate it. Step time is expected to rise while "
            "the world is short-handed and return to baseline on recovery; "
            "outliers are clipped because checkpoint steps run roughly an order "
            "of magnitude longer and would otherwise flatten the range.",
            [
                wr.LinePlot(
                    title="Goodput",
                    x="t_rel",
                    y=[D.K_GOODPUT],
                    title_x="wall clock (s)",
                    title_y="trained seconds / wall seconds",
                    range_y=(0.0, 1.0),
                    layout=wr.Layout(w=12, h=7),
                ),
                wr.LinePlot(
                    title="Step time (checkpoint spikes clipped)",
                    x="train_step",
                    y=[D.K_MS],
                    title_x="training step",
                    title_y="ms / step",
                    ignore_outliers=True,
                    layout=wr.Layout(w=12, h=7),
                ),
            ],
        ),
        (
            "4 · Model quality",
            "**Plotted against training step, not wall clock, because loss is a "
            "function of work done rather than time elapsed.** The loss curve is "
            "never smoothed: smoothing across a rollback would invent data that "
            "was never computed. A resume shows here as a brief overlap where the "
            "same steps are executed twice.",
            [
                wr.LinePlot(
                    title="Training loss",
                    x="train_step",
                    y=[D.K_LOSS],
                    title_x="training step",
                    title_y="cross-entropy loss",
                    smoothing_factor=0,
                    layout=wr.Layout(w=12, h=7),
                ),
                wr.LinePlot(
                    title="Tokens processed",
                    x="t_rel",
                    y=[D.K_TOKENS],
                    title_x="wall clock (s)",
                    title_y="tokens (billions)",
                    layout=wr.Layout(w=12, h=7),
                ),
            ],
        ),
        (
            "5 · Cost",
            "**Unit cost is the figure that survives comparison across runs of "
            "different length.** Cumulative spend rises with wall clock whether "
            "or not the fleet is training, so a failure appears here as a "
            "steepening of dollars per unit of work rather than as a step change "
            "in total spend.",
            [
                wr.LinePlot(
                    title="Cumulative spend",
                    x="t_rel",
                    y=[D.K_USD],
                    title_x="wall clock (s)",
                    title_y="USD",
                    layout=wr.Layout(w=12, h=7),
                ),
                wr.LinePlot(
                    title="Unit cost",
                    x="t_rel",
                    y=[D.K_USD_1K],
                    title_x="wall clock (s)",
                    title_y="USD per 1000 steps",
                    layout=wr.Layout(w=12, h=7),
                ),
            ],
        ),
        (
            "6 · Sentinels",
            "**These panels are expected to be boring.** A whole-group restart "
            "means the supervisor gave up on the current membership and discarded "
            "healthy survivors, so this series staying flat at zero is a result in "
            "itself. `degraded` marks every interval below target world size.",
            [
                wr.LinePlot(
                    title="Whole-group restarts (expected: flat at zero)",
                    x="t_rel",
                    y=[D.K_WGR],
                    title_x="wall clock (s)",
                    title_y="restarts",
                    layout=wr.Layout(w=12, h=6),
                ),
                wr.LinePlot(
                    title="Degraded intervals",
                    x="t_rel",
                    y=[D.K_DEGRADED],
                    title_x="wall clock (s)",
                    title_y="1 = below target world",
                    layout=wr.Layout(w=12, h=6),
                ),
            ],
        ),
    ]


def apply(entity: str, project: str = "spot-train", name: str = "Distributed training") -> str:
    """Create/overwrite the saved workspace view. Returns its URL."""
    import wandb_workspaces.workspaces as ws

    sections = []
    for title, _caption, panels in _sections(entity):
        # No caption panels. Each consumed a full-width row and read as filler;
        # panel titles and axis labels carry the meaning instead, and the
        # reasoning lives in this module's docstrings and docs/e5-results.md.
        sections.append(ws.Section(name=title, panels=panels, is_open=True))
    view = ws.Workspace(entity=entity, project=project, name=name, sections=sections)
    view.save()
    return view.url


def main() -> int:
    import os

    entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        import wandb

        entity = wandb.Api().default_entity
    url = apply(entity, os.environ.get("WANDB_PROJECT", "spot-train"))
    print(f"workspace: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
