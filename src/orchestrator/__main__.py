"""CLI: ``spot-orchestrate {setup,stage-data [--remote],bake-ami,baseline,spot,preempt,ddp,
ddp-preempt,multinode,multinode-shrink,multinode-preempt,scaling-experiment,
scaling-clean,scaling-preempt} [--dry-run]``,
``spot-orchestrate resume <run_id> [--budget N] [--market ...]``,
``spot-orchestrate compare <run_id> [<run_id> ...]``, and
``spot-orchestrate orch {up,status,logs,down}`` — the durable remote
orchestrator, which runs an experiment on an always-on t3.micro so a multi-day
run does not need this laptop awake.

You run this; it needs your AWS creds in the environment. A git-ignored ``.env``
in the current directory is loaded into the environment on startup (values are
never printed). ``--dry-run`` makes every AWS call a no-op that just logs what it
would do — use it to review before spending anything.
"""

from __future__ import annotations

import argparse
import os
import sys


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a local .env (KEY=VALUE lines). Never echoes
    values; does not override variables already set in the environment."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def main() -> None:
    _load_dotenv()

    # --dry-run is attached to each subcommand (e.g. `setup --dry-run`) via a
    # shared parent parser, so it reads naturally after the command.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="log AWS calls, execute none")

    parser = argparse.ArgumentParser(prog="spot-orchestrate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "setup",
        "bake-ami",
        "baseline",
        "spot",
        "preempt",
        "ddp",
        "ddp-preempt",
        "multinode",
        "multinode-shrink",
        "multinode-preempt",
        "calibrate",
        "scaling-experiment",
        "scaling-clean",
        "scaling-preempt",
    ):
        sub.add_parser(name, parents=[common])

    # stage-data: local by default (unchanged); --remote runs the identical
    # prepare+upload on a throwaway EC2 box, which is the only way corpora like
    # full OpenWebText (~110 GB transient, 17 GB uploaded) can be staged at all.
    stage_parser = sub.add_parser(
        "stage-data", parents=[common], help="prepare the dataset and upload the bins to S3"
    )
    stage_parser.add_argument(
        "--remote",
        action="store_true",
        help="prepare IN AWS on one throwaway box (same-region upload; it self-terminates)",
    )
    stage_parser.add_argument(
        "--no-attach", action="store_true", help="--remote: launch and exit instead of streaming"
    )
    stage_parser.add_argument(
        "--attach",
        dest="prep_id",
        default="",
        help="--remote: reattach to a prep already running (its id)",
    )

    res_parser = sub.add_parser("resume", parents=[common])
    res_parser.add_argument("run_id", help="existing run id to resume from its latest checkpoint")
    res_parser.add_argument(
        "--budget", type=int, default=None, help="training seconds (default: BASELINE_SECONDS)"
    )
    res_parser.add_argument(
        "--market",
        choices=["on-demand", "spot"],
        default=None,
        help="instance market (default: inferred from the run id's kind)",
    )
    ext_parser = sub.add_parser("extend", parents=[common])
    ext_parser.add_argument(
        "run_id", help="completed multi-node run to continue from its checkpoint"
    )
    ext_parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="new TOTAL training seconds (not an increment) — budget-in-checkpoint "
        "computes TRAIN_BUDGET_SECONDS minus trained_seconds",
    )
    cmp_parser = sub.add_parser("compare", parents=[common])
    cmp_parser.add_argument("run_ids", nargs="+", help="run ids to compare (2+ recommended)")

    logs_parser = sub.add_parser(
        "logs",
        parents=[common],
        help="live per-node log dashboard: arrow keys switch nodes, dead nodes freeze",
    )
    logs_parser.add_argument("run_id", help="run id to view (live or finished)")
    logs_parser.add_argument(
        "--node", type=int, default=None, help="node index to select first (default: first node)"
    )
    logs_parser.add_argument(
        "--interval", type=float, default=None, help="poll seconds (default: LOG_STREAM_SECONDS)"
    )
    logs_parser.add_argument(
        "--uri",
        default=None,
        help="run base override — a local dir or s3://… (default: s3://BUCKET/runs/<run_id>)",
    )
    logs_parser.add_argument(
        "--grid",
        action="store_true",
        help="show all node logs at once in a tiled grid (up to 8 panes); g toggles",
    )
    logs_parser.add_argument(
        "--plain",
        action="store_true",
        help="append-tail one node (--node) with join/death notices — for tmux panes/pipes",
    )

    # --- remote orchestrator: run the control plane on a t3.micro, not here --- #
    orch_parser = sub.add_parser(
        "orch",
        parents=[common],
        help="durable remote orchestrator: run an experiment on an always-on t3.micro",
    )
    orch_sub = orch_parser.add_subparsers(dest="orch_command", required=True)
    orch_up = orch_sub.add_parser(
        "up",
        parents=[common],
        help="launch the control plane and attach to the live dashboard (Ctrl-C detaches)",
    )
    orch_up.add_argument(
        "--experiment",
        required=True,
        help="experiment the remote orchestrator runs (e.g. multinode)",
    )
    orch_up.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="env override for the run, repeatable (e.g. --env NODES=8)",
    )
    orch_up.add_argument(
        "--run-id",
        default="",
        help="adopt an EXISTING run instead of minting a new one — the way back "
        "from a dead control-plane box (the fleet keeps training without it)",
    )
    orch_up.add_argument(
        "--force",
        action="store_true",
        help="with --run-id, re-enter even a run that already wrote metrics.json",
    )
    orch_up.add_argument("--no-attach", action="store_true", help="launch and exit (scripted use)")
    orch_up.add_argument("--node", type=int, default=None, help="dashboard: node tab to open")
    orch_up.add_argument("--interval", type=float, default=None, help="dashboard: poll seconds")
    orch_up.add_argument("--grid", action="store_true", help="dashboard: open the tiled grid")
    orch_status = orch_sub.add_parser(
        "status",
        parents=[common],
        help="one screen: heartbeat age, epoch/world, step/loss, elapsed vs budget, cost",
    )
    orch_status.add_argument("--id", dest="orch_id", default="", help="orchestrator id")
    orch_logs = orch_sub.add_parser(
        "logs", parents=[common], help="reattach to the active orchestrator's dashboard"
    )
    orch_logs.add_argument("--id", dest="orch_id", default="", help="orchestrator id")
    orch_logs.add_argument("--run", dest="run_id", default="", help="run id (skips discovery)")
    orch_logs.add_argument("--node", type=int, default=None)
    orch_logs.add_argument("--interval", type=float, default=None)
    orch_logs.add_argument("--grid", action="store_true")
    orch_logs.add_argument("--plain", action="store_true")
    orch_down = orch_sub.add_parser("down", parents=[common], help="terminate the control plane")
    orch_down.add_argument("--id", dest="orch_id", default="", help="orchestrator id")
    orch_down.add_argument(
        "--all", action="store_true", help="also terminate the training fleet it was driving"
    )
    orch_down.add_argument("--yes", "-y", action="store_true", help="skip the confirmation")
    # What the systemd unit on the box runs — you never type this yourself.
    orch_agent = orch_sub.add_parser(
        "_agent", parents=[common], help="(internal) run the experiment ON the control-plane box"
    )
    orch_agent.add_argument("--orch-id", dest="orch_id", required=True)
    orch_agent.add_argument("--experiment", required=True)

    fleet_parser = sub.add_parser(
        "fleet", parents=[common], help="inference fleet (ROADMAP Part 1)"
    )
    fleet_sub = fleet_parser.add_subparsers(dest="fleet_command", required=True)
    fleet_common = argparse.ArgumentParser(add_help=False)
    fleet_common.add_argument(
        "--local", action="store_true", help="run the fleet as local processes (no AWS)"
    )
    fleet_up = fleet_sub.add_parser("up", parents=[common, fleet_common])
    fleet_up.add_argument("--workers", type=int, default=None, help="default: 2 local, 4 cloud")
    fleet_up.add_argument(
        "--run", default="", help="run id whose latest checkpoint the fleet serves (cloud mode)"
    )
    fleet_up.add_argument("--router-port", type=int, default=8000)
    fleet_up.add_argument(
        "--checkpoint-uri",
        default="checkpoints/",
        help="checkpoint dir/prefix the workers serve (local path or s3://)",
    )
    fleet_up.add_argument(
        "--data-local-dir",
        default="third_party/nanoGPT/data/shakespeare_char",
        help="dir holding meta.pkl (the char codec)",
    )
    fleet_serve = fleet_sub.add_parser(
        "serve",
        parents=[common],
        help="minimal cloud serve: ONE box, no router — curl it directly",
    )
    fleet_serve.add_argument("--run", required=True, help="run id whose latest checkpoint to serve")
    fleet_sub.add_parser("status", parents=[common, fleet_common])
    fleet_sub.add_parser("down", parents=[common, fleet_common])
    fleet_kill = fleet_sub.add_parser("kill-worker", parents=[common, fleet_common])
    fleet_kill.add_argument("--worker-id", default=None)
    fleet_mon = fleet_sub.add_parser(
        "monitor",
        parents=[common, fleet_common],
        help="live per-worker queue/load table; --wandb mirrors it",
    )
    fleet_mon.add_argument("--url", default="", help="router base URL (default: discover)")
    fleet_mon.add_argument("--interval", type=float, default=2.0)
    fleet_mon.add_argument("--wandb", action="store_true", help="mirror ticks to Weights & Biases")
    fleet_pre = fleet_sub.add_parser(
        "preempt",
        parents=[common, fleet_common],
        help="preemption experiment: calibrated load, kill one worker, latency verdict",
    )
    fleet_pre.add_argument("--run", default="", help="run id to serve if the fleet must be booted")
    fleet_pre.add_argument("--workers", type=int, default=None, help="only used when booting")
    fleet_pre.add_argument("--duration", type=int, default=150, help="loadgen seconds")
    fleet_pre.add_argument("--kill-after", type=int, default=60, help="kill timing (seconds)")
    fleet_pre.add_argument("--rps", type=float, default=0.0, help="0 = auto-calibrate to 70%%")
    fleet_pre.add_argument(
        "--keep", action="store_true", help="don't tear down a fleet this experiment booted"
    )

    sw_parser = sub.add_parser(
        "spotwatch",
        parents=[common],
        help="unattended GPU spot-availability collector (Lambda + 10-min tick)",
    )
    sw_sub = sw_parser.add_subparsers(dest="spotwatch_command", required=True)
    sw_sub.add_parser("deploy", parents=[common], help="create/update the collector (idempotent)")
    sw_sub.add_parser("down", parents=[common], help="remove the rule, function and role")
    sw_report = sw_sub.add_parser("report", parents=[common], help="analyse the collected JSONL")
    sw_report.add_argument(
        "--since", type=float, default=72.0, help="hours to analyse (default 72)"
    )
    sw_report.add_argument(
        "--threshold", type=float, default=7.0, help="placement score counted as 'good' (1-10)"
    )
    sw_report.add_argument("--region", default="", help="home region (default: AWS_REGION)")
    sw_report.add_argument(
        "--type", dest="instance_type", default="", help="home GPU type (default: INSTANCE_TYPE)"
    )

    args = parser.parse_args()

    if args.command in ("spotwatch", "orch"):
        from . import aws
        from .config import OrchestratorConfig

        aws.set_dry_run(args.dry_run)
        cfg = OrchestratorConfig()
        aws.set_region(cfg.region)
        if args.command == "spotwatch":
            from . import spotwatch

            if args.spotwatch_command == "deploy":
                spotwatch.deploy(cfg)
            elif args.spotwatch_command == "down":
                spotwatch.down(cfg)
            elif args.spotwatch_command == "report":
                spotwatch.report(
                    cfg,
                    since_hours=args.since,
                    threshold=args.threshold,
                    home_region=args.region,
                    home_type=args.instance_type,
                )
        else:
            from . import orch

            if args.orch_command == "up":
                pairs = {}
                for item in args.env:
                    if "=" not in item:
                        parser.error(f"--env expects K=V, got {item!r}")
                    k, _, v = item.partition("=")
                    pairs[k.strip()] = v
                orch.up(
                    cfg,
                    experiment=args.experiment,
                    env_overrides=pairs,
                    no_attach=args.no_attach,
                    node=args.node,
                    interval=args.interval,
                    grid=args.grid,
                    run_id=args.run_id,
                    force=args.force,
                )
            elif args.orch_command == "status":
                orch.status(cfg, args.orch_id)
            elif args.orch_command == "logs":
                orch.logs(
                    cfg,
                    orch_id=args.orch_id,
                    run_id=args.run_id,
                    node=args.node,
                    interval=args.interval,
                    grid=args.grid,
                    plain=args.plain,
                )
            elif args.orch_command == "down":
                orch.down(cfg, orch_id=args.orch_id, all_=args.all, yes=args.yes)
            elif args.orch_command == "_agent":
                # Runs ON the control-plane box under systemd; its exit code is the
                # restart signal (0 = done, nonzero = systemd retries and resumes).
                sys.exit(orch.run_agent(cfg, orch_id=args.orch_id, experiment=args.experiment))
        if args.dry_run:
            print("\n[dry-run] no AWS calls were made.", file=sys.stderr)
        return

    if args.command == "fleet":
        from . import fleet

        if args.fleet_command == "preempt":
            from . import aws, fleet_preempt
            from .config import OrchestratorConfig

            aws.set_dry_run(args.dry_run)
            cfg = OrchestratorConfig.for_inference()
            aws.set_region(cfg.region)
            fleet_preempt.run_fleet_preempt(
                cfg,
                local=args.local,
                run_id=args.run,
                workers=args.workers,
                duration=args.duration,
                kill_after=args.kill_after,
                rps=args.rps,
                keep=args.keep,
            )
            if args.dry_run:
                print("\n[dry-run] no AWS calls were made.", file=sys.stderr)
            return

        if args.fleet_command == "monitor":
            from . import monitor
            from .config import OrchestratorConfig

            cfg = OrchestratorConfig.for_inference()
            url = args.url
            if not url and getattr(args, "local", False):
                url = fleet.router_url_local()
            elif not url:
                from . import aws

                aws.set_dry_run(args.dry_run)
                aws.set_region(cfg.region)
                url = fleet.router_url_cloud(cfg)
            if not url:
                sys.exit("fleet monitor: no running router found — pass --url or start a fleet")
            monitor.run_monitor(cfg, url, interval=args.interval, use_wandb=args.wandb)
            return

        # Local mode needs no AWS credentials or config.
        if getattr(args, "local", False):
            if args.fleet_command == "up":
                fleet.up_local(
                    workers=args.workers if args.workers is not None else 2,
                    router_port=args.router_port,
                    checkpoint_uri=args.checkpoint_uri,
                    data_local_dir=args.data_local_dir,
                )
            elif args.fleet_command == "status":
                fleet.status_local()
            elif args.fleet_command == "down":
                fleet.down_local()
            elif args.fleet_command == "kill-worker":
                fleet.kill_worker_local(args.worker_id)
            return

        # Cloud mode: same creds-after-dotenv discipline as the experiments.
        from . import aws
        from .config import OrchestratorConfig

        aws.set_dry_run(args.dry_run)
        cfg = OrchestratorConfig.for_inference()
        aws.set_region(cfg.region)
        if args.fleet_command == "serve":
            fleet.serve_cloud(cfg, run_id=args.run)
        elif args.fleet_command == "up":
            fleet.up_cloud(
                cfg,
                workers=args.workers if args.workers is not None else cfg.fleet_worker_count,
                run_id=args.run,
            )
        elif args.fleet_command == "status":
            fleet.status_cloud(cfg)
        elif args.fleet_command == "down":
            fleet.down_cloud(cfg)
        elif args.fleet_command == "kill-worker":
            fleet.kill_worker_cloud(cfg, args.worker_id)
        if args.dry_run:
            print("\n[dry-run] no AWS calls were made.", file=sys.stderr)
        return

    # Imported after dotenv so config picks up the loaded environment, and so
    # `aws` (the only creds-touching module) is configured before any call.
    from . import aws, dataset, experiments, setup
    from .config import OrchestratorConfig

    aws.set_dry_run(args.dry_run)
    cfg = OrchestratorConfig()
    aws.set_region(cfg.region)

    if args.command == "setup":
        setup.ensure_infra(cfg)
    elif args.command == "stage-data":
        # --attach implies --remote (you can only reattach to a remote prep).
        if args.remote or args.prep_id:
            from . import prep

            prep.run_remote_prep(cfg, attach=not args.no_attach, prep_id=args.prep_id)
        else:
            dataset.stage_data(cfg)
    elif args.command == "bake-ami":
        from . import bake

        bake.bake_ami(cfg)
    elif args.command == "baseline":
        experiments.run_baseline(cfg)
    elif args.command == "spot":
        experiments.run_spot(cfg)
    elif args.command == "preempt":
        experiments.run_preempt(cfg)
    elif args.command == "ddp":
        experiments.run_ddp(cfg)
    elif args.command == "ddp-preempt":
        experiments.run_preempt(cfg, ddp=True)
    elif args.command == "multinode":
        experiments.run_multinode(cfg)
    elif args.command == "multinode-shrink":
        experiments.run_multinode_shrink(cfg)
    elif args.command == "multinode-preempt":
        experiments.run_multinode_preempt(cfg)
    elif args.command == "calibrate":
        experiments.run_calibrate(cfg)
    elif args.command == "scaling-experiment":
        experiments.run_scaling_experiment(cfg)
    elif args.command == "scaling-clean":
        experiments.run_scaling_clean(cfg)
    elif args.command == "scaling-preempt":
        experiments.run_scaling_preempt(cfg)
    elif args.command == "resume":
        experiments.run_resume(cfg, args.run_id, budget=args.budget, market=args.market)
    elif args.command == "extend":
        experiments.run_extend(cfg, args.run_id, budget=args.budget)
    elif args.command == "compare":
        from . import compare

        compare.run_compare(cfg, args.run_ids)
    elif args.command == "logs":
        from . import logview

        logview.run_logs(
            cfg,
            args.run_id,
            uri=args.uri,
            interval=args.interval,
            node=args.node,
            grid=args.grid,
            plain=args.plain,
        )
    else:  # pragma: no cover — argparse enforces the choices
        parser.error(f"unknown command {args.command}")

    if args.dry_run:
        print("\n[dry-run] no AWS calls were made.", file=sys.stderr)


if __name__ == "__main__":
    main()
