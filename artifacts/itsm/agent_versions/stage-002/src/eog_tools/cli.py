"""Command line entry points for local EOG tool experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import AgentConfig, ReactAgent
from .bootstrap import bootstrap_sdk
from .runner import new_run_dir, run_deployment, run_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eog-tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap-sdk", help="install the authenticated, current service SDK wheel")
    for name in ("run-task", "run-deployment"):
        command = sub.add_parser(name)
        command.add_argument("--domain", default="calendar")
        command.add_argument("--model", default="gpt-5")
        command.add_argument("--max-steps", type=int, default=50)
        command.add_argument("--max-completion-tokens", type=int, default=16_384)
        command.add_argument("--tool-result-chars", type=int, default=4_000)
        command.add_argument("--runs-dir", type=Path, default=Path("runs"))
    task = sub.choices["run-task"]
    task.add_argument("--version", type=int, required=True)
    task.add_argument("--split", choices=("train", "test"), default="train")
    task.add_argument("--task-id", required=True)
    deployment = sub.choices["run-deployment"]
    group = deployment.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=1)
    group.add_argument("--full", action="store_true")
    evolving = sub.add_parser("run-self-evolving", help="evaluate stage-by-stage adaptation through CommandAgent")
    evolving.add_argument("--domain")
    evolving.add_argument("--limit", type=int)
    evolving.add_argument("--full", action="store_true", default=None)
    evolving.add_argument("--matrix", action="store_true", default=None)
    evolving.add_argument("--runs-dir", type=Path, default=Path("runs"))
    evolving.add_argument("--resume", type=Path, help="resume an existing run directory")
    evolving.add_argument("--retry-incomplete-adapt", action="store_true",
                          help="archive one incomplete adaptation and rerun its stage on resume")
    evolving.add_argument("--historical-scores", type=Path,
                          help="rounded terminal cohort scores from a pre-checkpoint run")
    evolving.add_argument("--optimize-timeout-minutes", type=int)
    evolving.add_argument("--adapt-timeout-minutes", type=int)
    evolving.add_argument("--task-timeout-minutes", type=int)
    evolving.add_argument("--total-rollouts", type=int)
    evolving.add_argument("--training-pool", choices=("current", "cumulative"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bootstrap-sdk":
        info = bootstrap_sdk(Path.cwd())
        print(f"installed {info['filename']} (sha256 {info['sha256']})")
        return 0
    from .bootstrap import require_sdk_version

    require_sdk_version()
    if not os.environ.get("EVAL_SERVICE_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
        print("EVAL_SERVICE_API_KEY and OPENAI_API_KEY must be set in this terminal", file=sys.stderr)
        return 2
    if args.command == "run-deployment" and not args.full and args.limit < 1:
        print("--limit must be positive", file=sys.stderr)
        return 2
    if args.command == "run-self-evolving":
        if args.retry_incomplete_adapt and not args.resume:
            print("--retry-incomplete-adapt requires --resume", file=sys.stderr)
            return 2
        import json
        from simple_agentic_evals import EvalClient
        from .command_agent import TimedCommandAgent
        from .self_evolving import _write_json, run_self_evolving

        run_dir = args.resume.resolve() if args.resume else new_run_dir(args.runs_dir).resolve()
        if args.resume and not run_dir.is_dir():
            print(f"run directory does not exist: {run_dir}", file=sys.stderr)
            return 2
        config_path = run_dir / "run-config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            requested = {
                "domain": args.domain, "matrix": args.matrix,
                "limit": None if args.full else args.limit,
                "total_rollouts": args.total_rollouts, "training_pool": args.training_pool,
            }
            for key, value in requested.items():
                if value is not None and value != config[key]:
                    print(f"resume option {key}={value!r} differs from saved {config[key]!r}", file=sys.stderr)
                    return 2
            if args.full and config["limit"] is not None:
                print("--full differs from the saved limited run", file=sys.stderr)
                return 2
            for key, value in (
                ("optimize_timeout_minutes", args.optimize_timeout_minutes),
                ("adapt_timeout_minutes", args.adapt_timeout_minutes),
                ("task_timeout_minutes", args.task_timeout_minutes),
            ):
                if value is not None:
                    config[key] = value
        else:
            if args.resume and (not args.domain or not args.full and args.limit is None
                                or args.matrix is None or args.total_rollouts is None
                                or args.training_pool is None):
                print("legacy resume requires --domain, --full or --limit, --matrix, "
                      "--total-rollouts, and --training-pool", file=sys.stderr)
                return 2
            if not args.domain or args.full and args.limit is not None:
                print("provide --domain and choose either --full or --limit", file=sys.stderr)
                return 2
            config = {
                "domain": args.domain, "limit": None if args.full else (1 if args.limit is None else args.limit),
                "matrix": bool(args.matrix), "total_rollouts": 6 if args.total_rollouts is None else args.total_rollouts,
                "training_pool": args.training_pool or "current",
                "optimize_timeout_minutes": 720 if args.optimize_timeout_minutes is None else args.optimize_timeout_minutes,
                "adapt_timeout_minutes": 750 if args.adapt_timeout_minutes is None else args.adapt_timeout_minutes,
                "task_timeout_minutes": 30 if args.task_timeout_minutes is None else args.task_timeout_minutes,
            }
        if (config["limit"] is not None and config["limit"] < 1 or
            min(config["total_rollouts"], config["optimize_timeout_minutes"],
                config["adapt_timeout_minutes"], config["task_timeout_minutes"]) < 1 or
            config["adapt_timeout_minutes"] <= config["optimize_timeout_minutes"]):
            print("limits and timeouts must be positive; adapt timeout must exceed optimize timeout",
                  file=sys.stderr)
            return 2
        if args.historical_scores is not None:
            if not args.resume:
                print("--historical-scores requires --resume", file=sys.stderr)
                return 2
            config["historical_scores"] = str(args.historical_scores.resolve())
        historical_path = Path(config["historical_scores"]) if config.get("historical_scores") else None
        if historical_path is not None and not historical_path.is_file():
            print(f"historical score file does not exist: {historical_path}", file=sys.stderr)
            return 2
        _write_json(config_path, config)
        agent = TimedCommandAgent(
            [sys.executable, "-m", "eog_tools.command_bridge", "solve"],
            adapt_command=[sys.executable, "-m", "eog_tools.command_bridge", "adapt"],
            output_dir=run_dir / "command-agent", state_dir=run_dir / "state",
            timeout=config["task_timeout_minutes"] * 60,
            adapt_timeout=config["adapt_timeout_minutes"] * 60,
        )
        os.environ["EOG_RELAI_TOTAL_ROLLOUTS"] = str(config["total_rollouts"])
        os.environ["EOG_RELAI_TIMEOUT_MINUTES"] = str(config["optimize_timeout_minutes"])
        os.environ["EOG_RELAI_TRAINING_POOL"] = config["training_pool"]
        with EvalClient() as eval_client:
            report = run_self_evolving(
                eval_client, agent, run_dir, config, historical_path,
                retry_incomplete_adapt=args.retry_incomplete_adapt,
            )
        report_path = run_dir / "report.json"
        print(f"report: {report_path}")
        print(f"state: {run_dir / 'state' / 'active.json'}")
        print(f"ACC: {report['accuracy']:.3f}  status: {report['status']}")
        return 0
    config = AgentConfig(
        model=args.model,
        max_steps=args.max_steps,
        max_completion_tokens=args.max_completion_tokens,
        tool_result_chars=args.tool_result_chars,
    )
    from openai import OpenAI
    from simple_agentic_evals import EvalClient

    eval_client = EvalClient()
    agent = ReactAgent(config, OpenAI())
    run_dir = new_run_dir(args.runs_dir)
    try:
        if args.command == "run-task":
            record = run_task(
                eval_client, agent, domain=args.domain, version=args.version,
                split=args.split, task_id=args.task_id, run_dir=run_dir,
            )
            print(f"result: {run_dir / 'result.json'}")
            if record["error"]:
                print(record["error"], file=sys.stderr)
                return 1
            print(f"score={record['grade']['score']:.3f} passed={record['grade']['passed']}")
            return 0
        record = run_deployment(
            eval_client, agent, domain=args.domain,
            limit=None if args.full else args.limit, run_dir=run_dir,
        )
        report = record["report"]
        print(f"report: {run_dir / 'report.json'}")
        print(f"score={report['accuracy']:.3f} pass={report['success_rate']:.3f}")
        return 0
    finally:
        eval_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
