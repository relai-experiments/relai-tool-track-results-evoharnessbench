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
    evolving.add_argument("--domain", required=True)
    evolving.add_argument("--limit", type=int, default=1)
    evolving.add_argument("--full", action="store_true")
    evolving.add_argument("--matrix", action="store_true")
    evolving.add_argument("--runs-dir", type=Path, default=Path("runs"))
    evolving.add_argument("--optimize-timeout-minutes", type=int, default=720)
    evolving.add_argument("--adapt-timeout-minutes", type=int, default=750)
    evolving.add_argument("--task-timeout-minutes", type=int, default=30)
    evolving.add_argument("--total-rollouts", type=int, default=6)
    evolving.add_argument("--training-pool", choices=("current", "cumulative"), default="current")
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
        if not args.full and args.limit < 1:
            parser_error = "--limit must be positive"
            print(parser_error, file=sys.stderr)
            return 2
        if min(args.optimize_timeout_minutes, args.adapt_timeout_minutes,
               args.task_timeout_minutes) < 1:
            print("timeout minutes must be positive", file=sys.stderr)
            return 2
        if args.adapt_timeout_minutes <= args.optimize_timeout_minutes:
            print("--adapt-timeout-minutes must exceed --optimize-timeout-minutes", file=sys.stderr)
            return 2
        from dataclasses import asdict
        import json
        from simple_agentic_evals import EvalClient, run_benchmark
        from .command_agent import TimedCommandAgent

        run_dir = new_run_dir(args.runs_dir).resolve()
        agent = TimedCommandAgent(
            [sys.executable, "-m", "eog_tools.command_bridge", "solve"],
            adapt_command=[sys.executable, "-m", "eog_tools.command_bridge", "adapt"],
            output_dir=run_dir / "command-agent", state_dir=run_dir / "state",
            timeout=args.task_timeout_minutes * 60,
            adapt_timeout=args.adapt_timeout_minutes * 60,
        )
        os.environ["EOG_RELAI_TOTAL_ROLLOUTS"] = str(args.total_rollouts)
        os.environ["EOG_RELAI_TIMEOUT_MINUTES"] = str(args.optimize_timeout_minutes)
        os.environ["EOG_RELAI_TRAINING_POOL"] = args.training_pool
        with EvalClient() as eval_client:
            report = run_benchmark(
                agent, "self_evolving", dataset="evovling_tools", benchmark="eog",
                domain=args.domain, limit=None if args.full else args.limit,
                matrix=args.matrix, client=eval_client, on_error="raise", progress=True,
            )
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str) + "\n")
        print(f"report: {report_path}")
        print(f"state: {run_dir / 'state' / 'active.json'}")
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
