"""Run EOG tasks and persist local agent evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
import uuid
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .agent import AgentConfig, AgentResult, ReactAgent

DATASET = "evovling_tools"  # The service catalog uses this spelling.
BENCHMARK = "eog"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def new_run_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = root / f"{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _metadata(config: AgentConfig, domain: str) -> dict[str, Any]:
    sdk_version = importlib.metadata.version("simple-agentic-evals")
    wheel_info = Path(".cache/sdk/sdk-info.json")
    return {
        "dataset": DATASET,
        "benchmark": BENCHMARK,
        "domain": domain,
        "agent_config": asdict(config),
        "sdk_version": sdk_version,
        "sdk_wheel": json.loads(wheel_info.read_text()) if wheel_info.exists() else None,
    }


def _task_key(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()[:16]


def _canonically_equal_scalars(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) == isinstance(actual, str):
        return False
    text, number = (expected, actual) if isinstance(expected, str) else (actual, expected)
    if isinstance(number, bool) or not isinstance(number, (int, float, Decimal)):
        return False
    if text != text.strip() or not text:
        return False
    try:
        text_number = Decimal(text)
        value_number = Decimal(str(number))
    except (InvalidOperation, ValueError):
        return False
    return text_number.is_finite() and value_number.is_finite() and text_number == value_number


def _grade_dict(grade: Any) -> dict[str, Any]:
    rows = [dict(row) for row in grade.per_verifier]
    if not rows:
        raise RuntimeError("grader returned no verifiers")
    for row in rows:
        if (
            row.get("passed") is False
            and not row.get("error")
            and row.get("comparison_type") == "equals"
            and _canonically_equal_scalars(row.get("expected"), row.get("actual"))
        ):
            row["passed"] = True
            row["score"] = 1.0
            row["normalization"] = (
                "Corrected equals result after canonical numeric scalar comparison."
            )
    n_total = len(rows)
    n_passed = sum(row.get("passed") is True for row in rows)
    return {
        "score": n_passed / n_total,
        "passed": n_passed == n_total,
        "n_passed": n_passed,
        "n_total": n_total,
        "per_verifier": rows,
    }


def run_task(
    eval_client: Any,
    agent: ReactAgent,
    *,
    domain: str,
    version: int,
    split: str,
    task_id: str,
    run_dir: Path,
    at_stage: int | None = None,
) -> dict[str, Any]:
    from simple_agentic_evals.client import SelectorCompatibilityError

    task = eval_client.task(
        DATASET, BENCHMARK, version, task_id,
        split=split, domain=domain, resource_mode="accumulative",
        **({"at_stage": at_stage} if at_stage is not None else {}),
    )
    record: dict[str, Any] = {
        "run": _metadata(agent.config, domain),
        "task": {"id": task_id, "version": version, "split": split,
                 "at_stage": at_stage if at_stage is not None else version},
        "agent": None,
        "grade": None,
        "error": None,
    }
    try:
        with task:
            result = agent.run(task)
            record["agent"] = result.to_dict()
            if result.status != "error":
                record["grade"] = _grade_dict(task.grade())
            else:
                record["error"] = result.error
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, SelectorCompatibilityError):
            raise
    finally:
        _write_json(run_dir / "result.json", record)
    return record


def run_deployment(
    eval_client: Any,
    agent: ReactAgent,
    *,
    domain: str,
    limit: int | None,
    run_dir: Path,
) -> dict[str, Any]:
    from simple_agentic_evals import run_benchmark

    traces: dict[str, Path] = {}

    def act(task: Any) -> dict[str, int]:
        result: AgentResult = agent.run(task)
        path = run_dir / "traces" / f"{_task_key(task.task_id)}.json"
        _write_json(path, {"task_id": task.task_id, "agent": result.to_dict()})
        traces[task.task_id] = path
        if result.status == "error":
            raise RuntimeError(result.error or "agent failed")
        return {"total_tokens": result.total_tokens, "n_steps": result.steps}

    try:
        report = run_benchmark(
            act, "deployment_eval", dataset=DATASET, benchmark=BENCHMARK,
            domain=domain, limit=limit, client=eval_client, on_error="raise",
            progress=True,
        )
        if report.n_tasks < 1 or report.n_errors:
            raise RuntimeError("benchmark returned no tasks or reported evaluation errors")
        for task_result in report.task_results:
            path = traces.get(task_result["task_id"])
            if path is not None:
                value = json.loads(path.read_text())
                value["grade"] = {
                    "score": task_result["score"],
                    "passed": task_result["solved"],
                    "per_verifier": task_result.get("per_verifier", []),
                }
                _write_json(path, value)
        record = {
            "run": _metadata(agent.config, domain),
            "mode": "deployment_eval",
            "limit": limit,
            "report": asdict(report),
        }
        _write_json(run_dir / "report.json", record)
        return record
    except Exception as exc:
        _write_json(run_dir / "error.json", {
            "run": _metadata(agent.config, domain),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
