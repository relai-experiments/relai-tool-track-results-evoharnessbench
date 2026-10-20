"""Run one cataloged EOG training task and emit RELAI's custom-run contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from eog_tools.agent import AgentConfig, ReactAgent
from eog_tools.runner import run_task


CATALOG_PATH = Path(__file__).with_name("catalog.json")


def harness_stage(cohort_stage: int) -> int | None:
    raw = os.environ.get("EOG_RELAI_HARNESS_STAGE")
    if raw is None:
        return None
    if not raw.isdecimal() or int(raw) < cohort_stage:
        raise ValueError(
            f"EOG_RELAI_HARNESS_STAGE must be an integer >= cohort stage {cohort_stage}; got {raw!r}"
        )
    return int(raw)


def trace_steps(agent: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not agent:
        return []
    steps: list[dict[str, Any]] = []
    for message in agent.get("initial_messages") or []:
        steps.append({"step_id": len(steps) + 1, "source": message.get("role", "user"), "message": message.get("content") or ""})
    for event in agent.get("events") or []:
        assistant = event.get("assistant") or {}
        steps.append({
            "step_id": len(steps) + 1,
            "source": "agent",
            "message": assistant.get("content") or "",
            "tool_calls": assistant.get("tool_calls") or [],
            "tool_results": event.get("tool_results") or [],
            "usage": event.get("usage"),
        })
    return steps


def to_relai_result(task_id: str, record: dict[str, Any], evaluators: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    agent = record.get("agent")
    grade = record.get("grade")
    error = record.get("error")
    result: dict[str, Any] = {
        "schema_version": "relai.custom_run.v1",
        "task_id": task_id,
        "trace": {"steps": trace_steps(agent)},
        "final_output": (agent or {}).get("final_message", ""),
        "evaluations": [],
        "evaluation_errors": [],
    }
    if error or not grade:
        result["failure"] = {"error_type": "EogTaskError", "message": error or "EOG grading returned no result"}
        result["evaluation_errors"] = [{"evaluator_id": "custom-grade", "error_type": "EogTaskError", "message": result["failure"]["message"]}]
    else:
        verifiers = grade.get("per_verifier") or []
        if evaluators:
            expected_names = [item["name"] for item in evaluators]
            actual_names = [item.get("name") for item in verifiers]
            if len(verifiers) != grade["n_total"] or actual_names != expected_names:
                message = f"EOG verifier mismatch: catalog={expected_names!r}, grade={actual_names!r}"
                result["failure"] = {"error_type": "EogVerifierMismatch", "message": message}
                result["evaluation_errors"] = [{"evaluator_id": "eog-verifiers", "error_type": "EogVerifierMismatch", "message": message}]
            else:
                for spec, item in zip(evaluators, verifiers):
                    if item.get("error"):
                        message = f"{item['name']}: {item['error']}"
                        result["evaluation_errors"].append({"evaluator_id": spec["id"], "error_type": "EogVerifierError", "message": message})
                        continue
                    score = item.get("score")
                    if not isinstance(score, (int, float)) or isinstance(score, bool):
                        result["evaluation_errors"].append({"evaluator_id": spec["id"], "error_type": "EogVerifierError", "message": f"{item['name']}: missing numeric score"})
                        continue
                    details = [f"{item['name']}: {'pass' if item.get('passed') else 'fail'}"]
                    for key in ("score", "expected", "actual", "comparison_type", "description", "details"):
                        if item.get(key) is not None and item.get(key) != "":
                            details.append(f"{key}={item[key]!r}")
                    result["evaluations"].append({
                        "evaluator_id": spec["id"], "score": score,
                        "min_score": 0.0, "max_score": 1.0,
                        "feedback": "; ".join(details), "result": item,
                    })
                if result["evaluation_errors"]:
                    result["failure"] = {"error_type": "EogVerifierError", "message": "EOG grader returned verifier errors"}
                elif abs(sum(item["score"] for item in result["evaluations"]) / len(result["evaluations"]) - grade["score"]) > 1e-9:
                    message = "EOG aggregate score differs from the mean of verifier scores"
                    result["evaluations"] = []
                    result["failure"] = {"error_type": "EogScoreMismatch", "message": message}
                    result["evaluation_errors"] = [{"evaluator_id": "eog-verifiers", "error_type": "EogScoreMismatch", "message": message}]
        else:
            feedback = "; ".join(f"{item.get('name', '?')}: {'pass' if item.get('passed') else 'fail'}" for item in verifiers)
            result["evaluations"] = [{
                "evaluator_id": "custom-grade", "score": grade["score"],
                "min_score": 0.0, "max_score": 1.0,
                "feedback": feedback or f"{grade['n_passed']}/{grade['n_total']} verifiers passed",
                "result": {"passed": grade["passed"], "n_passed": grade["n_passed"], "n_total": grade["n_total"], "per_verifier": verifiers},
            }]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    catalog = json.loads(CATALOG_PATH.read_text())
    entry = next((task for task in catalog["tasks"] if task["id"] == args.task_id), None)
    if entry is None:
        raise SystemExit(f"unknown EOG catalog task: {args.task_id}")
    metadata = entry["metadata"]
    from openai import OpenAI
    from simple_agentic_evals import EvalClient

    selected_stage = harness_stage(metadata["stage"])
    client = EvalClient()
    try:
        if selected_stage is not None:
            client.require_cohort_harness_selection()
        agent = ReactAgent(AgentConfig(), OpenAI())
        record = run_task(client, agent, domain=metadata["domain"], version=metadata["stage"],
                          split=metadata["split"], task_id=metadata["sdk_task_id"],
                          run_dir=args.output.parent, at_stage=selected_stage)
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_relai_result(args.task_id, record, entry.get("evaluators")), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
