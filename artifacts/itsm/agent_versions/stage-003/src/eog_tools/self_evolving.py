"""Checkpointed EOG self-evolving evaluation around the SDK task protocol."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from httpx import TransportError
from simple_agentic_evals.client import ServiceError
from simple_agentic_evals.command_agent import CommandAgentError

DATASET = "evovling_tools"
BENCHMARK = "eog"
RESOURCE_MODE = "accumulative"
RETRY_ATTEMPTS = 4
RETRY_DELAY_S = 15


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _key(kind: str, cohort: int | str, at_stage: int | str) -> str:
    if kind == "last_row":
        return "last-row-full"
    return f"{kind}-cohort-{cohort}-harness-{at_stage}"


def _stages(client: Any, domain: str) -> list[int]:
    for dataset in client.benchmarks()["datasets"]:
        if dataset["dataset"] != DATASET:
            continue
        for benchmark in dataset["benchmarks"]:
            if benchmark["benchmark"] != BENCHMARK:
                continue
            for item in benchmark["domains"]:
                if item["domain"] == domain:
                    stages = sorted(int(row["version"]) for row in item["versions"])
                    if stages:
                        return stages
    raise RuntimeError(f"EOG domain {domain!r} is not in the service catalog")


def _active_state(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "state/active.json"
    if not path.exists():
        return None
    state = _read_json(path)
    checkout = run_dir / "state/active"
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if actual != state["commit"]:
        raise RuntimeError(f"active checkout is at {actual}, but active.json records {state['commit']}")
    return state


def _stage_commit(run_dir: Path, stage: int) -> str:
    path = run_dir / "command-agent/adaptation" / f"stage-{stage:03d}" / "status.json"
    if not path.exists():
        raise RuntimeError(f"stage {stage} adaptation status is missing")
    status = _read_json(path)
    if status.get("phase") != "complete" or not status.get("active_commit"):
        raise RuntimeError(f"stage {stage} adaptation is not complete")
    return str(status["active_commit"])


def _optimizer_process_is_running(pid: Any) -> bool:
    """Recognize the optimizer recorded by a possibly stale status file."""
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # existence check only; sends no signal
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A PID may have been reused. When it exists, inspect its command so an
    # unrelated process does not block a stale adaptation retry.
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "stat=", "-o", "command="],
                                capture_output=True, text=True, check=False)
    except OSError:
        return True  # fail closed when process identity cannot be checked
    if result.returncode or not result.stdout.strip():
        return True
    status, _, command = result.stdout.strip().partition(" ")
    if status.startswith("Z"):
        return False
    return "relai" in command.lower() and "optimize" in command.lower()


def _archive_incomplete_adaptation(run_dir: Path, stage: int,
                                   previous_stage: int | None,
                                   status: dict[str, Any]) -> Path:
    """Remove one stale attempt from the active path while retaining its evidence."""
    stage_dir = run_dir / "command-agent/adaptation" / f"stage-{stage:03d}"
    if not stage_dir.is_dir() or not (stage_dir / "status.json").is_file():
        raise RuntimeError(f"cannot retry stage {stage}: adaptation status is missing")
    if status.get("phase") == "complete":
        raise RuntimeError(f"stage {stage} is already complete")
    if status.get("stage") not in (None, stage):
        raise RuntimeError(f"stage {stage} status identifies another stage")
    pid = status.get("optimizer_pid")
    if _optimizer_process_is_running(pid):
        raise RuntimeError(f"stage {stage} optimizer PID {pid} is still running; wait for it to finish")
    active = _active_state(run_dir)
    if previous_stage is None:
        if active is not None:
            raise RuntimeError(f"stage {stage} retry requires no completed active stage; found {active}")
        previous_commit = None
    else:
        previous_commit = _stage_commit(run_dir, previous_stage)
        if active is None or active.get("stage") != previous_stage or active.get("commit") != previous_commit:
            raise RuntimeError(f"stage {stage} retry requires the completed stage {previous_stage} "
                               f"checkout {previous_commit}; active state is {active}")
    attempts = stage_dir.parent / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = attempts / f"stage-{stage:03d}-{stamp}-{uuid.uuid4().hex[:8]}"
    _write_json(stage_dir / "retry-archive.json", {
        "archived_at": _utc_now(), "stage": stage,
        "original_phase": status.get("phase"), "original_optimizer_pid": pid,
        "previous_stage": previous_stage, "previous_agent_commit": previous_commit,
        "partial_relai_memory_retained": True,
        "note": "The optimizer attempt was archived; RELAI memory and optimizer state were not reset.",
    })
    stage_dir.rename(archive)
    return archive


def _safe_task_id(task_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
        raise ValueError(f"invalid task ID in service cohort: {task_id!r}")
    return task_id


def _manifest(run_dir: Path, client: Any, kind: str, cohort: int | str,
              at_stage: int | str, domain: str, limit: int | None) -> dict[str, Any]:
    path = run_dir / "evaluation" / _key(kind, cohort, at_stage) / "manifest.json"
    if path.exists():
        record = _read_json(path)
        expected = {"kind": kind, "cohort": cohort, "at_stage": at_stage,
                    "domain": domain, "limit": limit}
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"cell manifest does not match this run: {path}")
        return record
    ids = client.task_ids(DATASET, BENCHMARK, cohort, "test", domain, limit=limit)
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError(f"cell {_key(kind, cohort, at_stage)} has empty or duplicate task IDs")
    for task_id in ids:
        _safe_task_id(task_id)
    record = {"kind": kind, "cohort": cohort, "at_stage": at_stage,
              "domain": domain, "limit": limit, "task_ids": ids, "created_at": _utc_now()}
    _write_json(path, record)
    return record


def _historical(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError("historical scores must be a JSON array")
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in ("matrix", "before_adapt"):
            raise ValueError("historical rows need kind 'matrix' or 'before_adapt'")
        cohort, at_stage = row.get("cohort"), row.get("at_stage")
        n_tasks, accuracy = row.get("n_tasks"), row.get("accuracy")
        if type(cohort) is not int or type(at_stage) is not int or cohort < 1 or at_stage < cohort:
            raise ValueError(f"invalid historical cell: {row}")
        if type(n_tasks) is not int or n_tasks < 1:
            raise ValueError(f"invalid historical task count: {row}")
        if type(accuracy) not in (float, int) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"invalid historical accuracy: {row}")
        if row["kind"] == "before_adapt" and cohort != at_stage:
            raise ValueError("before-adaptation cohort and harness stage must match")
        key = _key(row["kind"], cohort, at_stage)
        if key in found:
            raise ValueError(f"duplicate historical score for {key}")
        found[key] = {**row, "source": "terminal_rounded", "exact": False}
    return found


def _transient(error: Exception, run_dir: Path) -> bool:
    if isinstance(error, TransportError):
        return True
    if isinstance(error, ServiceError):
        return error.status_code in (429, 500, 502, 503, 504) or (
            error.status_code == 404 and ("<html" in error.detail.lower() or "ngrok" in error.detail.lower()))
    if isinstance(error, CommandAgentError):
        # The solve subprocess turns hosted MCP failures into exit code 1.
        # Read only the task stderr named by CommandAgent, and retry only
        # recognizable HTTP failures on the hosted session endpoint.
        marker = "; see "
        if marker not in str(error):
            return False
        path = Path(str(error).split(marker, 1)[1]).resolve()
        if run_dir.resolve() not in path.parents or path.name != "stderr.log" or not path.is_file():
            return False
        log = path.read_text(errors="replace")
        return "/v1/sessions/" in log and bool(re.search(
            r"(?:HTTPStatusError|ServiceError):[^\n]*\b(?:429|500|502|503|504)\b", log))
    return False


def _task_result(client: Any, agent: Any, run_dir: Path, manifest: dict[str, Any],
                 task_id: str, commit: str, checkpoint: Path) -> dict[str, Any]:
    cohort, at_stage = manifest["cohort"], manifest["at_stage"]
    task = client.task(DATASET, BENCHMARK, cohort, task_id, "test", manifest["domain"],
                       RESOURCE_MODE, **({"at_stage": at_stage} if cohort != "full" else {}))
    workspaces = agent.output_dir / "tasks"
    before = set(workspaces.iterdir()) if workspaces.exists() else set()
    try:
        task.start()
        started = time.monotonic()
        usage = agent(task)
        elapsed = time.monotonic() - started
        grade = task.grade(keep_alive=True)
        new = ([p for p in workspaces.iterdir() if p.is_dir() and p not in before]
               if workspaces.exists() else [])
        trace = None
        if len(new) == 1 and (new[0] / "output/trace.json").is_file():
            trace = os.path.relpath(new[0] / "output/trace.json", run_dir)
        row = {
            "task_id": task_id, "score": float(grade.pass_rate),
            "solved": bool(grade.overall_success), "n_passed": grade.n_passed,
            "n_total": grade.n_total, "per_verifier": list(grade.per_verifier or []),
            "grade_raw": grade.raw, "latency_s": float(usage.get("latency_s", elapsed)),
            "total_tokens": usage.get("total_tokens"), "n_steps": usage.get("n_steps"),
            "trace_path": trace, "agent_commit": commit, "graded_at": _utc_now(),
            "source": "checkpointed_evaluation",
        }
        # DELETE can fail when the tunnel goes away, even after a valid grade.
        _write_json(checkpoint, row)
        return row
    finally:
        try:
            task.close()
        except Exception as error:
            print(f"warning: session cleanup failed for {task_id}: {error}", file=sys.stderr, flush=True)


def _measure(client: Any, agent: Any, run_dir: Path, manifest: dict[str, Any],
             task_id: str, commit: str, checkpoint: Path) -> None:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            _task_result(client, agent, run_dir, manifest, task_id, commit, checkpoint)
            return
        except Exception as error:
            if checkpoint.exists():
                return
            if attempt == RETRY_ATTEMPTS or not _transient(error, run_dir):
                raise
            delay = RETRY_DELAY_S * 2 ** (attempt - 1)
            print(f"  hosted service error for {task_id}; retry {attempt}/{RETRY_ATTEMPTS} "
                  f"in {delay}s", flush=True)
            time.sleep(delay)


def _cell_result(run_dir: Path, manifest: dict[str, Any],
                 historical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    kind, cohort, at_stage = (manifest[name] for name in ("kind", "cohort", "at_stage"))
    key = _key(kind, cohort, at_stage)
    ids = manifest["task_ids"]
    paths = [run_dir / "evaluation" / key / "grades" / f"{task_id}.json" for task_id in ids]
    rows = [_read_json(p) for p in paths if p.is_file()]
    if key in historical:
        if rows:
            raise RuntimeError(f"historical score would overwrite checkpointed grades for {key}")
        item = historical[key]
        if item["n_tasks"] != len(ids):
            raise RuntimeError(f"historical count for {key} is {item['n_tasks']}, service has {len(ids)}")
        agent_stage = at_stage - 1 if kind == "before_adapt" else at_stage
        return {"domain": manifest["domain"], "kind": kind, "stage": cohort,
                "at_stage": at_stage, "n_tasks": len(ids), "n_graded": None,
                "accuracy": item["accuracy"], "score_sum": None, "n_success": None,
                "success_rate": None, "per_task": None, "source": "terminal_rounded",
                "precision": "rounded_to_0.001", "agent_commit": _stage_commit(run_dir, agent_stage),
                "exact": False, "complete": True}
    for task_id, row in zip((p.stem for p in paths if p.is_file()), rows):
        if row.get("task_id") != task_id:
            raise RuntimeError(f"grade checkpoint has wrong task ID in {key}")
    complete = len(rows) == len(ids)
    score_sum = sum(row["score"] for row in rows)
    n_success = sum(bool(row["solved"]) for row in rows)
    return {"domain": manifest["domain"], "kind": kind,
            "stage": -1 if kind == "last_row" else cohort, "at_stage": at_stage,
            "n_tasks": len(ids), "n_graded": len(rows), "complete": complete,
            "score_sum": score_sum, "accuracy": score_sum / len(ids) if complete else None,
            "n_success": n_success, "success_rate": n_success / len(ids) if complete else None,
            "n_errors": 0, "agent_s": sum(float(row["latency_s"]) for row in rows),
            "total_tokens": sum(int(row["total_tokens"]) for row in rows if row.get("total_tokens") is not None),
            "n_with_tokens": sum(row.get("total_tokens") is not None for row in rows),
            "n_steps": sum(int(row.get("n_steps") or 0) for row in rows),
            "per_task": rows, "source": "checkpointed_evaluation", "exact": True}


def _report(run_dir: Path, domain: str, limit: int | None, stages: list[int],
            historical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifests = sorted((run_dir / "evaluation").glob("*/manifest.json"))
    cells = [_cell_result(run_dir, _read_json(path), historical) for path in manifests]
    matrix = sorted((row for row in cells if row["kind"] == "matrix"),
                    key=lambda row: (row["at_stage"], row["stage"]))
    before = sorted((row for row in cells if row["kind"] == "before_adapt"),
                    key=lambda row: row["stage"])
    last = next((row for row in cells if row["kind"] == "last_row"), None)
    by_cell = {(row["stage"], row["at_stage"]): row for row in matrix if row["complete"]}
    by_before = {row["stage"]: row for row in before if row["complete"]}
    bwt = fwt = None
    if len(stages) > 1 and all((i, stages[-1]) in by_cell and (i, i) in by_cell for i in stages[:-1]):
        weights = [by_cell[(i, stages[-1])]["n_tasks"] for i in stages[:-1]]
        bwt = sum(n * (by_cell[(i, stages[-1])]["accuracy"] - by_cell[(i, i)]["accuracy"])
                  for i, n in zip(stages[:-1], weights)) / sum(weights)
    if len(stages) > 1 and all(i in by_before and (i, i) in by_cell for i in stages[1:]):
        weights = [by_cell[(i, i)]["n_tasks"] for i in stages[1:]]
        fwt = sum(n * (by_cell[(i, i)]["accuracy"] - by_before[i]["accuracy"])
                  for i, n in zip(stages[1:], weights)) / sum(weights)
    complete = bool(last and last["complete"] and
                    (not matrix or (len(matrix) == sum(range(1, len(stages) + 1))
                                    and len(before) == len(stages) - 1
                                    and all(row["complete"] for row in matrix + before))))
    approximate = any(not row["exact"] for row in cells)
    return {"schema_version": 1, "generated_at": _utc_now(),
            "status": "complete" if complete else "incomplete", "mode": "self_evolving",
            "dataset": DATASET, "benchmark": BENCHMARK, "domain": domain,
            "resource_mode": RESOURCE_MODE, "scope_complete": limit is None,
            "matrix": matrix, "before_adapt": before, "last_row": [last] if last else [],
            "n_tasks": last["n_tasks"] if last and last["complete"] else None,
            "accuracy": last["accuracy"] if last else None,
            "success_rate": last["success_rate"] if last else None,
            "n_success": last["n_success"] if last else None,
            "n_errors": last.get("n_errors") if last else None,
            "agent_s": last.get("agent_s") if last else None,
            "total_tokens": last.get("total_tokens") if last else None,
            "n_with_tokens": last.get("n_with_tokens") if last else None,
            "n_steps": last.get("n_steps") if last else None,
            "bwt": bwt, "fwt": fwt,
            "transfer_precision": "approximate_from_rounded_terminal_scores" if approximate else "exact",
            "historical_limitations": (
                "Terminal scores have no saved task grades, verifier details, or success counts."
                if approximate else None),
            "per_stage_acc": {str(i): by_cell[(i, i)]["accuracy"] for i in stages if (i, i) in by_cell}}


def _save_report(run_dir: Path, domain: str, limit: int | None, stages: list[int],
                 historical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = _report(run_dir, domain, limit, stages, historical)
    _write_json(run_dir / "report.json", report)
    return report


def run_self_evolving(client: Any, agent: Any, run_dir: Path, config: dict[str, Any],
                      historical_path: Path | None = None,
                      retry_incomplete_adapt: bool = False) -> dict[str, Any]:
    domain, limit, matrix = config["domain"], config["limit"], config["matrix"]
    stages = _stages(client, domain)
    if matrix:
        client.require_cohort_harness_selection()
    historical = _historical(historical_path)
    used_historical: set[str] = set()
    state = _active_state(run_dir)
    if state is not None and state.get("training_pool") != config["training_pool"]:
        raise RuntimeError("resume training pool differs from saved active state")
    retried_adaptation = False

    def evaluate(kind: str, cohort: int | str, harness: int | str,
                 agent_stage: int) -> None:
        manifest = _manifest(run_dir, client, kind, cohort, harness, domain, limit)
        key = _key(kind, cohort, harness)
        if key in historical:
            used_historical.add(key)
        cell = _cell_result(run_dir, manifest, historical)
        if cell["complete"]:
            if key not in historical:
                expected_commit = _stage_commit(run_dir, agent_stage)
                if any(row.get("agent_commit") != expected_commit for row in cell["per_task"]):
                    raise RuntimeError(f"{key} contains grades from another agent commit")
            source = "terminal score" if key in historical else "checkpoints"
            print(f"  {key}: {cell['n_tasks']} tasks, acc {cell['accuracy']:.3f} ({source})", flush=True)
            _save_report(run_dir, domain, limit, stages, historical)
            return
        active = _active_state(run_dir)
        expected_commit = _stage_commit(run_dir, agent_stage)
        if active is None or active.get("stage") != agent_stage or active.get("commit") != expected_commit:
            raise RuntimeError(f"{key} requires stage {agent_stage} checkout {expected_commit}; "
                               f"active state is {active}")
        root = run_dir / "evaluation" / key / "grades"
        for index, task_id in enumerate(manifest["task_ids"], 1):
            checkpoint = root / f"{task_id}.json"
            if checkpoint.exists():
                if _read_json(checkpoint).get("agent_commit") != expected_commit:
                    raise RuntimeError(f"checkpoint agent commit differs for {task_id}")
                continue
            print(f"  {key} {index}/{len(manifest['task_ids'])} {task_id}", flush=True)
            try:
                _measure(client, agent, run_dir, manifest, task_id, expected_commit, checkpoint)
            finally:
                _save_report(run_dir, domain, limit, stages, historical)
        cell = _cell_result(run_dir, manifest, historical)
        print(f"  {key}: {cell['n_tasks']} tasks, acc {cell['accuracy']:.3f}", flush=True)

    print(f"{DATASET}/{BENCHMARK}  mode=self_evolving  resource={RESOURCE_MODE}  "
          f"domain={domain}", flush=True)
    _save_report(run_dir, domain, limit, stages, historical)
    for index, stage in enumerate(stages):
        if matrix and index:
            evaluate("before_adapt", stage, stage, stages[index - 1])
        status_path = run_dir / "command-agent/adaptation" / f"stage-{stage:03d}" / "status.json"
        status = _read_json(status_path) if status_path.exists() else None
        if status and status.get("phase") == "complete":
            print(f"  adapt {domain} stage v{stage}: already complete", flush=True)
        else:
            if status:
                if not retry_incomplete_adapt or retried_adaptation:
                    raise RuntimeError(f"stage {stage} has an incomplete adaptation; inspect {status_path}")
                archive = _archive_incomplete_adaptation(
                    run_dir, stage, stages[index - 1] if index else None, status)
                retried_adaptation = True
                print(f"  archived incomplete stage v{stage}: {archive}", flush=True)
            active = _active_state(run_dir)
            expected_previous = stages[index - 1] if index else None
            if (index == 0 and active is not None) or (
                index > 0 and (active is None or active.get("stage") != expected_previous)
            ):
                raise RuntimeError(f"cannot begin stage {stage}: unexpected active state {active}")
            print(f"  adapt {domain} stage v{stage}", flush=True)
            train = client.tasks(DATASET, BENCHMARK, stage, "train", domain,
                                 resource_mode=RESOURCE_MODE, limit=limit)
            agent.adapt(stage, train)
        if matrix:
            for cohort in stages[:index + 1]:
                evaluate("matrix", cohort, stage, stage)
    evaluate("last_row", "full", "full", stages[-1])
    unknown = set(historical) - used_historical
    if unknown:
        raise RuntimeError(f"unused historical scores: {sorted(unknown)}")
    return _save_report(run_dir, domain, limit, stages, historical)
