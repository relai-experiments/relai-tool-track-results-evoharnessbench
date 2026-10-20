"""Stable CommandAgent entry points for an evolving EOG ReAct implementation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DATASET = "evovling_tools"
BENCHMARK = "eog"
HEARTBEAT_SECONDS = 60


def _state_dir() -> Path:
    value = os.environ.get("EVAL_STATE_DIR")
    if not value:
        raise RuntimeError("EVAL_STATE_DIR is required")
    return Path(value).resolve()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _active_checkout(state_dir: Path) -> Path:
    active = state_dir / "active"
    if not active.exists():
        state_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(active), "HEAD"],
                       cwd=REPO, check=True)
    for relative in (
        ".relai/config.toml", ".relai/simulator.env",
        ".relai/.internal/optimizer-scope.json", ".relai/optimizer-scope.json",
    ):
        source, destination = REPO / relative, active / relative
        if not source.exists() or destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    if not (active / "src/eog_tools/agent.py").is_file():
        raise RuntimeError(f"active checkout is missing the ReAct agent: {active}")
    if not (active / ".relai/config.toml").is_file():
        raise RuntimeError("RELAI config is missing; run relai init --bare in the experiment repo")
    if not any((active / path).is_file() for path in (
        ".relai/.internal/optimizer-scope.json", ".relai/optimizer-scope.json",
    )):
        raise RuntimeError("RELAI optimizer scope is missing; configure it before self-evolving evaluation")
    return active


def _agent_module(active: Path) -> Any:
    import eog_tools

    active_package = str(active / "src/eog_tools")
    eog_tools.__path__.insert(0, active_package)
    sys.modules.pop("eog_tools.agent", None)
    return importlib.import_module("eog_tools.agent")


class _AllowedSession:
    def __init__(self, session: Any, allowed: list[str], enforce: bool) -> None:
        self.session = session
        self.allowed = set(allowed)
        self.enforce = enforce

    def list_tools(self) -> list[dict[str, Any]]:
        tools = self.session.list_tools()
        return [tool for tool in tools if tool["name"] in self.allowed] if self.enforce else tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.enforce and name not in self.allowed:
            raise ValueError(f"tool {name!r} is outside the task allowlist")
        return self.session.call_tool(name, arguments)

    def close(self) -> None:
        self.session.close()


class _TaskProxy:
    action_type = "mcp"

    def __init__(self, document: dict[str, Any], session: _AllowedSession) -> None:
        self.system_prompt = document.get("system_prompt") or ""
        self.user_prompt = document.get("user_prompt") or ""
        self.mcp_servers = [object()]
        self._session = session

    def mcp_session(self, _server: Any) -> _AllowedSession:
        return self._session


def solve() -> None:
    from openai import OpenAI
    from simple_agentic_evals import EvalClient, MCPSession

    task = json.loads(Path(os.environ["EVAL_TASK_JSON"]).read_text())
    state = json.loads(Path(os.environ["EVAL_TASK_STATE"]).read_text())
    if task.get("dataset") != DATASET or task.get("benchmark") != BENCHMARK:
        raise RuntimeError("this command only supports EOG tasks")
    active = _active_checkout(_state_dir())
    module = _agent_module(active)
    agent = module.ReactAgent(module.AgentConfig(), OpenAI())
    servers = state.get("servers") or []
    if not servers:
        raise RuntimeError("task state has no MCP server")
    server = servers[0]
    with EvalClient(base_url=state["base_url"]) as client:
        headers_by_server = json.loads(os.environ.get("EVAL_MCP_HEADERS_JSON", "{}"))
        headers = {**client.headers, **(headers_by_server.get(server["name"]) or {})}
        url = client.base_url + server["path"] if server.get("path") else server["url"]
        session = _AllowedSession(
            MCPSession(url, headers=headers), state.get("allowed_tools") or [],
            bool(state.get("enforce_allowlist")),
        )
        result = agent.run(_TaskProxy(task, session))
    output = Path(os.environ["EVAL_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "trace.json").write_text(json.dumps({
        "task_id": task["task_id"], "agent_commit": _git(active, "rev-parse", "HEAD"),
        "agent": result.to_dict(),
    }, indent=2, ensure_ascii=False, default=str) + "\n")
    Path(os.environ["EVAL_USAGE_JSON"]).write_text(json.dumps({
        "total_tokens": result.total_tokens, "n_steps": result.steps,
    }) + "\n")
    if result.status == "error":
        raise RuntimeError(result.error or "ReAct agent failed")


def _selectors(stage_document: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    index = {(item["metadata"]["domain"], item["metadata"]["stage"],
              item["metadata"]["sdk_task_id"]): item["id"] for item in catalog["tasks"]}
    selectors: list[str] = []
    for task in stage_document["tasks"]:
        selector = task["selector"]
        if task["dataset"] != DATASET or task["benchmark"] != BENCHMARK or selector.get("split") != "train":
            raise RuntimeError("adaptation received a task outside the EOG training split")
        key = (selector["domain"], int(selector["version"]), task["task_id"])
        if key not in index:
            raise RuntimeError(f"training task is absent from RELAI catalog: {key!r}")
        selectors.append(f"eog-tools::{index[key]}")
    if not selectors:
        raise RuntimeError("adaptation stage has no training tasks")
    return sorted(set(selectors))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, status: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _training_history(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    history = json.loads(path.read_text())
    return history["stages"]


def _last_event_at(artifact_root: Path) -> str | None:
    events = list(artifact_root.glob("*/events.jsonl"))
    if not events:
        return None
    return datetime.fromtimestamp(max(path.stat().st_mtime for path in events), timezone.utc).isoformat()


def _run_optimizer(command: list[str], active: Path, stage_dir: Path,
                   status: dict[str, Any], status_path: Path) -> int:
    secrets = [os.environ.get(name, "") for name in (
        "EVAL_SERVICE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "HF_TOKEN", "RELAI_API_KEY",
    )]
    secrets = [secret for secret in secrets if secret]
    log_path = stage_dir / "optimizer.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=active, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        status.update(phase="optimizing", optimizer_pid=process.pid,
                      optimizer_log=str(log_path))
        _write_status(status_path, status)

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                for secret in secrets:
                    line = line.replace(secret, "[REDACTED]")
                log.write(line)
                log.flush()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        while True:
            try:
                exit_code = process.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                status.update(heartbeat_at=_utc_now(), elapsed_s=round(time.monotonic() - started),
                              last_event_at=_last_event_at(Path(status["optimizer_artifacts"])))
                _write_status(status_path, status)
        reader.join()
        assert process.stdout is not None
        process.stdout.close()
        status.update(heartbeat_at=_utc_now(), elapsed_s=round(time.monotonic() - started),
                      last_event_at=_last_event_at(Path(status["optimizer_artifacts"])),
                      optimizer_exit_code=exit_code)
        _write_status(status_path, status)
        return exit_code


def adapt() -> None:
    state_dir = _state_dir()
    stage_path = Path(os.environ["EVAL_ADAPT_STAGE_JSON"]).resolve()
    stage_dir = stage_path.parent
    stage = json.loads(stage_path.read_text())
    stage_number = int(stage["stage"])
    training_pool = os.environ.get("EOG_RELAI_TRAINING_POOL", "current")
    if training_pool not in ("current", "cumulative"):
        raise RuntimeError(f"unknown RELAI training pool: {training_pool}")
    status_path = stage_dir / "status.json"
    status: dict[str, Any] = {
        "stage": stage_number, "phase": "preparing", "started_at": _utc_now(),
        "training_pool": training_pool, "harness_stage": stage_number,
        "selected_tasks": [], "optimizer_log": str(stage_dir / "optimizer.log"),
        "optimizer_artifacts": str(stage_dir / "optimizer-artifacts"),
    }
    _write_status(status_path, status)
    try:
        active = _active_checkout(state_dir)
        catalog = json.loads((active / ".relai/custom-adaptor/eog-tools/catalog.json").read_text())
        current_selectors = _selectors(stage, catalog)
        history_path = state_dir / "training-tasks.json"
        history = _training_history(history_path)
        selected_by_stage = {}
        if training_pool == "cumulative":
            missing = [number for number in range(1, stage_number)
                       if str(number) not in history]
            if missing:
                raise RuntimeError(f"cumulative training history is missing stages: {missing}")
            selected_by_stage.update({str(number): history[str(number)]
                                      for number in range(1, stage_number)})
        selected_by_stage[str(stage_number)] = current_selectors
        selectors = sorted({selector for cohort in selected_by_stage.values()
                            for selector in cohort})
        status.update(selected_tasks=selectors,
                      selected_tasks_by_stage=selected_by_stage,
                      selected_task_counts_by_stage={key: len(value)
                                                     for key, value in selected_by_stage.items()})
        _write_status(status_path, status)
        result_path = state_dir / "adapt-results" / f"stage-{stage_number}-{uuid.uuid4().hex}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        status["optimizer_result"] = str(result_path)
        command = [
            "relai", "optimize", "--no-pr", "--skip-final-eval",
            "--result-json", str(result_path),
            "--timeout", f"{os.environ.get('EOG_RELAI_TIMEOUT_MINUTES', '720')}m",
            "--observability-dir", status["optimizer_artifacts"],
            "--emit-events-jsonl",
        ]
        command.extend(["--total-rollouts", os.environ.get("EOG_RELAI_TOTAL_ROLLOUTS", "6")])
        command.extend(["--env", f"EOG_RELAI_HARNESS_STAGE={stage_number}"])
        for selector in selectors:
            command.extend(["--custom-tests", selector])
        exit_code = _run_optimizer(command, active, stage_dir, status, status_path)
        if exit_code:
            raise RuntimeError(f"RELAI optimize exited {exit_code}; see {status['optimizer_log']}")
        status["phase"] = "reading_result"
        _write_status(status_path, status)
        result = json.loads(result_path.read_text())
        status["relai_status"] = result.get("status")
        if result.get("repair_blocker") or result.get("status") not in {
            "accepted", "early_stopped", "terminated_after_step", "no_updates",
            "rejected", "no_accepted_updates",
        }:
            raise RuntimeError(f"RELAI optimization did not complete cleanly: {result.get('status')}")
        if result.get("accepted_updates", 0):
            status["phase"] = "applying_update"
            _write_status(status_path, status)
            worktree = Path(result["worktree_code_path"])
            commit = _git(worktree, "rev-parse", "HEAD")
            _git(active, "switch", "--detach", commit)
        status["active_commit"] = _git(active, "rev-parse", "HEAD")
        history[str(stage_number)] = current_selectors
        _write_status(history_path, {"stages": history})
        (state_dir / "active.json").write_text(json.dumps({
            "stage": stage_number, "commit": status["active_commit"],
            "optimizer_result": str(result_path), "selected_tasks": selectors,
            "training_pool": training_pool,
            "selected_task_counts_by_stage": status["selected_task_counts_by_stage"],
        }, indent=2) + "\n")
        status.update(phase="complete", completed_at=_utc_now())
        _write_status(status_path, status)
    except Exception as exc:
        status.update(phase="failed", completed_at=_utc_now(), error=str(exc))
        _write_status(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("solve", "adapt"))
    args = parser.parse_args()
    if args.command == "solve":
        solve()
    else:
        adapt()


if __name__ == "__main__":
    main()
