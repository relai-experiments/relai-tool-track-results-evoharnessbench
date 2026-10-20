"""Import the four selected EOG runs into this standalone evidence repository.

Run from any directory:
    python3 import_runs.py --tool-track ../tool-track

Only the Python standard library is required. This script reads the source
runs and Git repository, and writes exclusively beneath its own directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = {
    "calendar": ("20261002T195831Z-59fddc8d", "recovery/report.json"),
    "drive": ("20261007T124934Z-54148b3e", "report.json"),
    "csm": ("20261012T154955Z-95d47fda", "report.json"),
    "itsm": ("20261019T013854Z-06abfff3", "report.json"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def source_paths(repo: Path, commit: str) -> list[str]:
    names = git(repo, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    return [
        name
        for name in names
        if name in {"README.md", "pyproject.toml", "uv.lock"}
        or name.startswith("src/")
        or name.startswith("tests/")
        or (
            name.startswith(".relai/custom-adaptor/eog-tools/")
            and name.endswith(".py")
        )
    ]


def export_source(repo: Path, commit: str, destination: Path) -> None:
    paths = source_paths(repo, commit)
    if not any(path.endswith("src/eog_tools/agent.py") for path in paths):
        raise ValueError(f"agent source absent from {commit}")
    for path in paths:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git(repo, "show", f"{commit}:{path}"))


def rows(report: dict) -> list[dict]:
    final = report.get("last_row") or []
    if isinstance(final, dict):
        final = [final]
    return [*(report.get("matrix") or []), *(report.get("before_adapt") or []), *final]


def cell_name(row: dict) -> str:
    if row["kind"] in {"last_row", "final"}:
        return "last-row-full"
    cohort = row.get("cohort", row.get("stage"))
    return f"{row['kind']}-cohort-{cohort}-harness-{row['at_stage']}"


def import_run(domain: str, run_id: str, report_path: str, runs_root: Path, repo: Path) -> dict:
    source = runs_root / run_id
    target = ROOT / "artifacts" / domain
    run_target = target / "run"
    code_target = target / "agent_versions"
    if target.exists():
        shutil.rmtree(target)
    report = read_json(source / report_path)
    if report.get("domain") != domain or report.get("status") != "complete":
        raise ValueError(f"unexpected report status or domain for {run_id}")
    copy_file(source / report_path, run_target / report_path)
    for optional in ("run-config.json", "historical-scores.json", "state/training-tasks.json", "state/active.json"):
        if (source / optional).is_file():
            copy_file(source / optional, run_target / optional)

    stages = []
    stage_statuses = sorted((source / "command-agent" / "adaptation").glob("stage-*/status.json"))
    if not stage_statuses:
        raise ValueError(f"no adaptation stages in {run_id}")
    first_commit = read_json(stage_statuses[0])["active_commit"]
    base_commit = git(repo, "rev-parse", f"{first_commit}^").decode().strip()
    export_source(repo, base_commit, code_target / "initial")
    for status_path in stage_statuses:
        status = read_json(status_path)
        if status.get("phase") != "complete":
            raise ValueError(f"incomplete adaptation: {status_path}")
        stage = int(status["stage"])
        commit = status["active_commit"]
        git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
        branches = git(repo, "branch", "--format=%(refname:short)", "--points-at", commit).decode().splitlines()
        if not branches:
            raise ValueError(f"stage commit lacks a branch: {commit}")
        stage_dir = status_path.parent
        stage_rel = status_path.relative_to(source)
        copy_file(status_path, run_target / stage_rel)
        for filename in ("stage.json",):
            if (stage_dir / filename).is_file():
                copy_file(stage_dir / filename, run_target / stage_rel.parent / filename)
        optimizer_reports = sorted((stage_dir / "optimizer-artifacts").glob("*/optimization-report.json"))
        if len(optimizer_reports) != 1:
            raise ValueError(f"expected one optimizer report for {stage_dir}")
        optimizer_report = optimizer_reports[0]
        copy_file(optimizer_report, run_target / optimizer_report.relative_to(source))
        for name in ("pr-summary.json", "pr-description.md"):
            path = optimizer_report.parent / name
            if path.is_file():
                copy_file(path, run_target / path.relative_to(source))
        result_path = Path(status["optimizer_result"])
        if not result_path.is_absolute():
            result_path = source / result_path
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result_target = run_target / "state" / "adapt-results" / result_path.name
        copy_file(result_path, result_target)
        export_source(repo, commit, code_target / f"stage-{stage:03d}")
        stages.append(
            {
                "stage": stage,
                "commit": commit,
                "branches": branches,
                "selected_training_tasks": len(status["selected_tasks"]),
                "training_tasks_by_stage": status.get("selected_task_counts_by_stage"),
                "status": stage_rel.as_posix(),
                "optimizer_report": optimizer_report.relative_to(source).as_posix(),
                "optimizer_result": result_target.relative_to(run_target).as_posix(),
                "source_snapshot": f"agent_versions/stage-{stage:03d}",
            }
        )

    copied_traces: set[str] = set()
    cells = []
    for row in rows(report):
        name = cell_name(row)
        tasks = row.get("per_task")
        if tasks is None:
            if row.get("exact") is not False or row.get("precision") != "rounded_to_0.001":
                raise ValueError(f"missing task grades without terminal precision marker: {domain} {name}")
        elif len(tasks) != row["n_tasks"]:
            raise ValueError(f"task count mismatch: {domain} {name}")
        for task in tasks or []:
            trace_rel = Path(task["trace_path"])
            if trace_rel.is_absolute() or ".." in trace_rel.parts:
                raise ValueError(f"unsafe trace path: {trace_rel}")
            # Calendar's recovery report records paths relative to recovery/.
            if domain == "calendar":
                trace_rel = Path("recovery") / trace_rel
            if trace_rel.as_posix() not in copied_traces:
                copy_file(source / trace_rel, run_target / trace_rel)
                copied_traces.add(trace_rel.as_posix())
            trace = read_json(source / trace_rel)
            if trace.get("task_id") != task["task_id"] or trace.get("agent_commit") != task["agent_commit"]:
                raise ValueError(f"trace identity mismatch: {trace_rel}")
        if domain == "calendar" and tasks:
            grade_dir = source / "recovery" / "results" / (
                "final-full" if row["kind"] == "final" else f"cohort-{row['cohort']}-harness-{row['at_stage']}"
            )
            grade_files = sorted(grade_dir.glob("*.json"))
        elif tasks:
            grade_dir = source / "evaluation" / name / "grades"
            grade_files = sorted(grade_dir.glob("*.json"))
        else:
            grade_files = []
        if len(grade_files) != len(tasks or []):
            raise ValueError(f"checkpoint grade count mismatch: {domain} {name}")
        if {p.stem for p in grade_files} != {task["task_id"] for task in tasks or []}:
            raise ValueError(f"checkpoint task IDs mismatch: {domain} {name}")
        for grade_file in grade_files:
            copy_file(grade_file, run_target / grade_file.relative_to(source))
        cells.append(
            {
                "kind": row["kind"],
                "cohort": row.get("cohort", row.get("stage", "full")),
                "at_stage": row["at_stage"],
                "n_tasks": row["n_tasks"],
                "n_task_grades": len(tasks or []),
                "n_checkpoint_grades": len(grade_files),
                "accuracy": row["accuracy"],
                "source": row.get("source"),
                "exact": row.get("exact"),
                "precision": row.get("precision"),
            }
        )
    return {
        "domain": domain,
        "run_id": run_id,
        "source_report": report_path,
        "imported_report": (Path("artifacts") / domain / "run" / report_path).as_posix(),
        "initial_commit": base_commit,
        "initial_source_snapshot": "agent_versions/initial",
        "stages": stages,
        "cells": cells,
        "n_unique_traces": len(copied_traces),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums() -> None:
    artifact_root = ROOT / "artifacts"
    files = sorted(path for path in artifact_root.rglob("*") if path.is_file())
    content = "".join(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files)
    (ROOT / "checksums.sha256").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-track", type=Path, default=ROOT.parent / "tool-track")
    args = parser.parse_args()
    repo = args.tool_track.resolve()
    if not (repo / ".git").exists():
        parser.error(f"not a Git checkout: {repo}")
    runs_root = repo / "runs"
    manifest = {
        "schema_version": 1,
        "benchmark": "EvoHarnessBench EOG tool track",
        "metric": "mean per-task verifier fraction",
        "runs": [import_run(domain, run_id, report, runs_root, repo) for domain, (run_id, report) in RUNS.items()],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_checksums()
    for run in manifest["runs"]:
        print(f"{run['domain']}: {run['run_id']}, {len(run['cells'])} cells, {run['n_unique_traces']} traces")


if __name__ == "__main__":
    main()
