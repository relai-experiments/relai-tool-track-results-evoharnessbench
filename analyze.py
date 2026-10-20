"""Recompute the RELAI score tables from this repository's copied artifacts.

Run: python3 analyze.py [--check]
The optional check verifies artifact checksums, grade counts, trace identities,
and the published per-cell and full-evaluation scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(report: dict) -> list[dict]:
    final = report.get("last_row") or []
    if isinstance(final, dict):
        final = [final]
    return [*(report.get("matrix") or []), *(report.get("before_adapt") or []), *final]


def score(task: dict) -> float:
    return task["score"] if "score" in task else task["pass_rate"]


def cell_name(row: dict) -> str:
    if row["kind"] in {"last_row", "final"}:
        return "last-row-full"
    cohort = row.get("cohort", row.get("stage"))
    return f"{row['kind']}-cohort-{cohort}-harness-{row['at_stage']}"


def grade_dir(run_root: Path, domain: str, row: dict) -> Path:
    if domain == "calendar":
        name = "final-full" if row["kind"] == "final" else f"cohort-{row['cohort']}-harness-{row['at_stage']}"
        return run_root / "recovery" / "results" / name
    return run_root / "evaluation" / cell_name(row) / "grades"


def close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0):
        raise AssertionError(f"{context}: {actual} != {expected}")


def check_checksums() -> int:
    count = 0
    for line in (ROOT / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, path = line.split("  ", 1)
        target = ROOT / path
        if not target.is_file():
            raise AssertionError(f"missing artifact: {path}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            raise AssertionError(f"checksum mismatch: {path}")
        count += 1
    all_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "artifacts").rglob("*") if p.is_file()}
    listed = {line.split("  ", 1)[1] for line in (ROOT / "checksums.sha256").read_text().splitlines()}
    if all_files != listed:
        raise AssertionError("artifact/checksum inventory differs")
    return count


def analyze_run(entry: dict, check: bool) -> tuple[list[float], float, int, int]:
    domain = entry["domain"]
    run_root = ROOT / "artifacts" / domain / "run"
    report = load_json(ROOT / entry["imported_report"])
    if report["domain"] != domain or report["status"] != "complete":
        raise AssertionError(f"wrong report: {domain}")
    matrix = report["matrix"]
    by_stage: dict[int, list[dict]] = defaultdict(list)
    for row in matrix:
        by_stage[int(row["at_stage"])].append(row)
    stage_scores = []
    for stage in range(1, len(entry["stages"]) + 1):
        cells = by_stage[stage]
        if len(cells) != stage:
            raise AssertionError(f"{domain} stage {stage}: expected {stage} matrix cells")
        if {int(c.get("cohort", c.get("stage"))) for c in cells} != set(range(1, stage + 1)):
            raise AssertionError(f"{domain} stage {stage}: wrong cohort inventory")
        n = sum(c["n_tasks"] for c in cells)
        stage_scores.append(sum(c["accuracy"] * c["n_tasks"] for c in cells) / n)
    checked_grades = 0
    checked_traces: set[str] = set()
    for row in rows(report):
        context = f"{domain} {cell_name(row)}"
        tasks = row.get("per_task")
        if tasks is None:
            if row.get("exact") is not False or row.get("precision") != "rounded_to_0.001":
                raise AssertionError(f"{context}: terminal-only cell lacks precision marker")
            continue
        if len(tasks) != row["n_tasks"]:
            raise AssertionError(f"{context}: wrong task count")
        close(sum(score(task) for task in tasks) / len(tasks), row["accuracy"], context)
        if not check:
            continue
        grades = sorted(grade_dir(run_root, domain, row).glob("*.json"))
        if len(grades) != len(tasks):
            raise AssertionError(f"{context}: missing checkpoint grades")
        by_id = {task["task_id"]: task for task in tasks}
        if {path.stem for path in grades} != set(by_id):
            raise AssertionError(f"{context}: checkpoint task IDs differ")
        for path in grades:
            grade = load_json(path)
            task = by_id[path.stem]
            if grade["task_id"] != task["task_id"]:
                raise AssertionError(f"{context}: grade task identity differs")
            close(score(grade), score(task), f"{context} grade {path.stem}")
            trace_rel = Path(task["trace_path"])
            if trace_rel.is_absolute() or ".." in trace_rel.parts:
                raise AssertionError(f"{context}: unsafe trace path")
            if domain == "calendar":
                trace_rel = Path("recovery") / trace_rel
            trace = load_json(run_root / trace_rel)
            if trace.get("task_id") != task["task_id"] or trace.get("agent_commit") != task["agent_commit"]:
                raise AssertionError(f"{context}: trace identity differs")
            checked_grades += 1
            checked_traces.add(trace_rel.as_posix())
    final = report["last_row"]
    if isinstance(final, list):
        if len(final) != 1:
            raise AssertionError(f"{domain}: expected one final row")
        final = final[0]
    close(final["accuracy"], report["accuracy"], f"{domain} full evaluation")
    if check:
        if checked_traces != {
            p.relative_to(run_root).as_posix()
            for p in (run_root / "command-agent" / "tasks").glob("*/output/trace.json")
        } | {
            p.relative_to(run_root).as_posix()
            for p in (run_root / "recovery" / "command-agent" / "tasks").glob("*/output/trace.json")
        }:
            raise AssertionError(f"{domain}: trace inventory differs")
        if len(checked_traces) != entry["n_unique_traces"]:
            raise AssertionError(f"{domain}: trace count differs")
    return stage_scores, final["accuracy"], final["n_tasks"], checked_grades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify every copied grade, trace, and checksum")
    args = parser.parse_args()
    manifest = load_json(ROOT / "manifest.json")
    if args.check:
        n_files = check_checksums()
        print(f"Verified {n_files} artifact checksums")
    print("Domain | Weighted matrix scores by stage (%) | Final full eval (%) | Final tasks")
    print("--- | --- | --- | ---")
    for entry in manifest["runs"]:
        stages, final, n_tasks, n_grades = analyze_run(entry, args.check)
        print(f"{entry['domain']} | {', '.join(f'{100 * value:.4f}' for value in stages)} | {100 * final:.4f} | {n_tasks}")
        if args.check:
            print(f"  verified {n_grades} cell grades and {entry['n_unique_traces']} unique traces")


if __name__ == "__main__":
    main()
