"""Snapshot EOG training tasks and their published verifier contracts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen


CATALOG_PATH = Path(__file__).with_name("catalog.json")
DATASET = "evovling_tools"
BENCHMARK = "eog"
DATASET_REPO = "ZixuanKe/evovling_tools"
DATASET_REVISION = "1bf90b328c84117e6060fa00b599c5296d84e73e"


def fetch_rows(domain: str, version: int) -> list[dict[str, Any]]:
    url = (f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}"
           f"/eog/{domain}/v{version}/train.jsonl")
    with urlopen(url, timeout=30) as response:
        return [json.loads(line) for line in response if line.strip()]


def _verifiers(row: dict[str, Any]) -> list[dict[str, Any]]:
    verifiers = row["verifiers"]
    if isinstance(verifiers, str):
        verifiers = json.loads(verifiers)
    if not isinstance(verifiers, list) or not verifiers:
        raise ValueError(f"missing verifiers for {row['task_id']}")
    return verifiers


def _evaluator(verifier: dict[str, Any], index: int) -> dict[str, str]:
    config = verifier["validation_config"]
    name = verifier["name"]
    query = config["query"]
    description = verifier.get("description") or ""
    expected = config["expected_value"]
    comparison = config["comparison_type"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"empty verifier query: {name}")
    return {
        "id": f"eog-verifier-{index:02d}",
        "name": name,
        "description": (f"{name}\n{description}\nSQL: {query}\n"
                        f"Expected: {expected!r}; comparison: {comparison}"),
    }


def task_tags(domain: str, stage: int, sdk_task_id: str) -> list[str]:
    return [f"domain.{domain}", f"task.{domain}.v{stage}.{sdk_task_id}"]


def build_catalog(client: Any, row_loader: Any = fetch_rows) -> dict[str, Any]:
    template = json.loads(CATALOG_PATH.read_text())
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset in client.benchmarks()["datasets"]:
        if dataset["dataset"] != DATASET:
            continue
        for benchmark in dataset["benchmarks"]:
            if benchmark["benchmark"] != BENCHMARK:
                continue
            for domain in benchmark["domains"]:
                name = domain["domain"]
                if not name:
                    continue
                for stage in domain["versions"]:
                    version = stage["version"]
                    if not isinstance(version, int):
                        continue
                    ids = client.task_ids(DATASET, BENCHMARK, version, "train", name)
                    expected = stage.get("n_train")
                    if expected is not None and len(ids) != expected:
                        raise RuntimeError(f"{name}/v{version}: expected {expected} training tasks, got {len(ids)}")
                    rows = row_loader(name, version)
                    by_id = {row["task_id"]: row for row in rows}
                    if len(by_id) != len(rows) or set(by_id) != set(ids):
                        raise RuntimeError(f"{name}/v{version}: published training rows do not match SDK task IDs")
                    for sdk_id in ids:
                        catalog_id = f"{name}/v{version}/{sdk_id}"
                        if catalog_id in seen:
                            raise RuntimeError(f"duplicate catalog task ID: {catalog_id}")
                        seen.add(catalog_id)
                        row = by_id[sdk_id]
                        if row.get("domain") != name or row.get("version") != f"v{version}" or row.get("split") != "train":
                            raise RuntimeError(f"{catalog_id}: published row coordinates do not match SDK")
                        system_prompt, user_prompt = row["system_prompt"], row["user_prompt"]
                        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str) or not user_prompt.strip():
                            raise RuntimeError(f"{catalog_id}: missing task prompts")
                        tasks.append({
                            "id": catalog_id,
                            "tags": task_tags(name, version, sdk_id),
                            "description": "",
                            "input_summary": f"System prompt:\n{system_prompt}\n\nUser prompt:\n{user_prompt}",
                            "evaluators": [_evaluator(verifier, i) for i, verifier in enumerate(_verifiers(row), 1)],
                            "metadata": {"dataset": DATASET, "benchmark": BENCHMARK, "domain": name, "stage": version, "split": "train", "sdk_task_id": sdk_id, "dataset_revision": DATASET_REVISION},
                        })
    if not tasks:
        raise RuntimeError("EOG catalog contains no training tasks")
    template["tasks"] = sorted(tasks, key=lambda task: task["id"])
    return template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-existing-catalog", action="store_true", help="use existing SDK ID snapshot when the service key is unavailable")
    args = parser.parse_args()
    if args.from_existing_catalog:
        existing = json.loads(CATALOG_PATH.read_text())
        cohorts: dict[tuple[str, int], set[str]] = defaultdict(set)
        for entry in existing["tasks"]:
            meta = entry["metadata"]
            if meta["split"] != "train":
                raise RuntimeError("existing catalog contains non-training tasks")
            cohorts[(meta["domain"], meta["stage"])].add(meta["sdk_task_id"])

        class SnapshotClient:
            def benchmarks(self) -> dict[str, Any]:
                domains = defaultdict(list)
                for (domain, version), ids in cohorts.items():
                    domains[domain].append({"version": version, "n_train": len(ids)})
                return {"datasets": [{"dataset": DATASET, "benchmarks": [{"benchmark": BENCHMARK,
                        "domains": [{"domain": domain, "versions": versions} for domain, versions in domains.items()]}]}]}

            def task_ids(self, dataset: str, benchmark: str, version: int, split: str, domain: str) -> list[str]:
                return sorted(cohorts[(domain, version)])

        catalog = build_catalog(SnapshotClient())
    else:
        from simple_agentic_evals import EvalClient

        client = EvalClient()
        try:
            catalog = build_catalog(client)
        finally:
            client.close()
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {len(catalog['tasks'])} EOG training tasks to {CATALOG_PATH}")


if __name__ == "__main__":
    main()
