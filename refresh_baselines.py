"""Import the pinned EvoHarnessBench tool baseline stage scores.

Run: python3 refresh_baselines.py [--check]
The site's JavaScript embeds a JSON object assigned to LB_BASELINE_DATA.
Only six cumulative-harness EOG methods and four domains are retained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://mas-orchestra.salesforceresearch.ai/evoharness/static/baselines/data.js"
SOURCE_SHA256 = "a1ea84cfbccbedf721acb6e1ef37b2b026361696db477821ef11b6ebf0f81e06"
MARKER = "const LB_BASELINE_DATA = "
DOMAINS = ("calendar", "drive", "csm", "itsm")
SYSTEMS = (
    ("tools-react", "ReAct/Codex deployment"),
    ("tools-raw-memory", "Raw Memory"),
    ("tools-reasoning-bank", "Reasoning Bank"),
    ("tools-memtoolagent", "MemToolAgent"),
    ("tools-gepa", "GEPA"),
    ("tools-meta-harness", "Meta-Harness"),
)


def download() -> bytes:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "relai-eog-results/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def extract(raw: bytes) -> dict:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"source changed: SHA-256 {digest}; inspect the new file before updating the pin")
    source = raw.decode("utf-8")
    if source.count(MARKER) != 1:
        raise ValueError("baseline data assignment not found exactly once")
    data = json.JSONDecoder().raw_decode(source.split(MARKER, 1)[1])[0]
    if data.get("schema") != 3 or data.get("snapshot") != "August 2026":
        raise ValueError("unexpected baseline schema or snapshot")
    extracted = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_schema": data["schema"],
        "source_snapshot": data["snapshot"],
        "metric": "Score: mean fraction of task verifiers passed, percent",
        "uncertainty": "Population standard deviation over three runs",
        "domains": list(DOMAINS),
        "systems": [],
    }
    for key, label in SYSTEMS:
        system = data["systems"][key]
        if system.get("track") != "tools" or system.get("kind") != "cumulative":
            raise ValueError(f"wrong track or harness kind for {key}")
        streams = {}
        for domain in DOMAINS:
            stage_rows = system["streams"][domain]["stages"]
            if len(stage_rows) not in (3, 4):
                raise ValueError(f"unexpected stage count: {key}/{domain}")
            stages = []
            for expected_stage, row in enumerate(stage_rows, 1):
                if row["stage"] != expected_stage or row["cohorts"] != list(range(1, expected_stage + 1)):
                    raise ValueError(f"non-cumulative stage: {key}/{domain}/{expected_stage}")
                if row["runs"] != 3 or len(row["tasks"]) != 3:
                    raise ValueError(f"unexpected run count: {key}/{domain}/{expected_stage}")
                if not (0 <= row["score"] <= 100 and row["scoreSd"] >= 0):
                    raise ValueError(f"invalid score: {key}/{domain}/{expected_stage}")
                stages.append({
                    "stage": expected_stage,
                    "score": row["score"],
                    "score_sd": row["scoreSd"],
                    "runs": row["runs"],
                    "tasks_per_run": row["tasks"],
                })
            streams[domain] = stages
        extracted["systems"].append({"source_id": key, "name": label, "stages": streams})
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the pinned source with the committed data")
    args = parser.parse_args()
    extracted = extract(download())
    destination = ROOT / "baseline_scores.json"
    content = json.dumps(extracted, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if destination.read_text(encoding="utf-8") != content:
            raise SystemExit("baseline_scores.json differs from the pinned source")
        print("Baseline scores match the pinned website snapshot")
    else:
        destination.write_text(content, encoding="utf-8")
        print(f"Wrote {destination.name}: {len(extracted['systems'])} systems × {len(DOMAINS)} domains")


if __name__ == "__main__":
    main()
