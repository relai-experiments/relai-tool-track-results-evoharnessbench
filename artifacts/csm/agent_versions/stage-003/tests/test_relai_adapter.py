import importlib.util
import json
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ADAPTER = Path(__file__).resolve().parents[1] / ".relai/custom-adaptor/eog-tools"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ADAPTER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def benchmarks(self):
        return {"datasets": [{"dataset": "evovling_tools", "benchmarks": [{
            "benchmark": "eog", "domains": [
                {"domain": "calendar", "versions": [{"version": 1, "n_train": 1}, {"version": 2, "n_train": 1}]},
                {"domain": "email", "versions": [{"version": 1, "n_train": 1}]},
            ],
        }]}]}

    def task_ids(self, dataset, benchmark, version, split, domain):
        assert split == "train"
        return ["same-id"]


class RelaiAdapterTests(unittest.TestCase):
    def test_harness_stage_setting(self):
        adapter = load_script("run_task")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(adapter.harness_stage(1))
        with patch.dict(os.environ, {"EOG_RELAI_HARNESS_STAGE": "2"}):
            self.assertEqual(adapter.harness_stage(1), 2)
            self.assertEqual(adapter.harness_stage(2), 2)
            with self.assertRaisesRegex(ValueError, "cohort stage 3"):
                adapter.harness_stage(3)
        with patch.dict(os.environ, {"EOG_RELAI_HARNESS_STAGE": "invalid"}):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                adapter.harness_stage(1)

    def row(self, domain, version):
        return {"domain": domain, "version": f"v{version}", "split": "train", "task_id": "same-id",
                "system_prompt": "Follow policy.", "user_prompt": "Create an event.",
                "gym_servers_config": '{"x-access-token":"secret-token"}',
                "verifiers": json.dumps([{"name": "Event exists", "description": "Check event state",
                    "validation_config": {"query": "SELECT COUNT(*) FROM events WHERE title='X'",
                                          "expected_value": 1, "comparison_type": "equals"}}])}

    def test_catalog_keeps_stage_and_domain_identity(self):
        generator = load_script("generate_catalog")
        catalog = generator.build_catalog(FakeClient(), lambda domain, version: [self.row(domain, version)])
        self.assertEqual(len(catalog["tasks"]), 3)
        self.assertEqual(len({task["id"] for task in catalog["tasks"]}), 3)
        self.assertEqual({task["metadata"]["split"] for task in catalog["tasks"]}, {"train"})
        for task in catalog["tasks"]:
            metadata = task["metadata"]
            self.assertEqual(task["tags"], generator.task_tags(
                metadata["domain"], metadata["stage"], metadata["sdk_task_id"],
            ))
        self.assertEqual(len({task["tags"][1] for task in catalog["tasks"]}), 3)
        sample = catalog["tasks"][0]
        self.assertEqual(sample["description"], "")
        self.assertIn("System prompt:\nFollow policy.", sample["input_summary"])
        self.assertIn("User prompt:\nCreate an event.", sample["input_summary"])
        self.assertIn("SELECT COUNT(*) FROM events WHERE title='X'", sample["evaluators"][0]["description"])
        self.assertNotIn("secret-token", json.dumps(catalog))

    def test_checked_in_catalog_has_unique_task_tags(self):
        generator = load_script("generate_catalog")
        catalog = json.loads((ADAPTER / "catalog.json").read_text())
        task_tags = []
        for task in catalog["tasks"]:
            metadata = task["metadata"]
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(task["tags"], generator.task_tags(
                metadata["domain"], metadata["stage"], metadata["sdk_task_id"],
            ))
            self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_.-]+", tag)
                                for tag in task["tags"]))
            task_tags.append(task["tags"][1])
        self.assertEqual(len(task_tags), len(set(task_tags)))

    def test_catalog_rejects_wrong_dataset_row(self):
        row = self.row("calendar", 1)
        row["task_id"] = "wrong-id"
        with self.assertRaisesRegex(RuntimeError, "do not match SDK task IDs"):
            load_script("generate_catalog").build_catalog(FakeClient(), lambda _domain, _version: [row])

    def test_zero_score_is_a_scored_evaluation(self):
        adapter = load_script("run_task")
        record = {"agent": {"events": [], "final_message": "no"}, "grade": {
            "score": 0.0, "passed": False, "n_passed": 0, "n_total": 1,
            "per_verifier": [{"name": "booked", "passed": False}],
        }, "error": None}
        result = adapter.to_relai_result("calendar/v1/same-id", record)
        self.assertEqual(result["evaluations"][0]["score"], 0.0)
        self.assertNotIn("failure", result)
        self.assertEqual(result["evaluations"][0]["result"]["per_verifier"][0]["name"], "booked")

    def test_error_has_no_score(self):
        adapter = load_script("run_task")
        result = adapter.to_relai_result("calendar/v1/same-id", {"agent": None, "grade": None, "error": "SDK unavailable"})
        self.assertFalse(result["evaluations"])
        self.assertIn("SDK unavailable", result["failure"]["message"])

    def test_individual_verifiers_preserve_mean_and_feedback(self):
        adapter = load_script("run_task")
        names = ["calendar", "location", "color", "event", "attendees"]
        specs = [{"id": f"eog-verifier-{i:02d}", "name": name} for i, name in enumerate(names, 1)]
        rows = [{"name": name, "passed": i != 4, "score": float(i != 4),
                 "expected": 1, "actual": int(i != 4), "comparison_type": "equals",
                 "description": "", "details": None, "error": None} for i, name in enumerate(names, 1)]
        rows[0]["expected"] = "1"
        rows[0]["normalization"] = (
            "Corrected equals result after canonical numeric scalar comparison."
        )
        record = {"agent": {"events": [], "final_message": "done"}, "grade": {
            "score": 0.8, "passed": False, "n_passed": 4, "n_total": 5, "per_verifier": rows}, "error": None}
        result = adapter.to_relai_result("calendar/v1/same-id", record, specs)
        evaluations = result["evaluations"]
        self.assertEqual(len(evaluations), 5)
        self.assertEqual(
            [item["evaluator_id"] for item in evaluations],
            [spec["id"] for spec in specs],
        )
        self.assertAlmostEqual(sum(item["score"] for item in evaluations) / len(evaluations), 0.8)
        self.assertIn("actual=0", evaluations[3]["feedback"])
        self.assertEqual(evaluations[3]["evaluator_id"], "eog-verifier-04")
        rows[3]["name"] = "unexpected"
        mismatch = adapter.to_relai_result("calendar/v1/same-id", record, specs)
        self.assertFalse(mismatch["evaluations"])
        self.assertEqual(mismatch["failure"]["error_type"], "EogVerifierMismatch")
        rows[3]["name"] = "event"
        rows[3]["error"] = "database unavailable"
        errored = adapter.to_relai_result("calendar/v1/same-id", record, specs)
        self.assertEqual(errored["failure"]["error_type"], "EogVerifierError")
        self.assertEqual(errored["evaluation_errors"][0]["evaluator_id"], "eog-verifier-04")
        rows[3]["error"] = None
        record["grade"]["score"] = 0.6
        bad_mean = adapter.to_relai_result("calendar/v1/same-id", record, specs)
        self.assertEqual(bad_mean["failure"]["error_type"], "EogScoreMismatch")


if __name__ == "__main__":
    unittest.main()
