import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eog_tools.agent import AgentConfig, AgentResult
from eog_tools.runner import run_deployment, run_task
from simple_agentic_evals.client import SelectorCompatibilityError


class FakeTask:
    task_id = "sample-1"

    def __init__(self):
        self.closed = False
        self.graded = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def grade(self):
        self.graded = True
        return SimpleNamespace(
            pass_rate=0.5, overall_success=False, n_passed=1,
            n_total=2, per_verifier=[{"name": "first", "passed": True}],
        )


class RunnerTests(unittest.TestCase):
    def test_single_task_writes_trace_and_score(self):
        task = FakeTask()
        eval_client = SimpleNamespace(task=lambda *args, **kwargs: task)
        agent = SimpleNamespace(
            config=AgentConfig(),
            run=lambda task: AgentResult(status="done", steps=1, total_tokens=9),
        )
        with tempfile.TemporaryDirectory() as directory:
            record = run_task(
                eval_client, agent, domain="calendar", version=1,
                split="train", task_id="sample-1", run_dir=Path(directory),
            )
            saved = json.loads((Path(directory) / "result.json").read_text())
        self.assertEqual(record["grade"]["score"], 0.5)
        self.assertFalse(saved["grade"]["passed"])
        self.assertEqual(saved["agent"]["total_tokens"], 9)
        self.assertTrue(task.graded)
        self.assertTrue(task.closed)

    def test_agent_failure_is_not_scored_as_zero(self):
        task = FakeTask()
        eval_client = SimpleNamespace(task=lambda *args, **kwargs: task)
        agent = SimpleNamespace(
            config=AgentConfig(),
            run=lambda task: AgentResult(status="error", error="model unavailable"),
        )
        with tempfile.TemporaryDirectory() as directory:
            record = run_task(
                eval_client, agent, domain="hr", version=1,
                split="train", task_id="sample-1", run_dir=Path(directory),
            )
        self.assertIsNone(record["grade"])
        self.assertEqual(record["error"], "model unavailable")
        self.assertFalse(task.graded)
        self.assertTrue(task.closed)

    def test_prior_cohort_uses_selected_harness_stage(self):
        task = FakeTask()
        calls = []

        def select_task(*args, **kwargs):
            calls.append((args, kwargs))
            return task

        agent = SimpleNamespace(
            config=AgentConfig(),
            run=lambda _: AgentResult(status="done", steps=1, total_tokens=9),
        )
        with tempfile.TemporaryDirectory() as directory:
            record = run_task(
                SimpleNamespace(task=select_task), agent, domain="calendar", version=1,
                split="train", task_id="sample-1", run_dir=Path(directory), at_stage=2,
            )
        self.assertEqual(calls[0][0][2], 1)
        self.assertEqual(calls[0][1]["at_stage"], 2)
        self.assertEqual(record["task"]["at_stage"], 2)

    def test_default_harness_omits_explicit_stage(self):
        calls = []

        def select_task(*args, **kwargs):
            calls.append(kwargs)
            return FakeTask()

        agent = SimpleNamespace(
            config=AgentConfig(),
            run=lambda _: AgentResult(status="done", steps=1, total_tokens=9),
        )
        with tempfile.TemporaryDirectory() as directory:
            record = run_task(
                SimpleNamespace(task=select_task), agent, domain="calendar", version=1,
                split="train", task_id="sample-1", run_dir=Path(directory),
            )
        self.assertNotIn("at_stage", calls[0])
        self.assertEqual(record["task"]["at_stage"], 1)

    def test_selector_compatibility_failure_is_not_scored(self):
        class IncompatibleTask(FakeTask):
            def __enter__(self):
                raise SelectorCompatibilityError("service ignored at_stage")

        agent = SimpleNamespace(config=AgentConfig(), run=lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(SelectorCompatibilityError, "ignored at_stage"):
                run_task(
                    SimpleNamespace(task=lambda *args, **kwargs: IncompatibleTask()),
                    agent, domain="calendar", version=1, split="train",
                    task_id="sample-1", run_dir=root, at_stage=2,
                )
            saved = json.loads((root / "result.json").read_text())
        self.assertIsNone(saved["grade"])
        self.assertIn("SelectorCompatibilityError", saved["error"])

    def test_deployment_attaches_sdk_grade_to_local_trace(self):
        @dataclass
        class Report:
            accuracy: float = 0.5
            success_rate: float = 0.0
            n_tasks: int = 1
            n_errors: int = 0
            task_results: tuple = ({
                "task_id": "sample-1", "score": 0.5, "solved": False,
                "per_verifier": [{"name": "first", "passed": True}],
            },)

        agent = SimpleNamespace(
            config=AgentConfig(),
            run=lambda task: AgentResult(status="done", steps=2, total_tokens=9),
        )

        def fake_benchmark(callable_agent, mode, **kwargs):
            self.assertEqual(mode, "deployment_eval")
            self.assertEqual(kwargs["limit"], 1)
            self.assertEqual(kwargs["on_error"], "raise")
            self.assertEqual(callable_agent(SimpleNamespace(task_id="sample-1")),
                             {"total_tokens": 9, "n_steps": 2})
            return Report()

        with tempfile.TemporaryDirectory() as directory:
            with patch("simple_agentic_evals.run_benchmark", side_effect=fake_benchmark):
                record = run_deployment(
                    SimpleNamespace(), agent, domain="calendar", limit=1,
                    run_dir=Path(directory),
                )
            traces = list((Path(directory) / "traces").glob("*.json"))
            saved = json.loads(traces[0].read_text())
        self.assertEqual(record["report"]["accuracy"], 0.5)
        self.assertEqual(len(traces), 1)
        self.assertEqual(saved["grade"]["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
