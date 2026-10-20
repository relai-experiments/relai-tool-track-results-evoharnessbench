"""Offline checks for checkpointed self-evolving evaluation."""

import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simple_agentic_evals.command_agent import CommandAgentError

from eog_tools import self_evolving as evolving
from eog_tools.cli import main as cli_main


class FakeClient:
    def __init__(self):
        self.task_calls = []

    def benchmarks(self):
        return {"datasets": [{"dataset": "evovling_tools", "benchmarks": [{
            "benchmark": "eog", "domains": [{"domain": "drive", "versions": [
                {"version": 1}, {"version": 2}, {"version": 3}]}]}]}]}

    def require_cohort_harness_selection(self):
        pass

    def task_ids(self, _dataset, _benchmark, version, _split, _domain, limit=None):
        if version == "full":
            return [f"task-{stage}-{n}" for stage, count in ((1, 29), (2, 10), (3, 4))
                    for n in range(count)]
        return [f"task-{version}-{n}" for n in range({1: 29, 2: 10, 3: 4}[version])]

    def tasks(self, *args, **kwargs):
        self.task_calls.append((args, kwargs))
        return []


class ResumableTests(unittest.TestCase):
    def test_historical_scores_validate_and_report_rounded_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            source = run / "historical.json"
            evolving._write_json(source, [
                {"kind": "matrix", "cohort": 1, "at_stage": 1, "n_tasks": 29,
                 "accuracy": 0.706},
                {"kind": "before_adapt", "cohort": 2, "at_stage": 2,
                 "n_tasks": 10, "accuracy": 0.942},
            ])
            history = evolving._historical(source)
            evolving._write_json(
                run / "command-agent/adaptation/stage-001/status.json",
                {"phase": "complete", "active_commit": "commit-1"})
            manifest = evolving._manifest(run, FakeClient(), "matrix", 1, 1, "drive", None)
            cell = evolving._cell_result(run, manifest, history)
            self.assertEqual(cell["accuracy"], 0.706)
            self.assertIsNone(cell["per_task"])
            report = evolving._report(run, "drive", None, [1, 2, 3], history)
            self.assertEqual(report["transfer_precision"],
                             "approximate_from_rounded_terminal_scores")
            self.assertEqual(report["status"], "incomplete")
            source.write_text('[{"kind":"matrix","cohort":1,"at_stage":1,"n_tasks":29,"accuracy":2}]')
            with self.assertRaisesRegex(ValueError, "accuracy"):
                evolving._historical(source)

    def test_nested_hosted_503_is_transient_but_agent_error_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            stderr = run / "command-agent/tasks/task-1/stderr.log"
            stderr.parent.mkdir(parents=True)
            stderr.write_text("RuntimeError: HTTPStatusError: Server error '503 Service Unavailable' "
                              "for url 'https://service/v1/sessions/session-id/mcp/tool'")
            error = CommandAgentError(f"agent command exited 1; see {stderr}")
            self.assertTrue(evolving._transient(error, run))
            stderr.write_text("RuntimeError: model returned no choices")
            self.assertFalse(evolving._transient(error, run))

    def test_hosted_503_retries_fresh_task_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            stderr = run / "command-agent/tasks/task-1/stderr.log"
            stderr.parent.mkdir(parents=True)
            stderr.write_text("RuntimeError: HTTPStatusError: Server error '503 Service Unavailable' "
                              "for url 'https://service/v1/sessions/session-id/mcp/tool'")
            checkpoint = run / "evaluation/matrix-cohort-1-harness-2/grades/task-1.json"
            calls = []
            def one_attempt(*args):
                calls.append(1)
                if len(calls) == 1:
                    raise CommandAgentError(f"agent command exited 1; see {stderr}")
                evolving._write_json(checkpoint, {"task_id": "task-1", "score": 0.5})
            with patch.object(evolving, "_task_result", side_effect=one_attempt), \
                 patch.object(evolving.time, "sleep") as sleep:
                evolving._measure(None, None, run, {}, "task-1", "commit-2", checkpoint)
            self.assertEqual(len(calls), 2)
            sleep.assert_called_once()

    def test_grade_is_checkpointed_before_failed_cleanup(self):
        class Task:
            def start(self):
                pass

            def grade(self, keep_alive=False):
                assert keep_alive
                return SimpleNamespace(pass_rate=0.75, overall_success=False,
                                       n_passed=3, n_total=4, per_verifier=[],
                                       raw={"pass_rate": 0.75})

            def close(self):
                raise RuntimeError("service unavailable")

        class Client:
            def task(self, *args, **kwargs):
                return Task()

        class Agent:
            def __init__(self, root):
                self.output_dir = root / "command-agent"

            def __call__(self, task):
                return {"latency_s": 2.0, "total_tokens": 100, "n_steps": 4}

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "evaluation/matrix-cohort-1-harness-2/grades/task-1.json"
            manifest = {"cohort": 1, "at_stage": 2, "domain": "drive"}
            evolving._task_result(Client(), Agent(run), run, manifest, "task-1", "commit-2", checkpoint)
            self.assertEqual(json.loads(checkpoint.read_text())["score"], 0.75)

    def test_drive_resume_skips_two_adaptations_and_restores_stage_three_order(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            source = run / "historical-scores.json"
            evolving._write_json(source, [
                {"kind": "matrix", "cohort": 1, "at_stage": 1, "n_tasks": 29, "accuracy": 0.706},
                {"kind": "before_adapt", "cohort": 2, "at_stage": 2,
                 "n_tasks": 10, "accuracy": 0.942},
            ])
            for stage in (1, 2):
                evolving._write_json(
                    run / "command-agent/adaptation" / f"stage-{stage:03d}" / "status.json",
                    {"phase": "complete", "active_commit": f"commit-{stage}"})
            current = {"stage": 2}
            class Agent:
                def __init__(self):
                    self.adapted = []

                def adapt(self, stage, tasks):
                    self.adapted.append(stage)
                    current["stage"] = stage
                    evolving._write_json(
                        run / "command-agent/adaptation" / f"stage-{stage:03d}" / "status.json",
                        {"phase": "complete", "active_commit": f"commit-{stage}"})

            agent, client = Agent(), FakeClient()
            calls = []
            def measure(_client, _agent, root, manifest, task_id, commit, checkpoint):
                calls.append((manifest["kind"], manifest["cohort"], manifest["at_stage"], commit))
                evolving._write_json(checkpoint, {
                    "task_id": task_id, "score": 0.5, "solved": False,
                    "latency_s": 1.0, "total_tokens": 10, "n_steps": 1,
                    "agent_commit": commit, "per_verifier": [],
                })
            config = {"domain": "drive", "limit": None, "matrix": True,
                      "training_pool": "cumulative"}
            with patch.object(evolving, "_active_state", side_effect=lambda root: {
                "stage": current["stage"], "commit": f"commit-{current['stage']}",
                "training_pool": "cumulative"}), patch.object(evolving, "_measure", side_effect=measure):
                with redirect_stdout(io.StringIO()):
                    report = evolving.run_self_evolving(client, agent, run, config, source)
            self.assertEqual(agent.adapted, [3])
            self.assertEqual(calls[0][:3], ("matrix", 1, 2))
            self.assertIn(("before_adapt", 3, 3, "commit-2"), calls)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["accuracy"], 0.5)
            self.assertEqual(report["transfer_precision"],
                             "approximate_from_rounded_terminal_scores")

    def test_incomplete_adaptation_is_archived_without_erasing_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            stage = run / "command-agent/adaptation/stage-002"
            status = {"stage": 2, "phase": "optimizing", "optimizer_pid": 12345}
            evolving._write_json(stage / "status.json", status)
            (stage / "optimizer.log").write_text("accepted update in interrupted attempt\n")
            memory = run / "state/active/.relai/.internal/memory/tags/custom/eog-tools/domain.csm.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("partial stage 2 memory")
            with patch.object(evolving, "_optimizer_process_is_running", return_value=False), \
                 patch.object(evolving, "_active_state", return_value={
                     "stage": 1, "commit": "commit-1"}), \
                 patch.object(evolving, "_stage_commit", return_value="commit-1"):
                archive = evolving._archive_incomplete_adaptation(run, 2, 1, status)
            self.assertFalse(stage.exists())
            self.assertEqual((archive / "optimizer.log").read_text(),
                             "accepted update in interrupted attempt\n")
            self.assertTrue(json.loads((archive / "retry-archive.json").read_text())
                            ["partial_relai_memory_retained"])
            self.assertEqual(memory.read_text(), "partial stage 2 memory")
            failed_retry = {"stage": 2, "phase": "failed", "error": "service unavailable"}
            evolving._write_json(stage / "status.json", failed_retry)
            with patch.object(evolving, "_optimizer_process_is_running", return_value=False), \
                 patch.object(evolving, "_active_state", return_value={
                     "stage": 1, "commit": "commit-1"}), \
                 patch.object(evolving, "_stage_commit", return_value="commit-1"):
                second_archive = evolving._archive_incomplete_adaptation(run, 2, 1, failed_retry)
            self.assertNotEqual(archive, second_archive)
            self.assertTrue((archive / "optimizer.log").exists())
            self.assertTrue((second_archive / "status.json").exists())

    def test_live_optimizer_or_mismatched_active_state_blocks_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            stage = run / "command-agent/adaptation/stage-002"
            status = {"stage": 2, "phase": "optimizing", "optimizer_pid": 12345}
            evolving._write_json(stage / "status.json", status)
            with patch.object(evolving, "_optimizer_process_is_running", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    evolving._archive_incomplete_adaptation(run, 2, 1, status)
            with patch.object(evolving, "_optimizer_process_is_running", return_value=False), \
                 patch.object(evolving, "_active_state", return_value={
                     "stage": 2, "commit": "unexpected"}), \
                 patch.object(evolving, "_stage_commit", return_value="commit-1"):
                with self.assertRaisesRegex(RuntimeError, "requires the completed stage 1"):
                    evolving._archive_incomplete_adaptation(run, 2, 1, status)
            self.assertTrue((stage / "status.json").exists())

    def test_optimizer_pid_probe_handles_missing_and_uninspectable_processes(self):
        with patch.object(evolving.os, "kill", side_effect=ProcessLookupError), \
             patch.object(evolving.subprocess, "run") as process:
            self.assertFalse(evolving._optimizer_process_is_running(12345))
            process.assert_not_called()
        with patch.object(evolving.os, "kill", return_value=None), \
             patch.object(evolving.subprocess, "run", side_effect=PermissionError):
            self.assertTrue(evolving._optimizer_process_is_running(12345))

    def test_retry_option_reruns_only_incomplete_stage_then_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            source = run / "historical-scores.json"
            evolving._write_json(source, [
                {"kind": "matrix", "cohort": 1, "at_stage": 1, "n_tasks": 29, "accuracy": 0.706},
                {"kind": "before_adapt", "cohort": 2, "at_stage": 2,
                 "n_tasks": 10, "accuracy": 0.942},
            ])
            evolving._write_json(
                run / "command-agent/adaptation/stage-001/status.json",
                {"stage": 1, "phase": "complete", "active_commit": "commit-1"})
            stage_two = run / "command-agent/adaptation/stage-002/status.json"
            evolving._write_json(stage_two, {"stage": 2, "phase": "optimizing",
                                                "optimizer_pid": 12345})
            current = {"stage": 1}

            class Agent:
                def __init__(self):
                    self.adapted = []

                def adapt(self, stage, tasks):
                    self.adapted.append(stage)
                    current["stage"] = stage
                    evolving._write_json(
                        run / "command-agent/adaptation" / f"stage-{stage:03d}" / "status.json",
                        {"stage": stage, "phase": "complete",
                         "active_commit": f"commit-{stage}"})

            agent = Agent()

            def measure(_client, _agent, root, manifest, task_id, commit, checkpoint):
                evolving._write_json(checkpoint, {
                    "task_id": task_id, "score": 0.5, "solved": False,
                    "latency_s": 1.0, "total_tokens": 10, "n_steps": 1,
                    "agent_commit": commit, "per_verifier": [],
                })

            with patch.object(evolving, "_optimizer_process_is_running", return_value=False), \
                 patch.object(evolving, "_active_state", side_effect=lambda root: {
                     "stage": current["stage"], "commit": f"commit-{current['stage']}",
                     "training_pool": "cumulative"}), \
                 patch.object(evolving, "_measure", side_effect=measure):
                with redirect_stdout(io.StringIO()):
                    report = evolving.run_self_evolving(
                        FakeClient(), agent, run,
                        {"domain": "drive", "limit": None, "matrix": True,
                         "training_pool": "cumulative"}, source,
                        retry_incomplete_adapt=True)
            self.assertEqual(agent.adapted, [2, 3])
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(list((run / "command-agent/adaptation/attempts").iterdir())), 1)

    def test_legacy_resume_cli_saves_settings_and_historical_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-1"
            run.mkdir()
            source = run / "historical-scores.json"
            source.write_text("[]")
            argv = ["run-self-evolving", "--resume", str(run), "--domain", "drive",
                    "--full", "--matrix", "--total-rollouts", "100",
                    "--training-pool", "cumulative", "--historical-scores", str(source),
                    "--optimize-timeout-minutes", "1440", "--adapt-timeout-minutes", "1470",
                    "--retry-incomplete-adapt"]
            with patch.dict(os.environ, {"EVAL_SERVICE_API_KEY": "test", "OPENAI_API_KEY": "test"}), \
                 patch("eog_tools.bootstrap.require_sdk_version"), \
                 patch("simple_agentic_evals.EvalClient"), \
                 patch("eog_tools.self_evolving.run_self_evolving",
                       return_value={"accuracy": 0.5, "status": "complete"}) as run_eval:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(cli_main(argv), 0)
                    self.assertEqual(cli_main(["run-self-evolving", "--resume", str(run)]), 0)
            config = json.loads((run / "run-config.json").read_text())
            self.assertEqual(config["domain"], "drive")
            self.assertIsNone(config["limit"])
            self.assertEqual(config["total_rollouts"], 100)
            self.assertEqual(config["historical_scores"], str(source.resolve()))
            self.assertEqual(run_eval.call_count, 2)
            self.assertTrue(run_eval.call_args_list[0].kwargs["retry_incomplete_adapt"])
            self.assertFalse(run_eval.call_args_list[1].kwargs["retry_incomplete_adapt"])

    def test_new_run_preserves_current_pool_and_small_smoke_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"EVAL_SERVICE_API_KEY": "test", "OPENAI_API_KEY": "test"}), \
                 patch("eog_tools.bootstrap.require_sdk_version"), \
                 patch("simple_agentic_evals.EvalClient"), \
                 patch("eog_tools.self_evolving.run_self_evolving",
                       return_value={"accuracy": 0.5, "status": "complete"}):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(cli_main(["run-self-evolving", "--domain", "drive",
                                               "--runs-dir", directory]), 0)
            run = next(Path(directory).iterdir())
            config = json.loads((run / "run-config.json").read_text())
            self.assertEqual(config["training_pool"], "current")
            self.assertEqual(config["limit"], 1)
            self.assertFalse(config["matrix"])


if __name__ == "__main__":
    unittest.main()
