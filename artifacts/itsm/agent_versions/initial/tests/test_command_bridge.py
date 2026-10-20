import json
import os
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from eog_tools.cli import build_parser
from eog_tools.command_bridge import _AllowedSession, _active_checkout, _agent_module, _run_optimizer, _selectors, adapt


class FakeSession:
    def __init__(self):
        self.calls = []
        self.closed = False

    def list_tools(self):
        return [{"name": "allowed"}, {"name": "hidden"}]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True}

    def close(self):
        self.closed = True


class CommandBridgeTests(unittest.TestCase):
    def test_optimizer_log_and_heartbeat_are_live_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            status_path = root / "status.json"
            status = {"optimizer_artifacts": str(artifacts)}
            script = (
                "import pathlib,sys,time\n"
                "root=pathlib.Path(sys.argv[1])\n"
                "event=root/'run-1'/'events.jsonl'\n"
                "event.parent.mkdir(parents=True)\n"
                "event.write_text('{}\\n')\n"
                "print('secret-value', flush=True)\n"
                "time.sleep(0.3)\n"
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-value"}), \
                 patch("eog_tools.command_bridge.HEARTBEAT_SECONDS", 0.05):
                code = _run_optimizer([sys.executable, "-c", script, str(artifacts)],
                                      root, root, status, status_path)
            self.assertEqual(code, 0)
            self.assertEqual(status["optimizer_exit_code"], 0)
            self.assertIsNotNone(status["last_event_at"])
            self.assertIn("heartbeat_at", status)
            self.assertEqual(json.loads(status_path.read_text())["optimizer_pid"], status["optimizer_pid"])
            self.assertIn("[REDACTED]", (root / "optimizer.log").read_text())
            self.assertNotIn("secret-value", (root / "optimizer.log").read_text())

    def test_adapt_records_failure_and_optimizer_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage-001"
            stage_dir.mkdir()
            stage_path = stage_dir / "stage.json"
            stage_path.write_text(json.dumps({"stage": 1, "tasks": []}))
            active = root / "active"
            catalog_path = active / ".relai/custom-adaptor/eog-tools/catalog.json"
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text('{"tasks": []}')
            env = {"EVAL_STATE_DIR": str(root / "state"),
                   "EVAL_ADAPT_STAGE_JSON": str(stage_path),
                   "EOG_RELAI_TIMEOUT_MINUTES": "42"}
            with patch.dict(os.environ, env), \
                 patch("eog_tools.command_bridge._active_checkout", return_value=active), \
                 patch("eog_tools.command_bridge._selectors", return_value=["eog-tools::one"]), \
                 patch("eog_tools.command_bridge._run_optimizer", return_value=1) as optimizer:
                with self.assertRaisesRegex(RuntimeError, "exited"):
                    adapt()
            command = optimizer.call_args.args[0]
            self.assertEqual(command[command.index("--timeout") + 1], "42m")
            self.assertEqual(command[command.index("--env") + 1], "EOG_RELAI_HARNESS_STAGE=1")
            self.assertIn("--emit-events-jsonl", command)
            self.assertEqual(json.loads((stage_dir / "status.json").read_text())["phase"], "failed")

    def test_adapt_records_completed_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            state_dir.mkdir()
            stage_dir = root / "stage-001"
            stage_dir.mkdir()
            stage_path = stage_dir / "stage.json"
            stage_path.write_text(json.dumps({"stage": 1, "tasks": []}))
            active = root / "active"
            catalog_path = active / ".relai/custom-adaptor/eog-tools/catalog.json"
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text('{"tasks": []}')

            def complete_optimizer(command, *_args):
                Path(command[command.index("--result-json") + 1]).write_text(json.dumps({
                    "status": "no_updates", "accepted_updates": 0,
                }))
                return 0

            with patch.dict(os.environ, {"EVAL_STATE_DIR": str(state_dir),
                                      "EVAL_ADAPT_STAGE_JSON": str(stage_path)}), \
                 patch("eog_tools.command_bridge._active_checkout", return_value=active), \
                 patch("eog_tools.command_bridge._selectors", return_value=["eog-tools::one"]), \
                 patch("eog_tools.command_bridge._run_optimizer", side_effect=complete_optimizer), \
                 patch("eog_tools.command_bridge._git", return_value="abc123"):
                adapt()
            status = json.loads((stage_dir / "status.json").read_text())
            self.assertEqual(status["phase"], "complete")
            self.assertEqual(status["active_commit"], "abc123")
            self.assertEqual(status["selected_tasks"], ["eog-tools::one"])
            self.assertEqual(status["harness_stage"], 1)
            self.assertEqual(json.loads((state_dir / "active.json").read_text())["commit"], "abc123")
            self.assertEqual(json.loads((state_dir / "training-tasks.json").read_text()),
                             {"stages": {"1": ["eog-tools::one"]}})

    def test_cumulative_pool_uses_completed_stages_and_skips_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            active = state_dir / "active"
            catalog_path = active / ".relai/custom-adaptor/eog-tools/catalog.json"
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text(json.dumps({"tasks": [{
                "id": f"calendar/v{stage}/{task_id}",
                "metadata": {"domain": "calendar", "stage": stage,
                             "sdk_task_id": task_id},
            } for stage in (1, 2, 3)
               for task_id in (f"task-{stage}", f"unselected-{stage}")]}))
            commands = []
            fail_stage_two = True

            def run_optimizer(command, *_args):
                nonlocal fail_stage_two
                commands.append(command)
                if fail_stage_two and "stage-2-" in command[command.index("--result-json") + 1]:
                    fail_stage_two = False
                    return 1
                Path(command[command.index("--result-json") + 1]).write_text(json.dumps({
                    "status": "no_updates", "accepted_updates": 0,
                }))
                return 0

            def run_stage(number, pool="cumulative"):
                stage_dir = root / f"stage-{number:03d}"
                stage_dir.mkdir(exist_ok=True)
                stage_path = stage_dir / "stage.json"
                stage_path.write_text(json.dumps({"stage": number, "tasks": [{
                    "dataset": "evovling_tools", "benchmark": "eog",
                    "task_id": f"task-{number}",
                    "selector": {"domain": "calendar", "version": number,
                                 "split": "train"},
                }]}))
                with patch.dict(os.environ, {
                    "EVAL_STATE_DIR": str(state_dir),
                    "EVAL_ADAPT_STAGE_JSON": str(stage_path),
                    "EOG_RELAI_TRAINING_POOL": pool,
                }):
                    adapt()
                return json.loads((stage_dir / "status.json").read_text())

            with patch("eog_tools.command_bridge._active_checkout", return_value=active), \
                 patch("eog_tools.command_bridge._run_optimizer", side_effect=run_optimizer), \
                 patch("eog_tools.command_bridge._git", return_value="abc123"):
                first = run_stage(1)
                with self.assertRaisesRegex(RuntimeError, "exited"):
                    run_stage(2)
                history_path = state_dir / "training-tasks.json"
                self.assertEqual(list(json.loads(history_path.read_text())["stages"]), ["1"])
                second = run_stage(2)
                third = run_stage(3)
                current_only = run_stage(3, "current")

            self.assertEqual(first["selected_task_counts_by_stage"], {"1": 1})
            self.assertEqual(second["selected_task_counts_by_stage"], {"1": 1, "2": 1})
            self.assertEqual(third["selected_task_counts_by_stage"],
                             {"1": 1, "2": 1, "3": 1})
            self.assertEqual(third["selected_tasks"], [
                f"eog-tools::calendar/v{stage}/task-{stage}" for stage in (1, 2, 3)
            ])
            self.assertEqual([sum(arg == "--custom-tests" for arg in command)
                              for command in commands], [1, 2, 2, 3, 1])
            self.assertEqual([command[command.index("--env") + 1] for command in commands], [
                "EOG_RELAI_HARNESS_STAGE=1", "EOG_RELAI_HARNESS_STAGE=2",
                "EOG_RELAI_HARNESS_STAGE=2", "EOG_RELAI_HARNESS_STAGE=3",
                "EOG_RELAI_HARNESS_STAGE=3",
            ])
            self.assertEqual(current_only["selected_task_counts_by_stage"], {"3": 1})
            self.assertEqual(len(json.loads(history_path.read_text())["stages"]), 3)

    def test_cumulative_pool_requires_prior_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage-002"
            stage_dir.mkdir()
            stage_path = stage_dir / "stage.json"
            stage_path.write_text(json.dumps({"stage": 2, "tasks": []}))
            active = root / "active"
            catalog_path = active / ".relai/custom-adaptor/eog-tools/catalog.json"
            catalog_path.parent.mkdir(parents=True)
            catalog_path.write_text('{"tasks": []}')
            with patch.dict(os.environ, {
                "EVAL_STATE_DIR": str(root / "state"),
                "EVAL_ADAPT_STAGE_JSON": str(stage_path),
                "EOG_RELAI_TRAINING_POOL": "cumulative",
            }), patch("eog_tools.command_bridge._active_checkout", return_value=active), \
                 patch("eog_tools.command_bridge._selectors", return_value=["eog-tools::two"]), \
                 patch("eog_tools.command_bridge._run_optimizer") as optimizer:
                with self.assertRaisesRegex(RuntimeError, "missing stages: \\[1\\]"):
                    adapt()
            optimizer.assert_not_called()

    def test_training_pool_cli_defaults_to_current(self):
        parser = build_parser()
        # None means the flag was omitted; a fresh run resolves it to current,
        # while --resume loads the saved pool (which may be cumulative).
        self.assertEqual(parser.parse_args(["run-self-evolving", "--domain", "calendar"]).training_pool,
                         None)
        self.assertEqual(parser.parse_args(["run-self-evolving", "--domain", "calendar",
                                            "--training-pool", "cumulative"]).training_pool,
                         "cumulative")

    def test_active_checkout_copies_config_without_prior_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for command in (["git", "init", "-q"], ["git", "config", "user.name", "Test"],
                            ["git", "config", "user.email", "test@example.com"]):
                subprocess.run(command, cwd=repo, check=True)
            agent = repo / "src/eog_tools/agent.py"
            agent.parent.mkdir(parents=True)
            agent.write_text("agent = True\n")
            (repo / ".gitignore").write_text("runs/\n.relai/config.toml\n.relai/.internal/optimizer-scope.json\n")
            subprocess.run(["git", "add", ".gitignore", "src"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            scope = repo / ".relai/.internal/optimizer-scope.json"
            scope.parent.mkdir(parents=True)
            scope.write_text("{}\n")
            (repo / ".relai/config.toml").write_text("[simulator]\n")
            memory = repo / ".relai/.internal/memory"
            memory.mkdir()
            (memory / "old.md").write_text("prior run\n")
            with patch("eog_tools.command_bridge.REPO", repo):
                active = _active_checkout(repo / "runs/state")
            self.assertTrue((active / ".relai/config.toml").is_file())
            self.assertTrue((active / ".relai/.internal/optimizer-scope.json").is_file())
            self.assertFalse((active / ".relai/.internal/memory/old.md").exists())
            import eog_tools
            prior_path = list(eog_tools.__path__)
            prior_agent = sys.modules.get("eog_tools.agent")
            try:
                self.assertTrue(_agent_module(active).agent)
            finally:
                eog_tools.__path__[:] = prior_path
                sys.modules.pop("eog_tools.agent", None)
                if prior_agent is not None:
                    sys.modules["eog_tools.agent"] = prior_agent

    def test_tool_allowlist_filters_and_rejects_calls(self):
        raw = FakeSession()
        session = _AllowedSession(raw, ["allowed"], True)
        self.assertEqual(session.list_tools(), [{"name": "allowed"}])
        self.assertEqual(session.call_tool("allowed", {}), {"ok": True})
        with self.assertRaises(ValueError):
            session.call_tool("hidden", {})
        session.close()
        self.assertEqual(raw.calls, [("allowed", {})])
        self.assertTrue(raw.closed)

    def test_stage_mapping_uses_domain_stage_and_training_split(self):
        catalog = {"tasks": [{"id": "calendar/v2/task-1", "metadata": {
            "domain": "calendar", "stage": 2, "sdk_task_id": "task-1",
        }}]}
        stage = {"tasks": [{
            "dataset": "evovling_tools", "benchmark": "eog", "task_id": "task-1",
            "selector": {"domain": "calendar", "version": 2, "split": "train"},
        }]}
        self.assertEqual(_selectors(stage, catalog), ["eog-tools::calendar/v2/task-1"])
        stage["tasks"][0]["selector"]["split"] = "test"
        with self.assertRaisesRegex(RuntimeError, "training split"):
            _selectors(stage, catalog)
        stage["tasks"][0]["selector"]["split"] = "train"
        stage["tasks"][0]["selector"]["domain"] = "contacts"
        with self.assertRaisesRegex(RuntimeError, "absent from RELAI catalog"):
            _selectors(stage, catalog)


if __name__ == "__main__":
    unittest.main()
