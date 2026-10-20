import unittest
from unittest.mock import patch

from eog_tools.command_agent import TimedCommandAgent


class TimedCommandAgentTests(unittest.TestCase):
    def test_adaptation_uses_its_own_timeout_and_restores_task_timeout(self):
        with patch("simple_agentic_evals.CommandAgent.adapt", autospec=True) as parent_adapt:
            parent_adapt.side_effect = lambda agent, stage, tasks: {"timeout": agent.timeout}
            agent = TimedCommandAgent(["true"], timeout=1800, adapt_timeout=45000)
            self.assertEqual(agent.adapt(1, []), {"timeout": 45000})
            self.assertEqual(agent.timeout, 1800)

    def test_task_timeout_is_restored_on_adaptation_failure(self):
        with patch("simple_agentic_evals.CommandAgent.adapt", side_effect=RuntimeError("failed")):
            agent = TimedCommandAgent(["true"], timeout=1800, adapt_timeout=45000)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                agent.adapt(1, [])
            self.assertEqual(agent.timeout, 1800)


if __name__ == "__main__":
    unittest.main()
