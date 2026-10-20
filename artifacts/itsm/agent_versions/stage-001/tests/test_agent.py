import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eog_tools.agent import AgentConfig, MUTATION_FIDELITY_GUIDANCE, ReactAgent


class Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        value = {"role": "assistant"}
        if self.content is not None:
            value["content"] = self.content
        if self.tool_calls is not None:
            value["tool_calls"] = [{
                "id": call.id,
                "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            } for call in self.tool_calls]
        return value


def response(message, finish_reason="stop", tokens=5):
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=tokens)
    usage.model_dump = lambda exclude_none=True: {
        "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": tokens,
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


class FakeMCP:
    def __init__(self, fail=False):
        self.closed = False
        self.calls = []
        self.fail = fail

    def list_tools(self):
        return [{"name": "find", "inputSchema": {"type": "object"}}]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise ConnectionError("lost MCP connection")
        return {"content": "abcdef"}

    def close(self):
        self.closed = True


class AgentTests(unittest.TestCase):
    def task(self, mcp):
        return SimpleNamespace(
            action_type="mcp", mcp_servers=[object()],
            system_prompt="system", user_prompt="user",
            mcp_session=lambda server: mcp,
        )

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_tool_loop_records_trace_and_usage(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments='{"id": 4}'),
        )
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="finished")),
        ]
        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(tool_result_chars=10), client).run(self.task(mcp))
        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.total_tokens, 10)
        self.assertEqual(mcp.calls, [("find", {"id": 4})])
        self.assertTrue(mcp.closed)
        self.assertTrue(result.events[0]["tool_results"][0]["truncated"])
        self.assertEqual(captured[1]["messages"][-1]["tool_call_id"], "call-1")
        self.assertEqual(captured[0]["max_completion_tokens"], 16_384)
        self.assertEqual(
            result.initial_messages[0]["content"],
            f"system\n\n{MUTATION_FIDELITY_GUIDANCE}",
        )
        system_message = result.initial_messages[0]["content"]
        for instruction in (
            "perform them before refusing because identifiers, names, relationships, or requester context are missing",
            "using supplied attributes such as assignee, subject, state, reporter, and record relationships",
            "Ambiguity is a post-lookup condition",
            "an authoritative permission denial, or no required discovery tool",
            "If no requester identity is exposed, do not invent a missing-role blocker",
            "every deterministic policy-derived postcondition or coupled field",
            "the tool schema makes that field writable, include it in the mutation",
            "Compare every state-changing tool result with all intended postconditions",
            "a corrective mutation or authoritative read-back before finishing",
            "Do not claim success or speculate about backend timing",
            "Report failure only when the normal tools do not support the correction or reject it",
        ):
            with self.subTest(instruction=instruction):
                self.assertIn(instruction, system_message)
        self.assertLess(
            system_message.index("Ambiguity is a post-lookup condition"),
            system_message.index("Before acting, form and retain an internal checklist"),
        )
        self.assertTrue(system_message.startswith("system\n\n"))
        self.assertTrue(system_message.endswith(MUTATION_FIDELITY_GUIDANCE))
        self.assertEqual(result.initial_messages[1]["content"], "user")
        self.assertEqual(captured[0]["messages"][:2], result.initial_messages)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_mcp_failure_keeps_partial_trace_and_closes(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response(Message(tool_calls=[call]), "tool_calls"),
        )))
        mcp = FakeMCP(fail=True)
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(result.status, "error")
        self.assertIn("lost MCP connection", result.error)
        self.assertEqual(result.steps, 1)
        self.assertTrue(mcp.closed)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_length_stop_is_not_reported_done(self, _convert):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response(Message(content="partial"), "length"),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))
        self.assertEqual(result.status, "output_limit")

    def test_limits_must_be_positive(self):
        with self.assertRaises(ValueError):
            AgentConfig(max_steps=0)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_step_limit_stops_after_configured_turns(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response(Message(tool_calls=[call]), "tool_calls"),
        )))
        mcp = FakeMCP()
        result = ReactAgent(AgentConfig(max_steps=1), client).run(self.task(mcp))
        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps, 1)
        self.assertTrue(mcp.closed)


if __name__ == "__main__":
    unittest.main()
