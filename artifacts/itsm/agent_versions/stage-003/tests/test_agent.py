import copy
import json
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
    def __init__(self, fail=False, tools=None, outputs=None):
        self.closed = False
        self.calls = []
        self.fail = fail
        self.tools = tools or [{"name": "find", "inputSchema": {"type": "object"}}]
        self.outputs = list(outputs or [])

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise ConnectionError("lost MCP connection")
        if self.outputs:
            return self.outputs.pop(0)
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
        visible_result = result.events[0]["tool_results"][0]["content"]
        prefix, marker = visible_result.splitlines()
        self.assertEqual(prefix, json.dumps({"content": "abcdef"})[:10])
        self.assertEqual(json.loads(marker), {
            "tool_result_truncated": True,
            "omitted_chars": len(json.dumps({"content": "abcdef"})) - 10,
        })
        self.assertEqual(captured[1]["messages"][-2]["tool_call_id"], "call-1")
        self.assertEqual(captured[1]["messages"][-1]["role"], "system")
        self.assertEqual(captured[0]["max_completion_tokens"], 16_384)
        self.assertEqual(
            result.initial_messages[0]["content"],
            f"system\n\n{MUTATION_FIDELITY_GUIDANCE}",
        )
        system_message = result.initial_messages[0]["content"]
        for instruction in (
            "perform them before refusing because identifiers, names, relationships, or requester context are missing",
            "using supplied attributes such as assignee, subject, state, reporter, and record relationships",
            "progressively remove inferred or conjunctive filters",
            "Ambiguity is a post-lookup condition",
            "an authoritative permission denial or absence of a required discovery tool",
            "If no requester identity is exposed, do not invent a missing-role blocker",
            "a closed-world scope ledger",
            "explicitly requested mutation targets and changes",
            "explicit exclusions",
            "policy-required additions",
            "discovered-but-unapproved candidates",
            "discovery does not authorize mutation",
            "related, adjacent, synonym-matched, or continuity-beneficial entities",
            "map every mutation target and argument to its authorizing ledger entry",
            "Keep role or assignee selection separate from membership and other unrelated mutations",
            "mutate only the conservative supported subset and report the ambiguity",
            "do not expand scope through synonyms, inferred relationships, or nearby categories",
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
            system_message.index("Before acting, form and retain a closed-world scope ledger"),
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
    def test_empty_read_only_discovery_adds_recovery_after_raw_result(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="lookup_records", arguments='{"status": "open"}'),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="finished")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        empty = {"result": {"content": [{"type": "text", "text": "[]"}], "isError": False}}
        mcp = FakeMCP(
            tools=[{"name": "lookup_records", "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True}}],
            outputs=[empty],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        next_messages = captured[1]["messages"]
        self.assertEqual(next_messages[-2]["role"], "tool")
        self.assertEqual(next_messages[-2]["content"], json.dumps(empty))
        self.assertEqual(next_messages[-1]["role"], "system")
        self.assertIn("Remove inferred or conjunctive filters", next_messages[-1]["content"])
        self.assertEqual(result.events[0]["tool_results"][0]["content"], json.dumps(empty))
        self.assertTrue(mcp.closed)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_repeated_empty_discovery_escalates_and_discourages_equivalent_query(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="list_records", arguments='{"query": " Example "}'),
        )
        second_call = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="list_records", arguments='{"query": "example"}'),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(tool_calls=[second_call]), "tool_calls"),
            response(Message(content="finished")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP(outputs=[[], []])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ReactAgent(AgentConfig(), client).run(self.task(mcp))
        recovery = captured[2]["messages"][-1]["content"]
        self.assertIn("equivalent to an earlier empty query", recovery)
        self.assertIn("attempt at least one bounded minimally constrained search", recovery)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_truncated_read_only_discovery_adds_recovery_after_marked_result(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="list_records", arguments='{"status": "open"}'),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="finished")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        output = {"records": [{"id": index} for index in range(5)]}
        mcp = FakeMCP(outputs=[output])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(tool_result_chars=20), client).run(self.task(mcp))
        next_messages = captured[1]["messages"]
        self.assertEqual(next_messages[-2]["role"], "tool")
        self.assertIn('"tool_result_truncated":true', next_messages[-2]["content"])
        self.assertEqual(next_messages[-1]["role"], "system")
        recovery = next_messages[-1]["content"]
        self.assertIn("must not be treated as the full scope", recovery)
        self.assertIn("limit/offset or cursor pagination", recovery)
        self.assertIn("bounded, non-overlapping partitions", recovery)
        self.assertIn("Do not begin mutations that claim complete coverage", recovery)
        self.assertEqual(
            result.events[0]["tool_results"][0]["content"],
            next_messages[-2]["content"],
        )

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_repeated_truncated_discovery_escalates_then_untruncated_continues(self, _convert):
        first_call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="list_records", arguments='{"offset": 0}'),
        )
        repeated_call = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="list_records", arguments='{"offset": 0}'),
        )
        partition_call = SimpleNamespace(
            id="call-3", type="function",
            function=SimpleNamespace(name="list_records", arguments='{"offset": 5}'),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[first_call]), "tool_calls"),
            response(Message(tool_calls=[repeated_call]), "tool_calls"),
            response(Message(tool_calls=[partition_call]), "tool_calls"),
            response(Message(content="finished")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP(outputs=[
            {"records": ["a long first result"]},
            {"records": ["a long second result"]},
            {"x": 1},
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ReactAgent(AgentConfig(tool_result_chars=12), client).run(self.task(mcp))
        repeated_recovery = captured[2]["messages"][-1]["content"]
        self.assertIn("equivalent to an earlier truncated query", repeated_recovery)
        self.assertIn("strictly advance pagination or narrow a partition", repeated_recovery)
        self.assertEqual(captured[3]["messages"][-1]["role"], "tool")
        self.assertNotIn(
            "tool_result_truncated",
            captured[3]["messages"][-1]["content"],
        )

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_truncated_mutation_does_not_trigger_discovery_recovery(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="update_record", arguments='{"id": 1}'),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="finished")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        tools = [{
            "name": "update_record",
            "inputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
        }]
        mcp = FakeMCP(tools=tools, outputs=[{"updated": "a result larger than the limit"}])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ReactAgent(AgentConfig(tool_result_chars=10), client).run(self.task(mcp))
        self.assertEqual(captured[1]["messages"][-1]["role"], "tool")
        self.assertIn("tool_result_truncated", captured[1]["messages"][-1]["content"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_non_empty_and_mutation_results_do_not_trigger_recovery(self, _convert):
        read = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find_records", arguments="{}"),
        )
        mutate = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="update_record", arguments='{"id": 1}'),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[read]), "tool_calls"),
            response(Message(tool_calls=[mutate]), "tool_calls"),
            response(Message(content="finished")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        tools = [
            {"name": "find_records", "inputSchema": {"type": "object"},
             "annotations": {"readOnlyHint": True}},
            {"name": "update_record", "inputSchema": {"type": "object"},
             "annotations": {"readOnlyHint": True}},
        ]
        mcp = FakeMCP(tools=tools, outputs=[[{"id": 1}], []])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(captured[1]["messages"][-1]["role"], "tool")
        self.assertEqual(captured[2]["messages"][-1]["role"], "tool")

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
