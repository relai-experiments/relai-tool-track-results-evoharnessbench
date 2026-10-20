import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eog_tools.agent import (
    AgentConfig,
    FINALIZATION_AUDIT,
    ReactAgent,
    READ_TOOL_RESULT,
    TOOL_USE_POLICY,
)


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
    def __init__(self, fail=False, output=None, tools=None):
        self.closed = False
        self.calls = []
        self.fail = fail
        self.output = {"content": "abcdef"} if output is None else output
        self.tools = [{"name": "find", "inputSchema": {"type": "object"}}] if tools is None else tools

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise ConnectionError("lost MCP connection")
        if callable(self.output):
            return self.output(name, arguments)
        return self.output

    def close(self):
        self.closed = True


class AgentTests(unittest.TestCase):
    def task(self, mcp):
        return SimpleNamespace(
            action_type="mcp", mcp_servers=[object()],
            system_prompt="system", user_prompt="user",
            mcp_session=lambda server: mcp,
        )

    @patch("eog_tools.agent._to_openai_tools")
    def test_tool_loop_records_trace_and_usage(self, convert):
        tools = [{
            "type": "function",
            "function": {
                "name": "find",
                "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            },
        }]
        convert.return_value = tools
        arguments = {
            "enabled": True,
            "count": 4,
            "optional": None,
            "items": [1, {"nested": False}],
        }
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments=json.dumps(arguments)),
        )
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="draft finished")),
            response(Message(content="finished")),
        ]
        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.total_tokens, 15)
        self.assertEqual(result.final_message, "finished")
        self.assertEqual(mcp.calls, [("find", arguments)])
        self.assertTrue(mcp.closed)
        self.assertTrue(result.events[1]["finalization_audit_triggered"])
        self.assertFalse(result.events[0]["tool_results"][0]["truncated"])
        self.assertEqual(
            result.events[0]["tool_results"][0]["content"],
            json.dumps({"content": "abcdef"}),
        )
        self.assertEqual(captured[1]["messages"][-1]["tool_call_id"], "call-1")
        self.assertEqual(captured[0]["max_completion_tokens"], 16_384)
        self.assertEqual(captured[0]["tools"][:-1], tools)
        self.assertEqual(captured[0]["tools"][-1]["function"]["name"], READ_TOOL_RESULT)
        self.assertEqual(result.tools, captured[0]["tools"])
        self.assertEqual(
            result.initial_messages[0]["content"],
            f"system\n\n{TOOL_USE_POLICY}",
        )
        self.assertEqual(captured[2]["messages"][-1]["content"], FINALIZATION_AUDIT)
        self.assertEqual(captured[2]["messages"][-2]["content"], "draft finished")
        self.assertIn("Preserve explicit JSON types", captured[0]["messages"][0]["content"])
        self.assertIn("similarly named sibling field", captured[0]["messages"][0]["content"])
        self.assertIn("observation marked incomplete", captured[0]["messages"][0]["content"])
        self.assertIn(
            "empty narrow query is not sufficient evidence",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn(
            "remove optional suffix assumptions, relax combined predicates one at a time",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn(
            "use that candidate's exact returned resource ID and name",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn(
            "multiple plausible candidates remain after reasonable retrieval fallbacks are exhausted",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn(
            "Copying, duplicating, or cloning a resource preserves its observed source type",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn(
            "filename or extension is only a label and does not authorize conversion",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn(
            "compare any proposed MIME or format argument with the observed source type",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn(
            "Omit optional MIME or format arguments unless conversion is explicit",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn("Preserve user-specified names literally", captured[0]["messages"][0]["content"])
        self.assertIn(
            "Treat an unexpected type as a failed mutation",
            captured[0]["messages"][0]["content"],
        )
        self.assertIn("corrective normal tool call", captured[0]["messages"][0]["content"])
        self.assertIn(
            "reasonable broader retrieval fallbacks were exhausted",
            captured[2]["messages"][-1]["content"],
        )
        self.assertIn(
            "continue with its exact returned resource ID and name",
            captured[2]["messages"][-1]["content"],
        )
        self.assertIn(
            "ask for clarification only when multiple plausible candidates remain",
            captured[2]["messages"][-1]["content"],
        )
        self.assertIn(
            "make corrective normal tool calls now",
            captured[2]["messages"][-1]["content"],
        )
        self.assertEqual(result.initial_messages[1]["content"], "user")

    @patch("eog_tools.agent._to_openai_tools")
    def test_incomplete_result_can_be_read_across_multiple_continuations(self, convert):
        convert.return_value = [{
            "type": "function",
            "function": {"name": "find", "parameters": {"type": "object"}},
        }]
        find = SimpleNamespace(
            id="call-find", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        read_one = SimpleNamespace(
            id="call-read-1", type="function",
            function=SimpleNamespace(
                name=READ_TOOL_RESULT,
                arguments=json.dumps({"call_id": "call-find", "offset": 30, "max_chars": 30}),
            ),
        )
        read_two = SimpleNamespace(
            id="call-read-2", type="function",
            function=SimpleNamespace(
                name=READ_TOOL_RESULT,
                arguments=json.dumps({"call_id": "call-find", "offset": 60, "max_chars": 30}),
            ),
        )
        responses = [
            response(Message(tool_calls=[find]), "tool_calls"),
            response(Message(tool_calls=[read_one]), "tool_calls"),
            response(Message(tool_calls=[read_two]), "tool_calls"),
            response(Message(content="draft decisive file")),
            response(Message(content="found decisive file")),
        ]
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP(output={"content": ("A" * 50) + "decisive-file"})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(tool_result_chars=30), client).run(self.task(mcp))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 5)
        self.assertEqual(result.final_message, "found decisive file")
        self.assertEqual(mcp.calls, [("find", {})])
        self.assertTrue(result.events[0]["tool_results"][0]["truncated"])
        self.assertIn("incomplete=true", captured[1]["messages"][-1]["content"])
        self.assertIn("next_offset=30", captured[1]["messages"][-1]["content"])
        self.assertTrue(result.events[1]["tool_results"][0]["truncated"])
        self.assertIn("decisive-file", result.events[2]["tool_results"][0]["content"])
        self.assertFalse(result.events[2]["tool_results"][0]["truncated"])
        self.assertEqual(
            [event["tool_results"][0]["name"] for event in result.events[:3]],
            ["find", READ_TOOL_RESULT, READ_TOOL_RESULT],
        )
        self.assertTrue(result.events[3]["finalization_audit_triggered"])

    @patch("eog_tools.agent._to_openai_tools")
    def test_finalization_audit_corrects_semantically_wrong_sibling_field(self, convert):
        convert.return_value = [{
            "type": "function",
            "function": {
                "name": "update_record",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "publicMetadata": {"type": "object"},
                        "privateMetadata": {"type": "object"},
                    },
                },
            },
        }]
        wrong = SimpleNamespace(
            id="call-wrong", type="function",
            function=SimpleNamespace(
                name="update_record",
                arguments=json.dumps({"publicMetadata": {"status": "approved"}}),
            ),
        )
        corrective = SimpleNamespace(
            id="call-correct", type="function",
            function=SimpleNamespace(
                name="update_record",
                arguments=json.dumps({"privateMetadata": {"status": "approved"}}),
            ),
        )
        responses = [
            response(Message(tool_calls=[wrong]), "tool_calls"),
            response(Message(content="The private metadata was updated.")),
            response(Message(tool_calls=[corrective]), "tool_calls"),
            response(Message(content="The private metadata is now updated.")),
            response(Message(content="The private metadata is now updated.")),
        ]
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP(output=lambda _name, arguments: {
            "publicMetadata": arguments.get("publicMetadata"),
            "privateMetadata": arguments.get("privateMetadata"),
        })
        task = self.task(mcp)
        task.user_prompt = "Set the private metadata status to approved."
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(task)

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 5)
        self.assertEqual(mcp.calls, [
            ("update_record", {"publicMetadata": {"status": "approved"}}),
            ("update_record", {"privateMetadata": {"status": "approved"}}),
        ])
        self.assertTrue(result.events[1]["finalization_audit_triggered"])
        self.assertTrue(result.events[3]["finalization_audit_triggered"])
        self.assertEqual(captured[2]["messages"][-1]["content"], FINALIZATION_AUDIT)
        self.assertEqual(captured[4]["messages"][-1]["content"], FINALIZATION_AUDIT)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_tool_free_conversation_does_not_trigger_audit(self, _convert):
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return response(Message(content="No tools needed."))

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.final_message, "No tools needed.")
        self.assertEqual(len(captured), 1)
        self.assertNotIn("finalization_audit_triggered", result.events[0])

    @patch("eog_tools.agent._to_openai_tools")
    def test_invalid_continuation_handle_returns_tool_error_without_mcp_call(self, convert):
        convert.return_value = [{
            "type": "function",
            "function": {"name": "find", "parameters": {"type": "object"}},
        }]
        read = SimpleNamespace(
            id="call-read", type="function",
            function=SimpleNamespace(
                name=READ_TOOL_RESULT,
                arguments=json.dumps({"call_id": "missing", "offset": 0, "max_chars": 10}),
            ),
        )
        responses = [
            response(Message(tool_calls=[read]), "tool_calls"),
            response(Message(content="handled")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        mcp = FakeMCP()
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        tool_result = result.events[0]["tool_results"][0]
        self.assertEqual(result.status, "done")
        self.assertEqual(mcp.calls, [])
        self.assertIn("unknown tool result call_id", tool_result["content"])
        self.assertFalse(tool_result["truncated"])

    @patch("eog_tools.agent._to_openai_tools")
    def test_reserved_tool_name_collision_fails_before_model_call(self, convert):
        convert.return_value = [{
            "type": "function",
            "function": {"name": READ_TOOL_RESULT, "parameters": {"type": "object"}},
        }]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: self.fail("model should not be called"),
        )))
        mcp = FakeMCP(tools=[{"name": READ_TOOL_RESULT, "inputSchema": {"type": "object"}}])
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(result.status, "error")
        self.assertIn("MCP tool name is reserved", result.error)
        self.assertTrue(mcp.closed)

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

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_finalization_audit_respects_step_limit_and_token_accounting(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="draft")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(max_steps=2), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.total_tokens, 10)
        self.assertEqual(result.final_message, "")
        self.assertTrue(result.events[1]["finalization_audit_triggered"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_audit_corrective_tool_error_keeps_audit_trace(self, _convert):
        initial = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        corrective = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="find", arguments=json.dumps({"correct": True})),
        )
        responses = [
            response(Message(tool_calls=[initial]), "tool_calls"),
            response(Message(content="draft")),
            response(Message(tool_calls=[corrective]), "tool_calls"),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        mcp = FakeMCP()

        def call_tool(name, arguments):
            mcp.calls.append((name, arguments))
            if len(mcp.calls) == 2:
                raise ConnectionError("corrective call failed")
            return {"ok": True}

        mcp.call_tool = call_tool
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))

        self.assertEqual(result.status, "error")
        self.assertIn("corrective call failed", result.error)
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.total_tokens, 15)
        self.assertTrue(result.events[1]["finalization_audit_triggered"])
        self.assertTrue(mcp.closed)


if __name__ == "__main__":
    unittest.main()
