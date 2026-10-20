import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eog_tools.agent import (
    AgentConfig,
    EXECUTION_GUIDANCE,
    FINALIZATION_CHECKPOINT,
    RECOVERY_TOOL_NAME,
    ReactAgent,
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
        return self.output

    def close(self):
        self.closed = True


class SequencedMCP(FakeMCP):
    def __init__(self, outputs, tools):
        super().__init__(tools=tools)
        self.outputs = list(outputs)

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.outputs.pop(0)


class PersistingErrorMCP(FakeMCP):
    def __init__(self):
        super().__init__(tools=[
            {"name": "create_item", "inputSchema": {
                "type": "object", "properties": {"projectId": {"type": "string"}},
            }},
            {"name": "list_items", "inputSchema": {
                "type": "object", "properties": {"projectId": {"type": "string"}},
            }},
        ])
        self.persisted = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        self.persisted.append(dict(arguments))
        return {
            "result": {
                "content": [{"text": "Validation errors:\n- state: invalid"}],
                "isError": False,
            },
        }


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
            response(Message(content="proposed answer")),
            response(Message(content="finished")),
        ]
        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(tool_result_chars=10), client).run(self.task(mcp))
        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.total_tokens, 15)
        self.assertEqual(mcp.calls, [("find", {"id": 4})])
        self.assertTrue(mcp.closed)
        self.assertTrue(result.events[0]["tool_results"][0]["truncated"])
        preview = json.loads(result.events[0]["tool_results"][0]["content"])
        self.assertTrue(preview["incomplete_tool_result"])
        self.assertEqual(preview["total_chars"], len(json.dumps(mcp.output)))
        self.assertEqual(preview["tail"], json.dumps(mcp.output)[-5:])
        self.assertEqual(result.events[0]["tool_results"][0]["preview"]["handle"], preview["handle"])
        self.assertEqual(captured[1]["messages"][-1]["tool_call_id"], "call-1")
        self.assertEqual(captured[2]["messages"][-2]["content"], "proposed answer")
        self.assertEqual(captured[2]["messages"][-1]["content"], FINALIZATION_CHECKPOINT)
        self.assertTrue(result.events[1]["finalization_checkpoint"]["activated"])
        self.assertEqual(captured[0]["max_completion_tokens"], 16_384)
        self.assertTrue(any(
            tool.get("function", {}).get("name") == RECOVERY_TOOL_NAME
            for tool in captured[0]["tools"]
        ))
        self.assertIn("incomplete previews", result.initial_messages[0]["content"])
        self.assertTrue(result.initial_messages[0]["content"].endswith(EXECUTION_GUIDANCE))
        system_prompt = captured[0]["messages"][0]["content"]
        self.assertIn("minimal separator needed for continuation", system_prompt)
        self.assertIn("including whitespace and punctuation", system_prompt)
        self.assertIn("Successful storage is insufficient", system_prompt)
        self.assertIn("issue a corrective mutation", system_prompt)
        self.assertIn("inventory explicit numeric years, yearless dates", system_prompt)
        self.assertIn("single wall-clock-derived interpretation", system_prompt)
        self.assertIn("make one bounded recovery query", system_prompt)
        self.assertIn("do not make consequential mutations", system_prompt)
        self.assertIn("same destination or operation outranks", system_prompt)
        self.assertIn("use the most specific explicit anchor", system_prompt)
        self.assertIn("lifecycle fields such as status or state", system_prompt)
        self.assertIn("repair only that field", system_prompt)
        self.assertIn("write may have persisted", system_prompt)
        self.assertEqual(result.initial_messages[1]["content"], "user")

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_short_tool_result_is_unchanged(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="proposed answer")),
            response(Message(content="finished")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(tool_result_chars=100), client).run(self.task(mcp))
        expected = json.dumps(mcp.output, ensure_ascii=False)
        self.assertFalse(result.events[0]["tool_results"][0]["truncated"])
        self.assertEqual(result.events[0]["tool_results"][0]["content"], expected)
        self.assertEqual(captured[1]["messages"][-1]["content"], expected)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_ambiguous_write_blocks_immediate_duplicate_retry(self, _convert):
        create = lambda call_id: SimpleNamespace(
            id=call_id, type="function",
            function=SimpleNamespace(
                name="create_item",
                arguments='{"projectId": "project-1", "summary": "Deploy"}',
            ),
        )
        responses = [
            response(Message(tool_calls=[create("create-1")]), "tool_calls"),
            response(Message(tool_calls=[create("create-2")]), "tool_calls"),
            response(Message(content="proposed answer")),
            response(Message(content="finished")),
        ]
        mcp = PersistingErrorMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(len(mcp.calls), 1)
        self.assertEqual(len(mcp.persisted), 1)
        blocked = result.events[1]["tool_results"][0]
        self.assertTrue(blocked["local"])
        self.assertTrue(json.loads(blocked["content"])["reconciliation_required"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_relevant_read_releases_retry_after_confirmed_absence(self, _convert):
        calls = [
            SimpleNamespace(
                id="create-1", type="function",
                function=SimpleNamespace(
                    name="create_item",
                    arguments='{"projectId": "project-1", "summary": "Deploy"}',
                ),
            ),
            SimpleNamespace(
                id="list-1", type="function",
                function=SimpleNamespace(
                    name="list_items", arguments='{"projectId": "project-1"}',
                ),
            ),
            SimpleNamespace(
                id="create-2", type="function",
                function=SimpleNamespace(
                    name="create_item",
                    arguments='{"projectId": "project-1", "summary": "Deploy", "state": "ready"}',
                ),
            ),
        ]
        responses = [
            response(Message(tool_calls=[calls[0]]), "tool_calls"),
            response(Message(tool_calls=[calls[1]]), "tool_calls"),
            response(Message(tool_calls=[calls[2]]), "tool_calls"),
            response(Message(content="proposed answer")),
            response(Message(content="finished")),
        ]
        mcp = SequencedMCP(
            outputs=[
                {"result": {"content": [{"text": "1 validation error for Item\nstate invalid"}],
                            "isError": False}},
                {"result": {"content": [{"text": '{"items": []}'}], "isError": False}},
                {"result": {"content": [{"text": '{"id": "item-1"}'}], "isError": False}},
            ],
            tools=[
                {"name": "create_item", "inputSchema": {
                    "type": "object", "properties": {"projectId": {"type": "string"}},
                }},
                {"name": "list_items", "inputSchema": {
                    "type": "object", "properties": {"projectId": {"type": "string"}},
                }},
            ],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(result.status, "done")
        self.assertEqual([name for name, _ in mcp.calls],
                         ["create_item", "list_items", "create_item"])
        self.assertEqual(
            result.events[1]["tool_results"][0]["reconciliation"]["cleared_writes"],
            ["create_item"],
        )

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_successful_writes_are_not_gated(self, _convert):
        create_calls = [
            SimpleNamespace(
                id=f"create-{index}", type="function",
                function=SimpleNamespace(
                    name="create_item",
                    arguments=json.dumps({"projectId": "project-1", "summary": summary}),
                ),
            )
            for index, summary in enumerate(("Deploy", "Review"), 1)
        ]
        responses = [
            response(Message(tool_calls=[create_calls[0]]), "tool_calls"),
            response(Message(tool_calls=[create_calls[1]]), "tool_calls"),
            response(Message(content="proposed answer")),
            response(Message(content="finished")),
        ]
        mcp = SequencedMCP(
            outputs=[
                {"result": {"content": [{"text": '{"id": "item-1"}'}], "isError": False}},
                {"result": {"content": [{"text": '{"id": "item-2"}'}], "isError": False}},
            ],
            tools=[
                {"name": "create_item", "inputSchema": {
                    "type": "object", "properties": {"projectId": {"type": "string"}},
                }},
                {"name": "list_items", "inputSchema": {
                    "type": "object", "properties": {"projectId": {"type": "string"}},
                }},
            ],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(len(mcp.calls), 2)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_error_like_write_is_advisory_when_no_read_tool_exists(self, _convert):
        create = lambda call_id: SimpleNamespace(
            id=call_id, type="function",
            function=SimpleNamespace(
                name="create_item",
                arguments='{"projectId": "project-1", "summary": "Deploy"}',
            ),
        )
        responses = [
            response(Message(tool_calls=[create("create-1")]), "tool_calls"),
            response(Message(tool_calls=[create("create-2")]), "tool_calls"),
            response(Message(content="proposed answer")),
            response(Message(content="finished")),
        ]
        mcp = SequencedMCP(
            outputs=[
                {"result": {"content": [{"text": "Validation errors:\n- state invalid"}],
                            "isError": False}},
                {"result": {"content": [{"text": '{"id": "item-1"}'}], "isError": False}},
            ],
            tools=[{"name": "create_item", "inputSchema": {"type": "object"}}],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(len(mcp.calls), 2)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_recovers_content_after_original_boundary_without_mcp_call(self, _convert):
        output = {"padding": "x" * 4_100, "hidden_field": "recovered"}
        find_call = SimpleNamespace(
            id="call-find", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        calls = 0

        def create(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(Message(tool_calls=[find_call]), "tool_calls")
            if calls == 2:
                preview = json.loads(kwargs["messages"][-1]["content"])
                recovery_call = SimpleNamespace(
                    id="call-recover", type="function",
                    function=SimpleNamespace(
                        name=RECOVERY_TOOL_NAME,
                        arguments=json.dumps({
                            "handle": preview["handle"], "offset": 4_000, "length": 300,
                        }),
                    ),
                )
                return response(Message(tool_calls=[recovery_call]), "tool_calls")
            if calls == 3:
                recovered = json.loads(kwargs["messages"][-1]["content"])
                self.assertIn('"hidden_field": "recovered"', recovered["content"])
                self.assertTrue(recovered["end_of_result"])
                return response(Message(content="proposed answer"))
            self.assertEqual(kwargs["messages"][-1]["content"], FINALIZATION_CHECKPOINT)
            return response(Message(content="finished"))

        mcp = FakeMCP(output=output)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(result.status, "done")
        self.assertEqual(mcp.calls, [("find", {})])
        recovery = result.events[1]["tool_results"][0]
        self.assertTrue(recovery["local"])
        self.assertTrue(recovery["recovery"]["success"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_recovery_rejects_invalid_handles_and_offsets(self, _convert):
        find_call = SimpleNamespace(
            id="call-find", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        turn = 0
        valid_handle = None
        total_chars = None

        def create(**kwargs):
            nonlocal turn, valid_handle, total_chars
            turn += 1
            if turn == 1:
                return response(Message(tool_calls=[find_call]), "tool_calls")
            if turn == 2:
                preview = json.loads(kwargs["messages"][-1]["content"])
                valid_handle = preview["handle"]
                total_chars = preview["total_chars"]
                arguments = {"handle": "missing", "offset": 0, "length": 10}
            elif turn == 3:
                invalid = json.loads(kwargs["messages"][-1]["content"])
                self.assertEqual(invalid["error"]["type"], "invalid_handle")
                arguments = {"handle": valid_handle, "offset": total_chars, "length": 10}
            elif turn == 4:
                invalid = json.loads(kwargs["messages"][-1]["content"])
                self.assertEqual(invalid["error"]["type"], "invalid_offset")
                return response(Message(content="proposed answer"))
            else:
                self.assertEqual(kwargs["messages"][-1]["content"], FINALIZATION_CHECKPOINT)
                return response(Message(content="finished"))
            recovery_call = SimpleNamespace(
                id=f"call-recover-{turn}", type="function",
                function=SimpleNamespace(
                    name=RECOVERY_TOOL_NAME, arguments=json.dumps(arguments),
                ),
            )
            return response(Message(tool_calls=[recovery_call]), "tool_calls")

        mcp = FakeMCP(output={"content": "x" * 200})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(tool_result_chars=20), client).run(self.task(mcp))
        self.assertEqual(result.status, "done")
        self.assertEqual(mcp.calls, [("find", {})])
        self.assertFalse(result.events[1]["tool_results"][0]["recovery"]["success"])
        self.assertFalse(result.events[2]["tool_results"][0]["recovery"]["success"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_finalization_checkpoint_allows_correction_and_runs_once(self, _convert):
        first_call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments='{"id": 4}'),
        )
        corrective_call = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="find", arguments='{"id": 5}'),
        )
        captured = []
        responses = [
            response(Message(tool_calls=[first_call]), "tool_calls"),
            response(Message(content="proposed answer")),
            response(Message(tool_calls=[corrective_call]), "tool_calls"),
            response(Message(content="corrected answer")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(result.status, "done")
        self.assertEqual(result.final_message, "corrected answer")
        self.assertEqual(mcp.calls, [("find", {"id": 4}), ("find", {"id": 5})])
        self.assertEqual(sum(
            "finalization_checkpoint" in event for event in result.events
        ), 1)
        self.assertEqual(captured[2]["messages"][-1]["content"], FINALIZATION_CHECKPOINT)
        self.assertIn("persisted start dates, end dates, and time zones",
                      FINALIZATION_CHECKPOINT)
        self.assertIn("temporal-anchor mismatches", FINALIZATION_CHECKPOINT)
        self.assertIn("empty date-filtered retrievals", FINALIZATION_CHECKPOINT)
        self.assertIn("no bounded recovery was attempted", FINALIZATION_CHECKPOINT)
        self.assertIn("do not repeatedly widen", FINALIZATION_CHECKPOINT)
        self.assertIn("patch or delete duplicate resources", FINALIZATION_CHECKPOINT)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_no_tool_answer_finalizes_without_checkpoint(self, _convert):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response(Message(content="finished")),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))
        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 1)
        self.assertFalse(any("finalization_checkpoint" in event for event in result.events))

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_reserved_recovery_tool_name_collision_is_rejected(self, _convert):
        mcp = FakeMCP(tools=[{
            "name": RECOVERY_TOOL_NAME, "inputSchema": {"type": "object"},
        }])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: self.fail("model should not be called"),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        self.assertEqual(result.status, "error")
        self.assertIn("is reserved", result.error)
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


if __name__ == "__main__":
    unittest.main()
