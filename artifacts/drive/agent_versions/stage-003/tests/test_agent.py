import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eog_tools.agent import (
    ACTIONABILITY_AUDIT,
    AgentConfig,
    FINALIZATION_AUDIT,
    RECONCILIATION_CHECKPOINT,
    ReactAgent,
    READ_TOOL_RESULT,
    TOOL_USE_POLICY,
    _normalized_result_facts,
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

    def test_policy_retargets_authorization_constraints_and_channel_fallback(self):
        self.assertIn(
            "explicit request for a non-destructive mutation as authorization",
            TOOL_USE_POLICY,
        )
        self.assertIn(
            "Do not add restrictions, privacy hardening, access limits, or other state",
            TOOL_USE_POLICY,
        )
        self.assertIn(
            "hidden-state controls and access permissions as independent",
            TOOL_USE_POLICY,
        )
        self.assertIn(
            "does not specify its communication channel, select an available native channel",
            TOOL_USE_POLICY,
        )
        self.assertIn(
            "Classify every unmet obligation as actionable now, explicitly blocked by policy",
            FINALIZATION_AUDIT,
        )
        self.assertIn(
            "Reverse or relax any agent-created, unrequested state",
            FINALIZATION_AUDIT,
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
        self.assertEqual(captured[1]["messages"][-2]["tool_call_id"], "call-1")
        self.assertIn(RECONCILIATION_CHECKPOINT, captured[1]["messages"][-1]["content"])
        self.assertIn('"path": "$.content"', captured[1]["messages"][-1]["content"])
        self.assertIn('"value": "abcdef"', captured[1]["messages"][-1]["content"])
        self.assertEqual(captured[0]["max_completion_tokens"], 16_384)
        self.assertEqual(captured[0]["tools"][:-1], tools)
        self.assertEqual(captured[0]["tools"][-1]["function"]["name"], READ_TOOL_RESULT)
        self.assertEqual(result.tools, captured[0]["tools"])
        self.assertEqual(
            result.initial_messages[0]["content"],
            f"system\n\n{TOOL_USE_POLICY}",
        )
        self.assertIn(FINALIZATION_AUDIT, captured[2]["messages"][-1]["content"])
        self.assertIn("ORIGINAL USER REQUEST:\nuser", captured[2]["messages"][-1]["content"])
        self.assertIn("tool=find; status=success", captured[2]["messages"][-1]["content"])
        self.assertIn('"enabled": true', captured[2]["messages"][-1]["content"])
        self.assertIn("observed_facts=", captured[2]["messages"][-1]["content"])
        self.assertEqual(captured[2]["messages"][-2]["content"], "draft finished")
        self.assertIn("decompose the request into atomic obligations", captured[0]["messages"][0]["content"])
        self.assertIn("incidental notification", captured[0]["messages"][0]["content"])
        self.assertIn("Preserve explicit JSON types", captured[0]["messages"][0]["content"])
        self.assertIn("resource identifiers as discoverable", captured[0]["messages"][0]["content"])
        self.assertIn("existing labels and schema-allowed values", captured[0]["messages"][0]["content"])
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
        self.assertIn("incomplete=true", captured[1]["messages"][-2]["content"])
        self.assertIn("next_offset=30", captured[1]["messages"][-2]["content"])
        self.assertIn(RECONCILIATION_CHECKPOINT, captured[1]["messages"][-1]["content"])
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
        self.assertIn(FINALIZATION_AUDIT, captured[2]["messages"][-1]["content"])
        self.assertIn("tool=update_record", captured[2]["messages"][-1]["content"])
        self.assertIn(FINALIZATION_AUDIT, captured[4]["messages"][-1]["content"])
        self.assertEqual(captured[4]["messages"][-1]["content"].count("tool=update_record"), 2)

    @patch("eog_tools.agent._to_openai_tools")
    def test_post_mutation_checkpoint_exposes_changed_name_before_dependent_work(
        self, convert,
    ):
        convert.return_value = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("copy_record", "update_record")
        ]
        copy_call = SimpleNamespace(
            id="call-copy", type="function",
            function=SimpleNamespace(
                name="copy_record",
                arguments=json.dumps({"record_id": "record-1", "name": "Target.csv"}),
            ),
        )
        rename_call = SimpleNamespace(
            id="call-rename", type="function",
            function=SimpleNamespace(
                name="update_record",
                arguments=json.dumps({"record_id": "record-2", "name": "Target.csv"}),
            ),
        )
        responses = [
            response(Message(tool_calls=[copy_call]), "tool_calls"),
            response(Message(tool_calls=[rename_call]), "tool_calls"),
            response(Message(content="The copy is named Target.csv.")),
            response(Message(content="The copy is named Target.csv.")),
        ]
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        def output(name, _arguments):
            if name == "copy_record":
                return {"success": True, "record": {
                    "id": "record-2", "name": "Copy of Target.csv",
                }}
            return {"success": True, "record": {"id": "record-2", "name": "Target.csv"}}

        mcp = FakeMCP(output=output)
        task = self.task(mcp)
        task.user_prompt = "Copy record-1 as Target.csv, then continue processing that copy."
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(task)

        self.assertEqual(result.status, "done")
        self.assertEqual(mcp.calls, [
            ("copy_record", {"record_id": "record-1", "name": "Target.csv"}),
            ("update_record", {"record_id": "record-2", "name": "Target.csv"}),
        ])
        checkpoint = captured[1]["messages"][-1]["content"]
        self.assertIn(RECONCILIATION_CHECKPOINT, checkpoint)
        self.assertIn("Copy of Target.csv", checkpoint)
        self.assertIn("Target.csv", checkpoint)
        self.assertIn("before dependent work", checkpoint)
        final_audit = captured[3]["messages"][-1]["content"]
        self.assertIn('"value": "Copy of Target.csv"', final_audit)
        self.assertIn('"value": "Target.csv"', final_audit)

    @patch("eog_tools.agent._to_openai_tools")
    def test_reconciliation_covers_matching_reads_and_failed_calls(self, convert):
        convert.return_value = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("create_record", "get_record", "update_record")
        ]
        calls = [
            SimpleNamespace(
                id=f"call-{name}", type="function",
                function=SimpleNamespace(name=name, arguments="{}"),
            )
            for name in ("create_record", "get_record", "update_record")
        ]
        responses = [
            response(Message(tool_calls=[calls[0]]), "tool_calls"),
            response(Message(tool_calls=[calls[1]]), "tool_calls"),
            response(Message(tool_calls=[calls[2]]), "tool_calls"),
            response(Message(content="The failed update remains unmet.")),
            response(Message(content="The failed update remains unmet.")),
        ]
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        outputs = {
            "create_record": {"success": True, "record": {"name": "Expected"}},
            "get_record": {"status": "ok", "record": {"name": "Expected"}},
            "update_record": {"success": False, "error": "not editable"},
        }
        mcp = FakeMCP(output=lambda name, _arguments: outputs[name])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))

        self.assertEqual(result.status, "done")
        self.assertIn('"value": "Expected"', captured[1]["messages"][-1]["content"])
        self.assertIn("read-only or mutation", captured[2]["messages"][-1]["content"])
        self.assertIn("status=failure", captured[3]["messages"][-1]["content"])
        self.assertIn('"value": "not editable"', captured[3]["messages"][-1]["content"])
        self.assertIn("keep the postcondition explicitly unmet", captured[3]["messages"][-1]["content"])
        self.assertEqual(result.total_tokens, 25)

    def test_result_facts_decode_nested_json_and_enforce_bounds(self):
        nested = {
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "data": {"name": "Copy of Target.csv", "count": 2},
                    }),
                }],
            },
        }
        facts = _normalized_result_facts(nested)
        self.assertIn({
            "path": '$.result.content[0].text["$decoded"].data.name',
            "value": "Copy of Target.csv",
        }, facts)
        self.assertIn({
            "path": '$.result.content[0].text["$decoded"].success',
            "value": True,
        }, facts)

        bounded = _normalized_result_facts(
            {"items": [{"value": "x" * 100} for _ in range(20)]},
            max_items=3,
            max_chars=180,
            max_scalar_chars=20,
        )
        self.assertLessEqual(len(bounded), 3)
        self.assertLessEqual(
            len(json.dumps(bounded, ensure_ascii=False, default=str, sort_keys=True)),
            180,
        )
        self.assertTrue(any("[truncated]" in fact["value"] for fact in bounded))

    @patch("eog_tools.agent._to_openai_tools")
    def test_compound_collaboration_requires_reply_evidence_after_access_notification(
        self, convert,
    ):
        convert.return_value = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("grant_access", "list_comments", "create_reply")
        ]
        grant = SimpleNamespace(
            id="call-grant", type="function",
            function=SimpleNamespace(
                name="grant_access",
                arguments=json.dumps({
                    "user": "reviewer@example.com",
                    "notification_message": "Please review again.",
                }),
            ),
        )
        list_comments = SimpleNamespace(
            id="call-comments", type="function",
            function=SimpleNamespace(name="list_comments", arguments="{}"),
        )
        reply = SimpleNamespace(
            id="call-reply", type="function",
            function=SimpleNamespace(
                name="create_reply",
                arguments=json.dumps({"comment_id": "comment-7", "content": "Please review again."}),
            ),
        )
        responses = [
            response(Message(tool_calls=[grant]), "tool_calls"),
            response(Message(content="Access granted and the reviewer was notified.")),
            response(Message(tool_calls=[list_comments, reply]), "tool_calls"),
            response(Message(content="Access granted and the comment was replied to.")),
            response(Message(content="Access granted and the comment was replied to.")),
        ]
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP(output=lambda name, _arguments: {
            "success": True,
            "id": "permission-1" if name == "grant_access" else "comment-7",
        })
        task = self.task(mcp)
        task.user_prompt = (
            "Give the reviewer access, then reply to their existing comment asking for review."
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(task)

        self.assertEqual(result.status, "done")
        self.assertEqual([name for name, _arguments in mcp.calls], [
            "grant_access", "list_comments", "create_reply",
        ])
        first_audit = captured[2]["messages"][-1]["content"]
        self.assertIn("tool=grant_access", first_audit)
        self.assertNotIn("tool=create_reply", first_audit)
        self.assertIn("incidental to another mutation", first_audit)
        final_audit = captured[4]["messages"][-1]["content"]
        self.assertIn("tool=create_reply; status=success", final_audit)
        self.assertTrue(result.events[1]["finalization_audit_triggered"])
        self.assertTrue(result.events[3]["finalization_audit_triggered"])

    @patch("eog_tools.agent._to_openai_tools")
    def test_explicit_access_notification_needs_no_unrelated_reply(self, convert):
        convert.return_value = [{
            "type": "function",
            "function": {"name": "grant_access", "parameters": {"type": "object"}},
        }]
        grant = SimpleNamespace(
            id="call-grant", type="function",
            function=SimpleNamespace(
                name="grant_access",
                arguments=json.dumps({
                    "user": "reviewer@example.com",
                    "notification_message": "You now have access.",
                }),
            ),
        )
        responses = [
            response(Message(tool_calls=[grant]), "tool_calls"),
            response(Message(content="Access granted and notification sent.")),
            response(Message(content="Access granted and notification sent.")),
        ]
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP(output={"success": True, "id": "permission-1"})
        task = self.task(mcp)
        task.user_prompt = "Grant access and send a permission notification saying access is ready."
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(task)

        self.assertEqual(result.status, "done")
        self.assertEqual(mcp.calls, [("grant_access", {
            "user": "reviewer@example.com",
            "notification_message": "You now have access.",
        })])
        audit = captured[2]["messages"][-1]["content"]
        self.assertIn("tool=grant_access; status=success", audit)
        self.assertNotIn("tool=create_reply", audit)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_legitimate_tool_free_answer_is_accepted_after_one_actionability_audit(
        self, _convert,
    ):
        captured = []
        responses = [
            response(Message(content="No tools needed.")),
            response(Message(content="No tools needed.")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.total_tokens, 10)
        self.assertEqual(result.final_message, "No tools needed.")
        self.assertEqual(len(captured), 2)
        self.assertTrue(result.events[0]["actionability_audit_triggered"])
        self.assertNotIn("actionability_audit_triggered", result.events[1])
        self.assertEqual(captured[1]["messages"][-1]["content"], ACTIONABILITY_AUDIT)

    @patch("eog_tools.agent._to_openai_tools")
    def test_actionability_audit_recovers_file_id_request_into_tool_use(self, convert):
        convert.return_value = [{
            "type": "function",
            "function": {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }]
        search = SimpleNamespace(
            id="call-search", type="function",
            function=SimpleNamespace(
                name="search_files",
                arguments=json.dumps({"query": "yearly report"}),
            ),
        )
        responses = [
            response(Message(content="Please provide the file ID.")),
            response(Message(tool_calls=[search]), "tool_calls"),
            response(Message(content="Found and handled the report.")),
            response(Message(content="Found and handled the report.")),
        ]
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        mcp = FakeMCP(output={"files": [{"id": "file-1", "name": "Yearly Report"}]})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 4)
        self.assertEqual(result.total_tokens, 20)
        self.assertEqual(result.final_message, "Found and handled the report.")
        self.assertEqual(mcp.calls, [("search_files", {"query": "yearly report"})])
        self.assertTrue(result.events[0]["actionability_audit_triggered"])
        self.assertTrue(result.events[2]["finalization_audit_triggered"])
        self.assertEqual(captured[1]["messages"][-1]["content"], ACTIONABILITY_AUDIT)
        self.assertIn(FINALIZATION_AUDIT, captured[3]["messages"][-1]["content"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_actionability_audit_respects_step_limit_and_token_accounting(self, _convert):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response(Message(content="Please clarify.")),
        )))
        result = ReactAgent(AgentConfig(max_steps=1), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.total_tokens, 5)
        self.assertEqual(result.final_message, "")
        self.assertTrue(result.events[0]["actionability_audit_triggered"])

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
            response(Message(content="handled")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        mcp = FakeMCP()
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))
        tool_result = result.events[0]["tool_results"][0]
        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 3)
        self.assertEqual(mcp.calls, [])
        self.assertTrue(result.events[1]["actionability_audit_triggered"])
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
