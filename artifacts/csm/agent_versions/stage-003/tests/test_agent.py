import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eog_tools.agent import AgentConfig, ReactAgent


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
    def __init__(self, fail=False, tools=None):
        self.closed = False
        self.calls = []
        self.fail = fail
        self.tools = tools or [{"name": "find", "inputSchema": {"type": "object"}}]

    def list_tools(self):
        return self.tools

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
        self.assertTrue(result.initial_messages[0]["content"].startswith("system"))
        self.assertIn("constraint ledger", result.initial_messages[0]["content"])
        self.assertIn("identity constraints independently", result.initial_messages[0]["content"])
        self.assertIn("exact identifier fields", result.initial_messages[0]["content"])
        self.assertIn("requested outcomes as required final-state constraints",
                      result.initial_messages[0]["content"])
        self.assertIn("Classify every proposed mutation field",
                      result.initial_messages[0]["content"])
        self.assertIn("Omit optional fields unless the user selected them",
                      result.initial_messages[0]["content"])
        self.assertIn("compatibility with another value, discoverability",
                      result.initial_messages[0]["content"])
        self.assertIn('"A or B" as requiring only the selected branch',
                      result.initial_messages[0]["content"])
        self.assertIn('"if used" as validation guards',
                      result.initial_messages[0]["content"])
        self.assertIn("policy-permitted prerequisite transitions",
                      result.initial_messages[0]["content"])
        self.assertIn("authoritative policy prohibition", result.initial_messages[0]["content"])
        self.assertIn("metadata mismatches", result.initial_messages[0]["content"])
        self.assertIn("alternative record does not satisfy an explicitly identified target",
                      result.initial_messages[0]["content"])
        self.assertIn("Do not ask for confirmation when policy directs proceeding",
                      result.initial_messages[0]["content"])
        self.assertIn("lifecycle histories", result.initial_messages[0]["content"])
        self.assertEqual(result.initial_messages[1]["content"], "user")

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_mutation_final_answer_gets_verification_and_finalization_turns(self, _convert):
        mutation = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="update_case", arguments='{"case_id": 4}'),
        )
        correction = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="update_case", arguments='{"case_id": 4, "state": "new"}'),
        )
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        responses = [
            response(Message(tool_calls=[mutation]), "tool_calls"),
            response(Message(content="draft")),
            response(Message(tool_calls=[correction]), "tool_calls"),
            response(Message(content="verified checklist")),
            response(Message(content="Case 4 was updated successfully.")),
        ]
        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 5)
        self.assertEqual(result.final_message, "Case 4 was updated successfully.")
        self.assertEqual(len(mcp.calls), 2)
        checkpoints = [event for event in result.events if "verification_checkpoint" in event]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["verification_checkpoint"]["mutation_tools"], ["update_case"])
        finalizations = [event for event in result.events if "finalization_checkpoint" in event]
        self.assertEqual(len(finalizations), 1)
        self.assertEqual(
            finalizations[0]["finalization_checkpoint"]["mutation_tools"],
            ["update_case"],
        )
        reconciliations = [
            event for event in result.events if "reconciliation_checkpoint" in event
        ]
        self.assertEqual(len(reconciliations), 2)
        self.assertEqual(
            reconciliations[0]["reconciliation_checkpoint"]["mutation_tools"],
            ["update_case"],
        )
        self.assertEqual(captured[1]["messages"][-1]["role"], "user")
        reconciliation = captured[1]["messages"][-1]["content"]
        self.assertIn("authoritative policy consequences", reconciliation)
        self.assertIn("lifecycle-coupled records", reconciliation)
        self.assertIn("observed provenance", reconciliation)
        self.assertEqual(captured[2]["messages"][-1]["role"], "user")
        checkpoint = captured[2]["messages"][-1]["content"]
        self.assertIn("every explicit objective", checkpoint)
        self.assertIn("every policy-triggered obligation", checkpoint)
        self.assertIn("satisfied, genuinely prohibited by authoritative policy, or unresolved",
                      checkpoint)
        self.assertIn("state-independent and alternate-field lookups", checkpoint)
        self.assertIn("lifecycle-coupled or dependent record", checkpoint)
        self.assertIn("Audit every non-null relationship", checkpoint)
        self.assertIn("unsupported extra relationship", checkpoint)
        self.assertIn('"A or B" requires only the selected branch', checkpoint)
        self.assertIn('"if used" validates a relationship', checkpoint)
        self.assertIn("deciding policy condition and observed evidence", checkpoint)
        self.assertIn("available prerequisite or dependent correction", checkpoint)
        self.assertIn("continue using the normal MCP mutation tools", checkpoint)
        self.assertIn("when a supported correction is available", checkpoint)
        self.assertIn("Do not substitute an alternative record", checkpoint)
        self.assertEqual(captured[4]["messages"][-2]["content"], "verified checklist")
        self.assertEqual(captured[4]["messages"][-1]["role"], "user")
        finalization = captured[4]["messages"][-1]["content"]
        self.assertIn("concise user-facing completion", finalization)
        self.assertIn("relevant identifiers", finalization)
        self.assertIn("unresolved or authoritatively prohibited items", finalization)
        self.assertIn("Do not expose the internal constraint ledger", finalization)
        self.assertIn("Do not call tools or make further changes", finalization)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_read_only_batch_gets_no_reconciliation_checkpoint(self, _convert):
        read = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        responses = [
            response(Message(tool_calls=[read]), "tool_calls"),
            response(Message(content="finished")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "done")
        self.assertFalse(any("reconciliation_checkpoint" in event for event in result.events))
        self.assertEqual(captured[1]["messages"][-1]["role"], "tool")

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "find_user"}},
        {"type": "function", "function": {"name": "update_user"}},
    ])
    def test_premature_refusal_gets_authority_review_then_can_mutate(self, _convert):
        read = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find_user", arguments="{}"),
        )
        mutation = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="update_user", arguments='{"user_id": 4}'),
        )
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        responses = [
            response(Message(tool_calls=[read]), "tool_calls"),
            response(Message(content="Only administrators can do that.")),
            response(Message(tool_calls=[mutation]), "tool_calls"),
            response(Message(content="draft")),
            response(Message(content="verified")),
            response(Message(content="User 4 was updated.")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.final_message, "User 4 was updated.")
        reviews = [event for event in result.events if "authority_review_checkpoint" in event]
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["authority_review_checkpoint"]["mutation_tools"], ["update_user"])
        review = captured[2]["messages"][-1]["content"]
        self.assertIn("explicitly assigns the authenticated/current actor", review)
        self.assertIn("Available tools establish capability, not authorization", review)
        self.assertIn("Continue with the requested, sufficiently specified mutations", review)
        self.assertEqual(
            len([event for event in result.events if "verification_checkpoint" in event]),
            1,
        )

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "find_user"}},
        {"type": "function", "function": {"name": "update_user"}},
    ])
    def test_explicit_role_prohibition_can_still_finish_without_mutation(self, _convert):
        read = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find_user", arguments="{}"),
        )
        responses = [
            response(Message(tool_calls=[read]), "tool_calls"),
            response(Message(content="draft refusal")),
            response(Message(content="My explicitly assigned role prohibits this update.")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(result.events), 3)
        self.assertIn("authority_review_checkpoint", result.events[1])
        self.assertFalse(any("verification_checkpoint" in event for event in result.events))

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "find_user"}},
        {"type": "function", "function": {"name": "update_user"}},
    ])
    def test_informational_lookup_finishes_after_single_authority_review(self, _convert):
        read = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find_user", arguments="{}"),
        )
        responses = [
            response(Message(tool_calls=[read]), "tool_calls"),
            response(Message(content="draft answer")),
            response(Message(content="The requested user is active.")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "done")
        self.assertEqual(result.final_message, "The requested user is active.")
        self.assertEqual(
            len([event for event in result.events if "authority_review_checkpoint" in event]),
            1,
        )

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "find_user"}},
        {"type": "function", "function": {"name": "update_user"}},
    ])
    def test_authority_review_respects_step_limit(self, _convert):
        read = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find_user", arguments="{}"),
        )
        responses = [
            response(Message(tool_calls=[read]), "tool_calls"),
            response(Message(content="draft refusal")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(max_steps=2), client).run(self.task(FakeMCP()))

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.final_message, "")
        self.assertIn("authority_review_checkpoint", result.events[-1])

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_finalization_checkpoint_respects_step_limit(self, _convert):
        mutation = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="send_notification", arguments="{}"),
        )
        responses = [
            response(Message(tool_calls=[mutation]), "tool_calls"),
            response(Message(content="draft")),
            response(Message(content="verified checklist")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(max_steps=3), client).run(self.task(FakeMCP()))
        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.final_message, "")
        self.assertEqual(
            len([event for event in result.events if "verification_checkpoint" in event]),
            1,
        )
        self.assertIn("finalization_checkpoint", result.events[-1])

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
