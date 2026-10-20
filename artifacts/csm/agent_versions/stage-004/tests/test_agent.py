import copy
import json
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
    def __init__(self, fail=False, tools=None, output=None):
        self.closed = False
        self.calls = []
        self.fail = fail
        self.tools = tools or [{"name": "find", "inputSchema": {"type": "object"}}]
        self.output = output if output is not None else {"content": "abcdef"}

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise ConnectionError("lost MCP connection")
        return self.output

    def close(self):
        self.closed = True


class AgentTests(unittest.TestCase):
    def task(self, mcp, user_prompt=""):
        return SimpleNamespace(
            action_type="mcp", mcp_servers=[object()],
            system_prompt="system", user_prompt=user_prompt,
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
        preview = result.events[0]["tool_results"][0]["content"]
        self.assertIn("MCP tool result is incomplete", preview)
        self.assertIn("omitted_chars=", preview)
        self.assertIn("local__inspect_tool_result", preview)
        self.assertEqual(captured[1]["messages"][-1]["tool_call_id"], "call-1")
        tool_names = [tool["function"]["name"] for tool in captured[0]["tools"]
                      if "function" in tool]
        self.assertIn("local__inspect_tool_result", tool_names)
        self.assertIn("local__register_objectives", tool_names)
        self.assertIn("local__submit_completion", tool_names)
        self.assertEqual(captured[0]["max_completion_tokens"], 16_384)
        self.assertTrue(result.initial_messages[0]["content"].startswith("system"))
        self.assertIn("constraint ledger", result.initial_messages[0]["content"])
        self.assertIn("identity constraints independently", result.initial_messages[0]["content"])
        self.assertIn("exact identifier fields", result.initial_messages[0]["content"])
        self.assertIn("requested outcomes as required final-state constraints",
                      result.initial_messages[0]["content"])
        self.assertIn("subject, exact comparator", result.initial_messages[0]["content"])
        self.assertIn("threshold, required time scope, current value",
                      result.initial_messages[0]["content"])
        self.assertIn("anticipated delta from each proposed mutation",
                      result.initial_messages[0]["content"])
        self.assertIn("cumulative deltas of the planned mutation batch",
                      result.initial_messages[0]["content"])
        self.assertIn("for `x < N`, equality is a violation",
                      result.initial_messages[0]["content"])
        self.assertIn("for `x <= N`, equality may satisfy the bound",
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
        self.assertIn("compare near-equivalent records semantically",
                      result.initial_messages[0]["content"])
        self.assertIn("Do not ask for confirmation when policy directs proceeding",
                      result.initial_messages[0]["content"])
        self.assertIn("lifecycle histories", result.initial_messages[0]["content"])
        self.assertIn("Distinguish an operation label from the required final operational outcome",
                      result.initial_messages[0]["content"])
        self.assertIn("exact identity and version", result.initial_messages[0]["content"])
        self.assertIn("incident symptoms, timestamps, and policy constraints",
                      result.initial_messages[0]["content"])
        self.assertIn("result order, identifier magnitude, or timestamp alone",
                      result.initial_messages[0]["content"])
        self.assertIn("manually calling a link or update tool does not convert it",
                      result.initial_messages[0]["content"])
        self.assertIn("plausible alternatives rejected", result.initial_messages[0]["content"])
        self.assertIn("empty exact-name lookup is not proof of absence",
                      result.initial_messages[0]["content"])
        self.assertIn("explicitly truncated tool result as incomplete evidence",
                      result.initial_messages[0]["content"])
        self.assertIn("partial identifiers, including suffixes",
                      result.initial_messages[0]["content"])
        self.assertIn("default lifecycle state describes creation state only",
                      result.initial_messages[0]["content"])
        self.assertEqual(result.initial_messages[1]["content"], "")
        self.assertIn("compare near-equivalent existing records",
                      result.initial_messages[0]["content"])
        self.assertIn("structured attributes such as type, category, location",
                      result.initial_messages[0]["content"])
        self.assertIn("Require literal name identity only when",
                      result.initial_messages[0]["content"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "get_records"}},
    ])
    def test_truncated_result_can_be_recovered_by_query_and_paging(self, _convert):
        get_records = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="get_records", arguments="{}"),
        )
        captured = []
        handle = None

        def create(**kwargs):
            nonlocal handle
            captured.append(copy.deepcopy(kwargs))
            if len(captured) == 1:
                return response(Message(tool_calls=[get_records]), "tool_calls")
            if len(captured) == 2:
                preview = kwargs["messages"][-1]["content"]
                handle = preview.split("handle=", 1)[1].splitlines()[0]
                query = SimpleNamespace(
                    id="call-2", type="function",
                    function=SimpleNamespace(
                        name="local__inspect_tool_result",
                        arguments=f'{{"handle": "{handle}", "query": "target-9300"}}',
                    ),
                )
                return response(Message(tool_calls=[query]), "tool_calls")
            if len(captured) == 3:
                page = SimpleNamespace(
                    id="call-3", type="function",
                    function=SimpleNamespace(
                        name="local__inspect_tool_result",
                        arguments=f'{{"handle": "{handle}", "offset": 20, "length": 40}}',
                    ),
                )
                return response(Message(tool_calls=[page]), "tool_calls")
            return response(Message(content="found"))

        long_value = {"content": ("x" * 120) + "TARGET-9300" + ("y" * 120)}
        mcp = FakeMCP(output=long_value)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(tool_result_chars=40), client).run(self.task(mcp))

        self.assertEqual(result.status, "done")
        self.assertEqual(mcp.calls, [("get_records", {})])
        query_result = json.loads(result.events[1]["tool_results"][0]["content"])
        self.assertTrue(query_result["ok"])
        self.assertEqual(query_result["matches"][0]["match_offset"], 133)
        self.assertIn("TARGET-9300", query_result["matches"][0]["excerpt"])
        page_result = json.loads(result.events[2]["tool_results"][0]["content"])
        self.assertEqual(page_result["offset"], 20)
        self.assertEqual(page_result["length"], 40)
        self.assertTrue(page_result["has_more"])
        self.assertTrue(result.events[1]["tool_results"][0]["local"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "find"}},
    ])
    def test_invalid_or_expired_result_handle_fails_safely(self, _convert):
        inspect = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(
                name="local__inspect_tool_result",
                arguments='{"handle": "result_from_another_run", "offset": 0, "length": 20}',
            ),
        )
        responses = [
            response(Message(tool_calls=[inspect]), "tool_calls"),
            response(Message(content="not available")),
        ]
        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(), client).run(self.task(mcp))

        self.assertEqual(result.status, "done")
        self.assertEqual(mcp.calls, [])
        error = json.loads(result.events[0]["tool_results"][0]["content"])
        self.assertFalse(error["ok"])
        self.assertIn("unknown or expired", error["error"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "find"}},
    ])
    def test_stored_results_do_not_survive_a_run(self, _convert):
        find = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        first_responses = [
            response(Message(tool_calls=[find]), "tool_calls"),
            response(Message(content="done")),
        ]
        agent = ReactAgent(
            AgentConfig(tool_result_chars=100),
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: first_responses.pop(0),
            ))),
        )
        first = agent.run(self.task(FakeMCP()))
        old_handle = first.events[0]["tool_results"][0]["handle"]

        inspect = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(
                name="local__inspect_tool_result",
                arguments=json.dumps({
                    "handle": old_handle,
                    "offset": 0,
                    "length": 20,
                }),
            ),
        )
        second_responses = [
            response(Message(tool_calls=[inspect]), "tool_calls"),
            response(Message(content="expired")),
        ]
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: second_responses.pop(0),
        )))
        second = agent.run(self.task(FakeMCP()))

        error = json.loads(second.events[0]["tool_results"][0]["content"])
        self.assertFalse(error["ok"])
        self.assertIn("expired", error["error"])

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "find"}},
    ])
    def test_short_result_stays_inline_with_run_local_handle(self, _convert):
        call = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="find", arguments="{}"),
        )
        responses = [
            response(Message(tool_calls=[call]), "tool_calls"),
            response(Message(content="done")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(tool_result_chars=100), client).run(
            self.task(FakeMCP())
        )

        tool_result = result.events[0]["tool_results"][0]
        self.assertFalse(tool_result["truncated"])
        self.assertEqual(tool_result["content"], '{"content": "abcdef"}')
        self.assertTrue(tool_result["handle"].startswith("result_"))

    @patch("eog_tools.agent._to_openai_tools", return_value=[
        {"type": "function", "function": {"name": "local__inspect_tool_result"}},
    ])
    def test_reserved_local_tool_name_collision_is_rejected(self, _convert):
        result = ReactAgent(
            AgentConfig(),
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())),
        ).run(self.task(FakeMCP()))
        self.assertEqual(result.status, "error")
        self.assertIn("collides with reserved local tool", result.error)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_nonempty_request_exposes_only_objective_registration_first(self, _convert):
        captured = []
        register = SimpleNamespace(
            id="call-0", type="function",
            function=SimpleNamespace(
                name="local__register_objectives",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "lookup",
                        "outcome": "The requested record is found",
                        "depends_on": [],
                    }],
                }),
            ),
        )
        complete = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(
                name="local__submit_completion",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "lookup",
                        "status": "authoritatively_prohibited",
                        "evidence": "Authoritative policy prohibits disclosing the record.",
                    }],
                }),
            ),
        )
        responses = [
            response(Message(content="draft")),
            response(Message(tool_calls=[register]), "tool_calls"),
            response(Message(tool_calls=[complete]), "tool_calls"),
            response(Message(content="I cannot disclose that record.")),
        ]

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(
            self.task(FakeMCP(), "Find the protected record.")
        )

        self.assertEqual(result.status, "done")
        self.assertEqual(
            [tool["function"]["name"] for tool in captured[0]["tools"]],
            ["local__register_objectives"],
        )
        self.assertIn("objective_registration_checkpoint", result.events[0])
        later_tool_names = [tool["function"]["name"] for tool in captured[2]["tools"]
                            if "function" in tool]
        self.assertIn("find", later_tool_names)
        self.assertIn("local__submit_completion", later_tool_names)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_unresolved_objective_blocks_finalization_until_resolved(self, _convert):
        register = SimpleNamespace(
            id="call-0", type="function",
            function=SimpleNamespace(
                name="local__register_objectives",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "update-case",
                        "outcome": "Case 4 is updated to the requested state",
                        "depends_on": [],
                    }],
                }),
            ),
        )
        mutation = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="update_case", arguments='{"case_id": 4}'),
        )
        unresolved = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(
                name="local__submit_completion",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "update-case",
                        "status": "unresolved",
                        "evidence": "The requested state has not been reread yet.",
                    }],
                }),
            ),
        )
        correction = SimpleNamespace(
            id="call-3", type="function",
            function=SimpleNamespace(name="update_case", arguments='{"case_id": 4, "state": "new"}'),
        )
        complete = SimpleNamespace(
            id="call-4", type="function",
            function=SimpleNamespace(
                name="local__submit_completion",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "update-case",
                        "status": "satisfied",
                        "evidence": "A fresh read shows case 4 is new.",
                    }],
                }),
            ),
        )
        captured = []

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            return responses.pop(0)

        responses = [
            response(Message(tool_calls=[register]), "tool_calls"),
            response(Message(tool_calls=[mutation]), "tool_calls"),
            response(Message(content="draft")),
            response(Message(tool_calls=[unresolved]), "tool_calls"),
            response(Message(tool_calls=[correction]), "tool_calls"),
            response(Message(tool_calls=[complete]), "tool_calls"),
            response(Message(content="Case 4 was updated successfully.")),
        ]
        mcp = FakeMCP()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = ReactAgent(AgentConfig(), client).run(
            self.task(mcp, "Update case 4 to the requested state.")
        )

        self.assertEqual(result.status, "done")
        self.assertEqual(result.steps, 7)
        self.assertEqual(result.final_message, "Case 4 was updated successfully.")
        self.assertEqual(len(mcp.calls), 2)
        checkpoints = [
            event for event in result.events if "completion_required_checkpoint" in event
        ]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(
            checkpoints[0]["completion_required_checkpoint"]["mutation_tools"],
            ["update_case"],
        )
        finalizations = [event for event in result.events if "finalization_checkpoint" in event]
        self.assertEqual(len(finalizations), 1)
        self.assertEqual(
            finalizations[0]["finalization_checkpoint"]["mutation_tools"],
            ["update_case"],
        )
        rejected = json.loads(result.events[3]["tool_results"][0]["content"])
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["unresolved"], ["update-case"])
        reconciliations = [
            event for event in result.events if "reconciliation_checkpoint" in event
        ]
        self.assertEqual(len(reconciliations), 2)
        self.assertEqual(
            reconciliations[0]["reconciliation_checkpoint"]["mutation_tools"],
            ["update_case"],
        )
        self.assertEqual(captured[2]["messages"][-1]["role"], "user")
        reconciliation = captured[2]["messages"][-1]["content"]
        self.assertIn("authoritative policy consequences", reconciliation)
        self.assertIn("after the full mutation batch", reconciliation)
        self.assertIn("cumulative effect of multiple mutations", reconciliation)
        self.assertIn("Equality at a strict boundary", reconciliation)
        self.assertIn("inclusive boundary", reconciliation)
        self.assertIn("lifecycle-coupled records", reconciliation)
        self.assertIn("observed provenance", reconciliation)
        self.assertIn("required final operational outcome", reconciliation)
        self.assertIn("compare them explicitly", reconciliation)
        self.assertIn("automated search remains search-discovered", reconciliation)
        self.assertIn("plausible alternatives rejected", reconciliation)
        self.assertEqual(captured[3]["messages"][-1]["role"], "user")
        checkpoint = captured[3]["messages"][-1]["content"]
        self.assertIn("every explicit objective", checkpoint)
        self.assertIn("every policy-triggered obligation", checkpoint)
        self.assertIn("satisfied, genuinely prohibited by authoritative policy, or unresolved",
                      checkpoint)
        self.assertIn("original exact comparator, threshold, and required time scope",
                      checkpoint)
        self.assertIn("never reinterpret `<` as `<=`", checkpoint)
        self.assertIn("final-state invariant as a pre-mutation eligibility test",
                      checkpoint)
        self.assertIn("cumulative capacity consumption", checkpoint)
        self.assertIn("same original operator", checkpoint)
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
        self.assertIn("Reuse a semantically equivalent record", checkpoint)
        self.assertIn("local__submit_completion", checkpoint)
        self.assertIn("required final operational outcome", checkpoint)
        self.assertIn("title and body relevance", checkpoint)
        self.assertIn("identifier magnitude, or timestamp alone", checkpoint)
        self.assertIn("manually calling a link or update tool", checkpoint)
        self.assertIn("chosen value, its deciding evidence", checkpoint)
        self.assertIn("correct it through normal MCP tools before finalization", checkpoint)
        self.assertEqual(captured[6]["messages"][-1]["role"], "user")
        finalization = captured[6]["messages"][-1]["content"]
        self.assertIn("concise user-facing completion", finalization)
        self.assertIn("relevant identifiers", finalization)
        self.assertIn("authoritatively prohibited items", finalization)
        self.assertIn("Do not expose the internal constraint ledger", finalization)
        self.assertIn("Do not call tools or make further changes", finalization)

    @patch("eog_tools.agent._to_openai_tools", return_value=[{"type": "function"}])
    def test_later_mutation_invalidates_accepted_completion(self, _convert):
        register = SimpleNamespace(
            id="call-0", type="function",
            function=SimpleNamespace(
                name="local__register_objectives",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "update",
                        "outcome": "The requested update is complete",
                        "depends_on": [],
                    }],
                }),
            ),
        )
        complete = lambda call_id, evidence: SimpleNamespace(
            id=call_id, type="function",
            function=SimpleNamespace(
                name="local__submit_completion",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "update",
                        "status": "satisfied",
                        "evidence": evidence,
                    }],
                }),
            ),
        )
        mutation = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(name="update_case", arguments='{"case_id": 4}'),
        )
        responses = [
            response(Message(tool_calls=[register]), "tool_calls"),
            response(Message(tool_calls=[complete("call-1", "Initial evidence.")]), "tool_calls"),
            response(Message(tool_calls=[mutation]), "tool_calls"),
            response(Message(content="draft")),
            response(Message(tool_calls=[complete("call-3", "Fresh post-update evidence.")]),
                     "tool_calls"),
            response(Message(content="Updated.")),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(), client).run(
            self.task(FakeMCP(), "Update case 4.")
        )

        self.assertEqual(result.status, "done")
        self.assertTrue(result.events[2]["completion_invalidated"])
        self.assertIn("completion_required_checkpoint", result.events[3])
        self.assertEqual(
            len([event for event in result.events if "finalization_checkpoint" in event]),
            2,
        )

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
            len([event for event in result.events if "completion_required_checkpoint" in event]),
            0,
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
        self.assertFalse(any(
            "completion_required_checkpoint" in event for event in result.events
        ))

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
    def test_accepted_completion_checkpoint_respects_step_limit(self, _convert):
        register = SimpleNamespace(
            id="call-0", type="function",
            function=SimpleNamespace(
                name="local__register_objectives",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "notify",
                        "outcome": "The requested notification is sent",
                        "depends_on": [],
                    }],
                }),
            ),
        )
        mutation = SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="send_notification", arguments="{}"),
        )
        complete = SimpleNamespace(
            id="call-2", type="function",
            function=SimpleNamespace(
                name="local__submit_completion",
                arguments=json.dumps({
                    "objectives": [{
                        "id": "notify",
                        "status": "satisfied",
                        "evidence": "The MCP send result succeeded.",
                    }],
                }),
            ),
        )
        responses = [
            response(Message(tool_calls=[register]), "tool_calls"),
            response(Message(tool_calls=[mutation]), "tool_calls"),
            response(Message(tool_calls=[complete]), "tool_calls"),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: responses.pop(0),
        )))
        result = ReactAgent(AgentConfig(max_steps=3), client).run(
            self.task(FakeMCP(), "Send the notification.")
        )
        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.final_message, "")
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
