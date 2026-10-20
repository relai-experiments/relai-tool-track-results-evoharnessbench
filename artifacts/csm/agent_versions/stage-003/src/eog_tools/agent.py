"""A local GPT-5 ReAct loop over an EvoHarnessBench task's MCP tools."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


_MUTATION_SAFETY_CONTEXT = """
Mutation safety protocol:
- Before changing state, make an internal constraint ledger from the original request and authoritative policy: entity identity constraints, required final values, policy-required changes, and fields to preserve. Treat explicit requested outcomes as required final-state constraints.
- Classify every proposed mutation field as explicitly requested, schema-required, policy-required under a satisfied condition, or optional/defaulted. Omit optional fields unless the user selected them or authoritative policy requires them; compatibility with another value, discoverability, or the availability of a related record is not authority to populate a field.
- Interpret disjunctive requirements such as "A or B" as requiring only the selected branch. Treat conditional relationship rules such as "if used" as validation guards for a relationship when present, not as instructions to create that relationship.
- Discover each referenced entity from its identity constraints independently of requested final-state constraints. When the request may require transitioning an entity into a desired state, do not require that state in the initial lookup.
- Treat fuzzy search results as candidates only. Validate candidates against exact identifier fields, and retry state-independent or alternate-field lookups before concluding that the entity is absent.
- Inspect the located entity's current state, perform any policy-permitted prerequisite transitions implied by the requested final state when the active role has authority and policy does not require separate approval, and then complete the requested relationship or mutation.
- Apply this conflict hierarchy: an authoritative policy prohibition blocks action; inferred relevance concerns or metadata mismatches require preservation and investigation but do not override an exact requested target unless authoritative policy requires the values to match.
- Preserve every field that the request or policy does not authorize changing. A mismatch with a related record is not permission to rewrite either record.
- An existing alternative record does not satisfy an explicitly identified target and must not replace it merely because the alternative already has other preferred attributes.
- Reconcile explicit lifecycle histories, totals, and counts with the field's resulting-state semantics. When the requested resulting count is inferable, do not substitute a default increment.
- Use normal MCP mutation tools for authorized changes, reread affected primary records, and reconcile every requested objective with the current state before claiming completion. Do not ask for confirmation when policy directs proceeding; pause only to explain a genuine authoritative prohibition.
""".strip()

_POST_MUTATION_RECONCILIATION = """
Post-mutation reconciliation checkpoint: before continuing, recompute the authoritative policy consequences of the completed mutations. Inspect dependent or lifecycle-coupled records for state that became stale because a primary record changed, and use the normal MCP tools to correct any policy-required effects. For enum and relationship attributes, derive the exact value from the operation's observed provenance and the deciding policy condition, not from the relationship's intuitive purpose. Reread the affected records as needed, and do not claim the mutation batch is complete while a required dependent effect or provenance-sensitive value remains unverified.
""".strip()

_VERIFICATION_CHECKPOINT = """
Mutation verification checkpoint: do not provide the final answer yet. Separately enumerate (1) every explicit objective in the original request and (2) every policy-triggered obligation caused by the performed mutations. Mark each item satisfied, genuinely prohibited by authoritative policy, or unresolved, including objectives whose referenced entity was not previously found. Reread each affected primary record and each lifecycle-coupled or dependent record with the available MCP read tools, compare their current values with the original request, authoritative policy, and your constraint ledger, and preserve fields that were not authorized to change. Audit every non-null relationship in the affected records, including relationships not requested by the user: classify it as explicitly requested, schema-required, or policy-required under a satisfied condition, and treat any unsupported extra relationship as a discrepancy. For every exact enum or relationship value, identify the deciding policy condition and observed evidence, including how the operation or relationship was actually discovered or produced; compatibility, discoverability, or related-record availability alone is not sufficient. Apply disjunctive and conditional policy literally: "A or B" requires only the selected branch, and "if used" validates a relationship without requiring its creation. For an unresolved entity, retry valid state-independent and alternate-field lookups, validate exact identifier fields, inspect its current state, and perform policy-permitted prerequisite transitions before the requested relationship or mutation. If an objective or policy obligation is unresolved only because an available prerequisite or dependent correction has not yet been performed, and the active role is authorized to perform it without separate approval, continue using the normal MCP mutation tools until all primary and dependent states are verified. Do not substitute an alternative record for an explicitly identified target. Correct any remaining discrepancy through the normal MCP tools when a supported correction is available, including unsupported extra relationships, stale lifecycle state, totals, or counts, then provide the final response; ask for confirmation only when authoritative policy requires it, otherwise explain any genuine prohibition.
""".strip()

_FINAL_RESPONSE_CHECKPOINT = """
Final response checkpoint: using the verified results above, now provide a concise user-facing completion. Summarize the completed actions and relevant identifiers, and accurately disclose any unresolved or authoritatively prohibited items that matter to the request. Do not expose the internal constraint ledger, verification checklist, checkpoint narration, or tool trace. Do not call tools or make further changes; this turn is only to communicate the verified outcome.
""".strip()

_NO_MUTATION_AUTHORITY_CHECKPOINT = """
Authority and action completion checkpoint: you used read-only tools and are about to finish while mutation-capable tools are available but unused. Review the original request and authoritative policy once before deciding whether to stop.
- A descriptive catalog of domain roles and responsibilities binds the runtime actor only when the prompt explicitly assigns the authenticated/current actor to that catalog role. A generic domain label such as calling the assistant an agent does not by itself establish that authenticated role assignment.
- Available tools establish capability, not authorization. They never override an explicit actor-role assignment, operation-level prohibition, approval requirement, or other authoritative policy restriction.
- If an explicit assignment or prohibition genuinely blocks a requested mutation, return a concise policy-grounded final answer without using mutation tools.
- Otherwise, do not treat descriptive role language as a refusal basis. Continue with the requested, sufficiently specified mutations through the normal tools, then use the existing reconciliation and verification workflow before finalizing.
Do not ask for confirmation merely because the request requires mutation when policy already permits proceeding.
""".strip()

# EOG mutation tools use action-oriented operation names (for example
# update_case, set_case_assignment_group, link_case_knowledge, and
# send_notification); read operations use names such as find/search/list/retrieve.
_MUTATION_TOOL_PREFIXES = (
    "create_", "update_", "delete_", "set_", "assign_", "link_", "unlink_",
    "send_", "add_", "remove_", "close_", "cancel_", "reopen_", "resolve_",
)


def _to_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from simple_agentic_evals import to_openai_tools

    return to_openai_tools(mcp_tools)


def _is_mutation_tool(name: str) -> bool:
    return name.startswith(_MUTATION_TOOL_PREFIXES)


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return tool.get("name") if isinstance(tool.get("name"), str) else ""


@dataclass(frozen=True)
class AgentConfig:
    model: str = "gpt-5"
    max_steps: int = 50
    max_completion_tokens: int = 16_384
    tool_result_chars: int = 4_000

    def __post_init__(self) -> None:
        if min(self.max_steps, self.max_completion_tokens, self.tool_result_chars) < 1:
            raise ValueError("agent limits must be positive")


@dataclass
class AgentResult:
    status: str = "running"
    error: str | None = None
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    final_message: str = ""
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReactAgent:
    def __init__(self, config: AgentConfig, client: Any) -> None:
        self.config = config
        self.client = client

    def run(self, task: Any) -> AgentResult:
        result = AgentResult()
        start = time.monotonic()
        mcp = None
        try:
            if task.action_type != "mcp" or not task.mcp_servers:
                raise ValueError("this agent requires an EOG MCP task")
            mcp = task.mcp_session(task.mcp_servers[0])
            tools = _to_openai_tools(mcp.list_tools())
            if not tools:
                raise RuntimeError("task exposed no MCP tools")
            system_prompt = task.system_prompt or ""
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\n{_MUTATION_SAFETY_CONTEXT}".strip(),
                },
                {"role": "user", "content": task.user_prompt or ""},
            ]
            result.initial_messages = [dict(message) for message in messages]
            result.tools = tools
            mutation_tools_available = {
                name for tool in tools if (name := _tool_name(tool)) and _is_mutation_tool(name)
            }
            any_tool_used = False
            mutation_tools_used: set[str] = set()
            authority_review_requested = False
            verification_requested = False
            finalization_requested = False
            for step in range(1, self.config.max_steps + 1):
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=tools,
                    max_completion_tokens=self.config.max_completion_tokens,
                )
                if not response.choices:
                    raise RuntimeError("model returned no choices")
                choice = response.choices[0]
                message = choice.message
                usage = response.usage
                if usage is not None:
                    result.input_tokens += usage.prompt_tokens or 0
                    result.output_tokens += usage.completion_tokens or 0
                    result.total_tokens += usage.total_tokens or (
                        (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
                    )
                result.steps = step
                event: dict[str, Any] = {
                    "step": step,
                    "finish_reason": choice.finish_reason,
                    "assistant": message.model_dump(exclude_none=True),
                    "usage": usage.model_dump(exclude_none=True) if usage else None,
                    "tool_results": [],
                }
                result.events.append(event)
                messages.append(message.model_dump(exclude_none=True))
                if choice.finish_reason == "length":
                    result.status = "output_limit"
                    break
                if not message.tool_calls:
                    if (
                        choice.finish_reason == "stop"
                        and any_tool_used
                        and mutation_tools_available
                        and not mutation_tools_used
                        and not authority_review_requested
                    ):
                        authority_review_requested = True
                        event["authority_review_checkpoint"] = {
                            "mutation_tools": sorted(mutation_tools_available),
                        }
                        messages.append({
                            "role": "user",
                            "content": _NO_MUTATION_AUTHORITY_CHECKPOINT,
                        })
                        continue
                    if (
                        choice.finish_reason == "stop"
                        and mutation_tools_used
                        and not verification_requested
                    ):
                        verification_requested = True
                        event["verification_checkpoint"] = {
                            "mutation_tools": sorted(mutation_tools_used),
                        }
                        messages.append({"role": "user", "content": _VERIFICATION_CHECKPOINT})
                        continue
                    if (
                        choice.finish_reason == "stop"
                        and mutation_tools_used
                        and verification_requested
                        and not finalization_requested
                    ):
                        finalization_requested = True
                        event["finalization_checkpoint"] = {
                            "mutation_tools": sorted(mutation_tools_used),
                        }
                        messages.append({"role": "user", "content": _FINAL_RESPONSE_CHECKPOINT})
                        continue
                    result.final_message = message.content or ""
                    if choice.finish_reason == "stop":
                        result.status = "done"
                    else:
                        result.status = "error"
                        result.error = f"unexpected model finish reason: {choice.finish_reason}"
                    break
                batch_mutation_tools: set[str] = set()
                for call in message.tool_calls:
                    if call.type != "function":
                        raise RuntimeError(f"unsupported tool call type: {call.type}")
                    any_tool_used = True
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    output = mcp.call_tool(call.function.name, arguments)
                    if _is_mutation_tool(call.function.name):
                        mutation_tools_used.add(call.function.name)
                        batch_mutation_tools.add(call.function.name)
                    content = json.dumps(output, ensure_ascii=False, default=str)
                    truncated = len(content) > self.config.tool_result_chars
                    content = content[: self.config.tool_result_chars]
                    event["tool_results"].append({
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": arguments,
                        "content": content,
                        "truncated": truncated,
                    })
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                if batch_mutation_tools:
                    event["reconciliation_checkpoint"] = {
                        "mutation_tools": sorted(batch_mutation_tools),
                    }
                    messages.append({
                        "role": "user",
                        "content": _POST_MUTATION_RECONCILIATION,
                    })
            else:
                result.status = "step_limit"
        except Exception as exc:  # Save partial trace; the runner treats this as an error.
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            if mcp is not None:
                try:
                    mcp.close()
                except Exception as exc:
                    result.status = "error"
                    result.error = f"MCP close failed: {type(exc).__name__}: {exc}"
            result.latency_s = time.monotonic() - start
        return result
