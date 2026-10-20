"""A local GPT-5 ReAct loop over an EvoHarnessBench task's MCP tools."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


_LOCAL_INSPECTION_TOOL_NAME = "local__inspect_tool_result"
_LOCAL_OBJECTIVE_LEDGER_TOOL_NAME = "local__register_objectives"
_LOCAL_COMPLETION_TOOL_NAME = "local__submit_completion"
_LOCAL_INSPECTION_MAX_CHARS = 2_000
_LOCAL_INSPECTION_TOOL = {
    "type": "function",
    "function": {
        "name": _LOCAL_INSPECTION_TOOL_NAME,
        "description": (
            "Inspect a complete result retained from an earlier MCP tool call in this run. "
            "Use a case-insensitive query to find matching excerpts, or offset and length "
            "to page through the result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "query": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _LOCAL_INSPECTION_MAX_CHARS,
                },
            },
            "required": ["handle"],
            "additionalProperties": False,
        },
    },
}

_LOCAL_OBJECTIVE_LEDGER_TOOL = {
    "type": "function",
    "function": {
        "name": _LOCAL_OBJECTIVE_LEDGER_TOOL_NAME,
        "description": (
            "Register the explicit requested outcomes and their dependencies before using "
            "normal MCP tools. Call this once with a compact ledger grounded in the user's "
            "request; do not add inferred or optional work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objectives": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "outcome": {"type": "string"},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["id", "outcome", "depends_on"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["objectives"],
            "additionalProperties": False,
        },
    },
}

_LOCAL_COMPLETION_TOOL = {
    "type": "function",
    "function": {
        "name": _LOCAL_COMPLETION_TOOL_NAME,
        "description": (
            "Reconcile every registered objective against current MCP evidence. Finalization "
            "is allowed only when every objective is satisfied or authoritatively prohibited. "
            "Report unresolved work honestly; the tool will keep execution active."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objectives": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["satisfied", "authoritatively_prohibited", "unresolved"],
                            },
                            "evidence": {"type": "string"},
                        },
                        "required": ["id", "status", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["objectives"],
            "additionalProperties": False,
        },
    },
}


_SEMANTIC_SELECTION_GUIDANCE = """
Semantic mutation decision record:
- Distinguish an operation label from the required final operational outcome. For lifecycle work, derive the final state from the whole request and policy, not from the verb alone; record any policy-permitted intermediate transition needed to reach the requested handling outcome.
- A default lifecycle state describes creation state only. When the requested operation begins active work or starts a work-linked SLA, perform the policy-permitted transition to in_progress before linking or starting that SLA, even when the record was initially created as new.
- When retrieval returns multiple strong candidates, compare them explicitly using exact identity and version, title and body relevance, incident symptoms, timestamps, and policy constraints. Do not select by result order, identifier magnitude, or timestamp alone when those signals conflict with stronger semantic evidence.
- Before creating a named entity, compare near-equivalent existing records by normalized name meaning and the requested operational attributes. Keep structured attributes such as type, category, location, lifecycle state, and membership separate from the display name; reuse a semantically matching record when those attributes satisfy the request. Require literal name identity only when the request explicitly requires a rename, an exact display name, or creation of a distinct record.
- Preserve discovery provenance through later mutations. A record found through automated search remains search-discovered; manually calling a link or update tool does not convert it into a manual discovery.
- For each exact enum, lifecycle state, selected record, and relationship-purpose value, record the chosen value, its deciding evidence, and the plausible alternatives rejected. Treat a choice as unresolved when that comparison does not support it, and correct it through normal MCP tools before finalization.
""".strip()

_MUTATION_SAFETY_CONTEXT = """
Mutation safety protocol:
- Before changing state, make an internal constraint ledger from the original request and authoritative policy: entity identity constraints, required final values, policy-required changes, and fields to preserve. Treat explicit requested outcomes as required final-state constraints. For every quantitative constraint, record its subject, exact comparator (`<`, `<=`, `>`, `>=`, or `=`), threshold, required time scope, current value, and the anticipated delta from each proposed mutation.
- Before every count-changing mutation, compute the projected post-mutation value using the cumulative deltas of the planned mutation batch and compare that projection with the original comparator. Preserve strict versus inclusive bounds exactly: for `x < N`, equality is a violation and the mutation is permitted only when projected `x` remains strictly below `N`; for `x <= N`, equality may satisfy the bound. Reject the mutation or choose another valid candidate when its projected result would violate the constraint.
- Classify every proposed mutation field as explicitly requested, schema-required, policy-required under a satisfied condition, or optional/defaulted. Omit optional fields unless the user selected them or authoritative policy requires them; compatibility with another value, discoverability, or the availability of a related record is not authority to populate a field.
- Interpret disjunctive requirements such as "A or B" as requiring only the selected branch. Treat conditional relationship rules such as "if used" as validation guards for a relationship when present, not as instructions to create that relationship.
- Discover each referenced entity from its identity constraints independently of requested final-state constraints. When the request may require transitioning an entity into a desired state, do not require that state in the initial lookup.
- Treat fuzzy search results as candidates only. Validate candidates against exact identifier fields, and retry state-independent or alternate-field lookups before concluding that the entity is absent.
- An empty exact-name lookup is not proof of absence. Try bounded alternate-name lookups and, when broad collections are available, inspect the complete collection rather than relying on a truncated preview.
- Treat every explicitly truncated tool result as incomplete evidence. Use the local result-inspection tool to search or page the retained result whenever omitted content could change an entity-resolution, policy, or refusal decision.
- Treat partial identifiers, including suffixes, as discovery constraints rather than automatic blockers. Search available collections using the known fragment, then accept only a unique candidate validated against all available relational constraints such as account, product, location, lifecycle status, and contact primacy.
- Inspect the located entity's current state, perform any policy-permitted prerequisite transitions implied by the requested final state when the active role has authority and policy does not require separate approval, and then complete the requested relationship or mutation.
- Apply this conflict hierarchy: an authoritative policy prohibition blocks action; inferred relevance concerns or metadata mismatches require preservation and investigation but do not override an exact requested target unless authoritative policy requires the values to match.
- Preserve every field that the request or policy does not authorize changing. A mismatch with a related record is not permission to rewrite either record.
- Do not replace a target that the request explicitly distinguishes as a separate record. Otherwise, compare near-equivalent records semantically before deciding that creation is necessary.
- Reconcile explicit lifecycle histories, totals, and counts with the field's resulting-state semantics. When the requested resulting count is inferable, do not substitute a default increment.
- Use normal MCP mutation tools for authorized changes, reread affected primary records, and reconcile every requested objective with the current state before claiming completion. Do not ask for confirmation when policy directs proceeding; pause only to explain a genuine authoritative prohibition.
""".strip()

_POST_MUTATION_RECONCILIATION = f"""
Post-mutation reconciliation checkpoint: before continuing, recompute the authoritative policy consequences of the completed mutations. Recompute every affected aggregate, count, quota, or capacity value after the full mutation batch, including the cumulative effect of multiple mutations, and compare the observed result with the exact comparator and time scope recorded in the constraint ledger. Equality at a strict boundary such as `x < N` is a discrepancy, not threshold success; equality satisfies an inclusive boundary such as `x <= N` only when the original constraint used that comparator. Inspect dependent or lifecycle-coupled records for state that became stale because a primary record changed, and use the normal MCP tools to correct any policy-required effects. For enum and relationship attributes, derive the exact value from the operation's observed provenance and the deciding policy condition, not from the relationship's intuitive purpose. Reread the affected records as needed, and do not claim the mutation batch is complete while a required dependent effect, quantitative constraint, or provenance-sensitive value remains unverified.

{_SEMANTIC_SELECTION_GUIDANCE}
""".strip()

_VERIFICATION_CHECKPOINT = f"""
Mutation verification checkpoint: do not provide the final answer yet. Separately enumerate (1) every explicit objective in the original request and (2) every policy-triggered obligation caused by the performed mutations. Mark each item satisfied, genuinely prohibited by authoritative policy, or unresolved, including objectives whose referenced entity was not previously found. An objective with a quantitative constraint cannot be marked satisfied until you reread the observed post-mutation value and compare it with the original exact comparator, threshold, and required time scope from the constraint ledger; never reinterpret `<` as `<=`, `>` as `>=`, or a final-state invariant as a pre-mutation eligibility test. Recompute affected aggregates after all mutations, so cumulative capacity consumption is verified rather than checking each mutation against a stale starting value. Reread each affected primary record and each lifecycle-coupled or dependent record with the available MCP read tools, compare their current values with the original request, authoritative policy, and your constraint ledger, and preserve fields that were not authorized to change. Audit every non-null relationship in the affected records, including relationships not requested by the user: classify it as explicitly requested, schema-required, or policy-required under a satisfied condition, and treat any unsupported extra relationship as a discrepancy. For every exact enum or relationship value, identify the deciding policy condition and observed evidence, including how the operation or relationship was actually discovered or produced; compatibility, discoverability, or related-record availability alone is not sufficient. Apply disjunctive and conditional policy literally: "A or B" requires only the selected branch, and "if used" validates a relationship without requiring its creation. For an unresolved entity, retry valid state-independent and alternate-field lookups, validate exact identifier fields, inspect its current state, and perform policy-permitted prerequisite transitions before the requested relationship or mutation. If an objective or policy obligation is unresolved only because an available prerequisite or dependent correction has not yet been performed, and the active role is authorized to perform it without separate approval, continue using the normal MCP mutation tools until all primary and dependent states are verified. Reuse a semantically equivalent record unless the request explicitly distinguishes the target as a separate record or requires an exact rename. Correct any remaining discrepancy through the normal MCP tools when a supported correction is available, including unsupported extra relationships, stale lifecycle state, totals, counts, quotas, or capacities, then reread and compare again with the same original operator. Finally call {_LOCAL_COMPLETION_TOOL_NAME} with fresh evidence for every registered objective; ask for confirmation only when authoritative policy requires it, otherwise explain any genuine prohibition.

{_SEMANTIC_SELECTION_GUIDANCE}
""".strip()

_FINAL_RESPONSE_CHECKPOINT = """
Final response checkpoint: the local completion submission was accepted. Now provide a concise user-facing completion using the verified results above. Summarize the completed actions and relevant identifiers, and accurately disclose any authoritatively prohibited items that matter to the request. Do not expose the internal constraint ledger, verification checklist, checkpoint narration, or tool trace. Do not call tools or make further changes; any later mutation would invalidate this acceptance and require fresh reconciliation.
""".strip()

_OBJECTIVE_LEDGER_CONTEXT = f"""
Objective-led completion protocol:
- Before normal MCP work, call {_LOCAL_OBJECTIVE_LEDGER_TOOL_NAME} once to register every explicit requested outcome and dependency as compact durable working state.
- Keep MCP tool evidence distinct from assertions. After mutations, reread affected records and call {_LOCAL_COMPLETION_TOOL_NAME} with current evidence for every registered objective.
- Finalize only after that local completion submission is accepted. Missing, pending, or unresolved objectives keep the tool loop active; never relabel them as prohibited without authoritative policy evidence.
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


def _truncated_preview(
    tool_name: str,
    handle: str,
    content: str,
    excerpt_chars: int,
) -> str:
    head_chars = (excerpt_chars + 1) // 2
    tail_chars = excerpt_chars // 2
    head = content[:head_chars]
    tail = content[-tail_chars:] if tail_chars else ""
    omitted = max(0, len(content) - len(head) - len(tail))
    return (
        "MCP tool result is incomplete because it exceeded the inline preview limit.\n"
        f"tool={tool_name}\nhandle={handle}\n"
        f"total_chars={len(content)} omitted_chars={omitted}\n"
        f"Use {_LOCAL_INSPECTION_TOOL_NAME} with this handle and either a "
        "case-insensitive query or bounded offset/length paging before treating omitted "
        "evidence as absent.\n"
        f"--- head excerpt ---\n{head}\n--- tail excerpt ---\n{tail}"
    )


def _inspection_error(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _register_objectives(arguments: dict[str, Any]) -> tuple[dict[str, dict[str, Any]] | None, str]:
    objectives = arguments.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        return None, _inspection_error("objectives must be a non-empty array")
    ledger: dict[str, dict[str, Any]] = {}
    for item in objectives:
        if not isinstance(item, dict):
            return None, _inspection_error("each objective must be an object")
        objective_id = item.get("id")
        outcome = item.get("outcome")
        dependencies = item.get("depends_on")
        if not isinstance(objective_id, str) or not objective_id.strip():
            return None, _inspection_error("objective id must be a non-empty string")
        if objective_id in ledger:
            return None, _inspection_error(f"duplicate objective id: {objective_id}")
        if not isinstance(outcome, str) or not outcome.strip():
            return None, _inspection_error(f"objective {objective_id} needs a non-empty outcome")
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(value, str) or not value for value in dependencies)
        ):
            return None, _inspection_error(
                f"objective {objective_id} depends_on must be an array of objective ids"
            )
        ledger[objective_id] = {
            "outcome": outcome.strip(),
            "depends_on": list(dependencies),
        }
    for objective_id, item in ledger.items():
        unknown = [
            dependency for dependency in item["depends_on"]
            if dependency not in ledger or dependency == objective_id
        ]
        if unknown:
            return None, _inspection_error(
                f"objective {objective_id} has unknown or self dependencies: {unknown}"
            )
    return ledger, json.dumps({
        "ok": True,
        "registered": [
            {"id": objective_id, **item} for objective_id, item in ledger.items()
        ],
        "instruction": "Proceed through normal MCP tools and submit fresh terminal evidence.",
    }, ensure_ascii=False)


def _submit_completion(
    ledger: dict[str, dict[str, Any]],
    arguments: dict[str, Any],
) -> tuple[bool, str]:
    submissions = arguments.get("objectives")
    if not isinstance(submissions, list):
        return False, _inspection_error("objectives must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for item in submissions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return False, _inspection_error("each completion item needs an objective id")
        objective_id = item["id"]
        if objective_id in by_id:
            return False, _inspection_error(f"duplicate completion objective id: {objective_id}")
        by_id[objective_id] = item
    missing = sorted(set(ledger) - set(by_id))
    unknown = sorted(set(by_id) - set(ledger))
    unresolved: list[str] = []
    invalid: list[str] = []
    for objective_id in ledger:
        item = by_id.get(objective_id)
        if item is None:
            continue
        status = item.get("status")
        evidence = item.get("evidence")
        if status not in ("satisfied", "authoritatively_prohibited", "unresolved"):
            invalid.append(objective_id)
        elif not isinstance(evidence, str) or not evidence.strip():
            invalid.append(objective_id)
        elif status == "unresolved":
            unresolved.append(objective_id)
    if missing or unknown or invalid or unresolved:
        return False, json.dumps({
            "ok": False,
            "error": "completion rejected; continue normal MCP work and reconcile again",
            "missing": missing,
            "unknown": unknown,
            "invalid": invalid,
            "unresolved": unresolved,
        }, ensure_ascii=False)
    return True, json.dumps({
        "ok": True,
        "accepted": True,
        "instruction": "All registered objectives are terminal; provide the final response.",
    }, ensure_ascii=False)


def _inspect_stored_result(
    result_store: dict[str, str],
    arguments: dict[str, Any],
) -> str:
    handle = arguments.get("handle")
    if not isinstance(handle, str) or not handle:
        return _inspection_error("handle must be a non-empty string")
    content = result_store.get(handle)
    if content is None:
        return _inspection_error("unknown or expired result handle")

    query = arguments.get("query")
    has_paging = "offset" in arguments or "length" in arguments
    if query is not None and has_paging:
        return _inspection_error("provide query or offset/length paging, not both")
    if query is not None:
        if not isinstance(query, str) or not query:
            return _inspection_error("query must be a non-empty string")
        folded_content = content.casefold()
        folded_query = query.casefold()
        matches: list[dict[str, Any]] = []
        cursor = 0
        while len(matches) < 8:
            position = folded_content.find(folded_query, cursor)
            if position < 0:
                break
            start = max(0, position - 120)
            end = min(len(content), position + len(query) + 120)
            matches.append({
                "match_offset": position,
                "excerpt_offset": start,
                "excerpt": content[start:end],
            })
            cursor = position + max(1, len(query))
        has_more = folded_content.find(folded_query, cursor) >= 0
        return json.dumps({
            "ok": True,
            "mode": "query",
            "handle": handle,
            "query": query,
            "total_chars": len(content),
            "matches": matches,
            "has_more": has_more,
        }, ensure_ascii=False)

    if not has_paging:
        return _inspection_error("provide query or offset/length paging")
    offset = arguments.get("offset", 0)
    length = arguments.get("length", min(1_000, _LOCAL_INSPECTION_MAX_CHARS))
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return _inspection_error("offset must be a non-negative integer")
    if (
        isinstance(length, bool)
        or not isinstance(length, int)
        or not 1 <= length <= _LOCAL_INSPECTION_MAX_CHARS
    ):
        return _inspection_error(
            f"length must be an integer from 1 to {_LOCAL_INSPECTION_MAX_CHARS}"
        )
    if offset > len(content):
        return _inspection_error("offset exceeds result size")
    end = min(len(content), offset + length)
    return json.dumps({
        "ok": True,
        "mode": "page",
        "handle": handle,
        "offset": offset,
        "length": end - offset,
        "total_chars": len(content),
        "content": content[offset:end],
        "has_more": end < len(content),
    }, ensure_ascii=False)


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
            mcp_tools = list(_to_openai_tools(mcp.list_tools()))
            if not mcp_tools:
                raise RuntimeError("task exposed no MCP tools")
            mcp_tool_names = {
                name for tool in mcp_tools if (name := _tool_name(tool))
            }
            reserved_tool_names = {
                _LOCAL_INSPECTION_TOOL_NAME,
                _LOCAL_OBJECTIVE_LEDGER_TOOL_NAME,
                _LOCAL_COMPLETION_TOOL_NAME,
            }
            collisions = sorted(reserved_tool_names & mcp_tool_names)
            if collisions:
                raise RuntimeError(
                    f"MCP tool name collides with reserved local tool: {collisions[0]}"
                )
            tools = [
                *mcp_tools,
                _LOCAL_INSPECTION_TOOL,
                _LOCAL_OBJECTIVE_LEDGER_TOOL,
                _LOCAL_COMPLETION_TOOL,
            ]
            result_store: dict[str, str] = {}
            system_prompt = task.system_prompt or ""
            requires_objective_ledger = bool((task.user_prompt or "").strip())
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n{_MUTATION_SAFETY_CONTEXT}"
                        f"\n\n{_SEMANTIC_SELECTION_GUIDANCE}"
                        f"\n\n{_OBJECTIVE_LEDGER_CONTEXT}"
                    ).strip(),
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
            objective_ledger: dict[str, dict[str, Any]] = {}
            completion_accepted = False
            for step in range(1, self.config.max_steps + 1):
                available_tools = (
                    [_LOCAL_OBJECTIVE_LEDGER_TOOL]
                    if requires_objective_ledger and not objective_ledger
                    else tools
                )
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=available_tools,
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
                        and requires_objective_ledger
                        and not objective_ledger
                    ):
                        event["objective_registration_checkpoint"] = {}
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Register the explicit requested outcomes with "
                                f"{_LOCAL_OBJECTIVE_LEDGER_TOOL_NAME} before continuing."
                            ),
                        })
                        continue
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
                        and objective_ledger
                        and not completion_accepted
                    ):
                        event["completion_required_checkpoint"] = {
                            "mutation_tools": sorted(mutation_tools_used),
                            "objectives": sorted(objective_ledger),
                        }
                        messages.append({"role": "user", "content": _VERIFICATION_CHECKPOINT})
                        continue
                    result.final_message = message.content or ""
                    if choice.finish_reason == "stop":
                        result.status = "done"
                    else:
                        result.status = "error"
                        result.error = f"unexpected model finish reason: {choice.finish_reason}"
                    break
                batch_mutation_tools: set[str] = set()
                completion_accepted_this_batch = False
                for call in message.tool_calls:
                    if call.type != "function":
                        raise RuntimeError(f"unsupported tool call type: {call.type}")
                    any_tool_used = True
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    if call.function.name == _LOCAL_OBJECTIVE_LEDGER_TOOL_NAME:
                        if objective_ledger:
                            content = _inspection_error("objective ledger is already registered")
                        elif len(message.tool_calls) != 1:
                            content = _inspection_error(
                                "register objectives in a dedicated tool call turn"
                            )
                        else:
                            registered, content = _register_objectives(arguments)
                            if registered is not None:
                                objective_ledger = registered
                                completion_accepted = False
                        event["tool_results"].append({
                            "call_id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                            "content": content,
                            "truncated": False,
                            "local": True,
                        })
                    elif call.function.name == _LOCAL_COMPLETION_TOOL_NAME:
                        if not objective_ledger:
                            content = _inspection_error("register objectives before completion")
                        elif len(message.tool_calls) != 1:
                            content = _inspection_error(
                                "submit completion in a dedicated tool call turn"
                            )
                        else:
                            accepted, content = _submit_completion(
                                objective_ledger, arguments
                            )
                            completion_accepted = accepted
                            completion_accepted_this_batch = accepted
                        event["tool_results"].append({
                            "call_id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                            "content": content,
                            "truncated": False,
                            "local": True,
                        })
                    elif call.function.name == _LOCAL_INSPECTION_TOOL_NAME:
                        content = _inspect_stored_result(result_store, arguments)
                        event["tool_results"].append({
                            "call_id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                            "content": content,
                            "truncated": False,
                            "local": True,
                        })
                    else:
                        output = mcp.call_tool(call.function.name, arguments)
                        if _is_mutation_tool(call.function.name):
                            if completion_accepted:
                                event["completion_invalidated"] = True
                            completion_accepted = False
                            completion_accepted_this_batch = False
                            mutation_tools_used.add(call.function.name)
                            batch_mutation_tools.add(call.function.name)
                        full_content = json.dumps(output, ensure_ascii=False, default=str)
                        handle = f"result_{uuid.uuid4().hex}"
                        result_store[handle] = full_content
                        truncated = len(full_content) > self.config.tool_result_chars
                        content = (
                            _truncated_preview(
                                call.function.name,
                                handle,
                                full_content,
                                self.config.tool_result_chars,
                            )
                            if truncated
                            else full_content
                        )
                        event["tool_results"].append({
                            "call_id": call.id,
                            "name": call.function.name,
                            "arguments": arguments,
                            "content": content,
                            "truncated": truncated,
                            "handle": handle,
                            "total_chars": len(full_content),
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
                elif completion_accepted_this_batch:
                    event["finalization_checkpoint"] = {
                        "mutation_tools": sorted(mutation_tools_used),
                        "objectives": sorted(objective_ledger),
                    }
                    messages.append({"role": "user", "content": _FINAL_RESPONSE_CHECKPOINT})
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
