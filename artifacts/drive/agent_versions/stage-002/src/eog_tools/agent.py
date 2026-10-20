"""A local GPT-5 ReAct loop over an EvoHarnessBench task's MCP tools."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


READ_TOOL_RESULT = "read_tool_result"

TOOL_USE_POLICY = """Tool-use policy:
- Before acting, decompose the request into atomic obligations. For each obligation, track the target, requested mutation or outcome, requested communication channel, and the tool-result evidence that would prove completion.
- Keep obligations separate even when one tool offers incidental notification, message, metadata, or description fields. Such fields do not satisfy a distinct reply, comment, message, or content-edit obligation unless the user explicitly requested that channel.
- Before every tool call, compare the arguments with the user's explicit intent and the exposed tool schema.
- Treat file IDs and other resource identifiers as discoverable when exposed tools can search or list the relevant resources. Search before asking the user to supply them.
- Inspect current resource metadata, including existing labels and schema-allowed values, before asking the user to choose a metadata value.
- Treat semantic qualifiers such as application-specific, private, public, custom, or inherited as constraints on field selection. Updating a similarly named sibling field does not satisfy the request.
- Preserve explicit JSON types: booleans, numbers, null, arrays, and objects must retain their types unless the schema explicitly requires another type.
- An observation marked incomplete is not evidence that a search is exhaustive or has zero matches. Retrieve all relevant continuations with read_tool_result, and follow any service pagination, before making such a conclusion.
- Resource search predicates may be case-, punctuation-, or suffix-sensitive. An empty narrow query is not sufficient evidence that a resource is absent.
- For a failed resource lookup, start precise, then remove optional suffix assumptions, relax combined predicates one at a time, search distinctive tokens, follow all pagination, and inspect returned candidate metadata.
- Compare candidates semantically, including case-insensitive name comparison, without guessing or rewriting identifiers. Proceed only when one candidate clearly matches; use that candidate's exact returned resource ID and name for dependent actions. If multiple plausible candidates remain after reasonable retrieval fallbacks are exhausted, ask the user to clarify.
- Copying, duplicating, or cloning a resource preserves its observed source type or format unless the user explicitly requests conversion or export. A requested filename or extension is only a label and does not authorize conversion.
- Immediately before a copy, duplicate, or clone mutation, compare any proposed MIME or format argument with the observed source type. Omit optional MIME or format arguments unless conversion is explicit; if the schema requires one, use the observed source type.
- Preserve user-specified names literally. Track every requested mutation, verify each mutation response, and perform targeted reads of the resulting state before claiming completion.
- After a copy, duplicate, or clone mutation, verify that the resulting type matches the source type or the user's explicit conversion target. Treat an unexpected type as a failed mutation and correct it with normal tools before claiming completion.
- After a mutation, compare returned values and their types with the requested state. If a mismatch is repairable, make a corrective normal tool call before claiming success."""

ACTIONABILITY_AUDIT = """ACTIONABILITY AUDIT (one time):
The preceding assistant message is only a draft. Before accepting a tool-free completion, inspect the exposed tool schemas and classify each apparently missing input as one of: discoverable through tools, safely inferable from observed state, authorization-critical, or genuinely ambiguous. Resource identifiers and current metadata such as comments, permissions, labels, and allowed label values are discoverable when the tools expose relevant search, list, or get operations; call those tools before asking the user for that information. If retrieval or observed state leaves one unique safe path, continue with normal tool calls. Ask the user only when required authorization is absent or multiple plausible choices remain after reasonable retrieval. If no tool call is warranted, return the final answer without tools."""

FINALIZATION_AUDIT = """FINALIZATION AUDIT (mandatory):
The preceding assistant message is only a draft. Decompose the original request into atomic obligations, including each target, mutation or outcome, and requested communication channel. Match every obligation to independent external-tool evidence below; do not use one action as evidence for a different channel. Notification, message, metadata, or description fields incidental to another mutation do not prove that a distinct reply, comment, message, or content edit occurred unless that incidental channel was explicitly requested. read_tool_result is retrieval only and is not mutation evidence.

Audit every proposed completion claim against the original user request, the exposed tool schemas, the exact schema fields and arguments selected in tool calls, and the observed tool results. Pay particular attention to semantic qualifiers and similarly named sibling fields. Before claiming a resource is absent or asking the user to identify it, verify that reasonable broader retrieval fallbacks were exhausted, including relaxing case-, punctuation-, suffix-, and combined-predicate assumptions, searching distinctive tokens, following pagination, and inspecting candidate metadata. If one candidate clearly matches, continue with its exact returned resource ID and name; ask for clarification only when multiple plausible candidates remain. If an obligation has no matching successful evidence, retrieval was too narrow, or any requested field, value, type, or resulting state is missing or mismatched, do not finalize: make corrective normal tool calls now and verify their results. If the evidence supports every obligation and claim, return the final answer without tools. Never repeat an unsupported success claim."""

_AUDIT_REQUEST_CHARS = 4_000
_EVIDENCE_ARGUMENT_CHARS = 500
_EVIDENCE_IDENTIFIER_LIMIT = 8


def _to_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from simple_agentic_evals import to_openai_tools

    return to_openai_tools(mcp_tools)


def _continuation_tool(max_chars: int) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": READ_TOOL_RESULT,
            "description": (
                "Read another bounded character range from a prior incomplete tool result. "
                "Use the original tool call ID shown in the incomplete preview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "call_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": max_chars,
                    },
                },
                "required": ["call_id", "offset", "max_chars"],
                "additionalProperties": False,
            },
        },
    }


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
        elif isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def _result_chunk(content: str, call_id: str, offset: int, max_chars: int) -> str:
    end = min(len(content), offset + max_chars)
    has_more = end < len(content)
    return (
        f"[Tool result range: call_id={call_id}; incomplete={str(has_more).lower()}; "
        f"total_chars={len(content)}; range={offset}:{end}; "
        f"next_offset={end if has_more else 'none'}]\n"
        f"{content[offset:end]}"
    )


def _read_stored_result(
    observations: dict[str, str],
    arguments: dict[str, Any],
    max_chars: int,
) -> tuple[str, bool]:
    call_id = arguments.get("call_id")
    offset = arguments.get("offset")
    requested = arguments.get("max_chars")
    if not isinstance(call_id, str) or not call_id:
        return json.dumps({"error": "call_id must be a non-empty string"}), False
    if call_id not in observations:
        return json.dumps({"error": "unknown tool result call_id", "call_id": call_id}), False
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return json.dumps({"error": "offset must be a non-negative integer"}), False
    if not isinstance(requested, int) or isinstance(requested, bool) or not 1 <= requested <= max_chars:
        return json.dumps({
            "error": f"max_chars must be an integer between 1 and {max_chars}",
        }), False
    content = observations[call_id]
    if offset > len(content):
        return json.dumps({
            "error": "offset exceeds tool result length",
            "call_id": call_id,
            "total_chars": len(content),
        }), False
    end = min(len(content), offset + requested)
    return _result_chunk(content, call_id, offset, requested), end < len(content)


def _bounded_json(value: Any, max_chars: int) -> str:
    content = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "...[truncated]"


def _walk_result(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for item in value.values():
            values.extend(_walk_result(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_walk_result(item))
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            pass
        else:
            if decoded != value:
                values.extend(_walk_result(decoded))
    return values


def _result_status(output: Any) -> str:
    mappings = [value for value in _walk_result(output) if isinstance(value, dict)]
    if any(
        value.get("success") is False
        or str(value.get("status", "")).lower() in {"error", "failed", "failure"}
        or ("error" in value and value.get("error") not in (None, "", False))
        for value in mappings
    ):
        return "failure"
    if any(
        value.get("success") is True
        or str(value.get("status", "")).lower() in {"ok", "success", "succeeded"}
        for value in mappings
    ):
        return "success"
    return "success (no explicit error returned)"


def _result_identifiers(output: Any) -> list[str]:
    identifiers: list[str] = []
    for value in _walk_result(output):
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            normalized = key.lower().replace("_", "")
            if normalized == "id" or normalized.endswith("id"):
                if isinstance(item, (str, int)) and str(item) not in identifiers:
                    identifiers.append(str(item))
                    if len(identifiers) >= _EVIDENCE_IDENTIFIER_LIMIT:
                        return identifiers
    return identifiers


def _finalization_checkpoint(user_request: str, evidence: list[dict[str, Any]]) -> str:
    request = user_request[:_AUDIT_REQUEST_CHARS]
    if len(user_request) > _AUDIT_REQUEST_CHARS:
        request += "...[truncated]"
    lines = [FINALIZATION_AUDIT, "", "ORIGINAL USER REQUEST:", request, "", "EXTERNAL TOOL EVIDENCE:"]
    if not evidence:
        lines.append("- none")
    for index, item in enumerate(evidence, 1):
        identifiers = item["identifiers"]
        lines.append(
            f"- {index}. tool={item['name']}; status={item['status']}; "
            f"arguments={item['arguments']}; "
            f"resource_ids={_bounded_json(identifiers, _EVIDENCE_ARGUMENT_CHARS)}"
        )
    return "\n".join(lines)


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
            mcp_tools = mcp.list_tools()
            tools = _to_openai_tools(mcp_tools)
            if not tools:
                raise RuntimeError("task exposed no MCP tools")
            if READ_TOOL_RESULT in _tool_names(mcp_tools) | _tool_names(tools):
                raise RuntimeError(f"MCP tool name is reserved: {READ_TOOL_RESULT}")
            tools = [*tools, _continuation_tool(self.config.tool_result_chars)]
            observations: dict[str, str] = {}
            external_evidence: list[dict[str, Any]] = []
            system_prompt = task.system_prompt or ""
            user_prompt = task.user_prompt or ""
            system_context = (
                f"{system_prompt}\n\n{TOOL_USE_POLICY}" if system_prompt else TOOL_USE_POLICY
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_prompt},
            ]
            audit_pending = False
            external_tool_called = False
            actionability_audit_run = False
            result.initial_messages = [dict(message) for message in messages]
            result.tools = tools
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
                    if choice.finish_reason == "stop" and audit_pending:
                        event["finalization_audit_triggered"] = True
                        messages.append({
                            "role": "system",
                            "content": _finalization_checkpoint(user_prompt, external_evidence),
                        })
                        audit_pending = False
                        continue
                    if (
                        choice.finish_reason == "stop"
                        and not external_tool_called
                        and not actionability_audit_run
                    ):
                        event["actionability_audit_triggered"] = True
                        messages.append({"role": "system", "content": ACTIONABILITY_AUDIT})
                        actionability_audit_run = True
                        continue
                    result.final_message = message.content or ""
                    if choice.finish_reason == "stop":
                        result.status = "done"
                    else:
                        result.status = "error"
                        result.error = f"unexpected model finish reason: {choice.finish_reason}"
                    break
                for call in message.tool_calls:
                    if call.type != "function":
                        raise RuntimeError(f"unsupported tool call type: {call.type}")
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    if call.function.name == READ_TOOL_RESULT:
                        content, truncated = _read_stored_result(
                            observations, arguments, self.config.tool_result_chars,
                        )
                    else:
                        output = mcp.call_tool(call.function.name, arguments)
                        external_tool_called = True
                        audit_pending = True
                        external_evidence.append({
                            "name": call.function.name,
                            "arguments": _bounded_json(arguments, _EVIDENCE_ARGUMENT_CHARS),
                            "status": _result_status(output),
                            "identifiers": _result_identifiers(output),
                        })
                        full_content = json.dumps(output, ensure_ascii=False, default=str)
                        observations[call.id] = full_content
                        truncated = len(full_content) > self.config.tool_result_chars
                        content = (
                            _result_chunk(
                                full_content, call.id, 0, self.config.tool_result_chars,
                            )
                            if truncated else full_content
                        )
                    event["tool_results"].append({
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": arguments,
                        "content": content,
                        "truncated": truncated,
                    })
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
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
