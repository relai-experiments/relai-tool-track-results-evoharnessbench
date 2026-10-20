"""A local GPT-5 ReAct loop over an EvoHarnessBench task's MCP tools."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


RECOVERY_TOOL_NAME = "read_tool_result"
RECOVERY_MAX_CHARS = 4_000
RECOVERY_GUIDANCE = (
    "Tool results may include explicit incomplete previews. A preview is not evidence "
    "that omitted fields or resources are absent. When omitted content could affect the "
    f"task, use {RECOVERY_TOOL_NAME} to inspect the stored result before making an "
    "absence claim or a dependent mutation."
)
EXECUTION_GUIDANCE = (
    "When copying or transforming a resource, preserve every requested field that the "
    "available tools support. Keep references and destructive actions anchored to "
    "entities verified in tool results. For scalar text mutations, preserve the observed "
    "original text verbatim and honor any explicit placement or formatting instructions. "
    "Otherwise, use only the minimal separator needed for continuation; do not invent "
    "paragraph breaks or list formatting. Before mutating, compare the complete candidate "
    "value with the observed original and requested addition, including whitespace and "
    "punctuation. Mutation arguments show what was requested, not what persisted; inspect "
    "mutation results and read back consequential writes when needed before claiming "
    "success. Successful storage is insufficient if persisted text differs from the "
    "intended composition; issue a corrective mutation when necessary. Before any "
    "date-bearing mutation, inventory explicit numeric years, yearless dates, relative-time "
    "phrases, and the operation or entity each one scopes. An explicit year tied to the "
    "same destination or operation outranks a generic next-future-occurrence default. If "
    "temporal anchors genuinely conflict, request clarification when permitted. When the "
    "execution policy requires proceeding, use the most specific explicit anchor and state "
    "that interpretation rather than silently choosing the next future occurrence."
)
FINALIZATION_CHECKPOINT = (
    "Before finalizing, audit the work against the user's request using only observed "
    "tool evidence. Map every requested operation to a result, inspect mutation responses "
    "for dropped or defaulted fields, and read back consequential writes when persisted "
    "state is not established. Compare persisted start dates, end dates, and time zones "
    "with the original explicit years, yearless dates, relative-time phrases, and their "
    "operation or entity scope. Resolve unfinished destructive actions, identity mismatches, "
    "and temporal-anchor mismatches. If anything is missing or uncertain, issue corrective "
    "tool calls now; otherwise provide the final answer."
)


def _to_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from simple_agentic_evals import to_openai_tools

    return to_openai_tools(mcp_tools)


def _recovery_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": RECOVERY_TOOL_NAME,
            "description": (
                "Read a bounded chunk of an oversized tool result from this run. "
                "Use the opaque handle and offsets supplied in an incomplete preview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "Opaque handle from an incomplete tool-result preview.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Zero-based character offset in the serialized result.",
                    },
                    "length": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": RECOVERY_MAX_CHARS,
                        "description": "Maximum number of characters to return.",
                    },
                },
                "required": ["handle", "offset", "length"],
                "additionalProperties": False,
            },
        },
    }


def _tool_names(tools: list[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            if not isinstance(name, str):
                function = tool.get("function")
                name = function.get("name") if isinstance(function, dict) else None
        else:
            name = getattr(tool, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _preview(content: str, handle: str, limit: int) -> tuple[str, dict[str, Any]]:
    head_chars = (limit + 1) // 2
    tail_chars = limit - head_chars
    omitted_chars = len(content) - head_chars - tail_chars
    metadata = {
        "handle": handle,
        "total_chars": len(content),
        "omitted_chars": omitted_chars,
        "head_chars": head_chars,
        "tail_chars": tail_chars,
        "recovery_tool": RECOVERY_TOOL_NAME,
    }
    preview = {
        "incomplete_tool_result": True,
        **metadata,
        "head": content[:head_chars],
        "tail": content[-tail_chars:] if tail_chars else "",
        "instructions": (
            f"This is an incomplete preview. Use {RECOVERY_TOOL_NAME} with this handle, "
            "an offset, and a bounded length to inspect omitted content."
        ),
    }
    return json.dumps(preview, ensure_ascii=False), metadata


def _read_stored_result(
    store: dict[str, str], arguments: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    handle = arguments.get("handle")
    offset = arguments.get("offset")
    length = arguments.get("length")
    error_type = None
    message = None
    if not isinstance(handle, str) or handle not in store:
        error_type = "invalid_handle"
        message = "The result handle is invalid or no longer available in this run."
    elif not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset < len(store[handle]):
        error_type = "invalid_offset"
        message = f"offset must be between 0 and {len(store[handle]) - 1}."
    elif (
        not isinstance(length, int)
        or isinstance(length, bool)
        or not 1 <= length <= RECOVERY_MAX_CHARS
    ):
        error_type = "invalid_length"
        message = f"length must be between 1 and {RECOVERY_MAX_CHARS}."
    if error_type is not None:
        value = {"error": {"type": error_type, "message": message}}
        return json.dumps(value, ensure_ascii=False), {
            "success": False,
            "handle": handle,
            "offset": offset,
            "requested_length": length,
            "error": error_type,
        }

    content = store[handle]
    chunk = content[offset:offset + length]
    end_offset = offset + len(chunk)
    value = {
        "handle": handle,
        "offset": offset,
        "length": len(chunk),
        "end_offset": end_offset,
        "total_chars": len(content),
        "end_of_result": end_offset >= len(content),
        "content": chunk,
    }
    return json.dumps(value, ensure_ascii=False), {
        "success": True,
        "handle": handle,
        "offset": offset,
        "requested_length": length,
        "returned_length": len(chunk),
        "end_offset": end_offset,
        "total_chars": len(content),
        "end_of_result": end_offset >= len(content),
    }


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
        stored_results: dict[str, str] = {}
        external_mcp_called = False
        finalization_audit_run = False
        try:
            if task.action_type != "mcp" or not task.mcp_servers:
                raise ValueError("this agent requires an EOG MCP task")
            mcp = task.mcp_session(task.mcp_servers[0])
            mcp_tools = mcp.list_tools()
            if not mcp_tools:
                raise RuntimeError("task exposed no MCP tools")
            if RECOVERY_TOOL_NAME in _tool_names(mcp_tools):
                raise ValueError(f"MCP tool name {RECOVERY_TOOL_NAME!r} is reserved")
            tools = _to_openai_tools(mcp_tools)
            if RECOVERY_TOOL_NAME in _tool_names(tools):
                raise ValueError(f"MCP tool name {RECOVERY_TOOL_NAME!r} is reserved")
            tools.append(_recovery_tool())
            system_prompt = task.system_prompt or ""
            if system_prompt:
                system_prompt += "\n\n"
            system_prompt += f"{RECOVERY_GUIDANCE}\n\n{EXECUTION_GUIDANCE}"
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.user_prompt or ""},
            ]
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
                    if (
                        choice.finish_reason == "stop"
                        and external_mcp_called
                        and not finalization_audit_run
                    ):
                        finalization_audit_run = True
                        event["finalization_checkpoint"] = {
                            "activated": True,
                            "content": FINALIZATION_CHECKPOINT,
                        }
                        messages.append({"role": "user", "content": FINALIZATION_CHECKPOINT})
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
                    tool_event = {
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": arguments,
                    }
                    if call.function.name == RECOVERY_TOOL_NAME:
                        content, recovery = _read_stored_result(stored_results, arguments)
                        tool_event.update({
                            "content": content,
                            "truncated": False,
                            "local": True,
                            "recovery": recovery,
                        })
                    else:
                        output = mcp.call_tool(call.function.name, arguments)
                        external_mcp_called = True
                        full_content = json.dumps(output, ensure_ascii=False, default=str)
                        truncated = len(full_content) > self.config.tool_result_chars
                        if truncated:
                            handle = uuid.uuid4().hex
                            stored_results[handle] = full_content
                            content, preview = _preview(
                                full_content, handle, self.config.tool_result_chars,
                            )
                            tool_event["preview"] = preview
                        else:
                            content = full_content
                        tool_event.update({
                            "content": content,
                            "truncated": truncated,
                            "local": False,
                        })
                    event["tool_results"].append(tool_event)
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
            stored_results.clear()
            result.latency_s = time.monotonic() - start
        return result
