"""A local GPT-5 ReAct loop over an EvoHarnessBench task's MCP tools."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


_MUTATION_SAFETY_CONTEXT = """
Mutation safety protocol:
- Before changing state, make an internal constraint ledger from the original request and authoritative policy: required final values, policy-required changes, and fields to preserve.
- Preserve every field that the request or policy does not authorize changing. A mismatch with a related record is a conflict to investigate, not permission to rewrite either record.
- Reconcile explicit lifecycle histories, totals, and counts with the field's resulting-state semantics. When the requested resulting count is inferable, do not substitute a default increment.
- After state changes, reread affected primary records and compare their current values with every explicit constraint before claiming completion.
""".strip()

_VERIFICATION_CHECKPOINT = """
Mutation verification checkpoint: do not provide the final answer yet. Reread each affected primary record with the available MCP read tools, compare its current values with the original request's explicit constraints and your constraint ledger, and preserve fields that were not authorized to change. Correct any discrepancy through the normal MCP tools, including lifecycle totals or counts, then provide the final response.
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
            mutation_tools_used: set[str] = set()
            verification_requested = False
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
                        and mutation_tools_used
                        and not verification_requested
                    ):
                        verification_requested = True
                        event["verification_checkpoint"] = {
                            "mutation_tools": sorted(mutation_tools_used),
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
                for call in message.tool_calls:
                    if call.type != "function":
                        raise RuntimeError(f"unsupported tool call type: {call.type}")
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    output = mcp.call_tool(call.function.name, arguments)
                    if _is_mutation_tool(call.function.name):
                        mutation_tools_used.add(call.function.name)
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
