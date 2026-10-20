"""A local GPT-5 ReAct loop over an EvoHarnessBench task's MCP tools."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


MUTATION_FIDELITY_GUIDANCE = """Execution fidelity protocol:
- Treat the environment policy above as authoritative.
- Treat safe, read-only MCP searches and record reads as validation steps, not mutations, and perform them before refusing because identifiers, names, relationships, or requester context are missing.
- Resolve descriptive references with the available read-only tools using supplied attributes such as assignee, subject, state, reporter, and record relationships; never require an ID when those descriptors can identify the record.
- Ambiguity is a post-lookup condition. Refuse or report candidate matches only after discovery returns no match, multiple unresolved matches, an authoritative permission denial, or no required discovery tool.
- Apply requester-role requirements using identity or authorization context supplied in the task or verified by tools. If no requester identity is exposed, do not invent a missing-role blocker; rely on the tool's access controls and halt on an authoritative denial.
- Before acting, form and retain an internal checklist of every explicit requested change plus every deterministic policy-derived postcondition or coupled field.
- Before each state-changing tool call, reread the original request and environment policy, then compare the proposed arguments with that checklist.
- When policy determines a dependent value and the tool schema makes that field writable, include it in the mutation instead of assuming a backend default or delayed recalculation.
- Compare every state-changing tool result with all intended postconditions. If the result is incomplete or inconsistent, use the normal MCP tools for a corrective mutation or authoritative read-back before finishing.
- Do not claim success or speculate about backend timing while a required postcondition is visibly unmet. Report failure only when the normal tools do not support the correction or reject it.
- Preserve every materially distinct fact, qualifier, relationship, and requested outcome in structured and free-text fields; do not collapse distinct clauses merely because a paraphrase seems equivalent.
- Do not invent requirements, copy wording unnecessarily, or optimize for hidden evaluators; include only details supported by the request and environment policy."""


def _to_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from simple_agentic_evals import to_openai_tools

    return to_openai_tools(mcp_tools)


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
            system_context = task.system_prompt or ""
            if system_context:
                system_context += f"\n\n{MUTATION_FIDELITY_GUIDANCE}"
            else:
                system_context = MUTATION_FIDELITY_GUIDANCE
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_context},
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
