"""A local GPT-5 ReAct loop over an EvoHarnessBench task's MCP tools."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from fractions import Fraction
from typing import Any


RECOVERY_TOOL_NAME = "read_tool_result"
TIME_INTERVAL_TOOL_NAME = "calculate_time_interval"
LOCAL_TOOL_NAMES = frozenset((RECOVERY_TOOL_NAME, TIME_INTERVAL_TOOL_NAME))
RECOVERY_MAX_CHARS = 4_000
READ_TOOL_PREFIXES = (
    "check_", "describe_", "fetch_", "find_", "get_", "inspect_", "list_",
    "lookup_", "query_", "read_", "search_",
)
RECOVERY_GUIDANCE = (
    "Tool results may include explicit incomplete previews. A preview is not evidence "
    "that omitted fields or resources are absent. When omitted content could affect the "
    f"task, use {RECOVERY_TOOL_NAME} to inspect the stored result before making an "
    "absence claim or a dependent mutation."
)
EXECUTION_GUIDANCE = (
    "When copying or transforming a resource, preserve every requested field that the "
    "available tools support, including writable lifecycle fields such as status or state. "
    "If validation rejects one field, repair only that field and retain unrelated source "
    "fields rather than silently dropping them. Treat an error-like mutation result as "
    "ambiguous because the write may have persisted. Before retrying the same mutation, "
    "inspect relevant persisted state with an available read tool; if no such read path "
    "exists, do not blindly retry and state the uncertainty. Keep references and destructive "
    "actions anchored to entities verified in tool results. Before acting, classify each "
    "requested operation's scope as an explicitly named entity, a primary or default entity, "
    "a quantified collection, or an account- or ownership-wide category. From observed tool "
    "evidence, maintain a compact coverage ledger for each operation containing the discovered "
    "eligible resource identifiers, action status, and verification status. Reconcile every "
    "discovered eligible resource for collection- or account-scoped operations; selecting a "
    "primary or default alias accounts only for that entity when discovery shows others. If an "
    "account-level setting is unavailable but the same control exists per owned resource, cover "
    "the discovered owned-resource set unless the request explicitly limits the operation to one "
    "resource. Never silently narrow the requested scope. For scalar text mutations, preserve the observed "
    "original text verbatim and honor any explicit placement or formatting instructions. "
    "Otherwise, use only the minimal separator needed for continuation; do not invent "
    "paragraph breaks or list formatting. Before mutating, compare the complete candidate "
    "value with the observed original and requested addition, including whitespace and "
    "punctuation. Mutation arguments show what was requested, not what persisted; inspect "
    "mutation results and read back consequential writes when needed before claiming "
    "success. Successful storage is insufficient if persisted text differs from the "
    "intended composition; issue a corrective mutation when necessary. Before any "
    "date-bearing mutation, inventory explicit numeric years, yearless dates, relative-time "
    "phrases, and the operation or entity each one scopes. Preserve explicit years; do not "
    "replace them with wall-clock defaults. For a yearless date, use task and entity context "
    "before choosing a year. A single wall-clock-derived interpretation followed by an empty "
    "date-filtered discovery is not sufficient evidence that matching resources are absent. "
    "Before skipping dependent operations, make one bounded recovery query: use a contextual "
    "anchor when available; otherwise test one adjacent-year interval or the smallest useful "
    "range broadening supported by the tool. Do not repeatedly widen the search. If multiple "
    "populated interpretations remain materially plausible, request clarification when "
    "permitted and do not make consequential mutations unless the execution policy explicitly "
    "requires proceeding. An explicit year tied to the same destination or operation outranks "
    "a generic next-future-occurrence default. When the execution policy requires proceeding, "
    "use the most specific explicit anchor and state that interpretation rather than silently "
    "choosing the next future occurrence."
    f" When a consequential tool argument is numeric and derived from a date range, "
    f"duration, or time-unit conversion, call {TIME_INTERVAL_TOOL_NAME} first and use its "
    "validated result. Treat words such as 'through' and 'including' as inclusive endpoint "
    "semantics only when they scope calendar dates, not elapsed datetime intervals."
)
FINALIZATION_CHECKPOINT = (
    "Before finalizing, audit the work against the user's request using only observed "
    "tool evidence. For each requested operation, compare its discovered eligible-resource "
    "set with the completed-and-verified set in the coverage ledger. A primary or default "
    "resource does not satisfy a collection- or account-scoped operation when other eligible "
    "resources were discovered. Do not finalize while any eligible resource is unaccounted "
    "for; issue corrective MCP calls for uncovered resources now. Map every requested operation "
    "to a result, inspect mutation responses "
    "for dropped or defaulted fields, including lifecycle status or state, and read back "
    "consequential writes when persisted state is not established. Compare persisted start dates, end dates, and time zones "
    "with the original explicit years, yearless dates, relative-time phrases, and their "
    "operation or entity scope. Resolve unfinished destructive actions, identity mismatches, "
    "and temporal-anchor mismatches. After retried creates, inspect destination state and "
    "patch or delete duplicate resources produced during this run before claiming success. "
    "Audit empty date-filtered retrievals that caused requested "
    "downstream operations to be skipped. If such a retrieval used an ambiguous yearless date "
    "and no bounded recovery was attempted, make one contextual, adjacent-year, or minimally "
    "broadened recovery call before accepting absence; do not repeatedly widen the search or "
    "override an explicit year. Compare every date-derived numeric mutation argument with the "
    f"result from {TIME_INTERVAL_TOOL_NAME} and the endpoint semantics requested by the user. "
    "If anything is missing or uncertain, issue corrective tool "
    "calls now; otherwise provide the final answer."
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


def _time_interval_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TIME_INTERVAL_TOOL_NAME,
            "description": (
                "Calculate and validate an ISO date or timezone-aware datetime interval "
                "before using a date-derived numeric value in a consequential tool call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "ISO 8601 calendar date or timezone-aware datetime.",
                    },
                    "end": {
                        "type": "string",
                        "description": "ISO 8601 calendar date or timezone-aware datetime.",
                    },
                    "endpoint_semantics": {
                        "type": "string",
                        "enum": ["inclusive_calendar_dates", "end_exclusive"],
                        "description": (
                            "Use inclusive_calendar_dates for date wording such as through "
                            "or including; use end_exclusive for elapsed intervals."
                        ),
                    },
                    "output_unit": {
                        "type": "string",
                        "enum": ["days", "hours", "minutes", "seconds"],
                    },
                },
                "required": ["start", "end", "endpoint_semantics", "output_unit"],
                "additionalProperties": False,
            },
        },
    }


def _interval_error(
    error_type: str, message: str, arguments: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    value = {
        "error": {"type": error_type, "message": message},
        "inputs": {
            key: arguments.get(key)
            for key in ("start", "end", "endpoint_semantics", "output_unit")
        },
    }
    return json.dumps(value, ensure_ascii=False), {
        "success": False,
        "error": error_type,
    }


def _parse_temporal(value: Any) -> tuple[str, date | datetime]:
    if not isinstance(value, str) or not value:
        raise ValueError("must be a non-empty ISO 8601 string")
    if "T" not in value:
        return "date", date.fromisoformat(value)
    parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("datetime inputs must include an explicit UTC offset")
    return "datetime", parsed


def _calculate_time_interval(
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    semantics = arguments.get("endpoint_semantics")
    unit = arguments.get("output_unit")
    if semantics not in {"inclusive_calendar_dates", "end_exclusive"}:
        return _interval_error(
            "invalid_endpoint_semantics",
            "endpoint_semantics must be inclusive_calendar_dates or end_exclusive.",
            arguments,
        )
    unit_seconds = {"days": 86_400, "hours": 3_600, "minutes": 60, "seconds": 1}
    if unit not in unit_seconds:
        return _interval_error(
            "invalid_output_unit",
            "output_unit must be days, hours, minutes, or seconds.",
            arguments,
        )
    try:
        start_kind, start = _parse_temporal(arguments.get("start"))
        end_kind, end = _parse_temporal(arguments.get("end"))
    except (TypeError, ValueError) as exc:
        return _interval_error("invalid_temporal_input", str(exc), arguments)
    if start_kind != end_kind:
        return _interval_error(
            "mixed_temporal_types",
            "start and end must both be calendar dates or both be datetimes.",
            arguments,
        )
    if semantics == "inclusive_calendar_dates" and start_kind != "date":
        return _interval_error(
            "invalid_endpoint_semantics",
            "inclusive_calendar_dates is valid only for calendar-date inputs.",
            arguments,
        )
    if end < start:
        return _interval_error(
            "reversed_interval", "end must not be earlier than start.", arguments,
        )

    if start_kind == "date":
        day_count = (end - start).days + (1 if semantics == "inclusive_calendar_dates" else 0)
        seconds = day_count * 86_400
        formula = (
            f"({arguments['end']} - {arguments['start']}).days"
            + (" + 1" if semantics == "inclusive_calendar_dates" else "")
            + f" = {day_count} days; {day_count} * 86400 seconds"
        )
    else:
        delta = end - start
        day_count = None
        seconds = (
            delta.days * 86_400
            + delta.seconds
            + Fraction(delta.microseconds, 1_000_000)
        )
        formula = (
            f"{arguments['end']} - {arguments['start']} = {seconds} seconds"
        )
    converted = Fraction(seconds) / unit_seconds[unit]
    result_value: int | str = (
        converted.numerator
        if converted.denominator == 1
        else f"{converted.numerator}/{converted.denominator}"
    )
    value = {
        "inputs": {
            "start": arguments["start"],
            "end": arguments["end"],
            "endpoint_semantics": semantics,
            "output_unit": unit,
        },
        "endpoint_semantics": semantics,
        "day_count": day_count,
        "result": result_value,
        "output_unit": unit,
        "formula": formula + f"; / {unit_seconds[unit]} = {result_value} {unit}",
    }
    return json.dumps(value, ensure_ascii=False), {
        "success": True,
        "endpoint_semantics": semantics,
        "output_unit": unit,
        "result": result_value,
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


def _is_read_like_tool(name: str) -> bool:
    lowered = name.lower()
    return lowered == RECOVERY_TOOL_NAME or lowered.startswith(READ_TOOL_PREFIXES)


def _has_reconciliation_read(
    tools: list[Any], arguments: dict[str, Any],
) -> bool:
    argument_keys = {str(key).lower() for key in arguments}
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            schema = tool.get("inputSchema") or tool.get("input_schema")
        else:
            name = getattr(tool, "name", None)
            schema = (
                getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
            )
        if not isinstance(name, str) or not _is_read_like_tool(name):
            continue
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(properties, dict):
            readable_keys = {str(key).lower() for key in properties}
            if argument_keys & readable_keys:
                return True
    return False


def _tool_result_is_error_like(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered == "iserror" and nested is True:
                return True
            if lowered in {"error", "errors"} and nested not in (None, False, "", [], {}):
                return True
            if lowered == "success" and nested is False:
                return True
            if lowered == "status" and isinstance(nested, str):
                if nested.lower() in {"error", "failed", "failure", "invalid"}:
                    return True
        return any(_tool_result_is_error_like(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return any(_tool_result_is_error_like(item) for item in value)
    if not isinstance(value, str):
        return False

    text = value.strip()
    if not text:
        return False
    if text[:1] in "[{":
        try:
            return _tool_result_is_error_like(json.loads(text))
        except (TypeError, ValueError):
            pass
    lowered = text.lower()
    if lowered.startswith((
        "validation error", "validation errors", "invalid ", "error:",
        "error -", "failed", "failure:", "unable to ", "cannot ", "can't ",
    )):
        return True
    first_line = lowered.splitlines()[0]
    return (
        (" validation error" in first_line and first_line[:1].isdigit())
        or "input should be " in first_line
    )


def _normalize_tool_result(output: Any) -> tuple[str, bool]:
    content = json.dumps(output, ensure_ascii=False, default=str)
    return content, _tool_result_is_error_like(output)


def _argument_anchors(arguments: dict[str, Any]) -> set[str]:
    anchors: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                collect(nested)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            normalized = str(value).strip().lower()
            if len(normalized) >= 4:
                anchors.add(normalized)

    collect(arguments)
    return anchors


def _reconciliation_required(
    tool_name: str, pending: dict[str, Any],
) -> str:
    return json.dumps({
        "reconciliation_required": True,
        "pending_write": {
            "tool": tool_name,
            "identifier_anchors": sorted(pending["anchors"]),
        },
        "message": (
            "The previous call to this mutating tool returned an error-like result and may "
            "have persisted state. Inspect the relevant persisted state with a read-like "
            "external tool before retrying. If the resource exists, patch or delete that "
            "result when possible; retry creation only after the read establishes absence."
        ),
    }, ensure_ascii=False)


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
        pending_writes: dict[str, dict[str, Any]] = {}
        external_mcp_called = False
        finalization_audit_run = False
        try:
            if task.action_type != "mcp" or not task.mcp_servers:
                raise ValueError("this agent requires an EOG MCP task")
            mcp = task.mcp_session(task.mcp_servers[0])
            mcp_tools = mcp.list_tools()
            if not mcp_tools:
                raise RuntimeError("task exposed no MCP tools")
            collisions = LOCAL_TOOL_NAMES & _tool_names(mcp_tools)
            if collisions:
                raise ValueError(f"MCP tool name {sorted(collisions)[0]!r} is reserved")
            tools = _to_openai_tools(mcp_tools)
            collisions = LOCAL_TOOL_NAMES & _tool_names(tools)
            if collisions:
                raise ValueError(f"MCP tool name {sorted(collisions)[0]!r} is reserved")
            tools.extend((_recovery_tool(), _time_interval_tool()))
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
                    elif call.function.name == TIME_INTERVAL_TOOL_NAME:
                        content, calculation = _calculate_time_interval(arguments)
                        tool_event.update({
                            "content": content,
                            "truncated": False,
                            "local": True,
                            "calculation": calculation,
                        })
                    else:
                        name = call.function.name
                        pending = pending_writes.get(name)
                        if pending is not None:
                            content = _reconciliation_required(name, pending)
                            tool_event.update({
                                "content": content,
                                "truncated": False,
                                "local": True,
                                "reconciliation": {"required": True},
                            })
                        else:
                            output = mcp.call_tool(name, arguments)
                            external_mcp_called = True
                            full_content, error_like = _normalize_tool_result(output)
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
                            read_like = _is_read_like_tool(name)
                            anchors = _argument_anchors(arguments)
                            if read_like and not error_like and anchors:
                                cleared_writes = []
                                for write_name, write in list(pending_writes.items()):
                                    if anchors & write["anchors"]:
                                        del pending_writes[write_name]
                                        cleared_writes.append(write_name)
                                if cleared_writes:
                                    tool_event["reconciliation"] = {
                                        "cleared_writes": cleared_writes,
                                    }
                            elif (
                                not read_like
                                and error_like
                                and _has_reconciliation_read(mcp_tools, arguments)
                                and anchors
                            ):
                                pending_writes[name] = {
                                    "arguments": arguments,
                                    "anchors": anchors,
                                }
                            tool_event.update({
                                "content": content,
                                "truncated": truncated,
                                "local": False,
                                "error_like": error_like,
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
            pending_writes.clear()
            result.latency_s = time.monotonic() - start
        return result
