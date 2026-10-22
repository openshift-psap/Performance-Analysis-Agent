"""Keep Langfuse useful without recording whole LangGraph state at every node."""

import json
from typing import Any

from langfuse.langchain import CallbackHandler

_TOOL_INPUT_LIMIT = 3_000
_TOOL_OUTPUT_LIMIT = 12_000


def _chain_summary(value: Any) -> dict[str, Any]:
    """Chain nodes repeat state; generations and tools hold the useful detail."""
    summary: dict[str, Any] = {"state_omitted": True}
    if isinstance(value, dict):
        summary["keys"] = [str(key) for key in list(value)[:12]]
        messages = value.get("messages")
        if isinstance(messages, (list, tuple)):
            summary["message_count"] = len(messages)
    elif value is not None:
        summary["value_type"] = type(value).__name__
    return summary


def _bounded_value(value: Any, limit: int) -> Any:
    """Shorten a trace-only copy, leaving the result returned to the agent intact."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) <= limit:
        return value
    head = (limit - 200) * 3 // 4
    tail = limit - 200 - head
    return {
        "truncated_for_trace": True,
        "original_chars": len(text),
        "head": text[:head],
        "tail": text[-tail:],
    }


def _tool_input(value: Any) -> Any:
    """Strip injected ToolRuntime, whose state contains the entire chat history."""
    if isinstance(value, str):
        marker = "'runtime': ToolRuntime("
        marker_pos = value.find(marker)
        if marker_pos >= 0:
            return {
                "args_preview": value[:marker_pos].rstrip(" ,")[:_TOOL_INPUT_LIMIT],
                "runtime_state_omitted": True,
            }
    return _bounded_value(value, _TOOL_INPUT_LIMIT)


def _compact_inputs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "inputs" not in kwargs:
        return kwargs
    return {**kwargs, "inputs": _tool_input(kwargs["inputs"])}


class CompactLangfuseCallbackHandler(CallbackHandler):
    """Retain model/tool spans and nesting, but omit redundant state snapshots."""

    def on_chain_start(self, serialized, inputs, **kwargs):
        return super().on_chain_start(serialized, _chain_summary(inputs), **kwargs)

    def on_chain_end(self, outputs, **kwargs):
        compact = {**kwargs}
        if "inputs" in compact:
            compact["inputs"] = _chain_summary(compact["inputs"])
        return super().on_chain_end(_chain_summary(outputs), **compact)

    def on_chain_error(self, error, **kwargs):
        compact = {**kwargs}
        if "inputs" in compact:
            compact["inputs"] = _chain_summary(compact["inputs"])
        return super().on_chain_error(error, **compact)

    def on_tool_start(self, serialized, input_str, **kwargs):
        return super().on_tool_start(
            serialized, _tool_input(input_str), **_compact_inputs(kwargs)
        )

    def on_tool_end(self, output, **kwargs):
        return super().on_tool_end(
            _bounded_value(output, _TOOL_OUTPUT_LIMIT), **_compact_inputs(kwargs)
        )

    def on_tool_error(self, error, **kwargs):
        return super().on_tool_error(error, **_compact_inputs(kwargs))
