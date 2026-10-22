"""Tracing must not multiply full graph state into every observation."""

from langfuse.langchain import CallbackHandler

from psap_agent.src.core.trace_callback import (
    CompactLangfuseCallbackHandler,
    _bounded_value,
    _chain_summary,
    _tool_input,
)


def test_chain_state_is_summarized_without_mutating_it():
    state = {"messages": ["x" * 100_000], "remaining_steps": 10}
    summary = _chain_summary(state)

    assert summary == {
        "state_omitted": True,
        "keys": ["messages", "remaining_steps"],
        "message_count": 1,
    }
    assert len(state["messages"][0]) == 100_000


def test_tool_runtime_history_is_removed_from_trace_input():
    original = (
        "{'model': 'm', 'runtime': ToolRuntime(state={'messages': '" + "x" * 100_000
    )
    compact = _tool_input(original)

    assert compact["runtime_state_omitted"] is True
    assert "'model': 'm'" in compact["args_preview"]
    assert len(str(compact)) < 3_100


def test_large_tool_output_is_shortened_only_for_trace():
    output = {"data": "x" * 100_000}
    compact = _bounded_value(output, 12_000)

    assert compact["truncated_for_trace"] is True
    assert len(str(compact)) < 12_100
    assert len(output["data"]) == 100_000


def test_callback_forwards_compact_chain_and_tool_data(monkeypatch):
    calls = []
    monkeypatch.setattr(
        CallbackHandler,
        "on_chain_start",
        lambda self, serialized, inputs, **kwargs: calls.append(("chain", inputs)),
    )
    monkeypatch.setattr(
        CallbackHandler,
        "on_tool_start",
        lambda self, serialized, input_str, **kwargs: calls.append(("tool", input_str)),
    )
    monkeypatch.setattr(
        CallbackHandler,
        "on_chain_end",
        lambda self, outputs, **kwargs: calls.append(("chain_end", outputs, kwargs)),
    )
    monkeypatch.setattr(
        CallbackHandler,
        "on_tool_end",
        lambda self, output, **kwargs: calls.append(("tool_end", output, kwargs)),
    )
    callback = object.__new__(CompactLangfuseCallbackHandler)

    callback.on_chain_start({}, {"messages": ["x" * 100_000]}, run_id="chain")
    callback.on_tool_start(
        {},
        "{'runtime': ToolRuntime(state='" + "x" * 100_000,
        run_id="tool",
    )
    callback.on_chain_end(
        {"messages": ["x" * 100_000]},
        run_id="chain",
        inputs={"messages": ["x" * 100_000]},
    )
    callback.on_tool_end(
        {"data": "x" * 100_000},
        run_id="tool",
        inputs="{'runtime': ToolRuntime(state='" + "x" * 100_000,
    )

    assert calls[0][1]["state_omitted"] is True
    assert calls[1][1]["runtime_state_omitted"] is True
    assert calls[2][1]["state_omitted"] is True
    assert calls[2][2]["inputs"]["state_omitted"] is True
    assert calls[3][1]["truncated_for_trace"] is True
    assert calls[3][2]["inputs"]["runtime_state_omitted"] is True
