"""Tests for explicit request-to-Langfuse trace correlation."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, Mock

import pytest

from psap_agent.src.core import manager
from psap_agent.src.schema import ChatMessage

TRACE_ID = "0123456789abcdef0123456789abcdef"


def test_run_context_uses_generated_trace_id_for_callback(monkeypatch):
    create_trace_id = Mock(return_value=TRACE_ID)
    monkeypatch.setattr(
        manager.Langfuse,
        "create_trace_id",
        staticmethod(create_trace_id),
    )
    callback_handler = Mock()
    monkeypatch.setattr(manager, "CallbackHandler", callback_handler)

    trace_id, handler = manager._langfuse_context_for_run("graph-run-123")

    assert trace_id == TRACE_ID
    assert handler is callback_handler.return_value
    create_trace_id.assert_called_once_with(seed="graph-run-123")
    callback_handler.assert_called_once_with(trace_context={"trace_id": TRACE_ID})


def test_root_observation_records_request_and_trace_attributes(monkeypatch):
    observation = MagicMock()
    observation_context = MagicMock()
    observation_context.__enter__.return_value = observation
    client = MagicMock()
    client.start_as_current_observation.return_value = observation_context
    monkeypatch.setattr(manager, "get_client", lambda: client)
    request = manager.StreamRequest(
        message="compare these runs",
        model="gemini-3.8-flash",
        reasoning_effort="medium",
    )

    context, root = manager._start_langfuse_root_observation(
        TRACE_ID, request, "user-1", "session-1"
    )

    assert context is observation_context
    assert root is observation
    client.start_as_current_observation.assert_called_once_with(
        as_type="span",
        name="psap-agent-request",
        input={"query": "compare these runs"},
        metadata={
            "model": "gemini-3.8-flash",
            "reasoning_effort": "medium",
        },
        trace_context={"trace_id": TRACE_ID},
    )
    observation.update_trace.assert_called_once_with(
        input={"query": "compare these runs"},
        user_id="user-1",
        session_id="session-1",
    )


def test_langchain_callback_inherits_root_trace_without_forking_it(monkeypatch):
    class FakeCallbackHandler:
        pass

    monkeypatch.setattr(manager, "CallbackHandler", FakeCallbackHandler)
    old_handler = FakeCallbackHandler()
    other_callback = Mock()
    config = {"callbacks": [old_handler, other_callback]}

    manager._bind_langchain_callbacks_to_current_trace(config)

    assert config["callbacks"][0] is other_callback
    assert isinstance(config["callbacks"][1], FakeCallbackHandler)
    assert config["callbacks"][1] is not old_handler


def test_trace_output_uses_last_assistant_message():
    output = manager._trace_output_from_events(
        [
            {
                "type": "message",
                "content": {"type": "ai", "content": "First draft"},
            },
            {
                "type": "message",
                "content": {"type": "tool", "content": "Tool result"},
            },
            {
                "type": "message",
                "content": {"type": "ai", "content": "Final answer"},
            },
        ]
    )

    assert output == {"response": "Final answer"}


def test_trace_output_has_fallback_when_no_assistant_message():
    assert manager._trace_output_from_events([]) == {"status": "no_response"}


def test_trace_output_reconstructs_streamed_tokens_as_fallback():
    assert manager._trace_output_from_events(
        [
            {"type": "token", "content": "Final "},
            {"type": "token", "content": "answer"},
        ]
    ) == {"response": "Final answer"}


def test_root_output_updates_observation_and_trace():
    observation = MagicMock()
    output = {"response": "Final answer"}

    manager._record_langfuse_root_output(observation, output)

    observation.update.assert_called_once_with(output=output)
    observation.update_trace.assert_called_once_with(output=output)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_stream_response_records_root_output_on_success_and_failure(
    monkeypatch, fails
):
    observation = MagicMock()
    observation_context = MagicMock()
    request_trace = Mock(return_value=(observation_context, observation))
    callback_binding = Mock()
    monkeypatch.setattr(manager, "_start_langfuse_root_observation", request_trace)
    monkeypatch.setattr(
        manager, "_bind_langchain_callbacks_to_current_trace", callback_binding
    )
    monkeypatch.setattr(manager.settings, "ENABLE_REFLECTION", False)

    class FakeAgent:
        async def astream(self, **_kwargs):
            if fails:
                raise RuntimeError("simulated graph failure")
            yield (
                "updates",
                {"agent": {"messages": [manager.AIMessage(content="Final answer")]}},
            )

    @asynccontextmanager
    async def fake_get_psap_agent(*_args, **_kwargs):
        yield FakeAgent()

    async def fake_handle_input(_self, _request, _agent):
        return (
            {"input": {}, "config": {"callbacks": [object()]}},
            "run-1",
            "thread-1",
            "",
            TRACE_ID,
        )

    async def fake_evaluate_response(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(manager, "get_psap_agent", fake_get_psap_agent)
    monkeypatch.setattr(manager.AgentManager, "_handle_input", fake_handle_input)
    monkeypatch.setattr(
        manager.AgentManager,
        "_format_events",
        lambda *_args: [
            {"type": "message", "content": {"type": "ai", "content": "Final answer"}}
        ],
    )
    monkeypatch.setattr(manager, "evaluate_response", fake_evaluate_response)
    monkeypatch.setattr(manager, "get_memory_manager", lambda: MagicMock())
    from psap_agent.src.core import evaluator

    monkeypatch.setattr(evaluator, "should_store_memory", lambda _scores: False)

    events = [
        event
        async for event in manager.AgentManager().stream_response(
            manager.StreamRequest(message="query")
        )
    ]
    await asyncio.sleep(0)

    request_trace.assert_called_once_with(
        TRACE_ID, manager.StreamRequest(message="query"), "anonymous", "thread-1"
    )
    callback_binding.assert_called_once()
    observation_context.__exit__.assert_called_once_with(None, None, None)
    if fails:
        assert events[-1]["type"] == "error"
        observation.update.assert_any_call(
            output={"error": "Internal server error", "error_type": "agent_error"}
        )
        observation.update.assert_any_call(
            level="ERROR", status_message="simulated graph failure"
        )
    else:
        assert events[-1]["content"]["content"] == "Final answer"
        observation.update.assert_any_call(output={"response": "Final answer"})
        observation.update_trace.assert_any_call(output={"response": "Final answer"})


def test_streamed_ai_message_exposes_its_langfuse_trace_id():
    message = ChatMessage(
        type="ai",
        content="answer",
        run_id="graph-run-123",
        langfuse_trace_id=TRACE_ID,
    )

    content = manager.AgentManager()._convert_chat_message_to_simple_format(
        message, thread_id="thread-1", session_id="session-1"
    )

    assert content["run_id"] == "graph-run-123"
    assert content["langfuse_trace_id"] == TRACE_ID
