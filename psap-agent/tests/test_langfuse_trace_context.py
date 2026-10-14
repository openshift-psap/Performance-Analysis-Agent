"""Tests for explicit request-to-Langfuse trace correlation."""

from unittest.mock import Mock

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
