"""Regression tests for protecting Langfuse score trace associations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from psap_agent.src.core.evaluator import _is_valid_langfuse_trace_id, _run_single_eval


def test_langfuse_trace_id_validation_requires_32_lowercase_hex_characters():
    assert _is_valid_langfuse_trace_id("0123456789abcdef0123456789abcdef")
    assert not _is_valid_langfuse_trace_id("47a8627f-aa6d-4690-80ee-c7f012ffcbc2")
    assert not _is_valid_langfuse_trace_id("0123456789ABCDEF0123456789ABCDEF")
    assert not _is_valid_langfuse_trace_id("not-a-trace-id")


@pytest.mark.asyncio
async def test_invalid_trace_id_skips_judge_and_never_submits_score():
    with (
        patch("psap_agent.src.core.evaluator.create_chat_model") as create_model,
        patch("psap_agent.src.core.evaluator.get_client") as get_client,
    ):
        result = await _run_single_eval(
            criterion_name="correctness",
            criterion={},
            user_query="question",
            agent_response="answer",
            tool_calls_summary="No tool calls made.",
            trace_id="47a8627f-aa6d-4690-80ee-c7f012ffcbc2",
        )

    assert result == ("correctness", None)
    create_model.assert_not_called()
    get_client.assert_not_called()


@pytest.mark.asyncio
async def test_judge_score_uses_explicit_langfuse_trace_id():
    judge_model = Mock()
    judge_model.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content='{"score": 0.8, "reasoning": "supported"}',
            usage_metadata=None,
        )
    )
    langfuse_client = Mock()
    with (
        patch(
            "psap_agent.src.core.evaluator.create_chat_model",
            return_value=judge_model,
        ),
        patch(
            "psap_agent.src.core.evaluator.get_client",
            return_value=langfuse_client,
        ),
    ):
        result = await _run_single_eval(
            criterion_name="correctness",
            criterion={"description": "accuracy", "rubric": "score accuracy"},
            user_query="question",
            agent_response="answer",
            tool_calls_summary="No tool calls made.",
            trace_id="0123456789abcdef0123456789abcdef",
        )

    assert result == ("correctness", 0.8)
    langfuse_client.create_score.assert_called_once()
    assert (
        langfuse_client.create_score.call_args.kwargs["trace_id"]
        == "0123456789abcdef0123456789abcdef"
    )
