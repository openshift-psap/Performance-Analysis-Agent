"""Feedback route for the PSAP agent API.

This module provides endpoints for recording user feedback on agent responses
using Langfuse for analytics and monitoring purposes.
"""

import sys
import time

from fastapi import APIRouter
from langfuse import get_client

from psap_agent.src.schema import FeedbackRequest, FeedbackResponse

router = APIRouter()


def _log(msg: str) -> None:
    """Print message to stderr so it appears in container logs."""
    print(f"[FEEDBACK] {msg}", file=sys.stderr, flush=True)


def _resolve_trace_id(client, run_id: str, retries: int = 3, delay: float = 2.0) -> str | None:
    """Resolve a LangGraph run_id to the actual Langfuse trace_id.

    In Langfuse v3, the CallbackHandler generates OpenTelemetry trace IDs
    that differ from the LangGraph run_id. The run_id is stored in the
    trace metadata. We look it up via the API and return the real trace_id.

    Because the trace may not be fully ingested when feedback arrives,
    this retries up to ``retries`` times with a ``delay`` second pause
    between attempts.
    """
    for attempt in range(1, retries + 1):
        try:
            traces = client.api.trace.list(limit=30)
            for trace in traces.data:
                metadata = trace.metadata or {}
                if metadata.get("run_id") == run_id:
                    return trace.id
        except Exception as e:
            _log(f"trace lookup attempt {attempt} failed: {e}")

        if attempt < retries:
            _log(f"run_id={run_id} not found on attempt {attempt}, retrying in {delay}s...")
            time.sleep(delay)

    return None


@router.post("/v1/feedback")
async def feedback(feedback: FeedbackRequest) -> FeedbackResponse:
    """Record feedback for a specific agent run to Langfuse.

    This endpoint serves as a wrapper for the Langfuse create_feedback API,
    allowing credentials to be stored and managed in the service rather than
    requiring client-side credential management.

    In Langfuse v3, the trace_id is an OTel-generated hex ID that differs
    from the LangGraph run_id. This endpoint resolves the run_id to the
    actual trace_id before creating the score.

    Args:
        feedback: The feedback request containing run_id, key, score, and
            optional kwargs for additional metadata.

    Returns:
        A FeedbackResponse indicating successful feedback recording.
    """
    kwargs = feedback.kwargs or {}

    # Get Langfuse client (v3 singleton pattern)
    client = get_client()

    # Build a comment string from any extra metadata (e.g. user_query,
    # assistant_response) since create_score() only accepts specific
    # keyword arguments -- arbitrary kwargs would cause a TypeError.
    comment_parts = []
    for k, v in kwargs.items():
        comment_parts.append(f"{k}: {v}")
    comment = "\n".join(comment_parts) if comment_parts else None

    # Resolve the LangGraph run_id to the actual Langfuse trace_id.
    # In v3, these are different (OTel hex IDs vs UUID).
    trace_id = _resolve_trace_id(client, feedback.run_id)
    if trace_id:
        _log(f"Resolved run_id={feedback.run_id} -> trace_id={trace_id}")
    else:
        _log(f"Could not resolve run_id={feedback.run_id}, using as-is")
        trace_id = feedback.run_id

    client.create_score(
        trace_id=trace_id,
        name=feedback.key,
        value=feedback.score,
        comment=comment,
    )
    client.flush()

    _log(f"Score submitted: trace={trace_id}, key={feedback.key}, score={feedback.score}")
    return FeedbackResponse()
