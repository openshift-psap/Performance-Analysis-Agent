"""Feedback route for the PSAP agent API.

This module provides endpoints for recording user feedback on agent responses
using Langfuse for analytics and monitoring purposes.
"""

import asyncio
import re
import sys
import time

from fastapi import APIRouter, HTTPException
from langfuse import Langfuse, get_client

from psap_agent.src.schema import FeedbackRequest, FeedbackResponse

router = APIRouter()
_LANGFUSE_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _log(msg: str) -> None:
    """Print message to stderr so it appears in container logs."""
    print(f"[FEEDBACK] {msg}", file=sys.stderr, flush=True)


def _resolve_trace_id(client, run_id: str, retries: int = 3, delay: float = 2.0) -> str | None:
    """Resolve a LangGraph run_id to the actual Langfuse trace_id.

    Legacy clients send only the LangGraph run_id. Resolve it through trace
    metadata; callers must not use the run_id itself as a Langfuse trace ID.

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

    This endpoint serves as a wrapper for Langfuse score ingestion,
    allowing credentials to be stored and managed in the service rather than
    requiring client-side credential management.

    New clients should provide ``langfuse_trace_id`` from the streamed agent
    response. Legacy clients may provide only ``run_id``; that ID is looked up
    in Langfuse metadata and is never used directly as a score trace ID.

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

    trace_id = feedback.langfuse_trace_id
    if trace_id is not None:
        if not _LANGFUSE_TRACE_ID_RE.fullmatch(trace_id):
            raise HTTPException(
                status_code=422,
                detail="langfuse_trace_id must be 32 lowercase hexadecimal characters",
            )
        expected_trace_id = Langfuse.create_trace_id(seed=feedback.run_id)
        if trace_id != expected_trace_id:
            raise HTTPException(
                status_code=422,
                detail="langfuse_trace_id does not match the supplied run_id",
            )
        _log(f"Using explicit Langfuse trace ID for run_id={feedback.run_id}")
    else:
        trace_id = await asyncio.to_thread(_resolve_trace_id, client, feedback.run_id)
        if trace_id is None:
            _log(f"Could not resolve run_id={feedback.run_id}; feedback was not submitted")
            raise HTTPException(
                status_code=503,
                detail="Could not resolve the Langfuse trace for this response; retry feedback",
            )
        if not _LANGFUSE_TRACE_ID_RE.fullmatch(trace_id):
            _log(f"Resolved trace for run_id={feedback.run_id} had an invalid ID")
            raise HTTPException(
                status_code=503,
                detail="Resolved Langfuse trace ID is invalid; feedback was not submitted",
            )
        _log(f"Resolved legacy run_id={feedback.run_id} to a Langfuse trace")

    client.create_score(
        trace_id=trace_id,
        name=feedback.key,
        value=feedback.score,
        comment=comment,
    )
    client.flush()

    _log(f"Score submitted: trace={trace_id}, key={feedback.key}, score={feedback.score}")
    return FeedbackResponse()
