"""Analyze route for the automated analysis pipeline.

Provides POST /v1/analyze (submit job) and GET /v1/analyze/{job_id}/status
(poll status). Jobs run as asyncio background tasks with in-memory tracking.

Compiled LangGraph StateGraphs for each analysis level:
  - config        → config_graph        (inject_memory → react_agent → extract → judge → reconcile → store → red_flags)
  - model         → model_graph         (inject_facts → synthesize → extract → judge → reconcile → store)
  - version       → version_graph       (inject_facts → synthesize → extract → judge → reconcile → store)
  - consolidation → consolidation_graph (load_all_facts → consolidate → fold_red_flags → write_back → log_summary)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, status
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from psap_agent.src.core.graph.config_graph import build_config_graph
from psap_agent.src.core.graph.consolidation_graph import build_consolidation_graph
from psap_agent.src.core.graph.model_graph import build_model_graph
from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.core.graph.version_graph import build_version_graph
from psap_agent.src.core.memory.db import ensure_schema
from psap_agent.src.schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeStatusResponse,
)
from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

router = APIRouter(prefix="/v1", tags=["analyze"])


# ---------------------------------------------------------------------------
# Compiled graphs (built once, reused across jobs)
# ---------------------------------------------------------------------------

_config_graph = build_config_graph()
_model_graph = build_model_graph()
_version_graph = build_version_graph()
_consolidation_graph = build_consolidation_graph()


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------


@dataclass
class AnalyzeJob:
    job_id: str
    status: str  # accepted | running | completed | failed
    analysis_level: str
    request: AnalyzeRequest
    created_at: float
    completed_at: float | None = None
    report_preview: str | None = None
    facts_stored: int | None = None
    red_flags_added: int | None = None
    error: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


_jobs: dict[str, AnalyzeJob] = {}


# ---------------------------------------------------------------------------
# Model + tools helpers
# ---------------------------------------------------------------------------


def _create_main_model():
    """Create the main LLM (same model the interactive agent uses)."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=0.3,
    )


def _create_fact_model():
    """Create the smaller/cheaper LLM for fact extraction, judging, and reconciliation."""
    return ChatGoogleGenerativeAI(
        model=settings.FACT_MODEL,
        temperature=0.1,
    )


async def _load_mcp_tools() -> list:
    """Load MCP tools from the configured server."""
    try:
        client = MultiServerMCPClient(
            {
                "psap-mcp-server": {
                    "url": settings.MCP_SERVER_URL,
                    "transport": "streamable_http",
                },
            }
        )
        tools = await client.get_tools()
        logger.info("Loaded %d MCP tools for analysis pipeline", len(tools))
        return tools
    except Exception as e:
        logger.warning("Could not load MCP tools: %s", e)
        return []


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------


def _build_initial_state(request: AnalyzeRequest) -> AnalysisState:
    """Convert an AnalyzeRequest into the initial AnalysisState."""
    state: AnalysisState = {
        "analysis_level": request.analysis_level,
        "run_id": request.run_id,
        "rhaiis_version": request.rhaiis_version,
        "baseline_version": request.baseline_version,
    }

    if request.mode is not None:
        state["mode"] = request.mode
    if request.regression_detected is not None:
        state["regression_detected"] = request.regression_detected

    if request.config is not None:
        state["config"] = {
            "model": request.config.model,
            "accelerator": request.config.accelerator,
            "tp": request.config.tp,
            "prompt_toks": request.config.prompt_toks,
            "output_toks": request.config.output_toks,
            "concurrency": request.config.concurrency,
        }
    else:
        state["config"] = {}

    return state


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _run_job(job: AnalyzeJob) -> None:
    """Execute an analysis job using the appropriate graph."""
    job.status = "running"
    level = job.analysis_level
    logger.info(
        "Job %s running: level=%s, version=%s",
        job.job_id,
        level,
        job.request.rhaiis_version,
    )

    try:
        # Ensure memory schema exists
        await ensure_schema()

        # Prepare models
        main_model = _create_main_model()
        fact_model = _create_fact_model()

        configurable: dict = {
            "model": main_model,
            "fact_model": fact_model,
        }

        # Build initial state from request
        initial_state = _build_initial_state(job.request)

        # Select and invoke the right graph
        if level == "config":
            tools = await _load_mcp_tools()
            configurable["tools"] = tools
            result = await _config_graph.ainvoke(
                initial_state,
                config={"configurable": configurable},
            )
        elif level == "model":
            result = await _model_graph.ainvoke(
                initial_state,
                config={"configurable": configurable},
            )
        elif level == "version":
            result = await _version_graph.ainvoke(
                initial_state,
                config={"configurable": configurable},
            )
        elif level == "consolidation":
            result = await _consolidation_graph.ainvoke(
                initial_state,
                config={"configurable": configurable},
            )
        else:
            raise ValueError(f"Unknown analysis level: {level}")

        # Extract results
        report = result.get("report", "")
        facts_stored = result.get("facts_stored_count", 0)
        red_flags_added = result.get("red_flags_added_count", 0)

        job.status = "completed"
        job.facts_stored = facts_stored
        job.red_flags_added = red_flags_added
        job.report_preview = report[:500] if report else None

        logger.info(
            "Job %s completed: %d facts stored, %d red flags added",
            job.job_id,
            facts_stored,
            red_flags_added,
        )
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.exception("Job %s failed: %s", job.job_id, e)
    finally:
        job.completed_at = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_analysis(request: AnalyzeRequest) -> AnalyzeResponse:
    """Submit an analysis job for background execution."""
    job_id = f"job-{uuid.uuid4().hex[:12]}"

    job = AnalyzeJob(
        job_id=job_id,
        status="accepted",
        analysis_level=request.analysis_level,
        request=request,
        created_at=time.time(),
    )
    _jobs[job_id] = job

    job.task = asyncio.create_task(_run_job(job))

    logger.info(
        "Accepted job %s: level=%s, version=%s",
        job_id,
        request.analysis_level,
        request.rhaiis_version,
    )

    return AnalyzeResponse(
        job_id=job_id,
        analysis_level=request.analysis_level,
    )


@router.get(
    "/analyze/{job_id}/status",
    response_model=AnalyzeStatusResponse,
)
async def get_analysis_status(job_id: str) -> AnalyzeStatusResponse:
    """Poll the status of a submitted analysis job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )

    duration = None
    if job.completed_at:
        duration = round(job.completed_at - job.created_at, 2)
    elif job.status == "running":
        duration = round(time.time() - job.created_at, 2)

    return AnalyzeStatusResponse(
        job_id=job.job_id,
        status=job.status,
        analysis_level=job.analysis_level,
        report_preview=job.report_preview,
        facts_stored=job.facts_stored,
        red_flags_added=job.red_flags_added,
        error=job.error,
        duration_seconds=duration,
    )
