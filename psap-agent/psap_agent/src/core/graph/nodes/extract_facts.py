"""Fact extraction node — calls the smaller model to extract structured facts
and (for config-level) red flags from the analysis report.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.prompts.extraction import (
    FACT_EXTRACTION_SYSTEM,
    build_extraction_user_prompt,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _normalize_content(content: str | list) -> str:
    """Gemini 3.x can return content as a list of parts; join to str."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return "".join(parts)
    return content


def _parse_json(text: str | list) -> dict[str, Any]:
    """Best-effort extraction of JSON from an LLM response."""
    text = _normalize_content(text).strip()
    # Strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("extract_facts: failed to parse LLM response as JSON")
        return {"facts": [], "red_flags": []}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def extract_facts(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Extract structured facts (and red flags) from the report."""
    fact_model = config["configurable"]["fact_model"]

    user_prompt = build_extraction_user_prompt(
        report=state.get("report", ""),
        analysis_level=state.get("analysis_level", "config"),
        rhaiis_version=state.get("rhaiis_version", "?"),
        baseline_version=state.get("baseline_version", "?"),
        config=state.get("config"),
    )
    response = await fact_model.ainvoke(
        [
            SystemMessage(content=FACT_EXTRACTION_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    parsed = _parse_json(response.content)
    facts: list[dict] = parsed.get("facts", [])
    red_flags: list[str] = parsed.get("red_flags", [])

    logger.info(
        "extract_facts: extracted %d facts, %d red flags",
        len(facts),
        len(red_flags),
    )

    return {
        "extracted_facts": facts,
        "new_red_flags": red_flags,
    }
