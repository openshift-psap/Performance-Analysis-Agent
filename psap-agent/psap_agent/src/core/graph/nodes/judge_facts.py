"""LLM-as-judge node — validates each extracted fact against the report.

Drops facts the judge deems invalid.  Verdicts are logged for audit.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.prompts.judge import (
    FACT_VALIDATION_SYSTEM,
    build_judge_user_prompt,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


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


def _parse_verdicts(text: str | list) -> list[dict]:
    text = _normalize_content(text).strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        logger.warning("judge_facts: failed to parse verdicts JSON")
    return []


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def judge_facts(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Validate extracted facts against the report; drop invalid ones."""
    fact_model = config["configurable"]["fact_model"]
    facts = state.get("extracted_facts", [])
    report = state.get("report", "")

    if not facts:
        return {"validated_facts": []}

    user_prompt = build_judge_user_prompt(facts, report)
    response = await fact_model.ainvoke(
        [
            SystemMessage(content=FACT_VALIDATION_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    verdicts = _parse_verdicts(response.content)

    # Build index → verdict map
    verdict_map: dict[int, dict] = {}
    for v in verdicts:
        idx = v.get("fact_index")
        if idx is not None:
            verdict_map[int(idx)] = v

    validated: list[dict] = []
    for i, fact in enumerate(facts):
        v = verdict_map.get(i)
        if v and v.get("verdict") == "invalid":
            logger.info(
                "judge_facts: dropped fact %d — %s",
                i,
                v.get("reason", "no reason"),
            )
            continue
        validated.append(fact)

    logger.info(
        "judge_facts: %d/%d facts passed validation",
        len(validated),
        len(facts),
    )

    return {"validated_facts": validated}
