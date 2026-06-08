"""Storage node — writes reconciled facts and the report to the database.

Routes upserts to the correct table based on analysis_level.
Before inserting a new config-level fact, an LLM duplicate-check gate
compares the fact against existing DB entries to prevent semantic
duplicates (same finding, different wording).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.core.memory.queries import (
    get_config_facts,
    store_report,
    upsert_config_finding,
    upsert_model_summary,
    upsert_version_summary,
)
from psap_agent.src.prompts.dedup import (
    DEDUP_CHECK_SYSTEM,
    build_dedup_check_prompt,
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


def _parse_dedup_answer(text: str | list) -> tuple[bool, str | None]:
    """Parse the tag-based duplicate check response.

    Returns (is_duplicate, existing_id_or_none).
    """
    raw = _normalize_content(text).strip()
    answer_match = re.search(
        r"\[START_ANSWER\]\s*(.*?)\s*\[END_ANSWER\]", raw
    )
    if not answer_match:
        return False, None

    answer = answer_match.group(1).strip().lower()
    is_dup = answer == "duplicate"

    existing_id = None
    id_match = re.search(
        r"\[START_EXISTING_ID\]\s*(.*?)\s*\[END_EXISTING_ID\]", raw
    )
    if id_match:
        existing_id = id_match.group(1).strip()

    return is_dup, existing_id


async def _is_semantic_duplicate(
    fact: dict,
    existing_facts: list[dict],
    fact_model: Any,
) -> tuple[bool, str | None]:
    """Call the LLM to check if a fact is a semantic duplicate."""
    if not existing_facts:
        return False, None

    prompt = build_dedup_check_prompt(fact, existing_facts)
    response = await fact_model.ainvoke(
        [
            SystemMessage(content=DEDUP_CHECK_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )
    return _parse_dedup_answer(response.content)


async def store_facts(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Persist reconciled facts and the analysis report.

    For config-level inserts, each new fact goes through an LLM
    duplicate-check gate before being written to the database.
    """
    facts = state.get("reconciled_facts", [])
    level = state.get("analysis_level", "config")
    cfg = state.get("config", {})
    fact_model = config["configurable"]["fact_model"]
    count = 0
    dedup_skipped = 0

    for fact in facts:
        try:
            if level == "config":
                action_type = fact.get("action", "insert")
                if action_type == "insert":
                    version = fact.get(
                        "rhaiis_version", state.get("rhaiis_version", "")
                    )
                    profile = fact.get("profile", "")
                    existing = await get_config_facts(
                        rhaiis_version=version,
                        model_name=fact.get("model_name", cfg.get("model", "")),
                        accelerator=fact.get("accelerator", cfg.get("accelerator")),
                        profile=profile,
                        limit=100,
                    )
                    is_dup, dup_id = await _is_semantic_duplicate(
                        fact, existing, fact_model
                    )
                    if is_dup:
                        logger.info(
                            "store_facts: skipped duplicate fact "
                            "(matches %s): %s",
                            dup_id,
                            fact.get("fact_text", "")[:80],
                        )
                        dedup_skipped += 1
                        continue

                result = await upsert_config_finding(fact)
            elif level == "model":
                result = await upsert_model_summary(
                    {
                        "summary_text": fact.get("fact_text", ""),
                        "rhaiis_version": fact.get("rhaiis_version", state.get("rhaiis_version", "")),
                        "baseline_version": fact.get("baseline_version", state.get("baseline_version", "")),
                        "model_name": fact.get("model_name", cfg.get("model", "")),
                        "model_family": fact.get("model_family"),
                        "confidence": fact.get("confidence", "high"),
                        "status": fact.get("status", "active"),
                        "issue_thread_id": fact.get("issue_thread_id"),
                        "run_id": fact.get("run_id", state.get("run_id")),
                    }
                )
            elif level == "version":
                result = await upsert_version_summary(
                    {
                        "summary_text": fact.get("fact_text", ""),
                        "rhaiis_version": fact.get("rhaiis_version", state.get("rhaiis_version", "")),
                        "baseline_version": fact.get("baseline_version", state.get("baseline_version", "")),
                        "confidence": fact.get("confidence", "high"),
                        "status": fact.get("status", "active"),
                        "run_id": fact.get("run_id", state.get("run_id")),
                    }
                )
            else:
                continue

            action = result.get("action", "skipped")
            if action in ("inserted", "updated"):
                count += 1
            logger.debug("store_facts: %s → %s", fact.get("category", "?"), action)
        except Exception:
            logger.exception("store_facts: failed to upsert fact")

    report_text = state.get("report", "")
    if report_text:
        profile = f"{cfg.get('prompt_toks', 0)}/{cfg.get('output_toks', 0)}"
        try:
            await store_report(
                {
                    "analysis_level": level,
                    "report_text": report_text,
                    "rhaiis_version": state.get("rhaiis_version", ""),
                    "baseline_version": state.get("baseline_version", ""),
                    "model_name": cfg.get("model"),
                    "accelerator": cfg.get("accelerator"),
                    "tp_config": cfg.get("tp"),
                    "profile": profile if level == "config" else None,
                    "run_id": state.get("run_id"),
                }
            )
        except Exception:
            logger.exception("store_facts: failed to store report")

    logger.info(
        "store_facts: stored %d facts, skipped %d duplicates for level=%s",
        count,
        dedup_skipped,
        level,
    )
    return {"facts_stored_count": count}
