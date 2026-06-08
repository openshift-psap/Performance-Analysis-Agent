"""Reconciliation node — merges validated facts with existing memory.

Handles contradictions (Dreams-like):
  - insert:  no prior fact exists; store as new.
  - update:  prior fact exists with new info; produce merged narrative.
  - noop:    prior fact exists, no meaningful change; skip.

Also performs cross-version baseline annotation: when analyzing version B
against baseline A, it updates A's facts to record whether each issue
persists, is resolved, or improved in B.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.core.memory.queries import (
    get_config_facts,
    get_model_summary,
    get_version_summary,
)
from psap_agent.src.prompts.reconciliation import (
    BASELINE_ANNOTATION_SYSTEM,
    FACT_RECONCILIATION_SYSTEM,
    build_baseline_annotation_prompt,
    build_reconciliation_user_prompt,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


async def _get_existing_facts(state: AnalysisState) -> list[dict]:
    """Fetch existing facts from the appropriate table for reconciliation."""
    level = state.get("analysis_level", "config")
    cfg = state.get("config", {})

    if level == "config":
        profile = f"{cfg.get('prompt_toks', 0)}/{cfg.get('output_toks', 0)}"
        return await get_config_facts(
            rhaiis_version=state["rhaiis_version"],
            model_name=cfg.get("model", ""),
            accelerator=cfg.get("accelerator"),
            profile=profile,
            limit=50,
        )
    elif level == "model":
        return await get_model_summary(
            rhaiis_version=state["rhaiis_version"],
            model_name=cfg.get("model", ""),
            limit=10,
        )
    elif level == "version":
        return await get_version_summary(
            rhaiis_version=state["rhaiis_version"],
            limit=10,
        )
    return []


async def _get_baseline_facts(state: AnalysisState) -> list[dict]:
    """Fetch facts from the baseline version for cross-version annotation."""
    baseline_version = state.get("baseline_version", "")
    if not baseline_version:
        return []

    level = state.get("analysis_level", "config")
    cfg = state.get("config", {})

    if level == "config":
        profile = f"{cfg.get('prompt_toks', 0)}/{cfg.get('output_toks', 0)}"
        return await get_config_facts(
            rhaiis_version=baseline_version,
            model_name=cfg.get("model", ""),
            accelerator=cfg.get("accelerator"),
            profile=profile,
            limit=50,
        )
    elif level == "model":
        return await get_model_summary(
            rhaiis_version=baseline_version,
            model_name=cfg.get("model", ""),
            limit=10,
        )
    elif level == "version":
        return await get_version_summary(
            rhaiis_version=baseline_version,
            limit=10,
        )
    return []


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


def _parse_reconciled(text: str | list) -> list[dict]:
    text = _normalize_content(text).strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        logger.warning("reconcile_facts: failed to parse reconciliation JSON")
    return []


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def reconcile_facts(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Reconcile validated facts against existing memory.

    Two phases:
      1. Standard reconciliation — merge new facts with current version's memory.
      2. Baseline annotation — update the baseline version's facts with
         cross-version status notes (persists / resolved / improved).
    """
    fact_model = config["configurable"]["fact_model"]
    validated = state.get("validated_facts", [])

    if not validated:
        return {"reconciled_facts": []}

    existing = await _get_existing_facts(state)

    user_prompt = build_reconciliation_user_prompt(validated, existing)
    response = await fact_model.ainvoke(
        [
            SystemMessage(content=FACT_RECONCILIATION_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    decisions = _parse_reconciled(response.content)

    # Build reconciled fact list (skip noops)
    cfg = state.get("config", {})
    profile = f"{cfg.get('prompt_toks', 0)}/{cfg.get('output_toks', 0)}"
    reconciled: list[dict] = []

    for d in decisions:
        idx = d.get("fact_index", -1)
        action = d.get("action", "noop")
        if action == "noop":
            continue
        if idx < 0 or idx >= len(validated):
            continue

        original = validated[idx]
        reconciled.append(
            {
                "action": action,
                "fact_text": d.get("reconciled_text", original.get("fact_text", "")),
                "category": original.get("category", "metrics_comparison"),
                "confidence": original.get("confidence", "high"),
                "root_cause": original.get("root_cause"),
                "related_prs": original.get("related_prs"),
                "related_kernels": original.get("related_kernels"),
                "status": d.get("status", "active"),
                "existing_id": d.get("existing_id"),
                # Carry context for store_facts
                "rhaiis_version": state.get("rhaiis_version", ""),
                "baseline_version": state.get("baseline_version", ""),
                "model_name": cfg.get("model", ""),
                "accelerator": cfg.get("accelerator", ""),
                "profile": profile,
                "run_id": state.get("run_id"),
            }
        )

    logger.info(
        "reconcile_facts: %d inserts/updates out of %d validated facts",
        len(reconciled),
        len(validated),
    )

    # ------------------------------------------------------------------
    # Phase 2: Cross-version baseline annotation
    # ------------------------------------------------------------------
    baseline_version = state.get("baseline_version", "")
    current_version = state.get("rhaiis_version", "")

    if baseline_version and baseline_version != current_version:
        baseline_facts = await _get_baseline_facts(state)

        if baseline_facts:
            report_summary = state.get("report", "")[:3000]

            annotation_prompt = build_baseline_annotation_prompt(
                current_version=current_version,
                baseline_version=baseline_version,
                baseline_facts=baseline_facts,
                report_summary=report_summary,
            )

            annotation_response = await fact_model.ainvoke(
                [
                    SystemMessage(content=BASELINE_ANNOTATION_SYSTEM),
                    HumanMessage(content=annotation_prompt),
                ]
            )

            annotations = _parse_reconciled(annotation_response.content)

            baseline_update_count = 0
            for ann in annotations:
                idx = ann.get("baseline_fact_index", -1)
                action = ann.get("action", "noop")
                if action == "noop":
                    continue
                if idx < 0 or idx >= len(baseline_facts):
                    continue

                original_baseline = baseline_facts[idx]
                reconciled.append(
                    {
                        "action": "update",
                        "fact_text": ann.get(
                            "updated_text",
                            original_baseline.get("fact_text", ""),
                        ),
                        "category": original_baseline.get(
                            "category", "metrics_comparison"
                        ),
                        "confidence": original_baseline.get("confidence", "high"),
                        "root_cause": original_baseline.get("root_cause"),
                        "related_prs": original_baseline.get("related_prs"),
                        "related_kernels": original_baseline.get("related_kernels"),
                        "status": ann.get("status", "active"),
                        "existing_id": ann.get(
                            "existing_id", original_baseline.get("id")
                        ),
                        # Route to the BASELINE version's row in the DB
                        "rhaiis_version": baseline_version,
                        "baseline_version": original_baseline.get(
                            "baseline_version", ""
                        ),
                        "model_name": cfg.get("model", ""),
                        "accelerator": cfg.get("accelerator", ""),
                        "profile": profile,
                        "run_id": state.get("run_id"),
                    }
                )
                baseline_update_count += 1

            logger.info(
                "reconcile_facts: annotated %d baseline facts from %s with "
                "status in %s",
                baseline_update_count,
                baseline_version,
                current_version,
            )

    return {"reconciled_facts": reconciled}
