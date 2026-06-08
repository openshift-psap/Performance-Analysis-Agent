"""Memory injection nodes for each analysis level.

- inject_config_memory: queries prior config-level facts + red flags
- inject_model_facts: queries config findings for synthesis
- inject_version_facts: queries model summaries for synthesis
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.core.memory.injection import build_memory_context
from psap_agent.src.core.memory.queries import (
    get_config_facts,
    get_model_summary,
    get_version_summary,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# Config-level: prior facts from baseline + red flags from current run
# ---------------------------------------------------------------------------


async def inject_config_memory(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Query prior facts and red flags, build memory context for config analysis."""
    cfg = state.get("config", {})
    profile = f"{cfg.get('prompt_toks', 0)}/{cfg.get('output_toks', 0)}"

    red_flags_dir = config.get("configurable", {}).get("red_flags_dir")

    memory = await build_memory_context(
        rhaiis_version=state["rhaiis_version"],
        baseline_version=state["baseline_version"],
        model_name=cfg.get("model", ""),
        accelerator=cfg.get("accelerator"),
        profile=profile,
        red_flags_dir=red_flags_dir,
    )

    logger.info(
        "inject_config_memory: prior_facts=%d chars, red_flags=%d chars",
        len(memory.get("full_context", "")),
        len(memory.get("red_flags", "")),
    )

    return {
        "prior_facts": memory.get("full_context", ""),
        "red_flags": memory.get("red_flags", ""),
    }


# ---------------------------------------------------------------------------
# Model-level: all config findings for this model + prior model summary
# ---------------------------------------------------------------------------


def _format_facts_block(facts: list[dict], label: str) -> str:
    """Format a list of fact dicts into a readable text block."""
    if not facts:
        return ""
    lines = [f"## {label}\n"]
    for f in facts:
        text = f.get("fact_text", f.get("summary_text", ""))
        category = f.get("category", "")
        accel = f.get("accelerator", "")
        profile = f.get("profile", "")
        prefix_parts = [p for p in [category, accel, profile] if p]
        prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""
        lines.append(f"- {prefix}{text}")
    return "\n".join(lines)


async def inject_model_facts(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Gather all config-level findings for a model + prior model summary."""
    model_name = state["config"]["model"]

    current_facts = await get_config_facts(
        rhaiis_version=state["rhaiis_version"],
        model_name=model_name,
        limit=100,
    )

    prior_summaries = await get_model_summary(
        rhaiis_version=state["baseline_version"],
        model_name=model_name,
        limit=1,
    )

    sections: list[str] = []
    current_block = _format_facts_block(
        current_facts,
        f"Config-Level Findings for {model_name} ({state['rhaiis_version']})",
    )
    if current_block:
        sections.append(current_block)

    if prior_summaries:
        prior_text = prior_summaries[0].get("summary_text", "")
        if prior_text:
            sections.append(
                f"## Prior Model Summary ({state['baseline_version']})\n{prior_text}"
            )

    combined = "\n\n".join(sections)
    logger.info(
        "inject_model_facts: %d config facts, %d prior summaries, %d chars total",
        len(current_facts),
        len(prior_summaries),
        len(combined),
    )

    return {"prior_facts": combined}


# ---------------------------------------------------------------------------
# Version-level: all model summaries for this version + prior version summary
# ---------------------------------------------------------------------------


async def inject_version_facts(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Gather all model summaries for the version + prior version summary."""
    current_summaries = await get_model_summary(
        rhaiis_version=state["rhaiis_version"],
        limit=100,
    )

    prior_version = await get_version_summary(
        rhaiis_version=state["baseline_version"],
        limit=1,
    )

    sections: list[str] = []
    summary_block = _format_facts_block(
        current_summaries,
        f"Model Summaries ({state['rhaiis_version']})",
    )
    if summary_block:
        sections.append(summary_block)

    if prior_version:
        prior_text = prior_version[0].get("summary_text", "")
        if prior_text:
            sections.append(
                f"## Prior Version Summary ({state['baseline_version']})\n{prior_text}"
            )

    combined = "\n\n".join(sections)
    logger.info(
        "inject_version_facts: %d model summaries, %d prior version summaries, %d chars",
        len(current_summaries),
        len(prior_version),
        len(combined),
    )

    return {"prior_facts": combined}
