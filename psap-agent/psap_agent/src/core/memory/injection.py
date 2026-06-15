"""Memory injection logic for the PSAP agent.

Queries the database for prior facts and red flags, then constructs
a structured memory block to inject into the agent's context before
the analysis loop. Enforces a token budget (~2000 tokens total).
"""

from typing import Optional

from psap_agent.src.core.memory.queries import (
    get_config_facts,
    get_model_summary,
)
from psap_agent.src.core.memory.red_flags import RedFlagsManager
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

TOKEN_BUDGET_PRIOR_FACTS = 1000
TOKEN_BUDGET_RED_FLAGS = 500
TOKEN_BUDGET_MODEL_SUMMARY = 500
CHARS_PER_TOKEN = 4  # rough estimate for budget enforcement


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget, keeping most recent content."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] + "\n... (truncated to fit token budget)"


def _format_facts_by_status(facts: list[dict]) -> str:
    """Group facts by status and format into a memory block."""
    active = [f for f in facts if f.get("status", "active") == "active"]
    resolved = [f for f in facts if f.get("status") == "resolved"]

    sections = []

    if active:
        lines = ["### Active Issues (verify if still present):"]
        for f in active:
            category = f.get("category", "unknown")
            text = f.get("fact_text", f.get("summary_text", ""))
            version = f.get("rhaiis_version", "")
            line = f"- [{category}] {text}"
            if version:
                line += f" (from {version})"
            lines.append(line)
        sections.append("\n".join(lines))

    if resolved:
        lines = [
            "### Resolved Issues (context only — do NOT reinvestigate unless re-emerged):"
        ]
        for f in resolved:
            category = f.get("category", "unknown")
            text = f.get("fact_text", f.get("summary_text", ""))
            lines.append(f"- [{category}] {text}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


async def build_memory_context(
    rhaiis_version: str,
    baseline_version: str,
    model_name: str,
    accelerator: Optional[str] = None,
    profile: Optional[str] = None,
    red_flags_dir: Optional[str] = None,
) -> dict:
    """Build the full memory context for injection into agent prompt.

    Returns a dict with:
        - prior_facts: formatted facts from the previous version
        - red_flags: relevant red flags from current version run
        - full_context: the combined memory block string
    """
    sections = []
    prior_facts_text = ""
    red_flags_text = ""

    # 1. Query prior facts from the immediately previous version (baseline)
    try:
        prior_facts = await get_config_facts(
            rhaiis_version=baseline_version,
            model_name=model_name,
            accelerator=accelerator,
            profile=profile,
            limit=30,
        )
        if prior_facts:
            prior_facts_text = _format_facts_by_status(prior_facts)
            prior_facts_text = _truncate_to_budget(
                prior_facts_text, TOKEN_BUDGET_PRIOR_FACTS
            )
            logger.info(
                f"Injecting {len(prior_facts)} prior facts from {baseline_version}"
            )
    except Exception as e:
        logger.warning(f"Failed to query prior facts: {e}")

    # 2. Query model-level summary from previous version (if exists)
    model_summary_text = ""
    try:
        model_summaries = await get_model_summary(
            rhaiis_version=baseline_version,
            model_name=model_name,
            limit=1,
        )
        if model_summaries:
            summary = model_summaries[0]
            model_summary_text = summary.get("summary_text", "")
            model_summary_text = _truncate_to_budget(
                model_summary_text, TOKEN_BUDGET_MODEL_SUMMARY
            )
    except Exception as e:
        logger.warning(f"Failed to query model summary: {e}")

    # 3. Read red flags for current version
    try:
        rfm = RedFlagsManager(base_dir=red_flags_dir)
        red_flags_text = rfm.get_relevant_flags(rhaiis_version, model_name)
        if red_flags_text:
            red_flags_text = _truncate_to_budget(
                red_flags_text, TOKEN_BUDGET_RED_FLAGS
            )
            logger.info(
                f"Injecting red flags for {model_name} from {rhaiis_version}"
            )
    except Exception as e:
        logger.warning(f"Failed to read red flags: {e}")

    # 4. Assemble the memory block
    if prior_facts_text:
        sections.append(prior_facts_text)

    if model_summary_text:
        sections.append(
            f"### Prior Model Summary ({baseline_version}):\n{model_summary_text}"
        )

    if red_flags_text:
        sections.append(
            "### Current Red Flags (from other configs in this version run):\n"
            + red_flags_text
        )

    full_context = ""
    if sections:
        full_context = "## Prior Memory\n\n" + "\n\n".join(sections)

    total_tokens = _estimate_tokens(full_context)
    logger.info(f"Memory context built: ~{total_tokens} tokens")

    return {
        "prior_facts": prior_facts_text,
        "red_flags": red_flags_text,
        "model_summary": model_summary_text,
        "full_context": full_context,
    }
