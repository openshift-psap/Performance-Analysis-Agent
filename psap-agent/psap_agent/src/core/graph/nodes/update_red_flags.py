"""Red flags update node — appends cross-config red flags to the
ephemeral markdown file.  Only used in config-level graphs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.core.memory.red_flags import RedFlagsManager
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


async def update_red_flags(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Append newly extracted red flags to the red flags file."""
    red_flags = state.get("new_red_flags", [])
    if not red_flags:
        return {"red_flags_added_count": 0}

    cfg = state.get("config", {})
    model_name = cfg.get("model", "")
    version = state.get("rhaiis_version", "")

    red_flags_dir = config.get("configurable", {}).get("red_flags_dir")
    rfm = RedFlagsManager(base_dir=red_flags_dir)

    # Separate model-specific flags from ALL MODELS flags
    all_models_flags: list[str] = []
    model_flags: list[str] = []
    for flag in red_flags:
        lower = flag.lower()
        if "all models" in lower or "all configurations" in lower:
            all_models_flags.append(flag)
        else:
            model_flags.append(flag)

    added = 0
    if model_flags:
        rfm.append_flags(version, model_name, model_flags)
        added += len(model_flags)
    if all_models_flags:
        rfm.append_flags(version, "ALL MODELS", all_models_flags)
        added += len(all_models_flags)

    logger.info(
        "update_red_flags: added %d flags (%d model-specific, %d ALL MODELS)",
        added,
        len(model_flags),
        len(all_models_flags),
    )
    return {"red_flags_added_count": added}
