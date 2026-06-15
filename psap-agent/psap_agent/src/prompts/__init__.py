"""Prompt modules for the PSAP memory pipeline.

All prompts used by the analysis graph nodes are centralized here.
"""

from psap_agent.src.prompts.consolidation import (
    CONSOLIDATION_SYSTEM,
    build_consolidation_user_prompt,
)
from psap_agent.src.prompts.extraction import (
    FACT_EXTRACTION_SYSTEM,
    build_extraction_user_prompt,
)
from psap_agent.src.prompts.judge import (
    FACT_VALIDATION_SYSTEM,
    build_judge_user_prompt,
)
from psap_agent.src.prompts.model_synthesis import (
    MODEL_SYNTHESIS_SYSTEM,
    build_model_synthesis_user_prompt,
)
from psap_agent.src.prompts.reconciliation import (
    FACT_RECONCILIATION_SYSTEM,
    build_reconciliation_user_prompt,
)
from psap_agent.src.prompts.version_synthesis import (
    VERSION_SYNTHESIS_SYSTEM,
    build_version_synthesis_user_prompt,
)

__all__ = [
    "CONSOLIDATION_SYSTEM",
    "FACT_EXTRACTION_SYSTEM",
    "FACT_RECONCILIATION_SYSTEM",
    "FACT_VALIDATION_SYSTEM",
    "MODEL_SYNTHESIS_SYSTEM",
    "VERSION_SYNTHESIS_SYSTEM",
    "build_consolidation_user_prompt",
    "build_extraction_user_prompt",
    "build_judge_user_prompt",
    "build_model_synthesis_user_prompt",
    "build_reconciliation_user_prompt",
    "build_version_synthesis_user_prompt",
]
