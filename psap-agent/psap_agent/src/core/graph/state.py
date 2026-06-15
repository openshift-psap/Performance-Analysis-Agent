"""State schema for the PSAP analysis graph pipeline.

Defines the AnalysisState TypedDict used by config-level, model-level,
and version-level graphs.  All fields are optional (total=False) so that
each graph node only returns the keys it needs to set.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class AnalysisState(TypedDict, total=False):
    # ---- Input (set by the /v1/analyze endpoint) ----
    analysis_level: str  # "config" | "model" | "version" | "consolidation"
    mode: Optional[str]  # "shallow" | "deep" (config level only)
    regression_detected: Optional[bool]
    run_id: str
    rhaiis_version: str
    baseline_version: str
    config: dict[str, Any]  # model, accelerator, tp, prompt_toks, output_toks, concurrency

    # ---- Memory injection (set by inject_memory / inject_facts) ----
    prior_facts: str
    red_flags: str

    # ---- Agent / synthesis output ----
    messages: list[Any]  # ReAct message history (config level only)
    report: str

    # ---- Fact pipeline ----
    extracted_facts: list[dict[str, Any]]
    validated_facts: list[dict[str, Any]]
    reconciled_facts: list[dict[str, Any]]
    new_red_flags: list[str]

    # ---- Result metrics ----
    facts_stored_count: int
    red_flags_added_count: int
