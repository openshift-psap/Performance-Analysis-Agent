"""Version-level analysis graph.

Same pattern as the model graph but operates at version scope —
synthesises all model summaries into a single version-level report.

Pipeline:
  inject_facts → synthesize_report → extract_facts → judge_facts
    → reconcile_facts → store_facts → END
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from psap_agent.src.core.graph.nodes.extract_facts import extract_facts
from psap_agent.src.core.graph.nodes.inject_memory import inject_version_facts
from psap_agent.src.core.graph.nodes.judge_facts import judge_facts
from psap_agent.src.core.graph.nodes.reconcile_facts import reconcile_facts
from psap_agent.src.core.graph.nodes.store_facts import store_facts
from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.prompts.version_synthesis import (
    VERSION_SYNTHESIS_SYSTEM,
    build_version_synthesis_user_prompt,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# synthesize_report node
# ---------------------------------------------------------------------------


async def synthesize_version_report(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Single LLM call to synthesize model summaries into a version report."""
    model = config["configurable"]["model"]

    user_prompt = build_version_synthesis_user_prompt(
        rhaiis_version=state.get("rhaiis_version", "?"),
        baseline_version=state.get("baseline_version", "?"),
        model_summaries=state.get("prior_facts", "(none)"),
    )

    response = await model.ainvoke(
        [
            SystemMessage(content=VERSION_SYNTHESIS_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    report = response.content
    logger.info("synthesize_version_report: %d chars", len(report))
    return {"report": report}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_version_graph():
    """Compile the version-level analysis StateGraph."""
    graph = StateGraph(AnalysisState)

    graph.add_node("inject_facts", inject_version_facts)
    graph.add_node("synthesize_report", synthesize_version_report)
    graph.add_node("extract_facts", extract_facts)
    graph.add_node("judge_facts", judge_facts)
    graph.add_node("reconcile_facts", reconcile_facts)
    graph.add_node("store_facts", store_facts)

    graph.add_edge(START, "inject_facts")
    graph.add_edge("inject_facts", "synthesize_report")
    graph.add_edge("synthesize_report", "extract_facts")
    graph.add_edge("extract_facts", "judge_facts")
    graph.add_edge("judge_facts", "reconcile_facts")
    graph.add_edge("reconcile_facts", "store_facts")
    graph.add_edge("store_facts", END)

    return graph.compile()
