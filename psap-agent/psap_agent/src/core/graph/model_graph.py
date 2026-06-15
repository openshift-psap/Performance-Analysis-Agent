"""Model-level analysis graph.

Synthesis-only — no MCP tools, no ReAct loop.

Pipeline:
  inject_facts → synthesize_report → extract_facts → judge_facts
    → reconcile_facts → store_facts → END

The synthesize_report node makes a single LLM call with the main model
to summarize all config-level findings for a given model.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from psap_agent.src.core.graph.nodes.extract_facts import extract_facts
from psap_agent.src.core.graph.nodes.inject_memory import inject_model_facts
from psap_agent.src.core.graph.nodes.judge_facts import judge_facts
from psap_agent.src.core.graph.nodes.reconcile_facts import reconcile_facts
from psap_agent.src.core.graph.nodes.store_facts import store_facts
from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.prompts.model_synthesis import (
    MODEL_SYNTHESIS_SYSTEM,
    build_model_synthesis_user_prompt,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# synthesize_report node
# ---------------------------------------------------------------------------


async def synthesize_model_report(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Single LLM call to synthesize config findings into a model report."""
    model = config["configurable"]["model"]
    cfg = state.get("config", {})

    user_prompt = build_model_synthesis_user_prompt(
        model_name=cfg.get("model", "?"),
        rhaiis_version=state.get("rhaiis_version", "?"),
        baseline_version=state.get("baseline_version", "?"),
        config_facts=state.get("prior_facts", "(none)"),
    )

    response = await model.ainvoke(
        [
            SystemMessage(content=MODEL_SYNTHESIS_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    report = response.content
    logger.info("synthesize_model_report: %d chars", len(report))
    return {"report": report}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_model_graph():
    """Compile the model-level analysis StateGraph."""
    graph = StateGraph(AnalysisState)

    graph.add_node("inject_facts", inject_model_facts)
    graph.add_node("synthesize_report", synthesize_model_report)
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
