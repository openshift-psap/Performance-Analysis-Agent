"""Config-level analysis graph.

Pipeline:
  inject_memory → react_agent → extract_facts → judge_facts
    → reconcile_facts → store_facts → update_red_flags → END

The react_agent node wraps the existing prebuilt ReAct agent with a
memory-augmented system prompt.  It uses MCP tools for deep investigation
or a streamlined prompt for shallow (no-regression) runs.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from psap_agent.src.core.graph.nodes.extract_facts import extract_facts
from psap_agent.src.core.graph.nodes.inject_memory import inject_config_memory
from psap_agent.src.core.graph.nodes.judge_facts import judge_facts
from psap_agent.src.core.graph.nodes.reconcile_facts import reconcile_facts
from psap_agent.src.core.graph.nodes.store_facts import store_facts
from psap_agent.src.core.graph.nodes.update_red_flags import update_red_flags
from psap_agent.src.core.graph.state import AnalysisState
from psap_agent.src.core.prompt import get_analysis_system_prompt
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# react_agent node
# ---------------------------------------------------------------------------


def _build_user_message(state: AnalysisState) -> str:
    """Construct the user message that kicks off the ReAct loop."""
    cfg = state.get("config", {})
    regression = state.get("regression_detected", False)
    profile = f"{cfg.get('prompt_toks', 0)}/{cfg.get('output_toks', 0)}"

    return (
        f"Analyze the performance comparison between "
        f"{state.get('rhaiis_version', '?')} and {state.get('baseline_version', '?')} "
        f"for the following configuration:\n"
        f"- Model: {cfg.get('model', '?')}\n"
        f"- Accelerator: {cfg.get('accelerator', '?')}\n"
        f"- Tensor Parallel: {cfg.get('tp', '?')}\n"
        f"- Profile: {profile}\n"
        f"- Concurrency: {cfg.get('concurrency', 'N/A')}\n"
        f"- Regression detected: {'yes' if regression else 'no'}\n\n"
        "Produce a comprehensive analysis report."
    )


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


def _extract_report_text(messages: list) -> str:
    """Pull the report from the last AI message in the ReAct output."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return _normalize_content(msg.content)
        if isinstance(msg, dict) and msg.get("type") == "ai":
            return _normalize_content(msg.get("content", ""))
    return ""


async def react_agent_node(
    state: AnalysisState, config: RunnableConfig
) -> dict[str, Any]:
    """Run the full ReAct loop with MCP tools."""
    configurable = config.get("configurable", {})
    model = configurable["model"]
    tools = configurable.get("tools", [])

    system_prompt = get_analysis_system_prompt(
        memory_context=state.get("prior_facts"),
        mode=state.get("mode", "deep"),
    )
    user_message = _build_user_message(state)

    agent = create_react_agent(model=model, prompt=system_prompt, tools=tools)

    logger.info(
        "react_agent_node: invoking ReAct agent (mode=%s, tools=%d)",
        state.get("mode", "deep"),
        len(tools),
    )

    result = await agent.ainvoke({"messages": [("user", user_message)]})
    messages = result.get("messages", [])
    report = _extract_report_text(messages)

    logger.info(
        "react_agent_node: report length=%d chars, messages=%d",
        len(report),
        len(messages),
    )

    return {"report": report, "messages": messages}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_config_graph():
    """Compile the config-level analysis StateGraph."""
    graph = StateGraph(AnalysisState)

    graph.add_node("inject_memory", inject_config_memory)
    graph.add_node("react_agent", react_agent_node)
    graph.add_node("extract_facts", extract_facts)
    graph.add_node("judge_facts", judge_facts)
    graph.add_node("reconcile_facts", reconcile_facts)
    graph.add_node("store_facts", store_facts)
    graph.add_node("update_red_flags", update_red_flags)

    graph.add_edge(START, "inject_memory")
    graph.add_edge("inject_memory", "react_agent")
    graph.add_edge("react_agent", "extract_facts")
    graph.add_edge("extract_facts", "judge_facts")
    graph.add_edge("judge_facts", "reconcile_facts")
    graph.add_edge("reconcile_facts", "store_facts")
    graph.add_edge("store_facts", "update_red_flags")
    graph.add_edge("update_red_flags", END)

    return graph.compile()
