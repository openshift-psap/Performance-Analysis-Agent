"""A long tool loop must finish with a grounded answer, not a graph error."""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import ToolNode, create_react_agent

from psap_agent.src.core.agent import (
    _budgeted_model_and_prompt,
    _recoverable_tool_error,
)
from psap_agent.src.core.context_budget import bound_model_context


class AlwaysCallsTool(BaseChatModel):
    tools_enabled: bool = False
    tool_choice: str | None = None

    @property
    def _llm_type(self) -> str:
        return "always-calls-tool"

    def bind_tools(self, tools, **kwargs):
        assert tools
        choice = kwargs.get("tool_choice")
        return AlwaysCallsTool(
            tools_enabled=choice != "none", tool_choice=choice
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.tools_enabled:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {}, "id": "lookup-call"}],
            )
        else:
            message = AIMessage(
                content="The lookup is incomplete; no more tools are available."
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def lookup() -> str:
    """Return a sample evidence item."""
    return "sample evidence"


@tool
def failing_lookup() -> str:
    """Simulate a remote MCP schema-rejection response."""
    raise ToolException(
        "Error calling tool 'failing_lookup': 1 validation error\n"
        "category_filter\n  Unexpected keyword argument [input_value='secret']"
    )


class FailsThenAnswers(AlwaysCallsTool):
    def bind_tools(self, tools, **kwargs):
        assert tools
        return FailsThenAnswers(tools_enabled=True)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(content="The profile comparison was unavailable.")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "failing_lookup", "args": {}, "id": "bad-call"}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class HistorySensitiveModel(AlwaysCallsTool):
    """Mimic Gemini continuing tool calls from a prior function-call history."""

    def bind_tools(self, tools, **kwargs):
        assert tools
        choice = kwargs.get("tool_choice")
        return HistorySensitiveModel(
            tools_enabled=choice != "none", tool_choice=choice
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        has_function_history = any(
            isinstance(message, ToolMessage)
            or (isinstance(message, AIMessage) and bool(message.tool_calls))
            for message in messages
        )
        if self.tools_enabled or has_function_history:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {}, "id": "lookup-call"}],
            )
        else:
            message = AIMessage(content="Final answer from the collected evidence.")
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_model_and_prompt_switch_to_final_answer_mode():
    model = AlwaysCallsTool()
    selected, prompt = _budgeted_model_and_prompt(model, [lookup], "System rules")
    messages = [HumanMessage(content="Investigate")]

    assert selected({"remaining_steps": 30}, None).tools_enabled is True
    assert selected({"remaining_steps": 8}, None) is model
    assert (
        "tool-call budget"
        not in prompt({"remaining_steps": 30, "messages": messages})[0].content
    )
    assert (
        "tool-call budget"
        in prompt({"remaining_steps": 8, "messages": messages})[0].content
    )


def test_gemini_final_answer_uses_explicit_no_tool_choice():
    selected, _ = _budgeted_model_and_prompt(
        AlwaysCallsTool(), [lookup], "System rules", force_no_tool_choice=True
    )

    assert selected({"remaining_steps": 30}, None).tools_enabled is True
    final_model = selected({"remaining_steps": 8}, None)
    assert final_model.tools_enabled is False
    assert final_model.tool_choice == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("force_no_tool_choice", [False, True])
async def test_repeating_tool_model_finishes_before_graph_recursion_limit(
    force_no_tool_choice,
):
    model, prompt = _budgeted_model_and_prompt(
        AlwaysCallsTool(),
        [lookup],
        "System rules",
        force_no_tool_choice=force_no_tool_choice,
    )
    graph = create_react_agent(
        model=model,
        tools=[lookup],
        prompt=prompt,
        pre_model_hook=bound_model_context,
    ).with_config(recursion_limit=12)

    result = await graph.ainvoke({"messages": [HumanMessage(content="Investigate")]})

    assert result["messages"][-1].content.startswith("The lookup is incomplete")
    assert not result["messages"][-1].tool_calls


@pytest.mark.asyncio
async def test_final_pass_drops_function_history_that_causes_more_tool_calls():
    model, prompt = _budgeted_model_and_prompt(
        HistorySensitiveModel(),
        [lookup],
        "System rules",
        force_no_tool_choice=True,
    )
    graph = create_react_agent(
        model=model,
        tools=[lookup],
        prompt=prompt,
        pre_model_hook=bound_model_context,
    ).with_config(recursion_limit=12)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Investigate")]}
    )

    assert result["messages"][-1].content == "Final answer from the collected evidence."
    assert not result["messages"][-1].tool_calls


@pytest.mark.asyncio
async def test_mcp_validation_error_becomes_tool_result_not_agent_error():
    model, prompt = _budgeted_model_and_prompt(
        FailsThenAnswers(), [failing_lookup], "System rules"
    )
    graph = create_react_agent(
        model=model,
        tools=ToolNode([failing_lookup], handle_tool_errors=_recoverable_tool_error),
        prompt=prompt,
        pre_model_hook=bound_model_context,
    ).with_config(recursion_limit=20)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Investigate")]}
    )

    tool_result = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert "Unsupported argument 'category_filter'" in tool_result.content
    assert "secret" not in tool_result.content
    assert result["messages"][-1].content == "The profile comparison was unavailable."
