"""Model-input trimming must not corrupt LangGraph tool-call history."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from psap_agent.src.core.context_budget import bound_model_context


def _exchange(index: int, size: int = 20_000):
    call_id = f"call-{index}"
    return [
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": call_id}]),
        ToolMessage(content=f"start-{index}:" + "x" * size + f":end-{index}", tool_call_id=call_id),
    ]


def test_small_context_is_unchanged():
    messages = [HumanMessage(content="Find benchmarks"), *_exchange(0, 50)]
    result = bound_model_context({"messages": messages})["llm_input_messages"]
    assert result == messages
    assert result[0] is messages[0]


def test_large_tool_result_is_shortened_only_for_model():
    messages = [HumanMessage(content="Find benchmarks"), *_exchange(0, 30_000)]
    result = bound_model_context({"messages": messages})["llm_input_messages"]
    tool_result = result[-1]
    assert isinstance(result[0], SystemMessage)
    assert tool_result.tool_call_id == "call-0"
    assert len(tool_result.content) <= 20_000
    assert tool_result.content.startswith("start-0:")
    assert tool_result.content.endswith(":end-0")
    assert len(messages[-1].content) > 30_000


def test_structured_tool_content_is_bounded():
    messages = [
        HumanMessage(content="Inspect this"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "call-0"}]),
        ToolMessage(content=[{"type": "text", "text": "x" * 30_000}], tool_call_id="call-0"),
    ]
    result = bound_model_context({"messages": messages})["llm_input_messages"]
    assert isinstance(result[-1].content, str)
    assert len(result[-1].content) <= 20_000
    assert len(messages[-1].content[0]["text"]) == 30_000


def test_many_tool_calls_keep_valid_recent_pairs_and_latest_request():
    messages = [HumanMessage(content="Analyze this run")]
    for index in range(18):
        messages.extend(_exchange(index))

    result = bound_model_context({"messages": messages})["llm_input_messages"]
    assert len(result) < len(messages)
    assert any(isinstance(message, HumanMessage) and message.content == "Analyze this run" for message in result)
    assert isinstance(result[-1], ToolMessage)
    assert result[-1].tool_call_id == "call-17"
    calls = {
        call["id"]
        for message in result if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    assert all(message.tool_call_id in calls for message in result if isinstance(message, ToolMessage))
    assert len(messages) == 37  # Checkpointed history was not mutated.


def test_final_pass_has_bounded_evidence_without_function_call_history():
    messages = [HumanMessage(content="Earlier question"), AIMessage(content="Earlier answer")]
    messages.append(HumanMessage(content="Explain the MI355X regression"))
    for index in range(26):
        messages.extend(_exchange(index, 30_000))

    result = bound_model_context(
        {"messages": messages, "remaining_steps": 8}
    )["llm_input_messages"]

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    text = result[0].content
    assert "Explain the MI355X regression" in text
    assert "Earlier answer" in text
    assert "start-0:" in text
    assert ":end-25" in text
    assert "intermediate tool results omitted" in text
    assert len(text) < 140_000
    assert len(messages) == 55  # Only model input, not checkpoint state, is changed.
