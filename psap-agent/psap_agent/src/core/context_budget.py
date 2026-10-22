"""Bound model input without discarding the checkpointed conversation.

LangGraph calls this before *each* model invocation.  The original messages
remain in state (and in Langfuse); only the copy sent to the model is trimmed.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Character limits are intentionally conservative across Gemini, Claude and
# OpenAI.  Tool schemas and the system prompt consume context outside this cap.
_MAX_TOOL_RESULT_CHARS = 20_000
_MAX_MESSAGE_CHARS = 240_000
_TRUNCATION_NOTICE = "\n[Tool result shortened for context. Re-run with narrower filters if needed.]"
FINAL_ANSWER_STEP_RESERVE = 8
_MAX_FINAL_TOOL_RESULTS = 24
_MAX_FINAL_TOOL_RESULT_CHARS = 5_000


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _short_text(content, limit: int) -> str:
    text = _content_text(content)
    if len(text) <= limit:
        return text
    marker = "\n[Middle omitted; this is partial evidence.]\n"
    keep = limit - len(marker)
    head = keep * 3 // 4
    return text[:head] + marker + text[-(keep - head):]


def _final_answer_context(messages: list) -> list[HumanMessage]:
    """Give the final model evidence as text, without prior function-call history."""
    latest_human_index = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
        -1,
    )
    request = (
        _content_text(messages[latest_human_index].content)
        if latest_human_index >= 0
        else "No user request found in conversation state."
    )
    previous = [
        message
        for message in messages[:latest_human_index]
        if isinstance(message, (HumanMessage, AIMessage))
        and not (isinstance(message, AIMessage) and message.tool_calls)
        and _content_text(message.content).strip()
    ][-4:]
    prior_dialogue = "\n".join(
        f"{message.type}: {_short_text(message.content, 3_000)}"
        for message in previous
    )

    results = [
        message for message in messages[latest_human_index + 1 :]
        if isinstance(message, ToolMessage)
    ]
    if not results:
        results = [message for message in messages if isinstance(message, ToolMessage)][-_MAX_FINAL_TOOL_RESULTS:]
    omitted = max(0, len(results) - _MAX_FINAL_TOOL_RESULTS)
    if omitted:
        results = results[:4] + results[-(_MAX_FINAL_TOOL_RESULTS - 4):]
    evidence = "\n\n".join(
        f"[{index}. {message.name or 'tool'}] "
        + _short_text(message.content, _MAX_FINAL_TOOL_RESULT_CHARS)
        for index, message in enumerate(results, 1)
    )
    if omitted:
        evidence += f"\n\n[{omitted} intermediate tool results omitted.]"

    text = f"Current request:\n{request}"
    if prior_dialogue:
        text += f"\n\nRelevant prior conversation:\n{prior_dialogue}"
    text += (
        "\n\nTool evidence from this request (some results may be shortened; "
        "treat omitted parts as unknown):\n"
        + (evidence or "No successful tool evidence was collected.")
    )
    return [HumanMessage(content=text)]


def _message_size(message) -> int:
    """Estimate prompt size, including tool-call arguments hidden in content."""
    size = len(str(message.content))
    if isinstance(message, AIMessage):
        size += sum(len(str(call)) for call in message.tool_calls)
    return size


def _bound_tool_message(message: ToolMessage) -> ToolMessage:
    content = message.content
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return message
    keep = _MAX_TOOL_RESULT_CHARS - len(_TRUNCATION_NOTICE)
    head = keep * 3 // 4
    shortened = text[:head] + _TRUNCATION_NOTICE + text[-(keep - head):]
    return message.model_copy(update={"content": shortened})


def bound_model_context(state: dict) -> dict:
    """Keep the latest request and recent complete tool exchanges in model input.

    Every retained ToolMessage keeps its preceding AI tool-call message.  This
    avoids provider errors from orphaned tool responses while allowing older
    exchanges to be omitted once the budget is exhausted.
    """
    original = list(state.get("messages", []))
    if not original:
        return {"llm_input_messages": original}
    remaining_steps = state.get("remaining_steps")
    if isinstance(remaining_steps, int) and remaining_steps <= FINAL_ANSWER_STEP_RESERVE:
        return {"llm_input_messages": _final_answer_context(original)}

    # A human turn and its subsequent AI/tool messages form an indivisible
    # outer unit.  Recent turns are preferred to stale checkpoint history.
    turns = []
    for message in original:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([])
        turns[-1].append(_bound_tool_message(message) if isinstance(message, ToolMessage) else message)

    latest = turns[-1]
    # Within the current turn, keep complete AI/tool-call batches.  It is safe
    # to drop an old batch because its tool responses are dropped with it.
    batches = []
    for message in latest:
        if isinstance(message, AIMessage) and message.tool_calls:
            batches.append([message])
        elif isinstance(message, ToolMessage) and batches:
            batches[-1].append(message)
        else:
            batches.append([message])

    budget = _MAX_MESSAGE_CHARS
    kept = []
    for batch in reversed(batches):
        batch_size = sum(_message_size(message) for message in batch)
        if batch_size <= budget:
            kept[:0] = batch
            budget -= batch_size
        else:
            break

    # Always retain the latest human request.  If a very large user prompt
    # alone exceeds the budget, do not silently alter it.
    latest_human = next((m for m in reversed(latest) if isinstance(m, HumanMessage)), None)
    if latest_human is not None and latest_human not in kept:
        kept.insert(0, latest_human)
        budget -= _message_size(latest_human)

    for turn in reversed(turns[:-1]):
        turn_size = sum(_message_size(message) for message in turn)
        if turn_size > budget:
            break
        kept[:0] = turn
        budget -= turn_size

    if len(kept) != len(original) or any(a is not b for a, b in zip(kept, original)):
        logger.info(f"Bound model input from {len(original)} to {len(kept)} messages")
        # This is model input only; it is not written into the checkpoint.
        kept.insert(0, SystemMessage(content=(
            "Some earlier conversation or tool output was omitted to fit the model context. "
            "Re-run a tool with narrower filters when exact omitted data is needed."
        )))

    return {"llm_input_messages": kept}
