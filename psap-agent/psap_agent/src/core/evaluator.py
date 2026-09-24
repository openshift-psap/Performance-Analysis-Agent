"""LLM-as-Judge automated evaluation for agent responses.

Runs asynchronously after each agent response completes, scoring traces
in Langfuse on correctness, tool efficiency, hallucination, and completeness.
"""

import asyncio
import json
from typing import Any

from langfuse import get_client

from psap_agent.src.core.model_factory import create_chat_model
from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def _content_to_str(content) -> str:
    """Normalize message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)

EVAL_CRITERIA = {
    "correctness": {
        "description": "Did the agent answer the question accurately?",
        "rubric": (
            "Score 1.0: Response directly addresses the question with data from tool calls, memory context, "
            "or accurate general knowledge for non-data questions. "
            "Score 0.5: Partially correct, missing some data or minor inaccuracies. "
            "Score 0.0: Incorrect, fabricated data, or did not answer the question. "
            "IMPORTANT: The agent has TWO valid data sources: tool CALL/RESULT pairs AND memory context. "
            "For general knowledge questions (e.g. 'what is X?'), accurate answers from the agent's knowledge are valid. "
            "Check ALL sources before concluding data was fabricated."
        ),
    },
    "tool_efficiency": {
        "description": "Did the agent use the minimum number of tools needed?",
        "rubric": (
            "Score 1.0: Used only necessary tools, no redundant calls. "
            "Discovery/listing calls (e.g., discover_configurations, list_models) "
            "to find IDs or resolve names needed by subsequent tools count as "
            "necessary, not redundant. "
            "Score 0.5: One or two unnecessary tool calls. "
            "Score 0.0: Many redundant tool calls or used wrong tools entirely."
        ),
    },
    "hallucination": {
        "description": "Did the agent fabricate URLs, metrics, or data not returned by tools or memory context?",
        "rubric": (
            "Score 1.0: All data, URLs, and metrics are traceable to tool outputs OR memory context. "
            "Also score 1.0 if the response contains only general knowledge (e.g. explaining what a technology, "
            "company, or concept is) without making data-specific claims — general knowledge does NOT require tool evidence. "
            "Score 0.5: Minor embellishment but core data is tool-backed or memory-backed. "
            "Score 0.0: Contains fabricated URLs, invented metrics, or data-specific claims not from tools or memory. "
            "CRITICAL DISTINCTION: General knowledge about technologies, companies, or concepts (e.g. 'Nvidia makes GPUs', "
            "'vLLM is an inference engine', 'CUDA is a parallel computing platform') is NOT hallucination even without tool calls. "
            "Only flag as hallucination when SPECIFIC metrics, benchmark numbers, URLs, or dataset-specific facts are fabricated. "
            "The agent has TWO valid data sources: tool CALL/RESULT pairs AND memory context "
            "from past interactions. Data from EITHER source is NOT fabrication. "
            "Carefully cross-reference EVERY claim against ALL tool results AND memory context before flagging. "
            "CODE BLOCK CHECK: When the response contains reconstructed commands in code blocks, verify that "
            "specific numeric values and paths in those commands are traceable to tool outputs. Argument names "
            "can be inferred, but specific VALUES (e.g. a number like 1608, an endpoint like /v1/completions) "
            "must appear in the tool data. A number in a code block that exists nowhere in tool outputs is fabrication."
        ),
    },
    "completeness": {
        "description": "Did the agent address all parts of the user's query, using all relevant data from tool results?",
        "rubric": (
            "Score 1.0: All parts of the question fully addressed AND all relevant data "
            "from tool results is reflected in the response. "
            "Score 0.5: Most parts addressed, but the response OMITS data points that "
            "the tools returned and that are relevant to the query. For example, if the "
            "user asked for data at concurrency 1 and the tool returned it, but the "
            "response says 'no data available' — that is an omission, score 0.5 or lower. "
            "Score 0.0: Significant parts of the question unanswered or large amounts of "
            "relevant tool data omitted/contradicted. "
            "IMPORTANT: Compare the tool RESULTS against the response. If a tool returned "
            "data that directly answers part of the user's query but the response ignores "
            "or contradicts that data, this is incomplete — even if the response is otherwise "
            "well-structured."
        ),
    },
    "internal_consistency": {
        "description": "Are the numbers and claims within the response logically consistent with each other?",
        "rubric": (
            "Score 1.0: All numbers, percentages, and quantitative claims are internally consistent. "
            "Sub-components do not exceed totals, percentage breakdowns sum correctly, and no table "
            "contradicts another table in the same response. "
            "Score 0.5: Minor arithmetic imprecision (rounding differences). "
            "Score 0.0: Clear logical contradictions between numbers in the response "
            "(e.g., a sub-component duration exceeding the stated total, or a 'per block' metric "
            "that is impossible given other 'per block' metrics in the same response)."
        ),
    },
}

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge for an AI performance analysis agent. "
    "You evaluate agent responses on specific criteria.\n\n"
    "IMPORTANT: Each tool interaction below is shown as a CALL/RESULT pair. "
    "The RESULT directly below a CALL is the output of that specific call. "
    "Before claiming data was fabricated, carefully check ALL CALL/RESULT pairs -- "
    "the agent may have made many calls and the data you're looking for may be in a later pair.\n\n"
    "CRITICAL: Do NOT use your own training knowledge to judge factual accuracy. "
    "Your training data may be outdated. You can ONLY evaluate whether the agent's claims "
    "are supported by the tool outputs and memory context provided below. "
    "If the agent makes a claim and you cannot find it in tool outputs or memory, "
    "that is a potential hallucination. But NEVER say something 'does not exist' or "
    "'is incorrect' based on your own knowledge — you may simply be out of date.\n\n"
    "You MUST respond with valid JSON only, no other text. "
    "The JSON must have exactly these keys: 'score' (float 0.0-1.0) and 'reasoning' (string)."
)


def _build_judge_prompt(
    criterion_name: str,
    criterion: dict,
    user_query: str,
    agent_response: str,
    tool_calls_summary: str,
    memory_context: str = "",
) -> str:
    parts = [
        f"Evaluate the following agent interaction on: {criterion['description']}\n",
        f"Rubric:\n{criterion['rubric']}\n",
        f"User Query:\n{user_query}\n",
    ]
    if memory_context:
        parts.append(
            f"Memory Context (from past interactions -- the agent is allowed to reference this):\n"
            f"{memory_context}\n"
        )
    parts.extend([
        f"Tool Calls Made:\n{tool_calls_summary}\n",
        f"Agent Response:\n{agent_response}\n",
        f"Respond with JSON: {{\"score\": <0.0-1.0>, \"reasoning\": \"<brief explanation>\"}}",
    ])
    return "\n".join(parts)


def _parse_judge_response(response_text: str) -> dict:
    """Parse JSON from the judge response, stripping markdown fences."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


_MAX_RETRIES = 2
_RETRY_DELAY_SECS = 3.0


async def _run_single_eval(
    criterion_name: str,
    criterion: dict,
    user_query: str,
    agent_response: str,
    tool_calls_summary: str,
    trace_id: str,
    memory_context: str = "",
) -> tuple[str, float | None]:
    """Run a single LLM-as-Judge evaluation and score the trace.

    Retries on transient model/network failures and on JSON parse errors
    (with a nudge to return valid JSON on the second attempt).

    Returns:
        (criterion_name, score) -- score is None if evaluation failed.
    """
    judge_model = create_chat_model(
        settings.LLM_JUDGE_MODEL,
        temperature=0.0,
    )

    prompt = _build_judge_prompt(
        criterion_name, criterion, user_query, agent_response, tool_calls_summary,
        memory_context=memory_context,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            result = await judge_model.ainvoke(messages)
            response_text = _content_to_str(result.content)

            try:
                parsed = _parse_judge_response(response_text)
            except (json.JSONDecodeError, ValueError):
                if attempt < _MAX_RETRIES - 1:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Respond with ONLY a JSON object: "
                            '{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}'
                        ),
                    })
                    continue
                raise

            score = max(0.0, min(1.0, float(parsed["score"])))
            reasoning = parsed.get("reasoning", "")

            usage_meta = getattr(result, "usage_metadata", None)
            token_info = ""
            if usage_meta:
                inp = usage_meta.get("input_tokens", 0)
                out = usage_meta.get("output_tokens", 0)
                token_info = f" [tokens: {inp} in / {out} out]"

            client = get_client()
            client.create_score(
                trace_id=trace_id,
                name=f"llm-judge-{criterion_name}",
                value=score,
                comment=f"{reasoning}{token_info}",
            )
            logger.info(
                f"LLM-Judge scored {criterion_name}={score:.1f} for trace={trace_id}"
            )
            return criterion_name, score

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            logger.warning(f"LLM-Judge parse error for {criterion_name} (attempt {attempt + 1}): {e}")
        except Exception as e:
            last_error = e
            logger.warning(f"LLM-Judge transient error for {criterion_name} (attempt {attempt + 1}): {e}")
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY_SECS)

    logger.warning(f"LLM-Judge eval failed for {criterion_name} after {_MAX_RETRIES} attempts: {last_error}")
    return criterion_name, None


_CRITERIA_NEEDING_FULL_RESULTS = {"correctness", "hallucination", "completeness", "internal_consistency"}


_MEMORY_CONTEXT_MARKER = "[SYSTEM - Memory Context (use if relevant, ignore if not)]"


def _extract_conversation_parts(messages: list) -> tuple[str, str, str, str, str]:
    """Extract user query, memory context, agent response, and tool call summaries.

    Pairs each tool call with its result using tool_call_id so the judge
    can trace exactly which data came from which call.

    Returns:
        (user_query, agent_response, full_tool_summary, calls_only_summary, memory_context)
    """
    user_query = ""
    memory_context = ""
    agent_response = ""
    pending_calls: dict[str, str] = {}
    tool_exchanges: list[str] = []
    call_names: list[str] = []

    for msg in messages:
        msg_type = getattr(msg, "type", "")
        content = getattr(msg, "content", "")

        if msg_type == "human":
            raw = _content_to_str(content)
            if _MEMORY_CONTEXT_MARKER in raw:
                user_query, mem_part = raw.split(_MEMORY_CONTEXT_MARKER, 1)
                user_query = user_query.strip()
                memory_context = mem_part.strip()
            else:
                user_query = raw
        elif msg_type == "ai":
            if content:
                agent_response = _content_to_str(content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    call_id = tc.get("id", "")
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    call_desc = f"{name}({json.dumps(args, default=str)[:200]})"
                    pending_calls[call_id] = call_desc
                    call_names.append(call_desc)
        elif msg_type == "tool":
            call_id = getattr(msg, "tool_call_id", "")
            call_desc = pending_calls.pop(call_id, getattr(msg, "name", "tool"))
            result_text = _content_to_str(content)[:10000]
            tool_exchanges.append(f"CALL: {call_desc}\nRESULT: {result_text}")

    full_summary = "\n\n".join(tool_exchanges) if tool_exchanges else "No tool calls made."
    calls_only = "\n".join(f"- {c}" for c in call_names) if call_names else "No tool calls made."
    return user_query, agent_response, full_summary, calls_only, memory_context


_MEMORY_GATE_THRESHOLD = 1.0
_MEMORY_GATE_CRITERIA = ("hallucination", "correctness", "completeness", "internal_consistency")


async def evaluate_response(
    messages: list,
    trace_id: str,
    user_query: str = "",
    memory_context: str = "",
) -> dict[str, float]:
    """Run all LLM-as-Judge evaluations asynchronously.

    Args:
        memory_context: Pre-extracted memory context string. If not provided,
            falls back to extracting from the human message in collected_messages.

    Returns:
        Dict mapping criterion name to score (e.g. {"correctness": 1.0, "hallucination": 0.5}).
        Empty dict if evaluation is disabled or fails entirely.
    """
    if not settings.ENABLE_LLM_JUDGE:
        return {}

    try:
        extracted_query, agent_response, full_summary, calls_only, extracted_memory = (
            _extract_conversation_parts(messages)
        )
        user_query = user_query or extracted_query
        memory_context = memory_context or extracted_memory

        if not user_query or not agent_response:
            logger.debug("Skipping evaluation: missing user query or agent response")
            return {}

        tasks = [
            _run_single_eval(
                name,
                criterion,
                user_query,
                agent_response,
                full_summary if name in _CRITERIA_NEEDING_FULL_RESULTS else calls_only,
                trace_id,
                memory_context=memory_context,
            )
            for name, criterion in EVAL_CRITERIA.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(get_client().flush)

        scores: dict[str, float] = {}
        for r in results:
            if isinstance(r, tuple) and r[1] is not None:
                scores[r[0]] = r[1]
        return scores

    except Exception as e:
        logger.warning(f"Evaluation pipeline failed: {e}")
        return {}


def should_store_memory(scores: dict[str, float]) -> bool:
    """Check whether eval scores are good enough to store the interaction in memory.

    Returns True (safe to store) when:
    - No scores available (eval disabled or failed -- don't block memory)
    - All gated criteria (hallucination, correctness, completeness) score 1.0
    """
    if not scores:
        return True
    for criterion in _MEMORY_GATE_CRITERIA:
        value = scores.get(criterion)
        if value is not None and value < _MEMORY_GATE_THRESHOLD:
            logger.info(f"Memory gate: BLOCKED ({criterion}={value:.1f})")
            return False
    return True
