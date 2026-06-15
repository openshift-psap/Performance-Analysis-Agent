"""Reflection and tool-output verification for the PSAP agent.

Adds a critic node that reviews agent responses before they reach the user,
checking for hallucinated URLs, unsupported claims, and tool output validity.
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import get_client

from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)


def _content_to_str(content: Any) -> str:
    """Normalize message content to a plain string.

    Gemini can return content as a list of dicts (e.g. [{"type":"text","text":"..."}]).
    """
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

CRITIC_SYSTEM_PROMPT = """You are a quality-control critic for a performance analysis AI agent.
Your job is to review the agent's response BEFORE it reaches the user.

The agent has TWO valid data sources:
- Tool CALL/RESULT pairs from the current interaction
- Memory Context from past interactions (if provided below)
Data from EITHER source is valid and NOT fabrication.

CONFLICT RESOLUTION: When a live tool result EXPLICITLY contradicts memory context (e.g. tool says "model not found" but memory says it exists), the live tool result takes precedence. Memory context can become stale. Do NOT force the agent to override a clear tool result with potentially outdated memory. The agent should acknowledge both (e.g. "previously available but not currently listed") rather than asserting the memory-based claim as current fact.

CRITICAL DISTINCTION — General knowledge vs. data claims:
- **General knowledge** (e.g. "what is Nvidia?", "what does vLLM do?", "explain CUDA") does NOT require tool evidence. Well-known facts about companies, technologies, and concepts are valid without tool calls.
- **Data-specific claims** (e.g. metrics, benchmark numbers, model performance, URLs, configuration details) MUST come from tool outputs or memory context.
Only flag claims as unsupported when they are data-specific and lack tool/memory evidence.

Check for these issues:
1. HALLUCINATED URLS: Any URL in the response that was NOT returned by a tool call. The agent has a zero-URL-fabrication policy.
2. UNSUPPORTED DATA CLAIMS: Specific metrics, benchmark numbers, or dataset-specific facts that don't come from tool outputs OR memory context. General knowledge about technologies, companies, or concepts is NOT an unsupported claim.
3. MISSING TOOL EVIDENCE: Data-specific claims that should be backed by tool calls or memory but aren't. Do NOT require tool evidence for general knowledge.
4. INCOMPLETE RESPONSE: Parts of the user's question that weren't addressed.
5. INTERNAL CONTRADICTIONS: Numbers or claims within the response that contradict each other. For example, a sub-component duration that exceeds the stated total, or percentage breakdowns that don't add up. Cross-check all tables and quantitative claims against each other.
6. FABRICATED VALUES IN CODE BLOCKS: When the response includes reconstructed commands (in code blocks), verify that specific numeric values, paths, and endpoints are traceable to tool outputs or memory context. Common argument NAMES (e.g. --model, --tensor-parallel-size) can be inferred from context, but specific VALUES (e.g. a number of prompts, a port, an API endpoint path) must appear somewhere in the tool data. If a specific number in a code block does not appear in any tool output, flag it.

IMPORTANT: Asking the user for clarification IS a valid and complete response when:
- The user's query is ambiguous (e.g. "what's the performance?" without specifying which model/profile/config)
- Memory context contains data but it's unclear whether it matches what the user is currently asking about
- Multiple interpretations of the query exist
Do NOT flag clarifying questions as "incomplete" just because memory context has potentially relevant data.

Respond with EXACTLY this JSON format:
{
  "verdict": "pass" or "revise",
  "issues": ["list of specific issues found, empty if pass"],
  "revision_instructions": "specific instructions for the agent to fix the issues, empty string if pass"
}

CRITICAL: Do NOT use your own training knowledge to judge factual accuracy. Your training data may be outdated. You can ONLY evaluate whether the agent's claims are supported by tool outputs and memory context provided below. NEVER claim something "does not exist" or "is incorrect" based on your own knowledge — you may simply be out of date.

If the response is good (no hallucinations, claims are tool-backed or memory-backed, question fully answered), verdict MUST be "pass".
Only use "revise" for genuine quality problems, not style preferences."""


_VERSION_QUERY_PATTERNS = re.compile(
    r"(?:what(?:'s| is| are).*(?:new|feature|change|release|introduce|update).*(?:v?\d+\.\d+|vllm|rhaiis))"
    r"|(?:(?:v?\d+\.\d+).*(?:feature|change|release|note))",
    re.IGNORECASE,
)

_TABLE_ROW_PATTERN = re.compile(
    r"\|\s*\*{0,2}(?P<label>[^|*]+?)\*{0,2}\s*\|"
    r"[^|]*?"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*(?P<unit>µs|us|ms|s)"
)

_UNIT_TO_US = {"s": 1_000_000, "ms": 1_000, "us": 1, "µs": 1}


def _normalize_to_us(value: float, unit: str) -> float:
    """Convert a duration value to microseconds."""
    return value * _UNIT_TO_US.get(unit.lower().replace("µ", "u"), 1)


def _check_numerical_consistency(response: str) -> list[str]:
    """Check for internal contradictions in numerical claims.

    Detects cases where a sub-component duration exceeds a stated total,
    or percentage breakdowns that are clearly impossible.
    """
    issues: list[str] = []

    # Extract all duration values from tables
    table_rows: list[tuple[str, float, str]] = []
    for match in _TABLE_ROW_PATTERN.finditer(response):
        label = match.group("label").strip().lower()
        value_str = match.group("value").replace(",", "")
        unit = match.group("unit")
        try:
            value = float(value_str)
            table_rows.append((label, value, unit))
        except ValueError:
            continue

    # Look for "total" rows and check that components don't exceed them
    total_keywords = ("total", "wall time", "block wall", "overall")
    component_keywords = ("idle", "active", "overhead", "gap")

    totals_us: list[tuple[str, float]] = []
    components_us: list[tuple[str, float]] = []

    for label, value, unit in table_rows:
        value_us = _normalize_to_us(value, unit)
        if any(kw in label for kw in total_keywords):
            totals_us.append((label, value_us))
        elif any(kw in label for kw in component_keywords):
            components_us.append((label, value_us))

    for total_label, total_val in totals_us:
        for comp_label, comp_val in components_us:
            if comp_val > total_val * 1.1:
                issues.append(
                    f"INCONSISTENCY: '{comp_label}' ({comp_val:.0f}µs) exceeds "
                    f"stated '{total_label}' ({total_val:.0f}µs)"
                )

    # Check percentage breakdowns that claim to sum to 100%
    pct_pattern = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*%\s*(?:of\s+regression|of\s+total)",
        re.IGNORECASE,
    )
    pct_values = [float(m.group("value")) for m in pct_pattern.finditer(response)]
    if len(pct_values) >= 2:
        pct_sum = sum(pct_values)
        if pct_sum > 120:
            issues.append(
                f"INCONSISTENCY: percentage breakdown sums to {pct_sum:.0f}% "
                f"(values: {pct_values}), exceeds plausible 100%"
            )

    return issues


def verify_tool_outputs(messages: list[BaseMessage], user_query: str = "") -> list[str]:
    """Programmatic verification of tool outputs.

    Checks:
    - URLs in the final response are traceable to tool outputs
    - No-tool-call responses for questions that require tool evidence
    """
    issues = []
    tool_output_urls = set()
    tool_output_content = ""
    has_tool_calls = False

    for msg in messages:
        if getattr(msg, "type", "") == "tool":
            has_tool_calls = True
            content = str(getattr(msg, "content", ""))
            tool_output_content += content + "\n"
            urls = re.findall(r'https?://[^\s"\'<>]+', content)
            tool_output_urls.update(urls)

    final_ai_content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "content", ""):
            final_ai_content = _content_to_str(msg.content)
            break

    if not final_ai_content:
        return issues

    # Check: version-specific question answered without any tool calls
    if not has_tool_calls and user_query and _VERSION_QUERY_PATTERNS.search(user_query):
        if len(final_ai_content) > 200:
            issues.append(
                "NO TOOL CALLS for a version-specific question. You MUST call "
                "get_vllm_release_notes or compare_vllm_versions to answer questions "
                "about version features or changes. Do NOT answer from your own knowledge."
            )

    response_urls = re.findall(r'https?://[^\s"\'<>)]+', final_ai_content)
    for url in response_urls:
        url_clean = url.rstrip(".,;:!?)")
        if not any(url_clean in tool_url or tool_url in url_clean for tool_url in tool_output_urls):
            issues.append(f"URL not from tool output: {url_clean[:100]}")

    # Check: internal numerical contradictions
    consistency_issues = _check_numerical_consistency(final_ai_content)
    issues.extend(consistency_issues)

    return issues


async def run_critic(
    messages: list[BaseMessage],
    user_query: str,
    memory_context: str = "",
    trace_id: str = "",
) -> dict[str, Any]:
    """Run the critic on the agent's response.

    Returns:
        Dict with 'verdict' ('pass' or 'revise'), 'issues', and 'revision_instructions'.
    """
    programmatic_issues = verify_tool_outputs(messages, user_query=user_query)

    pending_calls: dict[str, str] = {}
    tool_exchanges: list[str] = []
    final_response = ""
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                final_response = _content_to_str(msg.content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    call_id = tc.get("id", "")
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    call_desc = f"{name}({json.dumps(args, default=str)[:200]})"
                    pending_calls[call_id] = call_desc
        elif getattr(msg, "type", "") == "tool":
            call_id = getattr(msg, "tool_call_id", "")
            call_desc = pending_calls.pop(call_id, getattr(msg, "name", "tool"))
            result_text = str(getattr(msg, "content", ""))[:30000]
            tool_exchanges.append(f"CALL: {call_desc}\nRESULT: {result_text}")

    if not final_response:
        return {"verdict": "pass", "issues": [], "revision_instructions": ""}

    tool_summary = "\n\n".join(tool_exchanges) if tool_exchanges else "No tool calls made."
    critic_prompt = f"User Query: {user_query}\n\n"
    if memory_context:
        critic_prompt += (
            f"Memory Context (from past interactions -- the agent is allowed to reference this):\n"
            f"{memory_context}\n\n"
        )
    critic_prompt += (
        f"Tool Calls and Results:\n{tool_summary}\n\n"
        f"Agent Response:\n{final_response}\n\n"
    )

    if programmatic_issues:
        critic_prompt += (
            "PROGRAMMATIC CHECKS FOUND THESE ISSUES:\n"
            + "\n".join(f"- {issue}" for issue in programmatic_issues)
            + "\n\nThese are CONFIRMED issues. Include them in your verdict.\n"
        )

    langfuse_gen = None
    if trace_id:
        try:
            client = get_client()
            langfuse_gen = client.start_generation(
                name="critic",
                model=settings.CRITIC_MODEL,
                input=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT[:200] + "..."},
                    {"role": "user", "content": critic_prompt},
                ],
                metadata={"run_id": trace_id, "component": "reflection", "programmatic_issues": programmatic_issues},
            )
        except Exception as e:
            logger.warning(f"Langfuse critic tracing failed (non-fatal): {e}")

    try:
        critic_model = ChatGoogleGenerativeAI(
            model=settings.CRITIC_MODEL,
            temperature=0.0,
        )

        result = await critic_model.ainvoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=critic_prompt),
            ]
        )

        response_text = _content_to_str(result.content).strip()
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(response_text)
        verdict = parsed.get("verdict", "pass")
        issues = parsed.get("issues", [])
        revision_instructions = parsed.get("revision_instructions", "")

        if programmatic_issues and verdict == "pass":
            verdict = "revise"
            issues.extend(programmatic_issues)
            revision_instructions = (
                f"Fix these programmatic issues: {'; '.join(programmatic_issues)}. "
                + revision_instructions
            )

        usage_meta = getattr(result, "usage_metadata", None)
        token_info = ""
        usage_details = {}
        if usage_meta:
            inp = usage_meta.get("input_tokens", 0)
            out = usage_meta.get("output_tokens", 0)
            usage_details = {"input": inp, "output": out}
            token_info = f" [tokens: {inp} in / {out} out]"

        if langfuse_gen:
            try:
                langfuse_gen.update(
                    output=parsed,
                    usage_details=usage_details,
                    metadata={"verdict": verdict, "issue_count": len(issues)},
                )
                langfuse_gen.end()
            except Exception:
                pass

        logger.info(f"Critic verdict: {verdict}, issues: {len(issues)}{token_info}")
        return {
            "verdict": verdict,
            "issues": issues,
            "revision_instructions": revision_instructions,
        }

    except Exception as e:
        if langfuse_gen:
            try:
                langfuse_gen.update(output={"error": str(e)}, level="ERROR")
                langfuse_gen.end()
            except Exception:
                pass
        logger.warning(f"Critic failed, defaulting to pass: {e}")
        if programmatic_issues:
            return {
                "verdict": "revise",
                "issues": programmatic_issues,
                "revision_instructions": f"Fix: {'; '.join(programmatic_issues)}",
            }
        return {"verdict": "pass", "issues": [], "revision_instructions": ""}
