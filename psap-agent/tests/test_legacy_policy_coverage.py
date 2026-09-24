"""Regression coverage for legacy policy rules split into curated skills.

These assertions intentionally check policy concepts, rather than the old
monolithic prompt's wording, so prompt slimming cannot silently drop a required
workflow safeguard.
"""

from psap_agent.src.core.curated_skills import load_curated_skill
from psap_agent.src.core.prompt import get_system_prompt


def _normalize(text: str) -> str:
    """Make policy assertions insensitive to Markdown line wrapping."""
    return " ".join(text.split())


def test_global_policy_rules_remain_in_the_stable_prompt():
    prompt = _normalize(get_system_prompt())

    for required_rule in (
        "tool result in this conversation supports it",
        "Every URL in a response must be copied exactly from a tool result",
        "internal analysis and guided customer-facing support",
        "equivalent hardware, not proxy data",
        "Format statistics as bullet lists",
        "load `clarification_and_comparison` before selecting data or tools",
        "load `vllm_performance_triage`",
    ):
        assert required_rule in prompt


def test_curated_skills_cover_the_migrated_legacy_workflow_rules():
    expected_rules = {
        "clarification_and_comparison": (
            "Every clarification question must use options returned by",
            "Never silently substitute a different accelerator",
            "arbitrary single concurrency point",
            "`tp_mismatch`",
        ),
        "benchmark_analysis": (
            "`compare_versions_comprehensive`",
            "Do not query two configurations separately",
            "`cv_conc` only with actual concurrency values",
            "follow-up request to show, graph, or visualize",
        ),
        "cost_analysis": (
            "Before calling `calculate_cost_efficiency`, ask for ITL P95",
            "Do not silently apply those defaults",
            "eight-GPU instance",
        ),
        "grafana_metrics": (
            "`grafana_available`",
            "GuideLLM timestamps",
            "`summary` field",
        ),
        "pytorch_profiles": (
            "`analyze_performance_insights`",
            "`compare_trace_structures`",
            "`get_kernel_call_stacks`",
        ),
        "deep_profiling": (
            "`compare_vllm_logs` after profiling",
            "single-request profile paradox",
            "Quantitative accounting using only actual profiling deltas",
        ),
        "kernel_source_analysis": (
            "`get_kernel_categories`",
            "`correlate_kernel_with_changes`",
            "`get_vllm_pull_request`",
        ),
        "vllm_logs": (
            "`pinned_dependencies`",
            "Do not infer these details from kernel names",
        ),
        "vllm_performance_triage": (
            "`get_vllm_performance_triage_guide` first",
            "primary source of truth",
        ),
    }

    for skill_name, rules in expected_rules.items():
        skill = _normalize(load_curated_skill(skill_name))
        for rule in rules:
            assert _normalize(rule) in skill, f"{skill_name} lost: {rule}"


def test_curated_benchmark_workflow_does_not_reintroduce_a_nonexistent_tool():
    """A stale legacy reference must not cause failed tool calls."""
    benchmark_skill = load_curated_skill("benchmark_analysis")

    assert "get_pareto_optimal" not in benchmark_skill
