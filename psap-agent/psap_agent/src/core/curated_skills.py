"""Registry and safe loader for curated PSAP workflow skills.

Curated skills are reviewed, task-specific instructions. They complement the
learned SkillDocumentStore recipes in ``memory.py``: curated skills explain
how to perform and interpret a workflow, while learned recipes suggest a
previously successful tool sequence.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import BaseTool, tool


@dataclass(frozen=True)
class CuratedSkill:
    """Metadata for one allowlisted workflow skill."""

    name: str
    filename: str
    description: str
    triggers: str


_SKILL_DIRECTORY = Path(__file__).with_name("curated_skill_documents")

_SKILLS: tuple[CuratedSkill, ...] = (
    CuratedSkill(
        name="clarification_and_comparison",
        filename="clarification_and_comparison.md",
        description="Data-backed clarification, configuration selection, and fair-comparison rules.",
        triggers="ambiguous model, version, accelerator, profile, TP, comparison, missing benchmark inputs",
    ),
    CuratedSkill(
        name="benchmark_analysis",
        filename="benchmark_analysis.md",
        description="Benchmark discovery, performance queries, fair comparisons, and regression analysis.",
        triggers="performance metrics, benchmark results, model or version comparisons, regressions, runtime arguments",
    ),
    CuratedSkill(
        name="cost_analysis",
        filename="cost_analysis.md",
        description="Cost-per-million-token analysis with latency SLOs and required reporting.",
        triggers="cost, CPMT, TCO, price, cost efficiency, cheapest configuration",
    ),
    CuratedSkill(
        name="grafana_metrics",
        filename="grafana_metrics.md",
        description="GPU/DCGM and vLLM metrics analysis from Grafana benchmark runs.",
        triggers="Grafana, GPU utilization, temperature, power, KV cache, vLLM metrics",
    ),
    CuratedSkill(
        name="deep_profiling",
        filename="deep_profiling.md",
        description="Evidence-led root-cause investigation of a performance change across versions.",
        triggers="why faster or slower, root cause, deep investigation, explain a regression",
    ),
    CuratedSkill(
        name="pytorch_profiles",
        filename="pytorch_profiles.md",
        description="PyTorch profiler availability, kernel comparison, and trace-structure analysis.",
        triggers="PyTorch profile, profiler trace, kernels, trace structure, CUDA timeline",
    ),
    CuratedSkill(
        name="kernel_source_analysis",
        filename="kernel_source_analysis.md",
        description="Kernel-to-source mapping, vLLM source and diff inspection, release and dependency verification.",
        triggers="kernel mapping, source code, code diff, release notes, dependency pin, Triton or PyTorch version",
    ),
    CuratedSkill(
        name="vllm_logs",
        filename="vllm_logs.md",
        description="vLLM log retrieval and side-by-side configuration comparison.",
        triggers="vLLM logs, startup logs, compare logs, engine configuration",
    ),
    CuratedSkill(
        name="vllm_performance_triage",
        filename="vllm_performance_triage.md",
        description="Structured diagnosis and remediation guidance for vLLM inference performance.",
        triggers="triage vLLM, improve inference performance, optimize latency, saturation, KV cache, remediation",
    ),
)

_SKILLS_BY_NAME = {skill.name: skill for skill in _SKILLS}


def list_curated_skills() -> tuple[CuratedSkill, ...]:
    """Return the fixed, ordered catalog of curated workflow skills."""
    return _SKILLS


def get_curated_skill(name: str) -> CuratedSkill | None:
    """Resolve a skill name from the allowlisted catalog."""
    return _SKILLS_BY_NAME.get(name.strip().lower())


def get_curated_skill_catalog() -> str:
    """Render concise skill metadata for the core system prompt."""
    return "\n".join(
        f"- `{skill.name}` — {skill.description} Use for: {skill.triggers}."
        for skill in _SKILLS
    )


def load_curated_skill(name: str) -> str:
    """Return an allowlisted skill document or a useful, safe error message."""
    skill = get_curated_skill(name)
    if skill is None:
        available = ", ".join(item.name for item in _SKILLS)
        return f"Unknown curated skill: {name!r}. Available skills: {available}."

    try:
        content = (_SKILL_DIRECTORY / skill.filename).read_text(encoding="utf-8").strip()
    except OSError:
        return (
            f"Curated skill '{skill.name}' is unavailable. Continue using the "
            "available tools and global integrity rules."
        )

    if not content:
        return f"Curated skill '{skill.name}' is empty and cannot be used."

    return f"# Loaded curated skill: {skill.name}\n\n{content}"


def create_load_skill_tool() -> BaseTool:
    """Create the local tool that exposes curated instructions to the agent."""

    catalog = "\n".join(
        f"- {skill.name}: {skill.description}"
        for skill in _SKILLS
    )

    @tool(
        "load_skill",
        description=(
            "Load one reviewed workflow skill before a specialized analysis. "
            "Use it only for a multi-step workflow, not for a simple metadata "
            "or availability question. You may load more than one skill when a "
            "workflow spans those domains.\n\nAvailable skills:\n" + catalog
        ),
    )
    def load_skill(skill_name: str) -> str:
        """Load the requested allowlisted workflow skill."""
        return load_curated_skill(skill_name)

    return load_skill
