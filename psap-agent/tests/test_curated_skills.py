"""Tests for curated workflow-skill discovery and loading."""

from psap_agent.src.core.curated_skills import (
    create_load_skill_tool,
    get_curated_skill_catalog,
    list_curated_skills,
    load_curated_skill,
)


class TestCuratedSkills:
    """Ensure the agent only loads reviewed, available skill documents."""

    def test_catalog_lists_every_registered_skill(self):
        skills = list_curated_skills()

        assert len(skills) >= 9
        catalog = get_curated_skill_catalog()
        for skill in skills:
            assert skill.name in catalog

        assert "clarification_and_comparison" in catalog
        assert "vllm_performance_triage" in catalog
        assert "downstream PRs" in catalog
        assert "cherry-picks" in catalog

    def test_loads_an_allowlisted_skill(self):
        content = load_curated_skill("cost_analysis")

        assert "Loaded curated skill: cost_analysis" in content
        assert "calculate_cost_efficiency" in content

    def test_rejects_unknown_skill_without_reading_a_path(self):
        content = load_curated_skill("../../etc/passwd")

        assert "Unknown curated skill" in content
        assert "benchmark_analysis" in content

    def test_tool_returns_skill_document(self):
        skill_tool = create_load_skill_tool()

        content = skill_tool.invoke({"skill_name": "grafana_metrics"})

        assert "Loaded curated skill: grafana_metrics" in content
        assert "query_grafana_metrics" in content

    def test_tool_loads_multiple_skills_in_one_call(self):
        skill_tool = create_load_skill_tool()

        content = skill_tool.invoke(
            {
                "skill_names": [
                    "deep_profiling",
                    "pytorch_profiles",
                    "vllm_logs",
                ]
            }
        )

        assert "Loaded curated skill: deep_profiling" in content
        assert "Loaded curated skill: pytorch_profiles" in content
        assert "Loaded curated skill: vllm_logs" in content

    def test_tool_deduplicates_skills_and_bounds_batch_size(self):
        skill_tool = create_load_skill_tool()

        deduplicated = skill_tool.invoke(
            {"skill_names": ["benchmark_analysis", " BENCHMARK_ANALYSIS "]}
        )
        assert deduplicated.count("Loaded curated skill: benchmark_analysis") == 1

        too_many = skill_tool.invoke(
            {
                "skill_names": [
                    "deep_profiling",
                    "pytorch_profiles",
                    "vllm_logs",
                    "benchmark_analysis",
                    "kernel_source_analysis",
                ]
            }
        )
        assert "Load at most 4 skills per call" in too_many

    def test_tool_rejects_batches_that_would_be_truncated(self):
        skill_tool = create_load_skill_tool()

        content = skill_tool.invoke(
            {
                "skill_names": [
                    "kernel_source_analysis",
                    "deep_profiling",
                    "benchmark_analysis",
                    "clarification_and_comparison",
                ]
            }
        )

        assert "exceed the combined output limit" in content
        assert "no workflow instructions are shortened" in content
