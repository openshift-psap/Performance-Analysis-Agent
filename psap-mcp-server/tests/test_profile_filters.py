"""Regression tests for exact decimal profile filtering across MCP tools."""

import pandas as pd
import pytest

from psap_mcp_server.src.tools import (
    compare_grafana_metrics_tool,
    compare_performance_tool,
    cost_efficiency_tool,
    dataset_metadata_tool,
    discover_configurations_tool,
    energy_efficiency_tool,
    query_performance_tool,
)


def _benchmark_data(prompt_tokens: int = 1024, output_tokens: int = 1024):
    return pd.DataFrame(
        [
            {
                "version": version,
                "model": "example-model",
                "accelerator": "H200",
                "TP": 1,
                "prompt toks": prompt_tokens,
                "output toks": output_tokens,
                "output_tok/sec": throughput,
                "ttft_median": 10.0,
                "ttft_p95": 20.0,
                "itl_median": 1.0,
                "itl_p95": 2.0,
                "intended concurrency": 1,
            }
            for version, throughput in (
                ("vLLM-0.26.0", 100.0),
                ("vLLM-0.28.0", 110.0),
            )
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "function_name", "empty_result_key"),
    [
        (query_performance_tool, "query_performance_metrics", "data"),
        (cost_efficiency_tool, "calculate_cost_efficiency", "cost_analysis"),
        (energy_efficiency_tool, "calculate_energy_efficiency", "energy_analysis"),
    ],
)
async def test_profile_filters_do_not_match_1024_for_one_k(
    monkeypatch, module, function_name, empty_result_key
):
    data = _benchmark_data()
    monkeypatch.setattr(module, "load_rhaiis_data", lambda: data.copy())

    result = await getattr(module, function_name)(
        model="example-model", accelerator="H200", profile="1k/1k"
    )

    assert result[empty_result_key] == []


@pytest.mark.asyncio
async def test_compare_configurations_does_not_match_1024_for_one_k(monkeypatch):
    data = _benchmark_data()
    monkeypatch.setattr(
        compare_performance_tool, "load_rhaiis_data", lambda: data.copy()
    )

    result = await compare_performance_tool.compare_configurations(
        [
            {
                "version": "vLLM-0.26.0",
                "model": "example-model",
                "accelerator": "H200",
                "profile": "1k/1k",
            },
            {
                "version": "vLLM-0.28.0",
                "model": "example-model",
                "accelerator": "H200",
                "profile": "1k/1k",
            },
        ]
    )

    assert result["status"] == "error"
    assert result["message"] == "Not enough valid configurations found for comparison"
    assert all(not configuration["found"] for configuration in result["comparison"])


@pytest.mark.asyncio
async def test_comprehensive_comparison_does_not_match_1024_for_one_k(monkeypatch):
    data = _benchmark_data()
    monkeypatch.setattr(
        compare_performance_tool, "load_rhaiis_data", lambda: data.copy()
    )

    result = await compare_performance_tool.compare_versions_comprehensive(
        "vLLM-0.26.0",
        "vLLM-0.28.0",
        model="example-model",
        accelerator="H200",
        profile="1k/1k",
    )

    assert result["status"] == "error"
    assert result["message"] == "No matching data after applying filters"


@pytest.mark.asyncio
async def test_dataset_metadata_does_not_label_1024_as_one_k(monkeypatch):
    data = _benchmark_data()
    monkeypatch.setattr(dataset_metadata_tool, "load_rhaiis_data", lambda: data)

    result = await dataset_metadata_tool.get_dataset_metadata("profiles")

    assert result["profiles"]["list"][0]["format_shorthand"] == "1024/1024"


@pytest.mark.asyncio
async def test_profile_discovery_does_not_label_1024_as_one_k(monkeypatch):
    data = _benchmark_data()
    monkeypatch.setattr(discover_configurations_tool, "load_rhaiis_data", lambda: data)

    result = await discover_configurations_tool.discover_configurations(
        discover_what="profiles"
    )

    assert result["unique_values"][0]["format_shorthand"] == "1024/1024"


@pytest.mark.asyncio
async def test_configuration_discovery_does_not_match_1024_for_one_k(monkeypatch):
    data = _benchmark_data()
    monkeypatch.setattr(discover_configurations_tool, "load_rhaiis_data", lambda: data)

    result = await discover_configurations_tool.discover_configurations(
        discover_what="models", profile="1k/1k"
    )

    assert result["total_records_matching_filters"] == 0


@pytest.mark.asyncio
async def test_grafana_run_lookup_does_not_match_1024_for_one_k(monkeypatch):
    data = _benchmark_data()
    monkeypatch.setattr(compare_grafana_metrics_tool, "load_rhaiis_data", lambda: data)

    result = await compare_grafana_metrics_tool.compare_grafana_metrics(
        run1_model="example-model",
        run1_version="vLLM-0.26.0",
        run1_accelerator="H200",
        run1_profile="1k/1k",
    )

    assert result["status"] == "error"
    assert "No data found for Run 1" in result["message"]
