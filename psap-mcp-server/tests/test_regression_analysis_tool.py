"""Tests for version-to-version regression analysis."""

import pandas as pd
import pytest

from psap_mcp_server.src.tools import regression_analysis_tool


def _benchmark_data(prompt_tokens: int, output_tokens: int) -> pd.DataFrame:
    rows = []
    for version, throughput, ttft in (
        ("RHAIIS-3.6-fast2", 100.0, 1.0),
        ("vLLM-0.28.0", 80.0, 1.2),
    ):
        rows.append(
            {
                "version": version,
                "model": "openai/gpt-oss-120b",
                "accelerator": "H200",
                "TP": 1,
                "prompt toks": prompt_tokens,
                "output toks": output_tokens,
                "output_tok/sec": throughput,
                "ttft_median": ttft,
                "itl_median": 0.1,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "token_count,profile",
    [
        (1000, "1k/1k"),
        (1000, "1000/1000"),
    ],
)
@pytest.mark.asyncio
async def test_profile_filter_uses_decimal_k_convention(
    monkeypatch, token_count, profile
):
    data = _benchmark_data(token_count, token_count)
    monkeypatch.setattr(
        regression_analysis_tool, "load_rhaiis_data", lambda: data.copy()
    )

    result = await regression_analysis_tool.analyze_regression(
        "RHAIIS-3.6-fast2",
        "vLLM-0.28.0",
        accelerator="H200",
        model="openai/gpt-oss-120b",
        profile=profile,
    )

    assert result["status"] == "success"
    assert result["summary"]["total_configurations_compared"] == 1


@pytest.mark.asyncio
async def test_profile_filter_does_not_treat_k_as_1024(monkeypatch):
    data = _benchmark_data(1024, 1024)
    monkeypatch.setattr(
        regression_analysis_tool, "load_rhaiis_data", lambda: data.copy()
    )

    result = await regression_analysis_tool.analyze_regression(
        "RHAIIS-3.6-fast2",
        "vLLM-0.28.0",
        accelerator="H200",
        model="openai/gpt-oss-120b",
        profile="1k/1k",
    )

    assert result["status"] == "error"
    assert result["error"] == "No matching benchmark data"
    assert "0 matching records" in result["message"]


@pytest.mark.asyncio
async def test_empty_profile_filter_returns_clear_error(monkeypatch):
    data = _benchmark_data(8000, 800)
    monkeypatch.setattr(
        regression_analysis_tool, "load_rhaiis_data", lambda: data.copy()
    )

    result = await regression_analysis_tool.analyze_regression(
        "RHAIIS-3.6-fast2",
        "vLLM-0.28.0",
        accelerator="H200",
        model="openai/gpt-oss-120b",
        profile="1k/1k",
    )

    assert result["status"] == "error"
    assert result["error"] == "No matching benchmark data"
    assert (
        "0 matching records for RHAIIS-3.6-fast2 and 0 for vLLM-0.28.0"
        in result["message"]
    )
