"""Unit and integration tests for NYSY profile tool.

Tests focus on:
1. CRITICAL: Nanosecond to microsecond conversion (divide by 1000, not multiply)
2. NYSY event parsing and stats extraction
3. Profile discovery (S3 + local)
4. MCP tool function behavior
5. Integration with PyTorch tool patterns
"""

import asyncio
import json
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from psap_mcp_server.src.tools.nysy_profile_tool import (
    _extract_nysy_stats,
    _parse_nysy_trace,
    _format_duration,
    _filter_by_category,
    _get_top_kernels,
    _get_category_breakdown,
    _merge_stats,
    _extract_bare_version,
    _match_version,
    _get_model_info,
    _parse_rank_from_filename,
    analyze_nysy_profile,
    compare_nysy_profiles,
    analyze_nysy_performance_insights,
    NYSY_CATEGORY_MAPPING,
    CATEGORY_GROUPS,
)


class TestNYSYConversion:
    """Test critical nanosecond to microsecond conversion."""

    def test_nanosecond_to_microsecond_conversion_basic(self):
        """CRITICAL: Verify ns→µs conversion divides by 1000, not multiplies."""
        events = [
            {
                "name": "test_kernel",
                "duration_ns": 100_000,  # 100 microseconds in nanoseconds
                "type": "kernel"
            }
        ]
        stats = _extract_nysy_stats(events)

        # Should be 100 microseconds, NOT 100,000,000
        assert stats["test_kernel"]["total_dur"] == 100.0, \
            "Conversion should divide by 1000, not multiply!"
        assert stats["test_kernel"]["avg_dur"] == 100.0

    def test_nanosecond_to_microsecond_conversion_1ms(self):
        """Test conversion for 1 millisecond (1,000,000 ns)."""
        events = [
            {
                "name": "kernel",
                "duration_ns": 1_000_000,  # 1 millisecond
                "type": "kernel"
            }
        ]
        stats = _extract_nysy_stats(events)

        # Should be 1000 microseconds (1 ms)
        assert stats["kernel"]["total_dur"] == 1000.0
        assert stats["kernel"]["avg_dur"] == 1000.0

    def test_nanosecond_to_microsecond_conversion_nanoseconds(self):
        """Test conversion for very small durations (nanoseconds)."""
        events = [
            {
                "name": "fast_kernel",
                "duration_ns": 1000,  # 1 microsecond
                "type": "kernel"
            }
        ]
        stats = _extract_nysy_stats(events)

        # Should be 1 microsecond
        assert stats["fast_kernel"]["total_dur"] == 1.0
        assert stats["fast_kernel"]["avg_dur"] == 1.0

    def test_multiple_events_conversion(self):
        """Test conversion with multiple events of same kernel."""
        events = [
            {"name": "kernel_a", "duration_ns": 100_000, "type": "kernel"},
            {"name": "kernel_a", "duration_ns": 150_000, "type": "kernel"},
            {"name": "kernel_a", "duration_ns": 50_000, "type": "kernel"},
        ]
        stats = _extract_nysy_stats(events)

        # Total: (100 + 150 + 50) = 300 microseconds
        assert stats["kernel_a"]["count"] == 3
        assert stats["kernel_a"]["total_dur"] == 300.0, \
            "Sum should be 300 microseconds, not 300,000,000"
        assert stats["kernel_a"]["avg_dur"] == 100.0
        assert stats["kernel_a"]["min_dur"] == 50.0
        assert stats["kernel_a"]["max_dur"] == 150.0

    def test_category_mapping(self):
        """Test NYSY category to canonical category mapping."""
        events = [
            {"name": "kernel1", "duration_ns": 100_000, "type": "kernel"},
            {"name": "nccl_allreduce", "duration_ns": 50_000, "type": "nccl"},
            {"name": "cuda_func", "duration_ns": 75_000, "type": "cuda_runtime"},
        ]
        stats = _extract_nysy_stats(events)

        assert stats["kernel1"]["cat"] == "kernel"
        assert stats["nccl_allreduce"]["cat"] == "nccl"
        assert stats["cuda_func"]["cat"] == "cuda_runtime"


class TestNYSYStatsExtraction:
    """Test stats extraction from NYSY events."""

    def test_extract_stats_aggregation(self):
        """Test stats aggregation from multiple events."""
        events = [
            {"name": "kernel_a", "duration_ns": 50_000, "type": "kernel"},
            {"name": "kernel_a", "duration_ns": 150_000, "type": "kernel"},
            {"name": "kernel_b", "duration_ns": 200_000, "type": "kernel"},
        ]
        stats = _extract_nysy_stats(events)

        # Kernel A
        assert stats["kernel_a"]["count"] == 2
        assert stats["kernel_a"]["total_dur"] == 200.0  # (50 + 150) µs
        assert stats["kernel_a"]["avg_dur"] == 100.0

        # Kernel B
        assert stats["kernel_b"]["count"] == 1
        assert stats["kernel_b"]["total_dur"] == 200.0
        assert stats["kernel_b"]["avg_dur"] == 200.0

    def test_zero_duration_events_ignored(self):
        """Test that zero-duration events are ignored."""
        events = [
            {"name": "kernel_a", "duration_ns": 100_000, "type": "kernel"},
            {"name": "kernel_b", "duration_ns": 0, "type": "kernel"},  # Should be ignored
        ]
        stats = _extract_nysy_stats(events)

        assert "kernel_a" in stats
        assert "kernel_b" not in stats  # Zero duration should not be in stats

    def test_missing_fields_handled(self):
        """Test that missing fields are handled gracefully."""
        events = [
            {"name": "kernel_a", "duration_ns": 100_000},  # Missing type
            {"duration_ns": 50_000, "type": "kernel"},  # Missing name
        ]
        stats = _extract_nysy_stats(events)

        assert stats["kernel_a"]["total_dur"] == 100.0
        assert stats["unknown"]["total_dur"] == 50.0


class TestSharedUtilities:
    """Test shared utility functions."""

    def test_format_duration_seconds(self):
        """Test duration formatting in seconds."""
        assert _format_duration(2_000_000) == "2.00s"

    def test_format_duration_milliseconds(self):
        """Test duration formatting in milliseconds."""
        assert _format_duration(5_000) == "5.00ms"

    def test_format_duration_microseconds(self):
        """Test duration formatting in microseconds."""
        assert _format_duration(100) == "100.00µs"

    def test_extract_bare_version(self):
        """Test version extraction."""
        assert _extract_bare_version("vLLM-0.20.0") == "0.20.0"
        assert _extract_bare_version("v0.20.0") == "0.20.0"
        assert _extract_bare_version("0.20.0") == "0.20.0"
        assert _extract_bare_version("vllm 0.20.0") == "0.20.0"

    def test_match_version(self):
        """Test version matching."""
        versions = ["vLLM-0.19.0", "vLLM-0.20.0", "vLLM-0.21.0"]
        assert _match_version("0.20.0", versions) == "vLLM-0.20.0"
        assert _match_version("v0.19.0", versions) == "vLLM-0.19.0"
        assert _match_version("0.22.0", versions) is None

    def test_parse_rank_from_filename(self):
        """Test rank extraction from filename."""
        assert _parse_rank_from_filename("rank0.nsys-rep") == 0
        assert _parse_rank_from_filename("rank3.nsys-rep") == 3
        assert _parse_rank_from_filename("nsys_rank7_profile.nsys-rep") == 7
        assert _parse_rank_from_filename("trace_1000_1100_0_20260229_120000.json") == 0

    def test_filter_by_category(self):
        """Test category filtering."""
        stats = {
            "kernel1": {"total_dur": 100, "cat": "kernel"},
            "nccl_op": {"total_dur": 50, "cat": "nccl"},
            "cuda_func": {"total_dur": 75, "cat": "cuda_runtime"},
        }

        # Filter by kernel
        filtered = _filter_by_category(stats, "kernel")
        assert len(filtered) == 1
        assert "kernel1" in filtered

        # Filter by communication (should include nccl)
        filtered = _filter_by_category(stats, "communication")
        assert "nccl_op" in filtered

        # Filter by all
        filtered = _filter_by_category(stats, "all")
        assert len(filtered) == 3

        # Filter by None (should return all)
        filtered = _filter_by_category(stats, None)
        assert len(filtered) == 3

    def test_get_top_kernels(self):
        """Test top kernels selection."""
        stats = {
            "kernel_a": {"total_dur": 500, "count": 2, "avg_dur": 250, "min_dur": 200, "max_dur": 300, "cat": "kernel"},
            "kernel_b": {"total_dur": 300, "count": 3, "avg_dur": 100, "min_dur": 50, "max_dur": 150, "cat": "kernel"},
            "kernel_c": {"total_dur": 100, "count": 1, "avg_dur": 100, "min_dur": 100, "max_dur": 100, "cat": "kernel"},
        }

        top = _get_top_kernels(stats, n=2)
        assert len(top) == 2
        assert top[0]["name"] == "kernel_a"
        assert top[1]["name"] == "kernel_b"

        # Check fields
        assert top[0]["total_dur_us"] == 500
        assert top[0]["count"] == 2

    def test_get_category_breakdown(self):
        """Test category breakdown."""
        stats = {
            "kernel1": {"total_dur": 300, "count": 5, "cat": "kernel"},
            "kernel2": {"total_dur": 200, "count": 3, "cat": "kernel"},
            "nccl_op": {"total_dur": 100, "count": 2, "cat": "nccl"},
        }

        breakdown = _get_category_breakdown(stats)

        assert "kernel" in breakdown
        assert breakdown["kernel"]["total_dur_us"] == 500
        assert breakdown["kernel"]["invocation_count"] == 8
        assert breakdown["kernel"]["unique_operations"] == 2

        assert "nccl" in breakdown
        assert breakdown["nccl"]["total_dur_us"] == 100
        assert breakdown["nccl"]["invocation_count"] == 2

    def test_merge_stats(self):
        """Test stats merging from multiple ranks."""
        stats1 = {
            "kernel_a": {"count": 2, "total_dur": 100, "min_dur": 40, "max_dur": 60, "cat": "kernel"},
            "kernel_b": {"count": 1, "total_dur": 50, "min_dur": 50, "max_dur": 50, "cat": "kernel"},
        }
        stats2 = {
            "kernel_a": {"count": 3, "total_dur": 150, "min_dur": 45, "max_dur": 55, "cat": "kernel"},
            "kernel_c": {"count": 1, "total_dur": 30, "min_dur": 30, "max_dur": 30, "cat": "kernel"},
        }

        merged = _merge_stats([stats1, stats2])

        # Kernel A: merged from both
        assert merged["kernel_a"]["count"] == 5
        assert merged["kernel_a"]["total_dur"] == 250
        assert merged["kernel_a"]["min_dur"] == 40
        assert merged["kernel_a"]["max_dur"] == 60

        # Kernel B: only in stats1
        assert merged["kernel_b"]["count"] == 1
        assert merged["kernel_b"]["total_dur"] == 50

        # Kernel C: only in stats2
        assert merged["kernel_c"]["count"] == 1
        assert merged["kernel_c"]["total_dur"] == 30

    def test_get_model_info(self):
        """Test model info building."""
        info = _get_model_info("H200/nemotron-120b", 2)

        assert info["display_name"] == "Nemotron-120B"
        assert info["gpus"] == "H200"
        assert info["ranks_stored"] == 2


class TestNYSYTraceParsing:
    """Test trace parsing and format handling."""

    def test_parse_json_trace(self):
        """Test parsing of JSON trace data."""
        trace_data = {
            "events": [
                {"name": "kernel_a", "duration_ns": 100_000, "type": "kernel"},
            ]
        }

        result = _parse_nysy_trace(trace_data)
        assert result is not None
        assert result == trace_data

    def test_parse_invalid_trace(self):
        """Test handling of invalid trace data."""
        result = _parse_nysy_trace(None)
        assert result is None

        result = _parse_nysy_trace(123)
        assert result is None


# ===================================================================== #
#  Integration Tests with mocked S3 and file system                     #
# ===================================================================== #

@pytest.mark.asyncio
class TestNYSYToolsIntegration:
    """Integration tests for MCP tool functions."""

    @pytest.mark.asyncio
    @patch("psap_mcp_server.src.tools.nysy_profile_tool._discover_nysy_profiles")
    @patch("psap_mcp_server.src.tools.nysy_profile_tool._load_or_extract_nysy_stats")
    async def test_analyze_nysy_profile_success(self, mock_load, mock_discover):
        """Test successful NYSY profile analysis."""
        # Mock discovery
        mock_discover.return_value = {
            "H200/nemotron-120b": {
                "vLLM-0.20.0": {
                    0: {"source": "local", "path": Path("/fake/path.nsys-rep")}
                }
            }
        }

        # Mock stats loading
        mock_load.return_value = {
            "kernel_a": {
                "count": 10,
                "total_dur": 500.0,
                "min_dur": 40.0,
                "max_dur": 60.0,
                "avg_dur": 50.0,
                "cat": "kernel",
            },
            "kernel_b": {
                "count": 5,
                "total_dur": 200.0,
                "min_dur": 30.0,
                "max_dur": 50.0,
                "avg_dur": 40.0,
                "cat": "kernel",
            },
        }

        result = await analyze_nysy_profile(
            version="v0.20.0",
            model="nemotron-120b",
            rank=0,
            aggregate_ranks=False,
            top_n=10,
        )

        assert result["status"] == "success"
        assert result["version"] == "vLLM-0.20.0"
        assert result["model"] == "Nemotron-120B (H200)"
        assert len(result["top_kernels"]) == 2
        assert result["top_kernels"][0]["name"] == "kernel_a"
        assert result["summary"]["total_traced_time_us"] == 700.0

    @pytest.mark.asyncio
    @patch("psap_mcp_server.src.tools.nysy_profile_tool._discover_nysy_profiles")
    async def test_analyze_nysy_profile_not_found(self, mock_discover):
        """Test error when profile not found."""
        mock_discover.return_value = {}

        result = await analyze_nysy_profile(
            version="v0.20.0",
            model="nonexistent",
        )

        assert result["status"] == "error"
        assert "profile data available" in result["message"].lower() or "not found" in result["message"].lower()

    @pytest.mark.asyncio
    @patch("psap_mcp_server.src.tools.nysy_profile_tool._discover_nysy_profiles")
    @patch("psap_mcp_server.src.tools.nysy_profile_tool._load_or_extract_nysy_stats")
    async def test_compare_nysy_profiles(self, mock_load, mock_discover):
        """Test NYSY profile comparison."""
        mock_discover.return_value = {
            "H200/nemotron-120b": {
                "vLLM-0.19.0": {0: {"source": "local"}},
                "vLLM-0.20.0": {0: {"source": "local"}},
            }
        }

        stats_old = {
            "kernel_a": {
                "count": 10,
                "total_dur": 600.0,
                "min_dur": 50.0,
                "max_dur": 70.0,
                "avg_dur": 60.0,
                "cat": "kernel",
            }
        }

        stats_new = {
            "kernel_a": {
                "count": 10,
                "total_dur": 500.0,
                "min_dur": 40.0,
                "max_dur": 60.0,
                "avg_dur": 50.0,
                "cat": "kernel",
            }
        }

        def load_side_effect(model, version, rank, force):
            if "0.19.0" in version:
                return stats_old
            return stats_new

        mock_load.side_effect = load_side_effect

        result = await compare_nysy_profiles(
            version1="v0.19.0",
            version2="v0.20.0",
            model="nemotron-120b",
        )

        assert result["status"] == "success"
        assert result["version1"] == "vLLM-0.19.0"
        assert result["version2"] == "vLLM-0.20.0"
        assert result["summary"]["delta_pct"] < 0  # Improvement

    @pytest.mark.asyncio
    @patch("psap_mcp_server.src.tools.nysy_profile_tool.analyze_nysy_profile")
    async def test_analyze_nysy_performance_insights(self, mock_analyze):
        """Test performance insights generation."""
        mock_analyze.return_value = {
            "status": "success",
            "version": "vLLM-0.20.0",
            "model": "Nemotron-120B (H200)",
            "summary": {"total_traced_time_us": 1000.0},
            "top_kernels": [
                {
                    "name": "fused_moe_kernel",
                    "total_dur_us": 600.0,
                    "count": 50,
                    "category": "kernel",
                },
                {
                    "name": "attention_kernel",
                    "total_dur_us": 300.0,
                    "count": 40,
                    "category": "kernel",
                },
            ],
            "category_breakdown": {
                "kernel": {
                    "total_dur_us": 1000.0,
                    "unique_operations": 2,
                }
            },
        }

        result = await analyze_nysy_performance_insights(
            version="v0.20.0",
            model="nemotron-120b",
        )

        assert result["status"] == "success"
        assert len(result["insights"]) > 0
        assert result["model"] == "Nemotron-120B (H200)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
