"""Large MCP datasets should remain retrievable a page at a time."""

import asyncio

import pandas as pd
import pytest

from psap_mcp_server.src.tools import (
    energy_efficiency_tool,
    pytorch_profile_tool,
    query_performance_tool,
    regression_analysis_tool,
    vllm_log_tool,
    vllm_release_notes_tool,
)


@pytest.mark.asyncio
async def test_performance_query_pages_and_trims_runtime_args(monkeypatch):
    data = pd.DataFrame(
        {
            "model": ["test-model"] * 30,
            "accelerator": ["H200"] * 30,
            "version": ["RHAIIS-3.6"] * 30,
            "TP": [1] * 30,
            "output_tok/sec": list(range(30)),
            "runtime_args": ["x" * 5_000] * 30,
        }
    )
    monkeypatch.setattr(query_performance_tool, "load_rhaiis_data", lambda: data)

    first = await query_performance_tool.query_performance_metrics(top_n=100)
    second = await query_performance_tool.query_performance_metrics(top_n=100, offset=25)
    full = await query_performance_tool.query_performance_metrics(
        top_n=1, include_full_runtime_args=True
    )

    assert first["summary"]["total_results"] == 30
    assert len(first["data"]) == 25
    assert first["summary"]["next_offset"] == 25
    assert len(first["data"][0]["runtime_args"]) == 1_000
    assert first["data"][0]["runtime_args_truncated"] is True
    assert len(second["data"]) == 5
    assert second["summary"]["next_offset"] is None
    assert len(full["data"][0]["runtime_args"]) == 5_000
    assert full["data"][0]["runtime_args_truncated"] is False


@pytest.mark.asyncio
async def test_profile_listing_pages_workloads(monkeypatch):
    workloads = {
        f"isl{i}_osl{i}": {0: {"source": "s3", "filename": f"{i}.json", "size_bytes": 100}}
        for i in range(65)
    }
    index = {"H200/example-model": {"tp1": {"vLLM-0.26.0": workloads}}}

    async def discover(model=None):
        return index

    monkeypatch.setattr(pytorch_profile_tool, "_discover_profiles_async", discover)
    first = await pytorch_profile_tool.list_available_profiles(include_details=True)
    second = await pytorch_profile_tool.list_available_profiles(offset=50, include_details=True)

    assert first["total_profiles"] == 65
    assert first["profiles_shown"] == 50
    assert first["next_offset"] == 50
    assert second["profiles_shown"] == 15
    assert second["next_offset"] is None


@pytest.mark.asyncio
async def test_regression_summary_counts_full_data_but_pages_details(monkeypatch):
    rows = []
    for version, throughput in (("old", 100.0), ("new", 80.0)):
        for i in range(40):
            rows.append({
                "version": version, "model": f"model-{i}", "accelerator": "H200",
                "TP": 1, "prompt toks": 1000, "output toks": 1000,
                "output_tok/sec": throughput, "ttft_median": 1.0, "itl_median": 1.0,
            })
    data = pd.DataFrame(rows)
    monkeypatch.setattr(regression_analysis_tool, "load_rhaiis_data", lambda: data)
    first = await regression_analysis_tool.analyze_regression("old", "new")
    second = await regression_analysis_tool.analyze_regression("old", "new", offset=10)

    assert first["summary"]["regressions_count"] == 40
    assert len(first["regressions"]) == 10
    assert first["pagination"]["next_offsets"]["regressions"] == 10
    assert len(second["regressions"]) == 10


@pytest.mark.asyncio
async def test_vllm_comparison_omits_duplicate_full_sections_by_default(monkeypatch):
    monkeypatch.setattr(vllm_log_tool, "_find_uuid_for_run", lambda version, *args: (version, None))
    monkeypatch.setattr(vllm_log_tool, "_fetch_log_by_uuid", lambda uuid: (uuid, f"{uuid}.log"))
    monkeypatch.setattr(vllm_log_tool, "_parse_vllm_log", lambda raw: {
        "engine_config": {"version": raw}, "warnings_errors_total": 2,
        "warnings_errors": ["warning"],
    })

    async def no_enrichment(*args):
        return None

    monkeypatch.setattr(vllm_log_tool, "_enrich_dependency_versions", no_enrichment)
    compact = await vllm_log_tool.compare_vllm_logs("old", "new")
    detailed = await vllm_log_tool.compare_vllm_logs("old", "new", include_parsed=True)

    assert compact["num_differences"] == 1
    assert compact["warning_counts"] == {"old": 2, "new": 2}
    assert "version1_parsed" not in compact
    assert detailed["version1_parsed"]["engine_config"]["version"] == "old"


def test_vllm_log_fetch_stops_at_startup_marker(monkeypatch):
    from psap_mcp_server.src.tools import s3_utils

    class Body:
        closed = False
        lines_read = 0

        def iter_lines(self, chunk_size):
            for line in (b"vLLM API server version 0.28", b"Application startup complete.", b"should not be read"):
                self.lines_read += 1
                yield line

        def close(self):
            self.closed = True

    body = Body()
    monkeypatch.setattr(s3_utils, "get_s3_client", lambda: type("S3", (), {"get_object": lambda self, **kwargs: {"Body": body}})())

    raw, _ = vllm_log_tool._fetch_log_by_uuid("test-uuid")
    assert "Application startup complete." in raw
    assert "should not be read" not in raw
    assert body.lines_read == 2
    assert body.closed is True


@pytest.mark.asyncio
async def test_release_notes_caps_multi_release_output(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return [{"tag_name": f"v{i}", "body": "x" * 10_000} for i in range(100)]

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            assert kwargs["params"]["per_page"] == 10
            return Response()

    monkeypatch.setattr(vllm_release_notes_tool.httpx, "AsyncClient", Client)
    result = await vllm_release_notes_tool.get_vllm_release_notes(max_releases=100)
    assert len(result["releases"]) == 10
    assert all(len(release["body"]) == 4_000 for release in result["releases"])


@pytest.mark.asyncio
async def test_energy_queries_are_concurrent_but_limited(monkeypatch):
    data = pd.DataFrame([
        {
            "version": "RHAIIS-3.6", "model": f"model-{i}", "accelerator": "H200",
            "TP": 1, "prompt toks": 1000, "output toks": 1000,
            "output_tok/sec": 100.0, "intended concurrency": 1,
            "uuid": "same-run" if i < 2 else f"run-{i}",
        }
        for i in range(8)
    ])
    monkeypatch.setattr(energy_efficiency_tool, "load_rhaiis_data", lambda: data)
    active = 0
    peak = 0
    calls = 0

    async def power(uuid):
        nonlocal active, peak, calls
        active += 1
        peak = max(peak, active)
        calls += 1
        await asyncio.sleep(0.01)
        active -= 1
        return 400.0

    monkeypatch.setattr(energy_efficiency_tool, "_get_gpu_power_from_grafana", power)
    result = await energy_efficiency_tool.calculate_energy_efficiency(top_n=100)

    assert result["status"] == "success"
    assert len(result["energy_analysis"]) == 8
    assert result["statistics"]["grafana_measured_count"] == 8
    assert calls == 7
    assert 1 < peak <= 4
