"""Evidence integrity checks for selecting PyTorch captures."""

import json

import pytest

from psap_mcp_server.src.tools import pytorch_profile_tool as profiles

MODEL = "H200/deepseek-ai--DeepSeek-V4-Flash"
TP = "tp4"
BASELINE = "vLLM-0.28.0"
CURRENT = "RHAIIS-3.6-fast2"
WORKLOAD = "isl8000_osl1000"
OTHER_WORKLOAD = "isl8000_osl800"
COMPARISONS = [
    profiles.compare_pytorch_profiles,
    profiles.compare_trace_structures,
    profiles.analyze_performance_insights,
]


@pytest.fixture
def traces(monkeypatch, tmp_path):
    async def discover(model=None):
        return profiles._discover_profiles_local(str(tmp_path))

    def write(version, workload, filename=None, *, duration=100):
        filename = filename or f"trace_rank0_pid123_run{workload}_range500-503.json"
        path = tmp_path / MODEL / TP / version / workload / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "traceEvents": [
                {"name": "gpu_kernel", "cat": "kernel", "ph": "X", "ts": 0, "dur": duration},
                {"name": "cudaGraphLaunch", "cat": "cuda_runtime", "ph": "X", "ts": 0, "dur": 3},
            ],
        }))
        return path.relative_to(tmp_path).as_posix()

    monkeypatch.setattr(profiles, "_discover_profiles_async", discover)
    return write


@pytest.mark.asyncio
async def test_duplicate_rank_zero_requires_exact_capture_and_preserves_cache_identity(traces):
    first = traces(BASELINE, WORKLOAD)
    second = traces(
        BASELINE, WORKLOAD,
        f"trace_rank0_pid456_run{WORKLOAD}_range500-503.json",
        duration=200,
    )
    current = traces(CURRENT, WORKLOAD)
    args = {"model": MODEL, "tp": TP, "workload": WORKLOAD}

    for aggregate in (False, True):
        result = await profiles.compare_pytorch_profiles(
            BASELINE, CURRENT, aggregate_ranks=aggregate, **args
        )
        assert result["error_type"] == "ambiguous_trace"
        assert result["comparison_performed"] is False
        assert {t["trace_id"] for t in result["candidates"]} == {first, second}
        assert {t["rank"] for t in result["candidates"]} == {0}

    missing = await profiles.analyze_pytorch_profile(BASELINE, rank=1, **args)
    assert missing["status"] == "error"

    for selected_id, expected_us in ((first, 100), (second, 200)):
        analysis = await profiles.compare_pytorch_profiles(
            BASELINE, CURRENT, baseline_trace_id=selected_id, current_trace_id=current,
            category="kernel", **args,
        )
        pairing = analysis["trace_pairing"]
        assert pairing["status"] == "matched"
        assert pairing["baseline"][0]["trace_id"] == selected_id
        assert pairing["baseline"][0]["benchmark_run_uuid"] is None
        assert pairing["baseline"][0]["capture_steps"] == [500, 503]
        assert pairing["current"][0]["trace_id"] == current
        assert analysis["summary"]["total_time1_us"] == expected_us


@pytest.mark.parametrize("compare", COMPARISONS)
@pytest.mark.asyncio
async def test_cross_workload_id_cannot_override_workload(traces, compare):
    traces(BASELINE, WORKLOAD)
    traces(CURRENT, WORKLOAD)
    other = traces(CURRENT, OTHER_WORKLOAD)
    result = await compare(
        BASELINE, CURRENT, model=MODEL, tp=TP, workload=WORKLOAD, current_trace_id=other
    )
    assert result["error_type"] == "trace_id_mismatch"
    assert result["comparison_performed"] is False
    assert result["candidates"][0]["workload"] == WORKLOAD


@pytest.mark.parametrize("compare", COMPARISONS)
@pytest.mark.asyncio
async def test_reject_trace_stored_under_wrong_workload(traces, compare):
    traces(BASELINE, WORKLOAD)
    traces(CURRENT, WORKLOAD, f"trace_rank0_pid123_run{OTHER_WORKLOAD}_range500-503.json")
    result = await compare(BASELINE, CURRENT, model=MODEL, tp=TP, workload=WORKLOAD)
    assert result["error_type"] == "trace_workload_mismatch"
    assert result["comparison_performed"] is False
    assert result["trace"]["filename_workload"] == OTHER_WORKLOAD


@pytest.mark.parametrize("compare", COMPARISONS)
@pytest.mark.asyncio
async def test_reject_different_capture_step_windows(traces, compare):
    traces(BASELINE, WORKLOAD)
    traces(CURRENT, WORKLOAD, f"trace_rank0_pid123_run{WORKLOAD}_range700-703.json")
    result = await compare(BASELINE, CURRENT, model=MODEL, tp=TP, workload=WORKLOAD)
    assert result["error_type"] == "trace_capture_mismatch"
    assert result["comparison_performed"] is False


@pytest.mark.asyncio
async def test_legacy_trace_remains_readable_without_causal_verdict(traces):
    traces(BASELINE, WORKLOAD, "trace_rank0_pid123.json")
    traces(CURRENT, WORKLOAD)
    result = await profiles.analyze_performance_insights(
        BASELINE, CURRENT, model=MODEL, tp=TP, workload=WORKLOAD
    )
    assert result["status"] == "success"
    assert result["trace_pairing"]["status"] == "unverified"
    assert result["overall_verdict"] == "UNVERIFIED_TRACE_PAIRING"
    assert result["conclusions"] == []
    assert result["kernel_launch_analysis"]["graph_launch_api_calls"] == {"v1": 1, "v2": 1}
