"""Safe handling for malformed profiler trace artifacts."""

import json
from types import SimpleNamespace

import pytest

from psap_mcp_server.src.tools import pytorch_profile_tool, s3_utils

MODEL_KEY = "H200/openai--gpt-oss-120b"
TP = "tp1"
WORKLOAD = "isl8000_osl800"
BROKEN_VERSION = "RHAIIS-3.6-fast1"
VALID_VERSION = "vLLM-0.28.0"


@pytest.fixture(autouse=True)
def reset_trace_caches():
    pytorch_profile_tool._malformed_trace_cache.clear()
    pytorch_profile_tool._stats_cache.clear()
    pytorch_profile_tool._raw_events_cache.clear()
    yield
    pytorch_profile_tool._malformed_trace_cache.clear()
    pytorch_profile_tool._stats_cache.clear()
    pytorch_profile_tool._raw_events_cache.clear()


def _profile_index(model_key=MODEL_KEY, ranks_by_version=None):
    accelerator, bare_model = model_key.split("/", 1)
    if ranks_by_version is None:
        ranks_by_version = {
            BROKEN_VERSION: (0,),
            VALID_VERSION: (0,),
        }
    return {
        model_key: {
            TP: {
                version: {
                    WORKLOAD: {
                        rank: {
                            "source": "s3",
                            "key": (
                                f"pytorch-profiles/rhaiis/{accelerator}/"
                                f"{bare_model}/{TP}/{version}/{WORKLOAD}/"
                                f"trace_rank{rank}.json"
                            ),
                            "filename": f"trace_rank{rank}.json",
                            "size_bytes": 128,
                        }
                        for rank in ranks
                    }
                }
                for version, ranks in ranks_by_version.items()
            }
        }
    }


class _Body:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload


class _FakeS3:
    def __init__(self, payloads):
        self.payloads = payloads
        self.get_object_calls = []

    def get_object(self, *, Bucket, Key):
        self.get_object_calls.append((Bucket, Key))
        return {"Body": _Body(self.payloads[Key])}


def _valid_trace_bytes():
    return json.dumps(
        {
            "traceEvents": [
                {
                    "name": "valid_kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 0,
                    "dur": 100,
                }
            ]
        }
    ).encode()


def _configure_fake_s3(
    monkeypatch,
    *,
    valid_versions=(),
    valid_traces=None,
    model_key=MODEL_KEY,
    ranks_by_version=None,
):
    index = _profile_index(model_key, ranks_by_version)
    payloads = {
        entry["key"]: _valid_trace_bytes()
        if (
            (version, rank) in valid_traces
            if valid_traces is not None
            else version in valid_versions
        )
        else b'{"traceEvents":[{"name":"truncated","ph":"X","dur":"unterminated'
        for version, workloads in index[model_key][TP].items()
        for ranks in workloads.values()
        for rank, entry in ranks.items()
    }
    fake_s3 = _FakeS3(payloads)

    async def discover_profiles(model=None):
        return index

    monkeypatch.setattr(
        pytorch_profile_tool, "settings", SimpleNamespace(S3_BUCKET="test-bucket")
    )
    monkeypatch.setattr(pytorch_profile_tool, "_discover_profiles_async", discover_profiles)
    monkeypatch.setattr(s3_utils, "get_s3_client", lambda: fake_s3)
    return fake_s3


def test_malformed_s3_trace_is_skipped_and_negative_cached(monkeypatch):
    index = _profile_index()
    trace_key = index[MODEL_KEY][TP][BROKEN_VERSION][WORKLOAD][0]["key"]
    fake_s3 = _configure_fake_s3(monkeypatch)

    assert pytorch_profile_tool._load_trace_from_s3(trace_key) is None
    assert pytorch_profile_tool._load_trace_from_s3(trace_key) is None
    assert len(fake_s3.get_object_calls) == 1


@pytest.mark.parametrize(
    ("model_key", "expected_display_name"),
    [
        (MODEL_KEY, "openai--gpt-oss-120b (H200)"),
        (
            "MI355X/openai--gpt-oss-120b",
            "openai--gpt-oss-120b (MI355X)",
        ),
    ],
)
@pytest.mark.asyncio
async def test_profile_comparison_returns_partial_when_one_trace_is_malformed(
    monkeypatch, model_key, expected_display_name
):
    _configure_fake_s3(
        monkeypatch, valid_versions=(VALID_VERSION,), model_key=model_key
    )

    result = await pytorch_profile_tool.compare_pytorch_profiles(
        BROKEN_VERSION,
        VALID_VERSION,
        model=model_key,
        tp=TP,
        workload=WORKLOAD,
    )

    assert result["status"] == "partial"
    assert result["operation_completed"] is False
    assert result["comparison_performed"] is False
    assert len(result["unavailable_traces"]) == 1
    unavailable = result["unavailable_traces"][0]
    assert unavailable["model"] == expected_display_name
    assert unavailable["tp"] == TP
    assert unavailable["version"] == BROKEN_VERSION
    assert unavailable["workload"] == WORKLOAD
    assert unavailable["rank"] == 0
    assert VALID_VERSION in result["available_trace_summaries"]
    assert "do not infer" in result["agent_guidance"].lower()


@pytest.mark.asyncio
async def test_comparison_tools_do_not_raise_for_malformed_trace(monkeypatch):
    _configure_fake_s3(monkeypatch, valid_versions=(VALID_VERSION,))

    structure_result = await pytorch_profile_tool.compare_trace_structures(
        BROKEN_VERSION,
        VALID_VERSION,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )
    insight_result = await pytorch_profile_tool.analyze_performance_insights(
        BROKEN_VERSION,
        VALID_VERSION,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )

    assert structure_result["status"] == "partial"
    assert structure_result["comparison_performed"] is False
    assert insight_result["status"] == "partial"
    assert insight_result["comparison_performed"] is False


@pytest.mark.asyncio
async def test_profile_comparison_is_unavailable_when_both_traces_are_malformed(monkeypatch):
    _configure_fake_s3(monkeypatch)

    result = await pytorch_profile_tool.compare_pytorch_profiles(
        BROKEN_VERSION,
        VALID_VERSION,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )

    assert result["status"] == "unavailable"
    assert result["operation_completed"] is False
    assert result["comparison_performed"] is False
    assert len(result["unavailable_traces"]) == 2


@pytest.mark.parametrize("missing_trace_is_indexed", [True, False])
@pytest.mark.asyncio
async def test_aggregate_comparison_never_compares_different_rank_sets(
    monkeypatch, missing_trace_is_indexed
):
    ranks_by_version = {
        BROKEN_VERSION: (0, 1),
        VALID_VERSION: (0, 1) if missing_trace_is_indexed else (0,),
    }
    _configure_fake_s3(
        monkeypatch,
        ranks_by_version=ranks_by_version,
        valid_traces={
            (BROKEN_VERSION, 0),
            (BROKEN_VERSION, 1),
            (VALID_VERSION, 0),
        },
    )

    result = await pytorch_profile_tool.compare_pytorch_profiles(
        BROKEN_VERSION,
        VALID_VERSION,
        aggregate_ranks=True,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )

    assert result["status"] == "partial"
    assert result["comparison_performed"] is False
    assert "summary" not in result
    assert result["rank_coverage"][BROKEN_VERSION]["loaded_ranks"] == [0, 1]
    assert result["rank_coverage"][VALID_VERSION]["loaded_ranks"] == [0]
    assert result["rank_coverage"][VALID_VERSION]["missing_ranks"] == [1]
    assert result["unavailable_traces"] == [{
        "model": "openai--gpt-oss-120b (H200)",
        "tp": TP,
        "version": VALID_VERSION,
        "workload": WORKLOAD,
        "rank": 1,
    }]


@pytest.mark.asyncio
async def test_aggregate_comparison_requires_all_ranks_even_if_both_sides_skip_the_same_one(
    monkeypatch,
):
    _configure_fake_s3(
        monkeypatch,
        ranks_by_version={
            BROKEN_VERSION: (0, 1),
            VALID_VERSION: (0, 1),
        },
        valid_traces={(BROKEN_VERSION, 0), (VALID_VERSION, 0)},
    )

    result = await pytorch_profile_tool.compare_pytorch_profiles(
        BROKEN_VERSION,
        VALID_VERSION,
        aggregate_ranks=True,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )

    assert result["status"] == "partial"
    assert result["comparison_performed"] is False
    assert "summary" not in result
    assert result["rank_coverage"][BROKEN_VERSION]["missing_ranks"] == [1]
    assert result["rank_coverage"][VALID_VERSION]["missing_ranks"] == [1]


@pytest.mark.asyncio
async def test_aggregate_comparison_succeeds_with_complete_matching_rank_sets(monkeypatch):
    _configure_fake_s3(
        monkeypatch,
        valid_versions=(BROKEN_VERSION, VALID_VERSION),
        ranks_by_version={
            BROKEN_VERSION: (0, 1),
            VALID_VERSION: (0, 1),
        },
    )

    result = await pytorch_profile_tool.compare_pytorch_profiles(
        BROKEN_VERSION,
        VALID_VERSION,
        aggregate_ranks=True,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )

    assert result["status"] == "success"
    assert result["comparison_performed"] is True
    assert result["comparison"]["scope"] == "all 2 ranks"
    assert result["summary"]["total_diff_pct"] == 0


@pytest.mark.asyncio
async def test_single_rank_comparison_ignores_unrequested_malformed_rank(monkeypatch):
    _configure_fake_s3(
        monkeypatch,
        ranks_by_version={
            BROKEN_VERSION: (0, 1),
            VALID_VERSION: (0, 1),
        },
        valid_traces={(BROKEN_VERSION, 0), (VALID_VERSION, 0)},
    )

    result = await pytorch_profile_tool.compare_pytorch_profiles(
        BROKEN_VERSION,
        VALID_VERSION,
        rank=0,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )

    assert result["status"] == "success"
    assert result["comparison_performed"] is True
    assert result["summary"]["total_diff_pct"] == 0


@pytest.mark.asyncio
async def test_single_trace_unavailability_does_not_mention_comparisons(monkeypatch):
    _configure_fake_s3(monkeypatch)

    result = await pytorch_profile_tool.analyze_pytorch_profile(
        BROKEN_VERSION,
        model=MODEL_KEY,
        tp=TP,
        workload=WORKLOAD,
    )

    assert result["status"] == "unavailable"
    assert result["analysis_performed"] is False
    assert "cross-version" not in result["message"]
    assert "comparison" not in result["agent_guidance"]


@pytest.mark.asyncio
async def test_ambiguous_model_match_requires_exact_key_without_guessing(monkeypatch):
    index = {
        **_profile_index("H200/openai--gpt-oss-120b"),
        **_profile_index("MI355X/openai--gpt-oss-120b"),
    }

    async def discover_profiles(model=None):
        return index

    monkeypatch.setattr(
        pytorch_profile_tool, "_discover_profiles_async", discover_profiles
    )

    analysis = await pytorch_profile_tool.analyze_pytorch_profile(
        VALID_VERSION, model="gpt-oss-120b"
    )
    comparison = await pytorch_profile_tool.compare_pytorch_profiles(
        BROKEN_VERSION, VALID_VERSION, model="gpt-oss-120b"
    )

    for result in (analysis, comparison):
        message = result["message"]
        assert result["status"] == "error"
        assert "H200/openai--gpt-oss-120b" in message
        assert "MI355X/openai--gpt-oss-120b" in message
        assert "Choose the exact composite key" in message
        assert "Do not select a candidate based on list order" in message
        assert "profile-listing call is not needed" in message
