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


def _profile_index(model_key=MODEL_KEY):
    accelerator, bare_model = model_key.split("/", 1)
    return {
        model_key: {
            TP: {
                BROKEN_VERSION: {
                    WORKLOAD: {
                        0: {
                            "source": "s3",
                            "key": (
                                f"pytorch-profiles/rhaiis/{accelerator}/"
                                f"{bare_model}/{TP}/{BROKEN_VERSION}/{WORKLOAD}/"
                                "trace_rank0.json"
                            ),
                            "filename": "trace_rank0.json",
                            "size_bytes": 114_000_000,
                        }
                    }
                },
                VALID_VERSION: {
                    WORKLOAD: {
                        0: {
                            "source": "s3",
                            "key": (
                                f"pytorch-profiles/rhaiis/{accelerator}/"
                                f"{bare_model}/{TP}/{VALID_VERSION}/{WORKLOAD}/"
                                "trace_rank0.json"
                            ),
                            "filename": "trace_rank0.json",
                            "size_bytes": 128,
                        }
                    }
                },
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


def _configure_fake_s3(monkeypatch, *, valid_versions=(), model_key=MODEL_KEY):
    index = _profile_index(model_key)
    payloads = {
        entry["key"]: _valid_trace_bytes()
        if version in valid_versions
        else b'{"traceEvents":[{"name":"truncated","ph":"X","dur":"unterminated'
        for version, workloads in index[model_key][TP].items()
        for ranks in workloads.values()
        for entry in ranks.values()
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
