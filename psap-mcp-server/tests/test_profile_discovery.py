"""Tests for asynchronous, cached PyTorch profile discovery."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from psap_mcp_server.src.tools import pytorch_profile_tool, s3_utils


@pytest.fixture(autouse=True)
def reset_profile_index_cache():
    with pytorch_profile_tool._profile_index_lock:
        pytorch_profile_tool._profile_index_cache = None
        pytorch_profile_tool._profile_index_ts = 0.0
        pytorch_profile_tool._profile_model_index_cache.clear()

    yield

    with pytorch_profile_tool._profile_index_lock:
        pytorch_profile_tool._profile_index_cache = None
        pytorch_profile_tool._profile_index_ts = 0.0
        pytorch_profile_tool._profile_model_index_cache.clear()


def test_profile_discovery_uses_ttl_cache_and_honors_forced_refresh(monkeypatch):
    calls = 0
    index = {"H200/example-model": {}}

    def discover_from_s3(model=None):
        nonlocal calls
        calls += 1
        return index

    monkeypatch.setattr(pytorch_profile_tool, "_discover_profiles_s3", discover_from_s3)

    assert pytorch_profile_tool._discover_profiles() is index
    assert pytorch_profile_tool._discover_profiles() is index
    assert calls == 1

    assert pytorch_profile_tool._discover_profiles(force_refresh=True) is index
    assert calls == 2
    assert pytorch_profile_tool._discover_profiles(model="example-model") == index
    assert calls == 2


def test_model_filtered_discovery_uses_separate_cache(monkeypatch):
    calls = []
    filtered_index = {"H200/example-model": {}}
    full_index = {
        "H200/example-model": {},
        "B200/other-model": {},
    }

    def discover_from_s3(model=None):
        calls.append(model)
        return filtered_index if model is not None else full_index

    monkeypatch.setattr(pytorch_profile_tool, "_discover_profiles_s3", discover_from_s3)

    assert (
        pytorch_profile_tool._discover_profiles(model="example-model") == filtered_index
    )
    assert (
        pytorch_profile_tool._discover_profiles(model="example-model") == filtered_index
    )
    assert calls == ["example-model"]
    assert pytorch_profile_tool._profile_index_cache is None

    assert pytorch_profile_tool._discover_profiles() == full_index
    assert calls == ["example-model", None]


@pytest.mark.asyncio
async def test_concurrent_async_discovery_is_offloaded_and_coalesced(monkeypatch):
    calls = 0
    index = {"H200/example-model": {}}

    def slow_discovery(model=None):
        nonlocal calls
        calls += 1
        time.sleep(0.15)
        return index

    monkeypatch.setattr(pytorch_profile_tool, "_discover_profiles_s3", slow_discovery)

    requests = [
        asyncio.create_task(pytorch_profile_tool._discover_profiles_async())
        for _ in range(4)
    ]
    heartbeat = asyncio.create_task(asyncio.sleep(0.01))

    # This must complete while the simulated S3 scan is still running.
    await heartbeat
    assert any(not request.done() for request in requests)

    results = await asyncio.gather(*requests)
    assert results == [index] * 4
    assert calls == 1


def test_s3_discovery_only_descends_into_matching_model(monkeypatch):
    profile_key = (
        "profiles/H200/google--gemma-4-31B-it-FP8-block/tp2/"
        "vLLM-0.20.0/isl1000_osl1000/trace_rank0.json"
    )
    unrelated_key = (
        "profiles/H200/example-unrelated-model/tp2/"
        "vLLM-0.20.0/isl1000_osl1000/trace_rank0.json"
    )
    objects = [profile_key, unrelated_key]
    directory_prefixes = set()
    for key in objects:
        parts = key.split("/")
        for depth in range(1, len(parts)):
            directory_prefixes.add("/".join(parts[:depth]) + "/")

    requested_prefixes = []

    class FakeS3:
        def list_objects_v2(self, *, Bucket, Prefix, Delimiter=None, **kwargs):
            requested_prefixes.append(Prefix)
            if Delimiter == "/":
                children = {
                    prefix[: prefix.find("/", len(Prefix)) + 1]
                    for prefix in directory_prefixes
                    if prefix.startswith(Prefix) and "/" in prefix[len(Prefix) :]
                }
                return {"CommonPrefixes": [{"Prefix": p} for p in sorted(children)]}

            contents = [
                {"Key": key, "Size": 128} for key in objects if key.startswith(Prefix)
            ]
            return {"Contents": contents, "IsTruncated": False}

    monkeypatch.setattr(
        pytorch_profile_tool,
        "settings",
        SimpleNamespace(S3_BUCKET="test-bucket", PROFILE_S3_PREFIX="profiles"),
    )
    monkeypatch.setattr(s3_utils, "get_s3_client", lambda: FakeS3())

    index = pytorch_profile_tool._discover_profiles_s3(model="Gemma-4-31B-it-FP8-block")

    assert index is not None
    assert list(index) == ["H200/google--gemma-4-31B-it-FP8-block"]
    assert any("gemma-4-31B-it-FP8-block/tp2/" in p for p in requested_prefixes)
    assert not any("example-unrelated-model/tp2/" in p for p in requested_prefixes)
