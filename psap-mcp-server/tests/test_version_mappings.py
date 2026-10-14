"""Tests for dynamically loaded RHAIIS-to-vLLM mappings."""

import asyncio

import httpx
import pytest

from psap_mcp_server.src.tools import vllm_release_notes_tool as mappings_tool


@pytest.fixture(autouse=True)
def reset_mapping_cache():
    """Keep module-level cache state isolated between tests."""
    mappings_tool._version_mappings_cache = {}
    mappings_tool._version_mappings_expires_at = 0.0
    mappings_tool._version_mappings_etag = None
    mappings_tool._version_mappings_last_modified = None
    mappings_tool._version_mappings_source = "bundled fallback"
    mappings_tool._version_mappings_lock = asyncio.Lock()


@pytest.mark.asyncio
async def test_remote_mappings_are_cached_and_used_for_resolution(monkeypatch):
    fetch_calls = 0

    async def fetch_remote():
        nonlocal fetch_calls
        fetch_calls += 1
        return {"RHAIIS-3.6": "v0.25.0"}, '"mapping-etag"', None

    monkeypatch.setattr(mappings_tool, "_fetch_remote_version_mappings", fetch_remote)

    mappings = await mappings_tool._load_version_mappings()
    resolved = await mappings_tool._resolve_to_vllm_version("rhaiis-3.6")
    mappings["RHAIIS-3.6"] = "mutated-copy"
    cached_mappings = await mappings_tool._load_version_mappings()

    assert mappings_tool._version_mappings_source == "GitHub"
    assert resolved == ("v0.25.0", "RHAIIS-3.6", True)
    assert cached_mappings == {"RHAIIS-3.6": "v0.25.0"}
    assert fetch_calls == 1


@pytest.mark.asyncio
async def test_remote_fetch_uses_configured_url_and_http_validators(monkeypatch):
    requested = []

    class Response:
        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self.payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    responses = [
        Response(
            200,
            {"rhaiis_to_vllm": {"RHAIIS-3.6": "v0.25.0"}},
            {
                "ETag": '"mapping-etag"',
                "Last-Modified": "Wed, 14 Oct 2026 17:00:00 GMT",
            },
        ),
        Response(304),
    ]

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 5.0
            assert kwargs["follow_redirects"] is True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers):
            requested.append((url, headers))
            return responses.pop(0)

    mapping_url = "https://raw.example.test/mappings.json"
    monkeypatch.setattr(
        mappings_tool.settings, "RHAIIS_VERSION_MAPPINGS_URL", mapping_url
    )
    monkeypatch.setattr(mappings_tool.httpx, "AsyncClient", FakeAsyncClient)

    mapping, etag, last_modified = await mappings_tool._fetch_remote_version_mappings()
    mappings_tool._version_mappings_etag = etag
    mappings_tool._version_mappings_last_modified = last_modified
    not_modified = await mappings_tool._fetch_remote_version_mappings()

    assert mapping == {"RHAIIS-3.6": "v0.25.0"}
    assert not_modified == (None, etag, last_modified)
    assert requested[0][0] == mapping_url
    assert requested[1][1]["If-None-Match"] == etag
    assert requested[1][1]["If-Modified-Since"] == last_modified


@pytest.mark.asyncio
async def test_bundled_mapping_is_used_when_remote_fetch_fails(monkeypatch):
    bundled = {"RHAIIS-3.5-GA": "v0.24.0"}

    async def fail_fetch():
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(mappings_tool, "_fetch_remote_version_mappings", fail_fetch)
    monkeypatch.setattr(
        mappings_tool, "_load_bundled_version_mappings", lambda: bundled
    )

    loaded = await mappings_tool._load_version_mappings()
    resolved = await mappings_tool._resolve_to_vllm_version("RHAIIS-3.5-GA")

    assert loaded == bundled
    assert resolved == ("v0.24.0", "RHAIIS-3.5-GA", True)
    assert mappings_tool._version_mappings_source == "bundled fallback"


@pytest.mark.asyncio
async def test_last_known_good_mapping_survives_remote_refresh_failure(monkeypatch):
    mappings_tool._version_mappings_cache = {"RHAIIS-3.6": "v0.25.0"}

    async def fail_fetch():
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(mappings_tool, "_fetch_remote_version_mappings", fail_fetch)
    monkeypatch.setattr(
        mappings_tool,
        "_load_bundled_version_mappings",
        lambda: pytest.fail("bundled fallback should not replace cached mappings"),
    )

    loaded = await mappings_tool._load_version_mappings()

    assert loaded == {"RHAIIS-3.6": "v0.25.0"}


def test_mapping_payload_must_have_nonempty_string_pairs():
    assert mappings_tool._validate_version_mappings(
        {"rhaiis_to_vllm": {" RHAIIS-3.6 ": " v0.25.0 "}}
    ) == {"RHAIIS-3.6": "v0.25.0"}

    with pytest.raises(ValueError, match="no mappings"):
        mappings_tool._validate_version_mappings({"rhaiis_to_vllm": {}})

    with pytest.raises(ValueError, match="non-empty string pairs"):
        mappings_tool._validate_version_mappings(
            {"rhaiis_to_vllm": {"RHAIIS-3.6": None}}
        )
