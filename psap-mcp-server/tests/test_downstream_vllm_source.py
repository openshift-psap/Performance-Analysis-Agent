"""Tests for private downstream vLLM source discovery and access."""

import base64

import pytest

from psap_mcp_server.src.tools import kernel_code_mapper_tool as source_tool
from psap_mcp_server.src.tools import vllm_log_tool
from psap_mcp_server.src.tools import vllm_release_notes_tool as mappings_tool

DOWNSTREAM_REPO = "example-org/downstream-vllm"


class FakeResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self.payload = payload
        self.text = text

    def json(self):
        return self.payload


class FakeAsyncClient:
    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, headers=None, params=None):
        self.calls.append((url, headers, params))
        return self.responses[url]


@pytest.mark.asyncio
async def test_downstream_tag_discovery_resolves_mapping_and_does_not_auto_select(
    monkeypatch,
):
    base = f"https://api.github.com/repos/{DOWNSTREAM_REPO}"
    matching_url = f"{base}/git/matching-refs/tags/v0.28.0"
    tag_url = f"{base}/git/tags/cpu-tag-object"
    commit_url = f"{base}/commits/b300-commit"
    calls = []
    responses = {
        f"{base}/git/ref/tags/v0.28.0%2Brhai.v3": FakeResponse(404, {}),
        matching_url: FakeResponse(
            200,
            [
                {
                    "ref": "refs/tags/v0.28.0+rhaiv.5.cpu",
                    "object": {"type": "tag", "sha": "cpu-tag-object"},
                },
                {
                    "ref": "refs/tags/v0.28.0+rhaiv.7",
                    "object": {"type": "commit", "sha": "b300-commit"},
                },
                {
                    "ref": "refs/tags/v0.27.0+rhaiv.99",
                    "object": {"type": "commit", "sha": "unrelated"},
                },
            ],
        ),
        tag_url: FakeResponse(
            200,
            {
                "message": "RHAIIS 3.6-fast2 CPU",
                "object": {"sha": "cpu-commit"},
                "tagger": {"date": "2026-10-20T12:00:00Z"},
            },
        ),
        commit_url: FakeResponse(
            200,
            {
                "commit": {
                    "message": "B300 CUDA architecture fix",
                    "committer": {"date": "2026-10-22T12:00:00Z"},
                }
            },
        ),
    }

    async def resolve_version(version):
        assert version == "RHAIIS-3.6-fast2"
        return "v0.28.0", version, True

    monkeypatch.setattr(mappings_tool, "_resolve_to_vllm_version", resolve_version)
    monkeypatch.setattr(
        source_tool.settings, "RHAIIS_DOWNSTREAM_VLLM_REPO", DOWNSTREAM_REPO
    )
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    result = await source_tool.get_downstream_vllm_tags(
        "RHAIIS-3.6-fast2",
        platform="cpu",
        runtime_version="0.28.0+rhai.v3",
    )

    assert result["status"] == "success"
    assert result["repo"] == DOWNSTREAM_REPO
    assert result["upstream_version"] == "v0.28.0"
    assert result["mapped_upstream_version"] == "v0.28.0"
    assert result["runtime_version"] == "0.28.0+rhai.v3"
    assert result["exact_runtime_match"] is False
    assert result["resolution_method"] == "rhaiis_mapping_fallback"
    assert result["used_mapping_fallback"] is True
    assert result["mapping_matches_runtime_base"] is True
    assert result["requires_exact_ref"] is True
    assert result["count"] == result["total_candidates"] == 2
    assert result["truncated"] is False
    assert "selected_ref" not in result
    by_ref = {candidate["ref"]: candidate for candidate in result["candidates"]}
    assert by_ref["v0.28.0+rhaiv.5.cpu"]["platform_match"] is True
    assert by_ref["v0.28.0+rhaiv.5.cpu"]["commit_sha"] == "cpu-commit"
    assert by_ref["v0.28.0+rhaiv.7"]["platform_hints"] == ["b300", "cuda"]
    assert by_ref["v0.28.0+rhaiv.7"]["platform_match"] is False
    assert all(
        call[2] == {"per_page": 100} for call in calls if call[0] == matching_url
    )


@pytest.mark.asyncio
async def test_runtime_log_version_exactly_identifies_downstream_tag(monkeypatch):
    base = f"https://api.github.com/repos/{DOWNSTREAM_REPO}"
    exact_ref_url = f"{base}/git/ref/tags/v0.28.0%2Brhai.v3"
    exact_commit_url = f"{base}/commits/exact-runtime-commit"
    calls = []
    responses = {
        exact_ref_url: FakeResponse(
            200,
            {
                "ref": "refs/tags/v0.28.0+rhai.v3",
                "object": {"type": "commit", "sha": "exact-runtime-commit"},
            },
        ),
        exact_commit_url: FakeResponse(
            200,
            {
                "commit": {
                    "message": "RHAIIS runtime build 3",
                    "committer": {"date": "2026-10-22T12:00:00Z"},
                }
            },
        ),
    }

    async def resolve_version(version):
        assert version == "RHAIIS-3.6-fast2"
        return "v0.28.0", version, True

    monkeypatch.setattr(mappings_tool, "_resolve_to_vllm_version", resolve_version)
    monkeypatch.setattr(
        source_tool.settings, "RHAIIS_DOWNSTREAM_VLLM_REPO", DOWNSTREAM_REPO
    )
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    result = await source_tool.get_downstream_vllm_tags(
        "RHAIIS-3.6-fast2",
        runtime_version="0.28.0+rhai.v3",
        max_tags=1,
    )

    assert result["status"] == "success"
    assert result["exact_runtime_match"] is True
    assert result["resolution_method"] == "exact_runtime_tag"
    assert result["used_mapping_fallback"] is False
    assert result["mapping_matches_runtime_base"] is True
    assert result["runtime_tag_match"]["ref"] == "v0.28.0+rhai.v3"
    # Exact log matches need not enumerate unrelated tag candidates.
    assert result["candidates"][0]["ref"] == "v0.28.0+rhai.v3"
    assert result["requires_exact_ref"] is False


@pytest.mark.asyncio
async def test_rhaiis_mapping_fallback_is_used_when_log_base_differs(monkeypatch):
    base = f"https://api.github.com/repos/{DOWNSTREAM_REPO}"
    mapped_refs_url = f"{base}/git/matching-refs/tags/v0.28.0"
    mapped_commit_url = f"{base}/commits/mapped-candidate"
    calls = []
    responses = {
        f"{base}/git/ref/tags/v0.27.0%2Brhai.v3": FakeResponse(404, {}),
        mapped_refs_url: FakeResponse(
            200,
            [
                {
                    "ref": "refs/tags/v0.28.0+rhaiv.5",
                    "object": {"type": "commit", "sha": "mapped-candidate"},
                }
            ],
        ),
        mapped_commit_url: FakeResponse(
            200,
            {"commit": {"message": "mapped candidate", "committer": {}}},
        ),
    }

    async def resolve_version(version):
        assert version == "RHAIIS-3.6-fast2"
        return "v0.28.0", version, True

    monkeypatch.setattr(mappings_tool, "_resolve_to_vllm_version", resolve_version)
    monkeypatch.setattr(
        source_tool.settings, "RHAIIS_DOWNSTREAM_VLLM_REPO", DOWNSTREAM_REPO
    )
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    result = await source_tool.get_downstream_vllm_tags(
        "RHAIIS-3.6-fast2", runtime_version="0.27.0+rhai.v3"
    )

    assert result["status"] == "success"
    assert result["resolution_method"] == "rhaiis_mapping_fallback"
    assert result["exact_runtime_match"] is False
    assert result["used_mapping_fallback"] is True
    assert result["runtime_version_base"] == "v0.27.0"
    assert result["mapped_upstream_version"] == "v0.28.0"
    assert result["upstream_version"] == "v0.28.0"
    assert result["mapping_matches_runtime_base"] is False
    assert [tag["ref"] for tag in result["candidates"]] == ["v0.28.0+rhaiv.5"]
    assert any(call[0] == mapped_refs_url for call in calls)


@pytest.mark.asyncio
async def test_fetch_vllm_logs_cross_checks_server_version(monkeypatch):
    raw_log = (
        "(APIServer pid=1) INFO 09-30 01:42:32 "
        "[api_utils.py:333] version 0.28.0+rhai.v3"
    )
    monkeypatch.setattr(
        vllm_log_tool, "_find_uuid_for_run", lambda *args: ("run-uuid", None)
    )
    monkeypatch.setattr(
        vllm_log_tool, "_fetch_log_by_uuid", lambda uuid: (raw_log, "logs/run-uuid.log")
    )
    monkeypatch.setattr(
        vllm_log_tool.settings,
        "RHAIIS_DOWNSTREAM_VLLM_REPO",
        DOWNSTREAM_REPO,
    )
    observed = {}

    async def resolve_downstream(**kwargs):
        observed.update(kwargs)
        return {"status": "success", "exact_runtime_match": True}

    monkeypatch.setattr(source_tool, "get_downstream_vllm_tags", resolve_downstream)

    result = await vllm_log_tool.fetch_vllm_logs(
        "RHAIIS-3.6-fast2", model="gpt-oss-120b", accelerator="H200"
    )

    assert result["status"] == "success"
    assert result["parsed"]["server_config"]["vllm_version"] == "0.28.0+rhai.v3"
    assert result["downstream_tag_resolution"]["exact_runtime_match"] is True
    assert observed == {
        "version": "RHAIIS-3.6-fast2",
        "platform": "H200",
        "max_tags": 10,
        "runtime_version": "0.28.0+rhai.v3",
    }


@pytest.mark.asyncio
async def test_downstream_tag_discovery_requires_repo_configuration(monkeypatch):
    monkeypatch.setattr(source_tool.settings, "RHAIIS_DOWNSTREAM_VLLM_REPO", None)

    result = await source_tool.get_downstream_vllm_tags("v0.28.0")

    assert result["status"] == "error"
    assert "RHAIIS_DOWNSTREAM_VLLM_REPO" in result["suggestion"]


def test_unknown_device_platform_is_not_reported_as_mismatch():
    # Specific GPU model names cannot be inferred from a generic CUDA tag.
    assert source_tool._platform_matches("H200", ["cuda"]) is None


def test_kernel_mapping_uses_exact_downstream_repo_and_ref():
    import asyncio

    result = asyncio.run(
        source_tool.map_kernel_to_vllm_code(
            "fused_moe_kernel",
            version="v0.28.0+rhaiv.5.cpu",
            repo=DOWNSTREAM_REPO,
        )
    )

    assert result["status"] == "success"
    assert result["repo"] == DOWNSTREAM_REPO
    assert result["version"] == "v0.28.0+rhaiv.5.cpu"
    assert result["github_urls"]
    assert all(
        f"github.com/{DOWNSTREAM_REPO}/" in item["url"]
        for item in result["github_urls"]
    )
    assert all("v0.28.0%2Brhaiv.5.cpu" in item["url"] for item in result["github_urls"])


@pytest.mark.parametrize(
    "repo",
    [
        "https://github.com/neuralmagic/nm-vllm-ent",
        "owner/../repository",
        "owner/..",
        "owner/repository/extra",
        "../repository",
    ],
)
@pytest.mark.asyncio
async def test_github_source_tools_reject_invalid_repo_before_network(monkeypatch, repo):
    def unexpected_request(**kwargs):
        pytest.fail("Invalid repository must be rejected before making a request")

    monkeypatch.setattr(source_tool.httpx, "AsyncClient", unexpected_request)

    results = [
        await source_tool.map_kernel_to_vllm_code(
            "fused_moe_kernel", repo=repo, search_github=True
        ),
        await source_tool.fetch_vllm_source("README.md", "main", repo=repo),
        await source_tool.get_vllm_code_diff("base", "head", repo=repo),
        await source_tool.correlate_kernel_with_changes(
            "fused_moe_kernel", "base", "head", repo=repo
        ),
    ]

    assert all(result["status"] == "error" for result in results)
    assert all("owner/repository" in result["message"] for result in results)


@pytest.mark.asyncio
async def test_fetch_source_keeps_downstream_ref_and_repo(monkeypatch):
    repo = DOWNSTREAM_REPO
    path = "vllm/model_executor/layers/fused_moe/fused_moe.py"
    ref = "v0.28.0+rhaiv.5.cpu"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    calls = []
    content = "# downstream implementation\n"
    encoded = base64.b64encode(content.encode()).decode()
    response = FakeResponse(
        200,
        {"encoding": "base64", "content": encoded, "size": len(content)},
    )
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient({url: response}, calls),
    )

    result = await source_tool.fetch_vllm_source(path, ref, repo=repo)

    assert result["status"] == "success"
    assert result["repo"] == repo
    assert result["version"] == ref
    assert result["content"] == content
    assert result["github_url"].endswith(f"/blob/v0.28.0%2Brhaiv.5.cpu/{path}")
    assert calls[0][2] == {"ref": ref}


@pytest.mark.asyncio
async def test_downstream_diff_compares_exact_refs_in_selected_repo(monkeypatch):
    repo = DOWNSTREAM_REPO
    from_ref = "v0.28.0+rhaiv.3.cpu"
    to_ref = "v0.28.0+rhaiv.5.cpu"
    url = (
        f"https://api.github.com/repos/{repo}/compare/"
        "v0.28.0%2Brhaiv.3.cpu...v0.28.0%2Brhaiv.5.cpu"
    )
    calls = []
    response = FakeResponse(
        200,
        {
            "commits": [{"sha": "commit"}],
            "files": [
                {
                    "filename": "vllm/model_executor/layers/fused_moe/fused_moe.py",
                    "status": "modified",
                    "additions": 1,
                    "deletions": 1,
                    "changes": 2,
                    "patch": "@@ -1 +1 @@\n-old\n+new",
                }
            ],
        },
    )
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient({url: response}, calls),
    )

    result = await source_tool.get_vllm_code_diff(from_ref, to_ref, repo=repo)

    assert result["status"] == "success"
    assert result["repo"] == repo
    assert result["version_range"] == {"from": from_ref, "to": to_ref}
    assert result["github_compare_url"] == (
        f"https://github.com/{DOWNSTREAM_REPO}/compare/"
        "v0.28.0%2Brhaiv.3.cpu...v0.28.0%2Brhaiv.5.cpu"
    )
    assert calls[0][0] == url


@pytest.mark.asyncio
async def test_same_repo_diff_can_return_bounded_commits_without_patch_text(monkeypatch):
    repo = DOWNSTREAM_REPO
    from_ref = "v0.28.0+rhaiv.3.cpu"
    to_ref = "v0.28.0+rhaiv.5.cpu"
    url = (
        f"https://api.github.com/repos/{repo}/compare/"
        "v0.28.0%2Brhaiv.3.cpu...v0.28.0%2Brhaiv.5.cpu"
    )
    response = FakeResponse(
        200,
        {
            "total_commits": 3,
            "commits": [
                {
                    "sha": "older",
                    "html_url": "https://github.com/example/commit/older",
                    "commit": {"message": "Older change", "author": {"name": "A"}},
                },
                {
                    "sha": "middle",
                    "html_url": "https://github.com/example/commit/middle",
                    "commit": {
                        "message": "Middle change",
                        "author": {"name": "B"},
                    },
                },
                {
                    "sha": "newer",
                    "html_url": "https://github.com/example/commit/newer",
                    "commit": {
                        "message": "Newest change\nbody",
                        "author": {"name": "C", "date": "2026-10-28T12:00:00Z"},
                    },
                },
            ],
            "files": [
                {
                    "filename": "vllm/core/scheduler.py",
                    "status": "modified",
                    "additions": 1,
                    "deletions": 1,
                    "changes": 2,
                    "patch": "@@ -1 +1 @@\n-old\n+new",
                }
            ],
        },
    )
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient({url: response}, []),
    )

    result = await source_tool.get_vllm_code_diff(
        from_ref,
        to_ref,
        repo=repo,
        include_commits=True,
        include_patches=False,
        max_commits=2,
    )

    assert result["status"] == "success"
    assert result["commits_returned"] == 2
    assert result["commits_truncated"] is True
    assert result["commits"][0]["sha"] == "newer"
    assert result["commits"][0]["author"] == "C"
    assert result["commits"][1]["sha"] == "middle"
    assert result["files"][0]["patch"] is None
    assert result["files"][0]["patch_included"] is False


@pytest.mark.asyncio
async def test_same_repo_diff_reports_unlisted_file_without_calling_it_an_error(
    monkeypatch,
):
    repo = DOWNSTREAM_REPO
    from_ref = "v0.28.0+rhaiv.3.cpu"
    to_ref = "v0.28.0+rhaiv.5.cpu"
    url = (
        f"https://api.github.com/repos/{repo}/compare/"
        "v0.28.0%2Brhaiv.3.cpu...v0.28.0%2Brhaiv.5.cpu"
    )
    calls = []
    response = FakeResponse(
        200,
        {
            "commits": [{"sha": "commit"}],
            "files": [
                {
                    "filename": "vllm/v1/core/scheduler.py",
                    "status": "modified",
                    "additions": 1,
                    "deletions": 1,
                    "changes": 2,
                    "patch": "@@ -1 +1 @@\n-old\n+new",
                }
            ],
        },
    )
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient({url: response}, calls),
    )

    result = await source_tool.get_vllm_code_diff(
        from_ref,
        to_ref,
        file_path="vllm/v1/core/engine/core.py",
        repo=repo,
    )

    assert result["status"] == "success"
    assert result["comparison_status"] == "completed"
    assert result["file_changed"] is False
    assert result["file_change_status"] == "not_in_changed_files"
    assert result["files"] == []
    assert result["files_returned"] == 0
    assert "may be unchanged or the path may not exist" in result["message"]
    assert result["version_range"] == {"from": from_ref, "to": to_ref}
    assert calls[0][0] == url
