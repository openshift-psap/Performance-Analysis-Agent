"""Tests for comparing vLLM source across distinct GitHub repositories."""

import base64

import pytest

from psap_mcp_server.src.tools import kernel_code_mapper_tool as source_tool


BASE_REPO = "upstream/vllm"
COMPARISON_REPO = "downstream/vllm"
API = source_tool.GITHUB_API_BASE


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload
        self.text = ""

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


def _blob(content):
    raw = content.encode("utf-8")
    return {
        "encoding": "base64",
        "size": len(raw),
        "content": base64.b64encode(raw).decode("ascii"),
    }


@pytest.mark.asyncio
async def test_cross_repo_diff_reports_added_removed_and_modified_files(monkeypatch):
    base_commit_url = f"{API}/repos/{BASE_REPO}/commits/tags/v0.28.0"
    comparison_commit_url = (
        f"{API}/repos/{COMPARISON_REPO}/commits/tags/v0.28.0%2Brhai.v3"
    )
    base_tree_url = f"{API}/repos/{BASE_REPO}/git/trees/base-tree"
    comparison_tree_url = f"{API}/repos/{COMPARISON_REPO}/git/trees/comparison-tree"
    responses = {
        base_commit_url: FakeResponse(
            200, {"sha": "base-commit", "commit": {"tree": {"sha": "base-tree"}}}
        ),
        comparison_commit_url: FakeResponse(
            200,
            {
                "sha": "comparison-commit",
                "commit": {"tree": {"sha": "comparison-tree"}},
            },
        ),
        base_tree_url: FakeResponse(
            200,
            {
                "truncated": False,
                "tree": [
                    {
                        "path": "vllm/attention/backend.py",
                        "sha": "attention-base",
                        "size": 18,
                        "mode": "100644",
                        "type": "blob",
                    },
                    {
                        "path": "removed.py",
                        "sha": "removed-blob",
                        "size": 4,
                        "mode": "100644",
                        "type": "blob",
                    },
                ],
            },
        ),
        comparison_tree_url: FakeResponse(
            200,
            {
                "truncated": False,
                "tree": [
                    {
                        "path": "vllm/attention/backend.py",
                        "sha": "attention-comparison",
                        "size": 18,
                        "mode": "100644",
                        "type": "blob",
                    },
                    {
                        "path": "added.py",
                        "sha": "added-blob",
                        "size": 8,
                        "mode": "100644",
                        "type": "blob",
                    },
                ],
            },
        ),
        f"{API}/repos/{BASE_REPO}/git/blobs/attention-base": FakeResponse(
            200, _blob("old_value = 1\n")
        ),
        f"{API}/repos/{COMPARISON_REPO}/git/blobs/attention-comparison": FakeResponse(
            200, _blob("new_value = 2\n")
        ),
        f"{API}/repos/{BASE_REPO}/git/blobs/removed-blob": FakeResponse(
            200, _blob("gone\n")
        ),
        f"{API}/repos/{COMPARISON_REPO}/git/blobs/added-blob": FakeResponse(
            200, _blob("arrived\n")
        ),
    }
    calls = []
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    result = await source_tool.get_cross_repo_vllm_diff(
        BASE_REPO, "v0.28.0", COMPARISON_REPO, "v0.28.0+rhai.v3"
    )

    assert result["status"] == "success"
    assert result["base"]["commit_sha"] == "base-commit"
    assert result["comparison"]["commit_sha"] == "comparison-commit"
    assert result["stats"] == {"added": 1, "removed": 1, "modified": 1}
    assert set(result["changed_paths"]) == {
        "vllm/attention/backend.py",
        "added.py",
        "removed.py",
    }
    by_path = {file["filename"]: file for file in result["files"]}
    assert by_path["vllm/attention/backend.py"]["status"] == "modified"
    assert "-old_value = 1" in by_path["vllm/attention/backend.py"]["patch"]
    assert "+new_value = 2" in by_path["vllm/attention/backend.py"]["patch"]
    assert by_path["added.py"]["status"] == "added"
    assert by_path["removed.py"]["status"] == "removed"
    assert any(call[0] == base_tree_url and call[2] == {"recursive": "1"} for call in calls)


@pytest.mark.asyncio
async def test_cross_repo_diff_rejects_truncated_repository_tree(monkeypatch):
    base_commit_url = f"{API}/repos/{BASE_REPO}/commits/tags/base"
    comparison_commit_url = f"{API}/repos/{COMPARISON_REPO}/commits/tags/head"
    base_tree_url = f"{API}/repos/{BASE_REPO}/git/trees/base-tree"
    comparison_tree_url = f"{API}/repos/{COMPARISON_REPO}/git/trees/comparison-tree"
    responses = {
        base_commit_url: FakeResponse(
            200, {"sha": "base-commit", "commit": {"tree": {"sha": "base-tree"}}}
        ),
        comparison_commit_url: FakeResponse(
            200,
            {
                "sha": "comparison-commit",
                "commit": {"tree": {"sha": "comparison-tree"}},
            },
        ),
        base_tree_url: FakeResponse(200, {"truncated": True, "tree": []}),
        comparison_tree_url: FakeResponse(200, {"truncated": False, "tree": []}),
    }
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, []),
    )

    result = await source_tool.get_cross_repo_vllm_diff(
        BASE_REPO, "base", COMPARISON_REPO, "head"
    )

    assert result["status"] == "error"
    assert result["side"] == "base"
    assert "truncated" in result["message"]
    assert "file_path" in result["suggestion"]


@pytest.mark.asyncio
async def test_cross_repo_diff_supports_focused_file_and_missing_file(monkeypatch):
    base_commit_url = f"{API}/repos/{BASE_REPO}/commits/tags/base"
    comparison_commit_url = f"{API}/repos/{COMPARISON_REPO}/commits/tags/head"
    base_tree_url = f"{API}/repos/{BASE_REPO}/git/trees/base-tree"
    comparison_tree_url = f"{API}/repos/{COMPARISON_REPO}/git/trees/comparison-tree"
    responses = {
        base_commit_url: FakeResponse(
            200, {"sha": "base-commit", "commit": {"tree": {"sha": "base-tree"}}}
        ),
        comparison_commit_url: FakeResponse(
            200,
            {
                "sha": "comparison-commit",
                "commit": {"tree": {"sha": "comparison-tree"}},
            },
        ),
        base_tree_url: FakeResponse(
            200,
            {
                "truncated": False,
                "tree": [
                    {
                        "path": "same.py",
                        "sha": "same-blob",
                        "size": 4,
                        "mode": "100644",
                        "type": "blob",
                    }
                ],
            },
        ),
        comparison_tree_url: FakeResponse(
            200,
            {
                "truncated": False,
                "tree": [
                    {
                        "path": "same.py",
                        "sha": "same-blob",
                        "size": 4,
                        "mode": "100644",
                        "type": "blob",
                    }
                ],
            },
        ),
    }
    monkeypatch.setattr(
        source_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, []),
    )

    missing = await source_tool.get_cross_repo_vllm_diff(
        BASE_REPO, "base", COMPARISON_REPO, "head", file_path="missing.py"
    )
    unchanged = await source_tool.get_cross_repo_vllm_diff(
        BASE_REPO, "base", COMPARISON_REPO, "head", file_path="same.py"
    )

    assert missing["status"] == "error"
    assert "not found at either ref" in missing["message"]
    assert unchanged["status"] == "success"
    assert unchanged["total_files_changed"] == 0
    assert unchanged["base"]["commit_sha"] == "base-commit"
    assert unchanged["comparison"]["commit_sha"] == "comparison-commit"
    assert unchanged["file_changed"] is False


@pytest.mark.asyncio
async def test_cross_repo_diff_validates_repository_names_without_network():
    result = await source_tool.get_cross_repo_vllm_diff(
        "https://github.com/upstream/vllm", "base", COMPARISON_REPO, "head"
    )

    assert result["status"] == "error"
    assert "owner/repository" in result["message"]


@pytest.mark.asyncio
async def test_cross_repo_diff_rejects_same_repository_without_network():
    result = await source_tool.get_cross_repo_vllm_diff(
        "NeuralMagic/nm-vllm-ent",
        "de2414062f7ef3da10a086851ac149f43a16b86b",
        "neuralmagic/nm-vllm-ent",
        "273c9a3f467773827be6ed2ddbb780e636557794",
    )

    assert result["status"] == "error"
    assert "same repository" in result["message"]
    assert "get_vllm_code_diff" in result["suggestion"]
