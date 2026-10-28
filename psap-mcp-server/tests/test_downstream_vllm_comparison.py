"""Tests for comparing two downstream RHAIIS releases and PR provenance."""

import httpx
import pytest

from psap_mcp_server.src.tools import downstream_vllm_comparison_tool as compare_tool
from psap_mcp_server.src.tools.kernel_code_mapper_tool import VLLM_REPO

DOWNSTREAM_REPO = "example-org/nm-vllm-ent"


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


def _http_response(url, status, payload):
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("GET", url),
    )


def test_patch_signature_ignores_hunk_offsets_and_context():
    first = {
        "files": [
            {
                "filename": "vllm/core/scheduler.py",
                "patch": "@@ -10,2 +10,2 @@\n context\n-old_value = 1\n+new_value = 2",
            }
        ]
    }
    second = {
        "files": [
            {
                "filename": "vllm/core/scheduler.py",
                "patch": "@@ -90,2 +90,2 @@\n other context\n-old_value = 1\n+new_value = 2",
            }
        ]
    }

    assert compare_tool._patch_signature(first) == compare_tool._patch_signature(second)


def test_patch_signature_is_unavailable_for_incomplete_patch_payload():
    assert (
        compare_tool._patch_signature({"files": [{"filename": "binary.bin"}]}) is None
    )


def test_commit_matcher_separates_exact_trailer_and_patch_evidence():
    upstream = [
        {
            "sha": "a" * 40,
            "message": "Upstream scheduler change",
            "pull_requests": [{"number": 101}],
            "_patch_signature": "patch-a",
        },
        {
            "sha": "b" * 40,
            "message": "Upstream cache change",
            "pull_requests": [{"number": 102}],
            "_patch_signature": "patch-b",
        },
        {
            "sha": "c" * 40,
            "message": "Upstream attention change",
            "pull_requests": [{"number": 103}],
            "_patch_signature": "patch-c",
        },
        {
            "sha": "f" * 40,
            "message": "Unmatched upstream cleanup",
            "pull_requests": [],
            "_patch_signature": "patch-f",
        },
    ]
    downstream = [
        {
            "sha": "a" * 40,
            "message": "Shared upstream commit",
            "pull_requests": [{"number": 201}],
            "upstream_pr_references": [],
        },
        {
            "sha": "d" * 40,
            "message": f"Backport\n(cherry picked from commit {'b' * 40})",
            "pull_requests": [{"number": 202}],
            "upstream_pr_references": [],
            "_patch_signature": "different-patch",
        },
        {
            "sha": "e" * 40,
            "message": "Equivalent code change, no source reference",
            "pull_requests": [{"number": 203}],
            "upstream_pr_references": [],
            "_patch_signature": "patch-c",
        },
    ]

    result = compare_tool._match_upstream_commits(upstream, downstream)

    assert [match["match_type"] for match in result["matches"]] == [
        "same_commit_sha",
        "cherry_pick_trailer",
        "patch_equivalent_possible_cherry_pick",
    ]
    assert result["matches"][2]["confidence"] == "medium"
    assert (
        "does not prove cherry-pick provenance"
        in result["matches"][2]["interpretation"]
    )
    assert len(result["unmatched_upstream_commits"]) == 1
    assert result["unmatched_downstream_commits"] == []


@pytest.mark.asyncio
async def test_enrichment_reads_repo_local_prs_and_commit_patch(monkeypatch):
    sha = "d" * 40
    pr_url = f"https://api.github.com/repos/{DOWNSTREAM_REPO}/commits/{sha}/pulls"
    commit_url = f"https://api.github.com/repos/{DOWNSTREAM_REPO}/commits/{sha}"
    calls = []
    responses = {
        pr_url: _http_response(
            pr_url,
            200,
            [
                {
                    "number": 27,
                    "title": "Backport scheduler fix",
                    "body": "Backport of https://github.com/vllm-project/vllm/pull/1234",
                    "html_url": "https://github.com/example-org/nm-vllm-ent/pull/27",
                    "state": "closed",
                    "merged_at": "2026-10-01T12:00:00Z",
                }
            ],
        ),
        commit_url: _http_response(
            commit_url,
            200,
            {
                "files": [
                    {
                        "filename": "vllm/core/scheduler.py",
                        "patch": "@@ -1 +1 @@\n-old\n+new",
                    }
                ]
            },
        ),
    }
    monkeypatch.setattr(
        compare_tool.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, calls),
    )

    result = await compare_tool._enrich_commits(
        DOWNSTREAM_REPO,
        [{"sha": sha, "message": "Backport scheduler fix"}],
        {sha},
    )

    assert result[0]["pull_request_lookup"] == "found"
    assert result[0]["pull_requests"][0]["number"] == 27
    assert result[0]["upstream_pr_references"] == [1234]
    assert result[0]["_patch_signature"]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_tool_rejects_upstream_versions_without_resolving_tags(monkeypatch):
    async def unexpected_resolution(*args, **kwargs):
        raise AssertionError("upstream inputs must not enter the downstream workflow")

    monkeypatch.setattr(compare_tool, "get_downstream_vllm_tags", unexpected_resolution)

    result = await compare_tool.compare_downstream_vllm_releases(
        "v0.26.0",
        "v0.28.0",
        "0.26.0+rhaiv.8",
        "0.28.0+rhaiv.4",
    )

    assert result["status"] == "error"
    assert "only for comparing two RHAIIS downstream releases" in result["message"]


@pytest.mark.asyncio
async def test_tool_requires_exact_runtime_tag_matches(monkeypatch):
    monkeypatch.setattr(
        compare_tool.settings,
        "RHAIIS_DOWNSTREAM_VLLM_REPO",
        DOWNSTREAM_REPO,
    )
    calls = []

    async def resolve(release, runtime_version):
        calls.append((release, runtime_version))
        return {
            "status": "success",
            "repo": DOWNSTREAM_REPO,
            "runtime_version": runtime_version,
            "exact_runtime_match": release.endswith("fast1"),
            "runtime_tag_match": {"ref": "v0.26.0+rhaiv.8"}
            if release.endswith("fast1")
            else None,
            "candidates": [{"ref": "v0.28.0+rhaiv.4"}],
            "selection_guidance": "Confirm the exact build tag.",
        }

    async def unexpected_diff(*args, **kwargs):
        raise AssertionError("source comparisons require exact tags on both sides")

    monkeypatch.setattr(compare_tool, "get_downstream_vllm_tags", resolve)
    monkeypatch.setattr(compare_tool, "get_vllm_code_diff", unexpected_diff)

    result = await compare_tool.compare_downstream_vllm_releases(
        "RHAIIS-3.6-fast1",
        "RHAIIS-3.6-fast2",
        "0.26.0+rhaiv.8",
        "0.28.0+rhaiv.4",
    )

    assert result["status"] == "needs_exact_downstream_tag"
    assert set(result["unresolved_sides"]) == {"comparison"}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_tool_compares_both_upstream_and_downstream_ranges(monkeypatch):
    monkeypatch.setattr(
        compare_tool.settings,
        "RHAIIS_DOWNSTREAM_VLLM_REPO",
        DOWNSTREAM_REPO,
    )

    async def resolve(release, runtime_version):
        baseline = release.endswith("fast1")
        return {
            "status": "success",
            "repo": DOWNSTREAM_REPO,
            "rhaiis_version": release,
            "runtime_version": runtime_version,
            "runtime_version_base": "v0.26.0" if baseline else "v0.28.0",
            "mapped_upstream_version": "v0.26.0" if baseline else "v0.28.0",
            "mapping_matches_runtime_base": True,
            "exact_runtime_match": True,
            "runtime_tag_match": {
                "ref": "v0.26.0+rhaiv.8" if baseline else "v0.28.0+rhaiv.4"
            },
        }

    upstream_sha = "a" * 40
    cherry_pick_sha = "b" * 40

    async def compare(version1, version2, **kwargs):
        calls.append((version1, version2, kwargs))
        repo = kwargs["repo"]
        commits = (
            [{"sha": upstream_sha, "message": "Upstream scheduler improvement"}]
            if repo == VLLM_REPO
            else [
                {
                    "sha": cherry_pick_sha,
                    "message": f"Backport\n(cherry picked from commit {upstream_sha})",
                }
            ]
        )
        return {
            "status": "success",
            "repo": repo,
            "version_range": {"from": version1, "to": version2},
            "total_commits": 1,
            "total_files_changed": 1,
            "commits": commits,
            "commits_truncated": False,
            "files": [{"filename": "vllm/v1/core/sched/scheduler.py"}],
        }

    calls = []

    async def enrich(repo, commits, patch_shas):
        result = []
        for commit in commits:
            pr_number = 10 if repo == VLLM_REPO else 20
            result.append(
                {
                    **commit,
                    "pull_requests": [
                        {
                            "number": pr_number,
                            "title": "Scheduler performance change",
                            "html_url": f"https://github.com/{repo}/pull/{pr_number}",
                            "state": "closed",
                            "merged_at": "2026-10-01T12:00:00Z",
                        }
                    ],
                    "pull_request_lookup": "found",
                    "upstream_pr_references": [],
                    **(
                        {"_patch_signature": None}
                        if commit["sha"] in patch_shas
                        else {}
                    ),
                }
            )
        return result

    monkeypatch.setattr(compare_tool, "get_downstream_vllm_tags", resolve)
    monkeypatch.setattr(compare_tool, "get_vllm_code_diff", compare)
    monkeypatch.setattr(compare_tool, "_enrich_commits", enrich)

    result = await compare_tool.compare_downstream_vllm_releases(
        "RHAIIS-3.6-fast1",
        "RHAIIS-3.6-fast2",
        "0.26.0+rhaiv.8",
        "0.28.0+rhaiv.4",
    )

    assert result["status"] == "success"
    assert len(calls) == 2
    assert {(call[0], call[1], call[2]["repo"]) for call in calls} == {
        ("v0.26.0", "v0.28.0", VLLM_REPO),
        (
            "v0.26.0+rhaiv.8",
            "v0.28.0+rhaiv.4",
            DOWNSTREAM_REPO,
        ),
    }
    assert result["upstream_comparison"]["pull_requests"][0]["number"] == 10
    assert result["downstream_comparison"]["pull_requests"][0]["number"] == 20
    match = result["upstream_downstream_commit_matches"]["matches"][0]
    assert match["match_type"] == "cherry_pick_trailer"
    assert match["upstream_commit"]["sha"] == upstream_sha
    assert match["downstream_commit"]["sha"] == cherry_pick_sha
