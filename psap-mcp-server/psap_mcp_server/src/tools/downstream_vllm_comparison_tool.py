"""Compare two RHAIIS downstream releases with their upstream vLLM bases.

This workflow is intentionally limited to downstream-to-downstream releases.
It compares both the mapped/runtime upstream range and the exact downstream
tag range, then associates commits with repository-local pull requests and
identifies exact or patch-equivalent upstream commits where possible.
"""

import asyncio
import hashlib
import re
from typing import Any, Dict, List, Optional, Set

import httpx

from psap_mcp_server.src.settings import settings
from psap_mcp_server.src.tools.kernel_code_mapper_tool import (
    GITHUB_API_BASE,
    VLLM_REPO,
    _get_github_headers,
    get_downstream_vllm_tags,
    get_vllm_code_diff,
)
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

_MAX_COMMIT_ENRICHMENT = 50
_COMMIT_REQUEST_CONCURRENCY = 5
_UPSTREAM_PR_URL = re.compile(
    r"https?://github\.com/vllm-project/vllm/pull/(\d+)", re.IGNORECASE
)
_UPSTREAM_PR_SHORTHAND = re.compile(r"vllm-project/vllm#(\d+)", re.IGNORECASE)
_CHERRY_PICK_TRAILER = re.compile(
    r"cherry[- ]picked from commit\s+([0-9a-f]{7,40})", re.IGNORECASE
)


def _upstream_pr_references(text: str) -> Set[int]:
    """Extract only explicitly qualified upstream vLLM PR references."""
    refs = _UPSTREAM_PR_URL.findall(text) + _UPSTREAM_PR_SHORTHAND.findall(text)
    return {int(ref) for ref in refs}


def _patch_signature(commit_payload: Dict[str, Any]) -> Optional[str]:
    """Hash changed lines for a conservative cross-repository cherry-pick hint.

    This is not Git's patch-id algorithm. It ignores hunk offsets and context,
    while retaining file paths and ordered added/removed lines. A matching
    signature is evidence of equivalent changes, not proof of cherry-pick
    provenance.
    """
    files = commit_payload.get("files") or []
    if not files:
        return None

    normalized_files = []
    for changed_file in files:
        patch = changed_file.get("patch")
        if not isinstance(patch, str):
            return None
        operations = []
        for line in patch.splitlines():
            if line.startswith(("+++", "---", "@@", "\\")):
                continue
            if line.startswith(("+", "-")):
                content = re.sub(r"\s+", "", line[1:])
                operations.append(f"{line[0]}{content}")
        if operations:
            filename = changed_file.get("filename", "")
            normalized_files.append(f"{filename}\n" + "\n".join(operations))

    if not normalized_files:
        return None
    canonical = "\n---FILE---\n".join(sorted(normalized_files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _summarize_pull_request(pr: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "number": pr.get("number"),
        "title": str(pr.get("title", ""))[:300],
        "html_url": pr.get("html_url"),
        "state": pr.get("state"),
        "merged_at": pr.get("merged_at"),
    }


async def _enrich_commits(
    repo: str,
    commits: List[Dict[str, Any]],
    patch_signature_shas: Set[str],
) -> List[Dict[str, Any]]:
    """Attach repository-local PRs and optional patch signatures to commits."""
    if not commits:
        return []

    semaphore = asyncio.Semaphore(_COMMIT_REQUEST_CONCURRENCY)
    headers = _get_github_headers()

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def enrich(commit: Dict[str, Any]) -> Dict[str, Any]:
            sha = str(commit.get("sha") or "")
            base = f"{GITHUB_API_BASE}/repos/{repo}/commits/{sha}"
            async with semaphore:
                requests = [
                    client.get(
                        f"{base}/pulls", headers=headers, params={"per_page": 10}
                    )
                ]
                needs_patch = sha in patch_signature_shas
                if needs_patch:
                    requests.append(client.get(base, headers=headers))
                responses = await asyncio.gather(*requests, return_exceptions=True)

            pull_requests: List[Dict[str, Any]] = []
            pull_request_status = "unavailable"
            upstream_pr_refs = _upstream_pr_references(str(commit.get("message", "")))
            pr_response = responses[0]
            if (
                isinstance(pr_response, httpx.Response)
                and pr_response.status_code == 200
            ):
                payload = pr_response.json()
                pull_request_status = "found" if payload else "none_associated"
                for pr in payload:
                    body = str(pr.get("body") or "")
                    title = str(pr.get("title") or "")
                    upstream_pr_refs.update(_upstream_pr_references(f"{title}\n{body}"))
                    pull_requests.append(_summarize_pull_request(pr))
            elif (
                isinstance(pr_response, httpx.Response)
                and pr_response.status_code == 404
            ):
                # GitHub may use 404 for inaccessible private resources as well
                # as missing resources, so do not claim that no PR exists.
                pull_request_status = "not_found_or_inaccessible"
            elif isinstance(pr_response, httpx.Response):
                pull_request_status = f"http_{pr_response.status_code}"
            else:
                pull_request_status = "request_failed"

            signature = None
            if needs_patch and len(responses) > 1:
                detail_response = responses[1]
                if (
                    isinstance(detail_response, httpx.Response)
                    and detail_response.status_code == 200
                ):
                    signature = _patch_signature(detail_response.json())

            result = {
                **commit,
                "pull_requests": pull_requests,
                "pull_request_lookup": pull_request_status,
                "upstream_pr_references": sorted(upstream_pr_refs),
            }
            if needs_patch:
                result["_patch_signature"] = signature
            return result

        return await asyncio.gather(*(enrich(commit) for commit in commits))


def _unique_pull_requests(
    commits: List[Dict[str, Any]], repo: str
) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for commit in commits:
        sha = commit.get("sha")
        for pr in commit.get("pull_requests", []):
            key = str(pr.get("html_url") or f"{repo}#{pr.get('number')}")
            summary = by_key.setdefault(
                key,
                {"repo": repo, **pr, "associated_commit_shas": []},
            )
            if sha and sha not in summary["associated_commit_shas"]:
                summary["associated_commit_shas"].append(sha)
    return list(by_key.values())


def _commit_for_output(commit: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in commit.items() if not key.startswith("_")}


def _match_upstream_commits(
    upstream_commits: List[Dict[str, Any]], downstream_commits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Match exact SHAs, explicit cherry-pick trailers, PR links, and patches."""
    upstream_by_sha = {commit.get("sha"): commit for commit in upstream_commits}
    upstream_by_pr: Dict[int, List[Dict[str, Any]]] = {}
    upstream_by_signature: Dict[str, List[Dict[str, Any]]] = {}
    for commit in upstream_commits:
        for pr in commit.get("pull_requests", []):
            number = pr.get("number")
            if isinstance(number, int):
                upstream_by_pr.setdefault(number, []).append(commit)
        signature = commit.get("_patch_signature")
        if signature:
            upstream_by_signature.setdefault(signature, []).append(commit)

    matches = []
    matched_upstream_shas: Set[str] = set()
    matched_downstream_shas: Set[str] = set()

    for downstream in downstream_commits:
        downstream_sha = downstream.get("sha")
        candidates: List[tuple[Dict[str, Any], str, str]] = []

        exact = upstream_by_sha.get(downstream_sha)
        if exact:
            candidates = [(exact, "same_commit_sha", "high")]
        else:
            message = str(downstream.get("message", ""))
            trailer = _CHERRY_PICK_TRAILER.search(message)
            if trailer:
                source_sha = trailer.group(1).lower()
                trailer_matches = [
                    (commit, "cherry_pick_trailer", "high")
                    for sha, commit in upstream_by_sha.items()
                    if isinstance(sha, str) and sha.lower().startswith(source_sha)
                ]
                if len(trailer_matches) == 1:
                    candidates = trailer_matches

            if not candidates:
                referenced_prs = set(downstream.get("upstream_pr_references", []))
                pr_matches = [
                    commit
                    for pr_number in referenced_prs
                    for commit in upstream_by_pr.get(pr_number, [])
                ]
                if len(pr_matches) == 1:
                    candidates = [(pr_matches[0], "upstream_pr_reference", "medium")]

            if not candidates:
                signature = downstream.get("_patch_signature")
                signature_matches = (
                    upstream_by_signature.get(signature, []) if signature else []
                )
                if len(signature_matches) == 1:
                    candidates = [
                        (
                            signature_matches[0],
                            "patch_equivalent_possible_cherry_pick",
                            "medium",
                        )
                    ]

        # Do not silently choose between multiple upstream candidates.
        for upstream, match_type, confidence in candidates:
            upstream_sha = upstream.get("sha")
            if not downstream_sha or not upstream_sha:
                continue
            matches.append(
                {
                    "upstream_commit": _commit_for_output(upstream),
                    "downstream_commit": _commit_for_output(downstream),
                    "match_type": match_type,
                    "confidence": confidence,
                    "interpretation": (
                        "Commit identity is shared."
                        if match_type == "same_commit_sha"
                        else "The downstream commit explicitly points to an upstream commit."
                        if match_type == "cherry_pick_trailer"
                        else "Downstream metadata references an upstream PR in this range."
                        if match_type == "upstream_pr_reference"
                        else "Changed lines match, but this does not prove cherry-pick provenance."
                    ),
                }
            )
            matched_upstream_shas.add(upstream_sha)
            matched_downstream_shas.add(downstream_sha)

    return {
        "matches": matches,
        "unmatched_upstream_commits": [
            _commit_for_output(commit)
            for commit in upstream_commits
            if commit.get("sha") not in matched_upstream_shas
        ],
        "unmatched_downstream_commits": [
            _commit_for_output(commit)
            for commit in downstream_commits
            if commit.get("sha") not in matched_downstream_shas
        ],
        "coverage_note": (
            "Unmatched commits are not proven downstream-only: the inspected "
            "commit window may be truncated, GitHub may not expose a patch, or "
            "a cherry-pick may not preserve a source SHA, PR link, or matching patch."
        ),
    }


async def compare_downstream_vllm_releases(
    release1: str,
    release2: str,
    runtime_version1: str,
    runtime_version2: str,
    max_commits: int = 20,
) -> Dict[str, Any]:
    """Compare two RHAIIS downstream releases and their mapped upstream ranges.

    This tool is intentionally for two downstream RHAIIS releases only. It
    requires each server-reported vLLM build string to resolve to an exact
    downstream Git tag; mapping-derived candidates are never selected silently.

    TOOL_NAME=compare_downstream_vllm_releases
    DISPLAY_NAME=Compare Two RHAIIS Downstream Releases
    USECASE=Compare two downstream RHAIIS builds to identify upstream vLLM changes, downstream changes, repository-local PRs, and evidence of cherry-picked commits.
    INSTRUCTIONS=Use only for two RHAIIS releases; provide each release and its server-log vLLM version; require exact downstream tag matches; compare the corresponding upstream base range and exact downstream tag range; report PRs separately by repository and mark cherry-pick matches with their evidence and confidence; do not claim unmatched means downstream-only
    INPUT_DESCRIPTION=release1 and release2 (str): RHAIIS release labels; runtime_version1 and runtime_version2 (str): exact vLLM build versions from each run's server logs; max_commits (int): newest commits inspected per range, 1-50 (default 20)
    OUTPUT_DESCRIPTION=Exact downstream tag resolutions, upstream and downstream commit/file comparisons with repository-local PRs, and conservative matching evidence for shared or likely cherry-picked commits
    EXAMPLES=compare_downstream_vllm_releases("RHAIIS-3.6-fast1", "RHAIIS-3.6-fast2", "0.26.0+rhaiv.8", "0.28.0+rhaiv.4")
    PREREQUISITES=RHAIIS_DOWNSTREAM_VLLM_REPO must be configured; GITHUB_TOKEN needs Contents read access and Pull requests read access for private downstream PR details
    RELATED_TOOLS=get_downstream_vllm_tags, get_vllm_code_diff, get_vllm_pull_request, fetch_vllm_logs

    Args:
        release1: Baseline RHAIIS release label.
        release2: Comparison RHAIIS release label.
        runtime_version1: Exact server-reported vLLM build string for release1.
        runtime_version2: Exact server-reported vLLM build string for release2.
        max_commits: Newest commits to enrich in each comparison, clamped to 1-50.

    Returns:
        Two repository-specific commit/PR comparisons and conservative
        cross-reference evidence. The result does not assert performance cause.
    """
    releases = (release1.strip(), release2.strip())
    if any(
        not re.match(r"^RHAIIS(?:[-\s]|$)", release, re.IGNORECASE)
        for release in releases
    ):
        return {
            "status": "error",
            "message": (
                "This tool is only for comparing two RHAIIS downstream releases. "
                "For two upstream vLLM versions, use get_vllm_code_diff with the upstream repo."
            ),
        }

    runtimes = (runtime_version1.strip(), runtime_version2.strip())
    if not all(runtimes):
        return {
            "status": "error",
            "message": "Both exact server-log vLLM runtime versions are required.",
            "suggestion": "Fetch each run's server logs and pass parsed.server_config.vllm_version.",
        }

    repo = (settings.RHAIIS_DOWNSTREAM_VLLM_REPO or "").strip()
    if not repo:
        return {
            "status": "error",
            "message": "The downstream vLLM repository is not configured.",
            "suggestion": "Set RHAIIS_DOWNSTREAM_VLLM_REPO to the GitHub owner/repository.",
        }

    max_commits = max(1, min(int(max_commits), _MAX_COMMIT_ENRICHMENT))
    resolutions = await asyncio.gather(
        get_downstream_vllm_tags(releases[0], runtime_version=runtimes[0]),
        get_downstream_vllm_tags(releases[1], runtime_version=runtimes[1]),
    )
    resolution_by_side = dict(zip(("baseline", "comparison"), resolutions))
    for side, resolution in resolution_by_side.items():
        if resolution.get("status") != "success":
            return {
                "status": "error",
                "stage": "downstream_tag_resolution",
                "failed_side": side,
                "release": releases[0 if side == "baseline" else 1],
                "tag_resolution": resolution,
            }

    unresolved = {
        side: {
            "release": releases[0 if side == "baseline" else 1],
            "runtime_version": resolution.get("runtime_version"),
            "exact_runtime_match": resolution.get("exact_runtime_match"),
            "candidates": resolution.get("candidates", []),
            "selection_guidance": resolution.get("selection_guidance"),
        }
        for side, resolution in resolution_by_side.items()
        if not resolution.get("exact_runtime_match")
        or not (resolution.get("runtime_tag_match") or {}).get("ref")
    }
    if unresolved:
        return {
            "status": "needs_exact_downstream_tag",
            "comparison_type": "two_downstream_rhaiis_releases",
            "message": (
                "At least one server-log build string did not exactly resolve to a "
                "downstream Git tag. No source or PR comparison was run; do not "
                "choose a mapping fallback candidate automatically."
            ),
            "unresolved_sides": unresolved,
        }

    baseline, comparison = resolutions
    downstream_refs = (
        baseline["runtime_tag_match"]["ref"],
        comparison["runtime_tag_match"]["ref"],
    )
    upstream_refs = (
        baseline.get("runtime_version_base") or baseline.get("mapped_upstream_version"),
        comparison.get("runtime_version_base")
        or comparison.get("mapped_upstream_version"),
    )
    if not all(upstream_refs):
        return {
            "status": "partial",
            "stage": "upstream_base_resolution",
            "message": "Exact downstream tags were resolved, but an upstream base could not be established for both releases.",
            "downstream_refs": downstream_refs,
            "tag_resolutions": resolution_by_side,
        }

    upstream_diff, downstream_diff = await asyncio.gather(
        get_vllm_code_diff(
            upstream_refs[0],
            upstream_refs[1],
            max_files=8,
            repo=VLLM_REPO,
            include_commits=True,
            include_patches=False,
            max_commits=max_commits,
        ),
        get_vllm_code_diff(
            downstream_refs[0],
            downstream_refs[1],
            max_files=8,
            repo=repo,
            include_commits=True,
            include_patches=False,
            max_commits=max_commits,
        ),
    )

    if (
        upstream_diff.get("status") != "success"
        or downstream_diff.get("status") != "success"
    ):
        return {
            "status": "partial",
            "comparison_type": "two_downstream_rhaiis_releases",
            "tag_resolutions": resolution_by_side,
            "upstream_comparison": upstream_diff,
            "downstream_comparison": downstream_diff,
            "message": "One or both GitHub source comparisons failed; do not infer missing changes or PRs from the failed range.",
        }

    upstream_commits = upstream_diff.get("commits", [])
    downstream_commits = downstream_diff.get("commits", [])
    exact_shared_shas = {
        commit.get("sha") for commit in upstream_commits if commit.get("sha")
    } & {commit.get("sha") for commit in downstream_commits if commit.get("sha")}
    upstream_needing_patch = {
        commit.get("sha")
        for commit in upstream_commits
        if commit.get("sha") not in exact_shared_shas
    }
    downstream_needing_patch = {
        commit.get("sha")
        for commit in downstream_commits
        if commit.get("sha") not in exact_shared_shas
    }

    enriched_upstream, enriched_downstream = await asyncio.gather(
        _enrich_commits(VLLM_REPO, upstream_commits, upstream_needing_patch),
        _enrich_commits(repo, downstream_commits, downstream_needing_patch),
    )
    correspondence = _match_upstream_commits(enriched_upstream, enriched_downstream)

    warnings = []
    for side, resolution in resolution_by_side.items():
        if resolution.get("mapping_matches_runtime_base") is False:
            warnings.append(
                f"{side}: the logged runtime base differs from the configured RHAIIS-to-vLLM mapping; upstream comparison uses the base parsed from the exact runtime tag."
            )
    if upstream_diff.get("commits_truncated"):
        warnings.append(
            "The upstream commit comparison is limited to the newest inspected commit window."
        )
    if downstream_diff.get("commits_truncated"):
        warnings.append(
            "The downstream commit comparison is limited to the newest inspected commit window."
        )
    if any(
        commit.get("pull_request_lookup") not in {"found", "none_associated"}
        for commit in enriched_upstream
    ):
        warnings.append(
            "Some upstream commit-to-PR lookups were unavailable or inaccessible."
        )
    if any(
        commit.get("pull_request_lookup") not in {"found", "none_associated"}
        for commit in enriched_downstream
    ):
        warnings.append(
            "Some downstream commit-to-PR lookups were unavailable or inaccessible; check private-repository Pull requests read permission."
        )

    upstream_prs = _unique_pull_requests(enriched_upstream, VLLM_REPO)
    downstream_prs = _unique_pull_requests(enriched_downstream, repo)
    upstream_diff["commits"] = enriched_upstream
    downstream_diff["commits"] = enriched_downstream
    upstream_diff["pull_requests"] = upstream_prs
    downstream_diff["pull_requests"] = downstream_prs

    return {
        "status": "success",
        "comparison_type": "two_downstream_rhaiis_releases",
        "releases": {"baseline": releases[0], "comparison": releases[1]},
        "tag_resolutions": resolution_by_side,
        "upstream_comparison": upstream_diff,
        "downstream_comparison": downstream_diff,
        "upstream_downstream_commit_matches": correspondence,
        "warnings": warnings,
        "causality_note": (
            "PRs and commit matches are candidates/evidence, not proof of a performance cause. "
            "Correlate relevant files with the measured regression and profiler/log evidence."
        ),
    }
