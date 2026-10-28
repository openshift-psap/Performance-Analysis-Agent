"""MCP tool for mapping PyTorch kernel names to vLLM source code locations.

This tool helps correlate kernel-level performance data from PyTorch profiler
with the actual vLLM source code that implements those operations. Useful for
understanding which code paths are responsible for performance characteristics.

Also provides dynamic source code fetching and same- or cross-repository diff
capabilities via the GitHub Contents, Git Trees, and Compare APIs, enabling the
agent to read actual vLLM implementation code and inspect release changes.
"""

import asyncio
import base64
import difflib
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from psap_mcp_server.src.settings import settings
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
VLLM_REPO = "vllm-project/vllm"
_GITHUB_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _normalize_github_repository(
    repo: Optional[str],
    *,
    default: Optional[str] = VLLM_REPO,
    argument_name: str = "repo",
) -> str:
    """Validate and normalize a GitHub repository before URL construction."""
    target_repo = (repo or "").strip()
    if not target_repo and default is not None:
        target_repo = default

    if (
        not _GITHUB_REPOSITORY_PATTERN.fullmatch(target_repo)
        or any(part in {".", ".."} for part in target_repo.split("/"))
    ):
        raise ValueError(f"{argument_name} must use GitHub owner/repository format.")
    return target_repo

# Curated mapping of kernel patterns to likely vLLM source locations
# Format: (pattern, [(path, confidence, description)])
KERNEL_MAPPINGS = [
    # Flash Attention kernels
    (r"flash_attn.*|FlashAttn.*", [
        ("vllm/attention/backends/flash_attn.py", "high", "Flash Attention backend implementation"),
        ("vllm/attention/ops/", "medium", "Attention operation implementations"),
        ("csrc/attention/", "medium", "C++/CUDA attention kernels"),
    ]),
    
    # PagedAttention kernels
    (r"paged_attention.*|PagedAttention.*", [
        ("vllm/attention/backends/", "high", "Paged attention backend"),
        ("csrc/attention/attention_kernels.cu", "high", "CUDA paged attention kernels"),
    ]),
    
    # MoE (Mixture of Experts) kernels - important for DeepSeek
    (r".*moe.*|.*expert.*|.*MoE.*|fused_moe.*", [
        ("vllm/model_executor/layers/fused_moe/", "high", "Fused MoE layer implementations"),
        ("csrc/moe/", "high", "C++/CUDA MoE kernels"),
        ("vllm/model_executor/layers/moe/", "medium", "MoE layer Python code"),
    ]),
    
    # Quantization kernels
    (r".*awq.*|.*gptq.*|.*fp8.*|.*int8.*|.*quant.*", [
        ("vllm/model_executor/layers/quantization/", "high", "Quantization implementations"),
        ("csrc/quantization/", "high", "CUDA quantization kernels"),
    ]),
    
    # GEMM / Matrix multiplication
    (r"aten::mm|aten::bmm|aten::matmul|aten::linear|cutlass.*gemm.*|cublas.*gemm.*", [
        ("vllm/model_executor/layers/linear.py", "medium", "Linear layer implementations"),
        ("csrc/", "low", "Custom CUDA kernels may override"),
    ]),
    
    # Activation functions
    (r"aten::silu|aten::gelu|aten::relu|.*activation.*|silu_and_mul.*", [
        ("vllm/model_executor/layers/activation.py", "high", "Activation function implementations"),
        ("csrc/activation_kernels.cu", "medium", "CUDA activation kernels"),
    ]),
    
    # LayerNorm / RMSNorm
    (r".*layer_norm.*|.*rms_norm.*|.*LayerNorm.*|.*RMSNorm.*|aten::layer_norm", [
        ("vllm/model_executor/layers/layernorm.py", "high", "LayerNorm implementations"),
        ("csrc/layernorm_kernels.cu", "medium", "CUDA LayerNorm kernels"),
    ]),
    
    # Rotary embeddings
    (r".*rotary.*|.*rope.*|.*RoPE.*", [
        ("vllm/model_executor/layers/rotary_embedding.py", "high", "Rotary embedding implementations"),
        ("csrc/pos_encoding_kernels.cu", "medium", "CUDA position encoding kernels"),
    ]),
    
    # Sampling / Top-k/p
    (r".*sample.*|.*topk.*|.*topp.*|.*argmax.*", [
        ("vllm/model_executor/layers/sampler.py", "high", "Sampling implementations"),
        ("csrc/", "low", "CUDA sampling kernels"),
    ]),
    
    # Communication / NCCL
    (r"nccl.*|ncclAllReduce.*|ncclAllGather.*|c10d.*", [
        ("vllm/distributed/", "high", "Distributed communication code"),
        ("vllm/executor/", "medium", "Executor implementations"),
    ]),
    
    # Memory operations
    (r"aten::copy_|aten::clone|aten::contiguous|aten::to|cudaMemcpy.*", [
        ("vllm/worker/", "medium", "Worker implementations handle memory"),
        ("vllm/attention/backends/", "low", "Attention backends may copy tensors"),
    ]),
    
    # KV Cache operations
    (r".*kv_cache.*|.*cache_copy.*|.*reshape_and_cache.*", [
        ("vllm/attention/backends/", "high", "KV cache operations in attention backends"),
        ("csrc/cache_kernels.cu", "high", "CUDA cache kernels"),
        ("vllm/worker/cache_engine.py", "medium", "Cache engine implementation"),
    ]),
    
    # Embedding operations
    (r"aten::embedding|.*embed.*", [
        ("vllm/model_executor/layers/vocab_parallel_embedding.py", "high", "Embedding layer implementations"),
    ]),
    
    # Softmax
    (r"aten::softmax|aten::_softmax|.*softmax.*", [
        ("vllm/attention/", "medium", "Softmax used in attention"),
        ("csrc/attention/", "medium", "CUDA attention with fused softmax"),
    ]),
    
    # Custom vLLM ops
    (r"vllm::.*", [
        ("csrc/", "high", "Custom vLLM CUDA ops"),
        ("vllm/", "medium", "Python wrapper code"),
    ]),
    
    # Triton kernels
    (r"triton.*|Triton.*", [
        ("vllm/attention/ops/", "high", "Triton attention kernels"),
        ("vllm/model_executor/layers/fused_moe/", "medium", "Triton MoE kernels"),
    ]),
]

# PyTorch ATen operations that are standard library (not vLLM-specific)
PYTORCH_STDLIB_OPS = [
    r"aten::empty.*",
    r"aten::zeros.*",
    r"aten::ones.*",
    r"aten::view.*",
    r"aten::reshape.*",
    r"aten::transpose.*",
    r"aten::permute.*",
    r"aten::squeeze.*",
    r"aten::unsqueeze.*",
    r"aten::cat.*",
    r"aten::stack.*",
    r"aten::split.*",
    r"aten::chunk.*",
    r"aten::select.*",
    r"aten::slice.*",
    r"aten::index.*",
    r"aten::as_strided.*",
    r"aten::expand.*",
    r"aten::repeat.*",
    r"aten::fill_.*",
    r"aten::zero_.*",
    r"aten::add.*",
    r"aten::sub.*",
    r"aten::mul.*",
    r"aten::div.*",
    r"aten::pow.*",
    r"aten::sqrt.*",
    r"aten::rsqrt.*",
    r"aten::exp.*",
    r"aten::log.*",
    r"aten::abs.*",
    r"aten::neg.*",
    r"aten::sum.*",
    r"aten::mean.*",
    r"aten::max.*",
    r"aten::min.*",
    r"aten::where.*",
    r"aten::masked.*",
    r"aten::scatter.*",
    r"aten::gather.*",
]


def _get_github_headers() -> Dict[str, str]:
    """Get headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = getattr(settings, 'GITHUB_TOKEN', None)
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


async def _normalize_version(version: str) -> str:
    """Normalize version string to a vLLM GitHub tag.

    Handles RHAIIS product versions (e.g. ``RHAIIS-3.3``) by resolving them
    through the cached GitHub mapping (with the bundled JSON as fallback), as
    well as plain vLLM versions
    (e.g. ``0.13.0`` or ``v0.13.0``).
    """
    version = version.strip()

    if version.upper().startswith("RHAIIS"):
        try:
            from psap_mcp_server.src.tools.vllm_release_notes_tool import (
                _resolve_to_vllm_version,
            )
            resolved, _, was_mapped = await _resolve_to_vllm_version(version)
            if was_mapped:
                return resolved
            logger.warning(
                f"RHAIIS version '{version}' not found in mappings — "
                f"passing through as-is"
            )
        except ImportError:
            logger.warning("Could not import version resolver; using raw version")

    if not version.startswith("v"):
        version = f"v{version}"
    return version


def _tag_platform_hints(tag: str, description: str) -> List[str]:
    """Infer explicit platform labels from a tag name and its Git metadata."""
    searchable = f"{tag} {description}".lower()
    patterns = {
        "cpu": r"\bcpu\b",
        "tpu": r"\btpu\b",
        "rocm": r"\brocm\b",
        "spyre": r"\bspyre\b",
        "rubin": r"\brubin\b",
        "b300": r"\b(?:b300|gb300)\b",
        "hpu": r"\bhpu\b",
        "cuda": r"\bcuda\b|\bnvidia\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, searchable)]


def _platform_matches(requested: Optional[str], hints: List[str]) -> Optional[bool]:
    """Return whether a candidate explicitly matches a requested platform."""
    if not requested:
        return None

    platform = requested.strip().lower().replace("_", "-")
    aliases = {
        "nvidia": "gpu",
        "nvidia-gpu": "gpu",
        "gb300": "b300",
    }
    platform = aliases.get(platform, platform)
    if not hints:
        return None
    if platform in hints:
        return True

    # A generic CUDA label cannot rule out a specific GPU model (for example,
    # H200), and missing metadata is not evidence of incompatibility.
    gpu_platforms = {"cuda", "b300", "rubin"}
    other_platforms = {"cpu", "tpu", "rocm", "spyre", "hpu"}
    if platform == "gpu":
        if any(hint in gpu_platforms for hint in hints):
            return True
        if any(hint in other_platforms for hint in hints):
            return False
    elif platform in other_platforms and any(
        hint in gpu_platforms or hint in other_platforms for hint in hints
    ):
        return False
    elif platform in gpu_platforms and any(
        hint in other_platforms for hint in hints
    ):
        return False
    return None


async def _describe_downstream_tag(
    client: httpx.AsyncClient, ref: Dict[str, Any], repo: str
) -> Dict[str, Any]:
    """Read annotation or commit metadata for one Git tag ref."""
    tag_name = str(ref.get("ref", "")).removeprefix("refs/tags/")
    git_object = ref.get("object", {})
    object_type = git_object.get("type")
    object_sha = git_object.get("sha")
    description = ""
    commit_date = None
    commit_sha = object_sha

    if object_type == "tag" and object_sha:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{repo}/git/tags/{object_sha}",
            headers=_get_github_headers(),
        )
        if response.status_code == 200:
            tag_data = response.json()
            description = tag_data.get("message", "").strip()
            commit_sha = tag_data.get("object", {}).get("sha", object_sha)
            commit_date = tag_data.get("tagger", {}).get("date")
    elif object_type == "commit" and object_sha:
        response = await client.get(
            f"{GITHUB_API_BASE}/repos/{repo}/commits/{object_sha}",
            headers=_get_github_headers(),
        )
        if response.status_code == 200:
            commit_data = response.json()
            commit = commit_data.get("commit", {})
            description = commit.get("message", "").strip()
            commit_date = commit.get("committer", {}).get("date")

    return {
        "ref": tag_name,
        "commit_sha": commit_sha,
        "date": commit_date,
        "description": description[:500],
        "platform_hints": _tag_platform_hints(tag_name, description),
        "github_url": f"https://github.com/{repo}/tree/{quote(tag_name, safe='')}",
    }


def _normalize_downstream_version_tag(version: str) -> Optional[str]:
    """Normalize a log version to its possible downstream Git tag.

    Runtime logs generally omit the leading ``v`` used by Git tags. Only that
    leading prefix is normalized; the downstream build suffix must match
    exactly before a tag is considered verified.
    """
    value = version.strip()
    for prefix in ("vLLM-", "vllm-", "vLLM ", "vllm "):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    if value.startswith("v") and len(value) > 1 and value[1].isdigit():
        value = value[1:]
    if not re.fullmatch(r"\d+(?:\.\d+)+(?:\+[A-Za-z0-9._-]+)?", value):
        return None
    return f"v{value}"


def _upstream_base_from_tag(tag: str) -> Optional[str]:
    """Remove a downstream build suffix while preserving the upstream version."""
    match = re.fullmatch(r"(v\d+(?:\.\d+)+)(?:\+[A-Za-z0-9._-]+)?", tag)
    return match.group(1) if match else None


async def _fetch_downstream_tag_refs(
    client: httpx.AsyncClient, repo: str, upstream_version: str
) -> Dict[str, Any]:
    """Fetch refs for an exact upstream version and its downstream variants."""
    prefix = f"tags/{upstream_version}"
    url = (
        f"{GITHUB_API_BASE}/repos/{repo}/git/matching-refs/"
        f"{quote(prefix, safe='/')}"
    )
    response = await client.get(
        url,
        headers=_get_github_headers(),
        params={"per_page": 100},
    )
    if response.status_code != 200:
        return {
            "status": "error",
            "http_status": response.status_code,
            "message": f"GitHub tag lookup failed: HTTP {response.status_code}",
        }

    refs = []
    for ref in response.json():
        tag_name = ref.get("ref", "").removeprefix("refs/tags/")
        if tag_name == upstream_version or tag_name.startswith(f"{upstream_version}+"):
            refs.append(ref)
    return {"status": "success", "refs": refs}


async def get_downstream_vllm_tags(
    version: str,
    platform: Optional[str] = None,
    max_tags: int = 25,
    runtime_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a logged runtime version to a downstream tag when possible.

    First checks whether the server-reported runtime version exactly matches a
    tag in the configured downstream repository. If it does not, the RHAIIS to
    upstream version mapping supplies candidate tags as a fallback. Fallback
    candidates are never treated as an exact match.

    TOOL_NAME=get_downstream_vllm_tags
    DISPLAY_NAME=List Downstream vLLM Tags
    USECASE=Find exact downstream tag candidates for an RHAIIS or upstream vLLM version before reading downstream source.
    INSTRUCTIONS=1. Provide the RHAIIS release, 2. If available, pass runtime_version from fetch_vllm_logs, 3. Treat only an exact runtime tag match as verified; otherwise use mapped candidates as fallback and do not guess.
    INPUT_DESCRIPTION=version (str): RHAIIS release or upstream vLLM version used for fallback; runtime_version (str, optional): exact vLLM build version reported by server logs; platform (str, optional): platform hint; max_tags (int): Maximum candidates returned (1-50, default 25)
    OUTPUT_DESCRIPTION=Exact runtime tag match when found, otherwise candidate downstream refs from the RHAIIS-to-upstream mapping fallback, with metadata and platform hints
    EXAMPLES=get_downstream_vllm_tags("RHAIIS-3.6-fast2", runtime_version="0.28.0+rhai.v3"), get_downstream_vllm_tags("RHAIIS-3.6-fast2", platform="cpu")
    PREREQUISITES=Set RHAIIS_DOWNSTREAM_VLLM_REPO to a GitHub owner/repository. Private repositories also require GITHUB_TOKEN with read access.
    RELATED_TOOLS=fetch_vllm_source, map_kernel_to_vllm_code, get_vllm_code_diff, get_cross_repo_vllm_diff

    Args:
        version: RHAIIS product version or upstream vLLM base version.
        platform: Optional platform hint used to annotate candidate matches.
        max_tags: Maximum number of candidate tags to return (clamped to 1-50).
        runtime_version: Optional vLLM version parsed from a server log. An exact
            downstream tag match takes precedence over the RHAIIS mapping.

    Returns:
        Exact runtime tag match when present, or fallback candidate refs based on
        the mapped upstream version.
    """
    try:
        requested_version = version.strip()
        target_repo = (settings.RHAIIS_DOWNSTREAM_VLLM_REPO or "").strip()
        if not target_repo:
            return {
                "status": "error",
                "message": "The downstream vLLM repository is not configured.",
                "suggestion": "Set RHAIIS_DOWNSTREAM_VLLM_REPO to the GitHub owner/repository.",
            }
        try:
            target_repo = _normalize_github_repository(
                target_repo, default=None, argument_name="RHAIIS_DOWNSTREAM_VLLM_REPO"
            )
        except ValueError as exc:
            return {
                "status": "error",
                "message": str(exc),
            }

        from psap_mcp_server.src.tools.vllm_release_notes_tool import (
            _resolve_to_vllm_version,
        )

        upstream_version, rhaiis_version, was_mapped = await _resolve_to_vllm_version(
            requested_version
        )
        runtime_tag = (
            _normalize_downstream_version_tag(runtime_version)
            if runtime_version
            else None
        )
        runtime_base = (
            _upstream_base_from_tag(runtime_tag) if runtime_tag else None
        )
        is_rhaiis_release = requested_version.upper().startswith("RHAIIS")
        if is_rhaiis_release and not was_mapped and not runtime_tag:
            return {
                "status": "error",
                "message": f"No upstream vLLM mapping is available for {requested_version}.",
                "suggestion": "Check get_version_mappings() or provide the upstream vLLM version explicitly.",
            }

        mapped_upstream_version = None
        if was_mapped or not is_rhaiis_release:
            normalized_mapping = await _normalize_version(upstream_version)
            mapped_upstream_version = (
                _upstream_base_from_tag(normalized_mapping) or normalized_mapping
            )

        primary_version = runtime_base or mapped_upstream_version
        if not primary_version:
            return {
                "status": "error",
                "repo": target_repo,
                "message": "Neither the server runtime version nor the RHAIIS mapping provided a usable vLLM version.",
            }

        max_tags = max(1, min(int(max_tags), 50))

        async with httpx.AsyncClient(timeout=30.0) as client:
            exact_runtime_match = False
            matched_runtime_tag = None
            candidates = []
            total_candidates = 0
            resolved_version = mapped_upstream_version or primary_version
            resolution_method = "rhaiis_mapping"
            fallback_reason = None

            if runtime_tag:
                exact_ref_url = (
                    f"{GITHUB_API_BASE}/repos/{target_repo}/git/ref/tags/"
                    f"{quote(runtime_tag, safe='')}"
                )
                exact_response = await client.get(
                    exact_ref_url,
                    headers=_get_github_headers(),
                )
                if exact_response.status_code == 200:
                    exact_ref = exact_response.json()
                    if exact_ref.get("ref") == f"refs/tags/{runtime_tag}":
                        matched_runtime_tag = await _describe_downstream_tag(
                            client, exact_ref, target_repo
                        )
                        matched_runtime_tag["platform_match"] = _platform_matches(
                            platform, matched_runtime_tag["platform_hints"]
                        )
                        candidates = [matched_runtime_tag]
                        total_candidates = 1
                        exact_runtime_match = True
                        resolved_version = runtime_base
                        resolution_method = "exact_runtime_tag"
                elif exact_response.status_code != 404:
                    return {
                        "status": "error",
                        "repo": target_repo,
                        "message": f"GitHub denied the exact runtime tag lookup: HTTP {exact_response.status_code}",
                        "suggestion": "Check GITHUB_TOKEN access and repository permissions.",
                    }

            if not exact_runtime_match:
                # The log's build string did not identify an exact repository
                # tag, so fall back to RHAIIS -> upstream mapping candidates.
                fallback_version = mapped_upstream_version or runtime_base
                if not fallback_version:
                    return {
                        "status": "error",
                        "repo": target_repo,
                        "message": "No exact runtime tag was found and no upstream base version is available for fallback.",
                        "suggestion": "Check the RHAIIS version mapping or provide a valid runtime version.",
                    }

                ref_result = await _fetch_downstream_tag_refs(
                    client, target_repo, fallback_version
                )
                if ref_result["status"] != "success":
                    status_code = ref_result.get("http_status")
                    if status_code == 403:
                        message = "GitHub denied the tag lookup. Check token approval, repository access, and API rate limits."
                    elif status_code == 404:
                        message = "GitHub could not find the downstream repository or refs; private-repository access may also be missing."
                    else:
                        message = ref_result.get("message", "GitHub tag lookup failed.")
                    return {
                        "status": "error",
                        "repo": target_repo,
                        "message": message,
                        "suggestion": "Verify the configured repository, GITHUB_TOKEN access, and the mapped upstream version.",
                    }

                refs = ref_result["refs"]
                semaphore = asyncio.Semaphore(5)

                async def describe_limited(ref: Dict[str, Any]) -> Dict[str, Any]:
                    async with semaphore:
                        return await _describe_downstream_tag(
                            client, ref, target_repo
                        )

                candidates = await asyncio.gather(
                    *(describe_limited(ref) for ref in refs)
                )
                candidates.sort(
                    key=lambda item: item.get("date") or "", reverse=True
                )
                for candidate in candidates:
                    candidate["platform_match"] = _platform_matches(
                        platform, candidate["platform_hints"]
                    )
                total_candidates = len(candidates)
                candidates = candidates[:max_tags]
                resolved_version = fallback_version
                resolution_method = (
                    "rhaiis_mapping_fallback"
                    if runtime_version and mapped_upstream_version
                    else "rhaiis_mapping"
                    if mapped_upstream_version
                    else "runtime_base_candidates"
                )
                if not runtime_version:
                    fallback_reason = "No runtime version was available in the server log."
                elif not runtime_tag:
                    fallback_reason = (
                        "The server-reported runtime version could not be normalized; "
                        "using the mapped upstream base as fallback."
                    )
                else:
                    fallback_reason = (
                        "No exact downstream tag matched the server-reported runtime "
                        "version; using the mapped upstream base as fallback."
                    )
                if (
                    runtime_base
                    and mapped_upstream_version
                    and runtime_base != mapped_upstream_version
                ):
                    fallback_reason += (
                        f" The logged base ({runtime_base}) differs from the mapping "
                        f"({mapped_upstream_version})."
                    )

        return {
            "status": "success",
            "repo": target_repo,
            "rhaiis_version": rhaiis_version,
            "upstream_version": resolved_version,
            "mapped_upstream_version": mapped_upstream_version,
            "runtime_version": runtime_version,
            "runtime_version_tag": runtime_tag,
            "runtime_version_base": runtime_base,
            "runtime_tag_match": matched_runtime_tag,
            "exact_runtime_match": exact_runtime_match,
            "mapping_matches_runtime_base": (
                runtime_base == mapped_upstream_version
                if runtime_base and mapped_upstream_version
                else None
            ),
            "resolution_method": resolution_method,
            "used_mapping_fallback": resolution_method
            in {"rhaiis_mapping", "rhaiis_mapping_fallback"},
            "fallback_reason": fallback_reason,
            "platform_requested": platform,
            "candidates": candidates,
            "count": len(candidates),
            "total_candidates": total_candidates,
            "truncated": total_candidates > len(candidates),
            "requires_exact_ref": not exact_runtime_match,
            "selection_guidance": (
                "The runtime tag is verified by an exact match to the version reported in the server log."
                if exact_runtime_match
                else "These are fallback candidates from the RHAIIS-to-upstream mapping, not a verified runtime tag. Match tag descriptions and platform to the actual build; do not guess from recency."
            ),
            "next_step": (
                "Pass runtime_tag_match.ref and this repo to the source tools."
                if exact_runtime_match
                else "Confirm the exact build tag before passing a candidate ref and this repo to source tools."
            ),
            "message": (
                f"Found {len(candidates)} downstream candidate tag(s) for {resolved_version} in "
                f"{target_repo}."
            ),
        }
    except (httpx.TimeoutException, httpx.NetworkError) as e:
        logger.warning(f"Downstream GitHub tag lookup failed ({type(e).__name__})")
        return {
            "status": "error",
            "repo": (settings.RHAIIS_DOWNSTREAM_VLLM_REPO or "").strip() or None,
            "message": "Could not reach GitHub to list downstream tags.",
        }
    except Exception as e:
        logger.error(f"Error listing downstream vLLM tags ({type(e).__name__})")
        return {
            "status": "error",
            "repo": (settings.RHAIIS_DOWNSTREAM_VLLM_REPO or "").strip() or None,
            "message": f"Failed to list downstream tags: {type(e).__name__}",
        }


def _is_pytorch_stdlib(kernel_name: str) -> bool:
    """Check if kernel is a standard PyTorch operation."""
    for pattern in PYTORCH_STDLIB_OPS:
        if re.match(pattern, kernel_name, re.IGNORECASE):
            return True
    return False


def _find_kernel_mapping(kernel_name: str) -> List[Dict[str, str]]:
    """Find source file mappings for a kernel name using curated patterns."""
    results = []
    
    for pattern, mappings in KERNEL_MAPPINGS:
        if re.search(pattern, kernel_name, re.IGNORECASE):
            for path, confidence, description in mappings:
                results.append({
                    "path": path,
                    "confidence": confidence,
                    "description": description,
                    "matched_pattern": pattern,
                })
    
    return results


def _build_github_url(path: str, version: str, repo: str = VLLM_REPO) -> str:
    """Build GitHub URL for a file path at a specific version."""
    return f"https://github.com/{repo}/blob/{quote(version, safe='')}/{path}"


def _build_github_search_url(kernel_name: str, repo: str = VLLM_REPO) -> str:
    """Build GitHub search URL for a kernel name."""
    # Clean up kernel name for search
    search_term = kernel_name.replace("::", " ").replace("_", " ")
    return f"https://github.com/{repo}/search?q={search_term}"


async def map_kernel_to_vllm_code(
    kernel_name: str,
    version: str = "v0.13.0",
    search_github: bool = False,
    profile_model: Optional[str] = None,
    profile_tp: Optional[str] = None,
    profile_workload: Optional[str] = None,
    repo: str = VLLM_REPO,
) -> Dict[str, Any]:
    """Map a PyTorch kernel/operation name to vLLM source code locations.
    
    Given a kernel name from PyTorch profiler traces, find the likely vLLM
    source files that implement or invoke that kernel. Useful for correlating
    performance data with actual code to understand implementation details.

    When profile coordinates (profile_model, profile_tp, profile_workload) are
    provided alongside the version, the tool loads the actual profile trace and
    returns ground-truth call stack attribution if the profile was collected
    with ``with_stack=True``.  Falls back to heuristic pattern matching for
    profiles without stacks.
    
    TOOL_NAME=map_kernel_to_vllm_code
    DISPLAY_NAME=Map Kernel to vLLM Code
    USECASE=Map PyTorch profiler kernel names to vLLM source code files. Use this after identifying slow kernels with compare_pytorch_profiles to understand the code responsible for performance characteristics. When profile coordinates are provided, returns ground-truth call stacks from the profile (if collected with stack traces enabled).
    INSTRUCTIONS=1. Provide a kernel name from PyTorch profiler (e.g., "flash_attn_v2_fwd", "fused_moe", "aten::mm"), 2. Optionally specify a vLLM version, 3. Optionally provide profile_model, profile_tp, profile_workload to get real call stack attribution from the profile, 4. Results include likely source files and GitHub URLs
    INPUT_DESCRIPTION=kernel_name (str): Kernel name from PyTorch profiler; version (str): upstream version or exact downstream tag; repo (str): GitHub repo (default vllm-project/vllm; use the repo returned by get_downstream_vllm_tags with an exact downstream tag); search_github (bool): Also search the selected repo; profile_model (str, optional): Model name to look up real stacks; profile_tp (str, optional): TP config; profile_workload (str, optional): ISL/OSL workload
    OUTPUT_DESCRIPTION=Dictionary with likely source files, confidence levels, GitHub URLs, and real call stack attribution when available
    EXAMPLES=map_kernel_to_vllm_code("flash_attn_v2_fwd"), map_kernel_to_vllm_code("fused_moe_kernel", version="v0.21.0", profile_model="deepseek-r1", profile_tp="tp8", profile_workload="isl1000_osl1000")
    PREREQUISITES=For downstream source search, use the repo returned by get_downstream_vllm_tags. Private repositories require GITHUB_TOKEN with read access.
    RELATED_TOOLS=compare_pytorch_profiles, analyze_pytorch_profile, compare_vllm_versions, get_vllm_pull_request
    
    Args:
        kernel_name: The kernel name from PyTorch profiler (e.g., "flash_attn_v2_fwd")
        version: vLLM version for GitHub URLs (default: "v0.13.0")
        search_github: If True, also search GitHub API for the kernel name
        profile_model: Model name to load profile stacks from (e.g., "deepseek-r1")
        profile_tp: Tensor parallelism config (e.g., "tp8")
        profile_workload: ISL/OSL workload (e.g., "isl1000_osl1000")
        repo: GitHub repository to link/search. For downstream source, first use
            get_downstream_vllm_tags and pass an exact downstream tag.
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - kernel_name: The input kernel name
        - is_pytorch_stdlib: Whether this is a standard PyTorch operation
        - likely_source_files: List of likely source file locations
        - profile_call_stacks: Real call stacks from profile (if available)
        - github_urls: GitHub URLs to view the files
        - search_suggestions: How to find more information
    """
    try:
        try:
            target_repo = _normalize_github_repository(repo)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        version = (
            await _normalize_version(version)
            if target_repo == VLLM_REPO
            else version.strip()
        )
        
        # Check if it's a standard PyTorch op
        is_stdlib = _is_pytorch_stdlib(kernel_name)
        
        # Attempt to load real call stacks from profile data
        profile_call_stacks = None
        if profile_model and not is_stdlib:
            try:
                from psap_mcp_server.src.tools.pytorch_profile_tool import (
                    _extract_bare_version,
                    _load_or_extract_stats,
                    _resolve_profile,
                )
                profile_version = version.split("+", 1)[0]
                bare_version = f"vLLM-{_extract_bare_version(profile_version)}"
                model_key, matched_tp, matched_version, matched_workload, error, index = await _resolve_profile(
                    profile_model, profile_tp, bare_version, profile_workload
                )
                if not error and model_key and matched_tp and matched_version and matched_workload:
                    stats = _load_or_extract_stats(
                        model_key, matched_tp, matched_version, matched_workload, 0,
                        index=index,
                    )
                    if stats and kernel_name in stats:
                        stacks = stats[kernel_name].get("call_stacks", {})
                        if stacks:
                            sorted_stacks = sorted(stacks.items(), key=lambda x: x[1], reverse=True)
                            profile_call_stacks = [
                                {"call_stack": s, "invocation_count": c}
                                for s, c in sorted_stacks[:5]
                            ]
            except Exception as e:
                logger.warning(f"Could not load profile stacks for {kernel_name}: {e}")

        # Find heuristic mappings
        mappings = _find_kernel_mapping(kernel_name)

        # Build result
        likely_files = []
        github_urls = []
        
        for mapping in mappings:
            file_info = {
                "path": mapping["path"],
                "confidence": mapping["confidence"],
                "description": mapping["description"],
            }
            likely_files.append(file_info)
            
            # Build GitHub URL (handle directory paths)
            if mapping["path"].endswith("/"):
                github_urls.append({
                    "path": mapping["path"],
                    "url": f"https://github.com/{target_repo}/tree/{quote(version, safe='')}/{mapping['path'].rstrip('/')}",
                    "type": "directory",
                })
            else:
                github_urls.append({
                    "path": mapping["path"],
                    "url": _build_github_url(mapping["path"], version, target_repo),
                    "type": "file",
                })
        
        # GitHub search (optional)
        github_search_results = None
        if search_github and not is_stdlib:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    headers = _get_github_headers()
                    search_query = kernel_name.replace("::", " ").replace("_", "+")
                    search_url = f"{GITHUB_API_BASE}/search/code?q={search_query}+repo:{target_repo}&per_page=10"
                    
                    response = await client.get(search_url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        github_search_results = {
                            "total_count": data.get("total_count", 0),
                            "items": [
                                {
                                    "path": item.get("path"),
                                    "url": item.get("html_url"),
                                }
                                for item in data.get("items", [])[:5]
                            ],
                        }
            except Exception as e:
                logger.warning(f"GitHub search failed: {e}")
        
        # Build suggestions
        suggestions = []
        if profile_call_stacks:
            suggestions.append("Real call stacks available from profile — use these for ground-truth attribution")
        if is_stdlib:
            suggestions.append("This is a standard PyTorch operation implemented in PyTorch core, not vLLM")
            suggestions.append("Performance may be affected by how vLLM uses this op (tensor shapes, data types)")
        elif not mappings and not profile_call_stacks:
            suggestions.append(f"No curated mapping found for '{kernel_name}'")
            suggestions.append(f"Try searching GitHub: {_build_github_search_url(kernel_name, target_repo)}")
            suggestions.append("Check if this kernel is from a third-party library (e.g., flash-attn, triton)")
        elif target_repo != VLLM_REPO:
            suggestions.append(
                f"Inspect these locations with fetch_vllm_source using version='{version}' and repo='{target_repo}'"
            )
            suggestions.append(
                "Use get_vllm_code_diff with two exact refs from this repository to compare downstream changes"
            )
        else:
            suggestions.append("Use get_vllm_pull_request to find PRs that modified these files")
            suggestions.append("Use compare_vllm_versions to see changes between releases")
        
        result = {
            "status": "success",
            "kernel_name": kernel_name,
            "version": version,
            "repo": target_repo,
            "is_pytorch_stdlib": is_stdlib,
            "likely_source_files": likely_files,
            "github_urls": github_urls,
            "github_search_results": github_search_results,
            "search_suggestions": suggestions,
            "message": f"Found {len(likely_files)} likely source location(s) for '{kernel_name}'" if likely_files
                      else f"No specific mapping for '{kernel_name}' - try GitHub search",
            "github_manual_search_url": _build_github_search_url(kernel_name, target_repo),
        }

        if profile_call_stacks:
            result["profile_call_stacks"] = profile_call_stacks
            result["attribution_source"] = "profile_stack_trace"
            result["message"] = (
                f"Ground-truth call stack from profile for '{kernel_name}' "
                f"({len(profile_call_stacks)} unique path(s))"
            )
        else:
            result["attribution_source"] = "heuristic_pattern_match"

        return result
        
    except Exception as e:
        logger.error(f"Error mapping kernel to code: {e}")
        return {
            "status": "error",
            "message": f"Failed to map kernel: {str(e)}",
        }


async def get_kernel_categories() -> Dict[str, Any]:
    """Get information about kernel categories and their vLLM implementations.
    
    Returns a guide to kernel categories in PyTorch profiler and how they
    map to vLLM code. Useful for understanding the structure of profile data.
    
    TOOL_NAME=get_kernel_categories
    DISPLAY_NAME=Get Kernel Categories
    USECASE=Understand kernel categories in PyTorch profiler and how they map to vLLM. Use this to learn about different operation types before analyzing profiles.
    INSTRUCTIONS=Call without arguments to get category reference
    INPUT_DESCRIPTION=No arguments required
    OUTPUT_DESCRIPTION=Dictionary with kernel category information and vLLM mapping guide
    EXAMPLES=get_kernel_categories()
    PREREQUISITES=None
    RELATED_TOOLS=analyze_pytorch_profile, map_kernel_to_vllm_code
    
    Returns:
        Dictionary with kernel category information.
    """
    return {
        "status": "success",
        "categories": {
            "kernel": {
                "description": "CUDA kernels (GPU compute operations)",
                "vllm_relevance": "high",
                "examples": ["flash_attn_v2_fwd", "fused_moe_kernel", "paged_attention_v1"],
                "vllm_locations": ["csrc/", "vllm/attention/ops/"],
            },
            "cpu_op": {
                "description": "CPU operations (PyTorch ATen ops)",
                "vllm_relevance": "medium",
                "examples": ["aten::mm", "aten::copy_", "aten::index"],
                "vllm_locations": ["Standard PyTorch, but vLLM controls usage"],
            },
            "cuda_runtime": {
                "description": "CUDA runtime API calls",
                "vllm_relevance": "low",
                "examples": ["cudaLaunchKernel", "cudaStreamSynchronize"],
                "vllm_locations": ["System CUDA library"],
            },
            "nccl": {
                "description": "NCCL collective communication operations",
                "vllm_relevance": "high",
                "examples": ["ncclAllReduce", "ncclAllGather"],
                "vllm_locations": ["vllm/distributed/"],
            },
            "cuda_memory": {
                "description": "CUDA memory operations",
                "vllm_relevance": "medium",
                "examples": ["cudaMalloc", "cudaMemcpy"],
                "vllm_locations": ["Memory managed by PyTorch/vLLM worker"],
            },
        },
        "vllm_key_areas": {
            "attention": {
                "kernels": ["flash_attn*", "paged_attention*", "*attention*"],
                "files": ["vllm/attention/", "csrc/attention/"],
            },
            "moe": {
                "kernels": ["*moe*", "*expert*", "fused_moe*"],
                "files": ["vllm/model_executor/layers/fused_moe/", "csrc/moe/"],
            },
            "quantization": {
                "kernels": ["*awq*", "*gptq*", "*fp8*", "*quant*"],
                "files": ["vllm/model_executor/layers/quantization/", "csrc/quantization/"],
            },
            "communication": {
                "kernels": ["nccl*", "c10d*"],
                "files": ["vllm/distributed/"],
            },
        },
        "message": "Use these categories with analyze_pytorch_profile(category='...') to filter results",
    }


async def correlate_kernel_with_changes(
    kernel_name: str,
    version1: str,
    version2: str,
    repo: str = VLLM_REPO,
) -> Dict[str, Any]:
    """Correlate a kernel's performance change with vLLM code changes.
    
    Given a kernel that shows performance regression or improvement, find
    related code changes between versions. Combines kernel mapping with
    version comparison to help identify root causes.
    
    TOOL_NAME=correlate_kernel_with_changes
    DISPLAY_NAME=Correlate Kernel with Code Changes
    USECASE=Investigate why a specific kernel's performance changed between versions. Use after compare_pytorch_profiles identifies a regression or improvement to find the responsible code changes.
    INSTRUCTIONS=1. Provide the kernel name showing performance change, 2. Specify the two versions being compared, 3. Results combine kernel mapping with changelog analysis
    INPUT_DESCRIPTION=kernel_name (str): Kernel showing performance change; version1 (str): Baseline version/tag; version2 (str): Comparison version/tag; repo (str): GitHub repository to compare (default vllm-project/vllm)
    OUTPUT_DESCRIPTION=Dictionary with kernel mapping, related code areas, and suggestions for investigation
    EXAMPLES=correlate_kernel_with_changes("flash_attn_v2_fwd", "v0.11.2", "v0.13.0")
    PREREQUISITES=None
    RELATED_TOOLS=compare_pytorch_profiles, map_kernel_to_vllm_code, compare_vllm_versions, get_vllm_pull_request
    
    Args:
        kernel_name: The kernel showing performance change
        version1: Baseline vLLM version or exact downstream tag.
        version2: Comparison vLLM version or exact downstream tag.
        repo: GitHub repository. For downstream refs, use the repo returned by
            get_downstream_vllm_tags.
    
    Returns:
        Dictionary with correlation analysis and investigation suggestions.
    """
    try:
        try:
            target_repo = _normalize_github_repository(repo)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        if target_repo == VLLM_REPO:
            version1 = await _normalize_version(version1)
            version2 = await _normalize_version(version2)
        
        # Get kernel mapping
        mapping_result = await map_kernel_to_vllm_code(
            kernel_name, version2, repo=target_repo
        )
        
        # Build investigation guide
        investigation_steps = []
        focus_areas = []
        
        if mapping_result.get("is_pytorch_stdlib"):
            investigation_steps.append(
                "This is a standard PyTorch op - investigate how vLLM's usage changed"
            )
            investigation_steps.append(
                "Check for changes in tensor shapes, batch sizes, or data types"
            )
            focus_areas.append("general")
        else:
            likely_files = mapping_result.get("likely_source_files", [])
            
            # Determine focus areas based on kernel type
            if any("attention" in f.get("path", "").lower() for f in likely_files):
                focus_areas.append("kernel/attention")
            if any("moe" in f.get("path", "").lower() for f in likely_files):
                focus_areas.append("kernel/attention")  # MoE often grouped with kernels
            if any("quant" in f.get("path", "").lower() for f in likely_files):
                focus_areas.append("quantization")
            if any("distributed" in f.get("path", "").lower() for f in likely_files):
                focus_areas.append("scheduling")
            
            if not focus_areas:
                focus_areas.append("general")
            
            for f in likely_files[:3]:
                investigation_steps.append(f"Review changes to {f['path']} between versions")
        
        if target_repo == VLLM_REPO:
            investigation_steps.append(
                f"Use compare_vllm_versions('{version1}', '{version2}', focus_areas={focus_areas}) "
                "to find related release notes"
            )
            investigation_steps.append(
                "Look for PR numbers in release notes and use get_vllm_pull_request() to investigate"
            )
        else:
            investigation_steps.append(
                f"Use get_vllm_code_diff('{version1}', '{version2}', repo='{target_repo}') "
                "to inspect downstream source changes"
            )
        
        return {
            "status": "success",
            "kernel_name": kernel_name,
            "repo": target_repo,
            "version_range": {"from": version1, "to": version2},
            "kernel_mapping": mapping_result,
            "suggested_focus_areas": focus_areas,
            "investigation_steps": investigation_steps,
            "quick_links": {
                "github_compare": f"https://github.com/{target_repo}/compare/{quote(version1, safe='')}...{quote(version2, safe='')}",
                "github_search": _build_github_search_url(kernel_name, target_repo),
            },
            "message": f"Correlation analysis for '{kernel_name}' between {version1} and {version2}",
            "next_tool_calls": (
                [
                    f"compare_vllm_versions('{version1}', '{version2}', focus_areas={focus_areas})",
                    f"get_vllm_release_notes(version='{version2}')",
                ]
                if target_repo == VLLM_REPO
                else [
                    f"get_vllm_code_diff('{version1}', '{version2}', repo='{target_repo}')"
                ]
            ),
        }
        
    except Exception as e:
        logger.error(f"Error correlating kernel with changes: {e}")
        return {
            "status": "error",
            "message": f"Failed to correlate kernel: {str(e)}",
        }


# Max file size to return to the agent (avoid blowing up LLM context)
_MAX_SOURCE_BYTES = 50_000


async def fetch_vllm_source(
    file_path: str,
    version: str = "v0.13.0",
    repo: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch source code from GitHub at a specific version tag.
    
    Retrieves the full file content from a GitHub repository at the given
    version/ref. Defaults to upstream vLLM and also supports private repositories
    when GITHUB_TOKEN has read access.
    
    No pre-cloning is needed -- files are fetched on demand via the GitHub
    Contents API.
    
    TOOL_NAME=fetch_vllm_source
    DISPLAY_NAME=Fetch GitHub Source Code
    USECASE=Read source code at a specific upstream or downstream version/ref. Use get_downstream_vllm_tags first for RHAIIS builds, then fetch with the exact downstream tag and repo.
    INSTRUCTIONS=1. Provide a file path relative to the repo root, 2. Specify an exact tag/ref, 3. For RHAIIS downstream code, first call get_downstream_vllm_tags and use its repo value plus a confirmed exact tag; never substitute only the upstream version
    INPUT_DESCRIPTION=file_path (str): Path relative to repo root; version (str): Exact Git tag/ref (default v0.13.0); repo (str, optional): GitHub repo in owner/name format (default vllm-project/vllm; for downstream use the repo returned by get_downstream_vllm_tags)
    OUTPUT_DESCRIPTION=Dictionary with file content, metadata, and GitHub URL
    EXAMPLES=fetch_vllm_source("vllm/model_executor/layers/fused_moe/fused_moe.py", "v0.13.0"), fetch_vllm_source(".ci/docker/triton_version.txt", "v2.9.1", repo="pytorch/pytorch")
    PREREQUISITES=Internet access to GitHub API. Private repositories require GITHUB_TOKEN with Contents read permission and any required organization approval.
    RELATED_TOOLS=map_kernel_to_vllm_code, get_vllm_code_diff, get_cross_repo_vllm_diff, correlate_kernel_with_changes
    
    Args:
        file_path: Path relative to the repo root.
        version: Git tag or ref to fetch from (default: "v0.13.0").
        repo: GitHub repository in "owner/name" format (default: vllm-project/vllm).
              For RHAIIS downstream source, use the repo returned by
              get_downstream_vllm_tags; for dependencies, use the relevant repo.
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - file_path: The requested path
        - version: The version fetched
        - content: The file content (truncated if very large)
        - size_bytes: Original file size
        - truncated: Whether content was truncated
        - github_url: Direct link to view the file on GitHub
    """
    try:
        try:
            target_repo = _normalize_github_repository(repo)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        if target_repo == VLLM_REPO:
            version = await _normalize_version(version)
        file_path = file_path.lstrip("/")

        url = f"{GITHUB_API_BASE}/repos/{target_repo}/contents/{file_path}"
        params = {"ref": version}

        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = _get_github_headers()
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 404:
                return {
                    "status": "error",
                    "message": f"GitHub returned 404 for '{file_path}' at ref '{version}' in {target_repo}; the file/ref may be absent or the token may not have access.",
                    "suggestion": f"Verify the exact ref and repository access: https://github.com/{target_repo}/tree/{quote(version, safe='')}/{file_path}",
                }
            if response.status_code == 403:
                return {
                    "status": "error",
                    "message": "GitHub denied access (403). Check token permissions/organization approval and API rate limits.",
                }
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"GitHub API error: HTTP {response.status_code}",
                    "details": response.text[:500],
                }

            data = response.json()

            # Handle directory listings
            if isinstance(data, list):
                entries = [
                    {"name": e.get("name"), "type": e.get("type"), "path": e.get("path")}
                    for e in data
                ]
                return {
                    "status": "success",
                    "file_path": file_path,
                    "version": version,
                    "repo": target_repo,
                    "type": "directory",
                    "entries": entries,
                    "message": f"'{file_path}' is a directory with {len(entries)} entries at {version} in {target_repo}",
                    "github_url": f"https://github.com/{target_repo}/tree/{quote(version, safe='')}/{file_path}",
                }

            # Decode file content
            encoding = data.get("encoding", "")
            raw_content = data.get("content", "")
            size_bytes = data.get("size", 0)

            if encoding == "base64":
                content = base64.b64decode(raw_content).decode("utf-8", errors="replace")
            else:
                content = raw_content

            truncated = False
            if len(content) > _MAX_SOURCE_BYTES:
                content = content[:_MAX_SOURCE_BYTES]
                truncated = True

            return {
                "status": "success",
                "file_path": file_path,
                "version": version,
                "repo": target_repo,
                "type": "file",
                "content": content,
                "size_bytes": size_bytes,
                "line_count": content.count("\n") + 1,
                "truncated": truncated,
                "github_url": f"https://github.com/{target_repo}/blob/{quote(version, safe='')}/{file_path}",
                "message": f"Fetched {file_path} at {version} from {target_repo} ({size_bytes:,} bytes, {content.count(chr(10))+1} lines)"
                + (" [truncated]" if truncated else ""),
            }

    except httpx.TimeoutException:
        return {"status": "error", "message": "GitHub API request timed out"}
    except Exception as e:
        logger.error(f"Error fetching source from {repo or VLLM_REPO}: {e}")
        return {"status": "error", "message": f"Failed to fetch source: {str(e)}"}


async def get_vllm_code_diff(
    version1: str,
    version2: str,
    file_path: Optional[str] = None,
    max_files: int = 10,
    repo: str = VLLM_REPO,
    include_commits: bool = False,
    include_patches: bool = True,
    max_commits: int = 25,
) -> Dict[str, Any]:
    """Fetch actual code diff between two vLLM versions, optionally for a specific file.
    
    Uses the GitHub Compare API to retrieve patch content showing exactly what
    lines changed between two releases. This is the key tool for explaining
    the mechanism behind performance changes -- it lets you see the actual
    code modifications.
    
    TOOL_NAME=get_vllm_code_diff
    DISPLAY_NAME=Get vLLM Code Diff
    USECASE=Inspect actual source changes between two refs in the same upstream or downstream vLLM repository. Use after identifying performance-relevant source files to understand code changes; use get_cross_repo_vllm_diff when the refs belong to different repositories.
    INSTRUCTIONS=1. For two upstream refs, omit repo; 2. For a two-downstream RHAIIS comparison that needs both upstream/downstream commit and PR attribution, use compare_downstream_vllm_releases; 3. For a same-repo source diff, first resolve exact downstream tags with get_downstream_vllm_tags and pass the same repo; 4. Optionally scope to a file; 5. Use get_cross_repo_vllm_diff only for source snapshots in different repositories
    INPUT_DESCRIPTION=version1 (str): Baseline exact tag/ref; version2 (str): Comparison exact tag/ref; file_path (str, optional): Specific file path to diff; max_files (int): Max files to return (default 10); repo (str): Single GitHub repo for both refs; include_commits (bool): Include bounded commit summaries (default false); include_patches (bool): Include file patches (default true); max_commits (int): Maximum commit summaries to return (1-50, default 25)
    OUTPUT_DESCRIPTION=Dictionary with changed files, stats, optional bounded commit summaries, and GitHub compare URL; a successful comparison with file_changed=false means the requested path was not in the changed-file list, but may be unchanged or absent at the refs
    EXAMPLES=get_vllm_code_diff("v0.11.2", "v0.13.0", "vllm/model_executor/layers/fused_moe/fused_moe.py"), get_vllm_code_diff("v0.28.0+build.1", "v0.28.0+build.2", repo="example-org/downstream-vllm", include_commits=True, include_patches=False)
    PREREQUISITES=Internet access to GitHub API. Private downstream refs require GITHUB_TOKEN with Contents read access and any required organization approval.
    RELATED_TOOLS=fetch_vllm_source, map_kernel_to_vllm_code, compare_vllm_versions, get_cross_repo_vllm_diff, correlate_kernel_with_changes
    
    Args:
        version1: Baseline version (e.g., "v0.11.2").
        version2: Comparison version (e.g., "v0.13.0").
        file_path: If provided, only return the diff for this file.
        max_files: Maximum number of file patches to return (default: 10).
        repo: GitHub repository to compare. For downstream refs, use the repo
            returned by get_downstream_vllm_tags. Private repositories require
            GITHUB_TOKEN access.
        include_commits: Include a bounded, newest-first list of commit metadata.
        include_patches: Whether to include patch text for changed files.
        max_commits: Maximum commit summaries to include, clamped to 1-50.
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - version_range: The two versions compared
        - files: List of changed files, with patch content unless disabled
        - commits: Optional newest-first commit summaries and truncation status
        - file_changed: False when a requested file is not in the completed
          comparison's changed-file list; this alone does not prove the path
          exists at either ref
        - stats: Overall diff statistics
        - github_compare_url: Link to the full comparison on GitHub
    """
    try:
        try:
            target_repo = _normalize_github_repository(repo)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        if target_repo == VLLM_REPO:
            version1 = await _normalize_version(version1)
            version2 = await _normalize_version(version2)

        url = (
            f"{GITHUB_API_BASE}/repos/{target_repo}/compare/"
            f"{quote(version1, safe='')}...{quote(version2, safe='')}"
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = _get_github_headers()
            response = await client.get(url, headers=headers)

            if response.status_code == 404:
                return {
                    "status": "error",
                    "message": f"Could not compare {version1}...{version2} in {target_repo}. Check the exact refs and private-repository access.",
                }
            if response.status_code == 403:
                return {
                    "status": "error",
                    "message": "GitHub denied access (403). Check token permissions, organization approval, and API rate limits.",
                }
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"GitHub API error: HTTP {response.status_code}",
                    "details": response.text[:500],
                }

            data = response.json()

            commit_records = data.get("commits", [])
            total_commits = data.get("total_commits", len(commit_records))
            all_files = data.get("files", [])
            commit_metadata: Dict[str, Any] = {}
            if include_commits:
                max_commits = max(1, min(int(max_commits), 50))
                newest_commits = reversed(commit_records[-max_commits:])
                summarized_commits = []
                for item in newest_commits:
                    commit = item.get("commit", {})
                    author = commit.get("author") or {}
                    summarized_commits.append(
                        {
                            "sha": item.get("sha"),
                            "message": str(commit.get("message", ""))[:1200],
                            "author": author.get("name"),
                            "date": author.get("date"),
                            "html_url": item.get("html_url"),
                        }
                    )
                commit_metadata = {
                    "commits": summarized_commits,
                    "commits_returned": len(summarized_commits),
                    "commits_truncated": len(commit_records) > len(summarized_commits)
                    or int(total_commits) > len(commit_records),
                    "commit_window": "newest",
                }

            # Performance-related path keywords for prioritisation
            perf_keywords = [
                "attention", "moe", "fused_moe", "quantization", "fp8",
                "kernel", "csrc", "distributed", "scheduler", "cache",
                "activation", "layernorm", "linear", "sampler", "deep_gemm",
            ]

            if file_path:
                file_path = file_path.lstrip("/")
                matched = [f for f in all_files if f.get("filename") == file_path]
                if not matched:
                    available = [f.get("filename") for f in all_files if any(kw in f.get("filename", "").lower() for kw in perf_keywords)][:20]
                    return {
                        "status": "success",
                        "comparison_status": "completed",
                        "file_changed": False,
                        "file_path": file_path,
                        "file_change_status": "not_in_changed_files",
                        "repo": target_repo,
                        "version_range": {"from": version1, "to": version2},
                        "total_files_changed": len(all_files),
                        "files_returned": 0,
                        "files": [],
                        "message": (
                            f"The comparison completed, but '{file_path}' was not in "
                            "GitHub's changed-file list. It may be unchanged or the "
                            "path may not exist at these refs; this result does not "
                            "distinguish those cases."
                        ),
                        "performance_related_changed_files": available,
                        **commit_metadata,
                    }
                target_files = matched
            else:
                # Prioritise performance-relevant files
                def _perf_score(f: dict) -> int:
                    name = f.get("filename", "").lower()
                    return sum(1 for kw in perf_keywords if kw in name)

                scored = [(f, _perf_score(f)) for f in all_files]
                scored.sort(key=lambda x: (-x[1], -x[0].get("changes", 0)))
                target_files = [f for f, _ in scored[:max_files]]

            files_result: List[Dict[str, Any]] = []
            total_patch_size = 0
            patch_budget = _MAX_SOURCE_BYTES

            for f in target_files:
                patch = f.get("patch", "")
                if not include_patches:
                    patch = ""
                # Truncate individual patches that are too large
                if total_patch_size + len(patch) > patch_budget:
                    remaining = max(0, patch_budget - total_patch_size)
                    patch = patch[:remaining] + "\n... [patch truncated]"
                    truncated = True
                else:
                    truncated = False

                total_patch_size += len(patch)

                files_result.append({
                    "filename": f.get("filename"),
                    "status": f.get("status"),  # added, removed, modified, renamed
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "changes": f.get("changes", 0),
                    "patch": patch if include_patches else None,
                    "patch_truncated": truncated if include_patches else False,
                    "patch_included": include_patches,
                })

                if total_patch_size >= patch_budget:
                    break

            return {
                "status": "success",
                "repo": target_repo,
                "version_range": {"from": version1, "to": version2},
                "total_commits": total_commits,
                "total_files_changed": len(all_files),
                "files_returned": len(files_result),
                "files": files_result,
                **commit_metadata,
                "stats": {
                    "total_additions": sum(f.get("additions", 0) for f in all_files),
                    "total_deletions": sum(f.get("deletions", 0) for f in all_files),
                },
                "github_compare_url": f"https://github.com/{target_repo}/compare/{quote(version1, safe='')}...{quote(version2, safe='')}",
                "message": (
                    f"Diff {version1}...{version2}: {len(all_files)} files changed, "
                    f"returning {len(files_result)} "
                    + (f"(filtered to '{file_path}')" if file_path else "(top performance-relevant)")
                ),
            }

    except httpx.TimeoutException:
        return {"status": "error", "message": "GitHub API request timed out (comparison may be very large)"}
    except Exception as e:
        logger.error(f"Error fetching vLLM code diff: {e}")
        return {"status": "error", "message": f"Failed to fetch diff: {str(e)}"}


_MAX_CROSS_REPO_BLOB_BYTES = 500_000
_MAX_CROSS_REPO_PATCH_BYTES = 50_000
_MAX_CROSS_REPO_PATHS = 200


async def _fetch_repo_tree(
    client: httpx.AsyncClient, repo: str, ref: str
) -> Dict[str, Any]:
    """Resolve a ref to a commit and return its recursive Git tree."""
    headers = _get_github_headers()
    api_ref = ref
    if api_ref.startswith("refs/tags/"):
        api_ref = f"tags/{api_ref.removeprefix('refs/tags/')}"
    elif api_ref.startswith("refs/heads/"):
        api_ref = f"heads/{api_ref.removeprefix('refs/heads/')}"
    elif not api_ref.startswith(("tags/", "heads/")) and not re.fullmatch(
        r"[0-9a-fA-F]{40,64}", api_ref
    ):
        # Version refs are tags by default. Branch refs can be passed explicitly
        # as heads/<branch>; full commit SHAs need no namespace.
        api_ref = f"tags/{api_ref}"
    commit_url = (
        f"{GITHUB_API_BASE}/repos/{repo}/commits/{quote(api_ref, safe='/')}"
    )
    commit_response = await client.get(commit_url, headers=headers)
    if commit_response.status_code != 200:
        return {
            "status": "error",
            "http_status": commit_response.status_code,
            "message": (
                f"Could not resolve ref '{ref}' in {repo}: "
                f"GitHub returned HTTP {commit_response.status_code}."
            ),
        }

    commit_data = commit_response.json()
    commit_sha = commit_data.get("sha")
    tree_sha = commit_data.get("commit", {}).get("tree", {}).get("sha")
    if not commit_sha or not tree_sha:
        return {
            "status": "error",
            "message": f"GitHub returned incomplete commit metadata for {repo}@{ref}.",
        }

    tree_url = f"{GITHUB_API_BASE}/repos/{repo}/git/trees/{tree_sha}"
    tree_response = await client.get(
        tree_url, headers=headers, params={"recursive": "1"}
    )
    if tree_response.status_code != 200:
        return {
            "status": "error",
            "http_status": tree_response.status_code,
            "message": (
                f"Could not read the Git tree for {repo}@{ref}: "
                f"GitHub returned HTTP {tree_response.status_code}."
            ),
        }

    tree_data = tree_response.json()
    if tree_data.get("truncated"):
        return {
            "status": "error",
            "message": (
                f"GitHub truncated the recursive tree for {repo}@{ref}; "
                "a complete repository comparison cannot be guaranteed."
            ),
            "suggestion": "Retry with file_path set to the specific source file.",
        }

    return {
        "status": "success",
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "entries": {
            entry["path"]: entry
            for entry in tree_data.get("tree", [])
            if entry.get("type") in {"blob", "commit"}
        },
    }


async def _fetch_github_blob(
    client: httpx.AsyncClient, repo: str, sha: str
) -> Dict[str, Any]:
    """Fetch and decode a text blob, rejecting oversized or binary content."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/git/blobs/{sha}"
    response = await client.get(url, headers=_get_github_headers())
    if response.status_code != 200:
        return {
            "status": "error",
            "message": f"Could not fetch blob {sha} from {repo}: HTTP {response.status_code}.",
        }

    payload = response.json()
    size = payload.get("size", 0)
    if size > _MAX_CROSS_REPO_BLOB_BYTES:
        return {
            "status": "skipped",
            "message": (
                f"Blob is {size} bytes; patches are limited to "
                f"{_MAX_CROSS_REPO_BLOB_BYTES} bytes per file."
            ),
        }
    if payload.get("encoding") != "base64":
        return {"status": "skipped", "message": "GitHub did not return a base64 text blob."}

    try:
        raw_content = base64.b64decode(payload.get("content", ""))
        if b"\0" in raw_content:
            return {"status": "skipped", "message": "Binary files are not rendered as text patches."}
        return {"status": "success", "content": raw_content.decode("utf-8")}
    except (ValueError, UnicodeDecodeError):
        return {"status": "skipped", "message": "Blob is not valid UTF-8 text."}


async def get_cross_repo_vllm_diff(
    base_repo: str,
    base_ref: str,
    comparison_repo: str,
    comparison_ref: str,
    file_path: Optional[str] = None,
    max_files: int = 10,
) -> Dict[str, Any]:
    """Compare source trees across two GitHub repositories and exact refs.

    Unlike GitHub's Compare API (used by get_vllm_code_diff), this tool supports
    different repositories on each side, such as upstream vLLM and a downstream
    RHAIIS fork. Refs must be supplied explicitly so a mapped upstream version
    is not mistaken for a verified downstream build tag.

    TOOL_NAME=get_cross_repo_vllm_diff
    DISPLAY_NAME=Compare vLLM Code Across Repositories
    USECASE=Compare source across different repositories, especially an upstream vLLM version against its confirmed RHAIIS downstream build.
    INSTRUCTIONS=1. Use this only when the two refs are in different repositories; same-repository inputs are rejected, 2. For upstream-versus-RHAIIS compare the mapped upstream base ref with the exact downstream tag confirmed from logs/build metadata, 3. Set file_path for a focused file diff or omit it to compare repository trees, 4. Never treat an inferred downstream candidate as verified; for two refs in one repository use get_vllm_code_diff
    INPUT_DESCRIPTION=base_repo (str): GitHub owner/repository for baseline; base_ref (str): exact tag, heads/<branch>, or full commit SHA; comparison_repo (str): GitHub owner/repository for comparison; comparison_ref (str): exact tag, heads/<branch>, or full commit SHA; file_path (str, optional): exact repo-relative file path; max_files (int): maximum patches returned (default 10, max 30)
    OUTPUT_DESCRIPTION=Changed paths and bounded unified text patches, with each side's repository/ref/commit SHA and source links
    EXAMPLES=get_cross_repo_vllm_diff("vllm-project/vllm", "v0.28.0", "example-org/downstream-vllm", "v0.28.0+build.3", file_path="vllm/attention/backends/flash_attn.py")
    PREREQUISITES=GitHub API access; private repositories require GITHUB_TOKEN with read access. For RHAIIS use get_downstream_vllm_tags and fetch_vllm_logs first to establish the refs.
    RELATED_TOOLS=get_downstream_vllm_tags, fetch_vllm_logs, fetch_vllm_source, get_vllm_code_diff

    Args:
        base_repo: Baseline GitHub repository in owner/name format.
        base_ref: Exact baseline tag, ``heads/<branch>``, or full commit SHA.
        comparison_repo: Comparison GitHub repository in owner/name format.
        comparison_ref: Exact comparison tag, ``heads/<branch>``, or full commit SHA.
        file_path: Optional exact repo-relative file to compare. If omitted, the
            tool compares repository trees and returns prioritized changed files.
        max_files: Maximum number of patches to return, clamped to 1-30.

    Returns:
        Structured repository/ref metadata, changed-path counts, and bounded
        unified patches. Large, binary, or non-UTF-8 files are listed without a
        text patch. Truncated GitHub trees are rejected rather than reported as
        complete comparisons.
    """
    try:
        base_repo = _normalize_github_repository(
            base_repo, default=None, argument_name="base_repo"
        )
        comparison_repo = _normalize_github_repository(
            comparison_repo, default=None, argument_name="comparison_repo"
        )
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    if not base_ref.strip() or not comparison_ref.strip():
        return {"status": "error", "message": "Both exact refs are required."}

    base_ref = base_ref.strip()
    comparison_ref = comparison_ref.strip()
    if base_repo.casefold() == comparison_repo.casefold():
        return {
            "status": "error",
            "message": (
                "Both refs are in the same repository. Use get_vllm_code_diff "
                "for a same-repository comparison."
            ),
            "suggestion": (
                f"get_vllm_code_diff('{base_ref}', '{comparison_ref}', "
                f"repo='{base_repo}')"
            ),
        }
    file_path = file_path.strip().lstrip("/") if file_path else None
    max_files = max(1, min(int(max_files), 30))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            base_tree, comparison_tree = await asyncio.gather(
                _fetch_repo_tree(client, base_repo, base_ref),
                _fetch_repo_tree(client, comparison_repo, comparison_ref),
            )
            for side, repo, ref, tree in (
                ("base", base_repo, base_ref, base_tree),
                ("comparison", comparison_repo, comparison_ref, comparison_tree),
            ):
                if tree.get("status") != "success":
                    return {
                        "status": "error",
                        "side": side,
                        "repo": repo,
                        "ref": ref,
                        "message": tree.get("message", "Could not read repository tree."),
                        **({"suggestion": tree["suggestion"]} if tree.get("suggestion") else {}),
                    }

            base_entries = base_tree["entries"]
            comparison_entries = comparison_tree["entries"]
            if file_path:
                paths = [file_path] if file_path in base_entries or file_path in comparison_entries else []
            else:
                paths = sorted(set(base_entries) | set(comparison_entries))

            changed = []
            for path in paths:
                base_entry = base_entries.get(path)
                comparison_entry = comparison_entries.get(path)
                if base_entry is None:
                    change_status = "added"
                elif comparison_entry is None:
                    change_status = "removed"
                elif (
                    base_entry.get("sha") != comparison_entry.get("sha")
                    or base_entry.get("mode") != comparison_entry.get("mode")
                    or base_entry.get("type") != comparison_entry.get("type")
                ):
                    change_status = "modified"
                else:
                    continue
                changed.append(
                    {
                        "filename": path,
                        "status": change_status,
                        "base_entry": base_entry,
                        "comparison_entry": comparison_entry,
                    }
                )

            if file_path and not paths:
                return {
                    "status": "error",
                    "message": f"File '{file_path}' was not found at either ref.",
                    "suggestion": "Provide an exact file path relative to each repository root.",
                }
            if file_path and not changed:
                return {
                    "status": "success",
                    "base": {
                        "repo": base_repo,
                        "ref": base_ref,
                        "commit_sha": base_tree["commit_sha"],
                        "tree_sha": base_tree["tree_sha"],
                    },
                    "comparison": {
                        "repo": comparison_repo,
                        "ref": comparison_ref,
                        "commit_sha": comparison_tree["commit_sha"],
                        "tree_sha": comparison_tree["tree_sha"],
                    },
                    "total_files_changed": 0,
                    "files_returned": 0,
                    "files": [],
                    "file_path": file_path,
                    "file_changed": False,
                    "message": f"'{file_path}' exists at both refs and has no content or mode changes.",
                }

            performance_keywords = (
                "attention", "moe", "fused_moe", "quantization", "fp8",
                "kernel", "csrc", "distributed", "scheduler", "cache",
                "activation", "layernorm", "linear", "sampler", "deep_gemm",
            )
            changed.sort(
                key=lambda item: (
                    -sum(keyword in item["filename"].lower() for keyword in performance_keywords),
                    item["filename"],
                )
            )
            selected = changed[:max_files]
            files_result = []
            total_patch_bytes = 0
            patch_budget_exhausted = False

            for item in selected:
                path = item["filename"]
                base_entry = item["base_entry"]
                comparison_entry = item["comparison_entry"]
                file_result = {
                    "filename": path,
                    "status": item["status"],
                    "base_sha": base_entry.get("sha") if base_entry else None,
                    "comparison_sha": comparison_entry.get("sha") if comparison_entry else None,
                    "base_url": (
                        f"https://github.com/{base_repo}/blob/{quote(base_ref, safe='/')}/{quote(path, safe='/')}"
                        if base_entry and base_entry.get("type") == "blob" else None
                    ),
                    "comparison_url": (
                        f"https://github.com/{comparison_repo}/blob/{quote(comparison_ref, safe='/')}/{quote(path, safe='/')}"
                        if comparison_entry and comparison_entry.get("type") == "blob" else None
                    ),
                    "patch": None,
                    "patch_available": False,
                }
                if patch_budget_exhausted:
                    file_result["patch_message"] = "Overall patch response budget was exhausted."
                    files_result.append(file_result)
                    continue

                if any(
                    entry is not None and entry.get("type") != "blob"
                    for entry in (base_entry, comparison_entry)
                ):
                    file_result["patch_message"] = "Submodule/tree entries are listed but not rendered as file patches."
                    files_result.append(file_result)
                    continue

                if any(
                    entry is not None and entry.get("size", 0) > _MAX_CROSS_REPO_BLOB_BYTES
                    for entry in (base_entry, comparison_entry)
                ):
                    file_result["patch_message"] = (
                        f"Text patches are limited to {_MAX_CROSS_REPO_BLOB_BYTES} bytes per source file."
                    )
                    files_result.append(file_result)
                    continue

                base_content = ""
                comparison_content = ""
                blob_error = None
                for repo, entry, label in (
                    (base_repo, base_entry, "base"),
                    (comparison_repo, comparison_entry, "comparison"),
                ):
                    if entry is None:
                        continue
                    blob_result = await _fetch_github_blob(client, repo, entry["sha"])
                    if blob_result["status"] != "success":
                        blob_error = blob_result.get("message", "Could not read source blob.")
                        break
                    if label == "base":
                        base_content = blob_result["content"]
                    else:
                        comparison_content = blob_result["content"]
                if blob_error:
                    file_result["patch_message"] = blob_error
                    files_result.append(file_result)
                    continue

                base_label = f"{base_repo}@{base_ref}/{path}"
                comparison_label = f"{comparison_repo}@{comparison_ref}/{path}"
                patch_lines = list(
                    difflib.unified_diff(
                        base_content.splitlines(),
                        comparison_content.splitlines(),
                        fromfile=base_label,
                        tofile=comparison_label,
                        lineterm="",
                    )
                )
                patch = "\n".join(patch_lines)
                if patch:
                    remaining = _MAX_CROSS_REPO_PATCH_BYTES - total_patch_bytes
                    if len(patch) > remaining:
                        patch = patch[:max(0, remaining)] + "\n... [overall patch truncated]"
                        file_result["patch_truncated"] = True
                        patch_budget_exhausted = True
                    else:
                        file_result["patch_truncated"] = False
                    total_patch_bytes += len(patch)
                    file_result["additions"] = sum(
                        line.startswith("+") and not line.startswith("+++")
                        for line in patch_lines
                    )
                    file_result["deletions"] = sum(
                        line.startswith("-") and not line.startswith("---")
                        for line in patch_lines
                    )
                    file_result["patch"] = patch
                    file_result["patch_available"] = True
                else:
                    file_result["patch_message"] = "Tree metadata changed, but decoded text content is identical."
                files_result.append(file_result)

            status_counts = {
                status: sum(item["status"] == status for item in changed)
                for status in ("added", "removed", "modified")
            }
            return {
                "status": "success",
                "base": {
                    "repo": base_repo,
                    "ref": base_ref,
                    "commit_sha": base_tree["commit_sha"],
                    "tree_sha": base_tree["tree_sha"],
                },
                "comparison": {
                    "repo": comparison_repo,
                    "ref": comparison_ref,
                    "commit_sha": comparison_tree["commit_sha"],
                    "tree_sha": comparison_tree["tree_sha"],
                },
                "total_files_changed": len(changed),
                "changed_paths": [
                    item["filename"] for item in changed[:_MAX_CROSS_REPO_PATHS]
                ],
                "changed_paths_returned": min(len(changed), _MAX_CROSS_REPO_PATHS),
                "changed_paths_truncated": len(changed) > _MAX_CROSS_REPO_PATHS,
                "files_returned": len(files_result),
                "files": files_result,
                "stats": status_counts,
                "patches_truncated": len(changed) > len(files_result) or patch_budget_exhausted,
                "truncated": (
                    len(changed) > len(files_result)
                    or len(changed) > _MAX_CROSS_REPO_PATHS
                    or patch_budget_exhausted
                ),
                "message": (
                    f"Cross-repository diff found {len(changed)} changed paths; "
                    f"returning {len(files_result)} prioritized files."
                ),
            }
    except httpx.TimeoutException:
        return {"status": "error", "message": "GitHub API request timed out during cross-repository comparison."}
    except Exception as e:
        logger.error(f"Error comparing vLLM repositories: {e}")
        return {"status": "error", "message": f"Failed to compare repositories: {str(e)}"}
