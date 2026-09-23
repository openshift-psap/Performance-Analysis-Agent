"""MCP tool for mapping PyTorch kernel names to vLLM source code locations.

This tool helps correlate kernel-level performance data from PyTorch profiler
with the actual vLLM source code that implements those operations. Useful for
understanding which code paths are responsible for performance characteristics.

Also provides dynamic source code fetching and cross-version diff capabilities
via the GitHub Contents and Compare APIs, enabling the agent to read actual
vLLM implementation code and see exactly what changed between releases.
"""

import base64
import re
from typing import Any, Dict, List, Optional

import httpx

from psap_mcp_server.src.settings import settings
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
VLLM_REPO = "vllm-project/vllm"

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


def _normalize_version(version: str) -> str:
    """Normalize version string to a vLLM GitHub tag.

    Handles RHAIIS product versions (e.g. ``RHAIIS-3.3``) by resolving them
    through the version mappings file, as well as plain vLLM versions
    (e.g. ``0.13.0`` or ``v0.13.0``).
    """
    version = version.strip()

    if version.upper().startswith("RHAIIS"):
        try:
            from psap_mcp_server.src.tools.vllm_release_notes_tool import (
                _resolve_to_vllm_version,
            )
            resolved, _, was_mapped = _resolve_to_vllm_version(version)
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


def _build_github_url(path: str, version: str) -> str:
    """Build GitHub URL for a file path at a specific version."""
    return f"https://github.com/{VLLM_REPO}/blob/{version}/{path}"


def _build_github_search_url(kernel_name: str) -> str:
    """Build GitHub search URL for a kernel name."""
    # Clean up kernel name for search
    search_term = kernel_name.replace("::", " ").replace("_", " ")
    return f"https://github.com/{VLLM_REPO}/search?q={search_term}"


async def map_kernel_to_vllm_code(
    kernel_name: str,
    version: str = "v0.13.0",
    search_github: bool = False,
    profile_model: Optional[str] = None,
    profile_tp: Optional[str] = None,
    profile_workload: Optional[str] = None,
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
    INPUT_DESCRIPTION=kernel_name (str): Kernel name from PyTorch profiler; version (str): vLLM version for GitHub URLs (default: v0.13.0); search_github (bool): Also search GitHub for the kernel name; profile_model (str, optional): Model name to look up real stacks; profile_tp (str, optional): TP config; profile_workload (str, optional): ISL/OSL workload
    OUTPUT_DESCRIPTION=Dictionary with likely source files, confidence levels, GitHub URLs, and real call stack attribution when available
    EXAMPLES=map_kernel_to_vllm_code("flash_attn_v2_fwd"), map_kernel_to_vllm_code("fused_moe_kernel", version="v0.21.0", profile_model="deepseek-r1", profile_tp="tp8", profile_workload="isl1000_osl1000")
    PREREQUISITES=None
    RELATED_TOOLS=compare_pytorch_profiles, analyze_pytorch_profile, compare_vllm_versions, get_vllm_pull_request
    
    Args:
        kernel_name: The kernel name from PyTorch profiler (e.g., "flash_attn_v2_fwd")
        version: vLLM version for GitHub URLs (default: "v0.13.0")
        search_github: If True, also search GitHub API for the kernel name
        profile_model: Model name to load profile stacks from (e.g., "deepseek-r1")
        profile_tp: Tensor parallelism config (e.g., "tp8")
        profile_workload: ISL/OSL workload (e.g., "isl1000_osl1000")
    
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
        version = _normalize_version(version)
        
        # Check if it's a standard PyTorch op
        is_stdlib = _is_pytorch_stdlib(kernel_name)
        
        # Attempt to load real call stacks from profile data
        profile_call_stacks = None
        if profile_model and not is_stdlib:
            try:
                from psap_mcp_server.src.tools.pytorch_profile_tool import (
                    _resolve_profile,
                    _load_or_extract_stats,
                    _extract_bare_version,
                    _match_version,
                )
                bare_version = f"vLLM-{_extract_bare_version(version)}"
                model_key, matched_tp, matched_version, matched_workload, error, index = _resolve_profile(
                    profile_model, profile_tp, bare_version, profile_workload
                )
                if not error and model_key and matched_tp and matched_version and matched_workload:
                    stats = _load_or_extract_stats(model_key, matched_tp, matched_version, matched_workload, 0)
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
                    "url": f"https://github.com/{VLLM_REPO}/tree/{version}/{mapping['path'].rstrip('/')}",
                    "type": "directory",
                })
            else:
                github_urls.append({
                    "path": mapping["path"],
                    "url": _build_github_url(mapping["path"], version),
                    "type": "file",
                })
        
        # GitHub search (optional)
        github_search_results = None
        if search_github and not is_stdlib:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    headers = _get_github_headers()
                    search_query = kernel_name.replace("::", " ").replace("_", "+")
                    search_url = f"{GITHUB_API_BASE}/search/code?q={search_query}+repo:{VLLM_REPO}&per_page=10"
                    
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
            suggestions.append(f"Try searching GitHub: {_build_github_search_url(kernel_name)}")
            suggestions.append("Check if this kernel is from a third-party library (e.g., flash-attn, triton)")
        else:
            suggestions.append(f"Use get_vllm_pull_request to find PRs that modified these files")
            suggestions.append(f"Use compare_vllm_versions to see changes between releases")
        
        result = {
            "status": "success",
            "kernel_name": kernel_name,
            "version": version,
            "is_pytorch_stdlib": is_stdlib,
            "likely_source_files": likely_files,
            "github_urls": github_urls,
            "github_search_results": github_search_results,
            "search_suggestions": suggestions,
            "message": f"Found {len(likely_files)} likely source location(s) for '{kernel_name}'" if likely_files 
                      else f"No specific mapping for '{kernel_name}' - try GitHub search",
            "github_manual_search_url": _build_github_search_url(kernel_name),
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
) -> Dict[str, Any]:
    """Correlate a kernel's performance change with vLLM code changes.
    
    Given a kernel that shows performance regression or improvement, find
    related code changes between versions. Combines kernel mapping with
    version comparison to help identify root causes.
    
    TOOL_NAME=correlate_kernel_with_changes
    DISPLAY_NAME=Correlate Kernel with Code Changes
    USECASE=Investigate why a specific kernel's performance changed between versions. Use after compare_pytorch_profiles identifies a regression or improvement to find the responsible code changes.
    INSTRUCTIONS=1. Provide the kernel name showing performance change, 2. Specify the two versions being compared, 3. Results combine kernel mapping with changelog analysis
    INPUT_DESCRIPTION=kernel_name (str): Kernel showing performance change; version1 (str): Baseline version; version2 (str): Comparison version
    OUTPUT_DESCRIPTION=Dictionary with kernel mapping, related code areas, and suggestions for investigation
    EXAMPLES=correlate_kernel_with_changes("flash_attn_v2_fwd", "v0.11.2", "v0.13.0")
    PREREQUISITES=None
    RELATED_TOOLS=compare_pytorch_profiles, map_kernel_to_vllm_code, compare_vllm_versions, get_vllm_pull_request
    
    Args:
        kernel_name: The kernel showing performance change
        version1: Baseline vLLM version
        version2: Comparison vLLM version
    
    Returns:
        Dictionary with correlation analysis and investigation suggestions.
    """
    try:
        version1 = _normalize_version(version1)
        version2 = _normalize_version(version2)
        
        # Get kernel mapping
        mapping_result = await map_kernel_to_vllm_code(kernel_name, version2)
        
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
        
        investigation_steps.append(
            f"Use compare_vllm_versions('{version1}', '{version2}', focus_areas={focus_areas}) "
            "to find related release notes"
        )
        investigation_steps.append(
            "Look for PR numbers in release notes and use get_vllm_pull_request() to investigate"
        )
        
        return {
            "status": "success",
            "kernel_name": kernel_name,
            "version_range": {"from": version1, "to": version2},
            "kernel_mapping": mapping_result,
            "suggested_focus_areas": focus_areas,
            "investigation_steps": investigation_steps,
            "quick_links": {
                "github_compare": f"https://github.com/{VLLM_REPO}/compare/{version1}...{version2}",
                "github_search": _build_github_search_url(kernel_name),
            },
            "message": f"Correlation analysis for '{kernel_name}' between {version1} and {version2}",
            "next_tool_calls": [
                f"compare_vllm_versions('{version1}', '{version2}', focus_areas={focus_areas})",
                f"get_vllm_release_notes(version='{version2}')",
            ],
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
    version/ref. Defaults to the vLLM repository but can target any public
    GitHub repo (e.g., "pytorch/pytorch") for verifying upstream dependencies.
    
    No pre-cloning is needed -- files are fetched on demand via the GitHub
    Contents API.
    
    TOOL_NAME=fetch_vllm_source
    DISPLAY_NAME=Fetch GitHub Source Code
    USECASE=Read source code at a specific version. Defaults to vLLM but can fetch from any public GitHub repo. Use to read kernel implementations, dependency pins (e.g., PyTorch's triton_version.txt), or configuration files.
    INSTRUCTIONS=1. Provide the file path relative to the repo root, 2. Specify the version tag or ref, 3. Optionally set repo to fetch from a different GitHub repo (e.g., "pytorch/pytorch")
    INPUT_DESCRIPTION=file_path (str): Path relative to repo root; version (str): Git tag (default: v0.13.0); repo (str, optional): GitHub repo in "owner/name" format (default: vllm-project/vllm)
    OUTPUT_DESCRIPTION=Dictionary with file content, metadata, and GitHub URL
    EXAMPLES=fetch_vllm_source("vllm/model_executor/layers/fused_moe/fused_moe.py", "v0.13.0"), fetch_vllm_source(".ci/docker/triton_version.txt", "v2.9.1", repo="pytorch/pytorch"), fetch_vllm_source("requirements/cuda.txt", "v0.16.0")
    PREREQUISITES=Internet access to GitHub API. Optional: GITHUB_TOKEN for higher rate limits
    RELATED_TOOLS=map_kernel_to_vllm_code, get_vllm_code_diff, correlate_kernel_with_changes
    
    Args:
        file_path: Path relative to the repo root.
        version: Git tag or ref to fetch from (default: "v0.13.0").
        repo: GitHub repository in "owner/name" format (default: vllm-project/vllm).
              Use for upstream dependency verification, e.g., "pytorch/pytorch".
    
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
        target_repo = repo or VLLM_REPO
        if target_repo == VLLM_REPO:
            version = _normalize_version(version)
        file_path = file_path.lstrip("/")

        url = f"{GITHUB_API_BASE}/repos/{target_repo}/contents/{file_path}"
        params = {"ref": version}

        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = _get_github_headers()
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 404:
                return {
                    "status": "error",
                    "message": f"File '{file_path}' not found at version {version}.",
                    "suggestion": f"Check the path exists: https://github.com/{target_repo}/tree/{version}/{file_path}",
                }
            if response.status_code == 403:
                return {
                    "status": "error",
                    "message": "GitHub API rate limit exceeded. Set GITHUB_TOKEN for higher limits.",
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
                    "github_url": f"https://github.com/{target_repo}/tree/{version}/{file_path}",
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
                "github_url": f"https://github.com/{target_repo}/blob/{version}/{file_path}",
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
) -> Dict[str, Any]:
    """Fetch actual code diff between two vLLM versions, optionally for a specific file.
    
    Uses the GitHub Compare API to retrieve patch content showing exactly what
    lines changed between two releases. This is the key tool for explaining
    the mechanism behind performance changes -- it lets you see the actual
    code modifications.
    
    TOOL_NAME=get_vllm_code_diff
    DISPLAY_NAME=Get vLLM Code Diff
    USECASE=See exactly what code changed between two vLLM versions. Use after identifying performance-relevant source files with map_kernel_to_vllm_code to understand the mechanism behind kernel timing changes.
    INSTRUCTIONS=1. Provide two version tags, 2. Optionally scope to a specific file path, 3. Without file_path returns top changed performance-relevant files with patches
    INPUT_DESCRIPTION=version1 (str): Baseline version tag; version2 (str): Comparison version tag; file_path (str, optional): Specific file to diff; max_files (int): Max files to return patches for (default 10)
    OUTPUT_DESCRIPTION=Dictionary with patch content for changed files, stats, and GitHub compare URL
    EXAMPLES=get_vllm_code_diff("v0.11.2", "v0.13.0", "vllm/model_executor/layers/fused_moe/fused_moe.py"), get_vllm_code_diff("v0.11.2", "v0.13.0")
    PREREQUISITES=Internet access to GitHub API. Optional: GITHUB_TOKEN for higher rate limits
    RELATED_TOOLS=fetch_vllm_source, map_kernel_to_vllm_code, compare_vllm_versions, correlate_kernel_with_changes
    
    Args:
        version1: Baseline version (e.g., "v0.11.2").
        version2: Comparison version (e.g., "v0.13.0").
        file_path: If provided, only return the diff for this file.
        max_files: Maximum number of file patches to return (default: 10).
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - version_range: The two versions compared
        - files: List of changed files with patch content
        - stats: Overall diff statistics
        - github_compare_url: Link to the full comparison on GitHub
    """
    try:
        version1 = _normalize_version(version1)
        version2 = _normalize_version(version2)

        url = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/compare/{version1}...{version2}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = _get_github_headers()
            response = await client.get(url, headers=headers)

            if response.status_code == 404:
                return {
                    "status": "error",
                    "message": f"Could not compare {version1}...{version2}. Check that both tags exist.",
                }
            if response.status_code == 403:
                return {
                    "status": "error",
                    "message": "GitHub API rate limit exceeded. Set GITHUB_TOKEN for higher limits.",
                }
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"GitHub API error: HTTP {response.status_code}",
                    "details": response.text[:500],
                }

            data = response.json()

            total_commits = len(data.get("commits", []))
            all_files = data.get("files", [])

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
                        "status": "error",
                        "message": f"File '{file_path}' was not changed between {version1} and {version2}.",
                        "suggestion": "The file may not have been modified, or the path may be wrong.",
                        "performance_related_changed_files": available,
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
                    "patch": patch,
                    "patch_truncated": truncated,
                })

                if total_patch_size >= patch_budget:
                    break

            return {
                "status": "success",
                "version_range": {"from": version1, "to": version2},
                "total_commits": total_commits,
                "total_files_changed": len(all_files),
                "files_returned": len(files_result),
                "files": files_result,
                "stats": {
                    "total_additions": sum(f.get("additions", 0) for f in all_files),
                    "total_deletions": sum(f.get("deletions", 0) for f in all_files),
                },
                "github_compare_url": f"https://github.com/{VLLM_REPO}/compare/{version1}...{version2}",
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
