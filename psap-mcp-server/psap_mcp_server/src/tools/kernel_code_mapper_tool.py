"""MCP tool for mapping PyTorch kernel names to vLLM source code locations.

This tool helps correlate kernel-level performance data from PyTorch profiler
with the actual vLLM source code that implements those operations. Useful for
understanding which code paths are responsible for performance characteristics.
"""

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
    """Normalize version string to GitHub tag format."""
    version = version.strip()
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
) -> Dict[str, Any]:
    """Map a PyTorch kernel/operation name to vLLM source code locations.
    
    Given a kernel name from PyTorch profiler traces, find the likely vLLM
    source files that implement or invoke that kernel. Useful for correlating
    performance data with actual code to understand implementation details.
    
    TOOL_NAME=map_kernel_to_vllm_code
    DISPLAY_NAME=Map Kernel to vLLM Code
    USECASE=Map PyTorch profiler kernel names to vLLM source code files. Use this after identifying slow kernels with compare_pytorch_profiles to understand the code responsible for performance characteristics.
    INSTRUCTIONS=1. Provide a kernel name from PyTorch profiler (e.g., "flash_attn_v2_fwd", "fused_moe", "aten::mm"), 2. Optionally specify a vLLM version, 3. Results include likely source files and GitHub URLs
    INPUT_DESCRIPTION=kernel_name (str): Kernel name from PyTorch profiler; version (str): vLLM version for GitHub URLs (default: v0.13.0); search_github (bool): Also search GitHub for the kernel name
    OUTPUT_DESCRIPTION=Dictionary with likely source files, confidence levels, and GitHub URLs
    EXAMPLES=map_kernel_to_vllm_code("flash_attn_v2_fwd"), map_kernel_to_vllm_code("fused_moe_kernel", version="v0.11.2")
    PREREQUISITES=None
    RELATED_TOOLS=compare_pytorch_profiles, analyze_pytorch_profile, compare_vllm_versions, get_vllm_pull_request
    
    Args:
        kernel_name: The kernel name from PyTorch profiler (e.g., "flash_attn_v2_fwd")
        version: vLLM version for GitHub URLs (default: "v0.13.0")
        search_github: If True, also search GitHub API for the kernel name
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - kernel_name: The input kernel name
        - is_pytorch_stdlib: Whether this is a standard PyTorch operation
        - likely_source_files: List of likely source file locations
        - github_urls: GitHub URLs to view the files
        - search_suggestions: How to find more information
    """
    try:
        version = _normalize_version(version)
        
        # Check if it's a standard PyTorch op
        is_stdlib = _is_pytorch_stdlib(kernel_name)
        
        # Find mappings
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
        if is_stdlib:
            suggestions.append("This is a standard PyTorch operation implemented in PyTorch core, not vLLM")
            suggestions.append("Performance may be affected by how vLLM uses this op (tensor shapes, data types)")
        elif not mappings:
            suggestions.append(f"No curated mapping found for '{kernel_name}'")
            suggestions.append(f"Try searching GitHub: {_build_github_search_url(kernel_name)}")
            suggestions.append("Check if this kernel is from a third-party library (e.g., flash-attn, triton)")
        else:
            suggestions.append(f"Use get_vllm_pull_request to find PRs that modified these files")
            suggestions.append(f"Use compare_vllm_versions to see changes between releases")
        
        return {
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
