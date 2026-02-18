"""MCP tool for fetching vLLM release notes from GitHub.

This tool fetches release information and changelogs from the vLLM GitHub repository
to help explain performance changes between different vLLM versions.

Supports automatic resolution of RHAIIS product versions to upstream vLLM versions
using a configurable mapping file.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import re

import httpx

from psap_mcp_server.src.settings import settings
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
VLLM_REPO = "vllm-project/vllm"

# Version mappings file path
VERSION_MAPPINGS_PATH = Path(__file__).parent / "version_mappings.json"


def _load_version_mappings() -> Dict[str, str]:
    """Load RHAIIS to vLLM version mappings from config file.
    
    Returns:
        Dictionary mapping RHAIIS versions to vLLM versions.
    """
    try:
        if VERSION_MAPPINGS_PATH.exists():
            with open(VERSION_MAPPINGS_PATH) as f:
                data = json.load(f)
                return data.get("rhaiis_to_vllm", {})
    except Exception as e:
        logger.warning(f"Could not load version mappings: {e}")
    return {}


def _resolve_to_vllm_version(version: str) -> Tuple[str, Optional[str], bool]:
    """Resolve a version string to an upstream vLLM version.
    
    Handles both RHAIIS product versions (e.g., "RHAIIS-3.2.5") and 
    direct vLLM versions (e.g., "0.8.0" or "v0.8.0").
    
    Args:
        version: Version string (RHAIIS or vLLM format)
        
    Returns:
        Tuple of (resolved_vllm_version, original_rhaiis_version_or_none, was_mapped)
        - resolved_vllm_version: The vLLM version in "vX.Y.Z" format
        - original_rhaiis_version_or_none: The RHAIIS version if it was mapped, None otherwise
        - was_mapped: True if the version was mapped from RHAIIS
    """
    version = version.strip()
    
    # Check if it's a RHAIIS version
    if version.upper().startswith("RHAIIS"):
        # Normalize to proper case
        normalized = version.upper().replace("RHAIIS", "RHAIIS")
        
        # Load mappings
        mappings = _load_version_mappings()
        
        # Try exact match first
        if normalized in mappings:
            return mappings[normalized], normalized, True
        
        # Try case-insensitive match
        for rhaiis_ver, vllm_ver in mappings.items():
            if rhaiis_ver.upper() == normalized.upper():
                return vllm_ver, rhaiis_ver, True
        
        # Not found in mappings
        return version, normalized, False
    
    # It's a direct vLLM version - normalize format
    if not version.startswith("v"):
        version = f"v{version}"
    
    return version, None, False


async def get_version_mappings() -> Dict[str, Any]:
    """Get RHAIIS to vLLM version mappings.
    
    Returns the configured mappings between RHAIIS product versions and their 
    corresponding upstream vLLM versions. Use this to understand which vLLM 
    version underlies a specific RHAIIS release.
    
    TOOL_NAME=get_version_mappings
    DISPLAY_NAME=Get RHAIIS to vLLM Version Mappings
    USECASE=View the mapping between RHAIIS product versions and upstream vLLM versions. Useful to understand which vLLM changes apply to a specific RHAIIS release.
    INSTRUCTIONS=Call without arguments to see all available mappings
    INPUT_DESCRIPTION=No arguments required
    OUTPUT_DESCRIPTION=Dictionary with all RHAIIS to vLLM version mappings and instructions for updating
    EXAMPLES=get_version_mappings()
    PREREQUISITES=None
    RELATED_TOOLS=get_vllm_release_notes, compare_vllm_versions
    
    Returns:
        Dictionary containing:
        - status: "success"
        - mappings: Dictionary of RHAIIS version -> vLLM version
        - available_rhaiis_versions: List of RHAIIS versions with mappings
        - mapping_file: Path to the mappings config file
        - instructions: How to add new mappings
    """
    mappings = _load_version_mappings()
    
    return {
        "status": "success",
        "mappings": mappings,
        "available_rhaiis_versions": list(mappings.keys()),
        "count": len(mappings),
        "mapping_file": str(VERSION_MAPPINGS_PATH),
        "instructions": "To add a new mapping, edit version_mappings.json and add an entry like: \"RHAIIS-X.Y.Z\": \"vA.B.C\"",
        "note": "RHAIIS versions are automatically resolved to vLLM versions when using get_vllm_release_notes() or compare_vllm_versions()",
    }

# Performance-related keywords to highlight in release notes
PERFORMANCE_KEYWORDS = [
    "performance", "throughput", "latency", "memory", "gpu", "cuda",
    "optimization", "speedup", "faster", "slower", "regression",
    "kernel", "attention", "flash", "paged", "kv cache", "prefill",
    "decode", "batch", "scheduling", "tensor parallel", "parallelism",
    "quantization", "fp8", "int8", "awq", "gptq", "speculative",
    "chunked", "prefix caching", "continuous batching", "vram",
    "deepseek", "llama", "mistral", "qwen", "mixtral",
]


def _get_github_headers() -> Dict[str, str]:
    """Get headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # Add token if configured for higher rate limits
    github_token = getattr(settings, 'GITHUB_TOKEN', None)
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def _normalize_version(version: str) -> str:
    """Normalize version string to match GitHub tag format.
    
    vLLM uses 'v0.6.0' format for tags.
    """
    version = version.strip()
    if not version.startswith("v"):
        version = f"v{version}"
    return version


def _extract_performance_changes(body: str) -> List[Dict[str, str]]:
    """Extract performance-related changes from release notes body.
    
    Searches for lines containing performance keywords and returns them
    with context about what type of change it is.
    """
    if not body:
        return []
    
    changes = []
    lines = body.split("\n")
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check if line contains any performance keyword
        matched_keywords = [kw for kw in PERFORMANCE_KEYWORDS if kw in line_lower]
        
        if matched_keywords and line.strip():
            # Clean up the line
            clean_line = line.strip()
            # Remove markdown list markers
            clean_line = re.sub(r'^[\*\-\+]\s*', '', clean_line)
            # Remove PR references for cleaner output
            pr_match = re.search(r'\(#(\d+)\)', clean_line)
            pr_number = pr_match.group(1) if pr_match else None
            
            # Categorize the change
            category = "general"
            if any(kw in line_lower for kw in ["memory", "vram", "kv cache", "gpu"]):
                category = "memory"
            elif any(kw in line_lower for kw in ["latency", "ttft", "tpot", "e2e"]):
                category = "latency"
            elif any(kw in line_lower for kw in ["throughput", "tok/s", "tokens per second"]):
                category = "throughput"
            elif any(kw in line_lower for kw in ["kernel", "cuda", "attention", "flash"]):
                category = "kernel/attention"
            elif any(kw in line_lower for kw in ["quantization", "fp8", "int8", "awq", "gptq"]):
                category = "quantization"
            elif any(kw in line_lower for kw in ["batch", "scheduling", "continuous"]):
                category = "scheduling"
            elif any(kw in line_lower for kw in ["speculative", "draft"]):
                category = "speculative_decoding"
            
            changes.append({
                "text": clean_line[:500],  # Limit length
                "category": category,
                "keywords": matched_keywords[:5],  # Limit keywords
                "pr_number": pr_number,
            })
    
    return changes


async def get_vllm_release_notes(
    version: Optional[str] = None,
    include_all_releases: bool = False,
    max_releases: int = 10,
    performance_only: bool = False,
) -> Dict[str, Any]:
    """Get vLLM release notes from GitHub.
    
    Supports both RHAIIS product versions (e.g., "RHAIIS-3.2.5") and upstream vLLM 
    versions (e.g., "0.8.0"). RHAIIS versions are automatically mapped to their 
    corresponding upstream vLLM versions using the version_mappings.json config file.
    
    TOOL_NAME=get_vllm_release_notes
    DISPLAY_NAME=Get vLLM Release Notes
    USECASE=Fetch release notes and changelogs from vLLM GitHub repository to understand what changed in specific versions. Supports RHAIIS versions (auto-mapped to upstream vLLM) and direct vLLM versions.
    INSTRUCTIONS=1. Specify a version (RHAIIS like "RHAIIS-3.2.5" or vLLM like "0.8.0"), 2. Set performance_only=True to filter for performance-related changes only, 3. Use to correlate version changes with benchmark results
    INPUT_DESCRIPTION=version (str, optional): Version - supports RHAIIS format "RHAIIS-3.2.5" or vLLM format "0.8.0"/"v0.8.0"; include_all_releases (bool): Get multiple releases; max_releases (int): Max releases to return; performance_only (bool): Filter for performance changes only
    OUTPUT_DESCRIPTION=Dictionary with release information including version, date, release notes body, and performance-related changes. For RHAIIS versions, includes the mapping used.
    EXAMPLES=get_vllm_release_notes(version="RHAIIS-3.2.5"), get_vllm_release_notes(version="0.8.0"), get_vllm_release_notes(performance_only=True)
    PREREQUISITES=Internet access to GitHub API. Optional: GITHUB_TOKEN for higher rate limits
    RELATED_TOOLS=analyze_regression, compare_configurations, compare_vllm_versions
    
    Args:
        version: Version to fetch - supports RHAIIS format (e.g., "RHAIIS-3.2.5") or 
                 vLLM format (e.g., "0.8.0" or "v0.8.0"). RHAIIS versions are 
                 automatically mapped to upstream vLLM versions.
        include_all_releases: If True and no version specified, return multiple releases
        max_releases: Maximum number of releases to return (default: 10)
        performance_only: If True, only return performance-related changes
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - releases: List of release information with notes and performance changes
        - version_mapping: If RHAIIS version was used, shows the mapping
        - message: Status message
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = _get_github_headers()
            
            version_mapping_info = None
            
            if version:
                # Resolve version (handles RHAIIS -> vLLM mapping)
                resolved_version, rhaiis_version, was_mapped = _resolve_to_vllm_version(version)
                
                # Check if RHAIIS version couldn't be mapped
                if rhaiis_version and not was_mapped:
                    available_mappings = _load_version_mappings()
                    return {
                        "status": "error",
                        "message": f"RHAIIS version '{rhaiis_version}' not found in version mappings.",
                        "available_rhaiis_versions": list(available_mappings.keys()),
                        "suggestion": "Update version_mappings.json with the correct RHAIIS to vLLM mapping, or provide the vLLM version directly (e.g., 'v0.8.0')",
                        "mapping_file": str(VERSION_MAPPINGS_PATH),
                    }
                
                if was_mapped:
                    version_mapping_info = {
                        "rhaiis_version": rhaiis_version,
                        "vllm_version": resolved_version,
                        "note": f"Mapped {rhaiis_version} → {resolved_version}",
                    }
                
                # Fetch specific version
                tag = resolved_version  # Already normalized by _resolve_to_vllm_version
                url = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/releases/tags/{tag}"
                
                response = await client.get(url, headers=headers)
                
                if response.status_code == 404:
                    return {
                        "status": "error",
                        "message": f"Version {tag} not found. vLLM uses format 'v0.6.0' for release tags.",
                        "suggestion": "Use get_vllm_release_notes(include_all_releases=True) to see available versions",
                    }
                elif response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"GitHub API error: HTTP {response.status_code}",
                        "details": response.text[:500],
                    }
                
                release_data = response.json()
                releases = [release_data]
            else:
                # Fetch multiple releases
                url = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/releases"
                params = {"per_page": min(max_releases, 100)}
                
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"GitHub API error: HTTP {response.status_code}",
                        "details": response.text[:500],
                    }
                
                releases = response.json()[:max_releases]
            
            # Process releases
            processed_releases = []
            for release in releases:
                body = release.get("body", "") or ""
                performance_changes = _extract_performance_changes(body)
                
                # If performance_only, skip releases with no performance changes
                if performance_only and not performance_changes:
                    continue
                
                release_info = {
                    "version": release.get("tag_name"),
                    "name": release.get("name"),
                    "published_at": release.get("published_at"),
                    "html_url": release.get("html_url"),
                    "prerelease": release.get("prerelease", False),
                    "performance_changes": performance_changes,
                    "performance_change_count": len(performance_changes),
                }
                
                # Include full body if not filtering for performance only
                if not performance_only:
                    # Truncate very long bodies
                    release_info["body"] = body[:10000] if len(body) > 10000 else body
                    release_info["body_truncated"] = len(body) > 10000
                
                processed_releases.append(release_info)
            
            if not processed_releases:
                result = {
                    "status": "success",
                    "releases": [],
                    "message": "No releases found matching criteria" + (
                        " (no performance-related changes found)" if performance_only else ""
                    ),
                }
                if version_mapping_info:
                    result["version_mapping"] = version_mapping_info
                return result
            
            result = {
                "status": "success",
                "releases": processed_releases,
                "count": len(processed_releases),
                "message": f"Retrieved {len(processed_releases)} release(s)" + (
                    " with performance-related changes" if performance_only else ""
                ),
                "github_repo": f"https://github.com/{VLLM_REPO}",
            }
            if version_mapping_info:
                result["version_mapping"] = version_mapping_info
            return result
            
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "GitHub API request timed out",
        }
    except Exception as e:
        logger.error(f"Error fetching vLLM release notes: {e}")
        return {
            "status": "error",
            "message": f"Failed to fetch release notes: {str(e)}",
        }


async def compare_vllm_versions(
    version1: str,
    version2: str,
    focus_areas: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare release notes between two vLLM versions to identify relevant changes.
    
    Supports both RHAIIS product versions (e.g., "RHAIIS-3.2.4") and upstream vLLM 
    versions (e.g., "0.7.3"). RHAIIS versions are automatically mapped to their 
    corresponding upstream vLLM versions.
    
    TOOL_NAME=compare_vllm_versions
    DISPLAY_NAME=Compare vLLM Version Changes
    USECASE=Compare changelogs between two versions to identify changes that might explain performance differences. Supports RHAIIS versions (auto-mapped) and vLLM versions. Particularly useful when analyzing benchmark regressions.
    INSTRUCTIONS=1. Provide two versions to compare (older first) - can be RHAIIS or vLLM format, 2. Optionally specify focus_areas like ["memory", "latency", "throughput"], 3. Use results to correlate code changes with benchmark data
    INPUT_DESCRIPTION=version1 (str): First/older version (e.g., "RHAIIS-3.2.4" or "0.7.3"); version2 (str): Second/newer version (e.g., "RHAIIS-3.2.5" or "0.8.0"); focus_areas (list, optional): Categories to focus on
    OUTPUT_DESCRIPTION=Dictionary with changes between versions, categorized by type. Includes version mapping info if RHAIIS versions were used.
    EXAMPLES=compare_vllm_versions("RHAIIS-3.2.4", "RHAIIS-3.2.5"), compare_vllm_versions("0.7.3", "0.8.0", focus_areas=["memory", "latency"])
    PREREQUISITES=Internet access to GitHub API
    RELATED_TOOLS=get_vllm_release_notes, analyze_regression
    
    Args:
        version1: First/older version - supports RHAIIS format (e.g., "RHAIIS-3.2.4") 
                  or vLLM format (e.g., "0.7.3" or "v0.7.3")
        version2: Second/newer version - supports RHAIIS format or vLLM format
        focus_areas: Optional list of categories to focus on:
            - "memory": Memory usage, KV cache, VRAM changes
            - "latency": TTFT, TPOT, E2E latency changes
            - "throughput": Tokens/second, request rate changes
            - "kernel/attention": CUDA kernels, attention mechanisms
            - "quantization": FP8, INT8, AWQ, GPTQ changes
            - "scheduling": Batching, scheduling changes
            - "speculative_decoding": Speculative decoding changes
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - version_range: Versions being compared (includes RHAIIS mapping if used)
        - changes_by_category: Changes grouped by performance category
        - total_performance_changes: Total number of performance-related changes
        - summary: High-level summary of changes
    """
    try:
        # Resolve versions (handles RHAIIS -> vLLM mapping)
        resolved1, rhaiis1, mapped1 = _resolve_to_vllm_version(version1)
        resolved2, rhaiis2, mapped2 = _resolve_to_vllm_version(version2)
        
        # Check for unmapped RHAIIS versions
        unmapped = []
        if rhaiis1 and not mapped1:
            unmapped.append(rhaiis1)
        if rhaiis2 and not mapped2:
            unmapped.append(rhaiis2)
        
        if unmapped:
            available_mappings = _load_version_mappings()
            return {
                "status": "error",
                "message": f"RHAIIS version(s) not found in mappings: {', '.join(unmapped)}",
                "available_rhaiis_versions": list(available_mappings.keys()),
                "suggestion": "Update version_mappings.json with the correct RHAIIS to vLLM mapping, or provide vLLM versions directly",
                "mapping_file": str(VERSION_MAPPINGS_PATH),
            }
        
        # Build version mapping info
        version_mapping_info = None
        if mapped1 or mapped2:
            version_mapping_info = {}
            if mapped1:
                version_mapping_info["version1"] = {"rhaiis": rhaiis1, "vllm": resolved1}
            if mapped2:
                version_mapping_info["version2"] = {"rhaiis": rhaiis2, "vllm": resolved2}
        
        tag1 = resolved1
        tag2 = resolved2
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = _get_github_headers()
            
            # Fetch all releases to find versions between tag1 and tag2
            url = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/releases"
            params = {"per_page": 100}
            
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"GitHub API error: HTTP {response.status_code}",
                }
            
            all_releases = response.json()
            
            # Find releases in range
            releases_in_range = []
            found_start = False
            found_end = False
            
            # Releases are in reverse chronological order (newest first)
            for release in all_releases:
                tag = release.get("tag_name", "")
                
                if tag == tag2:
                    found_end = True
                    
                if found_end and not found_start:
                    releases_in_range.append(release)
                    
                if tag == tag1:
                    found_start = True
                    break
            
            # If we found end but not start, we might need more releases
            if not found_start:
                # Check if version1 exists
                url_v1 = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/releases/tags/{tag1}"
                resp_v1 = await client.get(url_v1, headers=headers)
                if resp_v1.status_code == 404:
                    return {
                        "status": "error",
                        "message": f"Version {tag1} not found in vLLM releases",
                        "available_versions": [r["tag_name"] for r in all_releases[:20]],
                    }
            
            if not found_end:
                # Check if version2 exists
                url_v2 = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/releases/tags/{tag2}"
                resp_v2 = await client.get(url_v2, headers=headers)
                if resp_v2.status_code == 404:
                    return {
                        "status": "error",
                        "message": f"Version {tag2} not found in vLLM releases",
                        "available_versions": [r["tag_name"] for r in all_releases[:20]],
                    }
            
            # Remove the first version (tag1) from the range since we want changes AFTER it
            if releases_in_range and releases_in_range[-1].get("tag_name") == tag1:
                releases_in_range = releases_in_range[:-1]
            
            if not releases_in_range:
                result = {
                    "status": "success",
                    "version_range": {"from": tag1, "to": tag2},
                    "message": f"No releases found between {tag1} and {tag2}",
                    "changes_by_category": {},
                    "total_performance_changes": 0,
                }
                if version_mapping_info:
                    result["version_mapping"] = version_mapping_info
                return result
            
            # Extract and categorize changes from all releases in range
            changes_by_category: Dict[str, List[Dict]] = {
                "memory": [],
                "latency": [],
                "throughput": [],
                "kernel/attention": [],
                "quantization": [],
                "scheduling": [],
                "speculative_decoding": [],
                "general": [],
            }
            
            versions_included = []
            
            for release in releases_in_range:
                body = release.get("body", "") or ""
                tag = release.get("tag_name", "")
                versions_included.append(tag)
                
                performance_changes = _extract_performance_changes(body)
                
                for change in performance_changes:
                    category = change["category"]
                    
                    # Skip if focus_areas specified and category not in focus
                    if focus_areas and category not in focus_areas:
                        continue
                    
                    change_with_version = {
                        **change,
                        "version": tag,
                        "release_url": release.get("html_url"),
                    }
                    
                    if category in changes_by_category:
                        changes_by_category[category].append(change_with_version)
                    else:
                        changes_by_category["general"].append(change_with_version)
            
            # Remove empty categories
            changes_by_category = {k: v for k, v in changes_by_category.items() if v}
            
            # Count total changes
            total_changes = sum(len(v) for v in changes_by_category.values())
            
            # Generate summary
            summary_parts = []
            if changes_by_category.get("memory"):
                summary_parts.append(f"{len(changes_by_category['memory'])} memory-related changes")
            if changes_by_category.get("latency"):
                summary_parts.append(f"{len(changes_by_category['latency'])} latency-related changes")
            if changes_by_category.get("throughput"):
                summary_parts.append(f"{len(changes_by_category['throughput'])} throughput-related changes")
            if changes_by_category.get("kernel/attention"):
                summary_parts.append(f"{len(changes_by_category['kernel/attention'])} kernel/attention changes")
            if changes_by_category.get("quantization"):
                summary_parts.append(f"{len(changes_by_category['quantization'])} quantization changes")
            
            result = {
                "status": "success",
                "version_range": {
                    "from": tag1,
                    "to": tag2,
                },
                "versions_included": versions_included,
                "releases_analyzed": len(releases_in_range),
                "changes_by_category": changes_by_category,
                "total_performance_changes": total_changes,
                "summary": ", ".join(summary_parts) if summary_parts else "No performance-related changes detected",
                "focus_areas_applied": focus_areas,
                "message": f"Analyzed {len(releases_in_range)} release(s) between {tag1} and {tag2}, found {total_changes} performance-related changes",
                "github_compare_url": f"https://github.com/{VLLM_REPO}/compare/{tag1}...{tag2}",
            }
            
            # Add version mapping info if RHAIIS versions were used
            if version_mapping_info:
                result["version_mapping"] = version_mapping_info
                # Update message to mention RHAIIS versions
                if rhaiis1 and rhaiis2:
                    result["message"] = f"Compared {rhaiis1} ({tag1}) to {rhaiis2} ({tag2}): {total_changes} performance-related changes in {len(releases_in_range)} release(s)"
                elif rhaiis1:
                    result["message"] = f"Compared {rhaiis1} ({tag1}) to {tag2}: {total_changes} performance-related changes in {len(releases_in_range)} release(s)"
                elif rhaiis2:
                    result["message"] = f"Compared {tag1} to {rhaiis2} ({tag2}): {total_changes} performance-related changes in {len(releases_in_range)} release(s)"
            
            return result
            
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "GitHub API request timed out",
        }
    except Exception as e:
        logger.error(f"Error comparing vLLM versions: {e}")
        return {
            "status": "error",
            "message": f"Failed to compare versions: {str(e)}",
        }


def _categorize_pr_labels(labels: List[str]) -> Dict[str, List[str]]:
    """Categorize PR labels into meaningful groups.
    
    Args:
        labels: List of label names from the PR
        
    Returns:
        Dictionary with categorized labels
    """
    categories = {
        "type": [],        # bug, feature, enhancement, etc.
        "component": [],   # attention, scheduler, quantization, etc.
        "priority": [],    # critical, high, etc.
        "status": [],      # ready, needs-review, etc.
        "other": [],
    }
    
    type_keywords = ["bug", "feature", "enhancement", "fix", "refactor", "docs", "test", "ci", "perf", "breaking"]
    component_keywords = ["attention", "scheduler", "quantization", "memory", "kernel", "cuda", "rocm", "tpu", 
                          "model", "api", "serving", "speculative", "prefix", "kv", "distributed", "tensor"]
    priority_keywords = ["critical", "high", "medium", "low", "urgent", "priority"]
    status_keywords = ["ready", "review", "wip", "blocked", "draft", "approved"]
    
    for label in labels:
        label_lower = label.lower()
        categorized = False
        
        for kw in type_keywords:
            if kw in label_lower:
                categories["type"].append(label)
                categorized = True
                break
        
        if not categorized:
            for kw in component_keywords:
                if kw in label_lower:
                    categories["component"].append(label)
                    categorized = True
                    break
        
        if not categorized:
            for kw in priority_keywords:
                if kw in label_lower:
                    categories["priority"].append(label)
                    categorized = True
                    break
        
        if not categorized:
            for kw in status_keywords:
                if kw in label_lower:
                    categories["status"].append(label)
                    categorized = True
                    break
        
        if not categorized:
            categories["other"].append(label)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def _analyze_files_changed(files: List[Dict]) -> Dict[str, Any]:
    """Analyze the files changed in a PR to understand the scope.
    
    Args:
        files: List of file objects from GitHub API
        
    Returns:
        Analysis of files changed
    """
    analysis = {
        "total_files": len(files),
        "additions": 0,
        "deletions": 0,
        "by_directory": {},
        "by_type": {},
        "key_files": [],
    }
    
    # Key directories/files that often indicate performance impact
    performance_indicators = [
        "attention", "kernel", "cuda", "rocm", "quantization", "scheduler",
        "memory", "cache", "speculative", "distributed", "parallel",
        "benchmark", "perf", "csrc",  # C++ source files
    ]
    
    for file in files:
        filename = file.get("filename", "")
        additions = file.get("additions", 0)
        deletions = file.get("deletions", 0)
        
        analysis["additions"] += additions
        analysis["deletions"] += deletions
        
        # Categorize by directory
        parts = filename.split("/")
        if len(parts) > 1:
            directory = parts[0]
            if directory not in analysis["by_directory"]:
                analysis["by_directory"][directory] = 0
            analysis["by_directory"][directory] += 1
        
        # Categorize by file type
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1]
            if ext not in analysis["by_type"]:
                analysis["by_type"][ext] = 0
            analysis["by_type"][ext] += 1
        
        # Identify key performance-related files
        filename_lower = filename.lower()
        for indicator in performance_indicators:
            if indicator in filename_lower:
                analysis["key_files"].append({
                    "file": filename,
                    "changes": f"+{additions}/-{deletions}",
                    "indicator": indicator,
                })
                break
    
    # Limit key files to most significant
    analysis["key_files"] = analysis["key_files"][:20]
    
    return analysis


async def get_vllm_pull_request(
    pr_number: int,
    include_files: bool = True,
    include_comments: bool = False,
) -> Dict[str, Any]:
    """Get details about a specific vLLM pull request to understand what changed.
    
    Fetches comprehensive information about a PR including title, description,
    labels, files changed, and performance-related analysis. Useful for understanding
    specific changes referenced in release notes.
    
    TOOL_NAME=get_vllm_pull_request
    DISPLAY_NAME=Get vLLM Pull Request Details
    USECASE=Fetch detailed information about a specific vLLM pull request to understand what code changes were made. Useful when release notes reference a PR number (e.g., #12345) and you need to understand the impact.
    INSTRUCTIONS=1. Provide the PR number (e.g., 12345 from "#12345" in release notes), 2. Optionally include files changed for detailed analysis, 3. Use to understand specific changes that might affect performance
    INPUT_DESCRIPTION=pr_number (int): The pull request number (e.g., 12345); include_files (bool): Include list of files changed (default: True); include_comments (bool): Include PR comments (default: False)
    OUTPUT_DESCRIPTION=Dictionary with PR title, description, author, labels, files changed analysis, and performance-related categorization
    EXAMPLES=get_vllm_pull_request(12345), get_vllm_pull_request(pr_number=28439, include_files=True)
    PREREQUISITES=Internet access to GitHub API. Optional: GITHUB_TOKEN for higher rate limits
    RELATED_TOOLS=get_vllm_release_notes, compare_vllm_versions
    
    Args:
        pr_number: The pull request number (integer, not the full URL)
        include_files: Whether to fetch and analyze files changed (default: True)
        include_comments: Whether to include PR comments/discussion (default: False)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - pr_number: The PR number
        - title: PR title
        - description: PR description/body
        - author: PR author username
        - state: PR state (open, closed, merged)
        - labels: Categorized labels
        - created_at: When PR was created
        - merged_at: When PR was merged (if merged)
        - files_analysis: Analysis of files changed (if include_files=True)
        - performance_relevance: Assessment of performance impact potential
        - html_url: Link to the PR on GitHub
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = _get_github_headers()
            
            # Fetch PR details
            pr_url = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/pulls/{pr_number}"
            response = await client.get(pr_url, headers=headers)
            
            if response.status_code == 404:
                return {
                    "status": "error",
                    "message": f"Pull request #{pr_number} not found in vllm-project/vllm",
                    "suggestion": f"Check the PR number. Browse PRs at: https://github.com/{VLLM_REPO}/pulls",
                }
            elif response.status_code == 403:
                return {
                    "status": "error",
                    "message": "GitHub API rate limit exceeded",
                    "suggestion": "Set GITHUB_TOKEN environment variable for higher rate limits",
                }
            elif response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"GitHub API error: HTTP {response.status_code}",
                    "details": response.text[:500],
                }
            
            pr_data = response.json()
            
            # Extract basic PR info
            title = pr_data.get("title", "")
            body = pr_data.get("body", "") or ""
            author = pr_data.get("user", {}).get("login", "unknown")
            state = pr_data.get("state", "unknown")
            merged = pr_data.get("merged", False)
            merged_at = pr_data.get("merged_at")
            created_at = pr_data.get("created_at")
            html_url = pr_data.get("html_url")
            
            # Get and categorize labels
            labels = [label.get("name", "") for label in pr_data.get("labels", [])]
            categorized_labels = _categorize_pr_labels(labels)
            
            # Analyze PR body for performance-related content
            performance_changes = _extract_performance_changes(body)
            title_perf_changes = _extract_performance_changes(title)
            all_perf_indicators = performance_changes + title_perf_changes
            
            # Determine performance relevance
            perf_relevance = {
                "likely_performance_impact": len(all_perf_indicators) > 0,
                "indicators_found": len(all_perf_indicators),
                "categories_affected": list(set(p["category"] for p in all_perf_indicators)),
                "keywords_matched": list(set(kw for p in all_perf_indicators for kw in p.get("keywords", []))),
            }
            
            # Build result
            result = {
                "status": "success",
                "pr_number": pr_number,
                "title": title,
                "author": author,
                "state": "merged" if merged else state,
                "created_at": created_at,
                "merged_at": merged_at,
                "labels": categorized_labels,
                "labels_raw": labels,
                "description": body[:5000] if len(body) > 5000 else body,
                "description_truncated": len(body) > 5000,
                "performance_relevance": perf_relevance,
                "performance_changes_detected": all_perf_indicators[:10],  # Limit to 10
                "html_url": html_url,
                "message": f"Retrieved PR #{pr_number}: {title[:100]}{'...' if len(title) > 100 else ''}",
            }
            
            # Fetch files changed if requested
            if include_files:
                files_url = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/pulls/{pr_number}/files"
                files_response = await client.get(files_url, headers=headers, params={"per_page": 100})
                
                if files_response.status_code == 200:
                    files_data = files_response.json()
                    result["files_analysis"] = _analyze_files_changed(files_data)
                    result["files_count"] = len(files_data)
                    
                    # Update performance relevance based on files
                    key_files = result["files_analysis"].get("key_files", [])
                    if key_files:
                        perf_relevance["likely_performance_impact"] = True
                        perf_relevance["performance_related_files"] = len(key_files)
                else:
                    result["files_analysis"] = {"error": "Could not fetch files"}
            
            # Fetch comments if requested
            if include_comments:
                comments_url = f"{GITHUB_API_BASE}/repos/{VLLM_REPO}/issues/{pr_number}/comments"
                comments_response = await client.get(comments_url, headers=headers, params={"per_page": 50})
                
                if comments_response.status_code == 200:
                    comments_data = comments_response.json()
                    result["comments"] = [
                        {
                            "author": c.get("user", {}).get("login", "unknown"),
                            "body": c.get("body", "")[:1000],  # Truncate long comments
                            "created_at": c.get("created_at"),
                        }
                        for c in comments_data[:20]  # Limit to 20 comments
                    ]
                    result["comments_count"] = len(comments_data)
            
            return result
            
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "GitHub API request timed out",
        }
    except Exception as e:
        logger.error(f"Error fetching vLLM pull request: {e}")
        return {
            "status": "error",
            "message": f"Failed to fetch pull request: {str(e)}",
        }
