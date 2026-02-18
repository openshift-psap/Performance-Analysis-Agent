"""Generate Dashboard URL Tool.

This tool generates URLs to the performance dashboard with specific filter parameters,
allowing users to directly visualize the data being discussed.
"""

from typing import Dict, List, Optional
from urllib.parse import urlencode, quote


# Profile mapping from shorthand notation to full dashboard names
# MUST match exactly with assign_profile() function in dashboard.py
PROFILE_MAPPING = {
    "1k/1k": "Profile A: Balanced (1k/1k)",
    "1000/1000": "Profile A: Balanced (1k/1k)",
    "512/2048": "Profile B: Variable Workload (512/2k)",
    "512/2k": "Profile B: Variable Workload (512/2k)",
    "2048/128": "Profile C: Large Prompt (2k/128)",
    "2k/128": "Profile C: Large Prompt (2k/128)",
    "32000/256": "Profile D: Prefill Heavy (32k/256)",
    "32k/256": "Profile D: Prefill Heavy (32k/256)",
    "8000/1000": "Profile E: Prefill Heavy (8k/1k)",
    "8k/1k": "Profile E: Prefill Heavy (8k/1k)",
    "1000/100": "Profile F: Prefill Heavy (1k/100)",
    "1k/100": "Profile F: Prefill Heavy (1k/100)",
}


def normalize_profile(profile: str) -> str:
    """Convert shorthand profile notation to full dashboard profile name.
    
    Args:
        profile: Profile string in shorthand (e.g., "1k/1k") or full format
        
    Returns:
        Full profile name as used in the dashboard
    """
    # If it's already a full profile name, return it
    if profile.startswith("Profile "):
        return profile
    
    # Try to find in mapping
    return PROFILE_MAPPING.get(profile, profile)


async def generate_dashboard_url(
    view: str = "RHAIIS Dashboard",
    accelerators: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
    profile: Optional[str] = None,
    tp_sizes: Optional[List[int]] = None,
    base_url: str = "https://aidash.app.intlab.redhat.com",
) -> str:
    """Generate a URL to the performance dashboard with specified filters.
    
    This tool creates direct links to the dashboard with pre-applied filters,
    allowing users to visualize the exact data being analyzed.
    
    Args:
        view: Dashboard view ("RHAIIS Dashboard", "MLPerf Dashboard", or "LLM-D Dashboard")
        accelerators: List of accelerators (e.g., ["H200", "MI300X"])
        models: List of models (e.g., ["meta-llama/Llama-3.3-70B-Instruct"])
        versions: List of versions (e.g., ["RHAIIS-3.2.3", "RHAIIS-3.2.2"])
        profile: Profile in shorthand (e.g., "1k/1k") or full format (e.g., "Profile A: Balanced (1k/1k)")
        tp_sizes: List of tensor parallelism sizes (e.g., [8])
        base_url: Base URL of the dashboard (default: https://aidash.app.intlab.redhat.com)
        
    Returns:
        A formatted string containing the dashboard URL and filter summary
        
    Examples:
        >>> # Compare models across versions
        >>> generate_dashboard_url(
        ...     view="RHAIIS Dashboard",
        ...     accelerators=["H200"],
        ...     models=["deepseek-ai/DeepSeek-R1-0528"],
        ...     versions=["RHAIIS-3.2.3", "RHAIIS-3.2.2"],
        ...     profile="1k/1k",
        ...     tp_sizes=[8]
        ... )
        
        >>> # View cost analysis for a specific model
        >>> generate_dashboard_url(
        ...     view="RHAIIS Dashboard",
        ...     accelerators=["H200"],
        ...     models=["meta-llama/Llama-3.3-70B-Instruct"],
        ...     versions=["RHAIIS-3.2.3"],
        ...     profile="512/2k"
        ... )
    """
    # Validate view
    valid_views = ["RHAIIS Dashboard", "MLPerf Dashboard", "LLM-D Dashboard"]
    if view not in valid_views:
        return f"Error: Invalid view '{view}'. Must be one of: {', '.join(valid_views)}"
    
    # Build query parameters
    params = {"view": view}
    filters_applied = {}
    
    if accelerators:
        params["accelerators"] = ",".join(accelerators)
        filters_applied["accelerators"] = accelerators
    
    if models:
        params["models"] = ",".join(models)
        filters_applied["models"] = models
    
    if versions:
        params["versions"] = ",".join(versions)
        filters_applied["versions"] = versions
    
    if profile:
        # Normalize profile to full format
        full_profile = normalize_profile(profile)
        params["profile"] = full_profile
        filters_applied["profile"] = full_profile
    
    if tp_sizes:
        params["tp_sizes"] = ",".join(map(str, tp_sizes))
        filters_applied["tp_sizes"] = tp_sizes
    
    # Generate URL with proper encoding
    url = f"{base_url}/?{urlencode(params, quote_via=quote)}"
    
    # Create filter summary for display
    filter_summary = []
    if accelerators:
        filter_summary.append(f"Accelerators: {', '.join(accelerators)}")
    if models:
        model_display = [m.split("/")[-1] for m in models]  # Show short names
        filter_summary.append(f"Models: {', '.join(model_display)}")
    if versions:
        filter_summary.append(f"Versions: {', '.join(versions)}")
    if profile:
        filter_summary.append(f"Profile: {normalize_profile(profile)}")
    if tp_sizes:
        filter_summary.append(f"TP Sizes: {', '.join(map(str, tp_sizes))}")
    
    # Return formatted string with URL and summary
    summary = " | ".join(filter_summary) if filter_summary else "No filters applied"
    return f"Dashboard URL: {url}\n\nFilters Applied: {summary}\n\nView: {view}\nNote: Open this URL in your browser to see interactive visualizations of the data."

