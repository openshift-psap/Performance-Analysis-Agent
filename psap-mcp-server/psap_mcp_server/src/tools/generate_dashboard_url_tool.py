"""Generate Dashboard URL Tool.

This tool generates URLs to the performance dashboard with specific filter parameters,
allowing users to directly visualize the data being discussed.
"""

from typing import Dict, List, Optional
from urllib.parse import urlencode

from psap_mcp_server.src.settings import settings


# Profile mapping from shorthand notation to full dashboard names
# MUST match exactly with assign_profile() function in dashboard.py
PROFILE_MAPPING = {
    "1k/1k": "Profile A: Balanced (1k/1k)",
    "1000/1000": "Profile A: Balanced (1k/1k)",
    "512/2048": "Profile B: Variable Workload (512/2k)",
    "512/2k": "Profile B: Variable Workload (512/2k)",
    "2048/128": "Profile C: Large Prompt (2k/128)",
    "2k/128": "Profile C: Large Prompt (2k/128)",
    "8000/1000": "Profile D: Prefill Heavy (8k/1k)",
    "8k/1k": "Profile D: Prefill Heavy (8k/1k)",

}

VALID_PP_Y_VALUES = [
    "Throughput (Output tokens/second generated)",
    "Efficiency (Output tokens/sec per TP unit)",
    "Inter-Token Latency P95 (Time between tokens)",
    "Time to First Token P95 (Response start delay)",
    "Request Latency Median (Total request processing time)",
    "Request Latency Max (Maximum request processing time)",
    "Time Per Output Token P95 (Token generation time)",
    "Total Throughput (Total tokens/second processed)",
    "Request Count (Successful completions)",
    "Error Rate (% Failed requests)",
]

VALID_SECTIONS = ["cost_analysis", "performance_plots", "compare_versions", "runtime_configs", "view_logs"]


VALID_FULL_PROFILES = set(PROFILE_MAPPING.values())


def normalize_profile(profile: str) -> str:
    """Convert shorthand profile notation to full dashboard profile name.
    
    Args:
        profile: Profile string in shorthand (e.g., "1k/1k") or full format
        
    Returns:
        Full profile name as used in the dashboard
    """
    if profile in VALID_FULL_PROFILES:
        return profile

    mapped = PROFILE_MAPPING.get(profile)
    if mapped:
        return mapped

    # Try to extract shorthand from parentheses, e.g. "Profile B: Throughput (512/2k)" -> "512/2k"
    import re
    match = re.search(r"\(([^)]+)\)", profile)
    if match:
        shorthand = match.group(1)
        mapped = PROFILE_MAPPING.get(shorthand)
        if mapped:
            return mapped

    return profile


async def generate_dashboard_url(
    view: str = "RHAIIS Dashboard",
    accelerators: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    versions: Optional[List[str]] = None,
    profile: Optional[str] = None,
    tp_sizes: Optional[List[int]] = None,
    section: Optional[str] = None,
    # Performance plots parameters (section="performance_plots")
    pp_x: Optional[str] = None,
    pp_y: Optional[str] = None,
    pp_conc: Optional[int] = None,
    # Compare versions parameters (section="compare_versions")
    cv_v1: Optional[str] = None,
    cv_v2: Optional[str] = None,
    cv_gpu: Optional[str] = None,
    cv_profile: Optional[str] = None,
    cv_conc: Optional[List[int]] = None,
    base_url: Optional[str] = None,
) -> str:
    """Generate a URL to the performance dashboard with specified filters.
    
    This tool creates direct links to the dashboard with pre-applied filters,
    allowing users to visualize the exact data being analyzed.
    
    Supports four dashboard sections via the 'section' parameter:
    - "cost_analysis": Cost per million tokens view
    - "performance_plots": Charts of metrics vs concurrency (use pp_* params)
    - "compare_versions": Structured delta-table between two versions (use cv_* params)
    - "runtime_configs": vLLM runtime configurations and deployment parameters
    - "view_logs": vLLM server logs for a benchmark run
    
    Args:
        view: Dashboard view ("RHAIIS Dashboard", "MLPerf Dashboard", or "LLM-D Dashboard")
        accelerators: List of accelerators (e.g., ["H200", "MI300X"])
        models: List of models (e.g., ["meta-llama/Llama-3.3-70B-Instruct"])
        versions: List of versions (e.g., ["RHAIIS-3.2.3", "RHAIIS-3.2.2"])
        profile: Sidebar profile filter in shorthand (e.g., "1k/1k") or full format
        tp_sizes: List of tensor parallelism sizes (e.g., [8])
        section: Dashboard section ("cost_analysis", "performance_plots", "compare_versions", "runtime_configs", or "view_logs")
        pp_x: Performance plots X-axis. Only valid value: "Concurrency"
        pp_y: Performance plots Y-axis metric (e.g., "Throughput (Output tokens/second generated)").
            Valid values: "Throughput (Output tokens/second generated)", "Efficiency (Output tokens/sec per TP unit)",
            "Inter-Token Latency P95 (Time between tokens)", "Time to First Token P95 (Response start delay)",
            "Request Latency Median (Total request processing time)", "Request Latency Max (Maximum request processing time)",
            "Time Per Output Token P95 (Token generation time)", "Total Throughput (Total tokens/second processed)",
            "Request Count (Successful completions)", "Error Rate (% Failed requests)"
        pp_conc: Performance plots max concurrency filter (integer). Only used when pp_x="Concurrency".
        cv_v1: Compare versions — version 1 (e.g., "RHAIIS-3.3")
        cv_v2: Compare versions — version 2 (e.g., "RHAIIS-3.2.5")
        cv_gpu: Compare versions — accelerator (e.g., "H200")
        cv_profile: Compare versions — profile in shorthand (e.g., "1k/1k") or full format
        cv_conc: Compare versions — list of concurrency values to compare. Use ONLY values from actual benchmark data (e.g., from compare_configurations or compare_versions_comprehensive output). Omit if unknown.
        base_url: Base URL of the dashboard
        
    Returns:
        A formatted string containing the dashboard URL and filter summary
        
    Examples:
        >>> # Performance plot: throughput vs concurrency
        >>> generate_dashboard_url(
        ...     accelerators=["H200"],
        ...     models=["RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"],
        ...     versions=["RHAIIS-3.3"],
        ...     profile="1k/1k", tp_sizes=[4],
        ...     section="performance_plots",
        ...     pp_x="Concurrency",
        ...     pp_y="Throughput (Output tokens/second generated)",
        ...     pp_conc=650,
        ... )
        
        >>> # Compare two versions
        >>> generate_dashboard_url(
        ...     accelerators=["H200", "MI300X"],
        ...     models=["RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"],
        ...     versions=["RHAIIS-3.4-EA2"],
        ...     profile="512/2k", tp_sizes=[1, 4],
        ...     section="compare_versions",
        ...     cv_v1="RHAIIS-3.4-EA2", cv_v2="RHAIIS-3.4-EA1",
        ...     cv_gpu="H200", cv_profile="1k/1k",
        ...     cv_conc=[1, 50, 100, 200, 300],
        ... )
        
        >>> # Cost analysis
        >>> generate_dashboard_url(
        ...     accelerators=["H200"],
        ...     models=["RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"],
        ...     versions=["RHAIIS-3.3"],
        ...     profile="1k/1k", tp_sizes=[4],
        ...     section="cost_analysis",
        ... )
    """
    # Always use the configured base URL — ignore any base_url the caller passes.
    # The value comes from the DASHBOARD_BASE_URL environment variable through
    # the central MCP settings object. Never fall back to an internal URL.
    base_url = (settings.DASHBOARD_BASE_URL or "").strip().rstrip("/")
    if not base_url:
        return (
            "Dashboard URL unavailable: DASHBOARD_BASE_URL is not configured. "
            "Set it in the MCP server environment to enable dashboard links."
        )

    def _clean_list(val):
        """Sanitize list params — LLMs sometimes pass stringified arrays like '["H200"]'."""
        if val is None:
            return None
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("["):
                import ast
                try:
                    val = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    val = val.strip("[]").replace('"', '').replace("'", "")
                    val = [v.strip() for v in val.split(",") if v.strip()]
            else:
                val = [v.strip() for v in val.split(",") if v.strip()]
        cleaned = []
        for item in val:
            s = str(item).strip().strip('"').strip("'").strip("[").strip("]")
            if s:
                cleaned.append(s)
        return cleaned if cleaned else None

    accelerators = _clean_list(accelerators)
    models = _clean_list(models)
    versions = _clean_list(versions)
    if tp_sizes is not None:
        tp_cleaned = _clean_list(tp_sizes)
        tp_sizes = [int(x) for x in tp_cleaned] if tp_cleaned else None
    if cv_conc is not None:
        cv_cleaned = _clean_list(cv_conc)
        cv_conc = [int(x) for x in cv_cleaned] if cv_cleaned else None

    valid_views = ["RHAIIS Dashboard", "MLPerf Dashboard", "LLM-D Dashboard"]
    if view not in valid_views:
        return f"Error: Invalid view '{view}'. Must be one of: {', '.join(valid_views)}"

    if section and section not in VALID_SECTIONS:
        return f"Error: Invalid section '{section}'. Must be one of: {', '.join(VALID_SECTIONS)}"

    if pp_y and pp_y not in VALID_PP_Y_VALUES:
        return f"Error: Invalid pp_y '{pp_y}'. Must be one of: {VALID_PP_Y_VALUES}"

    # Build query parameters (order matters for readable URLs)
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
        full_profile = normalize_profile(profile)
        params["profile"] = full_profile
        filters_applied["profile"] = full_profile
    
    if tp_sizes:
        params["tp_sizes"] = ",".join(map(str, tp_sizes))
        filters_applied["tp_sizes"] = tp_sizes
    
    if section:
        params["section"] = section
        filters_applied["section"] = section

    # Section-specific parameters
    if section == "performance_plots":
        if pp_x:
            params["pp_x"] = pp_x
        if pp_y:
            params["pp_y"] = pp_y
        if pp_conc is not None and pp_x == "Concurrency":
            params["pp_conc"] = str(pp_conc)

    elif section == "compare_versions":
        if cv_v1:
            params["cv_v1"] = cv_v1
        if cv_v2:
            params["cv_v2"] = cv_v2
        if cv_gpu:
            params["cv_gpu"] = cv_gpu
        if cv_profile:
            params["cv_profile"] = normalize_profile(cv_profile)
        if cv_conc:
            params["cv_conc"] = ",".join(map(str, cv_conc))

    # Generate URL with proper encoding
    url = f"{base_url}/?{urlencode(params)}"
    
    # Create filter summary for display
    filter_summary = []
    if accelerators:
        filter_summary.append(f"Accelerators: {', '.join(accelerators)}")
    if models:
        model_display = [m.split("/")[-1] for m in models]
        filter_summary.append(f"Models: {', '.join(model_display)}")
    if versions:
        filter_summary.append(f"Versions: {', '.join(versions)}")
    if profile:
        filter_summary.append(f"Profile: {normalize_profile(profile)}")
    if tp_sizes:
        filter_summary.append(f"TP Sizes: {', '.join(map(str, tp_sizes))}")
    if section:
        filter_summary.append(f"Section: {section}")
    
    summary = " | ".join(filter_summary) if filter_summary else "No filters applied"
    return f"Dashboard URL: {url}\n\nFilters Applied: {summary}\n\nView: {view}\nNote: Open this URL in your browser to see interactive visualizations of the data."
