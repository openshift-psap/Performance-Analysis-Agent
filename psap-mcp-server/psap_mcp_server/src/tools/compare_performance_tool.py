"""Compare performance across different configurations.

This tool enables side-by-side comparison of performance metrics across models, accelerators, or versions.
Supports geometric mean calculations for comprehensive comparisons.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd

from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()


def _parse_version_tuple(version_str: str) -> Optional[tuple]:
    """Extract a comparable version tuple from a version string.

    Handles formats like:
      - "RHAIIS-3.2.4"  -> (3, 2, 4)
      - "v0.13.0"       -> (0, 13, 0)
      - "0.11.2"        -> (0, 11, 2)

    Returns None if no version numbers can be extracted.
    """
    # Strip common prefixes
    cleaned = re.sub(r'^(rhaiis-|v)', '', version_str.strip(), flags=re.IGNORECASE)
    # Extract dotted number sequences (e.g. "3.2.4")
    match = re.search(r'(\d+(?:\.\d+)+)', cleaned)
    if match:
        return tuple(int(x) for x in match.group(1).split('.'))
    return None


def _order_versions(version1: str, version2: str) -> tuple:
    """Return (older_version, newer_version) based on version numbers.

    If versions cannot be parsed, the original order is preserved
    (version1 treated as baseline/older, version2 as comparison/newer).
    """
    v1_tuple = _parse_version_tuple(version1)
    v2_tuple = _parse_version_tuple(version2)

    if v1_tuple is not None and v2_tuple is not None:
        if v1_tuple <= v2_tuple:
            return (version1, version2)
        else:
            return (version2, version1)

    # Cannot determine order — keep as-is
    return (version1, version2)


def _calculate_geom_mean(values: List[float]) -> Optional[float]:
    """Calculate geometric mean of positive values.
    
    Uses log-transform method for numerical stability:
    geom_mean = exp(mean(log(values)))
    """
    positive_values = [v for v in values if v is not None and v > 0]
    if not positive_values:
        return None
    return float(np.exp(np.mean(np.log(positive_values))))


def _calculate_geom_mean_change(changes: List[float]) -> float:
    """Calculate geometric mean of percentage changes using growth factors.
    
    Converts percentage changes to growth factors (1 + change/100),
    computes geometric mean, then converts back.
    
    This is the correct way to average ratios/multiplicative changes.
    """
    if not changes:
        return 0.0
    
    # Filter out changes that would result in non-positive growth factors (< -100%)
    valid_changes = [c for c in changes if c > -100]
    if not valid_changes:
        return 0.0
    
    # Convert to growth factors
    growth_factors = [1 + (c / 100) for c in valid_changes]
    
    # Compute geometric mean of growth factors
    geom_mean_factor = np.prod(growth_factors) ** (1 / len(growth_factors))
    
    # Convert back to percentage change
    return float((geom_mean_factor - 1) * 100)


def _to_python_native(value):
    """Convert numpy types to Python native types for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_python_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_python_native(v) for k, v in value.items()}
    if isinstance(value, set):
        return [_to_python_native(v) for v in value]
    return value


async def compare_configurations(
    configurations: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare performance metrics across multiple configurations.

    TOOL_NAME=compare_configurations
    DISPLAY_NAME=Compare Performance Configurations
    USECASE=Side-by-side comparison of different models, accelerators, or settings
    INSTRUCTIONS=1. Specify 2 or more configurations to compare, 2. Select metrics, 3. Get comparison results
    INPUT_DESCRIPTION=configurations (list): List of config dicts with keys: model, accelerator, version, tp; metrics (list, optional): Specific metrics to compare
    OUTPUT_DESCRIPTION=Dictionary with side-by-side comparison data, relative performance, and recommendations
    EXAMPLES=compare_configurations([{"model":"Llama-3.3-70B","accelerator":"H200"},{"model":"Llama-3.3-70B","accelerator":"MI300X"}])
    PREREQUISITES=At least 2 configurations to compare
    RELATED_TOOLS=query_performance_metrics, get_performance_rankings

    Args:
        configurations: List of configuration dictionaries, each containing:
            - model: Model name (optional)
            - accelerator: Accelerator type (optional)
            - version: RHAIIS version (optional)
            - tp: Tensor parallelism (optional)
            - profile: Profile in "ISL/OSL" format (e.g., "1k/1k", "2048/128") (optional)
        metrics: List of specific metrics to compare. Defaults to key metrics.

    Returns:
        Dictionary containing:
        - status: success/error
        - comparison: Side-by-side comparison data
        - relative_performance: Performance differences and ratios
        - winner: Best configuration for each metric
    """
    try:
        if not configurations or len(configurations) < 2:
            return {
                "status": "error",
                "error": "At least 2 configurations required for comparison",
                "message": "Please provide at least 2 configurations to compare",
            }

        df = load_rhaiis_data()
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load performance data",
                "message": "Could not load RHAIIS data file",
            }

        # Default metrics to compare
        if metrics is None:
            metrics = [
                "output_tok/sec",
                "ttft_median",
                "ttft_p95",
                "itl_median",
                "itl_p95",
                "tpot_median",
                "tpot_p95",
                "request_latency_median",
                "efficiency_ratio",
                "successful_requests",
                "errored_requests",
            ]

        # Query data for each configuration
        config_results = []
        for i, config in enumerate(configurations):
            filtered_df = df.copy()

            # Apply filters
            if config.get("model"):
                filtered_df = filtered_df[filtered_df["model"].str.contains(config["model"], case=False, na=False)]
            if config.get("accelerator"):
                filtered_df = filtered_df[filtered_df["accelerator"].str.contains(config["accelerator"], case=False, na=False)]
            if config.get("version"):
                # Use exact match for version to avoid matching "RHAIIS-3.2.3" when user wants "RHAIIS-3.2.3" (not async)
                # But use case-insensitive comparison to handle "rhaiis" vs "RHAIIS"
                filtered_df = filtered_df[filtered_df["version"].str.lower() == config["version"].lower()]
            if config.get("tp") is not None:
                filtered_df = filtered_df[filtered_df["TP"] == config["tp"]]
            
            # Profile filtering (format: "1k/1k" or "2048/128")
            if config.get("profile"):
                profile = config["profile"]
                if "/" in profile:
                    parts = profile.lower().split("/")
                    if len(parts) == 2:
                        try:
                            # Helper function to parse token counts with k notation
                            def parse_token_count(s):
                                s = s.strip()
                                if 'k' in s:
                                    num = float(s.replace('k', ''))
                                    # Try both decimal (k=1000) and binary (k=1024) interpretations
                                    return [int(num * 1000), int(num * 1024)]
                                return [int(s)]
                            
                            prompt_candidates = parse_token_count(parts[0])
                            output_candidates = parse_token_count(parts[1])
                            
                            # Try all combinations and find matches
                            match_found = False
                            for p_tok in prompt_candidates:
                                for o_tok in output_candidates:
                                    matched_df = filtered_df[
                                        (filtered_df["prompt toks"] == p_tok) &
                                        (filtered_df["output toks"] == o_tok)
                                    ]
                                    if not matched_df.empty:
                                        filtered_df = matched_df
                                        match_found = True
                                        break
                                if match_found:
                                    break
                        except (ValueError, AttributeError):
                            pass

            if filtered_df.empty:
                config_results.append({
                    "config_id": f"config_{i+1}",
                    "config": config,
                    "found": False,
                    "message": "No data found for this configuration",
                })
                continue

            # Get best result (highest throughput)
            if "output_tok/sec" in filtered_df.columns:
                best_row = filtered_df.sort_values("output_tok/sec", ascending=False).iloc[0]
            else:
                best_row = filtered_df.iloc[0]

            # Create profile string from prompt and output tokens
            profile_str = None
            if pd.notna(best_row.get("prompt toks")) and pd.notna(best_row.get("output toks")):
                profile_str = f"{int(best_row['prompt toks'])}/{int(best_row['output toks'])}"
            
            result = {
                "config_id": f"config_{i+1}",
                "config": config,
                "found": True,
                "actual_values": {
                    "model": best_row.get("model"),
                    "accelerator": best_row.get("accelerator"),
                    "version": best_row.get("version"),
                    "TP": int(best_row.get("TP")) if pd.notna(best_row.get("TP")) else None,
                    "profile": profile_str,
                    "concurrency": int(best_row.get("intended concurrency")) if pd.notna(best_row.get("intended concurrency")) else None,
                    "concurrency_note": "Concurrency at which peak throughput was achieved",
                },
                "metrics": {},
            }

            # Extract requested metrics
            for metric in metrics:
                if metric in best_row.index and pd.notna(best_row[metric]):
                    result["metrics"][metric] = float(best_row[metric])
                else:
                    result["metrics"][metric] = None

            config_results.append(result)

        # Filter out configs that weren't found
        valid_results = [r for r in config_results if r.get("found")]

        if len(valid_results) < 2:
            return {
                "status": "error",
                "comparison": config_results,
                "message": "Not enough valid configurations found for comparison",
            }

        # Check if profiles are different (unfair comparison)
        profiles = [r["actual_values"].get("profile") for r in valid_results]
        unique_profiles = [p for p in set(profiles) if p is not None]
        
        if len(unique_profiles) > 1:
            # Different profiles detected - this is an unfair comparison
            profile_list = ", ".join(unique_profiles)
            return {
                "status": "error",
                "comparison": config_results,
                "message": f"Cannot compare configurations with different profiles: {profile_list}. Please specify which profile to use for a fair comparison.",
                "profiles_found": unique_profiles,
                "error_type": "profile_mismatch",
            }

        # Check if TP (Tensor Parallelism) values are different (unfair comparison)
        tp_values = [r["actual_values"].get("TP") for r in valid_results]
        unique_tps = [tp for tp in set(tp_values) if tp is not None]
        
        if len(unique_tps) > 1:
            # Different TP values detected - check if user specified TP
            user_specified_tp = any(config.get("tp") is not None for config in configurations)
            
            if user_specified_tp:
                # User specified TP but still got mismatch (shouldn't happen, but handle it)
                tp_list = ", ".join([f"TP {tp}" for tp in sorted(unique_tps)])
                return {
                    "status": "error",
                    "comparison": config_results,
                    "message": f"Cannot compare configurations with different Tensor Parallelism (TP) values: {tp_list}. TP significantly impacts performance. Please specify which TP value to use for a fair comparison.",
                    "tp_values_found": unique_tps,
                    "error_type": "tp_mismatch",
                }
            else:
                # User didn't specify TP - try to find common TP and retry
                # This requires checking what TPs are available for each configuration
                tp_list = ", ".join([f"TP {tp}" for tp in sorted(unique_tps)])
                return {
                    "status": "error",
                    "comparison": config_results,
                    "message": f"Found different TP values when selecting peak throughput: {tp_list}. Recommend comparing at a common TP value (e.g., TP 1 if available for both). Please specify which TP value to use for a fair comparison.",
                    "tp_values_found": unique_tps,
                    "error_type": "tp_mismatch",
                    "suggestion": "Try adding 'tp': 1 to your configurations to compare at TP=1, or specify another common TP value.",
                }

        # Calculate relative performance
        relative_performance = {}
        winner = {}

        for metric in metrics:
            metric_values = []
            for result in valid_results:
                value = result["metrics"].get(metric)
                if value is not None:
                    metric_values.append({
                        "config_id": result["config_id"],
                        "value": value,
                    })

            if len(metric_values) < 2:
                continue

            # Determine if higher is better (throughput) or lower is better (latency)
            lower_is_better = any(x in metric.lower() for x in ["latency", "ttft", "itl", "tpot", "error"])

            if lower_is_better:
                best_value = min(metric_values, key=lambda x: x["value"])
                worst_value = max(metric_values, key=lambda x: x["value"])
                winner[metric] = best_value["config_id"]
                # Avoid division by zero
                if worst_value["value"] != 0:
                    improvement = ((worst_value["value"] - best_value["value"]) / worst_value["value"] * 100)
                else:
                    improvement = 0.0  # If worst is 0, no meaningful improvement calculation
            else:
                best_value = max(metric_values, key=lambda x: x["value"])
                worst_value = min(metric_values, key=lambda x: x["value"])
                winner[metric] = best_value["config_id"]
                # Avoid division by zero
                if worst_value["value"] != 0:
                    improvement = ((best_value["value"] - worst_value["value"]) / worst_value["value"] * 100)
                else:
                    # If worst is 0 and best is non-zero, it's infinite improvement
                    improvement = float('inf') if best_value["value"] > 0 else 0.0

            relative_performance[metric] = {
                "best": best_value,
                "worst": worst_value,
                "improvement_percent": round(improvement, 2) if improvement != float('inf') else "Infinite",
                "all_values": metric_values,
            }

        logger.info(f"Compared {len(valid_results)} configurations across {len(metrics)} metrics")

        return {
            "status": "success",
            "comparison": config_results,
            "relative_performance": relative_performance,
            "winner": winner,
            "message": f"Successfully compared {len(valid_results)} configurations",
        }

    except Exception as e:
        logger.error(f"Error comparing configurations: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to compare configurations",
        }


async def compare_versions_comprehensive(
    version1: str,
    version2: str,
    model: Optional[str] = None,
    accelerator: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprehensive version comparison with peak, mean, median, and geometric mean metrics.
    
    Compares two RHAIIS/vLLM versions across multiple aggregation methods to provide
    a complete picture of performance changes. Uses the same methodology as the
    performance dashboard for consistency.
    
    IMPORTANT: Versions are automatically ordered so that the newer version is compared
    against the older version (baseline). All percentage changes are expressed as
    "how the newer version changed relative to the older baseline". Positive changes
    in throughput and negative changes in latency indicate improvement.
    
    Geometric mean is better for comparing ratios/multiplicative changes
    because it correctly handles the asymmetry of percentage changes
    (e.g., +100% then -50% = 0% net change, not +25% from arithmetic mean).
    
    TOOL_NAME=compare_versions_comprehensive
    DISPLAY_NAME=Compare Versions Comprehensive
    USECASE=Get a complete performance comparison between two versions using peak, mean, median, AND geometric mean metrics. Use this for thorough version regression analysis.
    INSTRUCTIONS=1. Specify two versions to compare (order doesn't matter — newer is auto-detected), 2. Optionally filter by model/accelerator/profile, 3. Results show how the newer version changed relative to the older baseline
    INPUT_DESCRIPTION=version1 (str): First version; version2 (str): Second version; model (str, optional): Filter by model; accelerator (str, optional): Filter by accelerator; profile (str, optional): Filter by profile like "1k/1k"
    OUTPUT_DESCRIPTION=Dictionary with peak, mean, median, and geometric mean comparisons for each metric, plus overall verdict. Changes are always expressed as newer version relative to older baseline.
    EXAMPLES=compare_versions_comprehensive("RHAIIS-3.2.4", "RHAIIS-3.2.5"), compare_versions_comprehensive("v0.11.2", "v0.13.0", model="DeepSeek")
    PREREQUISITES=Performance data must be available for both versions
    RELATED_TOOLS=compare_configurations, analyze_regression, get_vllm_release_notes
    
    Args:
        version1: First version (e.g., "RHAIIS-3.2.4" or "v0.11.2"). Order doesn't matter.
        version2: Second version. Order doesn't matter — older/newer is auto-detected.
        model: Optional model filter
        accelerator: Optional accelerator filter  
        profile: Optional profile filter (e.g., "1k/1k", "2048/128")
    
    Returns:
        Dictionary with comprehensive comparison including:
        - metrics_comparison: Detailed comparison for each metric (newer vs older)
        - overall_verdict: Consensus-based determination of which version is better
        - summary: Human-readable summary of changes
    """
    try:
        df = load_rhaiis_data()
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load performance data",
                "message": "Could not load RHAIIS data file",
            }
        
        # Auto-detect version ordering: baseline (older) vs newer
        baseline_ver, newer_ver = _order_versions(version1, version2)
        
        # Apply filters
        def filter_version(df_in: pd.DataFrame, version: str) -> pd.DataFrame:
            return df_in[df_in["version"].str.lower() == version.lower()]
        
        df_baseline = filter_version(df, baseline_ver)
        df_newer = filter_version(df, newer_ver)
        
        if df_baseline.empty:
            return {
                "status": "error",
                "message": f"No data found for baseline version {baseline_ver}",
                "available_versions": df["version"].unique().tolist()[:20],
            }
        
        if df_newer.empty:
            return {
                "status": "error",
                "message": f"No data found for newer version {newer_ver}",
                "available_versions": df["version"].unique().tolist()[:20],
            }
        
        # Apply optional filters
        if model:
            df_baseline = df_baseline[df_baseline["model"].str.contains(model, case=False, na=False)]
            df_newer = df_newer[df_newer["model"].str.contains(model, case=False, na=False)]
        
        if accelerator:
            df_baseline = df_baseline[df_baseline["accelerator"].str.contains(accelerator, case=False, na=False)]
            df_newer = df_newer[df_newer["accelerator"].str.contains(accelerator, case=False, na=False)]
        
        if profile and "/" in profile:
            parts = profile.lower().split("/")
            if len(parts) == 2:
                def parse_token_count(s):
                    s = s.strip()
                    if 'k' in s:
                        num = float(s.replace('k', ''))
                        return [int(num * 1000), int(num * 1024)]
                    return [int(s)]
                
                prompt_candidates = parse_token_count(parts[0])
                output_candidates = parse_token_count(parts[1])
                
                for p_tok in prompt_candidates:
                    for o_tok in output_candidates:
                        df_b_filtered = df_baseline[(df_baseline["prompt toks"] == p_tok) & (df_baseline["output toks"] == o_tok)]
                        df_n_filtered = df_newer[(df_newer["prompt toks"] == p_tok) & (df_newer["output toks"] == o_tok)]
                        if not df_b_filtered.empty and not df_n_filtered.empty:
                            df_baseline = df_b_filtered
                            df_newer = df_n_filtered
                            break
        
        if df_baseline.empty or df_newer.empty:
            return {
                "status": "error",
                "message": "No matching data after applying filters",
            }
        
        # Metrics to compare with their properties
        metrics_config = {
            "output_tok/sec": {"display_name": "Output Throughput", "higher_is_better": True, "unit": "tok/s"},
            "total_tok/sec": {"display_name": "Total Throughput", "higher_is_better": True, "unit": "tok/s"},
            "request_latency_median": {"display_name": "E2E Latency (Median)", "higher_is_better": False, "unit": "s"},
            "ttft_median": {"display_name": "TTFT Median", "higher_is_better": False, "unit": "ms"},
            "ttft_p95": {"display_name": "TTFT P95", "higher_is_better": False, "unit": "ms"},
            "itl_median": {"display_name": "ITL Median", "higher_is_better": False, "unit": "ms"},
            "itl_p95": {"display_name": "ITL P95", "higher_is_better": False, "unit": "ms"},
        }
        
        # Get common concurrency levels
        baseline_concurrencies = set(df_baseline["intended concurrency"].dropna().unique())
        newer_concurrencies = set(df_newer["intended concurrency"].dropna().unique())
        common_concurrencies = baseline_concurrencies.intersection(newer_concurrencies)
        
        if not common_concurrencies:
            return {
                "status": "error",
                "message": "No common concurrency levels between versions",
                "baseline_concurrencies": sorted(list(baseline_concurrencies)),
                "newer_concurrencies": sorted(list(newer_concurrencies)),
            }
        
        df_baseline_common = df_baseline[df_baseline["intended concurrency"].isin(common_concurrencies)]
        df_newer_common = df_newer[df_newer["intended concurrency"].isin(common_concurrencies)]
        
        # Calculate comparisons for each metric
        # All changes are expressed as: (newer - baseline) / baseline * 100
        # So positive = newer is higher, negative = newer is lower
        metrics_comparison = {}
        newer_wins_metrics = []
        baseline_wins_metrics = []
        
        for metric_col, config in metrics_config.items():
            if metric_col not in df_baseline_common.columns:
                continue
            
            baseline_values = df_baseline_common[metric_col].dropna().tolist()
            newer_values = df_newer_common[metric_col].dropna().tolist()
            
            if not baseline_values or not newer_values:
                continue
            
            higher_is_better = config["higher_is_better"]
            
            # Calculate aggregations for each version
            baseline_peak = max(baseline_values) if higher_is_better else min(baseline_values)
            newer_peak = max(newer_values) if higher_is_better else min(newer_values)
            baseline_mean = np.mean(baseline_values)
            newer_mean = np.mean(newer_values)
            baseline_median = np.median(baseline_values)
            newer_median = np.median(newer_values)
            baseline_geom_mean = _calculate_geom_mean(baseline_values)
            newer_geom_mean = _calculate_geom_mean(newer_values)
            
            # Calculate percentage changes: (newer - baseline) / baseline * 100
            def calc_pct_change(newer_val, baseline_val):
                if baseline_val and baseline_val != 0:
                    return ((newer_val - baseline_val) / baseline_val) * 100
                return 0
            
            peak_change = calc_pct_change(newer_peak, baseline_peak)
            mean_change = calc_pct_change(newer_mean, baseline_mean)
            median_change = calc_pct_change(newer_median, baseline_median)
            geom_mean_change = calc_pct_change(newer_geom_mean, baseline_geom_mean) if newer_geom_mean and baseline_geom_mean else 0
            
            # Also calculate geometric mean of per-concurrency changes
            per_conc_changes = []
            for conc in common_concurrencies:
                baseline_at_conc = df_baseline_common[df_baseline_common["intended concurrency"] == conc][metric_col].dropna()
                newer_at_conc = df_newer_common[df_newer_common["intended concurrency"] == conc][metric_col].dropna()
                if not baseline_at_conc.empty and not newer_at_conc.empty:
                    b_val = baseline_at_conc.iloc[0]
                    n_val = newer_at_conc.iloc[0]
                    if b_val != 0:
                        per_conc_changes.append(((n_val - b_val) / b_val) * 100)
            
            geom_mean_of_changes = _calculate_geom_mean_change(per_conc_changes) if per_conc_changes else 0
            
            # Determine if the newer version improved on this metric
            # For higher_is_better (throughput): positive change = newer is better
            # For lower_is_better (latency):    negative change = newer is better
            def is_improvement(change):
                if higher_is_better:
                    return change >= 5
                else:
                    return change <= -5
            
            def is_regression(change):
                if higher_is_better:
                    return change <= -5
                else:
                    return change >= 5
            
            # Count improvements vs regressions across all 4 aggregation methods
            improvements = sum([
                1 if is_improvement(peak_change) else 0,
                1 if is_improvement(mean_change) else 0,
                1 if is_improvement(median_change) else 0,
                1 if is_improvement(geom_mean_change) else 0,
            ])
            regressions = sum([
                1 if is_regression(peak_change) else 0,
                1 if is_regression(mean_change) else 0,
                1 if is_regression(median_change) else 0,
                1 if is_regression(geom_mean_change) else 0,
            ])
            
            # Determine status — always framed from the newer version's perspective
            if improvements >= 3:
                status = "improved"
                emoji = "🟢"
                abs_change = abs(geom_mean_change)
                if higher_is_better:
                    direction_text = f"{newer_ver} is {abs_change:.1f}% higher than {baseline_ver}"
                else:
                    direction_text = f"{newer_ver} is {abs_change:.1f}% lower (better) than {baseline_ver}"
                newer_wins_metrics.append(config["display_name"])
            elif regressions >= 3:
                status = "regressed"
                emoji = "🔴"
                abs_change = abs(geom_mean_change)
                if higher_is_better:
                    direction_text = f"{newer_ver} is {abs_change:.1f}% lower (worse) than {baseline_ver}"
                else:
                    direction_text = f"{newer_ver} is {abs_change:.1f}% higher (worse) than {baseline_ver}"
                baseline_wins_metrics.append(config["display_name"])
            else:
                status = "similar"
                emoji = "🟡"
                direction_text = f"Similar between {baseline_ver} and {newer_ver}"
            
            metrics_comparison[config["display_name"]] = {
                "metric_column": metric_col,
                "higher_is_better": higher_is_better,
                "unit": config["unit"],
                "status": status,
                "emoji": emoji,
                "direction": direction_text,
                "baseline_values": {
                    "version": baseline_ver,
                    "peak": round(baseline_peak, 2),
                    "mean": round(baseline_mean, 2),
                    "median": round(baseline_median, 2),
                    "geometric_mean": round(baseline_geom_mean, 2) if baseline_geom_mean else None,
                },
                "newer_values": {
                    "version": newer_ver,
                    "peak": round(newer_peak, 2),
                    "mean": round(newer_mean, 2),
                    "median": round(newer_median, 2),
                    "geometric_mean": round(newer_geom_mean, 2) if newer_geom_mean else None,
                },
                "changes_vs_baseline": {
                    "note": f"Percentage change of {newer_ver} relative to {baseline_ver}",
                    "peak_pct": round(peak_change, 1),
                    "mean_pct": round(mean_change, 1),
                    "median_pct": round(median_change, 1),
                    "geometric_mean_pct": round(geom_mean_change, 1),
                    "geom_mean_of_changes_pct": round(geom_mean_of_changes, 1),
                },
                "consensus": {
                    "improvements_count": improvements,
                    "regressions_count": regressions,
                },
            }
        
        # Determine overall verdict
        total_newer_wins = len(newer_wins_metrics)
        total_baseline_wins = len(baseline_wins_metrics)
        
        if total_newer_wins > total_baseline_wins and total_newer_wins >= 2:
            verdict_emoji = "🟢"
            verdict_text = f"{newer_ver} performs better overall than {baseline_ver}"
        elif total_baseline_wins > total_newer_wins and total_baseline_wins >= 2:
            verdict_emoji = "🔴"
            verdict_text = f"{newer_ver} regressed overall compared to {baseline_ver}"
        else:
            verdict_emoji = "🟡"
            verdict_text = f"Performance is similar between {baseline_ver} and {newer_ver}"
        
        # Build summary
        summary_lines = [f"{verdict_emoji} **{verdict_text}**", ""]
        summary_lines.append(f"Baseline (older): **{baseline_ver}**")
        summary_lines.append(f"Comparison (newer): **{newer_ver}**")
        summary_lines.append("")
        
        if newer_wins_metrics:
            summary_lines.append(f"**{newer_ver} improved in:** {', '.join(newer_wins_metrics)}")
        if baseline_wins_metrics:
            summary_lines.append(f"**{newer_ver} regressed in:** {', '.join(baseline_wins_metrics)}")
        
        summary_lines.extend([
            "",
            "**Methodology:**",
            f"- All changes are expressed as: how {newer_ver} changed relative to {baseline_ver} (baseline)",
            "- Status based on consensus across 4 aggregations: Peak, Mean, Median, Geometric Mean",
            f"- 🟢 Improved: ≥3/4 aggregations show ≥5% improvement in {newer_ver}",
            f"- 🔴 Regressed: ≥3/4 aggregations show ≥5% regression in {newer_ver}",
            "- 🟡 Similar: Mixed results or <5% difference",
            "",
            "**Why Geometric Mean?**",
            "- Better for averaging ratios/percentages",
            "- Example: +100% then -50% = 0% net (geometric), not +25% (arithmetic)",
        ])
        
        # Convert all numpy types to Python native types for JSON serialization
        return _to_python_native({
            "status": "success",
            "versions": {
                "baseline_older": baseline_ver,
                "comparison_newer": newer_ver,
                "note": f"All changes expressed as {newer_ver} relative to {baseline_ver}",
            },
            "filters_applied": {
                "model": model,
                "accelerator": accelerator,
                "profile": profile,
            },
            "common_concurrencies": sorted(list(common_concurrencies)),
            "metrics_comparison": metrics_comparison,
            "overall_verdict": {
                "emoji": verdict_emoji,
                "text": verdict_text,
                "newer_wins_count": total_newer_wins,
                "baseline_wins_count": total_baseline_wins,
                "newer_wins_metrics": newer_wins_metrics,
                "baseline_wins_metrics": baseline_wins_metrics,
            },
            "summary": "\n".join(summary_lines),
            "message": f"Compared {newer_ver} (newer) against {baseline_ver} (baseline) across {len(metrics_comparison)} metrics at {len(common_concurrencies)} concurrency levels",
        })
        
    except Exception as e:
        logger.error(f"Error in comprehensive version comparison: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to compare versions",
        }

