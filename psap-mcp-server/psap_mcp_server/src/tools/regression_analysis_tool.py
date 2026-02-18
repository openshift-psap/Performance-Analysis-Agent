"""Analyze performance regressions between versions.

This tool detects performance improvements or degradations across different versions.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()


async def analyze_regression(
    version1: str,
    version2: str,
    accelerator: Optional[str] = None,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze performance changes between two versions.

    TOOL_NAME=analyze_regression
    DISPLAY_NAME=Analyze Performance Regression
    USECASE=Detect performance improvements or degradations between software versions
    INSTRUCTIONS=1. Specify two versions to compare, 2. Optionally filter by accelerator/model/profile, 3. Choose metrics to analyze
    INPUT_DESCRIPTION=version1 (str): First version (e.g., "3.2.2"); version2 (str): Second version (e.g., "3.2.3"); accelerator (str, optional): Filter by accelerator; model (str, optional): Filter by model; profile (str, optional): Filter by profile (e.g., "1k/1k", "512/2048"); metrics (list, optional): Metrics to compare
    OUTPUT_DESCRIPTION=Dictionary with detailed regression analysis, improvements, and degradations
    EXAMPLES=analyze_regression("3.2.2", "3.2.3"), analyze_regression("3.2.2", "3.2.3", accelerator="H200", model="Llama", profile="512/2048")
    PREREQUISITES=RHAIIS performance data with multiple versions
    RELATED_TOOLS=query_performance_metrics, compare_configurations

    Args:
        version1: First version to compare (baseline)
        version2: Second version to compare (new)
        accelerator: Optional accelerator filter
        model: Optional model filter
        profile: Optional profile filter (e.g., "1k/1k", "512/2048")
        metrics: List of metrics to analyze (defaults to key performance metrics)

    Returns:
        Dictionary containing:
        - status: success/error
        - baseline_version: Version 1 info
        - new_version: Version 2 info
        - improvements: List of configurations that improved
        - regressions: List of configurations that regressed
        - no_change: List of configurations with no significant change
        - summary: Overall analysis summary
    """
    try:
        df = load_rhaiis_data()
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load performance data",
                "message": "Could not load RHAIIS data file",
            }

        # Default metrics to analyze
        if metrics is None:
            metrics = ["output_tok/sec", "ttft_median", "itl_median"]

        # Filter data for both versions
        v1_df = df[df["version"].str.contains(version1, case=False, na=False)].copy()
        v2_df = df[df["version"].str.contains(version2, case=False, na=False)].copy()

        if v1_df.empty or v2_df.empty:
            return {
                "status": "error",
                "error": "Insufficient data for comparison",
                "message": f"Found {len(v1_df)} records for {version1} and {len(v2_df)} records for {version2}",
            }

        # Apply additional filters
        filters_applied = {"version1": version1, "version2": version2}

        if accelerator:
            v1_df = v1_df[v1_df["accelerator"].str.contains(accelerator, case=False, na=False)]
            v2_df = v2_df[v2_df["accelerator"].str.contains(accelerator, case=False, na=False)]
            filters_applied["accelerator"] = accelerator

        if model:
            v1_df = v1_df[v1_df["model"].str.contains(model, case=False, na=False)]
            v2_df = v2_df[v2_df["model"].str.contains(model, case=False, na=False)]
            filters_applied["model"] = model

        if profile:
            # Parse profile (e.g., "1k/1k" or "512/2048")
            profile_clean = profile.strip().lower()
            
            # Handle shorthand notation (k = 1024)
            def parse_profile_value(val: str) -> int:
                val = val.strip()
                if val.endswith('k'):
                    # Support both 1000-based and 1024-based
                    num = float(val[:-1])
                    # Try 1024-based first (more common in computing)
                    return int(num * 1024)
                return int(val)
            
            if '/' in profile_clean:
                parts = profile_clean.split('/')
                if len(parts) == 2:
                    try:
                        prompt_val = parse_profile_value(parts[0])
                        output_val = parse_profile_value(parts[1])
                        
                        # Filter by prompt toks and output toks
                        v1_df = v1_df[
                            (v1_df["prompt toks"] == prompt_val) & 
                            (v1_df["output toks"] == output_val)
                        ]
                        v2_df = v2_df[
                            (v2_df["prompt toks"] == prompt_val) & 
                            (v2_df["output toks"] == output_val)
                        ]
                        filters_applied["profile"] = f"{prompt_val}/{output_val}"
                    except (ValueError, KeyError):
                        pass  # Invalid profile format, skip filter

        # Create matching key for comparison
        # Use actual column names from CSV: "prompt toks", "output toks"
        match_cols = ["model", "accelerator", "TP", "prompt toks", "output toks"]
        v1_df["match_key"] = v1_df[match_cols].astype(str).agg("-".join, axis=1)
        v2_df["match_key"] = v2_df[match_cols].astype(str).agg("-".join, axis=1)

        # Find configurations present in both versions
        common_keys = set(v1_df["match_key"]) & set(v2_df["match_key"])

        if not common_keys:
            return {
                "status": "success",
                "improvements": [],
                "regressions": [],
                "no_change": [],
                "filters_applied": filters_applied,
                "message": "No common configurations found between versions for comparison",
            }

        improvements = []
        regressions = []
        no_change = []

        for key in common_keys:
            v1_row = v1_df[v1_df["match_key"] == key].iloc[0]
            v2_row = v2_df[v2_df["match_key"] == key].iloc[0]

            # Create profile string from prompt and output tokens
            profile_str = f"{int(v1_row['prompt toks'])}/{int(v1_row['output toks'])}" if pd.notna(v1_row.get('prompt toks')) and pd.notna(v1_row.get('output toks')) else None

            comparison = {
                "model": v1_row["model"],
                "accelerator": v1_row["accelerator"],
                "TP": int(v1_row["TP"]) if pd.notna(v1_row["TP"]) else None,
                "profile": profile_str,
                "metrics_changed": {},
            }

            has_improvement = False
            has_regression = False

            for metric in metrics:
                if metric not in v1_row or metric not in v2_row:
                    continue

                v1_val = v1_row[metric]
                v2_val = v2_row[metric]

                if pd.isna(v1_val) or pd.isna(v2_val):
                    continue

                v1_val = float(v1_val)
                v2_val = float(v2_val)

                if v1_val == 0:
                    continue

                # Calculate percent change
                pct_change = ((v2_val - v1_val) / v1_val) * 100

                # Determine if improvement or regression
                # For throughput (output_tok/sec), higher is better
                # For latency (ttft, itl), lower is better
                is_improvement = False
                is_regression_flag = False

                if "tok/sec" in metric or "throughput" in metric.lower():
                    # Higher is better
                    if pct_change > 2:  # More than 2% improvement threshold
                        is_improvement = True
                    elif pct_change < -2:  # More than 2% degradation threshold
                        is_regression_flag = True
                elif "ttft" in metric or "itl" in metric or "latency" in metric.lower():
                    # Lower is better
                    if pct_change < -2:  # More than 2% improvement (reduction)
                        is_improvement = True
                    elif pct_change > 2:  # More than 2% degradation (increase)
                        is_regression_flag = True

                comparison["metrics_changed"][metric] = {
                    f"{version1}_value": round(v1_val, 4),
                    f"{version2}_value": round(v2_val, 4),
                    "percent_change": round(pct_change, 2),
                    "absolute_change": round(v2_val - v1_val, 4),
                    "direction": "improvement" if is_improvement else ("regression" if is_regression_flag else "neutral"),
                }

                if is_improvement:
                    has_improvement = True
                if is_regression_flag:
                    has_regression = True

            # Categorize the comparison
            if has_regression and not has_improvement:
                regressions.append(comparison)
            elif has_improvement and not has_regression:
                improvements.append(comparison)
            elif has_improvement and has_regression:
                # Mixed results - categorize based on critical metrics
                if "output_tok/sec" in comparison["metrics_changed"]:
                    if comparison["metrics_changed"]["output_tok/sec"]["direction"] == "improvement":
                        improvements.append(comparison)
                    else:
                        regressions.append(comparison)
                else:
                    no_change.append(comparison)
            else:
                no_change.append(comparison)

        # Calculate summary statistics
        total_compared = len(common_keys)
        summary = {
            "total_configurations_compared": total_compared,
            "improvements_count": len(improvements),
            "regressions_count": len(regressions),
            "no_significant_change_count": len(no_change),
            "improvement_rate": round((len(improvements) / total_compared * 100), 2) if total_compared > 0 else 0,
            "regression_rate": round((len(regressions) / total_compared * 100), 2) if total_compared > 0 else 0,
        }

        logger.info(
            f"Analyzed regression between {version1} and {version2}: "
            f"{len(improvements)} improvements, {len(regressions)} regressions"
        )

        return {
            "status": "success",
            "baseline_version": version1,
            "new_version": version2,
            "improvements": improvements,
            "regressions": regressions,
            "no_change": no_change,
            "summary": summary,
            "filters_applied": filters_applied,
            "metrics_analyzed": metrics,
            "message": (
                f"Compared {total_compared} configurations: "
                f"{len(improvements)} improved, {len(regressions)} regressed, {len(no_change)} unchanged"
            ),
        }

    except Exception as e:
        logger.error(f"Error analyzing regression: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to analyze regression",
        }

