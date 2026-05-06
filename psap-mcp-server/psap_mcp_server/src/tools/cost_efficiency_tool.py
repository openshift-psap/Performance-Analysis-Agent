"""Calculate cost efficiency for AI inference.

This tool calculates cost per million tokens across different cloud providers and accelerators.
"""

from typing import Any, Dict, Optional

import pandas as pd

from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# Cloud provider pricing per hour (AWS on-demand)
ACCELERATOR_PRICING = {
    "B200": {
        "hourly_cost": 82.368,
        "provider": "AWS",
        "instance": "p6-b200.48xlarge",
        "configuration": "8xNVIDIA B200",
        "notes": "Full 8-GPU instance cost regardless of TP"
    },
    "H200": {
        "hourly_cost": 41.62,
        "provider": "AWS",
        "instance": "p6en.48xlarge",
        "configuration": "8xNVIDIA H200-144GB",
        "notes": "Full 8-GPU instance cost regardless of TP"
    },
    "H100": {
        "hourly_cost": 34.608,
        "provider": "AWS",
        "instance": "p5.48xlarge",
        "configuration": "8xNVIDIA H100",
        "notes": "Full 8-GPU instance cost regardless of TP"
    },
    "MI300X": {
        "hourly_cost": 48.00,
        "provider": "Azure",
        "instance": "ND96isr_MI300X_v5",
        "configuration": "8xAMD MI300X-192GB",
        "notes": "Full 8-GPU instance cost regardless of TP"
    },
    "TPU": {
        "hourly_cost": 2.70,
        "provider": "GCP",
        "instance": "TPU Trillium",
        "configuration": "Per-core pricing",
        "notes": "Per-core pricing, multiply by TP count"
    },
    "A100": {
        "hourly_cost": 11.80,
        "provider": "AWS",
        "instance": "p4d.24xlarge",
        "configuration": "8xNVIDIA A100",
        "notes": "Full 8-GPU instance cost regardless of TP"
    },
}


async def calculate_cost_efficiency(
    accelerator: Optional[str] = None,
    model: Optional[str] = None,
    version: Optional[str] = None,
    profile: Optional[str] = None,
    max_itl_p95_ms: Optional[float] = None,
    max_ttft_p95_ms: Optional[float] = None,
    top_n: int = 10,
    custom_pricing: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Calculate cost efficiency (cost per million tokens) with optional latency constraints.

    TOOL_NAME=calculate_cost_efficiency
    DISPLAY_NAME=Calculate Cost Efficiency with Latency SLOs
    USECASE=Calculate cost per million tokens using adjusted throughput (8-GPU instance for H200/MI300X) at optimal concurrency meeting latency SLOs
    INSTRUCTIONS=1. Ask user for latency requirements (ITL P95, TTFT P95) if not specified, 2. Specify filters, 3. Get cost analysis with latency-aware optimal concurrency
    INPUT_DESCRIPTION=accelerator (str, optional): Filter by accelerator; model (str, optional): Filter by model; version (str, optional): Filter by RHAIIS version; profile (str, optional): Workload profile (e.g., "1k/1k", "512/2k"); max_itl_p95_ms (float, optional): Maximum Inter-Token Latency P95 in ms (PSAP default: 65ms); max_ttft_p95_ms (float, optional): Maximum Time to First Token P95 in ms (PSAP default: 4000ms); top_n (int): Number of results; custom_pricing (dict, optional): Custom hourly pricing per accelerator
    OUTPUT_DESCRIPTION=Dictionary with cost efficiency metrics (adjusted throughput, TTMT, CPMT) and latency values (ITL P95, TTFT P95)
    EXAMPLES=calculate_cost_efficiency(accelerator="H200", profile="512/2k", max_itl_p95_ms=65, max_ttft_p95_ms=4000), calculate_cost_efficiency(model="gpt-oss-120b", version="RHAIIS-3.2.3", profile="1k/1k")
    PREREQUISITES=RHAIIS performance data with latency metrics (itl_p95, ttft_p95) and pricing information
    RELATED_TOOLS=query_performance_metrics, get_performance_rankings

    Args:
        accelerator: Optional accelerator filter
        model: Optional model filter
        version: Optional RHAIIS version filter (supports partial match)
        profile: Optional workload profile filter (e.g., "1k/1k", "512/2k", "2k/128")
        max_itl_p95_ms: Maximum Inter-Token Latency P95 in milliseconds (filters for latency SLO compliance)
        max_ttft_p95_ms: Maximum Time to First Token P95 in milliseconds (filters for latency SLO compliance)
        top_n: Number of results to return
        custom_pricing: Optional dictionary with custom hourly costs per accelerator

    Returns:
        Dictionary containing:
        - status: success/error
        - cost_analysis: List of configurations with cost metrics and latency values
        - best_value: Most cost-efficient configuration meeting latency SLOs
        - latency_constraints: Applied latency thresholds
        - pricing_info: Pricing information used
    """
    try:
        df = load_rhaiis_data()
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load performance data",
                "message": "Could not load RHAIIS data file",
            }

        # Use custom pricing if provided, otherwise use defaults
        pricing = custom_pricing if custom_pricing else ACCELERATOR_PRICING

        # Apply filters
        filtered_df = df.copy()
        filters_applied = {}

        if accelerator:
            filtered_df = filtered_df[filtered_df["accelerator"].str.contains(accelerator, case=False, na=False)]
            filters_applied["accelerator"] = accelerator

        if model:
            filtered_df = filtered_df[filtered_df["model"].str.contains(model, case=False, na=False)]
            filters_applied["model"] = model
        
        if version:
            # Use exact match (case-insensitive) to avoid matching "RHAIIS-3.2.3-async" when user wants "RHAIIS-3.2.3"
            filtered_df = filtered_df[filtered_df["version"].str.lower() == version.lower()]
            filters_applied["version"] = version

        if profile:
            if "prompt toks" in filtered_df.columns and "output toks" in filtered_df.columns:
                if "/" in profile:
                    parts = profile.lower().split("/")
                    if len(parts) == 2:
                        try:
                            def parse_token_count(s):
                                s = s.strip()
                                if 'k' in s:
                                    num = float(s.replace('k', ''))
                                    return [int(num * 1000), int(num * 1024)]
                                return [int(s)]

                            prompt_candidates = parse_token_count(parts[0])
                            output_candidates = parse_token_count(parts[1])

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

                            if match_found:
                                filters_applied["profile"] = profile
                        except (ValueError, AttributeError):
                            pass

        # Apply latency constraints (PSAP SLO compliance)
        latency_constraints = {}
        if max_itl_p95_ms is not None:
            if "itl_p95" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["itl_p95"] <= max_itl_p95_ms]
                latency_constraints["max_itl_p95_ms"] = max_itl_p95_ms
        
        if max_ttft_p95_ms is not None:
            if "ttft_p95" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["ttft_p95"] <= max_ttft_p95_ms]
                latency_constraints["max_ttft_p95_ms"] = max_ttft_p95_ms

        if filtered_df.empty:
            return {
                "status": "success",
                "cost_analysis": [],
                "filters_applied": filters_applied,
                "latency_constraints": latency_constraints,
                "message": "No results found matching the specified filters and latency constraints",
            }

        # Calculate cost efficiency
        cost_results = []
        for _, row in filtered_df.iterrows():
            acc = row.get("accelerator")
            if acc not in pricing:
                # Try to match partial accelerator name
                matched_acc = None
                for key in pricing.keys():
                    if key in acc:
                        matched_acc = key
                        break
                if not matched_acc:
                    continue
                acc = matched_acc

            raw_throughput = row.get("output_tok/sec")
            if pd.isna(raw_throughput) or raw_throughput <= 0:
                continue

            # Get base hourly cost
            base_hourly_cost = pricing[acc] if isinstance(pricing[acc], (int, float)) else pricing[acc].get("hourly_cost", 0)
            
            # Calculate total instance cost and adjusted throughput based on accelerator type
            tp_count = int(row.get("TP")) if pd.notna(row.get("TP")) else 1
            total_gpus = 8  # H200/MI300X have 8 GPUs per instance
            
            if acc == "TPU":
                # TPU: Pay per core used, so use raw throughput and multiply cost by TP count
                total_hourly_cost = base_hourly_cost * tp_count
                effective_throughput = raw_throughput
                adjusted_throughput = raw_throughput
                throughput_note = "Raw throughput (per-core pricing)"
            else:
                # H200/MI300X: Pay for full 8-GPU instance, so use adjusted throughput
                total_hourly_cost = base_hourly_cost
                # Adjusted throughput = what you'd get if using all 8 GPUs
                adjusted_throughput = raw_throughput * (total_gpus / tp_count) if tp_count > 0 else 0
                effective_throughput = adjusted_throughput
                throughput_note = f"Adjusted throughput (8-GPU instance, using TP={tp_count})"

            # Calculate TTMT: Time to Million Tokens (using effective throughput)
            million_tokens = 1_000_000
            ttmt_seconds = million_tokens / effective_throughput if effective_throughput > 0 else float('inf')
            ttmt_minutes = ttmt_seconds / 60
            ttmt_hours = ttmt_seconds / 3600

            # Calculate CPMT: Cost per Million Tokens
            # CPMT = (TTMT_seconds * total_hourly_cost) / 3600
            cost_per_million = (ttmt_seconds * total_hourly_cost) / 3600 if ttmt_seconds != float('inf') else float('inf')

            # Calculate efficiency score (effective throughput per dollar per hour)
            efficiency_score = effective_throughput / total_hourly_cost if total_hourly_cost > 0 else 0

            prompt_toks = int(row.get("prompt toks")) if pd.notna(row.get("prompt toks")) else None
            output_toks = int(row.get("output toks")) if pd.notna(row.get("output toks")) else None

            cost_results.append({
                "model": row.get("model"),
                "accelerator": row.get("accelerator"),
                "version": row.get("version"),
                "profile": f"{prompt_toks}/{output_toks}" if prompt_toks and output_toks else None,
                "TP": tp_count,
                "concurrency": int(row.get("intended_concurrency")) if pd.notna(row.get("intended_concurrency")) else None,
                "raw_throughput_tokens_per_sec": round(float(raw_throughput), 2),
                "adjusted_throughput_tokens_per_sec": round(float(adjusted_throughput), 2) if acc != "TPU" else None,
                "effective_throughput_tokens_per_sec": round(float(effective_throughput), 2),
                "throughput_note": throughput_note,
                "base_hourly_cost_usd": float(base_hourly_cost),
                "total_instance_cost_usd_per_hour": float(total_hourly_cost),
                "pricing_note": "Per-core pricing × TP" if acc == "TPU" else "Full 8-GPU instance",
                "time_to_million_tokens_seconds": round(float(ttmt_seconds), 2) if ttmt_seconds != float('inf') else None,
                "time_to_million_tokens_minutes": round(float(ttmt_minutes), 2) if ttmt_minutes != float('inf') else None,
                "time_to_million_tokens_hours": round(float(ttmt_hours), 4) if ttmt_hours != float('inf') else None,
                "cost_per_million_tokens_usd": round(float(cost_per_million), 4) if cost_per_million != float('inf') else None,
                "efficiency_score_tokens_per_dollar_hour": round(float(efficiency_score), 2),
                # Latency metrics (P95 values)
                "itl_p95_ms": float(row.get("itl_p95")) if pd.notna(row.get("itl_p95")) else None,
                "ttft_p95_ms": float(row.get("ttft_p95")) if pd.notna(row.get("ttft_p95")) else None,
                "ttft_median_ms": float(row.get("ttft_median")) if pd.notna(row.get("ttft_median")) else None,
                "error_rate_percent": (
                    float(row.get("errored_requests")) / (float(row.get("successful_requests")) + float(row.get("errored_requests"))) * 100
                ) if pd.notna(row.get("successful_requests")) and pd.notna(row.get("errored_requests")) and (row.get("successful_requests") + row.get("errored_requests")) > 0 else 0,
            })

        if not cost_results:
            return {
                "status": "success",
                "cost_analysis": [],
                "filters_applied": filters_applied,
                "latency_constraints": latency_constraints,
                "message": "No valid cost data could be calculated for the filtered results and latency constraints",
            }

        # Sort by cost per million tokens (ascending - lower is better)
        # Filter out None values first
        cost_results_valid = [r for r in cost_results if r["cost_per_million_tokens_usd"] is not None]
        cost_results_valid.sort(key=lambda x: x["cost_per_million_tokens_usd"])

        # Get top N
        top_results = cost_results_valid[:top_n]

        # Identify best value (lowest cost per million tokens)
        best_value = top_results[0] if top_results else None
        if best_value:
            best_value["data_source"] = (
                f"Actual benchmark data point at intended concurrency {best_value.get('concurrency')}. "
                "This is NOT interpolated — all values (throughput, ITL, TTFT) are from a single benchmark run at this exact concurrency level."
            )

        # Calculate statistics
        all_costs = [r["cost_per_million_tokens_usd"] for r in cost_results_valid if r["cost_per_million_tokens_usd"] is not None]
        statistics = {
            "total_configurations": len(cost_results_valid),
            "results_shown": len(top_results),
            "lowest_cost_per_million": min(all_costs) if all_costs else None,
            "highest_cost_per_million": max(all_costs) if all_costs else None,
            "average_cost_per_million": sum(all_costs) / len(all_costs) if all_costs else None,
        }

        # Prepare pricing info
        pricing_info = {}
        for acc, price in pricing.items():
            if isinstance(price, dict):
                pricing_info[acc] = price
            else:
                pricing_info[acc] = {"hourly_cost": price, "provider": "Custom"}

        logger.info(f"Calculated cost efficiency for {len(cost_results_valid)} configurations with latency constraints: {latency_constraints}")

        return {
            "status": "success",
            "cost_analysis": top_results,
            "best_value": best_value,
            "statistics": statistics,
            "pricing_info": pricing_info,
            "filters_applied": filters_applied,
            "latency_constraints": latency_constraints if latency_constraints else "None applied (all data)",
            "message": f"Cost efficiency calculated for {len(top_results)} configurations meeting latency SLOs",
        }

    except Exception as e:
        logger.error(f"Error calculating cost efficiency: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to calculate cost efficiency",
        }

