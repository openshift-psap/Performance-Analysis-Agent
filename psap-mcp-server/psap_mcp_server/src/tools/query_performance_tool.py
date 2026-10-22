"""Query performance metrics tool for RHAIIS benchmarks.

This tool allows querying specific performance metrics for models, accelerators, and configurations.
"""

import asyncio
from typing import Any, Dict, Optional

import pandas as pd

from psap_mcp_server.src.tools.benchmark_row_selection import select_latest_run_rows
from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.src.tools.profile_utils import parse_profile
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

_MAX_PAGE_SIZE = 25
_MAX_RUNTIME_ARGS_CHARS = 1_000


async def query_performance_metrics(
    model: Optional[str] = None,
    accelerator: Optional[str] = None,
    version: Optional[str] = None,
    tp: Optional[int] = None,
    profile: Optional[str] = None,
    top_n: int = 10,
    offset: int = 0,
    include_full_runtime_args: bool = False,
) -> Dict[str, Any]:
    """Query performance metrics from RHAIIS benchmarks.

    TOOL_NAME=query_performance_metrics
    DISPLAY_NAME=Query Performance Metrics
    USECASE=Get detailed performance metrics for AI models on different accelerators
    INSTRUCTIONS=1. Provide filters (model, accelerator, version, TP), 2. Get performance data, 3. Analyze results
    INPUT_DESCRIPTION=model (str, optional): Model name; accelerator (str, optional): H200/MI300X/TPU; version (str, optional): RHAIIS version; tp (int, optional): Tensor parallelism; profile (str, optional): Workload profile; top_n (int): Page size (max 25); offset (int): Rows to skip; include_full_runtime_args (bool): Explicitly include full runtime arguments
    OUTPUT_DESCRIPTION=Dictionary with filtered performance data including throughput, latency, error rates, and configuration details
    EXAMPLES=query_performance_metrics(model="Llama-3.3-70B", accelerator="H200"), query_performance_metrics(accelerator="MI300X", tp=8)
    PREREQUISITES=RHAIIS performance data must be loaded
    RELATED_TOOLS=compare_configurations, get_performance_rankings

    Args:
        model: Model name to filter (partial match supported)
        accelerator: Accelerator type (H200, MI300X, TPU, etc.)
        version: RHAIIS version to filter
        tp: Tensor parallelism value
        profile: Benchmark profile (e.g., "1k/1k", "512/2k")
        top_n: Number of results to return, capped at 25
        offset: Start row for pagination
        include_full_runtime_args: Include unabridged runtime arguments when needed

    Returns:
        Dictionary containing:
        - status: success/error
        - data: List of matching benchmark results
        - summary: Aggregate statistics
        - filters_applied: Dictionary of filters used
    """
    try:
        if top_n < 1 or offset < 0:
            return {"status": "error", "message": "top_n must be positive and offset must be non-negative"}
        page_size = min(top_n, _MAX_PAGE_SIZE)
        df = await asyncio.to_thread(load_rhaiis_data)
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load RHAIIS data",
                "message": "Could not load performance data file",
            }

        # Apply filters
        filtered_df = df.copy()
        filters_applied = {}

        if model:
            # Tokenized matching: split search term into tokens and match all
            # e.g., "maverick fp8" will match "Llama-4-Maverick-17B-128E-Instruct-FP8"
            model_tokens = model.lower().split()
            model_mask = filtered_df["model"].str.lower().apply(
                lambda x: all(token in x for token in model_tokens) if pd.notna(x) else False
            )
            filtered_df = filtered_df[model_mask]
            filters_applied["model"] = model

        if accelerator:
            filtered_df = filtered_df[filtered_df["accelerator"].str.contains(accelerator, case=False, na=False)]
            filters_applied["accelerator"] = accelerator

        if version:
            filtered_df = filtered_df[filtered_df["version"].str.contains(version, case=False, na=False)]
            filters_applied["version"] = version

        if tp is not None:
            filtered_df = filtered_df[filtered_df["TP"] == tp]
            filters_applied["TP"] = tp

        if profile:
            # Match profile based on prompt/output token counts
            if "prompt toks" in filtered_df.columns and "output toks" in filtered_df.columns:
                if "/" in profile:  # Format: "1k/1k", "2k/128", or "512/2k"
                    try:
                        prompt_tokens, output_tokens = parse_profile(profile)
                        filtered_df = filtered_df[
                            (filtered_df["prompt toks"] == prompt_tokens)
                            & (filtered_df["output toks"] == output_tokens)
                        ]
                        filters_applied["profile"] = profile
                    except (ValueError, AttributeError, KeyError):
                        pass

        filtered_df = select_latest_run_rows(filtered_df)

        if filtered_df.empty:
            return {
                "status": "success",
                "data": [],
                "summary": {"total_results": 0},
                "filters_applied": filters_applied,
                "message": "No results found matching the specified filters",
            }

        # Sort by throughput (descending) and limit results
        if "output_tok/sec" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("output_tok/sec", ascending=False)

        limited_df = filtered_df.iloc[offset : offset + page_size]

        # Prepare results
        results = []
        for _, row in limited_df.iterrows():
            # Check if Grafana metrics are available (requires uuid + time range)
            grafana_available = (
                pd.notna(row.get("uuid")) and
                pd.notna(row.get("guidellm_start_time_ms")) and
                pd.notna(row.get("guidellm_end_time_ms"))
            )
            
            runtime_args = row.get("runtime_args")
            runtime_args = str(runtime_args) if pd.notna(runtime_args) else None
            result = {
                "model": row.get("model"),
                "accelerator": row.get("accelerator"),
                "version": row.get("version"),
                "TP": int(row.get("TP")) if pd.notna(row.get("TP")) else None,
                "uuid": row.get("uuid") if pd.notna(row.get("uuid")) else None,
                "grafana_available": grafana_available,
                "grafana_note": "GPU/vLLM metrics available via query_grafana_metrics" if grafana_available else "GPU metrics not available (missing time range data)",
                "throughput": {
                    "output_tokens_per_sec": float(row.get("output_tok/sec")) if pd.notna(row.get("output_tok/sec")) else None,
                    "total_tokens_per_sec": float(row.get("total_tok/sec")) if pd.notna(row.get("total_tok/sec")) else None,
                },
                "latency": {
                    "ttft_median_ms": float(row.get("ttft_median")) if pd.notna(row.get("ttft_median")) else None,
                    "ttft_p95_ms": float(row.get("ttft_p95")) if pd.notna(row.get("ttft_p95")) else None,
                    "ttft_p99_ms": float(row.get("ttft_p99")) if pd.notna(row.get("ttft_p99")) else None,
                    "itl_median_ms": float(row.get("itl_median")) if pd.notna(row.get("itl_median")) else None,
                    "itl_p95_ms": float(row.get("itl_p95")) if pd.notna(row.get("itl_p95")) else None,
                    "itl_p99_ms": float(row.get("itl_p99")) if pd.notna(row.get("itl_p99")) else None,
                    "tpot_median_ms": float(row.get("tpot_median")) if pd.notna(row.get("tpot_median")) else None,
                    "tpot_p95_ms": float(row.get("tpot_p95")) if pd.notna(row.get("tpot_p95")) else None,
                    "tpot_p99_ms": float(row.get("tpot_p99")) if pd.notna(row.get("tpot_p99")) else None,
                    "request_latency_median_sec": float(row.get("request_latency_median")) if pd.notna(row.get("request_latency_median")) else None,
                    "request_latency_min_sec": float(row.get("request_latency_min")) if pd.notna(row.get("request_latency_min")) else None,
                    "request_latency_max_sec": float(row.get("request_latency_max")) if pd.notna(row.get("request_latency_max")) else None,
                },
                "concurrency": {
                    "measured": float(row.get("measured concurrency")) if pd.notna(row.get("measured concurrency")) else None,
                    "intended": float(row.get("intended concurrency")) if pd.notna(row.get("intended concurrency")) else None,
                },
                "requests": {
                    "successful": int(row.get("successful_requests")) if pd.notna(row.get("successful_requests")) else None,
                    "errored": int(row.get("errored_requests")) if pd.notna(row.get("errored_requests")) else None,
                    "error_rate_percent": (float(row.get("errored_requests")) / (float(row.get("successful_requests")) + float(row.get("errored_requests"))) * 100) 
                        if pd.notna(row.get("successful_requests")) and pd.notna(row.get("errored_requests")) and (row.get("successful_requests") + row.get("errored_requests")) > 0 else 0,
                },
                "profile": {
                    "prompt_tokens": int(row.get("prompt toks")) if pd.notna(row.get("prompt toks")) else None,
                    "output_tokens": int(row.get("output toks")) if pd.notna(row.get("output toks")) else None,
                },
                "efficiency_ratio": float(row.get("efficiency_ratio")) if pd.notna(row.get("efficiency_ratio")) else None,
                "runtime_args": (
                    runtime_args if include_full_runtime_args else runtime_args[:_MAX_RUNTIME_ARGS_CHARS]
                ) if runtime_args else None,
                "runtime_args_truncated": bool(
                    runtime_args and not include_full_runtime_args and len(runtime_args) > _MAX_RUNTIME_ARGS_CHARS
                ),
            }
            results.append(result)

        # Calculate summary statistics
        summary = {
            "total_results": len(filtered_df),
            "results_shown": len(results),
            "offset": offset,
            "next_offset": offset + len(results) if offset + len(results) < len(filtered_df) else None,
            "page_size": page_size,
            "avg_throughput": float(filtered_df["output_tok/sec"].mean()) if "output_tok/sec" in filtered_df.columns else None,
            "max_throughput": float(filtered_df["output_tok/sec"].max()) if "output_tok/sec" in filtered_df.columns else None,
            "min_throughput": float(filtered_df["output_tok/sec"].min()) if "output_tok/sec" in filtered_df.columns else None,
            "avg_ttft_ms": float(filtered_df["ttft_median"].mean()) if "ttft_median" in filtered_df.columns else None,
        }

        logger.info(f"Query returned {len(results)} results out of {len(filtered_df)} total matches")

        return {
            "status": "success",
            "data": results,
            "summary": summary,
            "filters_applied": filters_applied,
            "message": f"Found {len(results)} performance records",
        }

    except Exception as e:
        logger.error(f"Error querying performance metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to query performance metrics",
        }
