"""Calculate energy efficiency for GPU inference benchmarks.

Computes GPU energy consumption (kWh) and energy-per-token efficiency (Wh/1M tokens)
by fetching real GPU power data from Grafana/Prometheus. Falls back to estimated
power values when Grafana data is unavailable.
"""

from typing import Any, Dict, Optional

import pandas as pd

from psap_mcp_server.src.tools.grafana_metrics_tool import query_grafana_metrics
from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# Fallback average inference power per GPU (kW) when Grafana data is unavailable
FALLBACK_GPU_POWER_KW = {
    "H200": 0.475,
    "MI300X": 0.525,
}
DEFAULT_FALLBACK_POWER_KW = 0.400

# Per-concurrency benchmark duration changed from 600s to 450s starting RHAIIS 3.4
_CONCURRENCY_DURATION_SECONDS_DEFAULT = 600
_CONCURRENCY_DURATION_SECONDS_V34 = 450
_V34_THRESHOLD = (3, 4)


def _concurrency_duration_seconds(version: str) -> int:
    """Return per-concurrency benchmark duration based on RHAIIS version.

    RHAIIS 3.4+ uses 450s per concurrency level; earlier versions use 600s.
    """
    try:
        numeric_part = version.split("RHAIIS-", 1)[1] if "RHAIIS-" in version else version
        segments = numeric_part.split("-")[0].split(".")
        major_minor = (int(segments[0]), int(segments[1]))
        if major_minor >= _V34_THRESHOLD:
            return _CONCURRENCY_DURATION_SECONDS_V34
    except (IndexError, ValueError):
        pass
    return _CONCURRENCY_DURATION_SECONDS_DEFAULT


def _parse_profile_tokens(profile: str):
    """Parse a profile string like '1k/1k' into (prompt_toks, output_toks) candidate lists."""
    if "/" not in profile:
        return None, None
    parts = profile.lower().split("/")
    if len(parts) != 2:
        return None, None

    def parse_token_count(s):
        s = s.strip()
        if "k" in s:
            num = float(s.replace("k", ""))
            return [int(num * 1000), int(num * 1024)]
        return [int(s)]

    try:
        return parse_token_count(parts[0]), parse_token_count(parts[1])
    except (ValueError, AttributeError):
        return None, None


async def _get_gpu_power_from_grafana(deployment_uuid: str) -> Optional[float]:
    """Query Grafana for average GPU power (watts) for a benchmark run.

    Returns average power per GPU in watts, or None if unavailable.
    """
    try:
        result = await query_grafana_metrics(
            deployment_uuid=deployment_uuid,
            metrics=["gpu_power_usage"],
        )
        if result.get("status") != "success":
            return None

        power_data = result.get("metrics", {}).get("gpu_power_usage", {})
        summary = power_data.get("summary", {})
        avg_power = summary.get("avg")
        if avg_power is not None and avg_power > 0:
            return float(avg_power)
    except Exception as e:
        logger.debug(f"Grafana power query failed for {deployment_uuid}: {e}")
    return None


async def calculate_energy_efficiency(
    accelerator: Optional[str] = None,
    model: Optional[str] = None,
    version: Optional[str] = None,
    profile: Optional[str] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Calculate energy efficiency (energy per million tokens) for GPU inference benchmarks.

    Fetches real GPU power consumption from Grafana/Prometheus metrics and combines
    it with benchmark throughput data to compute energy efficiency. Falls back to
    estimated power values when Grafana data is unavailable (e.g., outside retention window).

    TOOL_NAME=calculate_energy_efficiency
    DISPLAY_NAME=Calculate Energy Efficiency for GPU Inference
    USECASE=Calculate GPU energy consumption (kWh) and energy per million tokens (Wh) using measured GPU power from Grafana metrics
    INSTRUCTIONS=1. Optionally filter by accelerator, model, version, and profile, 2. Tool fetches GPU power from Grafana for each benchmark group, 3. Returns energy metrics ranked by efficiency (lowest Wh/1M tokens first)
    INPUT_DESCRIPTION=accelerator (str, optional): Filter by accelerator (e.g., "H200", "MI300X"); model (str, optional): Filter by model name; version (str, optional): Filter by RHAIIS version; profile (str, optional): Workload profile (e.g., "1k/1k", "512/2k"); top_n (int): Number of results to return
    OUTPUT_DESCRIPTION=Dictionary with energy analysis per configuration including GPU power (kW), total energy (kWh), energy per 1M tokens (Wh), and whether power was measured from Grafana or estimated
    EXAMPLES=calculate_energy_efficiency(accelerator="H200", version="RHAIIS-3.2.5"), calculate_energy_efficiency(model="Llama-3.3-70B", profile="1k/1k"), calculate_energy_efficiency(accelerator="MI300X", profile="512/2k", top_n=5)
    PREREQUISITES=RHAIIS performance data with throughput metrics; Grafana access for measured GPU power (optional, falls back to estimates)
    RELATED_TOOLS=calculate_cost_efficiency, query_grafana_metrics

    Args:
        accelerator: Filter by accelerator type (e.g., "H200", "MI300X")
        model: Filter by model name (partial match)
        version: Filter by RHAIIS version (exact match, case-insensitive)
        profile: Workload profile as ISL/OSL (e.g., "1k/1k", "512/2k", "8k/1k")
        top_n: Number of results to return (default 10)

    Returns:
        Dictionary containing:
        - status: success/error
        - energy_analysis: List of configurations with energy metrics
        - most_efficient: Configuration with lowest energy per 1M tokens
        - statistics: Aggregate stats across all configurations
        - filters_applied: Filters that were used
    """
    try:
        df = load_rhaiis_data()
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load performance data",
                "message": "Could not load RHAIIS data file",
            }

        filtered_df = df.copy()
        filters_applied = {}

        if accelerator:
            filtered_df = filtered_df[
                filtered_df["accelerator"].str.contains(accelerator, case=False, na=False)
            ]
            filters_applied["accelerator"] = accelerator

        if model:
            filtered_df = filtered_df[
                filtered_df["model"].str.contains(model, case=False, na=False)
            ]
            filters_applied["model"] = model

        if version:
            filtered_df = filtered_df[
                filtered_df["version"].str.lower() == version.lower()
            ]
            filters_applied["version"] = version

        if profile:
            if "prompt toks" in filtered_df.columns and "output toks" in filtered_df.columns:
                prompt_candidates, output_candidates = _parse_profile_tokens(profile)
                if prompt_candidates and output_candidates:
                    match_found = False
                    for p_tok in prompt_candidates:
                        for o_tok in output_candidates:
                            matched_df = filtered_df[
                                (filtered_df["prompt toks"] == p_tok)
                                & (filtered_df["output toks"] == o_tok)
                            ]
                            if not matched_df.empty:
                                filtered_df = matched_df
                                match_found = True
                                break
                        if match_found:
                            break
                    if match_found:
                        filters_applied["profile"] = profile

        if filtered_df.empty:
            return {
                "status": "success",
                "energy_analysis": [],
                "filters_applied": filters_applied,
                "message": "No results found matching the specified filters",
            }

        # Determine concurrency column
        if "intended_concurrency" in filtered_df.columns:
            concurrency_col = "intended_concurrency"
        elif "intended concurrency" in filtered_df.columns:
            concurrency_col = "intended concurrency"
        elif "measured concurrency" in filtered_df.columns:
            concurrency_col = "measured concurrency"
        else:
            concurrency_col = None

        energy_results = []

        for (grp_model, grp_acc, grp_tp), group in filtered_df.groupby(
            ["model", "accelerator", "TP"]
        ):
            gpu_count = int(grp_tp)
            grp_version = group["version"].iloc[0] if "version" in group.columns else ""

            # Build profile string for output
            prompt_toks = int(group["prompt toks"].iloc[0]) if "prompt toks" in group.columns and pd.notna(group["prompt toks"].iloc[0]) else None
            output_toks = int(group["output toks"].iloc[0]) if "output toks" in group.columns and pd.notna(group["output toks"].iloc[0]) else None
            grp_profile = f"{prompt_toks}/{output_toks}" if prompt_toks and output_toks else None

            # Count concurrency levels and compute runtime
            if concurrency_col:
                num_concurrencies = group[concurrency_col].nunique()
            else:
                num_concurrencies = len(group)
            seconds_per_concurrency = _concurrency_duration_seconds(grp_version)
            runtime_seconds = num_concurrencies * seconds_per_concurrency
            runtime_minutes = runtime_seconds / 60
            runtime_hours = runtime_seconds / 3600

            # --- Fetch GPU power from Grafana ---
            # Pick a representative UUID from the group (first non-null)
            power_watts = None
            power_source = "fallback_estimate"
            uuid_used = None

            if "uuid" in group.columns:
                uuids = group["uuid"].dropna().unique()
                for uuid_candidate in uuids:
                    power_watts = await _get_gpu_power_from_grafana(str(uuid_candidate))
                    if power_watts is not None:
                        uuid_used = str(uuid_candidate)
                        power_source = "grafana"
                        break

            if power_watts is None:
                # Fallback to estimated values
                fallback_key = None
                for key in FALLBACK_GPU_POWER_KW:
                    if key in str(grp_acc):
                        fallback_key = key
                        break
                power_watts = FALLBACK_GPU_POWER_KW.get(fallback_key, DEFAULT_FALLBACK_POWER_KW) * 1000  # convert kW to W

            gpu_power_kw = power_watts / 1000.0
            total_power_kw = gpu_power_kw * gpu_count
            total_energy_kwh = total_power_kw * runtime_hours  # replicas=1 for RHAIIS

            # Throughput and token efficiency
            avg_throughput = 0.0
            total_tokens = 0.0
            energy_per_1m_tokens = None

            if "output_tok/sec" in group.columns:
                avg_throughput = group["output_tok/sec"].mean()
                if pd.notna(avg_throughput) and avg_throughput > 0:
                    total_tokens = avg_throughput * runtime_seconds
                    if total_tokens > 0:
                        energy_per_1m_tokens = (total_energy_kwh * 1e9) / total_tokens

            model_short = grp_model.split("/")[-1] if "/" in grp_model else grp_model

            energy_results.append({
                "model": grp_model,
                "model_short": model_short,
                "accelerator": grp_acc,
                "version": grp_version,
                "profile": grp_profile,
                "tp": gpu_count,
                "num_concurrencies": num_concurrencies,
                "seconds_per_concurrency": seconds_per_concurrency,
                "benchmark_duration_minutes": round(runtime_minutes, 1),
                "benchmark_duration_hours": round(runtime_hours, 2),
                "avg_gpu_power_kw": round(gpu_power_kw, 4),
                "total_power_draw_kw": round(total_power_kw, 4),
                "total_energy_kwh": round(total_energy_kwh, 4),
                "avg_throughput_tok_per_sec": round(float(avg_throughput), 2) if avg_throughput > 0 else None,
                "total_tokens_generated": int(total_tokens) if total_tokens > 0 else None,
                "energy_per_1m_tokens_wh": round(float(energy_per_1m_tokens), 2) if energy_per_1m_tokens else None,
                "power_data_source": power_source,
                "deployment_uuid_used": uuid_used,
            })

        if not energy_results:
            return {
                "status": "success",
                "energy_analysis": [],
                "filters_applied": filters_applied,
                "message": "No energy data could be calculated for the filtered results",
            }

        # Separate results with and without efficiency data for sorting
        with_efficiency = [r for r in energy_results if r["energy_per_1m_tokens_wh"] is not None]
        without_efficiency = [r for r in energy_results if r["energy_per_1m_tokens_wh"] is None]
        with_efficiency.sort(key=lambda x: x["energy_per_1m_tokens_wh"])

        sorted_results = with_efficiency + without_efficiency
        top_results = sorted_results[:top_n]

        most_efficient = with_efficiency[0] if with_efficiency else None

        all_efficiency_values = [r["energy_per_1m_tokens_wh"] for r in with_efficiency]
        all_energy_values = [r["total_energy_kwh"] for r in energy_results]
        grafana_count = sum(1 for r in energy_results if r["power_data_source"] == "grafana")

        statistics = {
            "total_configurations": len(energy_results),
            "results_shown": len(top_results),
            "grafana_measured_count": grafana_count,
            "fallback_estimated_count": len(energy_results) - grafana_count,
            "lowest_energy_per_1m_tokens_wh": min(all_efficiency_values) if all_efficiency_values else None,
            "highest_energy_per_1m_tokens_wh": max(all_efficiency_values) if all_efficiency_values else None,
            "avg_energy_per_1m_tokens_wh": round(sum(all_efficiency_values) / len(all_efficiency_values), 2) if all_efficiency_values else None,
            "lowest_total_energy_kwh": min(all_energy_values) if all_energy_values else None,
            "highest_total_energy_kwh": max(all_energy_values) if all_energy_values else None,
        }

        logger.info(
            f"Calculated energy efficiency for {len(energy_results)} configurations "
            f"({grafana_count} from Grafana, {len(energy_results) - grafana_count} estimated)"
        )

        return {
            "status": "success",
            "energy_analysis": top_results,
            "most_efficient": most_efficient,
            "statistics": statistics,
            "filters_applied": filters_applied,
            "message": f"Energy efficiency calculated for {len(top_results)} configurations",
        }

    except Exception as e:
        logger.error(f"Error calculating energy efficiency: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to calculate energy efficiency",
        }
