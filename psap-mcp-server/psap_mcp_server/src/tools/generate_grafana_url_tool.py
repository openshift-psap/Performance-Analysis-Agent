"""Generate Grafana Dashboard URL Tool.

This tool generates URLs to the Grafana GPU metrics dashboard for specific benchmark runs,
automatically selecting the correct dashboard based on the accelerator type and run date.

Dashboard selection logic (matches performance-dashboard/dashboard.py):
  - H200 runs before Jan 1 2026  -> H200_OLD dashboard
  - H200 runs on/after Jan 1 2026 -> H200_NEW dashboard
  - MI300X                        -> MI300X dashboard
"""

from typing import Dict, Optional
import pandas as pd

from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# Grafana base URL
GRAFANA_BASE_URL = "https://grafana-psap-obs.apps.ocp4.intlab.redhat.com"

# Dashboard configuration — mirrors GRAFANA_DASHBOARDS in dashboard.py
GRAFANA_DASHBOARDS = {
    "H200_OLD": {
        "dashboard_id": "6475e6106c33fe",
        "dashboard_name": "vllm-2b-dcgm-metrics-psap-8xh200-2",
    },
    "H200_NEW": {
        "dashboard_id": "7a3b910e7e827c",
        "dashboard_name": "vllm-2b-dcgm-metrics-psap-rhaiis-h200",
    },
    "MI300X": {
        "dashboard_id": "amd-ods-az-amd-01",
        "dashboard_name": "vllm-2b-rocm-gpu-metrics-ods-az-amd-01",
    },
}

# Jan 1, 2026 00:00:00 UTC in milliseconds — cutoff for H200 dashboard selection
H200_DASHBOARD_CUTOFF_MS = 1767225600000


def _select_dashboard(accelerator: str, start_ms: Optional[float]) -> Optional[str]:
    """Select the correct Grafana dashboard key based on accelerator and run date.

    Matches the logic in performance-dashboard/dashboard.py create_grafana_link().

    Args:
        accelerator: Accelerator name from the CSV (e.g. "H200", "MI300X")
        start_ms: Run start time in milliseconds (from guidellm_start_time_ms)

    Returns:
        Dashboard key ("H200_OLD", "H200_NEW", "MI300X") or None if unsupported.
    """
    acc_upper = (accelerator or "").strip().upper()

    if "H200" in acc_upper or "H100" in acc_upper:
        if start_ms is not None and start_ms >= H200_DASHBOARD_CUTOFF_MS:
            return "H200_NEW"
        return "H200_OLD"

    if "MI300" in acc_upper:
        return "MI300X"

    # Unsupported accelerator
    return None


async def generate_grafana_url(
    deployment_uuid: str,
    rate_interval: str = "1m",
) -> Dict:
    """Generate a Grafana dashboard URL for a specific benchmark run.

    Automatically selects the correct Grafana dashboard based on the accelerator
    type and run date, then builds a direct URL with the correct time range and
    filters pre-applied.

    Dashboard selection:
      - H200 runs before Jan 1 2026 use the old H200 dashboard
      - H200 runs on/after Jan 1 2026 use the new H200 dashboard
      - MI300X runs use the AMD dashboard

    TOOL_NAME=generate_grafana_url
    DISPLAY_NAME=Generate Grafana Dashboard URL for Benchmark Run
    USECASE=Create a direct link to the Grafana GPU metrics dashboard for a specific benchmark run, with correct time range and filters pre-applied
    INSTRUCTIONS=1. Provide deployment UUID from CSV, 2. Get a direct link to view GPU metrics in Grafana
    INPUT_DESCRIPTION=deployment_uuid (str, required): Deployment UUID from consolidated_dashboard.csv; rate_interval (str, optional): Rate interval for metrics (default: "1m")
    OUTPUT_DESCRIPTION=Dictionary with Grafana URL, dashboard info, and run details
    EXAMPLES=generate_grafana_url(deployment_uuid="94794a83-14a6-4dc0-8dc2-81645a7b05e6")
    PREREQUISITES=Valid deployment UUID from performance data
    RELATED_TOOLS=query_grafana_metrics, query_performance_metrics

    Args:
        deployment_uuid: Deployment UUID from the benchmark run (from CSV 'uuid' column)
        rate_interval: Rate interval for metric calculations (default: "1m")

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - url: Full Grafana dashboard URL
        - dashboard: Dashboard name and type
        - run_info: Details about the benchmark run
        - message: Status message
    """
    try:
        from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
        df = load_rhaiis_data()

        if df is None:
            return {
                "status": "error",
                "message": "Could not load performance data from CSV",
            }

        # Find the run with matching UUID
        run_data = df[df["uuid"] == deployment_uuid]

        if run_data.empty:
            return {
                "status": "error",
                "message": f"No run found with UUID: {deployment_uuid}",
                "deployment_uuid": deployment_uuid,
            }

        row = run_data.iloc[0]

        # Extract run metadata
        accelerator = row.get("accelerator", "")
        model = row.get("model", "")
        version = row.get("version", "")
        tp_value = int(row.get("TP")) if pd.notna(row.get("TP")) else None

        # Get time range from CSV (milliseconds)
        start_ms = row.get("guidellm_start_time_ms")
        end_ms = row.get("guidellm_end_time_ms")

        has_time_range = pd.notna(start_ms) and pd.notna(end_ms) and start_ms != "" and end_ms != ""
        start_ms_int = int(start_ms) if has_time_range else None
        end_ms_int = int(end_ms) if has_time_range else None

        # Select the correct dashboard
        dashboard_key = _select_dashboard(accelerator, start_ms_int)

        if dashboard_key is None:
            return {
                "status": "error",
                "message": f"No Grafana dashboard configured for accelerator: {accelerator}",
                "deployment_uuid": deployment_uuid,
                "supported_accelerators": ["H200", "H100", "MI300X"],
            }

        dashboard_config = GRAFANA_DASHBOARDS[dashboard_key]
        dashboard_id = dashboard_config["dashboard_id"]
        dashboard_name = dashboard_config["dashboard_name"]

        # Build URL — matches the format from dashboard.py create_grafana_link()
        if has_time_range:
            full_url = (
                f"{GRAFANA_BASE_URL}/d/{dashboard_id}/{dashboard_name}"
                f"?orgId=1&from={start_ms_int}&to={end_ms_int}"
                f"&timezone=browser&var-deployment_uuid={deployment_uuid}"
                f"&var-deployment_pod_name=$__all&var-rate_interval={rate_interval}"
            )
            time_range_info = {"from_ms": start_ms_int, "to_ms": end_ms_int}
        else:
            # Fallback: no time range available, use last 6 hours
            full_url = (
                f"{GRAFANA_BASE_URL}/d/{dashboard_id}/{dashboard_name}"
                f"?orgId=1&from=now-6h&to=now"
                f"&timezone=browser&var-deployment_uuid={deployment_uuid}"
                f"&var-deployment_pod_name=$__all&var-rate_interval={rate_interval}"
            )
            time_range_info = {"from": "now-6h", "to": "now", "note": "No timestamps in CSV, using default range"}

        return {
            "status": "success",
            "url": full_url,
            "dashboard": {
                "key": dashboard_key,
                "name": dashboard_name,
                "id": dashboard_id,
            },
            "run_info": {
                "deployment_uuid": deployment_uuid,
                "accelerator": accelerator,
                "model": model,
                "version": version,
                "tp": tp_value,
                "time_range": time_range_info,
            },
            "message": f"Grafana dashboard URL generated for {model} on {accelerator} (dashboard: {dashboard_key})",
            "note": "Open this URL in your browser to view real-time GPU metrics for this benchmark run.",
        }

    except Exception as e:
        logger.error(f"Error generating Grafana URL: {e}")
        return {
            "status": "error",
            "message": f"Error generating Grafana URL: {str(e)}",
            "deployment_uuid": deployment_uuid,
        }

