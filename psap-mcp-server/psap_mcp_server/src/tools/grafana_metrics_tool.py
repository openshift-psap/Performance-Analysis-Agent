"""MCP tool for querying Grafana metrics via Prometheus API.

This tool fetches GPU and vLLM performance metrics from Grafana dashboards for specific benchmark runs.
Supports both NVIDIA GPUs (DCGM metrics) and AMD MI300X GPUs (ROCm metrics).
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx
import pandas as pd

from psap_mcp_server.src.settings import settings


# GPU metric mappings for different accelerator types
# NVIDIA uses DCGM (Data Center GPU Manager) metrics
NVIDIA_GPU_METRICS = {
    "gpu_utilization": "DCGM_FI_DEV_GPU_UTIL",
    "gpu_memory_used": "DCGM_FI_DEV_FB_USED",
    "gpu_memory_free": "DCGM_FI_DEV_FB_FREE",
    "gpu_temperature": "DCGM_FI_DEV_GPU_TEMP",
    "gpu_power_usage": "DCGM_FI_DEV_POWER_USAGE",
    "gpu_sm_clock": "DCGM_FI_DEV_SM_CLOCK",
    "gpu_mem_clock": "DCGM_FI_DEV_MEM_CLOCK",
}

# AMD MI300X uses ROCm SMI metrics
AMD_GPU_METRICS = {
    "gpu_utilization": "gpu_gfx_activity",
    "gpu_memory_used": "gpu_used_vram",
    "gpu_memory_free": "gpu_free_vram",
    "gpu_temperature": "gpu_junction_temperature",
    "gpu_power_usage": "gpu_package_power",
    "gpu_sm_clock": 'gpu_clock{clock_type="GPU_CLOCK_TYPE_SYSTEM"}',
    "gpu_mem_clock": 'gpu_clock{clock_type="GPU_CLOCK_TYPE_MEMORY"}',
    # AMD-specific additional metrics
    "gpu_memory_temperature": "gpu_memory_temperature",
    "gpu_energy_consumed": "gpu_energy_consumed",
    "gpu_memory_controller_activity": "gpu_umc_activity",
}

# Accelerator type detection from CSV accelerator column
NVIDIA_ACCELERATORS = {"H200"}
AMD_ACCELERATORS = {"MI300X"}


def get_accelerator_type(accelerator: str) -> str:
    """Determine if accelerator is NVIDIA or AMD based on name."""
    accelerator_upper = accelerator.upper() if accelerator else ""
    
    for nvidia_acc in NVIDIA_ACCELERATORS:
        if nvidia_acc in accelerator_upper:
            return "nvidia"
    
    for amd_acc in AMD_ACCELERATORS:
        if amd_acc in accelerator_upper:
            return "amd"
    
    # Default to NVIDIA for unknown accelerators
    return "nvidia"


async def query_grafana_metrics(
    deployment_uuid: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Query vLLM and GPU metrics from Grafana for a specific benchmark run.
    
    Automatically detects GPU type (NVIDIA/AMD) from the CSV and queries appropriate metrics.
    
    TOOL_NAME=query_grafana_metrics
    DISPLAY_NAME=Query Grafana vLLM and GPU Metrics for Benchmark Run
    USECASE=Fetch real-time vLLM inference metrics AND GPU hardware metrics from Grafana for specific benchmark runs. Supports both NVIDIA (DCGM) and AMD MI300X (ROCm) GPUs.
    INSTRUCTIONS=1. Provide deployment UUID from CSV, 2. Optionally specify specific metrics, 3. Get detailed vLLM and GPU metrics
    INPUT_DESCRIPTION=deployment_uuid (str, required): Deployment UUID from consolidated_dashboard.csv; start_time (str, optional): Unix timestamp (default: from CSV); end_time (str, optional): Unix timestamp (default: from CSV); metrics (list[str], optional): Specific metrics to query (default: all metrics)
    OUTPUT_DESCRIPTION=Dictionary with vLLM metrics (request rates, latency) and GPU metrics (utilization, memory, temperature, power) - automatically uses correct metric names for NVIDIA or AMD GPUs
    EXAMPLES=query_grafana_metrics(deployment_uuid="94794a83-14a6-4dc0-8dc2-81645a7b05e6"), query_grafana_metrics(deployment_uuid="abc123", metrics=["gpu_utilization", "vllm_requests"])
    PREREQUISITES=Grafana API token set in GRAFANA_API_TOKEN environment variable, deployment UUID from performance data, data within retention period (typically 7 days)
    RELATED_TOOLS=query_performance_metrics, compare_configurations
    
    Args:
        deployment_uuid: Deployment UUID from the benchmark run (from CSV 'uuid' column)
        start_time: Unix timestamp (seconds since epoch) - optional, will use CSV times if not provided
        end_time: Unix timestamp (seconds since epoch) - optional, will use CSV times if not provided
        metrics: List of specific metrics to query. Available metrics:
            **vLLM Metrics:**
            - "vllm_request_rate": Request rate (requests/sec)
            - "vllm_request_total": Total successful requests (counter)
            - "vllm_kv_cache_usage": KV cache usage percentage
            - "vllm_requests_running": Number of requests currently running
            - "vllm_requests_waiting": Number of requests waiting
            - "vllm_ttft_avg": Average Time to First Token (seconds)
            - "vllm_tpot_avg": Average Time Per Output Token (seconds)
            - "vllm_e2e_latency_avg": Average End-to-End Latency (seconds)
            
            **GPU Metrics (works for both NVIDIA and AMD):**
            - "gpu_utilization": GPU utilization percentage (0-100%)
            - "gpu_memory_used": GPU memory used (bytes)
            - "gpu_memory_free": GPU memory free (bytes)
            - "gpu_temperature": GPU temperature (Celsius)
            - "gpu_power_usage": Power consumption (watts)
            - "gpu_sm_clock": SM/GFX clock speed (MHz)
            - "gpu_mem_clock": Memory clock speed (MHz)
            
            **AMD MI300X Additional Metrics:**
            - "gpu_hbm_temperature": HBM memory temperature (Celsius)
            - "gpu_energy_consumed": Total energy consumption (Joules)
            - "gpu_tensor_activity": Tensor/MMA core utilization (%)
            - "gpu_memory_controller_activity": Memory controller activity (%)
            
            - "all": All available metrics (default)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - deployment_uuid: UUID of the benchmark run
        - accelerator: GPU type detected (e.g., "H200", "MI300X")
        - accelerator_vendor: "nvidia" or "amd"
        - cluster: Cluster name where the run executed
        - time_range: {"start": start_time, "end": end_time}
        - metrics: Dictionary of metric name -> data points (includes all GPUs)
        - summary: Aggregated statistics (min, max, avg, p50, p95, p99) for each metric
        - message: Status message
    """
    # Validate Grafana configuration
    if not settings.GRAFANA_API_TOKEN:
        return {
            "status": "error",
            "message": "Grafana API token not configured. Please set GRAFANA_API_TOKEN environment variable.",
            "deployment_uuid": deployment_uuid,
        }
    
    # Load run metadata from CSV (time range, TP value, accelerator type, etc.)
    tp_value = None
    model_name = None
    accelerator = None
    accelerator_vendor = "nvidia"  # Default
    
    try:
        from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
        df = load_rhaiis_data()
        
        # Find the run with matching UUID
        run_data = df[df["uuid"] == deployment_uuid]
        if not run_data.empty:
            row = run_data.iloc[0]
            
            # Get accelerator type to determine which metrics to use
            accelerator = row.get("accelerator", "")
            accelerator_vendor = get_accelerator_type(accelerator)
            
            # Use Unix timestamps directly (divide ms by 1000)
            if not start_time or not end_time:
                start_ms = row.get("guidellm_start_time_ms")
                end_ms = row.get("guidellm_end_time_ms")
                
                if start_ms and end_ms:
                    start_time = str(int(start_ms / 1000))
                    end_time = str(int(end_ms / 1000))
            
            # Get TP value (number of GPUs used)
            tp_value = int(row.get("TP")) if pd.notna(row.get("TP")) else None
            model_name = row.get("model")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not load run metadata from CSV: {e}",
            "deployment_uuid": deployment_uuid,
        }
    
    if not start_time or not end_time:
        return {
            "status": "error",
            "message": "Time range not found in CSV and not provided. Please specify start_time and end_time.",
            "deployment_uuid": deployment_uuid,
        }
    
    # Select GPU metric mappings based on accelerator type
    gpu_metric_map = AMD_GPU_METRICS if accelerator_vendor == "amd" else NVIDIA_GPU_METRICS
    
    # If no specific metrics requested, fetch all available for this GPU type
    if not metrics or "all" in metrics:
        base_metrics = [
            # vLLM metrics (same for all GPUs)
            "vllm_request_rate",
            "vllm_request_total",
            "vllm_kv_cache_usage",
            "vllm_requests_running",
            "vllm_ttft_avg",
            "vllm_tpot_avg",
            "vllm_e2e_latency_avg",
            # Common GPU metrics
            "gpu_utilization",
            "gpu_memory_used",
            "gpu_memory_free",
            "gpu_temperature",
            "gpu_power_usage",
        ]
        
        # Add AMD-specific metrics if applicable
        if accelerator_vendor == "amd":
            base_metrics.extend([
                "gpu_hbm_temperature",
                "gpu_energy_consumed",
                "gpu_tensor_activity",
                "gpu_memory_controller_activity",
            ])
        
        metrics = base_metrics
    
    # First, query a vLLM metric to get the cluster label (needed for DCGM metrics)
    # DCGM metrics don't have deployment_uuid label, but they have cluster label
    cluster_name = None
    
    # Try to get cluster from vLLM metrics
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        url = f"{settings.GRAFANA_URL}/api/datasources/uid/{settings.GRAFANA_DATASOURCE_UID}/resources/api/v1/query_range"
        headers = {
            "Authorization": f"Bearer {settings.GRAFANA_API_TOKEN}",
            "Content-Type": "application/json",
        }
        
        # Query a simple vLLM metric to get cluster
        try:
            int(start_time)
            start_unix = start_time
            end_unix = end_time
        except (ValueError, TypeError):
            return {
                "status": "error",
                "message": "ISO timestamp format not supported. Please provide Unix timestamps or let the tool load times from CSV using deployment_uuid.",
                "deployment_uuid": deployment_uuid,
            }
        
        params = {
            "query": f'vllm:request_success_total{{deployment_uuid="{deployment_uuid}"}}',
            "start": start_unix,
            "end": end_unix,
            "step": "300",
        }
        
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                results_data = data.get("data", {}).get("result", [])
                if results_data:
                    # Extract cluster from first result
                    cluster_name = results_data[0].get("metric", {}).get("cluster")
        except Exception:
            pass  # Will handle missing cluster later
    
    if not cluster_name:
        # If no cluster found, can only query vLLM metrics
        cluster_note = "GPU metrics unavailable (cluster not found)"
    else:
        vendor_name = "ROCm/AMD" if accelerator_vendor == "amd" else "DCGM/NVIDIA"
        cluster_note = f"{vendor_name} GPU metrics queried from cluster: {cluster_name}"
    
    # Define Prometheus queries for each metric
    # vLLM metrics use deployment_uuid, DCGM metrics use cluster
    metric_queries = {
        # vLLM metrics
        "vllm_request_rate": f'rate(vllm:request_success_total{{deployment_uuid="{deployment_uuid}"}}[5m])',
        "vllm_request_total": f'vllm:request_success_total{{deployment_uuid="{deployment_uuid}"}}',
        "vllm_kv_cache_usage": f'vllm:kv_cache_usage_perc{{deployment_uuid="{deployment_uuid}"}}',
        "vllm_requests_running": f'vllm:num_requests_running{{deployment_uuid="{deployment_uuid}"}}',
        "vllm_requests_waiting": f'vllm:num_requests_waiting{{deployment_uuid="{deployment_uuid}"}}',
        # For histograms, query the average by dividing sum by count
        "vllm_ttft_avg": f'rate(vllm:time_to_first_token_seconds_sum{{deployment_uuid="{deployment_uuid}"}}[5m]) / rate(vllm:time_to_first_token_seconds_count{{deployment_uuid="{deployment_uuid}"}}[5m])',
        "vllm_tpot_avg": f'rate(vllm:time_per_output_token_seconds_sum{{deployment_uuid="{deployment_uuid}"}}[5m]) / rate(vllm:time_per_output_token_seconds_count{{deployment_uuid="{deployment_uuid}"}}[5m])',
        "vllm_e2e_latency_avg": f'rate(vllm:e2e_request_latency_seconds_sum{{deployment_uuid="{deployment_uuid}"}}[5m]) / rate(vllm:e2e_request_latency_seconds_count{{deployment_uuid="{deployment_uuid}"}}[5m])',
    }
    
    # Add GPU queries if cluster was found - use correct metrics based on GPU vendor
    if cluster_name:
        gpu_queries = {}
        
        for metric_key, prometheus_metric in gpu_metric_map.items():
            # Handle metrics that already have labels (like gpu_clock{clock_type="..."})
            if "{" in prometheus_metric:
                # Insert cluster label into existing label set
                base_metric, labels = prometheus_metric.split("{", 1)
                labels = labels.rstrip("}")
                gpu_queries[metric_key] = f'{base_metric}{{cluster="{cluster_name}",{labels}}}'
            else:
                gpu_queries[metric_key] = f'{prometheus_metric}{{cluster="{cluster_name}"}}'
        
        metric_queries.update(gpu_queries)
    
    # STEP 1: Identify active GPUs based on GPU utilization (if TP is known)
    # This ensures all GPU metrics use the same GPU IDs
    identified_active_gpus = None
    
    if tp_value and cluster_name:
        # Query GPU utilization to identify which GPUs are actually being used
        # Use correct metric based on GPU vendor
        gpu_util_metric = "gpu_gfx_activity" if accelerator_vendor == "amd" else "DCGM_FI_DEV_GPU_UTIL"
        
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            try:
                url = f"{settings.GRAFANA_URL}/api/datasources/uid/{settings.GRAFANA_DATASOURCE_UID}/resources/api/v1/query_range"
                headers = {
                    "Authorization": f"Bearer {settings.GRAFANA_API_TOKEN}",
                    "Content-Type": "application/json",
                }
                
                params = {
                    "query": f'{gpu_util_metric}{{cluster="{cluster_name}"}}',
                    "start": start_time,
                    "end": end_time,
                    "step": "60",
                }
                
                response = await client.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success" and data.get("data", {}).get("result"):
                        # Analyze GPU activity from utilization metric
                        gpu_activity = {}
                        for idx, series in enumerate(data["data"]["result"]):
                            labels = series.get("metric", {})
                            values = series.get("values", [])
                            gpu_id = labels.get("gpu") or labels.get("GPU") or labels.get("gpu_id") or labels.get("gpu_index")

                            if gpu_id is not None:
                                try:
                                    gpu_id_key = int(gpu_id)
                                except (ValueError, TypeError):
                                    gpu_id_key = idx
                            else:
                                # DCGM often uses UUID instead of a numeric gpu label.
                                # Fall back to series index so each GPU gets a unique key.
                                gpu_id_key = idx

                            activity_score = sum(1 for _, value in values if float(value) > 0)
                            gpu_activity[gpu_id_key] = activity_score

                        # Sort by activity and take top TP GPUs
                        if gpu_activity:
                            sorted_gpus = sorted(gpu_activity.items(), key=lambda x: x[1], reverse=True)
                            identified_active_gpus = [gpu_id for gpu_id, _ in sorted_gpus[:tp_value]]
            except Exception as e:
                # If we can't identify active GPUs, we'll fall back to the old behavior
                pass
    
    # Query Grafana API for requested metrics
    results = {}
    errors = []
    
    # Track actual data time range (min/max timestamps found in data)
    actual_start_time = None
    actual_end_time = None
    
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        for metric_name in metrics:
            if metric_name not in metric_queries:
                errors.append(f"Unknown metric: {metric_name}")
                continue
            
            query = metric_queries[metric_name]
            
            try:
                # Query Prometheus via Grafana datasource proxy API
                # Correct endpoint format: /api/datasources/uid/{uid}/resources/...
                url = f"{settings.GRAFANA_URL}/api/datasources/uid/{settings.GRAFANA_DATASOURCE_UID}/resources/api/v1/query_range"
                
                # Ensure we have Unix timestamps (Prometheus API requirement)
                # Check if already Unix timestamps (numeric strings) or ISO format
                try:
                    # Try to use as-is if it's already a Unix timestamp
                    int(start_time)
                    start_unix = start_time
                    end_unix = end_time
                except (ValueError, TypeError):
                    # Not a Unix timestamp, must be ISO format - not currently supported
                    # Use Unix timestamps from CSV via deployment_uuid parameter
                    return {
                        "status": "error",
                        "message": "ISO timestamp format not supported. Please provide Unix timestamps or let the tool load times from CSV using deployment_uuid.",
                        "deployment_uuid": deployment_uuid,
                    }
                
                # Calculate appropriate step size to avoid "exceeded maximum resolution" error
                # Grafana limits to 11,000 points per series
                # For typical benchmark runs (1-2 hours), use 1 minute resolution
                params = {
                    "query": query,
                    "start": start_unix,
                    "end": end_unix,
                    "step": "60",  # 60 seconds (1 minute resolution)
                }
                
                headers = {
                    "Authorization": f"Bearer {settings.GRAFANA_API_TOKEN}",
                    "Content-Type": "application/json",
                }
                
                response = await client.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("status") == "success" and data.get("data", {}).get("result"):
                        # Process Prometheus response
                        is_dcgm_metric = metric_name.startswith("gpu_")
                        
                        # STEP 1: Collect all GPU data and analyze activity
                        gpu_data = {}  # gpu_id -> list of data points

                        for idx, series in enumerate(data["data"]["result"]):
                            labels = series.get("metric", {})
                            values = series.get("values", [])

                            # Extract GPU ID - label varies by vendor
                            gpu_id = labels.get("gpu") or labels.get("GPU") or labels.get("gpu_id") or labels.get("gpu_index")

                            if is_dcgm_metric:
                                if gpu_id is not None:
                                    try:
                                        gpu_id_key = int(gpu_id)
                                    except (ValueError, TypeError):
                                        gpu_id_key = idx
                                else:
                                    # DCGM often uses UUID instead of a numeric
                                    # gpu label; fall back to series index.
                                    gpu_id_key = idx

                                if gpu_id_key not in gpu_data:
                                    gpu_data[gpu_id_key] = []

                                for timestamp, value in values:
                                    gpu_data[gpu_id_key].append({
                                        "timestamp": timestamp,
                                        "value": float(value),
                                        "labels": labels,
                                    })
                            else:
                                # Non-DCGM metrics - include all
                                if "all" not in gpu_data:
                                    gpu_data["all"] = []
                                for timestamp, value in values:
                                    gpu_data["all"].append({
                                        "timestamp": timestamp,
                                        "value": float(value),
                                        "labels": labels,
                                    })
                        
                        # STEP 2: Use pre-identified active GPUs or fallback to activity-based detection
                        active_gpus = []
                        metric_data = []
                        
                        if is_dcgm_metric and gpu_data and tp_value is not None:
                            # Use pre-identified active GPUs if available (from GPU utilization query)
                            if identified_active_gpus is not None:
                                active_gpus = identified_active_gpus
                            else:
                                # Fallback: Calculate activity score for each GPU (sum of non-zero values)
                                gpu_activity = {}
                                for gpu_id, points in gpu_data.items():
                                    if gpu_id != "all":
                                        activity_score = sum(1 for p in points if p["value"] > 0)
                                        data_point_count = len(points)
                                        gpu_activity[gpu_id] = (activity_score, data_point_count)
                                
                                # Sort GPUs by activity (non-zero values) and data point count
                                sorted_gpus = sorted(gpu_activity.items(), 
                                                   key=lambda x: (x[1][0], x[1][1]), 
                                                   reverse=True)
                                
                                # Take top TP GPUs with most activity
                                active_gpus = [gpu_id for gpu_id, _ in sorted_gpus[:tp_value]]
                            
                            # Collect data only from active GPUs (filter to only those in gpu_data)
                            available_active_gpus = [gpu_id for gpu_id in active_gpus if gpu_id in gpu_data]
                            
                            for gpu_id in available_active_gpus:
                                for point in gpu_data[gpu_id]:
                                    # Track min/max timestamps
                                    timestamp = point["timestamp"]
                                    if actual_start_time is None or timestamp < actual_start_time:
                                        actual_start_time = timestamp
                                    if actual_end_time is None or timestamp > actual_end_time:
                                        actual_end_time = timestamp
                                    
                                    metric_data.append({
                                        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                                        "value": point["value"],
                                        "labels": point["labels"],
                                    })
                            
                            # Update active_gpus to reflect what was actually used
                            active_gpus = available_active_gpus
                        else:
                            # Non-DCGM metrics or no TP filtering - include all data
                            for gpu_id, points in gpu_data.items():
                                for point in points:
                                    timestamp = point["timestamp"]
                                    if actual_start_time is None or timestamp < actual_start_time:
                                        actual_start_time = timestamp
                                    if actual_end_time is None or timestamp > actual_end_time:
                                        actual_end_time = timestamp
                                    
                                    metric_data.append({
                                        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                                        "value": point["value"],
                                        "labels": point["labels"],
                                    })
                        
                        # Calculate summary statistics
                        if metric_data:
                            values = [point["value"] for point in metric_data]
                            values.sort()
                            n = len(values)
                            
                            summary = {
                                "min": round(min(values), 2),
                                "max": round(max(values), 2),
                                "avg": round(sum(values) / n, 2),
                                "p50": round(values[int(n * 0.5)], 2),
                                "p95": round(values[int(n * 0.95)], 2),
                                "p99": round(values[int(n * 0.99)], 2) if n >= 100 else round(values[-1], 2),
                                "data_points": n,
                            }
                            
                            # Add GPU filtering note for DCGM metrics
                            if is_dcgm_metric and active_gpus:
                                gpu_list = sorted(active_gpus)
                                summary["active_gpus"] = gpu_list
                                # Indicate if GPUs were identified from utilization or from this metric
                                source = "identified from GPU utilization" if identified_active_gpus is not None else "filtered by activity"
                                summary["gpu_note"] = f"Statistics from GPU(s): {gpu_list} (TP={tp_value}, {source})" if tp_value else f"Statistics from GPU(s): {gpu_list}"
                        else:
                            summary = {"message": "No data points found"}
                        
                        results[metric_name] = {
                            "summary": summary,
                            "total_points": len(metric_data),
                            "sample_points": metric_data[:5],
                        }
                    else:
                        errors.append(f"{metric_name}: No data found")
                else:
                    errors.append(f"{metric_name}: HTTP {response.status_code} - {response.text[:200]}")
            
            except Exception as e:
                errors.append(f"{metric_name}: {str(e)}")
    
    if not results:
        vendor_label = "ROCm" if accelerator_vendor == "amd" else "DCGM"
        return {
            "status": "error",
            "message": f"No metrics could be retrieved for {accelerator} ({vendor_label}). Errors: {'; '.join(errors)}",
            "deployment_uuid": deployment_uuid,
            "accelerator": accelerator,
            "accelerator_vendor": accelerator_vendor,
            "cluster": cluster_name,
            "time_range_expected": {
                "start": start_time,
                "end": end_time,
                "start_iso": datetime.fromtimestamp(int(start_time)).isoformat(),
                "end_iso": datetime.fromtimestamp(int(end_time)).isoformat(),
            },
            "note": f"Expected {vendor_label} metrics for {accelerator}. Check if metrics are being collected for cluster '{cluster_name}'.",
        }
    
    # Count vLLM vs GPU metrics
    vllm_count = sum(1 for k in results.keys() if k.startswith("vllm_"))
    gpu_count = sum(1 for k in results.keys() if k.startswith("gpu_"))
    
    message_parts = []
    if vllm_count > 0:
        message_parts.append(f"{vllm_count} vLLM metrics")
    if gpu_count > 0:
        vendor_label = "ROCm" if accelerator_vendor == "amd" else "DCGM"
        message_parts.append(f"{gpu_count} {vendor_label} GPU metrics")
    
    # Convert actual timestamps to ISO format
    actual_start_iso = datetime.fromtimestamp(actual_start_time).isoformat() if actual_start_time else None
    actual_end_iso = datetime.fromtimestamp(actual_end_time).isoformat() if actual_end_time else None
    
    # Check if data range differs from expected range
    expected_start_iso = datetime.fromtimestamp(int(start_time)).isoformat()
    expected_end_iso = datetime.fromtimestamp(int(end_time)).isoformat()
    
    data_range_note = ""
    if actual_end_iso and actual_end_iso != expected_end_iso:
        duration_expected = (int(end_time) - int(start_time)) / 60  # minutes
        duration_actual = (actual_end_time - actual_start_time) / 60 if actual_start_time and actual_end_time else 0
        data_range_note = f"⚠️ CSV indicates run duration of {duration_expected:.1f} min, but Grafana data only available for {duration_actual:.1f} min (possible data retention/collection issue)"
    
    return {
        "status": "success",
        "deployment_uuid": deployment_uuid,
        "accelerator": accelerator,
        "accelerator_vendor": accelerator_vendor,
        "cluster": cluster_name,
        "model": model_name,
        "tensor_parallelism": {
            "tp": tp_value,
            "note": f"Using {tp_value} GPU(s)" if tp_value else "TP value not available"
        },
        "time_range_expected": {
            "start": start_time,
            "end": end_time,
            "start_iso": expected_start_iso,
            "end_iso": expected_end_iso,
            "note": "Time range from CSV (guidellm_start_time_ms/guidellm_end_time_ms)"
        },
        "time_range_actual": {
            "start": str(int(actual_start_time)) if actual_start_time else None,
            "end": str(int(actual_end_time)) if actual_end_time else None,
            "start_iso": actual_start_iso,
            "end_iso": actual_end_iso,
            "note": "Actual time range of data found in Grafana"
        },
        "data_completeness": {
            "complete": actual_end_iso == expected_end_iso if actual_end_iso else False,
            "note": data_range_note if data_range_note else "✅ Data complete for full benchmark duration"
        },
        "metrics": results,
        "errors": errors if errors else None,
        "message": f"Successfully retrieved {' and '.join(message_parts)} for deployment {deployment_uuid} on {accelerator} ({accelerator_vendor.upper()})" + (f" (TP={tp_value}, {tp_value} GPU(s) used)" if tp_value and gpu_count > 0 else ""),
        "note": cluster_note,
    }

