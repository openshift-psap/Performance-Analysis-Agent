"""Tool to compare vLLM and DCGM metrics between two benchmark runs."""

from typing import Dict, Any, List, Optional
from .performance_data_loader import load_rhaiis_data
from .grafana_metrics_tool import query_grafana_metrics
import pandas as pd


async def compare_grafana_metrics(
    run1_uuid: Optional[str] = None,
    run2_uuid: Optional[str] = None,
    run1_model: Optional[str] = None,
    run1_version: Optional[str] = None,
    run1_accelerator: Optional[str] = None,
    run1_profile: Optional[str] = None,
    run2_model: Optional[str] = None,
    run2_version: Optional[str] = None,
    run2_accelerator: Optional[str] = None,
    run2_profile: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    metric_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare vLLM and DCGM metrics between two benchmark runs.
    
    TOOL_NAME=compare_grafana_metrics
    DISPLAY_NAME=Compare Grafana Metrics Between Runs
    USECASE=Compare GPU utilization, temperature, power, and vLLM metrics between two different benchmark runs
    INSTRUCTIONS=1. Provide either UUIDs or model+version+accelerator+profile for both runs, 2. Optionally specify metric_type ('dcgm', 'vllm', or 'both'), 3. Tool will fetch and compare the specified metrics
    INPUT_DESCRIPTION=run1_uuid (str, optional): UUID of first run; run2_uuid (str, optional): UUID of second run; run1_model/version/accelerator/profile (str, optional): Details for first run; run2_model/version/accelerator/profile (str, optional): Details for second run; metrics (list, optional): Specific metrics to compare; metric_type (str, optional): Type of metrics ('dcgm', 'vllm', or 'both' - default is 'both')
    OUTPUT_DESCRIPTION=Dictionary with comparison of metrics between the two runs, including differences and percentage changes
    
    Args:
        run1_uuid: UUID of the first benchmark run
        run2_uuid: UUID of the second benchmark run
        run1_model: Model name for first run (if UUID not provided)
        run1_version: Version for first run (if UUID not provided)
        run1_accelerator: Accelerator for first run (if UUID not provided)
        run1_profile: Profile for first run (if UUID not provided)
        run2_model: Model name for second run (if UUID not provided)
        run2_version: Version for second run (if UUID not provided)
        run2_accelerator: Accelerator for second run (if UUID not provided)
        run2_profile: Profile for second run (if UUID not provided)
        metrics: List of specific metrics to compare (optional)
        metric_type: Type of metrics to compare - 'dcgm' for GPU metrics only, 'vllm' for vLLM metrics only, 'both' for all metrics (default: 'both')
        
    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - run1_info: Information about first run
        - run2_info: Information about second run
        - comparison: Detailed comparison of metrics
        - summary: High-level summary of differences
    """
    
    try:
        df = load_rhaiis_data()
        
        # Helper function to find UUID from parameters
        def find_uuid(model, version, accelerator, profile, run_label):
            if not all([model, version, accelerator, profile]):
                return None, f"Missing parameters for {run_label}. Need model, version, accelerator, and profile."
            
            # Parse profile
            if '/' in profile:
                parts = profile.split('/')
                isl = int(parts[0].replace('k', '000').replace('K', '000'))
                osl = int(parts[1].replace('k', '000').replace('K', '000'))
            else:
                return None, f"Invalid profile format for {run_label}: {profile}"
            
            # Filter data
            filtered = df[
                (df['model'].str.contains(model, case=False, na=False)) &
                (df['version'].str.contains(version, case=False, na=False)) &
                (df['accelerator'].str.contains(accelerator, case=False, na=False)) &
                (df['prompt toks'] == isl) &
                (df['output toks'] == osl)
            ]
            
            if len(filtered) == 0:
                return None, f"No data found for {run_label} with the specified parameters."
            
            # Check if UUID and time fields are populated
            valid_runs = filtered[
                pd.notna(filtered['uuid']) &
                pd.notna(filtered['guidellm_start_time_ms']) &
                pd.notna(filtered['guidellm_end_time_ms'])
            ]
            
            if len(valid_runs) == 0:
                return None, f"No Grafana data available for {run_label}. The run exists but doesn't have GPU metrics."
            
            # Take the first valid run (or you could add logic to select based on concurrency)
            return valid_runs.iloc[0]['uuid'], None
        
        # Get UUIDs for both runs
        if not run1_uuid:
            run1_uuid, error = find_uuid(run1_model, run1_version, run1_accelerator, run1_profile, "Run 1")
            if error:
                return {
                    "status": "error",
                    "message": error
                }
        
        if not run2_uuid:
            run2_uuid, error = find_uuid(run2_model, run2_version, run2_accelerator, run2_profile, "Run 2")
            if error:
                return {
                    "status": "error",
                    "message": error
                }
        
        # Query Grafana metrics for both runs
        run1_data = await query_grafana_metrics(deployment_uuid=run1_uuid, metrics=metrics)
        run2_data = await query_grafana_metrics(deployment_uuid=run2_uuid, metrics=metrics)
        
        if run1_data.get('status') != 'success':
            return {
                "status": "error",
                "message": f"Failed to fetch metrics for Run 1: {run1_data.get('message')}"
            }
        
        if run2_data.get('status') != 'success':
            return {
                "status": "error",
                "message": f"Failed to fetch metrics for Run 2: {run2_data.get('message')}"
            }
        
        # Compare metrics
        comparison = {}
        summary = []
        
        run1_metrics = run1_data.get('metrics', {})
        run2_metrics = run2_data.get('metrics', {})
        
        # Find common metrics
        common_metrics = set(run1_metrics.keys()) & set(run2_metrics.keys())
        
        if not common_metrics:
            return {
                "status": "error",
                "message": "No common metrics found between the two runs."
            }
        
        # Filter metrics based on metric_type
        if metric_type and metric_type.lower() != 'both':
            if metric_type.lower() == 'dcgm':
                # Filter to only DCGM metrics (GPU-related)
                common_metrics = {m for m in common_metrics if any(keyword in m.lower() for keyword in ['gpu', 'dcgm', 'temperature', 'power', 'memory', 'utilization'])}
            elif metric_type.lower() == 'vllm':
                # Filter to only vLLM metrics
                common_metrics = {m for m in common_metrics if 'vllm' in m.lower() or any(keyword in m.lower() for keyword in ['request', 'latency', 'cache', 'token'])}
        
        if not common_metrics:
            return {
                "status": "error",
                "message": f"No {metric_type if metric_type else 'common'} metrics found between the two runs."
            }
        
        for metric_name in sorted(common_metrics):
            run1_metric = run1_metrics[metric_name]
            run2_metric = run2_metrics[metric_name]
            
            run1_summary = run1_metric.get('summary', {})
            run2_summary = run2_metric.get('summary', {})
            
            metric_comparison = {
                "run1": run1_summary,
                "run2": run2_summary,
                "differences": {}
            }
            
            # Calculate differences for numerical values
            for stat_key in ['avg', 'min', 'max', 'p50', 'p95', 'p99']:
                if stat_key in run1_summary and stat_key in run2_summary:
                    val1 = run1_summary[stat_key]
                    val2 = run2_summary[stat_key]
                    
                    if val1 is not None and val2 is not None:
                        diff = val2 - val1
                        pct_change = ((val2 - val1) / val1 * 100) if val1 != 0 else None
                        
                        metric_comparison['differences'][stat_key] = {
                            "absolute_difference": round(diff, 2),
                            "percentage_change": round(pct_change, 2) if pct_change is not None else None,
                            "direction": "increased" if diff > 0 else "decreased" if diff < 0 else "no change"
                        }
            
            comparison[metric_name] = metric_comparison
            
            # Add to summary if there's a significant change in average
            if 'avg' in metric_comparison['differences']:
                avg_diff = metric_comparison['differences']['avg']
                if avg_diff['percentage_change'] is not None and abs(avg_diff['percentage_change']) > 5:
                    # Get versions for clear direction
                    run1_version = run1_data.get('version', 'Run 1')
                    run2_version = run2_data.get('version', 'Run 2')
                    
                    summary.append({
                        "metric": metric_name,
                        "change": f"{avg_diff['direction']} by {abs(avg_diff['percentage_change']):.1f}% (from {run1_version} to {run2_version})",
                        "from_value": run1_summary['avg'],
                        "to_value": run2_summary['avg'],
                        "from_version": run1_version,
                        "to_version": run2_version
                    })
        
        return {
            "status": "success",
            "run1_info": {
                "uuid": run1_uuid,
                "model": run1_data.get('model'),
                "version": run1_data.get('version'),
                "cluster": run1_data.get('cluster'),
                "time_range": run1_data.get('time_range_expected')
            },
            "run2_info": {
                "uuid": run2_uuid,
                "model": run2_data.get('model'),
                "version": run2_data.get('version'),
                "cluster": run2_data.get('cluster'),
                "time_range": run2_data.get('time_range_expected')
            },
            "comparison": comparison,
            "summary": summary,
            "message": f"Successfully compared {len(comparison)} {metric_type if metric_type else 'all'} metrics between the two runs.",
            "note": "Percentage changes greater than 5% are highlighted in the summary. Changes show the direction from Run 1 to Run 2.",
            "metric_type": metric_type if metric_type else "both"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error comparing Grafana metrics: {str(e)}"
        }

