"""Dataset metadata and discovery tool for RHAIIS benchmarks.

This tool provides information about available options in the dataset.
"""

from typing import Any, Dict, List

import pandas as pd

from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()


async def get_dataset_metadata(
    metadata_type: str = "all",
) -> Dict[str, Any]:
    """Get metadata about available options in the RHAIIS dataset.

    TOOL_NAME=get_dataset_metadata
    DISPLAY_NAME=Get Dataset Metadata
    USECASE=Discover all available models, accelerators, versions, profiles, and configurations in the dataset
    INSTRUCTIONS=1. Specify metadata type (all, accelerators, models, profiles, versions, or stats), 2. Get complete list of available options
    INPUT_DESCRIPTION=metadata_type (str): Type of metadata to retrieve - "all" (everything), "accelerators", "models", "profiles", "versions", or "stats"
    OUTPUT_DESCRIPTION=Dictionary with comprehensive metadata including all unique values for accelerators, models, profiles, versions, and dataset statistics
    EXAMPLES=get_dataset_metadata("all"), get_dataset_metadata("accelerators"), get_dataset_metadata("profiles")
    PREREQUISITES=RHAIIS performance data must be loaded
    RELATED_TOOLS=query_performance_metrics, compare_configurations

    Args:
        metadata_type: Type of metadata to return:
            - "all": All metadata (default)
            - "accelerators": List of all available accelerators
            - "models": List of all available models
            - "profiles": List of all available ISL/OSL (prompt/output token) combinations
            - "versions": List of all available RHAIIS versions
            - "stats": Dataset statistics

    Returns:
        Dictionary containing requested metadata
    """
    try:
        df = load_rhaiis_data()
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load RHAIIS data",
                "message": "Could not load performance data file",
            }

        result = {"status": "success"}

        # Get accelerators
        if metadata_type in ["all", "accelerators"]:
            accelerators = sorted(df["accelerator"].dropna().unique().tolist())
            result["accelerators"] = {
                "count": len(accelerators),
                "list": accelerators,
            }

        # Get models
        if metadata_type in ["all", "models"]:
            models = sorted(df["model"].dropna().unique().tolist())
            result["models"] = {
                "count": len(models),
                "list": models,
            }

        # Get profiles (ISL/OSL combinations)
        if metadata_type in ["all", "profiles"]:
            if "prompt toks" in df.columns and "output toks" in df.columns:
                # Get unique combinations
                profiles_df = df[["prompt toks", "output toks"]].dropna().drop_duplicates()
                profiles = []
                for _, row in profiles_df.iterrows():
                    prompt_tok = int(row["prompt toks"])
                    output_tok = int(row["output toks"])
                    
                    # Create both formats
                    exact_format = f"{prompt_tok}/{output_tok}"
                    
                    # Create shorthand format
                    if prompt_tok >= 1000 and prompt_tok % 1000 == 0:
                        prompt_short = f"{prompt_tok // 1000}k"
                    elif prompt_tok >= 1024 and prompt_tok % 1024 == 0:
                        prompt_short = f"{prompt_tok // 1024}k"
                    else:
                        prompt_short = str(prompt_tok)
                    
                    if output_tok >= 1000 and output_tok % 1000 == 0:
                        output_short = f"{output_tok // 1000}k"
                    elif output_tok >= 1024 and output_tok % 1024 == 0:
                        output_short = f"{output_tok // 1024}k"
                    else:
                        output_short = str(output_tok)
                    
                    shorthand_format = f"{prompt_short}/{output_short}"
                    
                    profiles.append({
                        "ISL_prompt_tokens": prompt_tok,
                        "OSL_output_tokens": output_tok,
                        "format_exact": exact_format,
                        "format_shorthand": shorthand_format,
                    })
                
                # Sort by prompt tokens, then output tokens
                profiles.sort(key=lambda x: (x["ISL_prompt_tokens"], x["OSL_output_tokens"]))
                
                result["profiles"] = {
                    "count": len(profiles),
                    "list": profiles,
                    "description": "ISL = Input Sequence Length (prompt tokens), OSL = Output Sequence Length (output tokens)",
                }

        # Get versions
        if metadata_type in ["all", "versions"]:
            versions = sorted(df["version"].dropna().unique().tolist())
            result["versions"] = {
                "count": len(versions),
                "list": versions,
            }

        # Get TP values
        if metadata_type in ["all", "stats"]:
            tp_values = sorted(df["TP"].dropna().unique().astype(int).tolist())
            result["tensor_parallelism_values"] = {
                "count": len(tp_values),
                "list": tp_values,
            }

        # Get dataset statistics
        if metadata_type in ["all", "stats"]:
            result["dataset_statistics"] = {
                "total_benchmark_runs": len(df),
                "total_unique_configurations": len(
                    df[["model", "accelerator", "version", "TP", "prompt toks", "output toks"]]
                    .drop_duplicates()
                ),
                "date_range": {
                    "earliest": df["guidellm_start_time_ms"].min() if "guidellm_start_time_ms" in df.columns else None,
                    "latest": df["guidellm_end_time_ms"].max() if "guidellm_end_time_ms" in df.columns else None,
                },
            }

        logger.info(f"Retrieved {metadata_type} metadata from RHAIIS dataset")

        return result

    except Exception as e:
        logger.error(f"Error getting dataset metadata: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to retrieve dataset metadata",
        }

