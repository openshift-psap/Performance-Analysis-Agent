"""Configuration discovery tool for RHAIIS benchmarks.

This tool discovers what configurations (models, accelerators, profiles, etc.) 
were actually tested for specific filters.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()


async def discover_configurations(
    discover_what: str = "models",
    version: Optional[str] = None,
    accelerator: Optional[str] = None,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    tp: Optional[int] = None,
) -> Dict[str, Any]:
    """Discover what configurations were actually tested with given filters.

    TOOL_NAME=discover_configurations
    DISPLAY_NAME=Discover Tested Configurations
    USECASE=Find out what models/accelerators/profiles/versions were actually tested in benchmark runs with specific filters
    INSTRUCTIONS=1. Specify what you want to discover (models, accelerators, profiles, versions, or tensor_parallelism), 2. Apply filters to narrow down the search, 3. Get back all unique values that match the filters
    INPUT_DESCRIPTION=discover_what (str): What to discover - "models", "accelerators", "profiles", "versions", or "tensor_parallelism"; Optional filters: version, accelerator, model, profile, tp
    OUTPUT_DESCRIPTION=Dictionary containing the discovered unique values, count, and filter summary
    EXAMPLES=discover_configurations("models", version="sglang-0.5.3"), discover_configurations("accelerators", model="deepseek-ai/DeepSeek-R1-0528"), discover_configurations("profiles", version="RHAIIS-3.2.3", accelerator="H200")
    PREREQUISITES=RHAIIS performance data must be loaded
    RELATED_TOOLS=get_dataset_metadata, query_performance_metrics

    This is specifically designed for questions like:
    - "What models were tested for sglang-0.5.3?"
    - "What accelerators did we test deepseek on?"
    - "What profiles are available for RHAIIS-3.2.3 on H200?"

    Args:
        discover_what: What to discover - "models", "accelerators", "profiles", "versions", or "tensor_parallelism"
        version: Filter by specific version (e.g., "sglang-0.5.3", "RHAIIS-3.2.3")
        accelerator: Filter by specific accelerator (e.g., "H200", "MI300X", "TPU")
        model: Filter by specific model
        profile: Filter by profile in format "ISL/OSL" (e.g., "1k/1k", "2048/128")
        tp: Filter by tensor parallelism value

    Returns:
        Dictionary containing discovered unique values and metadata
    """
    try:
        df = load_rhaiis_data()
        if df is None:
            return {
                "status": "error",
                "error": "Failed to load RHAIIS data",
                "message": "Could not load performance data file",
            }

        # Store original count
        original_count = len(df)

        # Apply filters
        filters_applied = {}
        
        if version:
            df = df[df["version"].str.contains(version, case=False, na=False)]
            filters_applied["version"] = version
        
        if accelerator:
            df = df[df["accelerator"] == accelerator]
            filters_applied["accelerator"] = accelerator
        
        if model:
            # Tokenized matching: split search term into tokens and match all
            # e.g., "maverick fp8" will match "Llama-4-Maverick-17B-128E-Instruct-FP8"
            model_tokens = model.lower().split()
            model_mask = df["model"].str.lower().apply(
                lambda x: all(token in x for token in model_tokens) if pd.notna(x) else False
            )
            df = df[model_mask]
            filters_applied["model"] = model
        
        if profile:
            # Parse profile (e.g., "1k/1k" or "2048/128")
            if "/" in profile:
                parts = profile.lower().split("/")
                # Parse prompt tokens
                prompt_str = parts[0].strip()
                if prompt_str.endswith("k"):
                    prompt_toks = int(float(prompt_str[:-1]) * 1000)
                else:
                    prompt_toks = int(prompt_str)
                
                # Parse output tokens
                output_str = parts[1].strip()
                if output_str.endswith("k"):
                    output_toks = int(float(output_str[:-1]) * 1000)
                else:
                    output_toks = int(output_str)
                
                df = df[
                    (df["prompt toks"] == prompt_toks) & 
                    (df["output toks"] == output_toks)
                ]
                filters_applied["profile"] = profile
        
        if tp is not None:
            df = df[df["TP"] == tp]
            filters_applied["tp"] = tp

        if len(df) == 0:
            return {
                "status": "success",
                "discovered": discover_what,
                "filters_applied": filters_applied,
                "count": 0,
                "unique_values": [],
                "message": f"No benchmark runs found matching the specified filters",
                "total_records_matching_filters": 0,
                "original_total_records": original_count,
            }

        # Discover unique values based on what was requested
        result = {
            "status": "success",
            "discovered": discover_what,
            "filters_applied": filters_applied,
            "total_records_matching_filters": len(df),
            "original_total_records": original_count,
        }

        if discover_what == "models":
            unique_values = sorted(df["model"].dropna().unique().tolist())
            result["unique_values"] = unique_values
            result["count"] = len(unique_values)
            
        elif discover_what == "accelerators":
            unique_values = sorted(df["accelerator"].dropna().unique().tolist())
            result["unique_values"] = unique_values
            result["count"] = len(unique_values)
            
        elif discover_what == "versions":
            unique_values = sorted(df["version"].dropna().unique().tolist())
            result["unique_values"] = unique_values
            result["count"] = len(unique_values)
            
        elif discover_what == "profiles":
            if "prompt toks" in df.columns and "output toks" in df.columns:
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
                
                result["unique_values"] = profiles
                result["count"] = len(profiles)
                result["description"] = "ISL = Input Sequence Length (prompt tokens), OSL = Output Sequence Length (output tokens)"
            else:
                result["unique_values"] = []
                result["count"] = 0
                
        elif discover_what == "tensor_parallelism":
            unique_values = sorted(df["TP"].dropna().unique().astype(int).tolist())
            result["unique_values"] = unique_values
            result["count"] = len(unique_values)
            
        else:
            return {
                "status": "error",
                "error": f"Invalid discover_what value: {discover_what}",
                "message": "discover_what must be one of: models, accelerators, profiles, versions, tensor_parallelism",
            }

        logger.info(
            f"Discovered {result['count']} unique {discover_what} "
            f"from {len(df)} records (filters: {filters_applied})"
        )

        return result

    except Exception as e:
        logger.error(f"Error discovering configurations: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to discover configurations",
        }

