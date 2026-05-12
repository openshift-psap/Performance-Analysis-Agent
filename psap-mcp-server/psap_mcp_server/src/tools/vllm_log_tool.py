"""MCP tool for fetching and comparing vLLM server logs.

Log files are stored in a flat layout at ``s3://<LOG_S3_BUCKET>/<LOG_S3_PREFIX>/<uuid>.log``.
The UUID for each benchmark run is recorded in the consolidated dashboard CSV loaded by
``load_rhaiis_data()``.  The tools in this module look up the UUID from the CSV using the
caller-supplied model / version / accelerator filters, then download and parse the log.
"""

import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from psap_mcp_server.src.settings import settings
from psap_mcp_server.src.tools.performance_data_loader import load_rhaiis_data
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()


# ===================================================================== #
#  UUID lookup from benchmark CSV                                        #
# ===================================================================== #

def _find_uuid_for_run(
    version: str,
    model: Optional[str] = None,
    accelerator: Optional[str] = None,
    tp: Optional[int] = None,
    profile: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Look up the run UUID from the benchmark CSV.

    Filters the RHAIIS DataFrame by version (required) and optionally by
    model, accelerator, TP, and profile, then returns the UUID of the
    matching deployment.

    Returns:
        (uuid, None) on success, or (None, error_message) on failure.
    """
    df = load_rhaiis_data()
    if df is None or df.empty:
        return None, "Failed to load benchmark data (RHAIIS CSV)."

    if "uuid" not in df.columns:
        return None, "Benchmark CSV does not contain a 'uuid' column."

    filtered = df.copy()

    filtered = filtered[filtered["version"].str.contains(version, case=False, na=False)]
    if filtered.empty:
        available = sorted(df["version"].dropna().unique().tolist())
        return None, f"Version '{version}' not found. Available versions: {available}"

    if model:
        model_tokens = model.lower().split()
        mask = filtered["model"].str.lower().apply(
            lambda x: all(tok in x for tok in model_tokens) if pd.notna(x) else False
        )
        filtered = filtered[mask]
        if filtered.empty:
            available = sorted(df["model"].dropna().unique().tolist())
            return None, f"Model '{model}' not found for version '{version}'. Available models: {available}"

    if accelerator:
        filtered = filtered[filtered["accelerator"].str.contains(accelerator, case=False, na=False)]
        if filtered.empty:
            available = sorted(df["accelerator"].dropna().unique().tolist())
            return None, f"Accelerator '{accelerator}' not found for version '{version}'. Available: {available}"

    if tp is not None and "TP" in filtered.columns:
        filtered = filtered[filtered["TP"] == tp]
        if filtered.empty:
            return None, f"TP={tp} not found for version '{version}'. Try without tp filter."

    if profile and "prompt toks" in filtered.columns and "output toks" in filtered.columns:
        parts = profile.lower().split("/")
        if len(parts) == 2:
            try:
                def _parse_tok(s: str) -> int:
                    s = s.strip()
                    if "k" in s:
                        return int(float(s.replace("k", "")) * 1000)
                    return int(s)

                p_tok, o_tok = _parse_tok(parts[0]), _parse_tok(parts[1])
                filtered = filtered[
                    (filtered["prompt toks"] == p_tok) & (filtered["output toks"] == o_tok)
                ]
                if filtered.empty:
                    return None, f"Profile '{profile}' not found for version '{version}'."
            except ValueError:
                pass

    valid = filtered[filtered["uuid"].notna()]
    if valid.empty:
        return None, (
            f"Found {len(filtered)} matching row(s) but none have a UUID. "
            "Log file unavailable for this run."
        )

    unique_uuids = valid["uuid"].unique().tolist()
    if len(unique_uuids) > 1:
        info_cols = ["model", "version", "accelerator", "uuid"]
        if "TP" in valid.columns:
            info_cols.insert(3, "TP")
        if "prompt toks" in valid.columns and "output toks" in valid.columns:
            info_cols.insert(3, "prompt toks")
            info_cols.insert(4, "output toks")
        combos = valid[info_cols].drop_duplicates(subset=["uuid"]).to_dict("records")
        return None, (
            f"Ambiguous: {len(unique_uuids)} distinct runs match. "
            f"Please specify tp and/or profile to narrow down. "
            f"Matching runs: {combos}"
        )

    return str(unique_uuids[0]), None


# ===================================================================== #
#  S3 log fetching                                                       #
# ===================================================================== #

def _fetch_log_by_uuid(uuid: str) -> Tuple[Optional[str], str]:
    """Download ``<uuid>.log`` from S3.

    Returns:
        (log_contents, s3_key) on success, or (None, s3_key) on failure.
    """
    bucket = settings.LOG_S3_BUCKET
    prefix = settings.LOG_S3_PREFIX.rstrip("/")
    s3_key = f"{prefix}/{uuid}.log"

    try:
        from psap_mcp_server.src.tools.s3_utils import get_s3_client
        s3 = get_s3_client()
        logger.info(f"Downloading log from S3: s3://{bucket}/{s3_key}")
        response = s3.get_object(Bucket=bucket, Key=s3_key)
        return response["Body"].read().decode("utf-8", errors="replace"), s3_key
    except Exception as exc:
        logger.error(f"Failed to download log s3://{bucket}/{s3_key}: {exc}")
        return None, s3_key


# ===================================================================== #
#  Log parsing                                                           #
# ===================================================================== #

def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _safe_int(text: str) -> Optional[int]:
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


_STARTUP_COMPLETE_MARKER = "Application startup complete."


def _truncate_at_startup(raw: str) -> str:
    """Return only the portion of the log up to and including the startup-complete line."""
    for i, line in enumerate(raw.splitlines()):
        if _STARTUP_COMPLETE_MARKER in line:
            return "\n".join(raw.splitlines()[: i + 1])
    return raw


def _parse_vllm_log(raw: str) -> Dict[str, Any]:
    """Parse a vLLM server log into structured sections.

    Extracts configuration, compilation, memory, and timing information
    using regex patterns matched against known vLLM log output formats.
    Only the startup portion of the log (up to "Application startup complete")
    is analyzed; request-serving output is discarded.
    """
    raw = _truncate_at_startup(raw)
    result: Dict[str, Any] = {
        "server_config": {},
        "engine_config": {},
        "compilation": {},
        "memory": {},
        "timing": {},
        "warnings_errors": [],
    }

    lines = raw.splitlines()

    for line in lines:
        # --- Server config ---
        m = re.search(r"vLLM API server version (\S+)", line)
        if m:
            result["server_config"]["vllm_version"] = m.group(1)

        if "vllm_version" not in result["server_config"]:
            m = re.search(r"version\s+(\d+\.\S+)", line)
            if m and "utils.py" in line:
                result["server_config"]["vllm_version"] = m.group(1)

        m = re.search(r"non-default args:\s*(\{.+\})", line)
        if m:
            try:
                import ast
                args = ast.literal_eval(m.group(1))
                result["server_config"]["non_default_args"] = args
                for key in ("model", "max_model_len", "gpu_memory_utilization",
                            "enable_prefix_caching", "trust_remote_code", "port"):
                    if key in args:
                        result["server_config"][key] = args[key]
            except Exception:
                result["server_config"]["non_default_args_raw"] = m.group(1)

        # --- Architecture ---
        m = re.search(r"Resolved architecture:\s*(\S+)", line)
        if m:
            result["engine_config"]["architecture"] = m.group(1)

        # --- Engine config from init log line ---
        if "Initializing a V1 LLM engine" in line or "Initializing an LLM engine" in line:
            em = re.search(r"LLM engine \(v(\S+?)\)", line)
            if em and "vllm_version" not in result["server_config"]:
                result["server_config"]["vllm_version"] = em.group(1)

            for field, pattern in [
                ("dtype", r"dtype=(\S+?)(?:,|$)"),
                ("quantization", r"quantization=(\S+?)(?:,|$)"),
                ("tensor_parallel_size", r"tensor_parallel_size=(\d+)"),
                ("pipeline_parallel_size", r"pipeline_parallel_size=(\d+)"),
                ("data_parallel_size", r"data_parallel_size=(\d+)"),
                ("max_seq_len", r"max_seq_len=(\d+)"),
                ("enforce_eager", r"enforce_eager=(\S+?)(?:,|$)"),
                ("enable_prefix_caching", r"enable_prefix_caching=(\S+?)(?:,|$)"),
                ("enable_chunked_prefill", r"enable_chunked_prefill=(\S+?)(?:,|$)"),
                ("load_format", r"load_format=(\S+?)(?:,|$)"),
                ("kv_cache_dtype", r"kv_cache_dtype=(\S+?)(?:,|$)"),
            ]:
                em = re.search(pattern, line)
                if em:
                    val = em.group(1)
                    if val.isdigit():
                        result["engine_config"][field] = int(val)
                    elif val in ("True", "False"):
                        result["engine_config"][field] = val == "True"
                    else:
                        result["engine_config"][field] = val

            # CUDA graph mode
            em = re.search(r"cudagraph_mode=<CUDAGraphMode\.(\S+?):", line)
            if em:
                result["engine_config"]["cudagraph_mode"] = em.group(1)

            # Max cudagraph capture size
            em = re.search(r"max_cudagraph_capture_size['\"]?:\s*(\d+)", line)
            if em:
                result["engine_config"]["max_cudagraph_capture_size"] = int(em.group(1))

            # Count cudagraph capture sizes
            em = re.search(r"cudagraph_capture_sizes['\"]?:\s*\[([^\]]+)\]", line)
            if em:
                sizes = [s.strip() for s in em.group(1).split(",") if s.strip()]
                result["engine_config"]["num_cudagraph_capture_sizes"] = len(sizes)

        # --- Chunked prefill ---
        m = re.search(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)", line)
        if m:
            result["engine_config"]["chunked_prefill"] = True
            result["engine_config"]["max_num_batched_tokens"] = int(m.group(1))

        m = re.search(r"Overriding max cuda graph capture size to (\d+)", line)
        if m:
            result["engine_config"]["max_cudagraph_capture_size_override"] = int(m.group(1))

        # --- Attention backend ---
        m = re.search(r"Using (\S+) attention backend", line)
        if m:
            result["compilation"]["attention_backend"] = m.group(1)

        m = re.search(r"Using FlashAttention version (\d+)", line)
        if m:
            result["compilation"]["flash_attention_version"] = int(m.group(1))

        m = re.search(r"vLLM is using nccl==(\S+)", line)
        if m:
            result["compilation"]["nccl_version"] = m.group(1)

        m = re.search(r"Enabled custom fusions:\s*(.+)", line)
        if m:
            result["compilation"]["custom_fusions"] = m.group(1).strip()

        # --- MoE stream ---
        if "separate cuda stream for MoE shared_experts" in line:
            result["compilation"]["moe_shared_experts_stream"] = True

        # --- Quantization backend ---
        m = re.search(r"\[mxfp4\.py:\d+\]\s*Using (\S+) backend", line)
        if m:
            result["compilation"]["quantization_backend"] = m.group(1)

        # --- Model loading ---
        m = re.search(r"Loading weights took (\S+) seconds", line)
        if m:
            result["memory"]["weights_load_time_s"] = _safe_float(m.group(1))

        m = re.search(r"Model loading took (\S+) GiB memory and (\S+) seconds", line)
        if m:
            result["memory"]["model_memory_gib"] = _safe_float(m.group(1))
            result["memory"]["model_load_time_s"] = _safe_float(m.group(2))

        # --- torch.compile ---
        m = re.search(r"Dynamo bytecode transform time:\s*(\S+)\s*s", line)
        if m:
            result["compilation"]["dynamo_transform_time_s"] = _safe_float(m.group(1))

        m = re.search(r"Compiling a graph .+ takes (\S+)\s*s", line)
        if m:
            result["compilation"]["graph_compile_time_s"] = _safe_float(m.group(1))

        m = re.search(r"torch\.compile (?:takes|took) (\S+)\s*s in total", line)
        if m:
            result["compilation"]["torch_compile_time_s"] = _safe_float(m.group(1))

        # --- KV cache ---
        m = re.search(r"Available KV cache memory:\s*(\S+)\s*GiB", line)
        if m:
            result["memory"]["kv_cache_memory_gib"] = _safe_float(m.group(1))

        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line)
        if m:
            result["memory"]["kv_cache_tokens"] = _safe_int(m.group(1).replace(",", ""))

        m = re.search(r"Maximum concurrency for .+ per request:\s*(\S+)x", line)
        if m:
            result["memory"]["max_concurrency"] = _safe_float(m.group(1))

        # --- CUDA graph capture ---
        m = re.search(r"Graph capturing finished in (\d+)\s*secs?, took (\S+)\s*GiB", line)
        if m:
            result["compilation"]["cuda_graph_capture_time_s"] = _safe_int(m.group(1))
            result["compilation"]["cuda_graph_memory_gib"] = _safe_float(m.group(2))

        m = re.search(r"Initial profiling/warmup run took (\S+)\s*s", line)
        if m:
            result["compilation"]["warmup_time_s"] = _safe_float(m.group(1))

        # --- Engine init ---
        m = re.search(r"init engine .+ took (\S+) seconds", line)
        if m:
            result["timing"]["engine_init_time_s"] = _safe_float(m.group(1))

        # --- TP/PP rank info ---
        m = re.search(r"world_size=(\d+)\s+rank=(\d+)", line)
        if m:
            result["engine_config"]["world_size"] = int(m.group(1))

        m = re.search(r"TP rank (\d+), EP rank (\d+)", line)
        if m:
            result["engine_config"]["tp_rank"] = int(m.group(1))
            result["engine_config"]["ep_rank"] = int(m.group(2))

        # --- Dependency versions (PyTorch, Triton, CUDA, etc.) ---
        m = re.search(r"(?:torch|pytorch)\s+version[:\s]+(\S+)", line, re.IGNORECASE)
        if m:
            result.setdefault("dependencies", {})["torch_version"] = m.group(1).rstrip(",")

        m = re.search(r"triton\s+version[:\s]+(\S+)", line, re.IGNORECASE)
        if m:
            result.setdefault("dependencies", {})["triton_version"] = m.group(1).rstrip(",")

        m = re.search(r"CUDA\s+version[:\s]+(\S+)", line, re.IGNORECASE)
        if m and ("dependencies" not in result or "cuda_version" not in result.get("dependencies", {})):
            result.setdefault("dependencies", {})["cuda_version"] = m.group(1).rstrip(",")

        m = re.search(r"(?:transformers)\s+version[:\s]+(\S+)", line, re.IGNORECASE)
        if m:
            result.setdefault("dependencies", {})["transformers_version"] = m.group(1).rstrip(",")

        m = re.search(r"(?:flash.?attn|flash.?attention)\s+version[:\s]+(\S+)", line, re.IGNORECASE)
        if m:
            result.setdefault("dependencies", {})["flash_attn_version"] = m.group(1).rstrip(",")

        # Catch "Using torch.xyz" or "torch xyz" version patterns common in vLLM logs
        if "torch" in line.lower() and not result.get("dependencies", {}).get("torch_version"):
            m = re.search(r"torch==(\S+)", line)
            if m:
                result.setdefault("dependencies", {})["torch_version"] = m.group(1)

        # --- Warnings and errors ---
        if "WARNING" in line or "ERROR" in line:
            cleaned = line.strip()
            if cleaned and "Loading safetensors" not in cleaned and "Capturing CUDA" not in cleaned:
                result["warnings_errors"].append(cleaned)

    # Remove empty sections
    return {k: v for k, v in result.items() if v}


# ===================================================================== #
#  Dependency version enrichment                                         #
# ===================================================================== #

def _extract_vllm_tag_from_version(version: str) -> Optional[str]:
    """Try to derive a vLLM GitHub tag from a version string.

    Handles formats like "vLLM-0.16.0", "v0.16.0", "0.16.0".
    Returns e.g. "v0.16.0" or None if unparseable.
    """
    v = version.strip()
    for prefix in ("vLLM-", "vllm-", "vLLM ", "vllm "):
        if v.startswith(prefix):
            v = v[len(prefix):]
            break
    v = v.lstrip("v")
    if re.match(r"^\d+\.\d+", v):
        return f"v{v}"
    return None


def _parse_requirements(text: str) -> Dict[str, str]:
    """Parse a requirements.txt-style file into {package: version_spec}."""
    deps: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("==", ">=", "<=", "~=", "!="):
            if sep in line:
                pkg, ver = line.split(sep, 1)
                deps[pkg.strip().lower()] = ver.strip()
                break
    return deps


async def _enrich_dependency_versions(
    parsed: Dict[str, Any],
    version: str,
) -> None:
    """Enrich parsed log data with all pinned dependency versions from GitHub.

    Fetches ``requirements/cuda.txt`` from the vLLM repo at the corresponding
    tag, parses every pinned package, and stores the full set under
    ``parsed["pinned_dependencies"]``.  Any versions already extracted from
    the log text (under ``parsed["dependencies"]``) are preserved and take
    precedence over the requirements file.

    Modifies *parsed* in place.
    """
    tag = _extract_vllm_tag_from_version(version)
    if not tag:
        return

    try:
        from psap_mcp_server.src.tools.kernel_code_mapper_tool import fetch_vllm_source

        result = await fetch_vllm_source("requirements/cuda.txt", tag)
        if result.get("status") != "success" or result.get("type") != "file":
            return

        reqs = _parse_requirements(result.get("content", ""))
        if not reqs:
            return

        pinned = parsed.setdefault("pinned_dependencies", {})
        pinned["_source"] = f"requirements/cuda.txt @ {tag}"
        for pkg, ver in sorted(reqs.items()):
            pinned[pkg] = ver

    except Exception as exc:
        logger.debug(f"Could not enrich dependency versions for {version}: {exc}")


# ===================================================================== #
#  MCP tools                                                             #
# ===================================================================== #

async def fetch_vllm_logs(
    version: str,
    model: Optional[str] = None,
    accelerator: Optional[str] = None,
    tp: Optional[int] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch and parse vLLM server logs for a specific version.

    Retrieves the vLLM log file from S3 for the given model/version/accelerator
    combination and returns a structured summary of the engine configuration,
    compilation settings, memory allocation, and timing information.

    Use this to understand the exact runtime configuration of a benchmark run,
    including quantization backend, attention backend, CUDA graph settings,
    model memory footprint, and compilation times.

    TOOL_NAME=fetch_vllm_logs
    DISPLAY_NAME=Fetch vLLM Logs
    USECASE=Read parsed vLLM server logs to understand the runtime configuration of a benchmark run. Use this alongside profiling tools to get the full picture of how the engine was configured.
    INSTRUCTIONS=1. Specify a version (e.g., "rhaiis-3.3" or "vLLM-0.13.0"), 2. Optionally specify model, accelerator, tp, and profile to disambiguate, 3. Returns structured config + compilation + memory + timing data
    INPUT_DESCRIPTION=version (str): Version; model (str, optional): Model name; accelerator (str, optional): GPU type; tp (int, optional): Tensor parallelism value; profile (str, optional): Workload profile e.g. "1k/1k"
    OUTPUT_DESCRIPTION=Dictionary with parsed log sections: server_config, engine_config, compilation, memory, timing, warnings_errors
    EXAMPLES=fetch_vllm_logs("vLLM-0.17.1", model="gpt-oss-120b", accelerator="H200", tp=4, profile="1k/1k")
    PREREQUISITES=Benchmark CSV must contain a uuid column; log files must exist in S3 at <LOG_S3_BUCKET>/logs/<uuid>.log
    RELATED_TOOLS=compare_vllm_logs, analyze_pytorch_profile, compare_pytorch_profiles

    Args:
        version: Version string (e.g., "rhaiis-3.3", "vLLM-0.13.0").
        model: Model name filter (e.g., "gpt-oss-120b", "deepseek-r1").
        accelerator: Accelerator filter (e.g., "H200", "B200").
        tp: Tensor parallelism filter (e.g., 4).
        profile: Workload profile filter (e.g., "1k/1k", "512/2k").

    Returns:
        Dictionary with parsed log data and metadata.
    """
    try:
        uuid, error = _find_uuid_for_run(version, model, accelerator, tp, profile)
        if error:
            return {"status": "error", "message": error}

        raw, s3_key = _fetch_log_by_uuid(uuid)
        if raw is None:
            return {"status": "error", "message": f"Failed to download log file: {s3_key}"}

        parsed = _parse_vllm_log(raw)

        return {
            "status": "success",
            "uuid": uuid,
            "version": version,
            "model": model,
            "accelerator": accelerator,
            "log_file": f"{uuid}.log",
            "parsed": parsed,
        }

    except Exception as exc:
        logger.error(f"fetch_vllm_logs failed: {exc}")
        return {"status": "error", "message": str(exc)}


async def compare_vllm_logs(
    version1: str,
    version2: str,
    model: Optional[str] = None,
    accelerator: Optional[str] = None,
    tp: Optional[int] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare vLLM server logs between two versions to identify configuration differences.

    Fetches and parses log files for both versions, then produces a side-by-side
    comparison highlighting only the fields that differ. This is essential for
    understanding performance changes that stem from configuration differences
    rather than code changes -- e.g., different quantization backends, attention
    backends, CUDA graph capture sizes, or memory allocation strategies.

    **Automatic dependency enrichment**: This tool automatically fetches
    ``requirements/cuda.txt`` from the vLLM GitHub repo for each version and
    includes ALL pinned dependency versions in the comparison under a
    ``pinned_dependencies`` section. This surfaces PyTorch, Triton, flash-attn,
    xformers, and every other pinned package — even when the server logs don't
    print them.

    TOOL_NAME=compare_vllm_logs
    DISPLAY_NAME=Compare vLLM Logs
    USECASE=Compare engine configurations AND all pinned dependency versions between two vLLM versions. Automatically enriches with every dependency from requirements/cuda.txt. Use this to find config and dependency differences that explain performance changes.
    INSTRUCTIONS=1. Provide two version strings, 2. Optionally specify model, accelerator, tp, and profile, 3. Returns only the fields that differ between versions plus full parsed data for both, 4. Check the 'pinned_dependencies' section for ALL dependency version changes (PyTorch, Triton, flash-attn, xformers, etc.)
    INPUT_DESCRIPTION=version1 (str): Baseline version; version2 (str): Comparison version; model (str, optional): Model name; accelerator (str, optional): GPU type; tp (int, optional): Tensor parallelism; profile (str, optional): Workload profile e.g. "1k/1k"
    OUTPUT_DESCRIPTION=Dictionary with config_differences (only changed fields including all pinned dependency versions), plus full parsed logs for both versions
    EXAMPLES=compare_vllm_logs("vLLM-0.16.0", "vLLM-0.17.1", model="gpt-oss-120b", accelerator="H200", tp=4, profile="1k/1k")
    PREREQUISITES=Benchmark CSV must contain uuid column; log files must exist in S3 for both versions
    RELATED_TOOLS=fetch_vllm_logs, compare_pytorch_profiles, analyze_performance_insights

    Args:
        version1: Baseline version string.
        version2: Comparison version string.
        model: Model name filter.
        accelerator: Accelerator filter.
        tp: Tensor parallelism filter (e.g., 4).
        profile: Workload profile filter (e.g., "1k/1k", "512/2k").

    Returns:
        Dictionary with differences and full parsed logs for both versions.
    """
    try:
        uuid1, err1 = _find_uuid_for_run(version1, model, accelerator, tp, profile)
        if err1:
            return {"status": "error", "message": f"Version 1 ({version1}): {err1}"}

        uuid2, err2 = _find_uuid_for_run(version2, model, accelerator, tp, profile)
        if err2:
            return {"status": "error", "message": f"Version 2 ({version2}): {err2}"}

        raw1, key1 = _fetch_log_by_uuid(uuid1)
        raw2, key2 = _fetch_log_by_uuid(uuid2)
        if raw1 is None:
            return {"status": "error", "message": f"Failed to download log: {key1}"}
        if raw2 is None:
            return {"status": "error", "message": f"Failed to download log: {key2}"}

        parsed1 = _parse_vllm_log(raw1)
        parsed2 = _parse_vllm_log(raw2)

        # Enrich with dependency versions from requirements/cuda.txt if not
        # already extracted from the logs.  This ensures the comparison always
        # surfaces PyTorch/Triton/CUDA version changes even when the log
        # format doesn't print them.
        await _enrich_dependency_versions(parsed1, version1)
        await _enrich_dependency_versions(parsed2, version2)

        # Build diff: only fields that changed
        differences: Dict[str, Dict[str, Any]] = {}
        all_sections = set(list(parsed1.keys()) + list(parsed2.keys()))
        all_sections.discard("warnings_errors")

        for section in sorted(all_sections):
            s1 = parsed1.get(section, {})
            s2 = parsed2.get(section, {})
            if not isinstance(s1, dict) or not isinstance(s2, dict):
                continue
            all_keys = set(list(s1.keys()) + list(s2.keys()))
            section_diff = {}
            for key in sorted(all_keys):
                v1 = s1.get(key)
                v2 = s2.get(key)
                if v1 != v2:
                    entry_diff: Dict[str, Any] = {version1: v1, version2: v2}
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1:
                        entry_diff["change_pct"] = round((v2 - v1) / v1 * 100, 1)
                    section_diff[key] = entry_diff
            if section_diff:
                differences[section] = section_diff

        return {
            "status": "success",
            "model": model,
            "accelerator": accelerator,
            "versions_compared": [version1, version2],
            "uuids": {version1: uuid1, version2: uuid2},
            "config_differences": differences,
            "num_differences": sum(len(v) for v in differences.values()),
            "version1_parsed": parsed1,
            "version2_parsed": parsed2,
        }

    except Exception as exc:
        logger.error(f"compare_vllm_logs failed: {exc}")
        return {"status": "error", "message": str(exc)}
