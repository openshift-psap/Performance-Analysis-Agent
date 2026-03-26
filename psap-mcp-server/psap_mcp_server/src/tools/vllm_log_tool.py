"""MCP tool for fetching and comparing vLLM server logs.

Log files sit alongside PyTorch profiler traces in S3 at
``s3://<bucket>/<PROFILE_S3_PREFIX>/<accelerator>/<model>/<version>/``.
They contain engine configuration, compilation timings, memory allocation,
CUDA graph capture details, and other runtime information that is critical
for understanding performance differences between versions.
"""

import re
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from psap_mcp_server.src.settings import settings
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# ---------------------------------------------------------------------------
# Log index cache  (composite_key -> version -> {source, key/path, filename})
# ---------------------------------------------------------------------------
_log_index_cache: Optional[Dict] = None
_log_index_ts: float = 0.0
_LOG_INDEX_TTL: int = 300  # seconds


# ===================================================================== #
#  Log discovery (S3)                                                    #
# ===================================================================== #

def _discover_logs_s3() -> Optional[Dict]:
    """Discover vLLM log files from S3.

    Walks the same ``s3://<bucket>/<prefix>/<accelerator>/<model>/<version>/``
    hierarchy used for profiler traces and indexes ``.txt`` and ``.log`` files.

    Returns::

        {"accelerator/model": {version: {"source": "s3", "key": ..., "filename": ...}}}

    Only the first matching log file per version is stored (there should be one).
    Returns ``None`` if S3 is not configured or unreachable.
    """
    bucket = settings.S3_BUCKET
    prefix = getattr(settings, "PROFILE_S3_PREFIX", "profiles/rhaiis")
    if not bucket:
        return None

    if not prefix.endswith("/"):
        prefix += "/"

    try:
        from psap_mcp_server.src.tools.s3_utils import get_s3_client
        s3 = get_s3_client()
    except ImportError:
        logger.warning("boto3 not installed -- S3 log discovery unavailable")
        return None
    except Exception as exc:
        logger.warning(f"Failed to create S3 client: {exc}")
        return None

    index: Dict[str, Dict[str, Dict]] = {}

    def _list_prefixes(parent_prefix: str) -> List[str]:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=parent_prefix, Delimiter="/")
        return [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]

    try:
        accel_prefixes = _list_prefixes(prefix)
        for accel_prefix in accel_prefixes:
            accelerator = accel_prefix.rstrip("/").split("/")[-1]
            model_prefixes = _list_prefixes(accel_prefix)

            for model_prefix in model_prefixes:
                model_name = model_prefix.rstrip("/").split("/")[-1]
                composite_key = f"{accelerator}/{model_name}"
                version_prefixes = _list_prefixes(model_prefix)

                for version_prefix in version_prefixes:
                    version_name = version_prefix.rstrip("/").split("/")[-1]
                    resp = s3.list_objects_v2(Bucket=bucket, Prefix=version_prefix)
                    for obj in resp.get("Contents", []):
                        key = obj["Key"]
                        filename = key.split("/")[-1]
                        if filename.endswith((".txt", ".log")) and "log" in filename.lower():
                            index.setdefault(composite_key, {})[version_name] = {
                                "source": "s3",
                                "key": key,
                                "filename": filename,
                                "size_bytes": obj.get("Size", 0),
                            }
                            break  # one log per version

        if index:
            summary = ", ".join(f"{m} ({len(vs)} version(s))" for m, vs in index.items())
            logger.info(f"S3 log discovery: {summary}")
        else:
            logger.info(f"S3 log discovery: no log files found under s3://{bucket}/{prefix}")

        return index

    except Exception as exc:
        logger.warning(f"S3 log discovery failed: {exc}")
        return None


def _discover_logs(force_refresh: bool = False) -> Dict:
    """Discover available log files with in-memory caching."""
    global _log_index_cache, _log_index_ts

    now = _time.time()
    if (
        not force_refresh
        and _log_index_cache is not None
        and (now - _log_index_ts) < _LOG_INDEX_TTL
    ):
        return _log_index_cache

    index = _discover_logs_s3()
    if not index:
        index = {}

    _log_index_cache = index
    _log_index_ts = now
    return index


# ===================================================================== #
#  Model / version matching (reuse logic from pytorch_profile_tool)      #
# ===================================================================== #

def _bare_model(composite_key: str) -> str:
    return composite_key.split("/", 1)[1] if "/" in composite_key else composite_key


def _key_accelerator(composite_key: str) -> str:
    return composite_key.split("/", 1)[0] if "/" in composite_key else "unknown"


def _match_model_keys(
    user_model: Optional[str],
    user_accelerator: Optional[str],
    available_keys: List[str],
) -> List[str]:
    """Match user-provided model/accelerator against composite keys."""
    if not available_keys:
        return []

    matches = available_keys

    if user_accelerator:
        accel_lower = user_accelerator.lower().strip()
        matches = [k for k in matches if _key_accelerator(k).lower() == accel_lower]

    if user_model:
        model_lower = user_model.lower().strip()
        filtered = []
        for k in matches:
            bare = _bare_model(k).lower()
            if model_lower == bare or model_lower in bare or bare in model_lower:
                filtered.append(k)
        matches = filtered

    return matches


def _match_version(user_version: str, available_versions: List[str]) -> Optional[str]:
    """Fuzzy-match a version string against available folder names."""
    user_lower = user_version.lower().strip()
    for v in available_versions:
        if v.lower() == user_lower:
            return v
    # Try stripping common prefixes for bare numeric comparison
    def _bare(ver: str) -> str:
        ver = ver.strip().lower()
        for pfx in ("vllm-", "vllm ", "rhaiis-", "rhaiis "):
            if ver.startswith(pfx):
                ver = ver[len(pfx):]
                break
        return ver.lstrip("v")

    user_bare = _bare(user_version)
    for v in available_versions:
        if _bare(v) == user_bare:
            return v
    return None


# ===================================================================== #
#  Log fetching                                                          #
# ===================================================================== #

def _fetch_log_from_s3(s3_key: str) -> Optional[str]:
    """Download a log file from S3 and return its contents as a string."""
    bucket = settings.S3_BUCKET
    if not bucket:
        return None
    try:
        from psap_mcp_server.src.tools.s3_utils import get_s3_client
        s3 = get_s3_client()
        logger.info(f"Downloading log from S3: s3://{bucket}/{s3_key}")
        response = s3.get_object(Bucket=bucket, Key=s3_key)
        return response["Body"].read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Failed to load log {s3_key}: {exc}")
        return None


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


def _parse_vllm_log(raw: str) -> Dict[str, Any]:
    """Parse a vLLM server log into structured sections.

    Extracts configuration, compilation, memory, and timing information
    using regex patterns matched against known vLLM log output formats.
    """
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

        m = re.search(r"torch\.compile takes (\S+)\s*s in total", line)
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

        # --- Warnings and errors ---
        if "WARNING" in line or "ERROR" in line:
            cleaned = line.strip()
            # Skip noisy progress bars and routine log lines
            if cleaned and "Loading safetensors" not in cleaned and "Capturing CUDA" not in cleaned:
                result["warnings_errors"].append(cleaned)

    # Remove empty sections
    return {k: v for k, v in result.items() if v}


# ===================================================================== #
#  Resolve helpers                                                       #
# ===================================================================== #

def _resolve_model_key(
    model: Optional[str],
    accelerator: Optional[str],
    index: Dict,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve user inputs to a single composite key, or return an error message."""
    available = sorted(index.keys())
    if not available:
        return None, "No vLLM log files found in S3."

    matches = _match_model_keys(model, accelerator, available)
    if not matches:
        return None, f"No logs found for model='{model}', accelerator='{accelerator}'. Available: {available}"
    if len(matches) > 1:
        return None, (
            f"Ambiguous: multiple matches found: {matches}. "
            f"Please specify both model and accelerator to disambiguate."
        )
    return matches[0], None


# ===================================================================== #
#  MCP tools                                                             #
# ===================================================================== #

async def fetch_vllm_logs(
    version: str,
    model: Optional[str] = None,
    accelerator: Optional[str] = None,
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
    INSTRUCTIONS=1. Specify a version (e.g., "rhaiis-3.3" or "vLLM-0.13.0"), 2. Optionally specify model and accelerator to disambiguate, 3. Returns structured config + compilation + memory + timing data
    INPUT_DESCRIPTION=version (str): Version folder name; model (str, optional): Model name; accelerator (str, optional): GPU type (e.g., "H200")
    OUTPUT_DESCRIPTION=Dictionary with parsed log sections: server_config, engine_config, compilation, memory, timing, warnings_errors
    EXAMPLES=fetch_vllm_logs("rhaiis-3.3", model="gpt-oss-120b", accelerator="H200")
    PREREQUISITES=Log files must be uploaded to S3 alongside profiler traces
    RELATED_TOOLS=compare_vllm_logs, analyze_pytorch_profile, compare_pytorch_profiles

    Args:
        version: Version folder name (e.g., "rhaiis-3.3", "vLLM-0.13.0").
        model: Model name filter (e.g., "gpt-oss-120b", "deepseek-r1").
        accelerator: Accelerator filter (e.g., "H200", "B200").

    Returns:
        Dictionary with parsed log data and metadata.
    """
    try:
        index = _discover_logs()
        model_key, error = _resolve_model_key(model, accelerator, index)
        if error:
            return {"status": "error", "message": error}

        versions = sorted(index[model_key].keys())
        matched_version = _match_version(version, versions)
        if matched_version is None:
            return {
                "status": "error",
                "message": f"Version '{version}' not found for {model_key}. Available: {versions}",
            }

        entry = index[model_key][matched_version]
        raw = _fetch_log_from_s3(entry["key"])
        if raw is None:
            return {"status": "error", "message": f"Failed to download log file: {entry['key']}"}

        parsed = _parse_vllm_log(raw)

        return {
            "status": "success",
            "model": model_key,
            "version": matched_version,
            "log_file": entry["filename"],
            "log_size_bytes": entry.get("size_bytes", 0),
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
) -> Dict[str, Any]:
    """Compare vLLM server logs between two versions to identify configuration differences.

    Fetches and parses log files for both versions, then produces a side-by-side
    comparison highlighting only the fields that differ. This is essential for
    understanding performance changes that stem from configuration differences
    rather than code changes -- e.g., different quantization backends, attention
    backends, CUDA graph capture sizes, or memory allocation strategies.

    TOOL_NAME=compare_vllm_logs
    DISPLAY_NAME=Compare vLLM Logs
    USECASE=Compare engine configurations between two vLLM versions. Use this to find config differences (quantization, attention backend, CUDA graphs, memory) that explain performance changes not visible in kernel profiling.
    INSTRUCTIONS=1. Provide two version strings, 2. Optionally specify model and accelerator, 3. Returns only the fields that differ between versions plus full parsed data for both
    INPUT_DESCRIPTION=version1 (str): Baseline version; version2 (str): Comparison version; model (str, optional): Model name; accelerator (str, optional): GPU type
    OUTPUT_DESCRIPTION=Dictionary with config_differences (only changed fields), plus full parsed logs for both versions
    EXAMPLES=compare_vllm_logs("rhaiis-3.2.5", "rhaiis-3.3", model="gpt-oss-120b", accelerator="H200")
    PREREQUISITES=Log files must exist for both versions in S3
    RELATED_TOOLS=fetch_vllm_logs, compare_pytorch_profiles, analyze_performance_insights

    Args:
        version1: Baseline version folder name.
        version2: Comparison version folder name.
        model: Model name filter.
        accelerator: Accelerator filter.

    Returns:
        Dictionary with differences and full parsed logs for both versions.
    """
    try:
        index = _discover_logs()
        model_key, error = _resolve_model_key(model, accelerator, index)
        if error:
            return {"status": "error", "message": error}

        versions = sorted(index[model_key].keys())
        mv1 = _match_version(version1, versions)
        mv2 = _match_version(version2, versions)
        if mv1 is None:
            return {"status": "error", "message": f"Version '{version1}' not found for {model_key}. Available: {versions}"}
        if mv2 is None:
            return {"status": "error", "message": f"Version '{version2}' not found for {model_key}. Available: {versions}"}

        entry1 = index[model_key][mv1]
        entry2 = index[model_key][mv2]

        raw1 = _fetch_log_from_s3(entry1["key"])
        raw2 = _fetch_log_from_s3(entry2["key"])
        if raw1 is None:
            return {"status": "error", "message": f"Failed to download log: {entry1['key']}"}
        if raw2 is None:
            return {"status": "error", "message": f"Failed to download log: {entry2['key']}"}

        parsed1 = _parse_vllm_log(raw1)
        parsed2 = _parse_vllm_log(raw2)

        # Build diff: only fields that changed
        differences: Dict[str, Dict[str, Any]] = {}
        all_sections = set(list(parsed1.keys()) + list(parsed2.keys()))
        # Skip warnings_errors from diff (list comparison is noisy)
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
                    entry_diff: Dict[str, Any] = {mv1: v1, mv2: v2}
                    # Add delta for numeric values
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1:
                        entry_diff["change_pct"] = round((v2 - v1) / v1 * 100, 1)
                    section_diff[key] = entry_diff
            if section_diff:
                differences[section] = section_diff

        return {
            "status": "success",
            "model": model_key,
            "versions_compared": [mv1, mv2],
            "config_differences": differences,
            "num_differences": sum(len(v) for v in differences.values()),
            "version1_parsed": parsed1,
            "version2_parsed": parsed2,
        }

    except Exception as exc:
        logger.error(f"compare_vllm_logs failed: {exc}")
        return {"status": "error", "message": str(exc)}
