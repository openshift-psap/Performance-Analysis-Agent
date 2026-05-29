"""MCP tool for analyzing NVIDIA Nsight Systems (NYSY) profiler traces from vLLM benchmark runs.

This tool complements the PyTorch profiler tool by providing GPU-level profiling with
nanosecond precision. NYSY files are binary (.nsys-rep) and must be exported to JSON first.

CRITICAL: NYSY uses nanoseconds; internally we store as microseconds.
Always divide nanoseconds by 1000, never multiply!

Dynamically discovers NYSY profile traces from either S3
(``s3://<bucket>/<NYSY_PROFILE_S3_PREFIX>/<accelerator>/<model>/<version>/``)
or a local directory fallback, extracts kernel statistics with GPU metrics,
and enables comparison between different vLLM versions.
"""

import json
import os
import re
import statistics
import subprocess
import tempfile
import time as _time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from psap_mcp_server.utils.pylogger import get_python_logger
from psap_mcp_server.src.settings import settings

logger = get_python_logger()

# Known model metadata (shared with pytorch_profile_tool)
KNOWN_MODELS: Dict[str, Dict[str, str]] = {
    "deepseek-r1": {
        "display_name": "DeepSeek-R1",
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
    },
    "gpt-oss": {
        "display_name": "GPT-OSS",
        "model_id": "gpt-oss",
    },
    "gpt-oss-120b": {
        "display_name": "GPT-OSS-120B",
        "model_id": "openai/gpt-oss-120b",
    },
    "nemotron-120b": {
        "display_name": "Nemotron-120B",
        "model_id": "nvidia/Nemotron-4-340B-Instruct",
    },
}

# Model aliases
_MODEL_ALIASES: Dict[str, str] = {
    "deepseek": "deepseek-r1",
    "deepseek-r1-0528": "deepseek-r1",
    "gpt-oss": "gpt-oss-120b",
    "gptoss": "gpt-oss-120b",
    "gpt_oss": "gpt-oss-120b",
    "gpt-oss-120b": "gpt-oss-120b",
    "gptoss120b": "gpt-oss-120b",
    "gpt_oss_120b": "gpt-oss-120b",
    "nemotron": "nemotron-120b",
    "nemotron-120b": "nemotron-120b",
}

# Local fallback directory
_DEFAULT_LOCAL_NYSY_BASE = "/app/nysy-profiles"
LOCAL_NYSY_BASE: str = (
    getattr(settings, "NYSY_PROFILE_BASE", None)
    or os.environ.get("NYSY_PROFILE_BASE", _DEFAULT_LOCAL_NYSY_BASE)
)

# Category groups (shared with pytorch_profile_tool)
CATEGORY_GROUPS: Dict[str, Optional[List[str]]] = {
    "kernel": ["kernel"],
    "cpu": ["cpu_op"],
    "cuda": ["cuda_runtime", "cuda_driver"],
    "communication": ["nccl", "c10d"],
    "memory": ["cuda_memory"],
    "all": None,
}

# Functional pipelines (shared with pytorch_profile_tool)
FUNCTIONAL_PIPELINES: Dict[str, Dict[str, Any]] = {
    "moe_execution": {
        "patterns": ["moe", "expert", "scatter", "gather", "fused_moe", "deep_gemm", "outplace_fused"],
        "description": "MoE routing + expert GEMM execution pipeline",
    },
    "attention": {
        "patterns": ["attention", "flash", "mla", "paged", "kv_cache", "reshape_and_cache"],
        "description": "Attention mechanism (MLA / MHA / GQA / Flash / Paged)",
    },
    "communication": {
        "patterns": ["nccl", "allreduce", "allgather", "multimem", "c10d", "all_reduce"],
        "description": "Inter-GPU collective communication",
    },
    "quantization": {
        "patterns": ["quant", "fp8", "int8", "dequant", "scale", "per_token_group"],
        "description": "Quantization / dequantization / scaling operations",
    },
    "normalization_activation": {
        "patterns": ["layernorm", "rmsnorm", "rms_norm", "silu", "gelu", "activation", "act_and_mul"],
        "description": "Normalization (RMSNorm/LayerNorm) and activation functions",
    },
    "gemm_linear": {
        "patterns": ["gemm", "matmul", "aten::mm", "aten::bmm", "aten::linear", "cublas", "cutlass"],
        "description": "Standalone matrix multiplication / linear layers (not MoE)",
    },
}

# NYSY event type to category mapping
NYSY_CATEGORY_MAPPING: Dict[str, str] = {
    "kernel": "kernel",
    "cuda_runtime": "cuda_runtime",
    "cuda_driver": "cuda_driver",
    "nccl": "nccl",
    "c10d": "c10d",
    "memory": "cuda_memory",
    "cpu": "cpu_op",
}

# Profile index cache
_nysy_index_cache: Optional[Dict] = None
_nysy_index_ts: float = 0.0
_NYSY_INDEX_TTL: int = 300

# Stats cache
_nysy_stats_cache: Dict[str, Tuple[float, Dict]] = {}
_NYSY_STATS_CACHE_TTL: int = 300


# ===================================================================== #
#  Helper functions                                                      #
# ===================================================================== #

def _parse_rank_from_filename(filename: str) -> Optional[int]:
    """Extract rank number from filename. See pytorch_profile_tool for logic."""
    match = re.search(r"rank(\d+)", filename)
    if match:
        return int(match.group(1))

    parts = filename.replace(".nsys-rep", "").replace(".json", "").split("_")
    if len(parts) >= 4 and parts[0] in ("trace", "nsys"):
        try:
            return int(parts[3])
        except (ValueError, IndexError):
            pass

    return None


def _extract_bare_version(version: str) -> str:
    """Extract bare numeric version. Shared with pytorch_profile_tool."""
    version = version.strip()
    for prefix in ("vllm-", "vllm ", "vLLM-", "vLLM "):
        if version.startswith(prefix):
            version = version[len(prefix):]
            break
    version = version.lstrip("v")
    return version


def _normalize_version(version: str) -> str:
    """Normalise to vLLM-X.Y.Z format."""
    bare = _extract_bare_version(version)
    return f"vLLM-{bare}"


def _match_version(user_version: str, available_versions: List[str]) -> Optional[str]:
    """Fuzzy-match version. Shared with pytorch_profile_tool."""
    user_bare = _extract_bare_version(user_version)
    for v in available_versions:
        if _extract_bare_version(v) == user_bare:
            return v
    return None


def _bare_model(composite_key: str) -> str:
    """Extract bare model name from composite key."""
    return composite_key.split("/", 1)[1] if "/" in composite_key else composite_key


def _key_accelerator(composite_key: str) -> str:
    """Extract accelerator from composite key."""
    return composite_key.split("/", 1)[0] if "/" in composite_key else "unknown"


def _match_all_models(user_model: Optional[str], available_models: List[str]) -> List[str]:
    """Fuzzy-match models. Shared logic with pytorch_profile_tool."""
    if not available_models:
        return []
    if user_model is None:
        return available_models

    lower = user_model.lower().strip()

    for key in available_models:
        if key.lower() == lower:
            return [key]

    alias = _MODEL_ALIASES.get(lower)
    if alias:
        alias_lower = alias.lower()
        matches = [k for k in available_models if _bare_model(k).lower() == alias_lower]
        if matches:
            return matches

    matches = []
    for key in available_models:
        bare = _bare_model(key).lower()
        if lower in key.lower() or lower in bare or bare in lower:
            matches.append(key)
            continue
        info = KNOWN_MODELS.get(bare, {})
        if lower in info.get("display_name", "").lower():
            matches.append(key)
            continue
        if lower in info.get("model_id", "").lower():
            matches.append(key)

    return matches


def _match_model(user_model: Optional[str], available_models: List[str]) -> Optional[str]:
    """Convenience wrapper for single match."""
    matches = _match_all_models(user_model, available_models)
    return matches[0] if matches else None


def _get_display_name(model_key: str) -> str:
    """Human-friendly display name."""
    bare = _bare_model(model_key)
    name = KNOWN_MODELS.get(bare, {}).get("display_name", bare)
    accel = _key_accelerator(model_key)
    return f"{name} ({accel})" if accel != "unknown" else name


def _get_model_info(model_key: str, num_ranks: int) -> Dict[str, Any]:
    """Build model info dict."""
    bare = _bare_model(model_key)
    accel = _key_accelerator(model_key)
    known = KNOWN_MODELS.get(bare, {})
    return {
        "display_name": known.get("display_name", bare),
        "model_id": known.get("model_id", bare),
        "gpus": accel,
        "ranks_stored": num_ranks,
        "note": (
            "ranks_stored is the number of trace files available, not the "
            "tensor-parallelism used in the benchmark. Only rank-0 traces "
            "may be stored for multi-GPU runs."
        ),
    }


# ===================================================================== #
#  Profile discovery (S3 + local)                                        #
# ===================================================================== #

def _discover_nysy_profiles_s3() -> Optional[Dict]:
    """Discover NYSY profiles from S3 with .nsys-rep files."""
    bucket = settings.S3_BUCKET
    prefix = getattr(settings, "NYSY_PROFILE_S3_PREFIX", "profiles/nsys")
    if not bucket:
        return None

    if not prefix.endswith("/"):
        prefix += "/"

    try:
        from psap_mcp_server.src.tools.s3_utils import get_s3_client
        s3 = get_s3_client()
    except ImportError:
        logger.warning("boto3 not installed -- S3 NYSY discovery unavailable")
        return None
    except Exception as exc:
        logger.warning(f"Failed to create S3 client: {exc}")
        return None

    index: Dict[str, Dict[str, Dict[int, Dict]]] = {}

    def _list_prefixes(parent_prefix: str) -> List[str]:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=parent_prefix, Delimiter="/")
        return [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]

    def _scan_version_traces(version_prefix: str, composite_key: str) -> None:
        """List .nsys-rep files under version prefix."""
        paginator_token = None
        auto_rank = 0
        while True:
            kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": version_prefix}
            if paginator_token:
                kwargs["ContinuationToken"] = paginator_token

            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                if not filename.endswith(".nsys-rep"):
                    continue

                rank = _parse_rank_from_filename(filename)
                if rank is None:
                    rank = auto_rank
                    auto_rank += 1

                version_dict = index.setdefault(composite_key, {}).setdefault(
                    version_prefix.rstrip("/").split("/")[-1], {}
                )
                while rank in version_dict:
                    rank = auto_rank
                    auto_rank += 1

                version_dict[rank] = {
                    "source": "s3",
                    "key": key,
                    "filename": filename,
                    "size_bytes": obj.get("Size", 0),
                }

            if resp.get("IsTruncated"):
                paginator_token = resp.get("NextContinuationToken")
            else:
                break

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
                    _scan_version_traces(version_prefix, composite_key)

        if index:
            summary = ", ".join(f"{m} ({len(vs)} version(s))" for m, vs in index.items())
            logger.info(f"S3 NYSY discovery: {summary}")
        else:
            logger.info(f"S3 NYSY discovery: no profiles under s3://{bucket}/{prefix}")

        return index

    except Exception as exc:
        logger.warning(f"S3 NYSY discovery failed: {exc}")
        return None


def _discover_nysy_profiles_local(base_dir: Optional[str] = None) -> Dict:
    """Discover NYSY profiles from local filesystem."""
    base = Path(base_dir or LOCAL_NYSY_BASE)
    index: Dict[str, Dict[str, Dict[int, Dict]]] = {}

    if not base.exists():
        logger.info(f"Local NYSY base not found: {base}")
        return index

    for accel_dir in sorted(base.iterdir()):
        if not accel_dir.is_dir() or accel_dir.name.startswith("."):
            continue
        accelerator = accel_dir.name

        for model_dir in sorted(accel_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            model_name = model_dir.name
            composite_key = f"{accelerator}/{model_name}"

            for version_dir in sorted(model_dir.iterdir()):
                if not version_dir.is_dir() or version_dir.name.startswith("."):
                    continue
                version_name = version_dir.name
                auto_rank = 0

                for f in sorted(version_dir.rglob("*.nsys-rep")):
                    if not f.is_file():
                        continue
                    rank = _parse_rank_from_filename(f.name)
                    if rank is None:
                        rank = auto_rank
                        auto_rank += 1

                    version_dict = index.setdefault(composite_key, {}).setdefault(
                        version_name, {}
                    )
                    while rank in version_dict:
                        rank = auto_rank
                        auto_rank += 1

                    version_dict[rank] = {
                        "source": "local",
                        "path": f,
                        "filename": f.name,
                        "size_bytes": f.stat().st_size,
                    }

    if index:
        summary = ", ".join(f"{m} ({len(vs)} version(s))" for m, vs in index.items())
        logger.info(f"Local NYSY discovery: {summary}")
    else:
        logger.info(f"Local NYSY discovery: no profiles under {base}")

    return index


def _discover_nysy_profiles(force_refresh: bool = False) -> Dict:
    """Discover NYSY profiles from S3, with caching."""
    global _nysy_index_cache, _nysy_index_ts

    now = _time.time()
    if (
        not force_refresh
        and _nysy_index_cache is not None
        and (now - _nysy_index_ts) < _NYSY_INDEX_TTL
    ):
        return _nysy_index_cache

    index = _discover_nysy_profiles_s3()

    if index:
        logger.info(f"NYSY index built from S3: {len(index)} model(s)")
    else:
        logger.warning(
            "No NYSY data found in S3. "
            "Upload .nsys-rep files to s3://<bucket>/<NYSY_PROFILE_S3_PREFIX>/<accelerator>/<model>/<version>/"
        )
        index = {}

    _nysy_index_cache = index
    _nysy_index_ts = now
    return index


# ===================================================================== #
#  NYSY trace parsing (CRITICAL: nanosecond conversion)                  #
# ===================================================================== #

def _export_nsys_to_json(nsys_file_path: str) -> Optional[Dict]:
    """Export NYSY binary to JSON using nsys export CLI.

    CRITICAL: This requires nsys CLI tool to be installed.
    Returns None if nsys is not available or export fails.
    """
    try:
        # Create temp file for JSON output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json_path = tmp.name

        # Run nsys export
        cmd = [
            "nsys", "export",
            "--type", "qdstrm_sql",
            "--output", json_path,
            nsys_file_path
        ]
        logger.info(f"Exporting NYSY: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"nsys export failed: {result.stderr}")
            return None

        # Read exported JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Clean up temp file
        try:
            os.unlink(json_path)
        except Exception:
            pass

        return data

    except FileNotFoundError:
        logger.error("nsys CLI tool not found. Install NVIDIA Nsight Systems.")
        return None
    except subprocess.TimeoutExpired:
        logger.error("nsys export timed out (>60s)")
        return None
    except Exception as exc:
        logger.error(f"Failed to export NYSY: {exc}")
        return None


def _parse_nysy_trace(trace_data: Any) -> Optional[Dict]:
    """Parse NYSY trace (JSON or binary .nsys-rep).

    If binary, attempts to export via nsys CLI.
    Returns extracted stats dict or None on error.
    """
    # If it's already parsed JSON, use it directly
    if isinstance(trace_data, dict):
        return trace_data

    # If it's a file path (string), try to load/export
    if isinstance(trace_data, str):
        if trace_data.endswith(".nsys-rep"):
            return _export_nsys_to_json(trace_data)
        else:
            # Assume it's JSON
            try:
                with open(trace_data, 'r') as f:
                    return json.load(f)
            except Exception as exc:
                logger.error(f"Failed to load NYSY JSON: {exc}")
                return None

    return None


def _extract_nysy_stats(events: List[Any]) -> Dict[str, Dict]:
    """Extract kernel statistics from NYSY events.

    CRITICAL: Convert nanoseconds to microseconds (divide by 1000, not multiply!)
    """
    stats: Dict[str, Dict] = defaultdict(
        lambda: {"count": 0, "total_dur": 0, "min_dur": float("inf"), "max_dur": 0, "cat": ""}
    )

    for event in events:
        kernel_name = event.get("name", "unknown")
        duration_ns = event.get("duration_ns", 0)

        # CRITICAL: Convert nanoseconds to microseconds (divide by 1000)
        duration_us = duration_ns / 1000

        if duration_us > 0:
            stats[kernel_name]["count"] += 1
            stats[kernel_name]["total_dur"] += duration_us
            stats[kernel_name]["min_dur"] = min(stats[kernel_name]["min_dur"], duration_us)
            stats[kernel_name]["max_dur"] = max(stats[kernel_name]["max_dur"], duration_us)

            # Map NYSY category to canonical category
            nysy_cat = event.get("type", "")
            canonical_cat = NYSY_CATEGORY_MAPPING.get(nysy_cat, nysy_cat)
            if not stats[kernel_name]["cat"]:
                stats[kernel_name]["cat"] = canonical_cat

    # Calculate averages and fix infinities
    for data in stats.values():
        data["avg_dur"] = data["total_dur"] / data["count"] if data["count"] > 0 else 0
        if data["min_dur"] == float("inf"):
            data["min_dur"] = 0

    return dict(stats)


# ===================================================================== #
#  Trace loading (S3 + local)                                           #
# ===================================================================== #

def _load_nysy_trace_from_s3(s3_key: str) -> Optional[dict]:
    """Download and parse NYSY trace from S3."""
    bucket = settings.S3_BUCKET
    if not bucket:
        return None

    try:
        from psap_mcp_server.src.tools.s3_utils import get_s3_client

        s3 = get_s3_client()
        logger.info(f"Downloading NYSY from S3: s3://{bucket}/{s3_key}")

        # Download to temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.nsys-rep', delete=False) as tmp:
            tmp_path = tmp.name
            response = s3.get_object(Bucket=bucket, Key=s3_key)
            tmp.write(response["Body"].read())

        # Export to JSON
        data = _export_nsys_to_json(tmp_path)

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return data

    except Exception as exc:
        logger.error(f"Failed to load NYSY from S3 {s3_key}: {exc}")
        return None


def _load_nysy_trace_from_local(filepath: Path) -> Optional[dict]:
    """Load NYSY trace from local filesystem."""
    if not filepath.exists():
        logger.warning(f"NYSY file not found: {filepath}")
        return None

    logger.info(f"Loading local NYSY: {filepath.name}")
    try:
        if filepath.suffix == ".nsys-rep":
            return _export_nsys_to_json(str(filepath))
        else:
            # Assume JSON
            with open(filepath, "r") as fh:
                return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error(f"Corrupted JSON in {filepath.name}: {exc}")
        return None
    except Exception as exc:
        logger.error(f"Failed to load {filepath.name}: {exc}")
        return None


def _load_or_extract_nysy_stats(
    model: str,
    version: str,
    rank: int,
    force_reload: bool = False,
) -> Optional[Dict[str, Dict]]:
    """Load and extract NYSY stats for model/version/rank.

    Checks cache, looks up in index, loads trace, extracts stats.
    """
    cache_key = f"{model}/{version}/rank{rank}"

    if not force_reload:
        cached = _get_cached_nysy_stats(cache_key)
        if cached is not None:
            logger.debug(f"NYSY stats cache hit: {cache_key}")
            return cached

    # Look up in profile index
    index = _discover_nysy_profiles()
    entry = index.get(model, {}).get(version, {}).get(rank)

    if entry is None:
        logger.warning(f"No NYSY trace found for {cache_key}")
        return None

    # Load the trace
    if entry["source"] == "s3":
        trace_json = _load_nysy_trace_from_s3(entry["key"])
    else:
        trace_json = _load_nysy_trace_from_local(entry["path"])

    if not trace_json:
        logger.warning(f"Failed to load NYSY trace for {cache_key}")
        return None

    # Extract events and stats
    events = trace_json.get("events", [])
    if not events:
        logger.warning(f"No events in NYSY trace for {cache_key}")
        return None

    stats = _extract_nysy_stats(events)
    _put_cached_nysy_stats(cache_key, stats)
    return stats


def _get_cached_nysy_stats(cache_key: str) -> Optional[Dict[str, Dict]]:
    """Get stats from cache if valid."""
    if cache_key in _nysy_stats_cache:
        ts, stats = _nysy_stats_cache[cache_key]
        if (_time.time() - ts) < _NYSY_STATS_CACHE_TTL:
            return stats
        del _nysy_stats_cache[cache_key]
    return None


def _put_cached_nysy_stats(cache_key: str, stats: Dict[str, Dict]) -> None:
    """Store stats in cache."""
    _nysy_stats_cache[cache_key] = (_time.time(), stats)


# ===================================================================== #
#  Shared utility functions (from pytorch_profile_tool)                  #
# ===================================================================== #

def _merge_stats(stats_list: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """Merge statistics from multiple ranks."""
    merged: Dict[str, Dict] = defaultdict(
        lambda: {"count": 0, "total_dur": 0, "min_dur": float("inf"), "max_dur": 0, "cat": ""}
    )

    for stats in stats_list:
        for name, data in stats.items():
            merged[name]["count"] += data["count"]
            merged[name]["total_dur"] += data["total_dur"]
            merged[name]["min_dur"] = min(
                merged[name]["min_dur"], data.get("min_dur", float("inf"))
            )
            merged[name]["max_dur"] = max(
                merged[name]["max_dur"], data.get("max_dur", 0)
            )
            if not merged[name]["cat"]:
                merged[name]["cat"] = data.get("cat", "")

    for data in merged.values():
        data["avg_dur"] = data["total_dur"] / data["count"] if data["count"] > 0 else 0
        if data["min_dur"] == float("inf"):
            data["min_dur"] = 0

    return dict(merged)


def _format_duration(us: float) -> str:
    """Format duration in microseconds to human readable."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    elif us >= 1_000:
        return f"{us / 1_000:.2f}ms"
    else:
        return f"{us:.2f}µs"


def _filter_by_category(stats: Dict[str, Dict], category: Optional[str]) -> Dict[str, Dict]:
    """Filter stats by category."""
    if category is None or category == "all":
        return stats

    categories = CATEGORY_GROUPS.get(category.lower())
    if categories is None:
        return stats

    lower_cats = [c.lower() for c in categories]
    return {
        name: data
        for name, data in stats.items()
        if data.get("cat", "").lower() in lower_cats
    }


def _get_top_kernels(stats: Dict[str, Dict], n: int = 30, sort_by: str = "total_dur") -> List[Dict]:
    """Get top N kernels sorted by metric."""
    sorted_kernels = sorted(stats.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

    result = []
    for name, data in sorted_kernels[:n]:
        result.append(
            {
                "name": name,
                "category": data.get("cat", "unknown"),
                "count": data["count"],
                "total_dur_us": data["total_dur"],
                "total_dur_human": _format_duration(data["total_dur"]),
                "avg_dur_us": data.get("avg_dur", 0),
                "avg_dur_human": _format_duration(data.get("avg_dur", 0)),
                "min_dur_us": data.get("min_dur", 0),
                "max_dur_us": data.get("max_dur", 0),
            }
        )

    return result


def _get_category_breakdown(stats: Dict[str, Dict]) -> Dict[str, Dict]:
    """Get time breakdown by category."""
    cat_totals: Dict[str, Dict] = defaultdict(
        lambda: {"total_dur": 0, "count": 0, "kernel_count": 0}
    )

    for _name, data in stats.items():
        cat = data.get("cat", "other") or "other"
        cat_totals[cat]["total_dur"] += data["total_dur"]
        cat_totals[cat]["count"] += data["count"]
        cat_totals[cat]["kernel_count"] += 1

    result = {}
    for cat, totals in sorted(cat_totals.items(), key=lambda x: x[1]["total_dur"], reverse=True):
        result[cat] = {
            "total_dur_us": totals["total_dur"],
            "total_dur_human": _format_duration(totals["total_dur"]),
            "invocation_count": totals["count"],
            "unique_operations": totals["kernel_count"],
        }

    return result


def _resolve_model_and_version(
    model: Optional[str],
    version: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Dict]:
    """Resolve user-provided model/version to discovered folder names."""
    index = _discover_nysy_profiles()

    if not index:
        return None, None, "No NYSY profile data available. Check S3 or local profile directory.", index

    available_models = sorted(index.keys())
    matches = _match_all_models(model, available_models)

    if not matches:
        return None, None, f"Model '{model}' not found. Available: {available_models}", index

    if len(matches) > 1:
        formatted = ", ".join(matches)
        return None, None, (
            f"Multiple models match '{model}': {formatted}. "
            f"Please specify accelerator, e.g. '{matches[0]}'"
        ), index

    model_key = matches[0]
    available_versions = sorted(index[model_key].keys())
    matched_version = _match_version(version, available_versions)

    if matched_version is None:
        return model_key, None, (
            f"Version '{version}' not found for {_get_display_name(model_key)}. "
            f"Available: {available_versions}"
        ), index

    return model_key, matched_version, None, index


def _get_available_ranks(index: Dict, model_key: str, version: str) -> List[int]:
    """Get sorted list of available ranks."""
    return sorted(index.get(model_key, {}).get(version, {}).keys())


# ===================================================================== #
#  MCP tool functions                                                    #
# ===================================================================== #

async def analyze_nysy_profile(
    version: str,
    rank: Optional[int] = None,
    aggregate_ranks: bool = False,
    top_n: int = 30,
    category: Optional[str] = None,
    force_reload: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze NVIDIA Nsight Systems profiler trace for a specific vLLM version.

    Load and analyze GPU kernel statistics from NYSY profiler traces with
    nanosecond precision. Useful for identifying GPU bottlenecks and
    understanding GPU utilization during inference.

    TOOL_NAME=analyze_nysy_profile
    DISPLAY_NAME=Analyze NYSY Profile
    USECASE=Analyze NVIDIA Nsight Systems profiler traces from vLLM benchmark runs to understand GPU kernel-level performance with nanosecond precision. Use to identify GPU bottlenecks and compare GPU metrics.
    INSTRUCTIONS=1. Specify a vLLM version (e.g., "v0.20.0"), 2. Optionally specify model (nemotron-120b or gpt-oss), 3. Optionally specify rank or set aggregate_ranks=True
    INPUT_DESCRIPTION=version (str): vLLM version; model (str, optional): Model name; rank (int, optional): GPU rank; aggregate_ranks (bool): Combine all ranks; top_n (int): Number of top kernels; category (str, optional): Filter by category; force_reload (bool): Ignore cache
    OUTPUT_DESCRIPTION=Dictionary with top kernels, category breakdown, and GPU metrics
    EXAMPLES=analyze_nysy_profile(version="v0.20.0"), analyze_nysy_profile(version="v0.20.0", model="nemotron-120b", aggregate_ranks=True)
    PREREQUISITES=NYSY profile traces must exist in S3 or local directory
    RELATED_TOOLS=compare_nysy_profiles, analyze_pytorch_profile, compare_pytorch_profiles

    Args:
        version: vLLM version (e.g., "v0.20.0", "0.20.0")
        rank: GPU rank to analyze (default: 0)
        aggregate_ranks: If True, aggregate stats from all ranks
        top_n: Number of top kernels to return (default: 30)
        category: Filter by category (kernel, cuda, communication, etc.)
        force_reload: Ignore cache and reload from files
        model: Model name (e.g., nemotron-120b, gpt-oss)

    Returns:
        Dictionary with status, version, model, top_kernels, category_breakdown, summary
    """
    try:
        model_key, matched_version, error, index = _resolve_model_and_version(model, version)
        if error:
            return {"status": "error", "message": error}

        assert model_key is not None and matched_version is not None

        available_ranks = _get_available_ranks(index, model_key, matched_version)
        num_ranks = len(available_ranks)
        display_name = _get_display_name(model_key)
        model_info = _get_model_info(model_key, num_ranks)

        if aggregate_ranks:
            stats_list = []
            loaded_ranks = []
            for r in available_ranks:
                s = _load_or_extract_nysy_stats(model_key, matched_version, r, force_reload)
                if s:
                    stats_list.append(s)
                    loaded_ranks.append(r)

            if not stats_list:
                return {
                    "status": "error",
                    "message": f"No NYSY trace files found for {display_name} {matched_version}",
                }

            stats = _merge_stats(stats_list)
            scope = f"all ranks ({len(loaded_ranks)} loaded)"
        else:
            r = rank if rank is not None else 0
            if r not in available_ranks:
                return {
                    "status": "error",
                    "message": f"Rank {r} not available for {display_name} {matched_version}. Available: {available_ranks}",
                }

            stats = _load_or_extract_nysy_stats(model_key, matched_version, r, force_reload)
            if not stats:
                return {
                    "status": "error",
                    "message": f"No NYSY trace found for {display_name} {matched_version} rank {r}",
                }
            scope = f"rank {r}"

        filtered_stats = _filter_by_category(stats, category)
        top_kernels = _get_top_kernels(filtered_stats, top_n)
        category_breakdown = _get_category_breakdown(stats)

        total_time = sum(d["total_dur"] for d in stats.values())
        filtered_time = sum(d["total_dur"] for d in filtered_stats.values())

        return {
            "status": "success",
            "version": matched_version,
            "model": display_name,
            "scope": scope,
            "profiler_type": "NVIDIA Nsight Systems (nanosecond precision)",
            "category_filter": category or "all",
            "top_kernels": top_kernels,
            "category_breakdown": category_breakdown,
            "summary": {
                "total_traced_time_us": total_time,
                "total_traced_time_human": _format_duration(total_time),
                "filtered_time_us": filtered_time,
                "filtered_time_human": _format_duration(filtered_time),
                "unique_operations": len(stats),
                "filtered_operations": len(filtered_stats),
            },
            "message": (
                f"Analyzed {display_name} {matched_version} ({scope}) with NYSY: "
                f"{len(filtered_stats)} unique kernels, {_format_duration(filtered_time)} total GPU time"
            ),
            "profile_info": {
                "model": model_info["model_id"],
                "gpus": model_info["gpus"],
                "ranks_stored": model_info["ranks_stored"],
                "note": model_info["note"],
            },
        }

    except Exception as exc:
        logger.error(f"Error analyzing NYSY profile: {exc}")
        return {"status": "error", "message": f"Failed to analyze NYSY profile: {exc}"}


async def compare_nysy_profiles(
    version1: str,
    version2: str,
    rank: Optional[int] = None,
    aggregate_ranks: bool = False,
    top_n: int = 30,
    category: Optional[str] = None,
    force_reload: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare NVIDIA Nsight Systems profiler traces between two vLLM versions.

    Compare GPU kernel performance to identify regressions and improvements
    with nanosecond-precision GPU timing.

    TOOL_NAME=compare_nysy_profiles
    DISPLAY_NAME=Compare NYSY Profiles
    USECASE=Compare GPU kernel performance between vLLM versions using NYSY profiler data. Use to identify GPU performance regressions/improvements with nanosecond precision.
    INSTRUCTIONS=1. Specify two versions to compare (e.g., "v0.19.0" and "v0.20.0"), 2. Optionally specify model, 3. Results show GPU kernels that got slower/faster
    INPUT_DESCRIPTION=version1 (str): Baseline version; version2 (str): Comparison version; model (str, optional): Model name; rank (int, optional): GPU rank; aggregate_ranks (bool): Combine ranks; top_n (int): Number of results; category (str, optional): Filter
    OUTPUT_DESCRIPTION=Dictionary with regressions, improvements, new/removed kernels, GPU performance comparison
    EXAMPLES=compare_nysy_profiles("v0.19.0", "v0.20.0"), compare_nysy_profiles("v0.19.0", "v0.20.0", model="nemotron-120b")
    PREREQUISITES=NYSY trace files must exist for both versions
    RELATED_TOOLS=analyze_nysy_profile, compare_pytorch_profiles, analyze_pytorch_profile

    Args:
        version1: Baseline version
        version2: Comparison version
        rank: GPU rank to compare (default: 0)
        aggregate_ranks: Aggregate all ranks
        top_n: Number of top changes to return (default: 30)
        category: Filter by category
        force_reload: Ignore cache
        model: Model name

    Returns:
        Dictionary with comparison results, regressions, improvements, summary
    """
    try:
        index = _discover_nysy_profiles(force_refresh=force_reload)

        if not index:
            return {"status": "error", "message": "No NYSY profile data available."}

        available_models = sorted(index.keys())
        matches = _match_all_models(model, available_models)
        if not matches:
            return {
                "status": "error",
                "message": f"Model '{model}' not found. Available: {available_models}",
            }
        if len(matches) > 1:
            return {
                "status": "error",
                "message": (
                    f"Multiple models match '{model}': {', '.join(matches)}. "
                    f"Please specify accelerator, e.g. '{matches[0]}'"
                ),
            }
        model_key = matches[0]

        available_versions = sorted(index[model_key].keys())
        mv1 = _match_version(version1, available_versions)
        mv2 = _match_version(version2, available_versions)

        if mv1 is None:
            return {
                "status": "error",
                "message": f"Version '{version1}' not found. Available: {available_versions}",
            }
        if mv2 is None:
            return {
                "status": "error",
                "message": f"Version '{version2}' not found. Available: {available_versions}",
            }

        display_name = _get_display_name(model_key)
        ranks_v1 = _get_available_ranks(index, model_key, mv1)
        ranks_v2 = _get_available_ranks(index, model_key, mv2)
        num_ranks = len(ranks_v1)
        model_info = _get_model_info(model_key, num_ranks)

        def load_version_stats(version: str, ranks: List[int]) -> Tuple[Optional[Dict], str]:
            if aggregate_ranks:
                parts = []
                for r in ranks:
                    s = _load_or_extract_nysy_stats(model_key, version, r, force_reload)
                    if s:
                        parts.append(s)
                if parts:
                    return _merge_stats(parts), f"all {len(parts)} ranks"
                return None, ""
            else:
                r = rank if rank is not None else 0
                return _load_or_extract_nysy_stats(model_key, version, r, force_reload), f"rank {r}"

        stats1, scope1 = load_version_stats(mv1, ranks_v1)
        stats2, scope2 = load_version_stats(mv2, ranks_v2)

        if not stats1:
            return {"status": "error", "message": f"No NYSY traces found for {display_name} {mv1}"}
        if not stats2:
            return {"status": "error", "message": f"No NYSY traces found for {display_name} {mv2}"}

        filtered1 = _filter_by_category(stats1, category)
        filtered2 = _filter_by_category(stats2, category)

        # Calculate diffs
        all_kernels = set(filtered1.keys()) | set(filtered2.keys())
        diffs = []
        for kernel in all_kernels:
            s1 = filtered1.get(kernel, {"total_dur": 0, "count": 0, "avg_dur": 0, "cat": ""})
            s2 = filtered2.get(kernel, {"total_dur": 0, "count": 0, "avg_dur": 0, "cat": ""})

            total1 = s1.get("total_dur", 0)
            total2 = s2.get("total_dur", 0)
            if total1 > 0 or total2 > 0:
                diff_abs = total2 - total1
                diff_pct = ((total2 - total1) / total1 * 100) if total1 > 0 else (100.0 if total2 > 0 else 0.0)

                count1 = s1.get("count", 0)
                count2 = s2.get("count", 0)
                avg1 = s1.get("avg_dur", 0)
                avg2 = s2.get("avg_dur", 0)
                count_change_pct = ((count2 - count1) / count1 * 100) if count1 > 0 else (100.0 if count2 > 0 else 0.0)
                avg_change_pct = ((avg2 - avg1) / avg1 * 100) if avg1 > 0 else (100.0 if avg2 > 0 else 0.0)

                diffs.append({
                    "name": kernel,
                    "category": s1.get("cat", "") or s2.get("cat", ""),
                    "total1_us": total1,
                    "total2_us": total2,
                    "total1_human": _format_duration(total1),
                    "total2_human": _format_duration(total2),
                    "count1": count1,
                    "count2": count2,
                    "count_change_pct": round(count_change_pct, 2),
                    "avg1_us": avg1,
                    "avg2_us": avg2,
                    "avg1_human": _format_duration(avg1),
                    "avg2_human": _format_duration(avg2),
                    "avg_change_pct": round(avg_change_pct, 2),
                    "diff_abs_us": diff_abs,
                    "diff_abs_human": _format_duration(abs(diff_abs)),
                    "diff_pct": round(diff_pct, 2),
                    "is_new": total1 == 0,
                    "is_removed": total2 == 0,
                })

        diffs_by_impact = sorted(diffs, key=lambda x: abs(x["diff_abs_us"]), reverse=True)

        # Categorize
        regressions = [d for d in diffs_by_impact if d["diff_abs_us"] > 0][:top_n]
        improvements = [d for d in diffs_by_impact if d["diff_abs_us"] < 0][:top_n]
        new_kernels = [d for d in diffs if d["is_new"]][:top_n]
        removed_kernels = [d for d in diffs if d["is_removed"]][:top_n]

        total1 = sum(s.get("total_dur", 0) for s in stats1.values())
        total2 = sum(s.get("total_dur", 0) for s in stats2.values())
        delta_us = total2 - total1
        delta_pct = (delta_us / total1 * 100) if total1 > 0 else 0

        return {
            "status": "success",
            "version1": mv1,
            "version2": mv2,
            "model": display_name,
            "scope": scope1,
            "profiler_type": "NVIDIA Nsight Systems (nanosecond precision)",
            "category_filter": category or "all",
            "regressions": regressions,
            "improvements": improvements,
            "new_kernels": new_kernels,
            "removed_kernels": removed_kernels,
            "summary": {
                "total1_us": total1,
                "total2_us": total2,
                "total1_human": _format_duration(total1),
                "total2_human": _format_duration(total2),
                "delta_us": delta_us,
                "delta_human": _format_duration(abs(delta_us)),
                "delta_pct": round(delta_pct, 2),
                "direction": "improvement" if delta_us < 0 else "regression",
                "regression_count": len(regressions),
                "improvement_count": len(improvements),
                "new_count": len(new_kernels),
                "removed_count": len(removed_kernels),
            },
            "message": (
                f"Compared {display_name} {mv1} vs {mv2}: "
                f"delta {_format_duration(abs(delta_us))} "
                f"({delta_pct:+.1f}%), {len(regressions)} regressions, {len(improvements)} improvements"
            ),
            "profile_info": {
                "model": model_info["model_id"],
                "gpus": model_info["gpus"],
                "ranks_stored": model_info["ranks_stored"],
                "note": model_info["note"],
            },
        }

    except Exception as exc:
        logger.error(f"Error comparing NYSY profiles: {exc}")
        return {"status": "error", "message": f"Failed to compare NYSY profiles: {exc}"}


async def analyze_nysy_performance_insights(
    version: str,
    model: Optional[str] = None,
    rank: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze NYSY profile for GPU performance insights.

    Provides high-level insights about GPU utilization, bottlenecks,
    and optimization opportunities from NYSY profiler data.

    TOOL_NAME=analyze_nysy_performance_insights
    DISPLAY_NAME=Analyze NYSY Performance Insights
    USECASE=Get high-level GPU performance insights from NYSY profiles - identify bottlenecks, utilization patterns, and optimization opportunities.
    INSTRUCTIONS=1. Specify a vLLM version, 2. Optionally specify model and rank
    INPUT_DESCRIPTION=version (str): vLLM version; model (str, optional): Model name; rank (int, optional): GPU rank (default: 0)
    OUTPUT_DESCRIPTION=Dictionary with GPU utilization insights, bottleneck analysis, optimization suggestions
    EXAMPLES=analyze_nysy_performance_insights(version="v0.20.0"), analyze_nysy_performance_insights(version="v0.20.0", model="nemotron-120b")
    PREREQUISITES=NYSY profile traces must be available
    RELATED_TOOLS=analyze_nysy_profile, compare_nysy_profiles

    Args:
        version: vLLM version
        model: Model name (optional)
        rank: GPU rank (default: 0)

    Returns:
        Dictionary with GPU utilization insights and recommendations
    """
    try:
        result = await analyze_nysy_profile(
            version=version,
            model=model,
            rank=rank if rank is not None else 0,
            aggregate_ranks=False,
            top_n=50,
            category=None,
            force_reload=False,
        )

        if result["status"] != "success":
            return result

        # Build insights from top kernels and categories
        top_kernels = result.get("top_kernels", [])
        category_breakdown = result.get("category_breakdown", {})
        total_time = result.get("summary", {}).get("total_traced_time_us", 0)

        insights = []

        # Category-level insights
        if category_breakdown:
            sorted_cats = sorted(
                category_breakdown.items(),
                key=lambda x: x[1]["total_dur_us"],
                reverse=True
            )
            top_cat = sorted_cats[0] if sorted_cats else None
            if top_cat:
                cat_name, cat_data = top_cat
                cat_pct = (cat_data["total_dur_us"] / total_time * 100) if total_time > 0 else 0
                insights.append({
                    "insight": "GPU time dominated by specific category",
                    "category": cat_name,
                    "percentage": round(cat_pct, 1),
                    "recommendation": f"Focus optimization efforts on {cat_name} operations ({cat_pct:.1f}% of GPU time)",
                })

        # Top kernel insights
        if top_kernels:
            top_kernel = top_kernels[0]
            kernel_pct = (top_kernel["total_dur_us"] / total_time * 100) if total_time > 0 else 0
            insights.append({
                "insight": "Single kernel dominates execution",
                "kernel": top_kernel["name"],
                "percentage": round(kernel_pct, 1),
                "count": top_kernel["count"],
                "recommendation": f"Optimize {top_kernel['name']} or consider kernel fusion to reduce overhead",
            })

            # Check for high-invocation kernels
            high_count = [k for k in top_kernels if k["count"] > 100]
            if high_count:
                insights.append({
                    "insight": "High kernel invocation count detected",
                    "kernel_count": len(high_count),
                    "recommendation": "Consider kernel fusion to reduce launch overhead and memory transfers",
                })

        return {
            "status": "success",
            "version": result["version"],
            "model": result["model"],
            "profiler_type": "NVIDIA Nsight Systems (nanosecond precision)",
            "insights": insights,
            "category_breakdown": category_breakdown,
            "top_kernels_sample": top_kernels[:10],
            "message": f"GPU performance insights for {result['model']} {result['version']}: {len(insights)} insights identified",
        }

    except Exception as exc:
        logger.error(f"Error analyzing NYSY insights: {exc}")
        return {"status": "error", "message": f"Failed to analyze NYSY insights: {exc}"}
