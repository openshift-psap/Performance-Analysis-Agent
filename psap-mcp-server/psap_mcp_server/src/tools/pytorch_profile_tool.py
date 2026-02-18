"""MCP tool for analyzing PyTorch profiler traces from vLLM benchmark runs.

This tool dynamically discovers Chrome trace JSON files from either S3
(``s3://<bucket>/<PROFILE_S3_PREFIX>/<model>/<version>/``) or a local
directory fallback, extracts kernel statistics, and enables comparison
between different vLLM versions to identify performance regressions and
improvements.

New profiles are auto-discovered -- just upload trace files to S3 under
the expected folder structure and the agent will find them.
"""

import json
import os
import re
import time as _time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from psap_mcp_server.utils.pylogger import get_python_logger
from psap_mcp_server.src.settings import settings

logger = get_python_logger()

# ---------------------------------------------------------------------------
# Known model metadata (optional enrichment).
# Models not listed here are still usable -- the folder name is used as-is.
# ---------------------------------------------------------------------------
KNOWN_MODELS: Dict[str, Dict[str, str]] = {
    "deepseek-r1": {
        "display_name": "DeepSeek-R1",
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "gpus": "H200",
    },
    "gpt-oss": {
        "display_name": "GPT-OSS",
        "model_id": "gpt-oss",
        "gpus": "H200",
    },
}

# Aliases map commonly-used names to the canonical S3/local folder names.
_MODEL_ALIASES: Dict[str, str] = {
    "deepseek": "deepseek-r1",
    "deepseek-r1-0528": "deepseek-r1",
    "gptoss": "gpt-oss",
    "gpt_oss": "gpt-oss",
}

# Local fallback directory (used when S3 is not configured or unreachable)
_DEFAULT_LOCAL_PROFILE_BASE = "/app/pytorch-profiles"
LOCAL_PROFILE_BASE: str = (
    getattr(settings, "PYTORCH_PROFILE_BASE", None)
    or os.environ.get("PYTORCH_PROFILE_BASE", _DEFAULT_LOCAL_PROFILE_BASE)
)

# Category groups for filtering
CATEGORY_GROUPS: Dict[str, Optional[List[str]]] = {
    "kernel": ["kernel"],
    "cpu": ["cpu_op"],
    "cuda": ["cuda_runtime", "cuda_driver"],
    "communication": ["nccl", "c10d"],
    "memory": ["cuda_memory"],
    "all": None,
}

# ---------------------------------------------------------------------------
# Profile index cache  (discovery results: model -> version -> rank -> entry)
# ---------------------------------------------------------------------------
_profile_index_cache: Optional[Dict] = None
_profile_index_ts: float = 0.0
_PROFILE_INDEX_TTL: int = 300  # seconds

# ---------------------------------------------------------------------------
# In-memory stats cache  (cache_key -> (timestamp, stats_dict))
# ---------------------------------------------------------------------------
_stats_cache: Dict[str, Tuple[float, Dict]] = {}
_STATS_CACHE_TTL: int = 300  # seconds


# ===================================================================== #
#  Helper functions                                                      #
# ===================================================================== #

def _parse_rank_from_filename(filename: str) -> Optional[int]:
    """Extract the rank number from a profiler trace filename.

    Handles patterns produced by PyTorch profiler, e.g.:
      trace_rank0_pid467_range2000-2010.json
      trace_rank3_pid458_range2000-2010_v0112.json
      rank7.json
    """
    match = re.search(r"rank(\d+)", filename)
    return int(match.group(1)) if match else None


def _extract_bare_version(version: str) -> str:
    """Extract the bare numeric version (e.g. '0.13.0') from any format.

    Handles: 'vLLM-0.13.0', 'v0.13.0', '0.13.0', 'vllm 0.13.0', etc.
    """
    version = version.strip()
    # Remove common prefixes (case-insensitive)
    for prefix in ("vllm-", "vllm ", "vLLM-", "vLLM "):
        if version.startswith(prefix):
            version = version[len(prefix):]
            break
    # Strip leading 'v' if present
    version = version.lstrip("v")
    return version


def _normalize_version(version: str) -> str:
    """Normalise a user-provided version string to ``vLLM-X.Y.Z`` format."""
    bare = _extract_bare_version(version)
    return f"vLLM-{bare}"


def _match_version(user_version: str, available_versions: List[str]) -> Optional[str]:
    """Fuzzy-match a user-provided version against discovered folder names.

    Compares the bare numeric portion so that 'v0.13.0', '0.13.0',
    'vLLM-0.13.0', and 'vllm 0.13.0' all match the same folder.
    """
    user_bare = _extract_bare_version(user_version)

    for v in available_versions:
        if _extract_bare_version(v) == user_bare:
            return v
    return None


def _match_model(user_model: Optional[str], available_models: List[str]) -> Optional[str]:
    """Fuzzy-match a user-provided model name against discovered folder names.

    Returns the matched folder name, or the first available model when
    *user_model* is ``None``.
    """
    if not available_models:
        return None

    if user_model is None:
        return available_models[0]

    lower = user_model.lower().strip()

    # Direct match
    if lower in available_models:
        return lower

    # Alias match
    alias = _MODEL_ALIASES.get(lower)
    if alias and alias in available_models:
        return alias

    # Substring / known-metadata match
    for model_key in available_models:
        if lower in model_key or model_key in lower:
            return model_key
        info = KNOWN_MODELS.get(model_key, {})
        if lower in info.get("display_name", "").lower():
            return model_key
        if lower in info.get("model_id", "").lower():
            return model_key

    return None


def _get_display_name(model_key: str) -> str:
    """Human-friendly display name for a model folder name."""
    return KNOWN_MODELS.get(model_key, {}).get("display_name", model_key)


def _get_model_info(model_key: str, num_ranks: int) -> Dict[str, Any]:
    """Build a model info dict from known metadata + discovered data."""
    known = KNOWN_MODELS.get(model_key, {})
    return {
        "display_name": known.get("display_name", model_key),
        "model_id": known.get("model_id", model_key),
        "gpus": known.get("gpus", "unknown"),
        "tensor_parallelism": num_ranks,
        "num_ranks": num_ranks,
    }


# ===================================================================== #
#  Profile discovery (S3 + local)                                        #
# ===================================================================== #

def _discover_profiles_s3() -> Optional[Dict]:
    """Discover available profiles from S3 by listing directories.

    Walks ``s3://<bucket>/<prefix>/<model>/<version>/`` and returns::

        {model: {version: {rank: {"source": "s3", "key": ..., ...}}}}

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
        logger.warning("boto3 not installed -- S3 profile discovery unavailable")
        return None
    except Exception as exc:
        logger.warning(f"Failed to create S3 client: {exc}")
        return None

    index: Dict[str, Dict[str, Dict[int, Dict]]] = {}

    try:
        # 1. List model folders
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        model_prefixes = [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]

        for model_prefix in model_prefixes:
            model_name = model_prefix.rstrip("/").split("/")[-1]

            # 2. List version folders
            resp2 = s3.list_objects_v2(
                Bucket=bucket, Prefix=model_prefix, Delimiter="/"
            )
            version_prefixes = [
                cp["Prefix"] for cp in resp2.get("CommonPrefixes", [])
            ]

            for version_prefix in version_prefixes:
                version_name = version_prefix.rstrip("/").split("/")[-1]

                # 3. List trace files (handle pagination)
                paginator_token = None
                while True:
                    kwargs: Dict[str, Any] = {
                        "Bucket": bucket,
                        "Prefix": version_prefix,
                    }
                    if paginator_token:
                        kwargs["ContinuationToken"] = paginator_token

                    resp3 = s3.list_objects_v2(**kwargs)
                    for obj in resp3.get("Contents", []):
                        key = obj["Key"]
                        filename = key.split("/")[-1]
                        if not filename.endswith(".json"):
                            continue
                        rank = _parse_rank_from_filename(filename)
                        if rank is None:
                            continue

                        index.setdefault(model_name, {}).setdefault(
                            version_name, {}
                        )[rank] = {
                            "source": "s3",
                            "key": key,
                            "filename": filename,
                            "size_bytes": obj.get("Size", 0),
                        }

                    if resp3.get("IsTruncated"):
                        paginator_token = resp3.get("NextContinuationToken")
                    else:
                        break

        if index:
            summary = ", ".join(
                f"{m} ({len(vs)} version(s))" for m, vs in index.items()
            )
            logger.info(f"S3 profile discovery: {summary}")
        else:
            logger.info(
                f"S3 profile discovery: no profiles under s3://{bucket}/{prefix}"
            )

        return index

    except Exception as exc:
        logger.warning(f"S3 profile discovery failed: {exc}")
        return None


def _discover_profiles_local(base_dir: Optional[str] = None) -> Dict:
    """Discover available profiles from the local filesystem.

    Scans ``base_dir/<model>/<version>/*.json`` for files containing
    ``rank`` in the filename.

    Returns::

        {model: {version: {rank: {"source": "local", "path": Path, ...}}}}
    """
    base = Path(base_dir or LOCAL_PROFILE_BASE)
    index: Dict[str, Dict[str, Dict[int, Dict]]] = {}

    if not base.exists():
        logger.info(f"Local profile base not found: {base}")
        return index

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model_name = model_dir.name

        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir() or version_dir.name.startswith("."):
                continue
            version_name = version_dir.name

            for f in sorted(version_dir.iterdir()):
                if not f.is_file() or not f.name.endswith(".json"):
                    continue
                rank = _parse_rank_from_filename(f.name)
                if rank is None:
                    continue

                index.setdefault(model_name, {}).setdefault(
                    version_name, {}
                )[rank] = {
                    "source": "local",
                    "path": f,
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                }

    if index:
        summary = ", ".join(
            f"{m} ({len(vs)} version(s))" for m, vs in index.items()
        )
        logger.info(f"Local profile discovery: {summary}")
    else:
        logger.info(f"Local profile discovery: no profiles under {base}")

    return index


def _discover_profiles(force_refresh: bool = False) -> Dict:
    """Discover available profiles from S3, with in-memory caching.

    Results are cached for ``_PROFILE_INDEX_TTL`` seconds.
    """
    global _profile_index_cache, _profile_index_ts

    now = _time.time()
    if (
        not force_refresh
        and _profile_index_cache is not None
        and (now - _profile_index_ts) < _PROFILE_INDEX_TTL
    ):
        return _profile_index_cache

    index = _discover_profiles_s3()

    if index:
        logger.info(f"Profile index built from S3: {len(index)} model(s)")
    else:
        logger.warning(
            "No profile data found in S3. "
            "Upload traces to s3://<bucket>/<PROFILE_S3_PREFIX>/<model>/<version>/"
        )
        index = {}

    _profile_index_cache = index
    _profile_index_ts = now
    return index


# ===================================================================== #
#  In-memory stats cache                                                 #
# ===================================================================== #

def _get_cached_stats(cache_key: str) -> Optional[Dict[str, Dict]]:
    """Return stats from in-memory cache if still valid."""
    if cache_key in _stats_cache:
        ts, stats = _stats_cache[cache_key]
        if (_time.time() - ts) < _STATS_CACHE_TTL:
            return stats
        del _stats_cache[cache_key]
    return None


def _put_cached_stats(cache_key: str, stats: Dict[str, Dict]) -> None:
    """Store stats in the in-memory cache."""
    _stats_cache[cache_key] = (_time.time(), stats)


# ===================================================================== #
#  Trace loading                                                         #
# ===================================================================== #

def _extract_kernel_stats(trace: dict) -> Dict[str, Dict]:
    """Extract kernel statistics from a Chrome trace JSON dict.

    Returns dict of ``kernel_name -> {count, total_dur, avg_dur, min_dur,
    max_dur, cat}``.
    """
    stats: Dict[str, Dict] = defaultdict(
        lambda: {"count": 0, "total_dur": 0, "min_dur": float("inf"), "max_dur": 0, "cat": ""}
    )

    for event in trace.get("traceEvents", []):
        if event.get("ph") == "X":
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)
            cat = event.get("cat", "")
            if dur > 0:
                stats[name]["count"] += 1
                stats[name]["total_dur"] += dur
                stats[name]["min_dur"] = min(stats[name]["min_dur"], dur)
                stats[name]["max_dur"] = max(stats[name]["max_dur"], dur)
                if not stats[name]["cat"]:
                    stats[name]["cat"] = cat

    for data in stats.values():
        data["avg_dur"] = data["total_dur"] / data["count"] if data["count"] > 0 else 0
        if data["min_dur"] == float("inf"):
            data["min_dur"] = 0

    return dict(stats)


def _load_trace_from_s3(s3_key: str) -> Optional[dict]:
    """Download and parse a Chrome trace JSON from S3."""
    bucket = settings.S3_BUCKET
    if not bucket:
        return None

    try:
        from psap_mcp_server.src.tools.s3_utils import get_s3_client

        s3 = get_s3_client()
        logger.info(f"Downloading trace from S3: s3://{bucket}/{s3_key}")
        response = s3.get_object(Bucket=bucket, Key=s3_key)
        body = response["Body"].read()
        return json.loads(body)
    except json.JSONDecodeError as exc:
        logger.error(f"Corrupted JSON in S3 trace {s3_key}: {exc}")
        return None
    except MemoryError:
        logger.error(f"Out of memory loading S3 trace {s3_key}")
        return None
    except Exception as exc:
        logger.error(f"Failed to load S3 trace {s3_key}: {exc}")
        return None


def _load_trace_from_local(filepath: Path) -> Optional[dict]:
    """Load a Chrome trace JSON file from local filesystem."""
    if not filepath.exists():
        logger.warning(f"Trace file not found: {filepath}")
        return None

    logger.info(f"Loading local trace: {filepath.name}")
    try:
        with open(filepath, "r") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error(f"Corrupted JSON in {filepath.name}: {exc}")
        return None
    except MemoryError:
        logger.error(f"Out of memory loading {filepath.name}")
        return None
    except Exception as exc:
        logger.error(f"Failed to load {filepath.name}: {exc}")
        return None


def _load_or_extract_stats(
    model: str,
    version: str,
    rank: int,
    force_reload: bool = False,
) -> Optional[Dict[str, Dict]]:
    """Load kernel stats for a given model/version/rank.

    1. Check in-memory cache
    2. Look up the file in the profile index (S3 or local)
    3. Download / read the trace, extract stats, cache them

    Returns ``None`` if the trace cannot be loaded.
    """
    cache_key = f"{model}/{version}/rank{rank}"

    if not force_reload:
        cached = _get_cached_stats(cache_key)
        if cached is not None:
            logger.debug(f"Stats cache hit: {cache_key}")
            return cached

    # Look up in the profile index
    index = _discover_profiles()
    entry = index.get(model, {}).get(version, {}).get(rank)

    if entry is None:
        logger.warning(f"No trace found for {cache_key}")
        return None

    # Load the trace
    if entry["source"] == "s3":
        trace = _load_trace_from_s3(entry["key"])
    else:
        trace = _load_trace_from_local(entry["path"])

    if not trace:
        logger.warning(f"Failed to load trace for {cache_key} - skipping")
        return None

    stats = _extract_kernel_stats(trace)
    _put_cached_stats(cache_key, stats)
    return stats


# ===================================================================== #
#  Pure utility functions (operate on stats dicts, no I/O)               #
# ===================================================================== #

def _merge_stats(stats_list: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """Merge statistics from multiple ranks into one aggregated view."""
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
        return f"{us:.2f}\u00b5s"


def _filter_by_category(stats: Dict[str, Dict], category: Optional[str]) -> Dict[str, Dict]:
    """Filter stats to only include a specified category."""
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
    """Get top N kernels sorted by specified metric."""
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


# ===================================================================== #
#  Internal helpers for tool functions                                   #
# ===================================================================== #

def _resolve_model_and_version(
    model: Optional[str],
    version: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Dict]:
    """Resolve user-provided model/version to discovered folder names.

    Returns (model_key, matched_version, error_message, index).
    If error_message is not None the caller should return it.
    """
    index = _discover_profiles()

    if not index:
        return None, None, "No profile data available. Check S3 configuration or local profile directory.", index

    available_models = sorted(index.keys())
    model_key = _match_model(model, available_models)
    if model_key is None:
        return None, None, (
            f"Model '{model}' not found. Available models: {available_models}"
        ), index

    available_versions = sorted(index[model_key].keys())
    matched_version = _match_version(version, available_versions)
    if matched_version is None:
        return model_key, None, (
            f"Version '{version}' not found for {_get_display_name(model_key)}. "
            f"Available versions: {available_versions}"
        ), index

    return model_key, matched_version, None, index


def _get_num_ranks(index: Dict, model_key: str, version: str) -> int:
    """Return the number of discovered ranks for a model/version."""
    ranks = index.get(model_key, {}).get(version, {})
    return max(ranks.keys()) + 1 if ranks else 0


def _get_available_ranks(index: Dict, model_key: str, version: str) -> List[int]:
    """Return a sorted list of discovered ranks."""
    return sorted(index.get(model_key, {}).get(version, {}).keys())


# ===================================================================== #
#  MCP tool functions                                                    #
# ===================================================================== #

async def analyze_pytorch_profile(
    version: str,
    rank: Optional[int] = None,
    aggregate_ranks: bool = False,
    top_n: int = 30,
    category: Optional[str] = None,
    force_reload: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze PyTorch profiler trace for a specific vLLM version and model.
    
    Load and analyze kernel statistics from PyTorch profiler traces to understand
    performance characteristics of vLLM inference. Useful for identifying slow
    operations and understanding where time is spent during inference.
    
    TOOL_NAME=analyze_pytorch_profile
    DISPLAY_NAME=Analyze PyTorch Profile
    USECASE=Analyze PyTorch profiler traces from vLLM benchmark runs to understand kernel-level performance. Use to identify slow operations, see time breakdown by category, and understand inference bottlenecks.
    INSTRUCTIONS=1. Specify a vLLM version (e.g., "v0.11.2" or "v0.13.0"), 2. Optionally specify model (deepseek or gpt-oss), 3. Optionally specify a rank or set aggregate_ranks=True, 4. Use category filter to focus on specific operation types
    INPUT_DESCRIPTION=version (str): vLLM version like "v0.11.2" or "v0.13.0"; model (str, optional): Model name - "deepseek" (default) or "gpt-oss"; rank (int, optional): GPU rank, defaults to 0; aggregate_ranks (bool): Combine all ranks; top_n (int): Number of top kernels; category (str, optional): Filter by category
    OUTPUT_DESCRIPTION=Dictionary with top kernels by time, category breakdown, and summary statistics
    EXAMPLES=analyze_pytorch_profile(version="v0.13.0"), analyze_pytorch_profile(version="v0.11.2", model="gpt-oss"), analyze_pytorch_profile(version="v0.13.0", model="deepseek", aggregate_ranks=True)
    PREREQUISITES=Profile traces must exist in S3 or local directory
    RELATED_TOOLS=compare_pytorch_profiles, map_kernel_to_vllm_code, compare_vllm_versions
    
    Args:
        version: vLLM version (e.g., "v0.11.2", "v0.13.0", "0.11.2")
        rank: GPU rank to analyze. If None and aggregate_ranks=False, uses rank 0
        aggregate_ranks: If True, aggregate statistics from all ranks
        top_n: Number of top kernels to return (default: 30)
        category: Filter by category (kernel, cpu, cuda, communication, memory, all)
        force_reload: If True, ignore cache and reload from trace files
        model: Model name - "deepseek" (default, TP=8) or "gpt-oss" (TP=4)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - version: The version analyzed
        - model: The model analyzed
        - top_kernels: List of top kernels by total time
        - category_breakdown: Time breakdown by operation category
        - summary: Overall statistics
    """
    try:
        model_key, matched_version, error, index = _resolve_model_and_version(model, version)
        if error:
            return {"status": "error", "message": error}

        assert model_key is not None and matched_version is not None  # mypy

        available_ranks = _get_available_ranks(index, model_key, matched_version)
        num_ranks = len(available_ranks)
        display_name = _get_display_name(model_key)
        model_info = _get_model_info(model_key, num_ranks)

        if aggregate_ranks:
            stats_list = []
            loaded_ranks = []
            for r in available_ranks:
                s = _load_or_extract_stats(model_key, matched_version, r, force_reload)
                if s:
                    stats_list.append(s)
                    loaded_ranks.append(r)

            if not stats_list:
                return {
                    "status": "error",
                    "message": f"No trace files could be loaded for {display_name} {matched_version}",
                }

            stats = _merge_stats(stats_list)
            scope = f"all ranks ({len(loaded_ranks)} loaded)"
        else:
            r = rank if rank is not None else 0
            if r not in available_ranks:
                return {
                    "status": "error",
                    "message": f"Rank {r} not available for {display_name} {matched_version}. Available ranks: {available_ranks}",
                }

            stats = _load_or_extract_stats(model_key, matched_version, r, force_reload)
            if not stats:
                return {
                    "status": "error",
                    "message": f"No trace file found for {display_name} {matched_version} rank {r}",
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
                f"Analyzed {display_name} {matched_version} ({scope}): "
                f"{len(filtered_stats)} unique operations, {_format_duration(filtered_time)} total time"
            ),
            "profile_info": {
                "model": model_info["model_id"],
                "gpus": model_info["gpus"],
                "tensor_parallelism": model_info["tensor_parallelism"],
            },
        }

    except Exception as exc:
        logger.error(f"Error analyzing PyTorch profile: {exc}")
        return {"status": "error", "message": f"Failed to analyze profile: {exc}"}


async def compare_pytorch_profiles(
    version1: str,
    version2: str,
    rank: Optional[int] = None,
    aggregate_ranks: bool = False,
    top_n: int = 30,
    category: Optional[str] = None,
    force_reload: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare PyTorch profiler traces between two vLLM versions.
    
    Compare kernel-level performance between two vLLM versions to identify
    regressions and improvements. Useful for understanding performance changes
    between releases and correlating with code changes.
    
    TOOL_NAME=compare_pytorch_profiles
    DISPLAY_NAME=Compare PyTorch Profiles
    USECASE=Compare kernel performance between two vLLM versions to identify regressions and improvements. Use this when investigating performance changes between releases or trying to understand why one version is slower/faster.
    INSTRUCTIONS=1. Specify two versions to compare (e.g., "v0.11.2" and "v0.13.0"), 2. Optionally specify model (deepseek or gpt-oss), 3. Results show which kernels got slower (regressions) and faster (improvements)
    INPUT_DESCRIPTION=version1 (str): First/baseline version; version2 (str): Second/comparison version; model (str, optional): Model - "deepseek" or "gpt-oss"; rank (int, optional): GPU rank; aggregate_ranks (bool): Combine all ranks; top_n (int): Number of results; category (str, optional): Filter by category
    OUTPUT_DESCRIPTION=Dictionary with regressions, improvements, new/removed kernels, and summary comparison
    EXAMPLES=compare_pytorch_profiles("v0.11.2", "v0.13.0"), compare_pytorch_profiles("v0.11.2", "v0.13.0", model="gpt-oss"), compare_pytorch_profiles("v0.11.2", "v0.13.0", model="deepseek", aggregate_ranks=True)
    PREREQUISITES=Profile traces must exist for both versions
    RELATED_TOOLS=analyze_pytorch_profile, map_kernel_to_vllm_code, compare_vllm_versions, get_vllm_release_notes
    
    Args:
        version1: First/baseline vLLM version (e.g., "v0.11.2")
        version2: Second/comparison vLLM version (e.g., "v0.13.0")
        rank: GPU rank to compare. If None and aggregate_ranks=False, uses rank 0
        aggregate_ranks: If True, aggregate statistics from all ranks
        top_n: Number of top regressions/improvements to return (default: 30)
        category: Filter by category (kernel, cpu, cuda, communication, memory, all)
        force_reload: If True, ignore cache and reload from trace files
        model: Model name - "deepseek" (default, TP=8) or "gpt-oss" (TP=4)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - comparison: Detailed comparison results
        - regressions: Kernels that got slower in version2
        - improvements: Kernels that got faster in version2
        - new_kernels: Kernels only in version2
        - removed_kernels: Kernels only in version1
        - summary: Overall comparison statistics
    """
    try:
        index = _discover_profiles(force_refresh=force_reload)

        if not index:
            return {"status": "error", "message": "No profile data available."}

        available_models = sorted(index.keys())
        model_key = _match_model(model, available_models)
        if model_key is None:
            return {
                "status": "error",
                "message": f"Model '{model}' not found. Available: {available_models}",
            }

        available_versions = sorted(index[model_key].keys())

        mv1 = _match_version(version1, available_versions)
        mv2 = _match_version(version2, available_versions)
        if mv1 is None:
            return {
                "status": "error",
                "message": f"Version '{version1}' not found for {_get_display_name(model_key)}. Available: {available_versions}",
            }
        if mv2 is None:
            return {
                "status": "error",
                "message": f"Version '{version2}' not found for {_get_display_name(model_key)}. Available: {available_versions}",
            }

        display_name = _get_display_name(model_key)

        # Determine ranks to load
        ranks_v1 = _get_available_ranks(index, model_key, mv1)
        ranks_v2 = _get_available_ranks(index, model_key, mv2)
        num_ranks = len(ranks_v1)
        model_info = _get_model_info(model_key, num_ranks)

        def load_version_stats(version: str, ranks: List[int]) -> Tuple[Optional[Dict], str]:
            if aggregate_ranks:
                parts = []
                for r in ranks:
                    s = _load_or_extract_stats(model_key, version, r, force_reload)
                    if s:
                        parts.append(s)
                if parts:
                    return _merge_stats(parts), f"all {len(parts)} ranks"
                return None, ""
            else:
                r = rank if rank is not None else 0
                return _load_or_extract_stats(model_key, version, r, force_reload), f"rank {r}"

        stats1, scope1 = load_version_stats(mv1, ranks_v1)
        stats2, scope2 = load_version_stats(mv2, ranks_v2)

        if not stats1:
            return {"status": "error", "message": f"No trace files found for {display_name} {mv1}"}
        if not stats2:
            return {"status": "error", "message": f"No trace files found for {display_name} {mv2}"}

        # Apply category filter
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
                diffs.append(
                    {
                        "name": kernel,
                        "category": s1.get("cat", "") or s2.get("cat", ""),
                        "total1_us": total1,
                        "total2_us": total2,
                        "total1_human": _format_duration(total1),
                        "total2_human": _format_duration(total2),
                        "count1": s1.get("count", 0),
                        "count2": s2.get("count", 0),
                        "avg1_us": s1.get("avg_dur", 0),
                        "avg2_us": s2.get("avg_dur", 0),
                        "diff_abs_us": diff_abs,
                        "diff_abs_human": _format_duration(abs(diff_abs)),
                        "diff_pct": round(diff_pct, 2),
                        "is_new": total1 == 0,
                        "is_removed": total2 == 0,
                    }
                )

        diffs_by_impact = sorted(diffs, key=lambda x: abs(x["diff_abs_us"]), reverse=True)

        regressions = [d for d in diffs_by_impact if d["diff_abs_us"] > 0 and not d["is_new"]][:top_n]
        improvements = [d for d in diffs_by_impact if d["diff_abs_us"] < 0 and not d["is_removed"]][:top_n]
        new_kernels = sorted([d for d in diffs if d["is_new"]], key=lambda x: x["total2_us"], reverse=True)[:top_n]
        removed_kernels = sorted([d for d in diffs if d["is_removed"]], key=lambda x: x["total1_us"], reverse=True)[:top_n]

        total_time1 = sum(d["total_dur"] for d in filtered1.values())
        total_time2 = sum(d["total_dur"] for d in filtered2.values())
        total_diff = total_time2 - total_time1
        total_diff_pct = (total_diff / total_time1 * 100) if total_time1 > 0 else 0.0

        regression_impact = sum(d["diff_abs_us"] for d in regressions)
        improvement_impact = sum(abs(d["diff_abs_us"]) for d in improvements)

        return {
            "status": "success",
            "comparison": {
                "version1": mv1,
                "version2": mv2,
                "model": display_name,
                "scope": scope1 if scope1 == scope2 else f"{mv1}: {scope1}, {mv2}: {scope2}",
                "category_filter": category or "all",
            },
            "regressions": regressions,
            "improvements": improvements,
            "new_kernels": new_kernels,
            "removed_kernels": removed_kernels,
            "summary": {
                "total_time1_us": total_time1,
                "total_time1_human": _format_duration(total_time1),
                "total_time2_us": total_time2,
                "total_time2_human": _format_duration(total_time2),
                "total_diff_us": total_diff,
                "total_diff_human": _format_duration(abs(total_diff)),
                "total_diff_pct": round(total_diff_pct, 2),
                "direction": "slower" if total_diff > 0 else "faster",
                "regression_count": len([d for d in diffs if d["diff_abs_us"] > 0 and not d["is_new"]]),
                "improvement_count": len([d for d in diffs if d["diff_abs_us"] < 0 and not d["is_removed"]]),
                "new_kernel_count": len([d for d in diffs if d["is_new"]]),
                "removed_kernel_count": len([d for d in diffs if d["is_removed"]]),
                "regression_impact_us": regression_impact,
                "regression_impact_human": _format_duration(regression_impact),
                "improvement_impact_us": improvement_impact,
                "improvement_impact_human": _format_duration(improvement_impact),
            },
            "message": (
                f"{display_name}: {mv2} is {abs(total_diff_pct):.1f}% "
                f"{'slower' if total_diff > 0 else 'faster'} than {mv1}. "
                f"Found {len(regressions)} regressions and {len(improvements)} improvements."
            ),
            "profile_info": {
                "model": model_info["model_id"],
                "gpus": model_info["gpus"],
                "tensor_parallelism": model_info["tensor_parallelism"],
            },
            "next_steps": [
                "Use map_kernel_to_vllm_code to find source files for top regressions",
                f"Use compare_vllm_versions('{mv1}', '{mv2}') to see release notes between versions",
                "Use get_vllm_pull_request to investigate specific PRs mentioned in release notes",
            ],
        }

    except Exception as exc:
        logger.error(f"Error comparing PyTorch profiles: {exc}")
        return {"status": "error", "message": f"Failed to compare profiles: {exc}"}


async def analyze_performance_insights(
    version1: str,
    version2: str,
    rank: int = 0,
    benchmark_shows_improvement: bool = True,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Deep analysis of profile differences with intelligent insights and conclusions.
    
    This tool provides expert-level analysis of PyTorch profiler data, explaining
    WHY performance changed between versions. It analyzes compute vs overhead,
    synchronization patterns, kernel fusion, and provides actionable conclusions.
    
    IMPORTANT: Profile total time can be misleading! This tool separates actual
    GPU compute from overhead and explains paradoxes like "more profile time but
    better benchmark throughput".
    
    TOOL_NAME=analyze_performance_insights
    DISPLAY_NAME=Analyze Performance Insights
    USECASE=Get expert-level analysis explaining WHY performance changed between versions. Use this when you need to understand the root cause of performance differences, especially when benchmark results seem to contradict profile data.
    INSTRUCTIONS=1. Provide two versions to compare, 2. Optionally specify model (deepseek or gpt-oss), 3. Set benchmark_shows_improvement based on your actual benchmark results
    INPUT_DESCRIPTION=version1 (str): Baseline version; version2 (str): Comparison version; model (str, optional): Model - "deepseek" or "gpt-oss"; rank (int): GPU rank (default 0); benchmark_shows_improvement (bool): Whether actual benchmarks show version2 is faster
    OUTPUT_DESCRIPTION=Dictionary with deep analysis including compute vs overhead breakdown, synchronization analysis, kernel fusion detection, and expert conclusions
    EXAMPLES=analyze_performance_insights("v0.11.2", "v0.13.0"), analyze_performance_insights("v0.11.2", "v0.13.0", model="gpt-oss")
    PREREQUISITES=Profile traces must exist for both versions
    RELATED_TOOLS=compare_pytorch_profiles, compare_vllm_versions, map_kernel_to_vllm_code
    
    Args:
        version1: Baseline vLLM version (e.g., "v0.11.2")
        version2: Comparison vLLM version (e.g., "v0.13.0")
        rank: GPU rank to analyze (default: 0)
        benchmark_shows_improvement: Set True if actual benchmarks show version2 is faster
        model: Model name - "deepseek" (default, TP=8) or "gpt-oss" (TP=4)
    
    Returns:
        Dictionary with deep analysis and expert conclusions.
    """
    try:
        # Resolve model & both versions
        index = _discover_profiles()
        if not index:
            return {"status": "error", "message": "No profile data available."}

        available_models = sorted(index.keys())
        model_key = _match_model(model, available_models)
        if model_key is None:
            return {"status": "error", "message": f"Model '{model}' not found. Available: {available_models}"}

        available_versions = sorted(index[model_key].keys())
        mv1 = _match_version(version1, available_versions)
        mv2 = _match_version(version2, available_versions)
        if mv1 is None:
            return {"status": "error", "message": f"Version '{version1}' not found. Available: {available_versions}"}
        if mv2 is None:
            return {"status": "error", "message": f"Version '{version2}' not found. Available: {available_versions}"}

        display_name = _get_display_name(model_key)

        stats1 = _load_or_extract_stats(model_key, mv1, rank)
        stats2 = _load_or_extract_stats(model_key, mv2, rank)

        if not stats1 or not stats2:
            return {"status": "error", "message": "Missing profile data for comparison"}

        # 1. Compute vs Overhead Analysis
        def categorize_time(stats: Dict) -> Dict[str, float]:
            compute = sum(d["total_dur"] for d in stats.values() if d.get("cat") == "kernel")
            overhead = sum(d["total_dur"] for d in stats.values() if d.get("cat") in ("cuda_runtime", "cpu_op"))
            sync = sum(d["total_dur"] for n, d in stats.items() if "synchronize" in n.lower() or "sync" in n.lower())
            profiler = sum(d["total_dur"] for n, d in stats.items() if "profiler" in n.lower() or "Profiler" in n)
            total = sum(d["total_dur"] for d in stats.values())
            return {"compute": compute, "overhead": overhead, "sync": sync, "profiler": profiler, "total": total}

        time1 = categorize_time(stats1)
        time2 = categorize_time(stats2)

        def _pct(a: float, b: float) -> float:
            return ((b - a) / a * 100) if a > 0 else 0.0

        compute_change_pct = _pct(time1["compute"], time2["compute"])
        overhead_change_pct = _pct(time1["overhead"], time2["overhead"])
        sync_change_pct = _pct(time1["sync"], time2["sync"])
        total_change_pct = _pct(time1["total"], time2["total"])

        # 2. Synchronization Analysis
        sync_ops: Dict[str, Dict] = {}
        for op in ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize"):
            s1 = stats1.get(op, {})
            s2 = stats2.get(op, {})
            if s1 or s2:
                sync_ops[op] = {
                    "v1_time_s": s1.get("total_dur", 0) / 1e6,
                    "v2_time_s": s2.get("total_dur", 0) / 1e6,
                    "v1_calls": s1.get("count", 0),
                    "v2_calls": s2.get("count", 0),
                    "change_pct": _pct(s1.get("total_dur", 0) or 1, s2.get("total_dur", 0)),
                }

        # 3. Kernel Launch Analysis
        launch_ops: Dict[str, Dict] = {}
        for op in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx"):
            s1 = stats1.get(op, {})
            s2 = stats2.get(op, {})
            if s1 or s2:
                launch_ops[op] = {
                    "v1_calls": s1.get("count", 0),
                    "v2_calls": s2.get("count", 0),
                    "change_pct": _pct(s1.get("count", 0) or 1, s2.get("count", 0)),
                }

        total_launches_v1 = sum(d["v1_calls"] for d in launch_ops.values())
        total_launches_v2 = sum(d["v2_calls"] for d in launch_ops.values())

        # 4. Kernel Fusion Detection
        only_v1 = {n: stats1[n] for n in stats1 if n not in stats2 and stats1[n].get("total_dur", 0) > 100_000}
        only_v2 = {n: stats2[n] for n in stats2 if n not in stats1 and stats2[n].get("total_dur", 0) > 100_000}

        fusion_detected: List[Dict] = []
        old_moe = [n for n in only_v1 if "moe" in n.lower() or "expert" in n.lower() or "gemm" in n.lower()]
        new_moe = [n for n in only_v2 if "moe" in n.lower() or "expert" in n.lower() or "fused" in n.lower()]
        if old_moe and new_moe:
            fusion_detected.append(
                {
                    "type": "MoE Kernel Fusion",
                    "description": "Multiple MoE/GEMM kernels replaced with fused implementation",
                    "removed_kernels": old_moe[:5],
                    "new_kernels": new_moe[:5],
                }
            )

        # 5. Domain-specific Analysis
        def analyze_domain(patterns: List[str]) -> List[Dict]:
            kernels = []
            for n in set(stats1.keys()) | set(stats2.keys()):
                if any(p in n.lower() for p in patterns):
                    t1 = stats1.get(n, {}).get("total_dur", 0) / 1e6
                    t2 = stats2.get(n, {}).get("total_dur", 0) / 1e6
                    if t1 > 0.1 or t2 > 0.1:
                        kernels.append(
                            {
                                "name": n[:60],
                                "v1_time_s": round(t1, 2),
                                "v2_time_s": round(t2, 2),
                                "change_pct": round(_pct(t1 or 1, t2), 1),
                            }
                        )
            return sorted(kernels, key=lambda x: abs(x["v2_time_s"] - x["v1_time_s"]), reverse=True)[:5]

        moe_analysis = analyze_domain(["moe", "expert", "fused_moe"])
        attention_analysis = analyze_domain(["attention", "flash", "mla"])

        # 6. Expert Conclusions
        conclusions: List[Dict] = []
        recommendations: List[str] = []

        profile_shows_slower = total_change_pct > 0
        if profile_shows_slower and benchmark_shows_improvement:
            conclusions.append(
                {
                    "type": "PARADOX_EXPLAINED",
                    "title": "Profile shows slower but benchmarks show faster - this is expected!",
                    "explanation": (
                        f"Total profile time increased by {total_change_pct:.1f}%, but actual GPU compute "
                        f"decreased by {abs(compute_change_pct):.1f}%. The extra profile time is from "
                        "overhead/instrumentation, not actual computation."
                    ),
                    "confidence": "high",
                }
            )

        if compute_change_pct < -5:
            conclusions.append(
                {
                    "type": "COMPUTE_IMPROVEMENT",
                    "title": f"GPU compute time reduced by {abs(compute_change_pct):.1f}%",
                    "explanation": (
                        f"Actual kernel execution time decreased from {time1['compute'] / 1e6:.1f}s to "
                        f"{time2['compute'] / 1e6:.1f}s. This is the real performance improvement."
                    ),
                    "confidence": "high",
                }
            )
        elif compute_change_pct > 5:
            conclusions.append(
                {
                    "type": "COMPUTE_REGRESSION",
                    "title": f"GPU compute time increased by {compute_change_pct:.1f}%",
                    "explanation": "Actual kernel execution time increased. Investigate specific kernel changes.",
                    "confidence": "high",
                }
            )

        if sync_change_pct < -10:
            conclusions.append(
                {
                    "type": "SYNC_IMPROVEMENT",
                    "title": f"Synchronization overhead reduced by {abs(sync_change_pct):.1f}%",
                    "explanation": "Less time spent waiting for GPU operations to complete. This improves parallelism and throughput.",
                    "confidence": "high",
                }
            )

        if total_launches_v1 > 0 and total_launches_v2 < total_launches_v1 * 0.9:
            launch_reduction = (1 - total_launches_v2 / total_launches_v1) * 100
            conclusions.append(
                {
                    "type": "KERNEL_FUSION",
                    "title": f"Kernel launches reduced by {launch_reduction:.1f}%",
                    "explanation": (
                        f"Fewer kernel launches ({total_launches_v1:,} -> {total_launches_v2:,}) indicates "
                        "kernel fusion. This reduces launch overhead and improves GPU utilization."
                    ),
                    "confidence": "high",
                }
            )

        for fusion in fusion_detected:
            conclusions.append(
                {
                    "type": "FUSION_DETECTED",
                    "title": fusion["type"],
                    "explanation": fusion["description"],
                    "details": fusion,
                    "confidence": "medium",
                }
            )

        if compute_change_pct < 0 and benchmark_shows_improvement:
            recommendations.append("The performance improvement is real - GPU compute time decreased.")
        if fusion_detected:
            recommendations.append("Kernel fusion is a key optimization - new fused kernels replace multiple separate operations.")
        if sync_change_pct < -10:
            recommendations.append("Reduced synchronization enables better GPU parallelism and throughput.")
        if overhead_change_pct > 20:
            recommendations.append("Overhead increased but this doesn't affect end-to-end performance if compute improved.")

        if compute_change_pct < -5 or (sync_change_pct < -10 and total_launches_v2 < total_launches_v1 * 0.95):
            overall_verdict = "GENUINE_IMPROVEMENT"
            verdict_explanation = f"{mv2} shows genuine performance improvements through reduced compute time, better parallelism, and kernel fusion."
        elif compute_change_pct > 5:
            overall_verdict = "GENUINE_REGRESSION"
            verdict_explanation = f"{mv2} shows genuine performance regression in GPU compute time."
        else:
            overall_verdict = "SIMILAR_PERFORMANCE"
            verdict_explanation = "Both versions have similar actual compute performance."

        return {
            "status": "success",
            "model": display_name,
            "versions": {"baseline": mv1, "comparison": mv2},
            "overall_verdict": overall_verdict,
            "verdict_explanation": verdict_explanation,
            "time_breakdown": {
                "v1": {k: round(v / 1e6, 2) for k, v in time1.items()},
                "v2": {k: round(v / 1e6, 2) for k, v in time2.items()},
                "changes": {
                    "compute_pct": round(compute_change_pct, 1),
                    "overhead_pct": round(overhead_change_pct, 1),
                    "sync_pct": round(sync_change_pct, 1),
                    "total_pct": round(total_change_pct, 1),
                },
            },
            "synchronization_analysis": sync_ops,
            "kernel_launch_analysis": {
                "total_launches_v1": total_launches_v1,
                "total_launches_v2": total_launches_v2,
                "change_pct": round(_pct(total_launches_v1 or 1, total_launches_v2), 1),
                "by_type": launch_ops,
            },
            "fusion_detection": fusion_detected,
            "domain_analysis": {
                "moe_kernels": moe_analysis,
                "attention_kernels": attention_analysis,
            },
            "new_kernels_significant": [
                {"name": n[:60], "time_s": round(d.get("total_dur", 0) / 1e6, 2)}
                for n, d in sorted(only_v2.items(), key=lambda x: x[1].get("total_dur", 0), reverse=True)[:5]
            ],
            "removed_kernels_significant": [
                {"name": n[:60], "time_s": round(d.get("total_dur", 0) / 1e6, 2)}
                for n, d in sorted(only_v1.items(), key=lambda x: x[1].get("total_dur", 0), reverse=True)[:5]
            ],
            "conclusions": conclusions,
            "recommendations": recommendations,
            "message": f"Deep analysis complete. {overall_verdict}: {verdict_explanation}",
        }

    except Exception as exc:
        logger.error(f"Error in performance insights analysis: {exc}")
        return {"status": "error", "message": f"Failed to analyze: {exc}"}


async def list_available_profiles(model: Optional[str] = None) -> Dict[str, Any]:
    """List all available PyTorch profile traces.
    
    Discover what profile data is available for analysis. Returns information
    about available models, versions, ranks, and profile metadata.
    
    TOOL_NAME=list_available_profiles
    DISPLAY_NAME=List Available Profiles
    USECASE=Discover what PyTorch profile traces are available for analysis. Use this first to see which models and versions have profile data.
    INSTRUCTIONS=Call without arguments to see all available profiles across all models, or specify a model to filter
    INPUT_DESCRIPTION=model (str, optional): Filter by model name - "deepseek" or "gpt-oss". If not specified, shows all models.
    OUTPUT_DESCRIPTION=Dictionary with available models, versions, ranks, and file information
    EXAMPLES=list_available_profiles(), list_available_profiles(model="gpt-oss")
    PREREQUISITES=None
    RELATED_TOOLS=analyze_pytorch_profile, compare_pytorch_profiles
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - models: Information about each available model
        - available_models: List of model names with profiles
    """
    try:
        index = _discover_profiles(force_refresh=True)

        if not index:
            return {
                "status": "success",
                "available_models": [],
                "models": {},
                "message": "No profile data found. Upload traces to S3 or configure local profile directory.",
                "data_source": "S3" if settings.S3_BUCKET else "local",
            }

        # Optionally filter to a single model
        if model is not None:
            available_models = sorted(index.keys())
            model_key = _match_model(model, available_models)
            if model_key is None:
                return {
                    "status": "error",
                    "message": f"Model '{model}' not found. Available: {available_models}",
                }
            models_to_show = {model_key: index[model_key]}
        else:
            models_to_show = index

        models_data: Dict[str, Any] = {}
        available: List[str] = []

        for model_key, versions in sorted(models_to_show.items()):
            available.append(model_key)
            profiles_list: List[Dict] = []

            for version_name, ranks in sorted(versions.items()):
                rank_list = sorted(ranks.keys())
                # Pick the first file to show size info
                first_entry = ranks[rank_list[0]]
                size_mb = first_entry.get("size_bytes", 0) / (1024 * 1024)

                profiles_list.append(
                    {
                        "version": version_name,
                        "ranks_available": rank_list,
                        "rank_count": len(rank_list),
                        "source": first_entry["source"],
                        "example_file": first_entry["filename"],
                        "file_size_approx_mb": round(size_mb, 1),
                    }
                )

            num_ranks = max(
                (len(ranks) for ranks in versions.values()), default=0
            )
            model_info = _get_model_info(model_key, num_ranks)

            models_data[model_key] = {
                "display_name": model_info["display_name"],
                "model_id": model_info["model_id"],
                "available_versions": sorted(versions.keys()),
                "profiles": profiles_list,
                "profile_info": {
                    "gpus": model_info["gpus"],
                    "tensor_parallelism": model_info["tensor_parallelism"],
                },
            }

        total_versions = sum(len(m["available_versions"]) for m in models_data.values())
        data_source = "S3" if settings.S3_BUCKET else "local"

        return {
            "status": "success",
            "available_models": available,
            "models": models_data,
            "data_source": data_source,
            "message": (
                f"Found profiles for {len(available)} model(s) with "
                f"{total_versions} version(s) total (source: {data_source})"
            ),
            "note": (
                "Use model parameter in analyze_pytorch_profile() and "
                "compare_pytorch_profiles() to specify which model to analyze."
            ),
        }

    except Exception as exc:
        logger.error(f"Error listing profiles: {exc}")
        return {"status": "error", "message": f"Failed to list profiles: {exc}"}


async def check_profile_status(model: Optional[str] = None) -> Dict[str, Any]:
    """Quick diagnostic check of profile data availability.
    
    Returns immediately with information about available profiles and data
    sources. Use this to diagnose issues before running comparisons.
    
    TOOL_NAME=check_profile_status
    DISPLAY_NAME=Check Profile Status
    USECASE=Quick diagnostic check to see if profile data is available. Use this first to verify setup before running expensive comparisons.
    INSTRUCTIONS=Call without arguments to check all models, or specify a model name
    INPUT_DESCRIPTION=model (str, optional): Model to check - "deepseek" or "gpt-oss". If not specified, checks all models.
    OUTPUT_DESCRIPTION=Dictionary with profile availability for each model
    EXAMPLES=check_profile_status(), check_profile_status(model="gpt-oss")
    PREREQUISITES=None
    RELATED_TOOLS=list_available_profiles, analyze_pytorch_profile
    
    Returns:
        Dictionary with diagnostic information.
    """
    try:
        index = _discover_profiles(force_refresh=True)

        if not index:
            return {
                "status": "success",
                "models": {},
                "all_models_ready": False,
                "available_models": [],
                "data_source": "none",
                "message": "No profile data found. Upload traces to S3 or configure local profile directory.",
                "s3_configured": bool(settings.S3_BUCKET),
                "s3_prefix": getattr(settings, "PROFILE_S3_PREFIX", "profiles/rhaiis"),
                "local_path": LOCAL_PROFILE_BASE,
            }

        # Optionally filter
        if model is not None:
            available_models = sorted(index.keys())
            model_key = _match_model(model, available_models)
            if model_key is None:
                return {
                    "status": "error",
                    "message": f"Model '{model}' not found. Available: {available_models}",
                }
            models_to_check = {model_key: index[model_key]}
        else:
            models_to_check = index

        models_status: Dict[str, Any] = {}
        all_ready = True

        for mkey, versions in sorted(models_to_check.items()):
            version_status: Dict[str, Any] = {}
            for vname, ranks in sorted(versions.items()):
                rank_list = sorted(ranks.keys())
                first_entry = ranks[rank_list[0]]
                version_status[vname] = {
                    "ranks_available": rank_list,
                    "rank_count": len(rank_list),
                    "source": first_entry["source"],
                    "file_size_mb": round(first_entry.get("size_bytes", 0) / (1024 * 1024), 1),
                    "ready": True,
                }

            models_status[mkey] = {
                "display_name": _get_display_name(mkey),
                "versions": version_status,
                "status": "ready",
                "ready_for_comparison": len(version_status) >= 2,
            }

            if len(version_status) < 2:
                all_ready = False

        data_source = "S3" if settings.S3_BUCKET else "local"

        return {
            "status": "success",
            "models": models_status,
            "all_models_ready": all_ready,
            "available_models": sorted(models_to_check.keys()),
            "data_source": data_source,
            "s3_configured": bool(settings.S3_BUCKET),
            "message": (
                f"All models ready for comparison (source: {data_source})"
                if all_ready
                else f"Some models need more versions for comparison (source: {data_source})"
            ),
        }

    except Exception as exc:
        logger.error(f"Error checking profile status: {exc}")
        return {"status": "error", "message": f"Failed to check status: {exc}"}
