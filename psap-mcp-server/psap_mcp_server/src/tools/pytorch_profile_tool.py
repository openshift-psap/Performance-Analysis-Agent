"""MCP tool for analyzing PyTorch profiler traces from vLLM benchmark runs.

This tool dynamically discovers Chrome trace JSON files from either S3
(``s3://<bucket>/<PROFILE_S3_PREFIX>/<accelerator>/<model>/<version>/``)
or a local directory fallback, extracts kernel statistics, and enables
comparison between different vLLM versions to identify performance
regressions and improvements.

New profiles are auto-discovered -- just upload trace files to S3 under
the expected folder structure and the agent will find them.
"""

import json
import os
import re
import statistics
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
    },
    "gpt-oss": {
        "display_name": "GPT-OSS",
        "model_id": "gpt-oss",
    },
    "gpt-oss-120b": {
        "display_name": "GPT-OSS-120B",
        "model_id": "openai/gpt-oss-120b",
    },
}

# Aliases map commonly-used names to the canonical S3/local folder names.
# Only needed for names where substring matching wouldn't work (e.g., "deepseek" → "deepseek-r1").
# New models uploaded to S3 are discovered automatically via substring matching in _match_all_models.
_MODEL_ALIASES: Dict[str, str] = {
    "deepseek": "deepseek-r1",
    "deepseek-r1-0528": "deepseek-r1",
    "gpt-oss": "gpt-oss-120b",
    "gptoss": "gpt-oss-120b",
    "gpt_oss": "gpt-oss-120b",
    "gpt-oss-120b": "gpt-oss-120b",
    "gptoss120b": "gpt-oss-120b",
    "gpt_oss_120b": "gpt-oss-120b",
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

# Functional pipeline grouping: maps high-level execution pipelines to kernel
# name patterns.  Used by analyze_performance_insights to produce a
# category-level breakdown comparable to what a human expert would write.
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
      trace_rank0_pid467_range2000-2010.json        -> rank 0  (explicit rank)
      trace_rank3_pid458_range2000-2010_v0112.json  -> rank 3
      rank7.json                                     -> rank 7
      trace_1050_1060_0_20260225_002159.json         -> rank 0  (3rd underscore-delimited segment)

    Returns None when no rank can be extracted — the caller should assign an
    auto-incrementing rank for such files.
    """
    match = re.search(r"rank(\d+)", filename)
    if match:
        return int(match.group(1))

    # Fallback: trace_<start>_<end>_<rank>_<date>_<time>.json
    parts = filename.replace(".json", "").split("_")
    if len(parts) >= 4 and parts[0] == "trace":
        try:
            return int(parts[3])
        except (ValueError, IndexError):
            pass

    return None


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


def _bare_model(composite_key: str) -> str:
    """Extract the bare model name from a composite ``accelerator/model`` key."""
    return composite_key.split("/", 1)[1] if "/" in composite_key else composite_key


def _key_accelerator(composite_key: str) -> str:
    """Extract the accelerator from a composite ``accelerator/model`` key."""
    return composite_key.split("/", 1)[0] if "/" in composite_key else "unknown"


def _match_all_models(user_model: Optional[str], available_models: List[str]) -> List[str]:
    """Fuzzy-match a user-provided model name against discovered composite keys.

    Composite keys have the form ``accelerator/model`` (e.g. ``H200/deepseek-r1``).
    The user may provide just the model part, just the accelerator, or both.

    Returns **all** matching composite keys (empty list if no match).
    """
    if not available_models:
        return []

    if user_model is None:
        return available_models

    lower = user_model.lower().strip()

    # Direct case-insensitive match on full composite key
    for key in available_models:
        if key.lower() == lower:
            return [key]

    # Alias: resolve bare model aliases, then collect all matching keys
    alias = _MODEL_ALIASES.get(lower)
    if alias:
        alias_lower = alias.lower()
        matches = [k for k in available_models if _bare_model(k).lower() == alias_lower]
        if matches:
            return matches

    # Substring match against the full key, bare model part, and known metadata
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
    """Convenience wrapper: returns the single match or first match."""
    matches = _match_all_models(user_model, available_models)
    return matches[0] if matches else None


def _get_display_name(model_key: str) -> str:
    """Human-friendly display name for a composite model key."""
    bare = _bare_model(model_key)
    name = KNOWN_MODELS.get(bare, {}).get("display_name", bare)
    accel = _key_accelerator(model_key)
    return f"{name} ({accel})" if accel != "unknown" else name


def _get_model_info(model_key: str, num_ranks: int) -> Dict[str, Any]:
    """Build a model info dict from known metadata + discovered data."""
    bare = _bare_model(model_key)
    accel = _key_accelerator(model_key)
    known = KNOWN_MODELS.get(bare, {})
    return {
        "display_name": known.get("display_name", bare),
        "model_id": known.get("model_id", bare),
        "gpus": accel,
        "tensor_parallelism": num_ranks,
        "num_ranks": num_ranks,
    }


# ===================================================================== #
#  Profile discovery (S3 + local)                                        #
# ===================================================================== #

def _discover_profiles_s3() -> Optional[Dict]:
    """Discover available profiles from S3 by listing directories.

    Walks ``s3://<bucket>/<prefix>/<accelerator>/<model>/<version>/`` and
    returns::

        {"accelerator/model": {version: {rank: {"source": "s3", "key": ..., ...}}}}

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

    def _list_prefixes(parent_prefix: str) -> List[str]:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=parent_prefix, Delimiter="/")
        return [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]

    def _scan_version_traces(version_prefix: str, composite_key: str) -> None:
        """List trace files under a version prefix and add them to the index.

        Any .json file in the version directory is treated as a profiler trace.
        If the filename encodes a rank (e.g. rank0, rank3), that rank is used.
        Otherwise, an auto-incrementing rank is assigned so that files with any
        naming convention are indexed.
        """
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
                if not filename.endswith(".json"):
                    continue

                rank = _parse_rank_from_filename(filename)
                if rank is None:
                    rank = auto_rank
                    auto_rank += 1

                version_dict = index.setdefault(composite_key, {}).setdefault(
                    version_prefix.rstrip("/").split("/")[-1], {}
                )
                # Avoid collisions: if rank already taken, bump auto_rank
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
        # 1. List accelerator folders  (e.g. H200/, MI300X/)
        accel_prefixes = _list_prefixes(prefix)

        for accel_prefix in accel_prefixes:
            accelerator = accel_prefix.rstrip("/").split("/")[-1]

            # 2. List model folders  (e.g. deepseek-r1/, gpt-oss/)
            model_prefixes = _list_prefixes(accel_prefix)

            for model_prefix in model_prefixes:
                model_name = model_prefix.rstrip("/").split("/")[-1]
                composite_key = f"{accelerator}/{model_name}"

                # 3. List version folders  (e.g. rhaiis-3.2.5/, vLLM-0.13.0/)
                version_prefixes = _list_prefixes(model_prefix)

                for version_prefix in version_prefixes:
                    _scan_version_traces(version_prefix, composite_key)

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

    Scans ``base_dir/<accelerator>/<model>/<version>/*.json`` and indexes
    all JSON files as profiler traces.

    Returns::

        {"accelerator/model": {version: {rank: {"source": "local", "path": Path, ...}}}}
    """
    base = Path(base_dir or LOCAL_PROFILE_BASE)
    index: Dict[str, Dict[str, Dict[int, Dict]]] = {}

    if not base.exists():
        logger.info(f"Local profile base not found: {base}")
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

                for f in sorted(version_dir.rglob("*.json")):
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
            "Upload traces to s3://<bucket>/<PROFILE_S3_PREFIX>/<accelerator>/<model>/<version>/"
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
#  Trace-structure analysis (block segmentation, streams, overhead)       #
# ===================================================================== #

def _extract_raw_events(trace: dict) -> List[Dict[str, Any]]:
    """Extract duration events from a Chrome trace, preserving temporal/stream info.

    Returns a list of dicts sorted by start time (ts), each with:
    name, cat, ts (start µs), dur (µs), end (µs), tid (stream/thread id), pid.
    Only includes complete duration events (ph == "X") with dur > 0.
    """
    events = []
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") == "X" and ev.get("dur", 0) > 0:
            ts = ev["ts"]
            dur = ev["dur"]
            events.append({
                "name": ev.get("name", "unknown"),
                "cat": ev.get("cat", ""),
                "ts": ts,
                "dur": dur,
                "end": ts + dur,
                "tid": ev.get("tid", 0),
                "pid": ev.get("pid", 0),
            })
    events.sort(key=lambda e: e["ts"])
    return events


def _classify_kernel(name: str) -> str:
    """Classify a kernel name into a functional pipeline using FUNCTIONAL_PIPELINES."""
    nl = name.lower()
    for pipe_name, pipe_cfg in FUNCTIONAL_PIPELINES.items():
        if any(pat in nl for pat in pipe_cfg["patterns"]):
            return pipe_name
    return "other"


def _build_kernel_signature(events: List[Dict], cat_filter: str = "kernel") -> List[str]:
    """Build an ordered sequence of pipeline labels from kernel events.

    Only considers events whose ``cat`` matches *cat_filter*. Each kernel is
    mapped to its pipeline label via ``_classify_kernel``.  This produces a
    compact "signature" string per kernel that is used for block-boundary
    detection (e.g. ["attention", "moe_execution", "communication", ...]).
    """
    return [_classify_kernel(ev["name"]) for ev in events if ev.get("cat") == cat_filter]


def _detect_blocks(events: List[Dict], min_kernels: int = 20) -> List[Dict[str, Any]]:
    """Detect repeating transformer block boundaries in a kernel event stream.

    Strategy:
    1. Extract only GPU kernel events (cat == "kernel"), preserving order.
    2. Build a pipeline-label signature for each kernel.
    3. Use a sliding-window approach: find the most common *anchor pattern*
       (the first few pipeline labels of a block) and use it to split the
       kernel stream into blocks.
    4. For each block, record start/end timestamps and constituent events.

    Returns a list of block dicts, each with:
      block_index, start_ts, end_ts, wall_time_us, gpu_active_us,
      kernel_count, events (list), pipeline_sequence (list of labels).
    """
    kernels = [ev for ev in events if ev.get("cat") == "kernel"]
    if len(kernels) < min_kernels:
        return []

    labels = [_classify_kernel(k["name"]) for k in kernels]

    # Find a repeating anchor: look for the most common 3-label prefix
    # that repeats at roughly regular intervals.
    anchor_len = 3
    if len(labels) < anchor_len * 2:
        return []

    prefix_counts: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(labels) - anchor_len + 1):
        key = "|".join(labels[i:i + anchor_len])
        prefix_counts[key].append(i)

    # Pick the anchor that appears most often and has reasonable spacing.
    best_anchor = None
    best_count = 0
    for key, positions in prefix_counts.items():
        if len(positions) > best_count and len(positions) >= 3:
            # Check that the median gap between occurrences is ≥ min_kernels
            gaps = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
            if gaps and statistics.median(gaps) >= min_kernels:
                best_anchor = key
                best_count = len(positions)

    if best_anchor is None:
        return []

    anchor_positions = prefix_counts[best_anchor]

    blocks: List[Dict[str, Any]] = []
    for idx, start_pos in enumerate(anchor_positions):
        end_pos = (anchor_positions[idx + 1] if idx + 1 < len(anchor_positions) else len(kernels))
        block_kernels = kernels[start_pos:end_pos]
        if not block_kernels:
            continue

        block_start = block_kernels[0]["ts"]
        block_end = max(k["end"] for k in block_kernels)
        gpu_active = sum(k["dur"] for k in block_kernels)

        blocks.append({
            "block_index": idx,
            "start_ts": block_start,
            "end_ts": block_end,
            "wall_time_us": block_end - block_start,
            "gpu_active_us": gpu_active,
            "idle_us": (block_end - block_start) - gpu_active,
            "kernel_count": len(block_kernels),
            "events": block_kernels,
            "pipeline_sequence": [_classify_kernel(k["name"]) for k in block_kernels],
        })

    return blocks


def _select_median_block(blocks: List[Dict]) -> Optional[Dict]:
    """Select the block closest to median wall time among all detected blocks.

    Skips the first and last blocks (potential warmup/cooldown).  If fewer
    than 3 blocks remain after trimming, uses all blocks.
    """
    if not blocks:
        return None

    candidates = blocks[1:-1] if len(blocks) > 3 else blocks
    if not candidates:
        return blocks[0]

    wall_times = [b["wall_time_us"] for b in candidates]
    median_wt = statistics.median(wall_times)

    return min(candidates, key=lambda b: abs(b["wall_time_us"] - median_wt))


def _analyze_streams(events: List[Dict]) -> Dict[str, Any]:
    """Analyse GPU stream utilization and overlap within a set of events.

    Groups events by ``tid`` (stream ID) and computes:
    - Per-stream: total active time, event count, time range
    - Cross-stream overlap (time intervals where ≥2 streams are active)
    - Wall-clock span and critical-path estimate
    """
    if not events:
        return {"streams": {}, "wall_time_us": 0, "overlap_us": 0, "critical_path_us": 0}

    by_stream: Dict[int, List[Dict]] = defaultdict(list)
    for ev in events:
        by_stream[ev["tid"]].append(ev)

    stream_summaries: Dict[str, Dict] = {}
    for tid, stream_events in sorted(by_stream.items()):
        active = sum(e["dur"] for e in stream_events)
        stream_summaries[str(tid)] = {
            "event_count": len(stream_events),
            "active_time_us": active,
            "start_ts": min(e["ts"] for e in stream_events),
            "end_ts": max(e["end"] for e in stream_events),
        }

    global_start = min(ev["ts"] for ev in events)
    global_end = max(ev["end"] for ev in events)
    wall_time = global_end - global_start

    # Compute overlap: merge all event intervals per stream, then find
    # total time where ≥2 streams have concurrent activity.
    # Use a sweep-line approach for efficiency.
    timeline_events: List[Tuple[float, int]] = []  # (time, +1 for start / -1 for end)
    for ev in events:
        timeline_events.append((ev["ts"], 1))
        timeline_events.append((ev["end"], -1))
    timeline_events.sort(key=lambda x: (x[0], x[1]))

    overlap_us = 0.0
    active_count = 0
    prev_time = global_start
    for t, delta in timeline_events:
        if active_count >= 2 and t > prev_time:
            overlap_us += t - prev_time
        active_count += delta
        prev_time = t

    # Critical path: wall time minus overlap gives the serialised path
    # through the streams.  This is a rough upper bound.
    critical_path = wall_time - overlap_us if overlap_us < wall_time else wall_time

    return {
        "streams": stream_summaries,
        "stream_count": len(by_stream),
        "wall_time_us": wall_time,
        "total_active_us": sum(s["active_time_us"] for s in stream_summaries.values()),
        "overlap_us": round(overlap_us),
        "critical_path_us": round(critical_path),
    }


def _build_ordered_pipeline_breakdown(events: List[Dict]) -> List[Dict[str, Any]]:
    """Build an ordered pipeline breakdown preserving execution sequence.

    Instead of just totals-per-pipeline, this groups *consecutive* kernel
    events that belong to the same pipeline into *segments*, preserving
    the order in which pipelines execute within a block.

    Returns a list of segments, each with:
      pipeline, description, start_ts, end_ts, wall_time_us,
      gpu_time_us, kernel_count, kernels (list of per-kernel summaries).
    """
    kernels = [ev for ev in events if ev.get("cat") == "kernel"]
    if not kernels:
        return []

    segments: List[Dict[str, Any]] = []
    current_pipe = None
    current_kernels: List[Dict] = []

    def _flush():
        if not current_kernels:
            return
        desc = FUNCTIONAL_PIPELINES.get(current_pipe, {}).get("description", "Other kernels")
        seg_start = current_kernels[0]["ts"]
        seg_end = max(k["end"] for k in current_kernels)
        gpu_time = sum(k["dur"] for k in current_kernels)

        kernel_summaries = []
        for k in current_kernels:
            kernel_summaries.append({
                "name": k["name"][:80],
                "dur_us": k["dur"],
                "stream": k["tid"],
            })

        segments.append({
            "pipeline": current_pipe,
            "description": desc,
            "start_ts": seg_start,
            "end_ts": seg_end,
            "wall_time_us": seg_end - seg_start,
            "gpu_time_us": gpu_time,
            "kernel_count": len(current_kernels),
            "kernels": kernel_summaries,
        })

    for k in kernels:
        pipe = _classify_kernel(k["name"])
        if pipe != current_pipe:
            _flush()
            current_pipe = pipe
            current_kernels = [k]
        else:
            current_kernels.append(k)
    _flush()

    return segments


def _compute_overhead(blocks: List[Dict], all_events: List[Dict]) -> Dict[str, Any]:
    """Compute inter-block and intra-block overhead metrics.

    - Inter-block gap: wall time between consecutive blocks (CPU dispatch,
      scheduling, memory allocation overhead).
    - Intra-block idle: per-block wall time minus GPU active time (stream
      idle gaps, sync waits within a block).
    """
    if not blocks:
        return {"inter_block_gaps": [], "avg_inter_block_us": 0, "avg_intra_block_idle_us": 0}

    inter_gaps: List[Dict] = []
    for i in range(len(blocks) - 1):
        gap = blocks[i + 1]["start_ts"] - blocks[i]["end_ts"]
        inter_gaps.append({
            "between": f"block_{blocks[i]['block_index']}_to_{blocks[i + 1]['block_index']}",
            "gap_us": max(0, gap),
        })

    intra_idles = [b.get("idle_us", 0) for b in blocks]

    gap_values = [g["gap_us"] for g in inter_gaps]
    avg_inter = statistics.mean(gap_values) if gap_values else 0
    median_inter = statistics.median(gap_values) if gap_values else 0
    avg_intra = statistics.mean(intra_idles) if intra_idles else 0

    total_wall = 0.0
    if len(blocks) >= 2:
        total_wall = blocks[-1]["end_ts"] - blocks[0]["start_ts"]
    total_gpu = sum(b["gpu_active_us"] for b in blocks)
    total_inter = sum(gap_values)
    total_intra = sum(intra_idles)

    return {
        "block_count": len(blocks),
        "total_wall_time_us": round(total_wall),
        "total_gpu_active_us": round(total_gpu),
        "total_inter_block_gap_us": round(total_inter),
        "total_intra_block_idle_us": round(total_intra),
        "avg_inter_block_gap_us": round(avg_inter),
        "median_inter_block_gap_us": round(median_inter),
        "avg_intra_block_idle_us": round(avg_intra),
        "inter_block_gaps": inter_gaps[:20],
        "overhead_pct": round((total_inter + total_intra) / total_wall * 100, 1) if total_wall > 0 else 0,
    }


def _build_structured_root_causes(
    pipeline_breakdown_v1: List[Dict],
    pipeline_breakdown_v2: List[Dict],
    stats1: Dict[str, Dict],
    stats2: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    """Build structured root-cause objects from two pipeline breakdowns.

    For each pipeline where the delta is significant (>5% or >1ms), produces
    a root-cause dict with per-kernel evidence, suggested source files, and
    investigation hints.
    """
    from psap_mcp_server.src.tools.kernel_code_mapper_tool import KERNEL_MAPPINGS

    # Build pipeline -> segment map for each version
    def _pipe_total(segments: List[Dict]) -> Dict[str, Dict]:
        totals: Dict[str, Dict] = defaultdict(lambda: {"gpu_us": 0, "kernels": []})
        for seg in segments:
            pipe = seg["pipeline"]
            totals[pipe]["gpu_us"] += seg["gpu_time_us"]
            for k in seg.get("kernels", []):
                totals[pipe]["kernels"].append(k)
        return dict(totals)

    v1_pipes = _pipe_total(pipeline_breakdown_v1)
    v2_pipes = _pipe_total(pipeline_breakdown_v2)

    all_pipes = set(v1_pipes.keys()) | set(v2_pipes.keys())
    root_causes: List[Dict[str, Any]] = []

    for pipe in sorted(all_pipes):
        v1 = v1_pipes.get(pipe, {"gpu_us": 0, "kernels": []})
        v2 = v2_pipes.get(pipe, {"gpu_us": 0, "kernels": []})

        delta_us = v2["gpu_us"] - v1["gpu_us"]
        delta_pct = ((delta_us / v1["gpu_us"]) * 100) if v1["gpu_us"] > 0 else (100.0 if v2["gpu_us"] > 0 else 0.0)

        if abs(delta_us) < 1000 and abs(delta_pct) < 5:
            continue

        # Aggregate kernels per name for each version
        def _agg_kernels(kernel_list: List[Dict]) -> List[Dict]:
            by_name: Dict[str, Dict] = defaultdict(lambda: {"total_us": 0, "count": 0})
            for k in kernel_list:
                by_name[k["name"]]["total_us"] += k["dur_us"]
                by_name[k["name"]]["count"] += 1
            return sorted(
                [{"name": n, "total_us": d["total_us"], "count": d["count"]} for n, d in by_name.items()],
                key=lambda x: x["total_us"],
                reverse=True,
            )[:10]

        v1_kernels = _agg_kernels(v1["kernels"])
        v2_kernels = _agg_kernels(v2["kernels"])

        # Determine investigation hint
        v1_names = {k["name"] for k in v1["kernels"]}
        v2_names = {k["name"] for k in v2["kernels"]}
        new_names = v2_names - v1_names
        removed_names = v1_names - v2_names
        v1_total_calls = sum(k.get("count", 1) for k in v1_kernels)
        v2_total_calls = sum(k.get("count", 1) for k in v2_kernels)

        hints = []
        if new_names and removed_names:
            hints.append(f"Kernel replacement detected: {len(removed_names)} removed, {len(new_names)} new")
        if v2_total_calls < v1_total_calls * 0.8:
            hints.append(f"Kernel call count reduced ({v1_total_calls} -> {v2_total_calls}), suggesting fusion")
        if v2_total_calls > v1_total_calls * 1.2:
            hints.append(f"Kernel call count increased ({v1_total_calls} -> {v2_total_calls}), possible decomposition")
        if delta_us < 0:
            hints.append(f"Pipeline improved by {_format_duration(abs(delta_us))}")
        else:
            hints.append(f"Pipeline regressed by {_format_duration(delta_us)}")

        # Suggest source files based on kernel names in this pipeline
        suggested_files = set()
        all_kernel_names = v1_names | v2_names
        for kname in all_kernel_names:
            for pattern, mappings in KERNEL_MAPPINGS:
                if re.search(pattern, kname, re.IGNORECASE):
                    for m in mappings:
                        if isinstance(m, dict):
                            suggested_files.add(m.get("path", ""))
                        elif isinstance(m, (list, tuple)) and len(m) >= 1:
                            suggested_files.add(m[0] if isinstance(m[0], str) else str(m[0]))
                    break
        suggested_files.discard("")

        direction = "improvement" if delta_us < 0 else "regression"
        desc = FUNCTIONAL_PIPELINES.get(pipe, {}).get("description", pipe)

        root_causes.append({
            "root_cause_id": f"{pipe}_{direction}",
            "pipeline": pipe,
            "description": desc,
            "direction": direction,
            "evidence": {
                "v1_gpu_us": round(v1["gpu_us"]),
                "v2_gpu_us": round(v2["gpu_us"]),
                "delta_us": round(delta_us),
                "delta_pct": round(delta_pct, 1),
                "v1_kernels": v1_kernels,
                "v2_kernels": v2_kernels,
            },
            "suggested_source_files": sorted(suggested_files)[:5],
            "investigation_hints": hints,
        })

    root_causes.sort(key=lambda rc: abs(rc["evidence"]["delta_us"]), reverse=True)
    return root_causes


# ---------------------------------------------------------------------------
# Raw-event loading (parallel to stats loading, preserves temporal info)
# ---------------------------------------------------------------------------

_raw_events_cache: Dict[str, Tuple[float, List[Dict]]] = {}
_RAW_EVENTS_CACHE_TTL: int = 300


def _load_raw_events(
    model: str,
    version: str,
    rank: int,
) -> Optional[List[Dict]]:
    """Load raw Chrome trace events for a model/version/rank.

    Checks the in-memory cache first.  Falls back to loading the trace
    JSON from S3 or local storage.
    """
    cache_key = f"raw:{model}/{version}/rank{rank}"
    if cache_key in _raw_events_cache:
        ts, evts = _raw_events_cache[cache_key]
        if (_time.time() - ts) < _RAW_EVENTS_CACHE_TTL:
            return evts
        del _raw_events_cache[cache_key]

    index = _discover_profiles()
    entry = index.get(model, {}).get(version, {}).get(rank)
    if entry is None:
        return None

    if entry["source"] == "s3":
        trace = _load_trace_from_s3(entry["key"])
    else:
        trace = _load_trace_from_local(entry["path"])

    if not trace:
        return None

    evts = _extract_raw_events(trace)
    _raw_events_cache[cache_key] = (_time.time(), evts)
    return evts


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
    matches = _match_all_models(model, available_models)

    if not matches:
        return None, None, (
            f"Model '{model}' not found. Available models: {available_models}"
        ), index

    if len(matches) > 1:
        formatted = ", ".join(matches)
        return None, None, (
            f"Multiple models match '{model}': {formatted}. "
            f"Please specify the accelerator, e.g. '{matches[0]}'"
        ), index

    model_key = matches[0]

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
                    f"Please specify the accelerator, e.g. '{matches[0]}'"
                ),
            }
        model_key = matches[0]

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

                count1 = s1.get("count", 0)
                count2 = s2.get("count", 0)
                avg1 = s1.get("avg_dur", 0)
                avg2 = s2.get("avg_dur", 0)
                count_change_pct = ((count2 - count1) / count1 * 100) if count1 > 0 else (100.0 if count2 > 0 else 0.0)
                avg_change_pct = ((avg2 - avg1) / avg1 * 100) if avg1 > 0 else (100.0 if avg2 > 0 else 0.0)

                diffs.append(
                    {
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

        # Pipeline breakdown (reuse FUNCTIONAL_PIPELINES from module level)
        def _pipeline_summary(s1: Dict, s2: Dict) -> List[Dict]:
            kernel_names = {n for n, d in s1.items() if d.get("cat") == "kernel"} | \
                           {n for n, d in s2.items() if d.get("cat") == "kernel"}
            assigned: Dict[str, str] = {}
            for name in kernel_names:
                nl = name.lower()
                for pipe_name, pipe_cfg in FUNCTIONAL_PIPELINES.items():
                    if any(pat in nl for pat in pipe_cfg["patterns"]):
                        assigned[name] = pipe_name
                        break
            buckets: Dict[str, Dict[str, float]] = {}
            for pipe_name in list(FUNCTIONAL_PIPELINES.keys()) + ["other"]:
                buckets[pipe_name] = {"v1": 0.0, "v2": 0.0, "v1_calls": 0, "v2_calls": 0}
            for name in kernel_names:
                pipe = assigned.get(name, "other")
                buckets[pipe]["v1"] += s1.get(name, {}).get("total_dur", 0)
                buckets[pipe]["v2"] += s2.get(name, {}).get("total_dur", 0)
                buckets[pipe]["v1_calls"] += s1.get(name, {}).get("count", 0)
                buckets[pipe]["v2_calls"] += s2.get(name, {}).get("count", 0)
            result = []
            for pn, b in sorted(buckets.items(), key=lambda x: max(x[1]["v1"], x[1]["v2"]), reverse=True):
                if b["v1"] == 0 and b["v2"] == 0:
                    continue
                desc = FUNCTIONAL_PIPELINES.get(pn, {}).get("description", "Other kernels")
                ch = ((b["v2"] - b["v1"]) / b["v1"] * 100) if b["v1"] > 0 else 0.0
                result.append({
                    "pipeline": pn,
                    "description": desc,
                    "v1_time_ms": round(b["v1"] / 1e3, 1),
                    "v2_time_ms": round(b["v2"] / 1e3, 1),
                    "change_pct": round(ch, 1),
                    "delta_ms": round((b["v2"] - b["v1"]) / 1e3, 1),
                })
            return result

        pipeline_breakdown = _pipeline_summary(filtered1, filtered2)

        # Event count summary (total events by category)
        def _event_counts(s: Dict) -> Dict[str, int]:
            counts: Dict[str, int] = defaultdict(int)
            for d in s.values():
                cat = d.get("cat", "other") or "other"
                counts[cat] += d.get("count", 0)
            counts["total"] = sum(counts.values())
            return dict(counts)

        event_counts_v1 = _event_counts(stats1)
        event_counts_v2 = _event_counts(stats2)

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
            "pipeline_breakdown": pipeline_breakdown,
            "event_counts": {
                "v1": event_counts_v1,
                "v2": event_counts_v2,
            },
            "profile_info": {
                "model": model_info["model_id"],
                "gpus": model_info["gpus"],
                "tensor_parallelism": model_info["tensor_parallelism"],
            },
            "next_steps": [
                "Use map_kernel_to_vllm_code to find source files for top regressions",
                f"Use compare_vllm_versions('{mv1}', '{mv2}') to see release notes between versions",
                "Use get_vllm_pull_request to investigate specific PRs mentioned in release notes",
                f"Use fetch_vllm_source to read actual implementation of changed kernels",
                f"Use get_vllm_code_diff('{mv1}', '{mv2}', '<file_path>') to see exact code changes",
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

        # 5b. Functional Pipeline Breakdown
        # Assigns every kernel (with cat=="kernel") to at most one pipeline
        # based on the first matching pattern set, then summarises per-pipeline.
        def _build_pipeline_breakdown(
            s1: Dict[str, Dict], s2: Dict[str, Dict]
        ) -> List[Dict[str, Any]]:
            all_kernel_names = set()
            for n, d in s1.items():
                if d.get("cat") == "kernel":
                    all_kernel_names.add(n)
            for n, d in s2.items():
                if d.get("cat") == "kernel":
                    all_kernel_names.add(n)

            assigned: Dict[str, str] = {}  # kernel_name -> pipeline_name
            for name in all_kernel_names:
                name_lower = name.lower()
                for pipe_name, pipe_cfg in FUNCTIONAL_PIPELINES.items():
                    if any(pat in name_lower for pat in pipe_cfg["patterns"]):
                        assigned[name] = pipe_name
                        break

            pipeline_stats: Dict[str, Dict[str, float]] = {}
            for pipe_name in FUNCTIONAL_PIPELINES:
                pipeline_stats[pipe_name] = {
                    "v1_us": 0.0, "v2_us": 0.0,
                    "v1_calls": 0, "v2_calls": 0,
                    "v1_kernels": 0, "v2_kernels": 0,
                }
            pipeline_stats["other"] = {
                "v1_us": 0.0, "v2_us": 0.0,
                "v1_calls": 0, "v2_calls": 0,
                "v1_kernels": 0, "v2_kernels": 0,
            }

            for name in all_kernel_names:
                pipe = assigned.get(name, "other")
                d1 = s1.get(name, {})
                d2 = s2.get(name, {})
                pipeline_stats[pipe]["v1_us"] += d1.get("total_dur", 0)
                pipeline_stats[pipe]["v2_us"] += d2.get("total_dur", 0)
                pipeline_stats[pipe]["v1_calls"] += d1.get("count", 0)
                pipeline_stats[pipe]["v2_calls"] += d2.get("count", 0)
                if d1.get("total_dur", 0) > 0:
                    pipeline_stats[pipe]["v1_kernels"] += 1
                if d2.get("total_dur", 0) > 0:
                    pipeline_stats[pipe]["v2_kernels"] += 1

            result = []
            for pipe_name, ps in sorted(
                pipeline_stats.items(),
                key=lambda x: max(x[1]["v1_us"], x[1]["v2_us"]),
                reverse=True,
            ):
                if ps["v1_us"] == 0 and ps["v2_us"] == 0:
                    continue
                desc = FUNCTIONAL_PIPELINES.get(pipe_name, {}).get("description", "Other kernels")
                change_pct = _pct(ps["v1_us"] or 1, ps["v2_us"])
                result.append({
                    "pipeline": pipe_name,
                    "description": desc,
                    "v1_time_ms": round(ps["v1_us"] / 1e3, 1),
                    "v2_time_ms": round(ps["v2_us"] / 1e3, 1),
                    "change_pct": round(change_pct, 1),
                    "v1_calls": int(ps["v1_calls"]),
                    "v2_calls": int(ps["v2_calls"]),
                    "v1_unique_kernels": int(ps["v1_kernels"]),
                    "v2_unique_kernels": int(ps["v2_kernels"]),
                    "delta_ms": round((ps["v2_us"] - ps["v1_us"]) / 1e3, 1),
                })
            return result

        pipeline_breakdown = _build_pipeline_breakdown(stats1, stats2)

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
            "pipeline_breakdown": pipeline_breakdown,
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

        # Optionally filter by model name
        if model is not None:
            available_models = sorted(index.keys())
            matches = _match_all_models(model, available_models)
            if not matches:
                return {
                    "status": "error",
                    "message": f"Model '{model}' not found. Available: {available_models}",
                }
            models_to_show = {k: index[k] for k in matches}
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
                "accelerator": _key_accelerator(model_key),
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
            matches = _match_all_models(model, available_models)
            if not matches:
                return {
                    "status": "error",
                    "message": f"Model '{model}' not found. Available: {available_models}",
                }
            models_to_check = {k: index[k] for k in matches}
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


async def analyze_trace_structure(
    version: str,
    rank: int = 0,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the temporal structure of a PyTorch profiler trace with transformer block segmentation.

    Goes beyond flat kernel aggregation by detecting repeating transformer
    block boundaries, selecting a representative median block, analysing GPU
    stream utilisation and overlap, and computing overhead metrics.

    TOOL_NAME=analyze_trace_structure
    DISPLAY_NAME=Analyze Trace Structure
    USECASE=Perform block-level structural analysis of a PyTorch profiler trace. Use this to understand the temporal structure of inference: how many transformer blocks were captured, what a single representative block looks like (operations in order, per-stream), and where overhead lives (inter-block gaps, intra-block idle).
    INSTRUCTIONS=1. Specify a vLLM version and optionally a model, 2. The tool detects transformer block boundaries automatically, 3. It picks a median-wall-time block as representative, 4. It returns ordered pipeline breakdown, stream analysis, and overhead accounting
    INPUT_DESCRIPTION=version (str): vLLM version like "v0.13.0"; model (str, optional): Model name; rank (int): GPU rank (default 0)
    OUTPUT_DESCRIPTION=Dictionary with block segmentation, median block detail, stream analysis, ordered pipeline breakdown, and overhead metrics
    EXAMPLES=analyze_trace_structure("v0.13.0"), analyze_trace_structure("v0.11.2", model="gpt-oss")
    PREREQUISITES=Profile traces must exist in S3 or local directory
    RELATED_TOOLS=compare_trace_structures, compare_pytorch_profiles, analyze_performance_insights

    Args:
        version: vLLM version (e.g., "v0.13.0")
        rank: GPU rank to analyze (default: 0)
        model: Model name (e.g., "deepseek", "gpt-oss")

    Returns:
        Dictionary with block segmentation, median block, stream analysis,
        ordered pipeline breakdown, and overhead accounting.
    """
    try:
        model_key, matched_version, error, index = _resolve_model_and_version(model, version)
        if error:
            return {"status": "error", "message": error}
        assert model_key is not None and matched_version is not None

        display_name = _get_display_name(model_key)

        raw_events = _load_raw_events(model_key, matched_version, rank)
        if not raw_events:
            return {
                "status": "error",
                "message": f"No trace found for {display_name} {matched_version} rank {rank}",
            }

        # Block segmentation
        blocks = _detect_blocks(raw_events)
        median_block = _select_median_block(blocks)

        # Build results
        block_summary = []
        for b in blocks:
            block_summary.append({
                "block_index": b["block_index"],
                "wall_time_us": b["wall_time_us"],
                "gpu_active_us": b["gpu_active_us"],
                "idle_us": b["idle_us"],
                "kernel_count": b["kernel_count"],
            })

        median_detail = None
        stream_analysis = None
        ordered_pipeline = None
        if median_block:
            median_events = median_block["events"]

            stream_analysis = _analyze_streams(median_events)
            ordered_pipeline = _build_ordered_pipeline_breakdown(median_events)

            # Compact kernel list for the median block
            median_kernels = []
            for k in median_events[:200]:
                median_kernels.append({
                    "name": k["name"][:80],
                    "pipeline": _classify_kernel(k["name"]),
                    "dur_us": k["dur"],
                    "stream": k["tid"],
                })

            median_detail = {
                "block_index": median_block["block_index"],
                "wall_time_us": median_block["wall_time_us"],
                "gpu_active_us": median_block["gpu_active_us"],
                "idle_us": median_block["idle_us"],
                "kernel_count": median_block["kernel_count"],
                "kernels": median_kernels,
            }

        overhead = _compute_overhead(blocks, raw_events)

        return {
            "status": "success",
            "version": matched_version,
            "model": display_name,
            "rank": rank,
            "total_events": len(raw_events),
            "block_segmentation": {
                "blocks_detected": len(blocks),
                "blocks": block_summary,
            },
            "median_block": median_detail,
            "stream_analysis": stream_analysis,
            "ordered_pipeline_breakdown": ordered_pipeline,
            "overhead": overhead,
            "message": (
                f"Analyzed {display_name} {matched_version} rank {rank}: "
                f"{len(blocks)} transformer blocks detected, "
                f"median block has {median_block['kernel_count'] if median_block else 0} kernels"
            ),
        }

    except Exception as exc:
        logger.error(f"Error in trace structure analysis: {exc}")
        return {"status": "error", "message": f"Failed to analyze trace structure: {exc}"}


async def compare_trace_structures(
    version1: str,
    version2: str,
    rank: int = 0,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare trace structure between two vLLM versions at the transformer block level.

    Runs block segmentation on both versions, selects median blocks, then
    compares them: ordered pipeline breakdowns, stream utilisation, overhead
    metrics, and structured root-cause objects.

    TOOL_NAME=compare_trace_structures
    DISPLAY_NAME=Compare Trace Structures
    USECASE=Compare the temporal structure of two vLLM versions at the transformer block level. Use this for deep root-cause analysis: it compares median blocks operation-by-operation (preserving execution order), analyses stream overlap changes, computes overhead deltas, and produces structured root-cause objects with per-kernel evidence and suggested source files.
    INSTRUCTIONS=1. Specify two vLLM versions, 2. Optionally specify model, 3. The tool segments both traces into blocks, picks median blocks, and compares them structurally
    INPUT_DESCRIPTION=version1 (str): Baseline version; version2 (str): Comparison version; model (str, optional): Model name; rank (int): GPU rank (default 0)
    OUTPUT_DESCRIPTION=Dictionary with per-version median block details, ordered pipeline comparison, stream analysis diff, overhead comparison, and structured root causes
    EXAMPLES=compare_trace_structures("v0.11.2", "v0.13.0"), compare_trace_structures("v0.11.2", "v0.13.0", model="gpt-oss")
    PREREQUISITES=Profile traces must exist for both versions
    RELATED_TOOLS=analyze_trace_structure, compare_pytorch_profiles, analyze_performance_insights

    Args:
        version1: Baseline vLLM version
        version2: Comparison vLLM version
        rank: GPU rank (default: 0)
        model: Model name

    Returns:
        Dictionary with structural comparison of median blocks including
        root causes, pipeline breakdowns, stream analysis, and overhead.
    """
    try:
        # Resolve model and both versions
        index = _discover_profiles()
        if not index:
            return {"status": "error", "message": "No profile data available."}

        available_models = sorted(index.keys())
        matches = _match_all_models(model, available_models)
        if not matches:
            return {"status": "error", "message": f"Model '{model}' not found. Available: {available_models}"}
        if len(matches) > 1:
            return {"status": "error", "message": f"Multiple models match '{model}': {', '.join(matches)}. Please specify."}
        model_key = matches[0]
        display_name = _get_display_name(model_key)

        available_versions = sorted(index[model_key].keys())
        mv1 = _match_version(version1, available_versions)
        mv2 = _match_version(version2, available_versions)
        if mv1 is None:
            return {"status": "error", "message": f"Version '{version1}' not found. Available: {available_versions}"}
        if mv2 is None:
            return {"status": "error", "message": f"Version '{version2}' not found. Available: {available_versions}"}

        # Load raw events for both versions
        raw1 = _load_raw_events(model_key, mv1, rank)
        raw2 = _load_raw_events(model_key, mv2, rank)
        if not raw1:
            return {"status": "error", "message": f"No trace found for {display_name} {mv1} rank {rank}"}
        if not raw2:
            return {"status": "error", "message": f"No trace found for {display_name} {mv2} rank {rank}"}

        # Block segmentation + median selection
        blocks1 = _detect_blocks(raw1)
        blocks2 = _detect_blocks(raw2)
        median1 = _select_median_block(blocks1)
        median2 = _select_median_block(blocks2)

        # Analyse each median block
        def _analyze_median(median_block: Optional[Dict]) -> Dict[str, Any]:
            if not median_block:
                return {"available": False}
            evts = median_block["events"]
            return {
                "available": True,
                "wall_time_us": median_block["wall_time_us"],
                "gpu_active_us": median_block["gpu_active_us"],
                "idle_us": median_block["idle_us"],
                "kernel_count": median_block["kernel_count"],
                "stream_analysis": _analyze_streams(evts),
                "ordered_pipeline": _build_ordered_pipeline_breakdown(evts),
            }

        detail1 = _analyze_median(median1)
        detail2 = _analyze_median(median2)

        # Overhead comparison
        overhead1 = _compute_overhead(blocks1, raw1)
        overhead2 = _compute_overhead(blocks2, raw2)

        overhead_comparison = {}
        for key in ("avg_inter_block_gap_us", "median_inter_block_gap_us", "avg_intra_block_idle_us", "overhead_pct"):
            v1_val = overhead1.get(key, 0)
            v2_val = overhead2.get(key, 0)
            delta = v2_val - v1_val
            pct = ((delta / v1_val) * 100) if v1_val else 0
            overhead_comparison[key] = {
                "v1": v1_val,
                "v2": v2_val,
                "delta": round(delta, 1),
                "change_pct": round(pct, 1),
            }

        # Build structured root causes from median block pipelines
        root_causes = []
        if detail1.get("available") and detail2.get("available"):
            root_causes = _build_structured_root_causes(
                detail1["ordered_pipeline"],
                detail2["ordered_pipeline"],
                {}, {},  # stats dicts not needed for pipeline-level root causes
            )

        # Median block wall time comparison
        wt1 = detail1.get("wall_time_us", 0)
        wt2 = detail2.get("wall_time_us", 0)
        wt_delta = wt2 - wt1
        wt_pct = ((wt_delta / wt1) * 100) if wt1 else 0

        return {
            "status": "success",
            "model": display_name,
            "comparison": {"version1": mv1, "version2": mv2, "rank": rank},
            "block_counts": {
                "v1": len(blocks1),
                "v2": len(blocks2),
            },
            "median_block_comparison": {
                "wall_time": {
                    "v1_us": wt1,
                    "v2_us": wt2,
                    "delta_us": round(wt_delta),
                    "change_pct": round(wt_pct, 1),
                    "direction": "slower" if wt_delta > 0 else "faster",
                },
                "v1": detail1,
                "v2": detail2,
            },
            "overhead_comparison": overhead_comparison,
            "overhead_v1": overhead1,
            "overhead_v2": overhead2,
            "root_causes": root_causes,
            "message": (
                f"{display_name}: Median block {mv2} is "
                f"{abs(wt_pct):.1f}% {'slower' if wt_delta > 0 else 'faster'} "
                f"than {mv1} ({_format_duration(abs(wt_delta))} delta). "
                f"Found {len(root_causes)} root cause(s)."
            ),
        }

    except Exception as exc:
        logger.error(f"Error comparing trace structures: {exc}")
        return {"status": "error", "message": f"Failed to compare trace structures: {exc}"}
