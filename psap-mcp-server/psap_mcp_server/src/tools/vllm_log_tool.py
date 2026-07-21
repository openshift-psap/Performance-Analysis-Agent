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

    exact = filtered[filtered["version"].str.lower() == version.lower()]
    if not exact.empty:
        filtered = exact
    else:
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


_NOISE_SUBSTRINGS = frozenset({
    "Loading safetensors", "Capturing CUDA", "AutoTuner", "Autotuning",
    "autotuner", "[Gloo]", "DeepGEMM warmup", "FutureWarning",
    "Registering", "cuda graph addresses", "Route:", "Methods:",
    "Started server process", "Waiting for application startup",
    "Application startup complete", "Available routes",
    "Starting vLLM", "dist-packages/torch/", "return func(*args",
    "Parse safetensors", "it/s]", "deprecated",
    "set OMP_NUM_THREADS", "Reducing Torch parallelism",
})

_NOISE_RE = re.compile(
    r"^\s*$"
    r"|rank=\d+"
    r"|pid=\d+\)\s*$"
    r"|▄|▀|█"
)


def _is_noise(line: str) -> bool:
    """Return True if a log line is noise (progress bars, empty, route listings, etc.)."""
    if _NOISE_RE.search(line):
        return True
    return any(s in line for s in _NOISE_SUBSTRINGS)


def _generic_extract_kv(line: str) -> Dict[str, str]:
    """Extract key=value pairs from a log line that look like configuration.

    Catches patterns like ``key=value``, ``key='value'``,
    ``key: value`` when they appear in structured log output.
    Returns a dict of extracted pairs (may be empty).
    """
    pairs: Dict[str, str] = {}
    for m in re.finditer(r"(\b[a-z][a-z0-9_]{2,})\s*[=:]\s*'?([^\s,;'}{)\]]+)'?", line):
        key, val = m.group(1), m.group(2)
        if key in ("rank", "pid", "port", "size", "local_rank", "world_size",
                    "level", "name", "type", "return", "self"):
            continue
        pairs[key] = val
    return pairs


def _generic_extract_using(line: str) -> Optional[Dict[str, str]]:
    """Extract generic 'Using X [for|backend|out of]' patterns.

    Returns a dict like ``{"component": "...", "selected": "...", "available": "..."}``
    or None.
    """
    m = re.search(
        r"Using\s+(\S+(?:\s+\S+)?)\s+"
        r"(?:for|backend|attention backend|MoE backend)"
        r"(?:\s+out of potential backends:\s*\[([^\]]+)\])?",
        line
    )
    if m:
        result = {"selected": m.group(1).strip("'\".")}
        if m.group(2):
            result["available"] = [b.strip().strip("'\"") for b in m.group(2).split(",")]
        return result

    m = re.search(r"Using\s+'?(\S+?)'?\s+(\S*(?:backend|kernel|method|sampler|strategy))", line, re.IGNORECASE)
    if m:
        return {"selected": m.group(1), "type": m.group(2)}
    return None


def _parse_vllm_log(raw: str) -> Dict[str, Any]:
    """Parse a vLLM server log into structured sections.

    Extracts configuration, compilation, memory, and timing information
    using regex patterns matched against known vLLM log output formats.
    Only the startup portion of the log (up to "Application startup complete")
    is analyzed; request-serving output is discarded.

    **Generic fallback**: Lines not captured by any specific regex pattern
    are processed by heuristic extractors that look for key=value pairs
    and "Using X for Y" patterns.  Any remaining significant INFO/WARNING
    lines are collected in ``unmatched_lines`` so that an LLM agent can
    interpret new patterns introduced in future vLLM versions without
    requiring parser code changes.
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
    matched_lines: set[int] = set()

    for line_idx, line in enumerate(lines):
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
                            "enable_prefix_caching", "trust_remote_code", "port",
                            "tool_call_parser", "reasoning_parser",
                            "kv_cache_dtype", "tensor_parallel_size",
                            "enable_expert_parallel", "enable_auto_tool_choice",
                            "enable_chunked_prefill", "max_cudagraph_capture_size",
                            "mamba_ssm_cache_dtype", "mm_encoder_tp_mode",
                            "max_num_seqs", "max_num_batched_tokens",
                            "async_scheduling"):
                    if key in args:
                        result["server_config"][key] = args[key]
            except Exception:
                result["server_config"]["non_default_args_raw"] = m.group(1)

        # --- Architecture ---
        m = re.search(r"Resolved architecture:\s*(\S+)", line)
        if m:
            result["engine_config"]["architecture"] = m.group(1)

        # --- Max model len ---
        m = re.search(r"Using max model len (\d+)", line)
        if m:
            result["engine_config"]["max_model_len"] = int(m.group(1))

        # --- NIXL ---
        if "NIXL is available" in line and "nixl_available" not in result["engine_config"]:
            result["engine_config"]["nixl_available"] = True

        # --- Async scheduling ---
        if "Asynchronous scheduling is enabled" in line and "async_scheduling" not in result["engine_config"]:
            result["engine_config"]["async_scheduling"] = True

        # --- IR op priority ---
        m = re.search(r"Final IR op priority after setting platform defaults:\s*(.+)", line)
        if m and "ir_op_priority" not in result["compilation"]:
            result["compilation"]["ir_op_priority"] = m.group(1).strip()

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
                ("decode_context_parallel_size", r"decode_context_parallel_size=(\d+)"),
                ("dcp_comm_backend", r"dcp_comm_backend=(\S+?)(?:,|$)"),
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

            # enable_flashinfer_autotune from kernel_config
            em = re.search(r"enable_flashinfer_autotune=(\S+?)(?:,|\))", line)
            if em:
                result["compilation"]["flashinfer_autotune_enabled"] = em.group(1) == "True"

            # moe_backend from kernel_config
            em = re.search(r"moe_backend='(\S+?)'", line)
            if em:
                result["compilation"]["moe_backend_config"] = em.group(1)

            # pass_config fusions from compilation_config
            for fuse_key in ("fuse_norm_quant", "fuse_act_quant", "fuse_attn_quant",
                            "fuse_gemm_comms", "fuse_allreduce_rms", "fuse_act_padding",
                            "fuse_rope_kvcache_cat_mla", "eliminate_noops"):
                em = re.search(rf"'{fuse_key}':\s*(True|False)", line)
                if em:
                    result["compilation"].setdefault("pass_config", {})[fuse_key] = em.group(1) == "True"

            # compile_ranges_endpoints (v0.18+) or compile_ranges_split_points (v0.14.x)
            for cr_key in ("compile_ranges_endpoints", "compile_ranges_split_points"):
                em = re.search(rf"'{cr_key}':\s*\[([^\]]*)\]", line)
                if em and em.group(1).strip():
                    endpoints = [int(x.strip()) for x in em.group(1).split(",") if x.strip()]
                    result["compilation"]["compile_ranges_endpoints"] = endpoints
                    break

            # custom_ops
            em = re.search(r"'custom_ops':\s*\[([^\]]*)\]", line)
            if em:
                ops = [x.strip().strip("'\"") for x in em.group(1).split(",") if x.strip()]
                result["compilation"]["custom_ops"] = ops

            # fast_moe_cold_start
            em = re.search(r"'fast_moe_cold_start':\s*(True|False)", line)
            if em:
                result["compilation"]["fast_moe_cold_start"] = em.group(1) == "True"

            # linear_backend from kernel_config (v0.22+)
            em = re.search(r"linear_backend='(\S+?)'", line)
            if em:
                result["compilation"]["linear_backend_config"] = em.group(1)

            # reasoning_parser from structured_outputs_config
            em = re.search(r"reasoning_parser='(\S*?)'", line)
            if em and em.group(1):
                result["engine_config"]["reasoning_parser"] = em.group(1)

        # --- Chunked prefill ---
        m = re.search(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)", line)
        if m:
            result["engine_config"]["chunked_prefill"] = True
            result["engine_config"]["max_num_batched_tokens"] = int(m.group(1))

        m = re.search(r"Overriding max cuda graph capture size to (\d+)", line)
        if m:
            result["engine_config"]["max_cudagraph_capture_size_override"] = int(m.group(1))

        # --- Selected linear kernel (FP8, CompressedTensors, etc.) ---
        m = re.search(r"Selected (\S+) for (\S+)", line)
        if m and ("LinearMethod" in m.group(2) or "LinearKernel" in m.group(1)):
            result["compilation"]["fp8_linear_kernel"] = m.group(1)
            result["compilation"]["fp8_linear_method"] = m.group(2)

        # --- DeepGEMM ---
        if "DeepGEMM" in line and "enabled" in line:
            m = re.search(r"DeepGEMM (\S+) enabled", line)
            if m:
                result["compilation"]["deepgemm_mode"] = m.group(1)
                result["compilation"]["deepgemm_enabled"] = True

        # --- MoE backend (runtime selection) ---
        m = re.search(r"Using (\S+) Fp8 MoE backend out of potential backends: \[([^\]]+)\]", line)
        if m:
            result["compilation"]["moe_backend"] = m.group(1)
            backends = [b.strip().strip("'") for b in m.group(2).split(",")]
            result["compilation"]["moe_available_backends"] = backends

        # Unquantized MoE backend (e.g. Qwen3-30B unquantized)
        m = re.search(r"Using (\S+) backend for Unquantized MoE", line)
        if m and "moe_backend" not in result["compilation"]:
            result["compilation"]["moe_backend"] = m.group(1)

        # MXFP4 MoE backend
        m = re.search(r"Using '(\S+)' Mxfp4 MoE backend", line)
        if m and "moe_backend" not in result["compilation"]:
            result["compilation"]["moe_backend"] = m.group(1)

        # --- Attention backend ---
        m = re.search(r"Using (\S+) attention backend out of potential backends: \[([^\]]+)\]", line)
        if m:
            result["compilation"]["attention_backend"] = m.group(1)
            backends = [b.strip().strip("'") for b in m.group(2).split(",")]
            result["compilation"]["attention_available_backends"] = backends
        elif not result["compilation"].get("attention_backend"):
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

        # MXFP4 fallback warning
        if "MXFP4 linear layer is not implemented" in line:
            result["compilation"]["mxfp4_linear_fallback"] = True

        # MXFP4 attention skip
        if "MXFP4 attention layer is not implemented" in line:
            result["compilation"]["mxfp4_attention_skip"] = True

        # MoEPrepareAndFinalize variant
        m = re.search(r"Using (MoEPrepareAndFinalize\S+)", line)
        if m:
            result["compilation"]["moe_prepare_finalize"] = m.group(1)

        # --- Sampler backend ---
        if "Using FlashInfer for top-p & top-k sampling" in line:
            result["compilation"]["sampler_backend"] = "flashinfer"

        # --- FlashInfer allreduce ---
        m = re.search(r"Auto-selected flashinfer allreduce backend:\s*(\S+)", line)
        if m:
            result["compilation"]["allreduce_backend"] = m.group(1)

        # Initialized FlashInfer Allreduce fusion workspace
        m = re.search(r"Initialized FlashInfer Allreduce norm fusion workspace with backend=(\S+)", line)
        if m:
            result["compilation"]["allreduce_fusion_backend"] = m.group(1)

        # --- All-reduce backends per group ---
        m = re.search(r"Using \[([^\]]+)\] all-reduce backends .+ for group '(\S+)' out of potential backends: \[([^\]]+)\]", line)
        if m:
            group_name = m.group(2)
            selected = [b.strip().strip("'") for b in m.group(1).split(",")]
            available = [b.strip().strip("'") for b in m.group(3).split(",")]
            key = f"allreduce_backends_{group_name.replace(':', '_')}"
            if key not in result["compilation"]:
                result["compilation"][key] = {"selected": selected, "available": available}

        # --- FlashInfer autotune skip ---
        if "Skipping FlashInfer autotune because it is disabled" in line:
            result["compilation"]["flashinfer_autotune_enabled"] = False

        # --- Mamba SSU backend ---
        m = re.search(r"Using (\S+) Mamba SSU backend", line)
        if m and "mamba_ssu_backend" not in result["compilation"]:
            result["compilation"]["mamba_ssu_backend"] = m.group(1)

        # --- Mamba warmup ---
        if "Warming up Mamba2 SSD Triton kernels" in line:
            result["compilation"]["mamba2_ssd_warmup"] = True

        # --- Expert parallelism info ---
        m = re.search(r"\[EP Rank (\d+)/(\d+)\] Expert parallelism is enabled\. Expert placement strategy: (\S+)\. Local/global number of experts: (\d+)/(\d+)", line)
        if m and "expert_parallel" not in result["engine_config"]:
            result["engine_config"]["expert_parallel"] = {
                "ep_size": int(m.group(2)),
                "strategy": m.group(3).rstrip("."),
                "local_experts": int(m.group(4)),
                "global_experts": int(m.group(5)),
            }

        # --- Profiler ranges ---
        m = re.search(r"\[profiler\] vLLM profiler installed - will profile ranges:\s*\[([^\]]+)\]", line)
        if m and "profiler_ranges" not in result["engine_config"]:
            result["engine_config"]["profiler_ranges"] = m.group(1).strip()

        # --- MLA attention backend ---
        m = re.search(r"Using (\S+) attention backend out of potential backends: \[([^\]]+)\]", line)
        if m and "MLA" in m.group(1):
            result["compilation"]["mla_attention_backend"] = m.group(1)
            mla_backends = [b.strip().strip("'") for b in m.group(2).split(",")]
            result["compilation"]["mla_available_backends"] = mla_backends

        m = re.search(r"Using (\S+) prefill for MLA", line)
        if m:
            result["compilation"]["mla_prefill_backend"] = m.group(1)

        # --- Mamba attention/page size ---
        m = re.search(r"Setting attention block size to (\d+) tokens", line)
        if m and "mamba_attention_block_size" not in result["engine_config"]:
            result["engine_config"]["mamba_attention_block_size"] = int(m.group(1))

        # --- Mamba SSM cache dtype override ---
        m = re.search(r"Updating mamba_ssm_cache_dtype to '(\S+)' for (\S+) model", line)
        if m:
            result["engine_config"]["mamba_ssm_cache_dtype"] = m.group(1)
            result["engine_config"]["mamba_ssm_cache_model"] = m.group(2)

        # --- FP8 MoE unequal scales ---
        if "input_scales that are not equal for fp8 MoE layer" in line:
            result["compilation"]["fp8_moe_unequal_scales"] = True

        # --- MoE config file ---
        m = re.search(r"Using configuration from (\S+) for MoE layer", line)
        if m:
            result["compilation"]["moe_config_file"] = m.group(1)

        # --- Compile cache hit (directly loaded) ---
        m = re.search(r"Directly load the compiled graph\(s\) for (?:compile range \((\d+), (\d+)\)|dynamic shape) from the cache, took (\S+)\s*s", line)
        if m:
            result["compilation"]["compile_cache_hit"] = True
            result["compilation"]["compile_cache_load_time_s"] = _safe_float(m.group(3))

        # --- AOT compile cache hit ---
        m = re.search(r"Directly load AOT compilation from path (\S+)", line)
        if m:
            result["compilation"]["aot_cache_hit"] = True

        # --- AOT compile save ---
        m = re.search(r"saved AOT compiled function to (\S+)", line)
        if m:
            result["compilation"]["aot_compiled_saved"] = True

        # --- Compile range details (per range) ---
        m = re.search(r"(?:Cache the graph of compile range|Compiling a graph for compile range) \((\d+), (\d+)\)", line)
        if m:
            ranges = result["compilation"].setdefault("compile_ranges", [])
            rng = (int(m.group(1)), int(m.group(2)))
            if rng not in ranges:
                ranges.append(rng)

        # --- Default sampling params ---
        m = re.search(r"(?:Using default (?:chat|completion) sampling params|Default (?:vLLM )?sampling parameters have been overridden)", line)
        if m:
            sp = re.search(r"[`']\{([^}]+)\}[`']", line)
            if sp and "default_sampling_params" not in result["engine_config"]:
                result["engine_config"]["default_sampling_params"] = "{" + sp.group(1) + "}"

        # --- Auto tool choice ---
        if '"auto" tool choice has been enabled' in line:
            result["engine_config"]["auto_tool_choice_enabled"] = True

        # --- Chat template content format ---
        m = re.search(r"Detected the chat template content format to be '(\S+)'", line)
        if m and "chat_template_format" not in result["engine_config"]:
            result["engine_config"]["chat_template_format"] = m.group(1)

        # --- NCCL version from ProcessGroupNCCL ---
        m = re.search(r"NCCL version (\S+)", line)
        if m and "nccl_version" not in result["compilation"]:
            result["compilation"]["nccl_version"] = m.group(1)

        # --- Num GPU blocks override ---
        m = re.search(r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)", line)
        if m and "num_gpu_blocks_override" not in result["memory"]:
            result["memory"]["num_gpu_blocks_override"] = int(m.group(2))

        # --- FP8 KV cache info ---
        if "Using fp8 data type to store kv cache" in line:
            result["memory"]["kv_cache_fp8"] = True

        # --- KV cache scaling factor ---
        m = re.search(r"Using KV cache scaling factor (\S+) for (\S+)", line)
        if m and "kv_cache_scaling_factor" not in result["memory"]:
            result["memory"]["kv_cache_scaling_factor"] = _safe_float(m.group(1))
            result["memory"]["kv_cache_scaling_dtype"] = m.group(2)

        # --- Valid backends (older format without "out of potential") ---
        m = re.search(r"Valid backends:\s*\[([^\]]+)\]", line)
        if m and "attention_available_backends" not in result["compilation"]:
            backends = [b.strip().strip("'") for b in m.group(1).split(",")]
            result["compilation"]["attention_available_backends"] = backends

        # --- Using FLASH_ATTN backend (older format) ---
        m = re.search(r"Using (\S+) backend\.", line)
        if m and "attention_backend" not in result["compilation"] and "ATTN" in m.group(1):
            result["compilation"]["attention_backend"] = m.group(1)

        # --- Supported tasks ---
        m = re.search(r"Supported tasks:\s*\[([^\]]+)\]", line)
        if m:
            tasks = [t.strip().strip("'") for t in m.group(1).split(",")]
            result["engine_config"]["supported_tasks"] = tasks

        # --- DP group leader info ---
        m = re.search(r"DP group leader:.*world_size=(\d+),\s*local_world_size=(\d+)", line)
        if m:
            result["engine_config"]["world_size"] = int(m.group(1))
            result["engine_config"]["local_world_size"] = int(m.group(2))

        # --- Asynchronous scheduling disabled ---
        if "Asynchronous scheduling is disabled" in line:
            result["engine_config"]["async_scheduling"] = False

        # --- FlashMLA KV cache block size ---
        m = re.search(r"Forcing kv cache block size to (\d+) for (\S+) backend", line)
        if m:
            result["engine_config"]["kv_cache_block_size"] = int(m.group(1))
            result["engine_config"]["kv_cache_block_backend"] = m.group(2)

        # --- Multimodal encoder info ---
        m = re.search(r"Encoder cache will be initialized with a budget of (\d+) tokens", line)
        if m and "encoder_cache_budget_tokens" not in result["engine_config"]:
            result["engine_config"]["encoder_cache_budget_tokens"] = int(m.group(1))

        m = re.search(r"Multi-modal warmup completed in (\S+)s", line)
        if m and "mm_warmup_time_s" not in result["timing"]:
            result["timing"]["mm_warmup_time_s"] = _safe_float(m.group(1))

        m = re.search(r"Using (?:backend )?(\S+) for (?:vit|MMEncoder)", line)
        if m and "mm_encoder_attention_backend" not in result["compilation"]:
            result["compilation"]["mm_encoder_attention_backend"] = m.group(1)

        # --- Compressed tensors / Marlin MoE ---
        m = re.search(r"Using (CompressedTensors\S+)", line)
        if m and "compressed_tensors_method" not in result["compilation"]:
            result["compilation"]["compressed_tensors_method"] = m.group(1)

        m = re.search(r"Using (\S+) backend for (\S+) MoE \(group_size=(\d+), num_bits=(\d+)\)", line)
        if m and "moe_quantized_backend" not in result["compilation"]:
            result["compilation"]["moe_quantized_backend"] = m.group(1)
            result["compilation"]["moe_quant_type"] = m.group(2)
            result["compilation"]["moe_group_size"] = int(m.group(3))
            result["compilation"]["moe_num_bits"] = int(m.group(4))

        # --- FlashInfer GDN prefill ---
        if "Using FlashInfer GDN prefill kernel" in line and "gdn_prefill_backend" not in result["compilation"]:
            result["compilation"]["gdn_prefill_backend"] = "FlashInfer"

        # --- Rope type legacy replacement ---
        if "Replacing legacy 'type' key with 'rope_type'" in line:
            result["engine_config"]["rope_type_legacy_replaced"] = True

        # --- NCCL DP sync disabled ---
        if "Disabling NCCL for DP synchronization" in line:
            result["engine_config"]["nccl_dp_sync_disabled"] = True

        # --- Weight download time ---
        m = re.search(r"Time spent downloading weights for (\S+):\s*(\S+)\s*seconds", line)
        if m and "weights_download_time_s" not in result["memory"]:
            result["memory"]["weights_download_time_s"] = _safe_float(m.group(2))

        # --- DeepGEMM MoE disabled for TP ---
        m = re.search(r"DeepGEMM MoE is disabled by default when TP size is >= (\d+)", line)
        if m:
            result["compilation"]["deepgemm_moe_disabled_tp_threshold"] = int(m.group(1))

        # --- FlashAttention version ---
        m = re.search(r"Using FlashAttention version (\d+)", line)
        if m and "flash_attn_version" not in result["compilation"]:
            result["compilation"]["flash_attn_version"] = int(m.group(1))

        # --- Detected ModelOpt checkpoint ---
        m = re.search(r"Detected ModelOpt (\S+) checkpoint \(quant_algo=(\S+)\)", line)
        if m:
            result["engine_config"]["modelopt_dtype"] = m.group(1)
            result["engine_config"]["modelopt_quant_algo"] = m.group(2).rstrip(")")

        # --- Mamba page size padding ---
        m = re.search(r"Padding mamba page size by (\S+)%", line)
        if m and "mamba_page_size_padding_pct" not in result["engine_config"]:
            result["engine_config"]["mamba_page_size_padding_pct"] = _safe_float(m.group(1))

        # --- FlashInfer autotune skipped ---
        if "Skipping FlashInfer autotune because it is disabled" in line:
            result["compilation"]["flashinfer_autotune_skipped"] = True

        # --- Using FlashInfer for top-p & top-k sampling ---
        if "Using FlashInfer for top-p & top-k sampling" in line:
            result["compilation"]["flashinfer_sampling"] = True

        # --- Slow tokenizer warning ---
        if "Using a slow tokenizer" in line and "slow_tokenizer_warning" not in result["engine_config"]:
            result["engine_config"]["slow_tokenizer_warning"] = True

        # --- gpt-oss always enable tool use ---
        if "we ignore --enable-auto-tool-choice and always enable tool use" in line:
            result["engine_config"]["force_tool_use"] = True

        # --- Shared memory broadcast block wait ---
        m = re.search(r"No available shared memory broadcast block found in (\d+) seconds", line)
        if m and "shm_broadcast_wait_s" not in result["timing"]:
            result["timing"]["shm_broadcast_wait_s"] = int(m.group(1))

        # --- IR op priority ---
        m = re.search(r"Final IR op priority after setting platform defaults:\s*IrOpPriorityConfig\(([^)]+)\)", line)
        if m and "ir_op_priority" not in result["compilation"]:
            result["compilation"]["ir_op_priority"] = m.group(1)

        # --- Auto-prefetch disabled ---
        m = re.search(r"Auto-prefetch is disabled because the filesystem \((\S+)\)", line)
        if m and "auto_prefetch_disabled_fs" not in result["memory"]:
            result["memory"]["auto_prefetch_disabled_fs"] = m.group(1).rstrip(")")

        # --- Resolved architecture ---
        m = re.search(r"Resolved architecture:\s*(\S+)", line)
        if m and "resolved_architecture" not in result["engine_config"]:
            result["engine_config"]["resolved_architecture"] = m.group(1)

        # --- Unknown vLLM environment variables ---
        m = re.search(r"Unknown vLLM environment variable detected:\s*(\S+)", line)
        if m:
            result["engine_config"].setdefault("unknown_env_vars", [])
            var = m.group(1)
            if var not in result["engine_config"]["unknown_env_vars"]:
                result["engine_config"]["unknown_env_vars"].append(var)

        # --- Using max model len ---
        m = re.search(r"Using max model len (\d+)", line)
        if m and "max_model_len" not in result["engine_config"]:
            result["engine_config"]["max_model_len"] = int(m.group(1))

        # --- CUDA graph memory profiling message ---
        m = re.search(r"CUDA graph memory profiling (?:is enabled|will be enabled)", line)
        if m and "cuda_graph_memory_profiling" not in result["compilation"]:
            result["compilation"]["cuda_graph_memory_profiling"] = True

        # --- Padding intermediate size (MoE weight reshaping) ---
        m = re.search(r"Padding intermediate size from (\d+) to (\d+)", line)
        if m and "moe_intermediate_padding" not in result["compilation"]:
            result["compilation"]["moe_intermediate_padding"] = f"{m.group(1)}->{m.group(2)}"

        # --- Mamba SSU backend ---
        m = re.search(r"Using (\S+) Mamba SSU backend", line)
        if m and "mamba_ssu_backend" not in result["compilation"]:
            result["compilation"]["mamba_ssu_backend"] = m.group(1)

        # --- Chat template warmup ---
        m = re.search(r"Chat template warmup completed in (\S+)(?:ms|s)", line)
        if m and "chat_template_warmup_ms" not in result["timing"]:
            result["timing"]["chat_template_warmup_ms"] = _safe_float(m.group(1))

        # --- Readonly multi-modal warmup ---
        m = re.search(r"Readonly multi-modal warmup completed in (\S+)s", line)
        if m:
            result["timing"]["readonly_mm_warmup_time_s"] = _safe_float(m.group(1))

        # --- Model loading ---
        m = re.search(r"Loading weights took (\S+) seconds", line)
        if m:
            result["memory"]["weights_load_time_s"] = _safe_float(m.group(1))

        m = re.search(r"Model loading took (\S+) GiB memory and (\S+) seconds", line)
        if m:
            result["memory"]["model_memory_gib"] = _safe_float(m.group(1))
            result["memory"]["model_load_time_s"] = _safe_float(m.group(2))

        # --- Checkpoint info ---
        m = re.search(r"Filesystem type for checkpoints:\s*(\S+)\.\s*Checkpoint size:\s*(\S+)\s*GiB\.\s*Available RAM:\s*(\S+)\s*GiB", line)
        if m:
            result["memory"]["checkpoint_filesystem"] = m.group(1).rstrip(".")
            result["memory"]["checkpoint_size_gib"] = _safe_float(m.group(2))
            result["memory"]["available_ram_gib"] = _safe_float(m.group(3))

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

        # --- CUDA graph profiling ---
        m = re.search(r"Profiling CUDA graph memory:\s*PIECEWISE=(\d+)\s*\(largest=(\d+)\),\s*FULL=(\d+)\s*\(largest=(\d+)\)", line)
        if m and "cuda_graph_piecewise_count" not in result["compilation"]:
            result["compilation"]["cuda_graph_piecewise_count"] = int(m.group(1))
            result["compilation"]["cuda_graph_piecewise_largest"] = int(m.group(2))
            result["compilation"]["cuda_graph_full_count"] = int(m.group(3))
            result["compilation"]["cuda_graph_full_largest"] = int(m.group(4))

        m = re.search(r"Estimated CUDA graph memory:\s*(\S+)\s*GiB", line)
        if m and "estimated_cuda_graph_memory_gib" not in result["memory"]:
            result["memory"]["estimated_cuda_graph_memory_gib"] = _safe_float(m.group(1))

        # --- CUDA graph capture ---
        m = re.search(r"Graph capturing finished in (\d+)\s*secs?, took (\S+)\s*GiB", line)
        if m:
            result["compilation"]["cuda_graph_capture_time_s"] = _safe_int(m.group(1))
            result["compilation"]["cuda_graph_memory_gib"] = _safe_float(m.group(2))

        # --- CUDA graph pool memory (actual vs estimated) ---
        m = re.search(r"CUDA graph pool memory:\s*(\S+)\s*GiB\s*\(actual\),\s*(\S+)\s*GiB\s*\(estimated\)", line)
        if m and "cuda_graph_pool_actual_gib" not in result["memory"]:
            result["memory"]["cuda_graph_pool_actual_gib"] = _safe_float(m.group(1))
            result["memory"]["cuda_graph_pool_estimated_gib"] = _safe_float(m.group(2))

        m = re.search(r"Initial profiling/warmup run took (\S+)\s*s", line)
        if m:
            result["compilation"]["warmup_time_s"] = _safe_float(m.group(1))

        # --- Engine init (with compilation breakdown) ---
        m = re.search(r"init engine .+ took (\S+)\s*s(?:econds?)?\s*(?:\(compilation:\s*(\S+)\s*s\))?", line)
        if m:
            result["timing"]["engine_init_time_s"] = _safe_float(m.group(1))
            if m.group(2):
                result["timing"]["engine_init_compilation_time_s"] = _safe_float(m.group(2))

        # --- Kernel JIT monitor ---
        if "Kernel JIT monitor activated" in line and "jit_monitor_activated" not in result["compilation"]:
            result["compilation"]["jit_monitor_activated"] = True

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

    # ------------------------------------------------------------------ #
    #  Second pass: generic extraction for lines the regexes didn't cover #
    # ------------------------------------------------------------------ #
    # Build a set of "fingerprints" from values already extracted by the
    # regex pass so we can skip lines that are already represented.
    known_fingerprints: set[str] = set()
    for section in ("server_config", "engine_config", "compilation", "memory", "timing"):
        for k, v in result.get(section, {}).items():
            known_fingerprints.add(k)
            sv = str(v)
            if len(sv) > 6:
                known_fingerprints.add(sv[:60])

    # Also fingerprint common phrases already handled by regexes
    _KNOWN_PHRASES = {
        "Asynchronous scheduling", "Loading weights", "Model loading took",
        "Dynamo bytecode transform", "torch.compile", "Graph capturing",
        "Available KV cache", "GPU KV cache size", "Maximum concurrency",
        "Profiling CUDA graph", "Estimated CUDA graph", "CUDA graph pool",
        "init engine", "Kernel JIT monitor", "Chunked prefill",
        "FlashAttention version", "DeepGEMM E8M0", "Encoder cache",
        "Multi-modal warmup", "Detected the chat template",
        "auto\" tool choice", "Default vLLM sampling", "Default sampling",
        "Supported tasks", "DP group leader", "FlashInfer Allreduce",
        "Auto-selected flashinfer", "Overriding num_gpu_blocks",
        "Using fp8 data type", "KV cache scaling factor",
        "Forcing kv cache block size", "Replacing legacy",
        "Disabling NCCL", "Time spent downloading",
        "DeepGEMM MoE is disabled", "Detected ModelOpt",
        "Padding mamba page size", "Skipping FlashInfer autotune",
        "FlashInfer for top-p", "slow tokenizer",
        "enable tool use", "shared memory broadcast",
        "IR op priority", "Auto-prefetch is disabled",
        "Resolved architecture", "Filesystem type for checkpoints",
        "Warming up Mamba", "Expert parallelism is enabled",
        "profiler", "Compile range", "Cache the graph",
        "Compiling a graph", "saved AOT compiled", "profiling/warmup",
        "non-default args", "Initializing a V1 LLM engine",
        "Initializing an LLM engine", "vLLM API server version",
        "NVIDIA_VISIBLE_DEVICES", "nixl", "NIXL",
        "Enabled custom fusions", "CompressedTensors",
        "MoEPrepareAndFinalize", "Setting attention block size",
        "allreduce backends", "MLA attention",
        "Using default chat sampling", "Using default completion sampling",
        "chat template warmup", "Chat template warmup", "Selected",
        "Mxfp4 MoE backend", "Unquantized MoE",
        "MXFP4 linear layer", "MXFP4 attention layer",
        "compile_cache_save_format",
        "Using max model len", "vLLM is using nccl",
        "Unknown vLLM environment variable", "rank",
        "CUDA graph memory profiling", "Padding intermediate size",
        "Mamba SSU backend", "Readonly multi-modal warmup",
        "Overriding max cuda graph", "Warming up chat template",
        "input_scales that are not equal", "uncalibrated q_scale",
        "Checkpoint does not provide a q scaling",
        "Using cache directory", "Marlin backend",
        "FlashInfer GDN prefill", "Using FlashInfer CUTLASS",
        "FP8 MoE backend", "attention backend out of",
        "Fp8 MoE backend out of",
        "vit attention", "MMEncoderAttention",
        "nccl_dp_sync", "UCX_RCACHE",
    }

    def _line_already_handled(payload: str) -> bool:
        for phrase in _KNOWN_PHRASES:
            if phrase in payload:
                return True
        for fp in known_fingerprints:
            if len(fp) > 6 and fp in payload:
                return True
        return False

    # Strip ANSI escapes and log prefixes to get clean payloads
    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
    _LOG_PREFIX_RE = re.compile(
        r"^(?:\[?[0-9;m\x1b]*\]?\s*)*"
        r"(?:\([^)]+\)\s*)?"
        r"(?:INFO|WARNING|ERROR)\s+"
        r"\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+"
        r"\[[^\]]+\]\s*"
    )

    auto_extracted: Dict[str, Any] = {}
    unmatched: list[str] = []
    seen_payloads: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or _is_noise(line):
            continue

        if "INFO" not in line and "WARNING" not in line:
            continue

        clean = _ANSI_RE.sub("", stripped)
        payload = _LOG_PREFIX_RE.sub("", clean).strip()
        if not payload or len(payload) < 15:
            continue

        # Deduplicate
        sig = payload[:80]
        if sig in seen_payloads:
            continue
        seen_payloads.add(sig)

        if _line_already_handled(payload):
            continue

        # Try generic "Using X ..." extraction
        using = _generic_extract_using(payload)
        if using:
            key = re.sub(r"[^a-z0-9]+", "_", using["selected"].lower()).strip("_")
            if key and len(key) > 2 and key not in auto_extracted:
                auto_extracted[key] = using
                continue

        # Collect remaining significant unmatched lines (payload only)
        if len(payload) > 15:
            unmatched.append(payload)

    if auto_extracted:
        result["auto_extracted"] = auto_extracted
    if unmatched:
        result["unmatched_lines"] = unmatched[:30]

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
