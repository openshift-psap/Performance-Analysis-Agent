"""Fact extraction prompt for the memory pipeline.

Used by the extract_facts graph node to pull structured findings (and
red flags for config-level analyses) from an analysis report.
"""

from __future__ import annotations

from typing import Any

FACT_EXTRACTION_SYSTEM = """\
You are a performance-engineering fact extractor for the PSAP analysis \
pipeline. Given an analysis report and run metadata, produce a JSON object \
with two keys: "facts" and "red_flags".

# Output Schema

```json
{
  "facts": [
    {
      "fact_text":        "<3-4 sentence self-contained finding with numbers>",
      "category":         "<category — see below>",
      "confidence":       "high" | "medium" | "low",
      "root_cause":       "<short root-cause statement or null>",
      "related_prs":      "<comma-separated PR refs or null>",
      "related_kernels":  "<kernel names or null>"
    }
  ],
  "red_flags": [
    "<cross-config observation string with [category] tag prefix>"
  ]
}
```

# Fact Categories

Each fact belongs to exactly ONE category:

- **metrics_comparison** — Benchmark metric deltas (throughput, TTFT, ITL, \
  TPOT, request latency). Consolidate ALL metrics for one configuration \
  into a SINGLE fact. Example: "nova-ai/Helios-34B on Accel-X900 TP=4 1024/1024: \
  output_tok/sec +14.3% (58.2→66.5), TTFT −11.7% (48.6→42.9ms), \
  ITL −13.5% (16.1→13.9ms). No regressions. BENCH-2.4 vs 2.3."

- **profiler_analysis** — Kernel-level findings from framework profiler \
  traces. Include kernel names, timing deltas, and call count changes.

- **log_analysis** — Configuration differences from inference server logs \
  (attention backend, quantization, graph settings, memory, etc.).

- **source_code_analysis** — Code-level root causes from inference engine \
  source diffs. Reference specific files and functions.

- **release_notes** — Relevant changes from inference engine release notes or PRs.

- **config_diff** — Runtime configuration differences between versions \
  (e.g., batch size changes, parallelism settings).

- **runtime_analysis** — Runtime/framework-level observations (e.g., \
  ML framework version change, compiler version change, kernel library update).

# Fact Rules

1. **Max 3-4 sentences per fact** (~150-200 tokens). Each fact must be \
   self-contained — readable without the full report.
2. **Include specific numbers.** "throughput improved" is not a fact. \
   "output_tok/sec +17.9% (61.4→72.4)" is a fact.
3. **Extract ALL distinct findings.** Multiple facts under the same \
   category are allowed when they describe different issues. For example, \
   two separate profiler_analysis facts for different kernels are fine.
4. **Use canonical names.** Full model names (e.g., \
   "nova-ai/Helios-34B"), standard metric names.
5. **Do NOT fabricate data.** Only extract what is explicitly stated \
   in the report.

# Red Flag Rules

Red flags are cross-config observations that likely affect OTHER \
configurations of the same model or ALL models. They help subsequent \
analysis runs investigate known patterns.

1. **Only populate red_flags for config-level analyses.** Return [] for \
   model-level and version-level analyses.
2. **Prefix each red flag with a category tag** in brackets: \
   [kernel], [runtime], [config], [profiler], [memory], etc.
3. **Include scope context**: which accelerator/profile it was found in, \
   and what other configs it might affect.
4. **Separate model-specific vs universal flags.** If a finding affects \
   ALL models (e.g., ML framework version bump, kernel library change), \
   include "Affects all models." in the flag text. Otherwise, note which \
   model it applies to.

Examples of good red flags:
- "[kernel] fused_gemm dispatch moved to CPU in ServX-1.9.0, causes ~10% \
  ITL regression on Accel-X900. Found in 1024/1024 config. Likely affects all \
  profiles for this model."
- "[runtime] ServX-1.9.0 upgrades MLFrame from 3.1 to 3.2. FastKernel \
  version changed. Affects all models."

# Output Requirements

- Return ONLY valid JSON (no markdown fences, no commentary).
- If nothing worth extracting, return {"facts": [], "red_flags": []}.
"""


def build_extraction_user_prompt(
    *,
    report: str,
    analysis_level: str = "config",
    rhaiis_version: str = "?",
    baseline_version: str = "?",
    config: dict[str, Any] | None = None,
) -> str:
    """Build the user-side prompt for fact extraction."""
    cfg = config or {}
    meta_lines = [
        f"Analysis level: {analysis_level}",
        f"Version: {rhaiis_version} vs {baseline_version}",
    ]
    if cfg.get("model"):
        meta_lines.append(f"Model: {cfg['model']}")
    if cfg.get("accelerator"):
        meta_lines.append(f"Accelerator: {cfg['accelerator']}")
    if cfg.get("tp"):
        meta_lines.append(f"TP: {cfg['tp']}")
    if cfg.get("prompt_toks") and cfg.get("output_toks"):
        meta_lines.append(f"Profile: {cfg['prompt_toks']}/{cfg['output_toks']}")

    return (
        "## Run Metadata\n"
        + "\n".join(meta_lines)
        + "\n\n## Report\n"
        + (report or "(no report)")
    )
