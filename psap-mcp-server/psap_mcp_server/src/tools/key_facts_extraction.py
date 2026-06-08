"""LLM-based key fact extraction from performance analysis reports.

Extracts consolidated performance findings — one fact per configuration
per analysis stage — returning JSON with metadata for deterministic retrieval.
"""

PERFORMANCE_FACT_EXTRACTION_PROMPT = """\
You are a **Performance Findings Extractor** for the performance \
analysis agent. Your job is to read performance analysis reports \
and extract discrete, self-contained findings suitable for storage \
in a key facts database.

# Fact Categories

Each fact belongs to exactly ONE category:

- **metrics_comparison**: Benchmark metric deltas. Consolidate ALL metrics \
  for one configuration into a SINGLE fact. Example: "nova-ai/Helios-34B on \
  Accel-X900 TP=4 1024/1024: output_tok/sec +14.3% (58.2->66.5), TTFT -11.7% \
  (48.6->42.9ms), ITL -13.5% (16.1->13.9ms). No regressions. BENCH-2.4 vs 2.3."

- **profiler_analysis**: Kernel-level findings from framework profiler traces.

- **log_analysis**: Configuration differences from inference server logs.

- **source_code_analysis**: Code-level root causes from inference engine source diffs.

- **release_notes**: Relevant changes from inference engine release notes / PRs.

- **version_summary**: Cross-configuration rollup for an entire version.

- **observation**: General findings that don't fit other categories.

# Key Rules

1. **One fact per configuration per category.** All metrics for one config \
   go in ONE metrics_comparison fact, not one per metric.
2. **Include numbers.** "throughput improved" is useless. \
   "output_tok/sec +17.9% (61.4->72.4)" is a fact.
3. **Facts must be self-contained.** Readable without the full report.
4. **Use canonical names.** Full model names, standard metric names.
5. **Return `{"findings": []}` if nothing worth extracting.**

# Output Format

```json
{
  "findings": [
    {
      "text": "Self-contained finding with all relevant numbers.",
      "category": "metrics_comparison",
      "rhaiis_version": "BENCH-2.4",
      "baseline_version": "BENCH-2.3",
      "model_name": "nova-ai/Helios-34B",
      "model_family": "moe",
      "accelerator": "Accel-X900",
      "tp_config": 4,
      "profile": "1024/1024",
      "root_cause": "kernel fusion in fused_gemm (if identified, else null)",
      "related_prs": "#4567, #3891 (if known, else null)",
      "related_kernels": "fused_gemm_kernel, scatter_op (if relevant, else null)",
      "confidence": "high"
    }
  ]
}
```
"""


def build_extraction_user_prompt(
    analysis_report: str,
    existing_facts: list[dict] | None = None,
    run_context: dict | None = None,
) -> str:
    """Build the user-side prompt for the extraction LLM call."""
    sections = []

    if run_context:
        ctx_lines = ["## Run Context"]
        for key, value in run_context.items():
            if value is not None:
                ctx_lines.append(f"- {key}: {value}")
        sections.append("\n".join(ctx_lines))

    if existing_facts:
        import json
        compact = [
            {"id": f.get("id", str(i)), "text": f.get("fact_text", f.get("text", "")),
             "category": f.get("category", ""), "version": f.get("rhaiis_version"),
             "model": f.get("model_name")}
            for i, f in enumerate(existing_facts)
        ]
        sections.append(f"## Existing Key Facts (do NOT re-extract duplicates)\n{json.dumps(compact, indent=2)}")

    sections.append(f"## Analysis Report\n{analysis_report}")
    sections.append("# Output:")
    return "\n\n".join(sections)
