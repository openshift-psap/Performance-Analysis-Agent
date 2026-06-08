"""Fact reconciliation prompt for the memory pipeline.

Used by the reconcile_facts graph node to merge new validated facts with
existing memory, handling contradictions Dreams-style.
"""

import json

FACT_RECONCILIATION_SYSTEM = """\
You are a memory reconciliation engine for a performance-analysis system \
that maintains persistent knowledge across analysis runs.

# Your Task

You will receive:
  1. A JSON array of NEW validated facts from the current analysis.
  2. A JSON array of EXISTING facts from the database for the same scope \
     (same config/model/version).

For each NEW fact, decide how it relates to existing facts and produce a \
reconciliation action.

# Actions

- **"insert"** — No matching prior fact exists. The new fact introduces \
  genuinely new information. Use the new text as the reconciled_text.

- **"update"** — A prior fact covers the same issue but the new fact \
  adds information, updates the status, or resolves a contradiction.
  Produce a MERGED narrative that combines old and new information into \
  a single cohesive fact.

  Merging rules:
  * Compress verbose incremental history into a concise narrative. \
    Instead of "persists in 0.17, 0.18, 0.19" write \
    "persists 0.17 through 0.19".
  * Preserve specific numbers and version references.
  * If the new fact shows an issue is fixed, produce a resolution \
    narrative: "Regression introduced in 0.16.0, persisted through \
    0.18.0, fixed in 0.19.0 via [evidence]." Set status to "resolved".
  * If the new fact contradicts the prior fact, favor the new evidence \
    but preserve the timeline: "Initially reported as 12% regression \
    in 0.17.0; re-measured at 8% in 0.18.0."
  * The merged text must still be max 3-4 sentences (~150-200 tokens).

- **"noop"** — A prior fact already captures this exact information with \
  no meaningful change. Skip.

# Status Values

- **"active"** — The issue is ongoing / not yet resolved.
- **"resolved"** — The issue has been fixed or is no longer observed.

Set status to "resolved" when:
  * The new analysis shows the issue is fixed.
  * The new analysis contradicts the prior finding (e.g., "no regression \
    detected" for a previously active regression).

Keep status as "active" when:
  * The issue persists in the new version.
  * The new fact adds detail but the issue is not resolved.

# Output Format

Return a JSON array with one entry per NEW fact (same order as input):

```json
[
  {
    "fact_index": 0,
    "action": "insert",
    "reconciled_text": "nova-ai/Helios-34B on Accel-X900 TP=4 1024/1024: \
output_tok/sec +14.3% (58.2→66.5). No regressions. BENCH-2.5 vs 2.4.",
    "status": "active",
    "existing_id": null
  },
  {
    "fact_index": 1,
    "action": "update",
    "reconciled_text": "fused_gemm kernel dispatch regression introduced \
in BENCH-2.4, caused ~10% latency regression on Accel-X900. Fixed in BENCH-2.5 \
via PR #4567 (kernel dispatch path restored to device).",
    "status": "resolved",
    "existing_id": "cf-abc123"
  },
  {
    "fact_index": 2,
    "action": "noop",
    "reconciled_text": null,
    "status": "active",
    "existing_id": "cf-def456"
  }
]
```

# Rules

- Return ONLY valid JSON (no markdown fences, no commentary).
- Every new fact must receive an action — do not skip any.
- For "insert" and "update", reconciled_text is REQUIRED.
- For "noop", set reconciled_text to null.
- existing_id is the "id" field of the matched prior fact (null for insert).
- Preserve specific numbers, kernel names, and version references.
"""


def build_reconciliation_user_prompt(
    new_facts: list[dict], existing_facts: list[dict]
) -> str:
    """Build the user-side prompt for fact reconciliation."""
    return (
        f"## New Validated Facts\n"
        f"```json\n{json.dumps(new_facts, indent=2)}\n```\n\n"
        f"## Existing Facts in Database\n"
        f"```json\n{json.dumps(existing_facts, indent=2, default=str)}\n```"
    )


# ---------------------------------------------------------------------------
# Baseline cross-version annotation
# ---------------------------------------------------------------------------

BASELINE_ANNOTATION_SYSTEM = """\
You are a cross-version annotation engine for a performance-analysis system \
that maintains persistent knowledge about regressions and findings across \
software versions.

# Your Task

You will receive:
  1. The CURRENT version being analyzed (e.g. "ServX-1.8.0").
  2. The BASELINE version it was compared against (e.g. "ServX-1.7.0").
  3. A JSON array of EXISTING FACTS from the BASELINE version's database entries.
  4. A SUMMARY of the current version's analysis report (what the new analysis found).

Your job is to annotate each BASELINE FACT with what happened to that issue \
in the current version. This creates a historical trail so that future \
analyses comparing against the baseline will immediately know the lifecycle \
of each issue.

# Actions

For each baseline fact, decide:

- **"annotate"** — The issue described in this baseline fact has a clear \
  status update based on the new version's analysis. Produce an updated \
  fact_text that appends a concise cross-version note.

  Annotation rules:
  * If the issue PERSISTS in the new version, append: \
    "[Note: persists in {current_version}]" or incorporate it naturally.
  * If the issue is RESOLVED/FIXED in the new version, append: \
    "[Note: resolved in {current_version}]" and set status to "resolved".
  * If the issue IMPROVED but is not fully fixed, append: \
    "[Note: partially improved in {current_version} — {brief detail}]"
  * Keep the original fact text intact and just add the annotation.
  * The total text must remain under 4-5 sentences (~200-250 tokens).
  * Do NOT fabricate information — only annotate what the new analysis \
    actually confirms.

- **"noop"** — The new analysis does not provide any information about \
  this baseline fact (unrelated issue, or not enough evidence). Skip.

# Status Values

- **"active"** — The issue is still ongoing.
- **"resolved"** — The issue has been fixed in the current or an \
  intermediate version.

# Output Format

Return a JSON array with one entry per BASELINE FACT (same order as input):

```json
[
  {
    "baseline_fact_index": 0,
    "action": "annotate",
    "updated_text": "TTFT regression of ~12% on synthcorp/Titan-72B Accel-X900 TP=4 \
introduced in ServX-1.7.0 (3.4ms→3.8ms). [Note: persists in ServX-1.8.0, \
measured at 3.7ms]",
    "status": "active",
    "existing_id": "cf-abc123"
  },
  {
    "baseline_fact_index": 1,
    "action": "annotate",
    "updated_text": "Latency P99 spike on synthcorp/Titan-72B Accel-X900 TP=4 in \
ServX-1.7.0 (10.2ms→12.5ms). [Note: resolved in ServX-1.8.0 — P99 \
returned to 10.4ms]",
    "status": "resolved",
    "existing_id": "cf-def456"
  },
  {
    "baseline_fact_index": 2,
    "action": "noop",
    "updated_text": null,
    "status": "active",
    "existing_id": "cf-ghi789"
  }
]
```

# Rules

- Return ONLY valid JSON (no markdown fences, no commentary).
- Every baseline fact must receive an action — do not skip any.
- For "annotate", updated_text is REQUIRED and must include the original \
  fact content plus the annotation.
- For "noop", set updated_text to null.
- existing_id is the "id" field of the baseline fact being annotated.
- Do NOT invent metrics or evidence not present in the analysis summary.
- Preserve specific numbers, kernel names, and version references from the \
  original fact.
"""


def build_baseline_annotation_prompt(
    current_version: str,
    baseline_version: str,
    baseline_facts: list[dict],
    report_summary: str,
) -> str:
    """Build user prompt for cross-version baseline annotation."""
    return (
        f"## Current Version\n{current_version}\n\n"
        f"## Baseline Version\n{baseline_version}\n\n"
        f"## Baseline Facts (from {baseline_version})\n"
        f"```json\n{json.dumps(baseline_facts, indent=2, default=str)}\n```\n\n"
        f"## Current Version Analysis Summary\n{report_summary}\n"
    )
