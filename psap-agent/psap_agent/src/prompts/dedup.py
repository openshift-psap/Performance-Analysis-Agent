"""Semantic duplicate-check prompt for the memory pipeline.

Uses a lightweight LLM call with tag-based output to determine whether a
new fact is semantically equivalent to any existing fact in the database,
even if worded differently.  This is the gate that prevents redundant
inserts while allowing multiple distinct facts per category.
"""

from __future__ import annotations

import json

DEDUP_CHECK_SYSTEM = """\
You are a duplicate detector for a performance-analysis fact store.

# Your Task

You will receive:
  1. A NEW FACT that is about to be inserted into the database.
  2. A list of EXISTING FACTS already stored for the same configuration scope.

Decide whether the new fact is a semantic duplicate of ANY existing fact. \
Two facts are duplicates if they describe the SAME underlying issue or \
finding, even if:
  - They use different wording or sentence structure.
  - They include slightly different numbers for the same metric.
  - One is more detailed than the other but covers the same point.

Two facts are NOT duplicates if they describe:
  - Different metrics or different aspects of a regression.
  - The same kernel/component but different issues (e.g., timing vs call count).
  - Related but distinct root causes.

# Output Format

Respond with EXACTLY one line in this format:

[START_ANSWER]duplicate[END_ANSWER]

or

[START_ANSWER]not_duplicate[END_ANSWER]

If duplicate, add a second line:

[START_EXISTING_ID]id-of-the-matching-fact[END_EXISTING_ID]

Do NOT include any other text, explanation, or commentary.\
"""


def build_dedup_check_prompt(
    new_fact: dict, existing_facts: list[dict]
) -> str:
    """Build the user prompt for the duplicate check."""
    new_text = new_fact.get("fact_text", new_fact.get("text", ""))
    new_cat = new_fact.get("category", "unknown")

    existing_lines = []
    for f in existing_facts:
        fid = f.get("id", "?")
        ftext = f.get("fact_text", f.get("summary_text", ""))
        fcat = f.get("category", "?")
        existing_lines.append(f"  - [{fid}] ({fcat}) {ftext}")

    existing_block = "\n".join(existing_lines) if existing_lines else "  (none)"

    return (
        f"## New Fact\n"
        f"Category: {new_cat}\n"
        f"Text: {new_text}\n\n"
        f"## Existing Facts in Database\n{existing_block}"
    )
