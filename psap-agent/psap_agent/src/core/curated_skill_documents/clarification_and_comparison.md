# Clarification and fair comparison

Use this workflow before a parameterized benchmark, cost, comparison, or
Grafana analysis when the user has not supplied one exact, unambiguous
configuration. Load the relevant domain skill as well after the request is
resolved.

## Establish intent and preserve user constraints

- First decide whether the user’s intent is clear. If a short or vague request
  could mean several things, ask what result, metric, run, or question they mean
  before choosing tools or parameters.
- Keep every explicit exclusion or constraint. Remove excluded versions, models,
  accelerators, or profiles before presenting results, and say that the result
  honors the exclusion. Never suggest or use an excluded item.
- When several dimensions are ambiguous, resolve them in one response rather
  than asking one question at a time.

## Ask with real, filtered choices

Every clarification question must use options returned by
`discover_configurations`, filtered by every known input. Never offer generic
examples as if they were available data.

- For a partial model name or an unspecified quantization/model variant, first
  discover matching exact model names with the known version, accelerator, and
  profile filters. If one exact model remains, proceed; if several remain, list
  those exact choices and ask the user; if none remain, broaden filters only to
  find data-backed alternatives.
- When a mentioned version may have variants (for example, a base and `-async`
  release), discover versions and ask the user to choose the exact variant.
- Before a comparison, discover TP values using the exact model, version,
  accelerator, and profile. If one TP exists, use it explicitly. If several
  exist, list them and ask for a common TP.
- If a comparison lacks a version, accelerator, or profile, discover the
  available choices and ask for the missing dimensions. Do not silently choose
  a default for those comparison inputs.
- For a cost request without a version, first resolve the exact model variant,
  then discover its available versions for the requested accelerator. If more
  than one exists, ask whether to analyze a specific version or find the
  lowest-cost version across them.
- For a GPU/Grafana request, resolve the exact model, version, accelerator, and
  profile before retrieving the benchmark run. Then use
  `query_performance_metrics` to obtain the UUID and check Grafana availability.

## Handle no data safely

If a tool returns no data, say so first, then use `discover_configurations` to
find what is actually available. For a model/version mismatch, offer only the
available versions for that exact model. For an accelerator mismatch, list the
available accelerators and ask before using a different one. Never silently
substitute a different accelerator, model, version, profile, or TP value.

## Make comparisons fairly

- Use `compare_configurations` for the default peak-throughput comparison; do
  not query two runs separately and calculate a comparison manually.
- Use `compare_versions_comprehensive` for a geometric-mean comparison across
  common concurrencies.
- These are the only headline comparison methods. Do not present an arbitrary
  single concurrency point, especially concurrency 1, as a representative
  version comparison.
- Never compare different profiles or TP values. If the comparison tool reports
  `profile_mismatch`, ask for a common available profile. If it reports
  `tp_mismatch` and recommends TP=1, retry once at TP=1; otherwise ask for the
  common TP.
- For several models, make a separate comparison for each model. State a
  per-model winner and then provide a concise overall summary. Use “improved” or
  “regressed” for versions of the same system; use “outperformed” or “achieved”
  for different systems.
