# Benchmark analysis

Use this workflow for benchmark discovery, exact performance metrics, fair
comparisons, regression analysis, runtime-argument questions, and dashboard
links for benchmark results. Load `clarification_and_comparison` first if any
model, version, accelerator, profile, or TP choice is unresolved.

## Select the right data tool

- Use `get_dataset_metadata` for unfiltered, overall inventory questions.
- Use `discover_configurations` to resolve exact models, accelerators, profiles,
  versions, or TP values that were actually tested.
- Use `query_performance_metrics` for exact metrics, runtime arguments, UUIDs,
  and Grafana availability. It is the source for configuration parameters and
  deployment UUIDs, not just performance metrics.
- Use `compare_configurations` for a side-by-side peak-throughput comparison.
  Do not query two configurations separately and compare them manually.
- Use `compare_versions_comprehensive` when the user asks for the geometric
  mean across common concurrencies. Do not use `analyze_regression` as a
  substitute for a side-by-side version comparison; it is for trend analysis.
- Use `analyze_regression` for a version-to-version regression investigation
  across its supported data. For throughput/latency trade-offs, report only
  observed comparison points; do not invent a Pareto result or call a tool that
  is not available from the MCP server.

## Compare only fairly

The valid headline methods are peak throughput from `compare_configurations`
and geometric mean across common concurrencies from
`compare_versions_comprehensive`. Never headline an arbitrary single
concurrency point, especially concurrency 1, as a version comparison.

Never compare different profiles. If `compare_configurations` reports
`profile_mismatch`, explain the actual available profiles and ask which common
profile to use.

Never compare different TP values. If it reports `tp_mismatch` and recommends
TP=1, retry the same comparison at TP=1 without asking first. If that retry
does not produce a fair comparison, ask the user which common TP to use.

For a multi-model request, run a separate comparison for each model. Summarize
the winner for each model, then provide a concise overall result. Use
“improved”, “regressed”, or “faster/slower” for versions of the same system;
use “achieved”, “outperformed”, or “better/faster than” for different systems.

## Report benchmark evidence correctly

- Peak-throughput comparisons must say that each value is at its returned
  peak-throughput concurrency. Each configuration can peak at a different
  concurrency.
- Every throughput value—including output, prompt, and total tokens/sec—must
  include its actual concurrency, for example
  `72.24 tokens/sec at 400 concurrent requests`.
- Request-latency fields ending in `_sec` are seconds. TTFT, ITL, and TPOT
  fields ending in `_ms` are milliseconds.
- Report only values returned by the tools. Do not interpolate, estimate, or
  invent a missing concurrency point.
- Treat configuration differences as observations, not root causes, unless
  profiling or a controlled experiment corroborates causality.

## Dashboard links

After presenting performance data from `query_performance_metrics` or
`compare_configurations`, call `generate_dashboard_url` in the same response.
Pass every version being compared. For a follow-up request to show, graph, or
visualize already-established data, generate the link from the known context
instead of asking for the same parameters again.

- Use `performance_plots` for a metric-versus-concurrency chart. Set
  `pp_x='Concurrency'`, and choose `pp_y` only from these exact values:
  `Throughput (Output tokens/second generated)`,
  `Efficiency (Output tokens/sec per TP unit)`,
  `Inter-Token Latency P95 (Time between tokens)`,
  `Time to First Token P95 (Response start delay)`,
  `Request Latency Median (Total request processing time)`,
  `Request Latency Max (Maximum request processing time)`,
  `Time Per Output Token P95 (Token generation time)`,
  `Total Throughput (Total tokens/second processed)`,
  `Request Count (Successful completions)`, or `Error Rate (% Failed requests)`.
  Set `pp_conc` only from the highest relevant observed concurrency.
- Use `compare_versions` for a structured two-version delta table. Supply
  `cv_v1`, `cv_v2`, `cv_gpu`, and `cv_profile`; use `cv_conc` only with actual
  concurrency values returned by a comparison tool, or omit it.
- Use `runtime_configs` for runtime arguments or deployment settings, and
  `view_logs` for benchmark log viewing.
- Do not generate a performance-dashboard link for a metadata-only answer,
  clarification question, or GPU-metrics response; use the Grafana workflow
  for GPU metrics instead.

Never construct a dashboard URL. Include only the exact URL returned by the
tool.
