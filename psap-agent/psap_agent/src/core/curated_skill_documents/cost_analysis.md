# Cost analysis

Use this workflow for cost per million tokens, TCO, price comparisons, or the
cheapest configuration that satisfies performance requirements. Load
`clarification_and_comparison` when the model variant, accelerator, or version
is unresolved.

## Before calculating

- Before calling `calculate_cost_efficiency`, ask for ITL P95 and TTFT P95
  requirements when the user has not stated them. Mention the PSAP defaults:
  ITL P95 <= 65 ms and TTFT P95 <= 4000 ms. Do not silently apply those
  defaults before giving the user that choice.
- Use the user-specified profile when present. Otherwise default to
  `512/2048` without asking for a profile.
- If that profile is unavailable, use an available profile and explicitly say
  which one was used and that `512/2048` was unavailable.
- Include a version filter when the user specified a RHAIIS version. When a
  version is missing and several exact-model versions exist, use the
  clarification workflow rather than selecting one silently.

## Calculate and present results

Call `calculate_cost_efficiency`. Report:

- The exact profile used.
- The latency requirements used, including whether they were user-supplied or
  the stated PSAP defaults.
- The intended concurrency returned by the tool.
- ITL P95 and TTFT P95 at that concurrency.
- The cost methodology: H200 and MI300X use adjusted throughput for an
  eight-GPU instance; TPU uses per-core pricing.

The tool returns actual benchmark observations. Do not interpolate values,
estimate a concurrency level, or present an unmeasured configuration as the
lowest cost.

## Link to the result

After presenting a cost analysis, call `generate_dashboard_url` with
`section='cost_analysis'`. Use only the exact URL from that tool result.
