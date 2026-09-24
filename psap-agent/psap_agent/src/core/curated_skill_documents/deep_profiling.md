# Deep performance investigation

Use this workflow when the user asks why one version is faster or slower, asks
for a root-cause analysis, or wants a rigorous explanation of a regression or
improvement. Load `clarification_and_comparison` if the compared configurations
are not exact.

## Required evidence order

Use this order. Do not replace measured profiling evidence with release notes,
PR descriptions, or general knowledge.

1. Establish the measured benchmark delta for two fair configurations: same
   model, accelerator, profile, TP, workload, and comparable versions.
2. Load `pytorch_profiles` and call `list_available_profiles` before selecting a
   trace. Then call `compare_pytorch_profiles` first for the chosen traces.
3. Call `analyze_performance_insights` with the observed benchmark direction,
   and `compare_trace_structures` for ordered pipeline, stream, and overhead
   evidence.
4. Load `vllm_logs` and call `compare_vllm_logs` after profiling to identify
   supporting configuration and pinned-dependency differences.
5. Load `kernel_source_analysis`. Prefer call stacks, then map material kernels,
   fetch relevant source for both versions, and inspect the actual code diff.

Do not use release notes, changelogs, or PR descriptions as root-cause evidence
during this workflow unless the user explicitly asks for them. They can explain
intent, but they never replace profile, log, and source evidence.

## When evidence is incomplete

If profile traces are unavailable or a profiling tool fails, say that
kernel-level root-cause analysis cannot be confirmed. You may still compare logs
as observations, clearly state the limitation, and recommend collecting traces
or running a controlled experiment. Do not use release notes as a substitute.

If benchmark throughput changes materially but total GPU kernel time changes
little, report the single-request profile paradox rather than inflating a small
kernel delta. Inspect `compare_trace_structures` for inter-block gaps,
intra-block idle time, stream overlap, batching, or scheduling effects. State
what profiling measures, what overhead evidence shows, what source evidence
suggests, and what remains unresolved.

## Root-cause standard

Separate three levels of certainty:

- **Observation:** a benchmark, log, or trace reports a difference.
- **Hypothesis:** the difference could explain part of the measured outcome.
- **Confirmed mechanism:** profiling or a controlled experiment connects the
  change to the observed performance delta.

Configuration differences, release-note items, and source diffs are not root
causes by themselves. Profiling evidence is ground truth when it conflicts with
a release-note or source-reading hypothesis.

For every material contributor, provide actual trace time, call-count, and
per-call-duration evidence. For multi-GPU models, examine communication kernels
and distinguish a call-count change from a per-call-duration change. Investigate
dispatch logic, synchronization, fusion, memory allocation, and stream
utilization as applicable; do not merely name a kernel or say it was “fused.”

## Source and log evidence

Logs are ground truth for explicit engine configuration, backend selection,
parallelism, CUDA-graph settings, memory/KV-cache values, compilation settings,
and pinned dependencies. Report meaningful changes in `pinned_dependencies`;
do not infer a configuration from kernel names when the logs contain the answer.

For each claimed mechanism, retrieve source or a code diff for the relevant
implementation and cite the retrieved file/change. If source retrieval fails,
state that error rather than claiming source analysis was completed.

## Final structure

Present the measured benchmark change first. Then provide:

1. Configuration observations from logs, if available.
2. For each contributor: profiling evidence, relevant log evidence, source-code
   evidence, and the certainty level.
3. Overhead evidence, including inter-block gaps or intra-block idle changes
   when available.
4. Quantitative accounting using only actual profiling deltas in time units.
   Never convert small trace deltas into estimated throughput impact, add
   unrelated savings, or present an unmeasured “at scale” result.
5. Remaining unknowns, recommended next steps, and confidence.

Every metric, kernel name, configuration detail, source reference, and causal
claim must be traceable to a tool result.
