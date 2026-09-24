# PyTorch profiler analysis

Use this workflow for profile availability, kernel hotspots, kernel regressions,
trace structure, CUDA timelines, or profiler-based performance explanations.

## Discover before analysis

Always call `list_available_profiles` first. Profile availability is dynamic;
never assume a model, TP, workload, rank, or version has a trace. Use
`check_profile_status` when a quick availability diagnostic is helpful.

Once the exact trace coordinates are known:

- Use `analyze_pytorch_profile` for one trace's hotspots and category totals.
- Use `compare_pytorch_profiles` first for kernel-level changes between two
  versions.
- Use `analyze_performance_insights` when the question asks why performance
  changed or when benchmark results and profile totals appear to disagree. Pass
  the observed benchmark direction rather than assuming it.
- Use `analyze_trace_structure` for an ordered pipeline/block view of one trace.
- Use `compare_trace_structures` for block-level structure, stream overlap,
  overhead deltas, and structured root-cause evidence across versions.
- Use `get_kernel_call_stacks` for ground-truth Python/C++ attribution when
  stacks are available.

## Interpret accurately

- Separate GPU compute, CPU operations, CUDA runtime overhead, NCCL, and other
  categories rather than treating one aggregate total as a mechanism.
- Examine total time, call count, and per-call duration. A total-time change can
  be caused by either count or duration.
- Prefer profile-provided `source_attribution` or call stacks over heuristic
  source mapping.
- A kernel appearing or disappearing is evidence of a trace difference, not by
  itself proof of its source-level cause.
- If trace data is unavailable, say so and stop short of a kernel-level causal
  conclusion. Do not fabricate kernel names, timings, or call counts.

For kernel-to-source or release attribution, load `kernel_source_analysis`
after identifying the material kernels. For an end-to-end root-cause analysis,
also load `deep_profiling`.
