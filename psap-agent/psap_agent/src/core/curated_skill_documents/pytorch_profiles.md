# PyTorch profiler analysis

Use this workflow for profile availability, kernel hotspots, kernel regressions,
trace structure, CUDA timelines, or profiler-based performance explanations.

## Resolve profiles efficiently

Profile availability is dynamic; never assume an unspecified model, TP,
workload, rank, or version has a trace. When exact selectors are known, call
`analyze_pytorch_profile` or `compare_pytorch_profiles` directly; these tools
resolve the profile index themselves. If a tool returns multiple exact model,
TP, version, or workload candidates, retry with the selected candidate rather
than issuing a separate listing call. Use `list_available_profiles` when
availability itself is the question or the returned candidates do not resolve
the ambiguity. Use `check_profile_status` for a quick readiness diagnostic.

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
- Avoid repeating a listing or profile comparison after the tool has already
  returned the needed availability or kernel deltas. Request additional profile
  analysis only to answer a distinct unresolved question.

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
