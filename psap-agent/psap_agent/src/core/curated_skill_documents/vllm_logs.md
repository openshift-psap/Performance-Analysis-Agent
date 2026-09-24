# vLLM log analysis

Use this workflow when the user asks to read benchmark logs, inspect vLLM
startup/configuration details, or compare logs between versions. For a deep
root-cause investigation, use this after profiler analysis and before source
attribution.

## Retrieve the requested evidence

- For logs from one run, use `fetch_vllm_logs` with the resolved benchmark
  configuration.
- When the user explicitly asks to compare or diff logs, call
  `compare_vllm_logs` directly. Do not substitute release notes, benchmark
  metrics, or another analysis if it fails; report the tool error instead.
- Use `query_performance_metrics` first when the exact configuration or UUID is
  needed to identify the target run.

## Read log facts accurately

`compare_vllm_logs` enriches the comparison with `pinned_dependencies`. Check
and report meaningful changes across PyTorch, Triton, flash-attn, xformers, and
other performance-relevant dependencies rather than mentioning only one.

Use the explicit log fields as ground truth for quantization and attention
backends, TP/PP/DP values, CUDA-graph mode and capture sizes, chunked prefill,
model memory and KV cache, compilation settings, engine initialization, dtype,
and resolved model architecture. Do not infer these details from kernel names
when logs provide the answer.

Report configuration differences exactly as observations. They can be a
potential contributor but are not a confirmed root cause unless profiling data
or a controlled experiment corroborates it. State log-backed facts directly;
use hedged language only for a causal interpretation.

## Link to logs

When a dashboard log link would help, use `generate_dashboard_url` with
`section='view_logs'` and include only the tool-returned URL.
