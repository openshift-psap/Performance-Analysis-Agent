# vLLM performance triage

Use this workflow when the user asks how to diagnose, triage, tune, optimize,
or remediate vLLM inference performance, including TTFT, ITL, saturation,
KV-cache/VRAM health, sequence lengths, distributed inference, or serving
configuration.

Call `get_vllm_performance_triage_guide` first, using a topic when the request
is narrow. Treat the returned guide as the primary source of truth for the
diagnostic order, criteria, Prometheus/PromQL queries, and remediation advice.

After presenting tool-backed guidance, you may add clearly labeled general
reasoning or best practices that the guide does not cover. Do not present that
additional reasoning as a tool-verified diagnosis. When a specific benchmark
run is involved, load the relevant benchmark, Grafana, or vLLM-log skill and
retrieve evidence before claiming a root cause.
