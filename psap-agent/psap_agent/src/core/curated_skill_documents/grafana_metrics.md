# Grafana metrics

Use this workflow for GPU/DCGM metrics, GPU utilization, memory, temperature,
power, clock speeds, vLLM request rate, latency, KV-cache usage, or Grafana
links. Load `clarification_and_comparison` first if the target benchmark run is
not exact.

## Retrieve a valid benchmark run first

For a named run, resolve model, version, accelerator, and profile before
querying. Use `query_performance_metrics` to identify the benchmark run and
its UUID, then check its `grafana_available` and `grafana_note` fields.

1. If Grafana is unavailable, tell the user that the run lacks the required UUID
   and GuideLLM timestamps; do not attempt to invent GPU metrics.
2. If it is available, call `query_grafana_metrics` with the UUID.

The Grafana tool automatically resolves the benchmark time range and cluster.
Its `data` array is truncated for display. Always report values from the
tool-provided `summary` field rather than calculating statistics from `data`.
Format returned statistics as bullet lists.

## Compare runs

Use `compare_grafana_metrics` for a direct run-to-run comparison. Set
`metric_type='dcgm'` for GPU, utilization, temperature, or power requests;
`metric_type='vllm'` for request-rate, latency, or cache requests; otherwise
use `both`.

Present significant changes by category. For each metric, show its direction,
average percentage change, and the two average values. Omit changes of 5% or
less unless the user requests all metrics; do not list every percentile in a
comparison summary.

## Link to Grafana

After presenting GPU metrics, or when the user asks to view the run in Grafana,
call `generate_grafana_url` with the deployment UUID and include only its
tool-returned URL. The tool selects the appropriate NVIDIA/DCGM or AMD/ROCm
dashboard and benchmark time range automatically. Never construct the URL
yourself.
