"""MCP tool providing vLLM performance triage guidance.

Structured reference material from the Red Hat Developer blog post
"5 steps to triage vLLM performance" (March 9, 2026).

Source: https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance
"""

from typing import Any, Dict, Optional

_SOURCE_URL = "https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance"

_TRIAGE_GUIDE: Dict[str, Dict[str, Any]] = {
    "workload_profiles": {
        "title": "Before You Start: Define What Success Looks Like",
        "step": 0,
        "summary": (
            "Before reviewing diagnostic metrics, identify your workload profile. "
            "Your optimization strategy depends on which profile applies."
        ),
        "profiles": {
            "throughput_sensitive": {
                "label": "Throughput-sensitive (offline/batch)",
                "description": (
                    "Processing large volumes of requests for data extraction, summarization, "
                    "or document classification. Individual request latency matters less than "
                    "total job completion time. Optimize for requests per second."
                ),
            },
            "latency_sensitive": {
                "label": "Latency-sensitive (online/interactive)",
                "description": (
                    "A human user is waiting for a streamed response (chatbot, code completion, "
                    "real-time voice). Time to first token (TTFT) and inter-token latency (ITL) "
                    "directly affect the user experience. Optimize for low latency at expected concurrency."
                ),
            },
            "bursty": {
                "label": "Bursty (semi-online)",
                "description": (
                    "Load is unpredictable, switching from idle to spiking. You need flexible "
                    "scaling and fast cold starts more than raw throughput or minimum latency."
                ),
            },
        },
    },
    "ttft_vs_itl": {
        "title": "Step 1: Isolate the Symptom — TTFT vs ITL",
        "step": 1,
        "summary": (
            "Total E2E latency tells you IF requests are slow. To understand WHY, break it "
            "into TTFT (queuing + prefill) and ITL (decode phase). These two phases interact: "
            "long prefills on a saturated server can starve decode steps of other in-flight "
            "requests, causing ITL spikes. vLLM mitigates this with chunked prefill (enabled "
            "by default), which breaks long prefills into smaller chunks that interleave with "
            "decode steps."
        ),
        "definitions": {
            "ttft": (
                "Time to First Token — time from when the server receives the request until the "
                "first token is generated. Includes queuing delay plus the prefill phase. On a "
                "saturated server, TTFT is often dominated by queuing time."
            ),
            "itl": (
                "Inter-Token Latency — time between each subsequent token during the decode phase. "
                "Determines how smooth text appears when streaming. High ITL typically suggests "
                "hardware bandwidth limits or high concurrency within a batch."
            ),
        },
        "diagnostic_logic": (
            "If E2E latency and TTFT are high but ITL is low, the server is likely overloaded "
            "(requests queuing) or input prompts are very long (long prefill). If ITL is high, "
            "the decode phase is bottlenecked — check hardware bandwidth and batch sizes."
        ),
        "promql": {
            "e2e_latency_p50": (
                'histogram_quantile(0.5, sum by(model_name, pod, le)(\n'
                '    rate(vllm:e2e_request_latency_seconds_bucket{}[1m]))\n'
                ')'
            ),
            "ttft_p50": (
                'histogram_quantile(0.5, sum by(model_name, pod, le)(\n'
                '    rate(vllm:time_to_first_token_seconds_bucket{}[1m]))\n'
                ')'
            ),
            "itl_p50": (
                'histogram_quantile(0.5, sum by(model_name, pod, le)(\n'
                '    rate(vllm:inter_token_latency_seconds_bucket{}[1m]))\n'
                ')'
            ),
        },
        "note": (
            "These metrics are server-side and do not include network or ingress latency. "
            "If client-measured latency is significantly higher, the gap is in the network "
            "path, load balancer, or proxy layer."
        ),
    },
    "saturation": {
        "title": "Step 2: Detect Server Saturation",
        "step": 2,
        "summary": (
            "Monitor the relationship between running and waiting requests to determine if "
            "the server is saturated. vLLM uses continuous batching to maximize throughput, "
            "but as running requests increase, individual latency rises."
        ),
        "metrics": {
            "running_requests": (
                "vllm:num_requests_running — requests actively being processed on the GPU (batch size)."
            ),
            "waiting_requests": (
                "vllm:num_requests_waiting — requests in the queue due to resource limits (queue depth)."
            ),
        },
        "diagnostic_logic": (
            "If num_requests_waiting is consistently above zero, queuing time drives high TTFT. "
            "If it is zero but TTFT remains high, the delay is from the prefill phase itself "
            "(processing long prompts), meaning the server is compute-bound."
        ),
        "log_examples": {
            "healthy": (
                "Engine 000: Running: 39 reqs, Waiting: 0 reqs, GPU KV cache usage: 68.9%, "
                "Prefix cache hit rate: 29.7%"
            ),
            "saturated": (
                "Engine 000: Running: 60 reqs, Waiting: 21 reqs, GPU KV cache usage: 99.8%, "
                "Prefix cache hit rate: 32.2%"
            ),
        },
        "note": (
            "A 'healthy' batch size depends on hardware and model. For a small model on a "
            "large GPU, vLLM can process hundreds of requests in a batch with acceptable latency."
        ),
    },
    "kv_cache": {
        "title": "Step 3: Evaluate VRAM and KV Cache Health",
        "step": 3,
        "summary": (
            "GPU memory is the primary constraint for concurrent request capacity. It stores "
            "model weights and the dynamic KV cache. If model weights are too large, there "
            "isn't enough KV cache space to process requests concurrently — this 'memory "
            "pressure' is the most common cause of queued waiting requests."
        ),
        "metrics": {
            "kv_cache_usage": (
                "vllm:kv_cache_usage_perc — percentage of KV cache in use. If consistently "
                "near 100%, new requests are forced to wait."
            ),
            "preemptions": (
                "vllm:num_preemptions_total — cumulative count of requests stopped mid-generation "
                "to free memory. If climbing, reduce --max-num-seqs."
            ),
        },
        "remediation": [
            "Quantize model weights (FP8, AWQ, GPTQ) to reduce VRAM footprint.",
            "Quantize KV cache with --kv-cache-dtype fp8 to double effective cache capacity.",
            "Use a smaller model if the task allows it.",
            "Use more or larger GPUs to free up space for KV cache.",
        ],
        "prefix_caching": (
            "vLLM prefix caching (enabled by default) stores repeated prompt prefixes so they "
            "don't need re-processing. A high hit rate saves compute and reduces TTFT. "
            "PromQL: rate(vllm:prefix_cache_hits_total[1m]) / rate(vllm:prefix_cache_queries_total[1m])"
        ),
        "startup_check": (
            "At startup, vLLM logs available KV cache memory. If very low (e.g., 4.4 GiB on "
            "an 80 GB GPU for a 32B FP16 model), the server will struggle with concurrent requests."
        ),
    },
    "sequence_lengths": {
        "title": "Step 4: Analyze Request Sequence Lengths",
        "step": 4,
        "summary": (
            "Latency metrics need the context of token counts. Input sequence length (ISL) and "
            "output sequence length (OSL) determine the KV cache footprint and compute time. "
            "Prefill and decode are highly asymmetrical: processing a 1000-token prompt (prefill) "
            "is parallel and typically takes less than a second, but generating 1000 tokens "
            "(decode) is sequential and can take 10–100+ seconds."
        ),
        "relationships": {
            "isl_and_ttft": (
                "Prefill time scales with input length. Long prompts mean naturally higher TTFT."
            ),
            "osl_and_itl": (
                "Generation time scales linearly with output tokens. At 40ms ITL, 1000 tokens "
                "takes 40 seconds but 100 tokens takes 4 seconds."
            ),
        },
        "promql": {
            "prompt_length_p95": "histogram_quantile(0.95, rate(vllm:request_prompt_tokens_bucket[5m]))",
            "output_length_p95": "histogram_quantile(0.95, rate(vllm:request_generation_tokens_bucket[5m]))",
            "output_length_p50": "histogram_quantile(0.5, rate(vllm:request_generation_tokens_bucket[5m]))",
        },
        "rag_note": (
            "In RAG pipelines, a short user query is expanded into a massive prompt with "
            "retrieved context, leading to high TTFT. Structure prompts so shared system "
            "prompt and static instructions come first — vLLM's prefix caching will cache "
            "and reuse these common prefixes across requests."
        ),
    },
    "distributed_inference": {
        "title": "Step 5: Review Distributed Inference Strategy",
        "step": 5,
        "summary": (
            "When a model is too large for one GPU, tensor parallelism (TP) shards computation "
            "across GPUs. This frees KV cache space per GPU but introduces communication "
            "overhead. If ITL is unexpectedly high, check GPU topology with nvidia-smi topo -m."
        ),
        "key_points": [
            "GPUs communicating over PCIe instead of NVLink can severely degrade performance.",
            (
                "Use the minimum TP degree that fits the model, then scale out with independent "
                "replicas for additional capacity."
            ),
            "Replicas have zero inter-replica communication overhead and scale linearly.",
            (
                "Two independent replicas at TP=2 typically outperform one instance at TP=4 in "
                "both throughput and scheduling flexibility."
            ),
            (
                "If hardware lacks NVLink entirely, keep TP=1 per GPU and use pure data or "
                "pipeline parallelism."
            ),
        ],
    },
    "remediation": {
        "title": "Remediation Steps",
        "step": 6,
        "summary": (
            "High-impact optimization paths once the bottleneck is identified. Performance "
            "issues are rarely solved by tuning CLI parameters alone — significant improvements "
            "usually require addressing underlying hardware or memory constraints."
        ),
        "strategies": {
            "right_size_model": {
                "label": "Right-size your model",
                "description": (
                    "A fine-tuned 8B model often matches or exceeds a general-purpose 70B model "
                    "on a narrow task at a fraction of the latency and cost. If the workload is "
                    "well-defined (classification, extraction, summarization), a smaller task-specific "
                    "model may eliminate the performance problem entirely."
                ),
            },
            "scale_hardware": {
                "label": "Scale your hardware",
                "description": (
                    "If consistently saturated (high num_requests_waiting), add more replicas. "
                    "If ITL is the bottleneck, move to GPUs with higher memory bandwidth (e.g., "
                    "L40S to H100)."
                ),
            },
            "quantization": {
                "label": "Use quantization",
                "description": (
                    "FP16 to FP8 halves the VRAM footprint and improves ITL. Use "
                    "--kv-cache-dtype fp8 to further double effective cache capacity. Ensure "
                    "hardware natively supports the chosen format (e.g., avoid FP8 on A100s)."
                ),
            },
            "speculative_decoding": {
                "label": "Implement speculative decoding",
                "description": (
                    "If ITL is the primary bottleneck on fast hardware with a quantized model, "
                    "speculative decoding can generate multiple tokens per forward pass. vLLM "
                    "supports native MTP, EAGLE3, draft models, and n-gram speculation."
                ),
            },
            "refine_distribution": {
                "label": "Refine distribution",
                "description": (
                    "Reduce tensor parallel (TP) degree to cut interconnect overhead. Two "
                    "replicas at TP=2 typically outperform one at TP=4. Without NVLink, keep "
                    "TP=1 and use pure data or pipeline parallelism."
                ),
            },
        },
    },
}

_TOPIC_ALIASES: Dict[str, str] = {
    "workload": "workload_profiles",
    "workloads": "workload_profiles",
    "profiles": "workload_profiles",
    "ttft": "ttft_vs_itl",
    "itl": "ttft_vs_itl",
    "latency": "ttft_vs_itl",
    "e2e": "ttft_vs_itl",
    "saturation": "saturation",
    "queue": "saturation",
    "running": "saturation",
    "waiting": "saturation",
    "kv_cache": "kv_cache",
    "kv": "kv_cache",
    "vram": "kv_cache",
    "memory": "kv_cache",
    "cache": "kv_cache",
    "preemption": "kv_cache",
    "sequence_lengths": "sequence_lengths",
    "isl": "sequence_lengths",
    "osl": "sequence_lengths",
    "tokens": "sequence_lengths",
    "rag": "sequence_lengths",
    "distributed": "distributed_inference",
    "distributed_inference": "distributed_inference",
    "tp": "distributed_inference",
    "tensor_parallelism": "distributed_inference",
    "scaling": "distributed_inference",
    "replicas": "distributed_inference",
    "remediation": "remediation",
    "optimization": "remediation",
    "fix": "remediation",
    "improve": "remediation",
    "quantization": "remediation",
    "speculative": "remediation",
}


async def get_vllm_performance_triage_guide(
    topic: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve vLLM performance triage guidance from the Red Hat Developer blog.

    TOOL_NAME=get_vllm_performance_triage_guide
    DISPLAY_NAME=vLLM Performance Triage Guide
    USECASE=When users ask how to diagnose, triage, or improve vLLM inference performance, understand TTFT vs ITL, detect server saturation, evaluate KV cache health, analyze sequence lengths, review distributed inference strategy, or get remediation advice.
    INSTRUCTIONS=1. Call with no topic to get the full 5-step triage guide, 2. Call with a specific topic to get targeted guidance (e.g., "ttft", "saturation", "kv_cache", "sequence_lengths", "distributed_inference", "remediation")
    INPUT_DESCRIPTION=topic (Optional[str]): A specific triage area to retrieve. Valid topics: "workload_profiles", "ttft_vs_itl" (or "ttft", "itl", "latency"), "saturation" (or "queue"), "kv_cache" (or "vram", "memory"), "sequence_lengths" (or "isl", "osl"), "distributed_inference" (or "tp", "scaling"), "remediation" (or "optimization", "fix"). Omit for the full guide.
    OUTPUT_DESCRIPTION=Dictionary with structured triage guidance including step titles, summaries, relevant Prometheus/PromQL queries, diagnostic logic, and remediation advice.
    EXAMPLES=get_vllm_performance_triage_guide(), get_vllm_performance_triage_guide("ttft"), get_vllm_performance_triage_guide("kv_cache"), get_vllm_performance_triage_guide("remediation")
    PREREQUISITES=None — this is a reference tool that requires no prior data or tool calls.
    RELATED_TOOLS=query_grafana_metrics, compare_grafana_metrics, fetch_vllm_logs, compare_vllm_logs

    Args:
        topic: Optional topic filter. If provided, returns guidance for that
            specific triage step. If omitted, returns the full guide.

    Returns:
        Dictionary with structured triage guidance and source attribution.
    """
    if topic is None:
        return {
            "status": "success",
            "operation": "get_vllm_performance_triage_guide",
            "scope": "full_guide",
            "source": _SOURCE_URL,
            "guide": _TRIAGE_GUIDE,
        }

    key = _TOPIC_ALIASES.get(topic.lower().strip(), topic.lower().strip())

    if key not in _TRIAGE_GUIDE:
        return {
            "status": "error",
            "operation": "get_vllm_performance_triage_guide",
            "error": f"Unknown topic: '{topic}'",
            "available_topics": list(_TRIAGE_GUIDE.keys()),
            "aliases": {k: v for k, v in _TOPIC_ALIASES.items() if v != k},
            "message": (
                f"Topic '{topic}' not recognized. Use one of the available "
                "topics or aliases, or omit the topic for the full guide."
            ),
        }

    return {
        "status": "success",
        "operation": "get_vllm_performance_triage_guide",
        "scope": "single_topic",
        "topic": key,
        "source": _SOURCE_URL,
        "guide": {key: _TRIAGE_GUIDE[key]},
    }
