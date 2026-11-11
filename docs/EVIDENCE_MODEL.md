# Evidence Model and Diagnostic Triage Specification

This document specifies the formal evidence model, evidence algebra, and diagnostic-triage procedure for **TRIAGE** (*Tool-guided Root-cause Investigation Agent with Grounded Evidence*). It is the companion specification for the ICPE 2027 Industry Track paper *TRIAGE: Evidence-Grounded Diagnosis of LLM Inference Performance Regressions*.

The document deliberately separates two things. Sections 1--6 state the **normative policy** the workflow is designed to enforce. Implementation notes record where the shipped code is weaker than that policy, because the agent enforces most of the procedure through reviewed workflow skills and a pre-delivery critic rather than through hard-coded finite-state checks. Where policy and code differ, the note describes the behavior of `master` at the time of writing; the paper's Threats to Validity section makes the same distinction.

---

## Table of Contents
1. [Inference Run Definition & Comparability](#1-inference-run-definition--comparability)
2. [Performance Metrics and Deltas](#2-performance-metrics-and-deltas)
3. [Evidence Algebra & Provenance](#3-evidence-algebra--provenance)
4. [Diagnostic Claims & Epistemic Status](#4-diagnostic-claims--epistemic-status)
5. [The Confirmation Gate Policy](#5-the-confirmation-gate-policy)
6. [Diagnostic Acceptance Predicate](#6-diagnostic-acceptance-predicate)
7. [Dual Feedback Gating Predicates](#7-dual-feedback-gating-predicates)
8. [Evidence-Ordered Regression Triage Algorithm](#8-evidence-ordered-regression-triage-algorithm)
9. [Context Complexity & Payload Boundedness](#9-context-complexity--payload-boundedness)
10. [Mapping to Implementation Code](#10-mapping-to-implementation-code)

---

## 1. Inference Run Definition & Comparability

Diagnosing LLM serving regressions requires ensuring that comparisons between baseline and candidate releases are experimentally well-posed.

### Run Definition
An LLM inference benchmark or production evaluation run is formalized as a 7-tuple:

$$r = \langle m, h, \pi, b, s^{\mathrm{in}}, s^{\mathrm{out}}, v \rangle$$

where:
* $m$: Model architecture, precision, and weight configuration (e.g., `Nemotron-3-Super-120B-A12B-FP8`).
* $h$: Hardware accelerator topology (e.g., NVIDIA H200 SXM, PCIe configuration, SM count).
* $\pi = \langle \pi_{\mathrm{TP}}, \pi_{\mathrm{DP}}, \pi_{\mathrm{EP}} \rangle$: Parallelism configuration (Tensor, Data, Expert Parallelism).
* $b$: Request concurrency or batch size factor.
* $s^{\mathrm{in}}$: Input prompt sequence length (tokens).
* $s^{\mathrm{out}}$: Output generation sequence length (tokens).
* $v$: Serving runtime engine release version and commit hash (e.g., vLLM `v0.21.0`).

### Strict Comparability Predicate
A diagnostic comparison between a baseline run $r_b$ and a regressed run $r_r$ is valid if and only if both runs satisfy configuration parity:

$$r_b \equiv r_r \iff \left(m, h, \pi, b, s^{\mathrm{in}}, s^{\mathrm{out}}\right)_b = \left(m, h, \pi, b, s^{\mathrm{in}}, s^{\mathrm{out}}\right)_r$$

If $r_b \not\equiv r_r$, the policy halts at Stage 1 ($S_1$: Comparison Fairness) with an explicit configuration mismatch notice, before speculative causal hypotheses can be propagated across asymmetric benchmark points.

**Implementation note.** The shipped comparison tools enforce some run dimensions. `compare_configurations` returns `profile_mismatch` or `tp_mismatch` when known selected profiles or TP values differ, and `compare_versions_comprehensive` checks or filters TP. Neither tool enforces hardware-topology, data-parallelism, or expert-parallelism parity. Concurrency coverage is handled differently. `compare_configurations` compares only up to the highest concurrency both runs share and returns a `concurrency_cap_note` when one run was benchmarked further. That is a fairness mitigation, not a hard halt, so the strict predicate above is the normative rule rather than a description of every path through the tool.

---

## 2. Performance Metrics and Deltas

For any target performance metric $M$ (e.g., aggregate throughput, time-to-first-token [TTFT], inter-token latency [ITL]), the observed fractional delta between baseline $r_b$ and candidate $r_r$ is defined as:

$$\Delta M(r_b, r_r) = \frac{M(r_r) - M(r_b)}{M(r_b)}$$

### Directional Conventions
* **Throughput Metrics** (Tokens/s, Requests/s):
  * $\Delta M < 0$: Performance regression.
  * $\Delta M > 0$: Performance optimization.
* **Latency & Duration Metrics** (TTFT, ITL, Kernel execution time $t_k$, Block duration $t_{\mathrm{block}}$):
  * $\Delta M > 0$: Latency slowdown (regression).
  * $\Delta M < 0$: Latency speedup (optimization).

*Duration slowdown* ($L_r / L_b - 1$) must not be conflated with *throughput drop* ($1 - T_r / T_b$), as non-linear continuous batching and KV-cache scheduling invalidate serial execution assumptions.

### Aggregate Throughput Ratios
Across $n$ standardized concurrency evaluation points, aggregate throughput ratios are evaluated via geometric mean:

$$\rho_G = \exp\left(\frac{1}{n} \sum_{i=1}^n \ln \frac{T_{r,i}}{T_{b,i}}\right)$$

---

## 3. Evidence Algebra & Provenance

Diagnostic tools exposed over the Model Context Protocol (MCP) produce structured evidence items belonging to an evidence set $\mathcal{E}$.

### Evidence Item
An individual evidence item is defined as:

$$e = \langle \tau, \kappa, p, \pi_e \rangle$$

where:
* $\tau \in \{1, 2, 3, 4\}$ denotes the operational evidence tier:
  * **Tier 1 ($\tau=1$)**: Benchmark Discovery and Fair Comparison (S3/CSV stores, aggregate deltas).
  * **Tier 2 ($\tau=2$)**: Deep Profiler and Trace Analysis (PyTorch traces, CUDA kernel timings, decode block intervals).
  * **Tier 3 ($\tau=3$)**: Cluster Telemetry, Logs, and Efficiency (DCGM utilization, SM clocks, engine startup logs).
  * **Tier 4 ($\tau=4$)**: Cross-Repository Source Intelligence (PR diffs, commit lineage, kernel source mapping).
* $\kappa$: Unique tool identifier (e.g., `compare_pytorch_profiles`, `inspect_engine_startup_logs`).
* $p$: Structured JSON/dictionary output payload.
* $\pi_e$: Provenance metadata tuple.

### Provenance Tuple
To guarantee end-to-end traceability, every evidence item carries immutable provenance:

$$\pi_e = \langle \kappa, \alpha, t, \iota \rangle$$

where $\alpha$ is the dictionary of invocation arguments, $t$ is the UTC timestamp of execution, and $\iota$ is the immutable upstream artifact identifier (e.g., Perfetto trace UUID, S3 object ETag, or Git commit SHA).

---

## 4. Diagnostic Claims & Epistemic Status

A diagnostic assertion synthesized by the agent is formalized as a claim triple:

$$q = \langle \phi, \sigma, E_q \rangle$$

where:
* $\phi$: The causal proposition (e.g., *"Un-tuned CUTLASS grouped-GEMM kernel template caused 63.5% throughput drop"*).
* $\sigma$: The discrete epistemic status of the claim.
* $E_q \subseteq \mathcal{E}$: The subset of supporting evidence items cited for proposition $\phi$.

### Epistemic Status Space
The epistemic status $\sigma$ is defined over a 3-state lattice:

$$\sigma \in \{\mathrm{obs}, \mathrm{hyp}, \mathrm{conf}\}$$

1. **$\mathrm{obs}$ (Observed Fact)**: A directly measured, descriptive metric delta without causal attribution (e.g., "P95 ITL increased by 19.1%").
2. **$\mathrm{hyp}$ (Hypothesis)**: A plausible causal conjecture that has not satisfied trace verification (e.g., "Scheduler queueing contention caused degradation").
3. **$\mathrm{conf}$ (Trace-Confirmed Mechanism)**: A causal mechanism rigorously localized to kernel execution traces or runtime configuration flags satisfying the confirmation gate.

---

## 5. The Confirmation Gate Policy

To suppress **diff bias** (the tendency of LLMs to hallucinate root causes from prominent source diffs without checking runtime relevance), TRIAGE requires a strict confirmation gate before any diagnostic claim can transition from hypothesis ($\mathrm{hyp}$) to confirmed status ($\mathrm{conf}$):

$$\sigma(q) = \mathrm{conf} \iff \exists e \in E_q : \tau(e) \in \{2, 3\} \land \mathrm{localises}(e, \phi) \land \neg \exists e' \in E_q : \mathrm{contradicts}(e', \phi)$$

### Formal Conditions
1. **Runtime Grounding ($\tau(e) \in \{2, 3\}$)**: At least one cited evidence item must originate from Tier 2 (execution traces) or Tier 3 (runtime telemetry and configuration). A claim supported exclusively by Tier 4 source code ($\tau=4$) cannot be confirmed, because source diffs demonstrate what changed in repository history, not what actually executed on the GPU.
2. **Bottleneck Localization ($\mathrm{localises}(e, \phi)$)**: The evidence item must mathematically isolate the claimed execution stall (e.g., demonstrating that kernel execution duration $\Delta t_k$ exceeds baseline variance by $>3\sigma$ or $>5\%$, or verifying a disabled autotuning flag in startup logs).
3. **Absence of Contradiction ($\neg\exists e' : \mathrm{contradicts}(e', \phi)$)**: No cited evidence item may contradict the proposition (e.g., an execution trace showing the suspect kernel or dispatch routine was never invoked).

**Implementation note.** The paper draws the same boundary: `localises` and `contradicts` are procedural judgments guided by workflow skills rather than compiler-verified assertions, so this equation formalizes an operational review policy, not a formal verification guarantee. In the current code the pre-delivery critic reviews evidence support and internal consistency but does not test the Tier 2/3 condition explicitly, and reflection can be disabled or skipped after an aborted pass. Hardening the gate into runtime middleware is pending.

---

## 6. Diagnostic Acceptance Predicate

A delivered diagnostic report $D = \{q_1, \dots, q_k\}$ is formally accepted ($A(D) = 1$) under external engineering evaluation if and only if its primary causal claim $q^*$ satisfies:

$$A(D) = \mathbb{I}\left[\sigma(q^*) = \mathrm{conf} \land \mathrm{fix}(q^*) = \mathrm{merged} \land \mathrm{panel}(q^*) = 1\right]$$

where:
* $\sigma(q^*) = \mathrm{conf}$: The agent's claim satisfied the trace-level confirmation gate.
* $\mathrm{panel}(q^*) = 1$: Independent human performance engineers and maintainers external to the authors review and adjudicate the trace evidence as factually correct.
* $\mathrm{fix}(q^*) = \mathrm{merged}$: Upstream or downstream maintainers author and merge a patch based on the diagnosis that empirically recovers benchmark throughput.

---

## 7. Dual Feedback Gating Predicates

TRIAGE decouples interactive report delivery from persistent memory updates through dual feedback gates:

```
                  ┌─────────────────────────────────────────┐
                  │          DELIVERED DIAGNOSIS            │
                  └────────────────────┬────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │  ASYNCHRONOUS EVAL    │             │   TOOL EFFICIENCY     │
        │  (Correctness, Ground,│             │   (Tool call counts,  │
        │   Consistency, Compl) │             │    Zero redundancy)   │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │ min(s_f,s_g,s_c,s_i)=1.0            │ s_e=1.0 & n_calls>=5
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │   G_mem = 1 (PASS)    │             │  G_recipe = 1 (PASS)  │
        │  Commit to Mem0/Qdrant│             │ Save to PostgreSQL    │
        └───────────────────────┘             └───────────────────────┘
```

Following delivery, an out-of-band asynchronous evaluator scores the interaction across five normalized dimensions $s = (s_f, s_g, s_c, s_i, s_e) \in [0, 1]^5$:
* $s_f$: Factual correctness (`correctness`)
* $s_g$: Groundedness / anti-hallucination (`hallucination`)
* $s_c$: Completeness (`completeness`)
* $s_i$: Internal quantitative consistency (`internal_consistency`)
* $s_e$: Tool efficiency (`tool_efficiency`)

The parenthetical names are the criterion keys used in the shipped evaluator, given here so the code references in Section 10 resolve.

### Episodic Memory Gate
Admitting an investigation's findings into persistent vector memory (Mem0 / Qdrant) requires a flawless score across all four correctness-related dimensions:

$$G_{\mathrm{mem}} = \mathbb{I}\left[\min\left(s_f, s_g, s_c, s_i\right) = 1.0\right]$$

If any score falls below 1.0, the session is rejected from long-term memory ($G_{\mathrm{mem}} = 0$) to prevent contaminating future investigations with flawed priors.

**Implementation note.** `should_store_memory` in `evaluator.py` applies this predicate over the four gated criteria (`hallucination`, `correctness`, `completeness`, `internal_consistency`) at a threshold of 1.0, but it checks only scores that are present and stores when the score dictionary is empty. The gate therefore fails open: an evaluator outage, or a criterion the evaluator failed to return, admits the interaction to memory instead of rejecting it. That is the opposite direction from the conservative policy above. It is what the shipped code does, not what this document recommends.

### Advisory Recipe Gate
Registering a repeatable sequence of tool calls as an advisory recipe in PostgreSQL requires:

$$G_{\mathrm{recipe}} = \mathbb{I}\left[s_e = 1.0 \land n_{\mathrm{calls}} \ge 5\right]$$

The gate requires a perfect tool-efficiency score across at least five diagnostic calls, and stored recipes remain advisory.

**Implementation note.** The shipped gate is a two-stage check. The post-processing hook in `manager.py` requires `tool_efficiency` to be present and equal to 1.0 before it calls `maybe_generate_skill`; a missing score skips generation. `maybe_generate_skill` in `memory.py` then requires five calls by default (`SKILL_GENERATION_THRESHOLD`, configurable) or four calls when the agent recovered from a tool error, and generation is disabled when `ENABLE_SKILL_DOCUMENTS` is off. The recovery path and the configurable threshold are the implementation's additions; the equation states the default policy.

---

## 8. Evidence-Ordered Regression Triage Algorithm

The following algorithm formalizes the end-to-end execution of TRIAGE across the five investigative stages ($S_1 \to S_5$). Tool names in the pseudocode are schematic; Section 10 maps each stage to the code that implements it.

```python
"""
Algorithm: Evidence-Ordered Regression Triage
Five-stage progression that gates source-diff analysis behind runtime evidence.
"""

def triage_regression(r_b: Run, r_r: Run, S_max: int = 50, T_max: float = 600.0) -> DiagnosticReport:
    start = monotonic()
    evidence = []
    steps = 0

    def budget_remaining() -> float:
        return T_max - (monotonic() - start)

    def call_tool(name: str, **kwargs) -> Evidence:
        nonlocal steps
        remaining = budget_remaining()
        if steps >= S_max or remaining <= 0:
            raise BudgetExhausted(steps=steps, elapsed=T_max - remaining)
        ev = invoke_mcp_tool(name, timeout=remaining, **kwargs)
        evidence.append(ev)
        steps += 1
        return ev

    # -------------------------------------------------------------
    # STAGE 1: Comparison Fairness (DR1)
    # -------------------------------------------------------------
    is_comparable, mismatches = check_comparability(r_b, r_r)
    if not is_comparable:
        return DiagnosticReport(
            status="CLARIFICATION_REQUIRED",
            summary=f"Runs are not comparable: {mismatches}",
            epistemic_status="obs",
            evidence=evidence,
        )

    try:
        # Tier 1: macro throughput and latency deltas
        call_tool("compute_benchmark_deltas", r_b=r_b, r_r=r_r)

        # ---------------------------------------------------------
        # STAGE 2: Micro kernel profiling (Tier 2)
        # ---------------------------------------------------------
        runtime_localized = False
        kernel_ev = call_tool("compare_pytorch_profiles", r_b=r_b, r_r=r_r, top_k=10)
        for k in kernel_ev.payload.get("regressed_kernels", []):
            if k.delta_duration_pct > 5.0 and k.p_value < 0.01:
                runtime_localized = True
                break

        # ---------------------------------------------------------
        # STAGE 3: Decode block and bubble analysis (Tier 2)
        # ---------------------------------------------------------
        if not runtime_localized:
            block_ev = call_tool("analyze_decode_block_bubbles", r_b=r_b, r_r=r_r)
            runtime_localized = bool(block_ev.payload.get("bubble_detected"))

        # ---------------------------------------------------------
        # STAGE 4: Engine logs and telemetry (Tier 3)
        # Localization signals: a configuration override in startup
        # logs, a pinned or updated runtime dependency, or throttling.
        # ---------------------------------------------------------
        if not runtime_localized:
            logs_ev = call_tool("inspect_engine_startup_logs", r_b=r_b, r_r=r_r)
            runtime_localized = bool(
                logs_ev.payload.get("config_override_detected")
                or logs_ev.payload.get("pin_mismatch")
                or logs_ev.payload.get("clock_throttling_detected")
            )

        # ---------------------------------------------------------
        # STAGE 5: Cross-repository source inspection (Tier 4)
        # Admissible only after a runtime observation localizes the
        # bottleneck; otherwise the diff is not diagnostic.
        # ---------------------------------------------------------
        if runtime_localized:
            call_tool("cross_repo_git_diff", r_b=r_b, r_r=r_r, focus=evidence)

    except BudgetExhausted as exc:
        return DiagnosticReport(
            status="BUDGET_EXHAUSTED",
            summary=f"Stopped after {exc.steps} steps; report covers partial evidence.",
            epistemic_status="hyp",
            evidence=evidence,
        )

    draft = synthesize_diagnostic_claims(evidence)

    # -------------------------------------------------------------
    # SYNCHRONOUS PRE-DELIVERY CRITIQUE (Gated Delivery)
    # -------------------------------------------------------------
    critic_passed = False
    for _ in range(MAX_REFLECTION_ITERATIONS):   # default 2
        critique = evaluate_critic(draft, evidence)
        if critique.passed:
            critic_passed = True
            break
        draft = revise_report(draft, critique.critique)

    if not critic_passed:
        return DiagnosticReport(
            status="EVIDENCE_GAP",
            summary="Critic reported unresolved evidence gaps after the revision limit.",
            epistemic_status="hyp",
            evidence=evidence,
            report=draft,
        )
    return draft
```

**Implementation note.** The shipped agent does not implement `triage_regression` as one function. It runs a LangGraph ReAct loop under a recursion limit of 50 (`agent.py`), so the step bound is enforced by the graph rather than by a per-stage counter. The time bound is the agent response timeout in `manager.py` (`AGENT_RESPONSE_TIMEOUT_SECONDS`, default 600 s; 900 s for the Gemini 4 Argon configuration), which wraps an entire agent pass rather than each tool call. The `runtime_localized` name is ours, but the payload flags it reads are returned by the tools. Deciding that a flag localizes the bottleneck to the suspected layer remains a skill-guided judgment, and Stage 5 admissibility is enforced by the workflow skill's instructions rather than by a runtime guard. The critic loop runs at most `MAX_REFLECTION_ITERATIONS` times (default 2). The current code then delivers the latest draft; the evidence-gap halt above is the specified behavior and is not yet implemented.

---

## 9. Context Complexity & Payload Boundedness

A naive agent ingesting multi-gigabyte PyTorch trace JSONs or multi-thousand commit histories exhausts context windows ($\mathcal{O}(|E|)$ token consumption).

TRIAGE bounds context consumption per triage pass to:

$$\mathcal{O}(S_{\max} \cdot \bar{p})$$

and across at most $P_{\max} \le 2$ pre-delivery critique revision cycles to:

$$\mathcal{O}(P_{\max} \cdot S_{\max} \cdot \bar{p})$$

where:
* $S_{\max} = 50$: Maximum allowed tool execution steps per pass.
* $\bar{p} \approx 1{,}200$ tokens: Target payload budget per tool invocation, enforced by hierarchical summarization and top-$k$ aggregation.
* $P_{\max} \le 2$: Maximum pre-delivery critique revision cycles.

Payloads that exceed $\bar{p}$ are summarized or truncated and marked with explicit partial-evidence indicators, so the agent does not mistake a summary view for complete evidence.

**Implementation note.** The current code bounds payloads in characters rather than tokens: `context_budget.py` truncates each tool result to `_MAX_TOOL_RESULT_CHARS = 20,000` characters and each assembled message history to `_MAX_MESSAGE_CHARS = 240,000` characters, keeping the head and tail of an oversized result and inserting the notice `[Tool result shortened for context. Re-run with narrower filters if needed.]`. Final-answer context is narrower still: at most 24 tool results, each cut to 5,000 characters. The 1,200-token target and the partial-evidence markers are specified behavior that the implementation has not yet adopted; until then, the truncation notice is the signal that the agent is looking at a summary view.

---

## 10. Mapping to Implementation Code

| Formal Concept | Mathematical Symbol | Implementation Location in Repository |
| :--- | :--- | :--- |
| **Comparability Check** | $r_b \equiv r_r$ (Section 1) | [`compare_performance_tool.py`](../psap-mcp-server/psap_mcp_server/src/tools/compare_performance_tool.py) |
| **Evidence Tiers ($T_1$--$T_4$)** | $\tau \in \{1,2,3,4\}$ | Tool registration in [`psap_mcp_server/src/mcp.py`](../psap-mcp-server/psap_mcp_server/src/mcp.py) |
| **Workflow Skills ($S_1$--$S_5$)**| Section 8 | [`psap_agent/src/core/curated_skills.py`](../psap-agent/psap_agent/src/core/curated_skills.py) |
| **Confirmation Gate** | $\sigma(q) = \mathrm{conf}$ (Section 5) | Policy: workflow skills plus pre-delivery critique, not a runtime invariant. The current critic checks evidence support, not the Tier 2/3 condition. |
| **Pre-Delivery Critic** | $\mathrm{Critic}(D, E)$ | [`reflection.py#run_critic`](../psap-agent/psap_agent/src/core/reflection.py), looped in [`manager.py`](../psap-agent/psap_agent/src/core/manager.py) |
| **Asynchronous Evaluator** | $s \in [0, 1]^5$ | [`psap_agent/src/core/evaluator.py`](../psap-agent/psap_agent/src/core/evaluator.py) |
| **Memory Gate** | $G_{\mathrm{mem}} = \mathbb{I}[\min=1.0]$ | [`evaluator.py#should_store_memory`](../psap-agent/psap_agent/src/core/evaluator.py#L390); fails open on empty scores |
| **Recipe Gate** | $G_{\mathrm{recipe}} = \mathbb{I}[s_e=1.0 \land n \ge 5]$ | Two-stage: `tool_efficiency` check in [`manager.py`](../psap-agent/psap_agent/src/core/manager.py), complexity test in [`memory.py#maybe_generate_skill`](../psap-agent/psap_agent/src/core/memory.py) |
| **Execution Budgets** | $S_{\max}, T_{\max}$ | Step bound: recursion limit 50 in [`agent.py`](../psap-agent/psap_agent/src/core/agent.py). Payload bound: [`context_budget.py`](../psap-agent/psap_agent/src/core/context_budget.py). No wall-clock deadline. |
