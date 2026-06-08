# Memory Layer & Analysis Graph Pipeline

This document describes the persistent memory system and the LangGraph analysis pipelines added on top of the original PSAP agent. The original project provides an interactive chat agent with MCP tools. The memory layer adds **automated multi-level analysis** with persistent knowledge that carries across runs.

---

## Table of Contents

1. [Overview](#overview)
2. [Database Schema](#database-schema)
3. [Memory Module (`core/memory/`)](#memory-module)
4. [Graph Module (`core/graph/`)](#graph-module)
5. [Prompt Templates (`prompts/`)](#prompt-templates)
6. [Analysis Pipelines Step by Step](#analysis-pipelines-step-by-step)
7. [New API Endpoints](#new-api-endpoints)
8. [Running the New Endpoints](#running-the-new-endpoints)
9. [Configuration](#configuration)
10. [Migrations](#migrations)

---

## Overview

The memory layer enables the agent to **remember findings across analysis runs**. Instead of treating each analysis as independent, the system:

- Stores structured facts extracted from every analysis report
- Injects relevant prior knowledge into subsequent analysis runs
- Tracks issue lifecycles across versions (active → resolved → stale)
- Shares cross-configuration observations via ephemeral "red flags"
- Periodically consolidates the fact store to prevent unbounded growth

The design is inspired by the **"Dreams" consolidation pattern**: extract → validate → reconcile → store → consolidate.

---

## Database Schema

Five tables, supported on both **PostgreSQL** (production) and **SQLite** (local dev):

| Table | Scope | Purpose |
|---|---|---|
| `config_findings` | Per-configuration (model + accelerator + TP + profile) | Granular performance facts with category, status, confidence, root cause, related PRs/kernels |
| `model_summaries` | Per-model | Rolled-up summaries synthesized from config-level findings |
| `version_summaries` | Per-version | Top-level summary across all models for a version |
| `analysis_reports` | Per-analysis | Full report text archive |
| `consolidation_log` | Per-consolidation run | Audit trail: how many facts were pruned, merged, folded |

Every fact row carries:
- `content_hash` (MD5) for exact-text deduplication
- `status`: `active` | `resolved` | `stale`
- `confidence`: `high` | `medium` | `low`
- `source`: `agent_analysis` | `red_flag_consolidation`
- `issue_thread_id`: optional grouping for related facts
- `run_id`: links back to the analysis run that produced it

**File:** `core/memory/db.py`

---

## Memory Module

Located at `core/memory/`. Four files:

### `db.py` — Connection & Schema Management

- Auto-detects backend: PostgreSQL if `POSTGRES_HOST` and `POSTGRES_DB` are set, otherwise falls back to SQLite (`memory.db`)
- `ensure_schema()` — creates all five tables idempotently on startup
- `content_hash(text)` — MD5 hash for dedup
- Thread-safe SQLite via a global lock

### `queries.py` — Read/Write Operations

All functions are async and backend-agnostic (Postgres via `asyncpg`, SQLite via `sqlite3`):

| Function | Description |
|---|---|
| `get_config_facts(version, model, accelerator, profile, ...)` | Query config-level findings with optional filters |
| `get_model_summary(version, model)` | Query model-level summaries |
| `get_version_summary(version)` | Query version-level summaries |
| `upsert_config_finding(finding)` | Insert or skip (dedup on content_hash per scope) |
| `upsert_model_summary(summary)` | Insert or update (dedup on version + model) |
| `upsert_version_summary(summary)` | Insert or update (dedup on version) |
| `store_report(report)` | Insert full report text |
| `update_fact_status(id, status)` | Change status (active → resolved → stale) |
| `delete_stale_facts(table)` | Hard-delete facts marked stale |

### `injection.py` — Memory Context Builder

`build_memory_context()` assembles the memory block injected into the agent's prompt before analysis. It queries three sources and enforces a **~2000 token budget**:

| Source | Budget | What It Contains |
|---|---|---|
| Prior facts from baseline version | ~1000 tokens | Active issues to verify, resolved issues for context |
| Model summary from baseline | ~500 tokens | High-level model behavior from prior version |
| Red flags from current version run | ~500 tokens | Cross-config observations from other analyses in this run |

Output is a structured `## Prior Memory` block with sections for active issues, resolved issues, model summary, and red flags.

### `red_flags.py` — Ephemeral Cross-Config Working Memory

`RedFlagsManager` manages markdown files on disk (`red_flags_{version}.md`). These are NOT in the database — they're fast, ephemeral, and designed for intra-run communication.

**Lifecycle:**
1. Created by the first config analysis in a version run
2. Appended to after each config analysis (model-specific + ALL MODELS sections)
3. Read by subsequent config analyses (relevant flags injected into context)
4. Folded into persistent facts during consolidation
5. Archived to `red_flags/archive/` after consolidation

**Structure:**
```markdown
## Red Flags - BENCH-2.5

### nova-ai/Helios-34B
- [kernel] fused_gemm dispatch changed, ~10% latency regression on Accel-X900

### ALL MODELS
- [runtime] ServX-1.9.0 upgrades MLFrame from 3.1 to 3.2
```

---

## Graph Module

Located at `core/graph/`. Contains four LangGraph `StateGraph` pipelines and six shared nodes.

### State (`state.py`)

All pipelines share `AnalysisState` (a `TypedDict`):

```python
class AnalysisState(TypedDict, total=False):
    # Input
    analysis_level: str       # "config" | "model" | "version" | "consolidation"
    mode: str                 # "shallow" | "deep"
    regression_detected: bool
    run_id: str
    rhaiis_version: str
    baseline_version: str
    config: dict              # model, accelerator, tp, profile, concurrency

    # Memory injection
    prior_facts: str
    red_flags: str

    # Agent output
    messages: list
    report: str

    # Fact pipeline
    extracted_facts: list[dict]
    validated_facts: list[dict]
    reconciled_facts: list[dict]
    new_red_flags: list[str]

    # Metrics
    facts_stored_count: int
    red_flags_added_count: int
```

### Pipelines

#### 1. Config Graph (`config_graph.py`)

The main workhorse — uses MCP tools and a full ReAct agent loop.

```
START → inject_memory → react_agent → extract_facts → judge_facts
      → reconcile_facts → store_facts → update_red_flags → END
```

- **inject_memory**: Queries prior facts + red flags, builds memory context
- **react_agent**: Runs the ReAct loop with MCP tools, produces an analysis report
- **extract_facts**: LLM extracts structured facts + red flags from the report
- **judge_facts**: LLM-as-judge validates each fact against the report
- **reconcile_facts**: LLM merges new facts with existing memory (insert/update/noop)
- **store_facts**: Writes to DB with semantic dedup gate
- **update_red_flags**: Appends cross-config red flags to the markdown file

#### 2. Model Graph (`model_graph.py`)

Synthesis-only — no MCP tools, single LLM call to synthesize config findings.

```
START → inject_facts → synthesize_report → extract_facts → judge_facts
      → reconcile_facts → store_facts → END
```

#### 3. Version Graph (`version_graph.py`)

Same pattern as model graph but synthesizes model summaries into a version rollup.

```
START → inject_facts → synthesize_report → extract_facts → judge_facts
      → reconcile_facts → store_facts → END
```

#### 4. Consolidation Graph (`consolidation_graph.py`)

Dreams-like 4-phase memory cleanup. Runs after a full version pipeline.

```
START → load_all_facts → consolidate → fold_red_flags → write_back
      → log_summary → END
```

- **load_all_facts**: Reads all facts from all tables + red flags file
- **consolidate**: LLM reviews everything and produces a mutation plan (merge/compress/delete/mark_stale)
- **fold_red_flags**: Promotes valuable red flags to permanent `config_findings`
- **write_back**: Applies all mutations to the DB, archives the red flags file
- **log_summary**: Writes consolidation audit trail

### Shared Nodes (`nodes/`)

| Node | File | Used By |
|---|---|---|
| `inject_config_memory` / `inject_model_facts` / `inject_version_facts` | `inject_memory.py` | Config / Model / Version graphs |
| `extract_facts` | `extract_facts.py` | All graphs except consolidation |
| `judge_facts` | `judge_facts.py` | All graphs except consolidation |
| `reconcile_facts` | `reconcile_facts.py` | All graphs except consolidation |
| `store_facts` | `store_facts.py` | All graphs except consolidation |
| `update_red_flags` | `update_red_flags.py` | Config graph only |

### Two-Model Architecture

The pipeline uses two LLM instances:

| Model | Config Key | Default | Used For |
|---|---|---|---|
| Main model | `GEMINI_MODEL` | `gemini-3-flash-preview` | ReAct agent loop (config graph), synthesis (model/version graphs) |
| Fact model | `FACT_MODEL` | `gemini-2.0-flash` | Extraction, judging, reconciliation, dedup check, consolidation |

The fact model is smaller/cheaper for the high-volume structured-output calls.

---

## Prompt Templates

Located at `prompts/`. Each file defines a system prompt and a user-prompt builder for one step of the pipeline:

| File | System Prompt | Purpose |
|---|---|---|
| `extraction.py` | `FACT_EXTRACTION_SYSTEM` | Extract structured facts + red flags from a report |
| `judge.py` | `FACT_VALIDATION_SYSTEM` | Validate extracted facts against the original report |
| `dedup.py` | `DEDUP_CHECK_SYSTEM` | Semantic duplicate check before DB insert |
| `reconciliation.py` | `FACT_RECONCILIATION_SYSTEM` + `BASELINE_ANNOTATION_SYSTEM` | Merge new facts with existing memory; annotate baseline facts |
| `consolidation.py` | `CONSOLIDATION_SYSTEM` | Dreams-like 4-phase fact store cleanup |
| `model_synthesis.py` | `MODEL_SYNTHESIS_SYSTEM` | Synthesize config findings into model summary |
| `version_synthesis.py` | `VERSION_SYNTHESIS_SYSTEM` | Synthesize model summaries into version rollup |

---

## Analysis Pipelines Step by Step

### How a Config-Level Analysis Uses Memory

```
1. inject_memory
   ├── Query baseline version's facts from DB
   ├── Query baseline model summary from DB
   ├── Read red flags file for current version
   └── Assemble into ~2000-token "Prior Memory" block

2. react_agent
   ├── Memory block injected into system prompt
   ├── Full ReAct loop with MCP tools
   └── Produces analysis report

3. extract_facts
   ├── Smaller LLM reads the report
   └── Outputs JSON: { facts: [...], red_flags: [...] }

4. judge_facts
   ├── LLM-as-judge validates each fact against report
   └── Drops facts with fabricated/misquoted data

5. reconcile_facts
   ├── Phase 1: Compare new facts against existing DB entries
   │   └── Decide: insert / update (merge narrative) / noop
   └── Phase 2: Annotate baseline version's facts
       └── "persists in current version" / "resolved" / "improved"

6. store_facts
   ├── For each insert: LLM semantic dedup gate
   ├── Upsert to config_findings / model_summaries / version_summaries
   └── Store full report to analysis_reports

7. update_red_flags (config graph only)
   ├── Separate model-specific vs ALL MODELS flags
   └── Append to red_flags_{version}.md (deduped)
```

### How Memory Chains Across Versions

```
v1.7.0 analysis → facts stored →
  v1.8.0 analysis (injected with 1.7.0 facts) → new facts stored, 1.7.0 facts annotated →
    v1.9.0 analysis (injected with 1.8.0 facts) → new facts stored, 1.8.0 facts annotated →
      consolidation → prune old resolved facts, merge duplicates, archive red flags
```

---

## New API Endpoints

Two endpoints were added beyond the original project's API:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/analyze` | POST | Submit an analysis job (returns immediately with `job_id`) |
| `/v1/analyze/{job_id}/status` | GET | Poll job status (accepted → running → completed/failed) |

### `POST /v1/analyze`

Submit a background analysis job. Returns `202 Accepted` with a `job_id`.

**Request body:**

```json
{
  "analysis_level": "config",
  "run_id": "run-abc123",
  "rhaiis_version": "BENCH-2.5",
  "baseline_version": "BENCH-2.4",
  "mode": "deep",
  "regression_detected": true,
  "config": {
    "model": "nova-ai/Helios-34B",
    "accelerator": "Accel-X900",
    "tp": 4,
    "prompt_toks": 1024,
    "output_toks": 1024,
    "concurrency": 200
  }
}
```

**Fields by analysis level:**

| Field | config | model | version | consolidation |
|---|---|---|---|---|
| `analysis_level` | required | required | required | required |
| `run_id` | required | required | required | required |
| `rhaiis_version` | required | required | required | required |
| `baseline_version` | required | required | required | required |
| `mode` | required (`"shallow"` / `"deep"`) | -- | -- | -- |
| `regression_detected` | required | -- | -- | -- |
| `config` | required (all fields) | required (`model` field) | -- | -- |

**Response:**

```json
{
  "job_id": "job-a1b2c3d4e5f6",
  "status": "accepted",
  "analysis_level": "config"
}
```

### `GET /v1/analyze/{job_id}/status`

Poll the status of a submitted job.

**Response:**

```json
{
  "job_id": "job-a1b2c3d4e5f6",
  "status": "completed",
  "analysis_level": "config",
  "report_preview": "First 500 chars of the report...",
  "facts_stored": 5,
  "red_flags_added": 2,
  "error": null,
  "duration_seconds": 45.2
}
```

Status values: `accepted` → `running` → `completed` | `failed`

---

## Running the New Endpoints

### Prerequisites

The analysis pipeline requires:
1. **Google Gemini API key** (for both the main model and the fact model)
2. **PSAP MCP Server** running (for config-level analysis tools)
3. **PostgreSQL** or **SQLite** for the memory store (SQLite is auto-created for local dev)

### Quick Start (Local Development)

```bash
cd psap-agent

# Install dependencies (includes new langgraph, asyncpg deps)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Set your API key
export GOOGLE_API_KEY="your-gemini-api-key"

# Start the agent (SQLite memory, no PostgreSQL needed)
USE_INMEMORY_SAVER=true python -m psap_agent.src.main
```

The server starts on `http://localhost:8081`. The `/v1/analyze` endpoint is automatically registered alongside the existing `/v1/stream` and other endpoints.

### Submitting an Analysis Job

```bash
# Config-level analysis (deep investigation with MCP tools)
curl -X POST http://localhost:8081/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_level": "config",
    "run_id": "run-001",
    "rhaiis_version": "BENCH-2.5",
    "baseline_version": "BENCH-2.4",
    "mode": "deep",
    "regression_detected": true,
    "config": {
      "model": "nova-ai/Helios-34B",
      "accelerator": "Accel-X900",
      "tp": 4,
      "prompt_toks": 1024,
      "output_toks": 1024
    }
  }'

# Model-level synthesis (no MCP tools needed)
curl -X POST http://localhost:8081/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_level": "model",
    "run_id": "run-001",
    "rhaiis_version": "BENCH-2.5",
    "baseline_version": "BENCH-2.4",
    "config": { "model": "nova-ai/Helios-34B", "accelerator": "", "tp": 0, "prompt_toks": 0, "output_toks": 0 }
  }'

# Version-level synthesis
curl -X POST http://localhost:8081/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_level": "version",
    "run_id": "run-001",
    "rhaiis_version": "BENCH-2.5",
    "baseline_version": "BENCH-2.4"
  }'

# Consolidation (cleanup after a full version run)
curl -X POST http://localhost:8081/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_level": "consolidation",
    "run_id": "run-001",
    "rhaiis_version": "BENCH-2.5",
    "baseline_version": "BENCH-2.4"
  }'
```

### Polling for Results

```bash
# Poll until status is "completed" or "failed"
curl http://localhost:8081/v1/analyze/job-a1b2c3d4e5f6/status
```

### Typical Orchestration Flow

A full version analysis runs in this order:

```
1. For each (model, accelerator, tp, profile) combination:
   POST /v1/analyze  { analysis_level: "config", ... }
   → poll until completed

2. For each model:
   POST /v1/analyze  { analysis_level: "model", ... }
   → poll until completed

3. Once all models are done:
   POST /v1/analyze  { analysis_level: "version", ... }
   → poll until completed

4. Finally:
   POST /v1/analyze  { analysis_level: "consolidation", ... }
   → poll until completed
```

Each step builds on the previous one's stored facts.

### With PostgreSQL (Production)

```bash
# Set PostgreSQL connection
export POSTGRES_HOST=your-pg-host
export POSTGRES_PORT=5432
export POSTGRES_DB=psap
export POSTGRES_USER=psap_user
export POSTGRES_PASSWORD=your-password

# Run migrations first (only needed once)
python migrations/migrate.py --backend postgres

# Start the agent
python -m psap_agent.src.main
```

---

## Configuration

New environment variables for the memory layer:

| Variable | Default | Description |
|---|---|---|
| `FACT_MODEL` | `gemini-2.0-flash` | Smaller LLM for fact extraction, judging, reconciliation |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Main LLM for ReAct agent and synthesis |
| `POSTGRES_HOST` | `pgvector` | Set to a real host to enable PostgreSQL; defaults to SQLite otherwise |

All existing configuration variables from the original project still apply.

---

## MCP Server Tools (Interactive Access)

Three read-only MCP tools were added to `psap-mcp-server` so the **interactive chat agent** can query the memory layer during conversations. These live in `psap-mcp-server/psap_mcp_server/src/tools/` and are registered in `mcp.py`.

| Tool | File | Description |
|---|---|---|
| `search_key_facts` | `key_facts_tool.py` | Search config-level findings with partial-match filters (version, model, accelerator, profile, category). Returns matching facts with metadata. |
| `get_version_summary` | `key_facts_tool.py` | Get the version-level summary and all model summaries for a given version. |
| `get_model_performance_history` | `key_facts_tool.py` | Track a model's performance trends across the last N versions (default 5). Groups model summaries by version. |

### Supporting Files

| File | Purpose |
|---|---|
| `key_facts_db.py` | Database layer for the MCP server side. Mirrors the agent-side `memory/db.py` schema (same 5 tables). Exists as a separate copy because the MCP server runs in its own process with its own settings module. |
| `key_facts_extraction.py` | LLM prompt template for extracting structured facts from reports. Used for standalone extraction outside the graph pipeline. |

### How They Connect

The MCP tools read from the **same database** that the analysis pipeline writes to. When the automated pipeline (`/v1/analyze`) stores facts via the agent-side `memory/queries.py`, the interactive chat agent can immediately query those facts through these MCP tools during a conversation.

```
Automated pipeline (agent-side)          Interactive chat (MCP-side)
─────────────────────────────            ──────────────────────────
/v1/analyze                              /v1/stream
  → graph pipeline                         → ReAct agent
    → store_facts node                       → search_key_facts tool
      → memory/queries.py                      → key_facts_db.py
        → PostgreSQL / SQLite  ◄──────────────→  PostgreSQL / SQLite
          (shared database)                       (same database)
```

A fourth tool, `store_key_facts`, is defined in `key_facts_tool.py` but is **not registered** in `mcp.py`. It provides write access for programmatic use but is intentionally kept out of the interactive agent's tool set to prevent uncontrolled writes during chat sessions.

---

## Migrations

If you already have an existing database with the original 3-table schema (config_findings, model_summaries, version_summaries), run the migration to add the new columns and tables:

```bash
# Auto-detect backend
python migrations/migrate.py

# Or specify explicitly
python migrations/migrate.py --backend postgres
python migrations/migrate.py --backend sqlite
python migrations/migrate.py --backend sqlite --sqlite-path /path/to/memory.db
```

The migration adds:
- `status` and `issue_thread_id` columns to existing tables
- `analysis_reports` table (new)
- `consolidation_log` table (new)
- Updated indexes

All statements are idempotent — safe to run multiple times.

For fresh installs, `ensure_schema()` is called automatically on startup and creates all tables with the full schema, so no migration is needed.
