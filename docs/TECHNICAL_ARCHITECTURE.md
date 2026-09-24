# AI Performance Agent - Technical Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Component Integration](#component-integration)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
6. [Tool Calling Mechanism](#tool-calling-mechanism)
7. [AI Agent Architecture](#ai-agent-architecture)
8. [Communication Patterns](#communication-patterns)
9. [State Management](#state-management)
10. [Integration Points](#integration-points)
11. [Observability & Feedback (Langfuse)](#observability--feedback-langfuse)

---

## System Overview

This system implements a **Tool-Augmented AI Agent** that uses the **Model Context Protocol (MCP)** to provide natural language access to performance data. The architecture separates concerns into four layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  • Streamlit Chat UI (User Interface)                       │
│  • Performance Dashboard (Data Visualization)               │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTP/REST + SSE streaming
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│  • LangGraph Agent (State Machine)                          │
│  • Google Gemini (LLM Reasoning, default: gemini-3.8-flash) │
│  • PostgreSQL (State Persistence + Skill Store)             │
└────────────────┬────────────────────────────────────────────┘
                 │ MCP Protocol (Streamable HTTP)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      TOOL LAYER                              │
│  • FastMCP Server (Tool Registry)                           │
│  • ~31 Performance Analysis Tools (15 tool modules)         │
│  • Data Loaders (CSV, Grafana API, S3)                      │
└─────────────────────────────────────────────────────────────┘
                 ▲
                 │ (runs after every response)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 SELF-IMPROVEMENT LAYER                       │
│  • Reflection & Critic (inline, before delivery)            │
│  • LLM-as-Judge (4 criteria, scores to Langfuse)            │
│  • Quality-gated Memory (Mem0 + Qdrant)                     │
│  • Skill Documents (reusable tool-call recipes)             │
│  See: SELF_IMPROVEMENT_ARCHITECTURE.md                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### 1. **Model Context Protocol (MCP)**
- **What it is:** Open protocol for standardized communication between AI systems and external tools
- **Role:** Enables the agent to discover and invoke tools without tight coupling
- **Implementation:** FastMCP library

### 2. **FastMCP (Python MCP Implementation)**
- **What it is:** Python library for building MCP servers
- **Role:** Tool registration, parameter validation, request/response handling
- **Why chosen:** Rapid development, decorator-based tool registration, automatic OpenAPI spec generation

### 3. **LangGraph (Agent Framework)**
- **What it is:** Framework for building stateful, multi-actor applications with LLMs
- **Role:** Orchestrates the agent's reasoning loop, manages conversation state, tool invocation flow
- **Why chosen:** Built-in state persistence, cycle detection, human-in-the-loop support

### 4. **LLM (Gemini + OpenAI + Claude)**
- **What it is:** Multi-model support -- Google Gemini (default), OpenAI, and Anthropic Claude via Vertex AI
- **Role:** Natural language understanding, query intent parsing, response generation, tool calling
- **Default model:** `gemini-3.8-flash` (configurable via `GEMINI_MODEL` env var)
- **Available models:** `gemini-3.8-flash`, `gemini-3.1-pro-preview`, `gpt-6-luna` (extra-high thinking), `gpt-6-sol` (medium thinking), `openai:<model-id>`, `claude-opus-4-6`, `claude-sonnet-4-6`
- **Per-request model selection:** Users can switch models from the Streamlit UI dropdown; the `model` field in `StreamRequest` overrides the server default, and OpenAI selections may provide `reasoning_effort` (for example, `xhigh`) for that request
- **Claude integration:** Uses `ChatAnthropicVertex` via `langchain-google-vertexai`, requires `ANTHROPIC_VERTEX_PROJECT_ID` and GCP auth
- **OpenAI integration:** Uses `ChatOpenAI` via `langchain-openai`, requires `OPENAI_API_KEY`; Streamlit includes fixed GPT-6 Luna and Sol choices, while `OPENAI_MODEL` optionally adds another choice. GPT-6 requests use the Responses API so tools work with their selected reasoning effort

### 5. **FastAPI (Web Framework)**
- **What it is:** Modern Python web framework
- **Role:** REST API for both agent and MCP server, request handling, async support
- **Why chosen:** High performance, automatic OpenAPI docs, native async/await support

### 6. **Streamlit (UI Framework)**
- **What it is:** Python framework for data apps
- **Role:** Interactive chat interface, performance dashboard visualization
- **Why chosen:** Rapid prototyping, built-in widgets, real-time updates

### 7. **PostgreSQL (Database)**
- **What it is:** Relational database
- **Role:** Persist agent conversation state, enable conversation history, checkpoint recovery, store skill documents as key-value JSON (via LangGraph `AsyncPostgresStore`)
- **Why chosen:** ACID compliance, JSON support, LangGraph native integration (no pgvector required)

### 8. **Pandas (Data Processing)**
- **What it is:** Data manipulation library
- **Role:** CSV processing, data filtering, statistical calculations
- **Why chosen:** Performance, rich API, native support for data operations

### 9. **Grafana API (Metrics Source)**
- **What it is:** Observability platform with REST API
- **Role:** Real-time GPU metrics (DCGM), vLLM inference metrics
- **Why chosen:** Industry standard, Prometheus integration, rich query language

### 10. **Langfuse (LLM Observability & Analytics)**
- **What it is:** Open-source observability and analytics platform for LLM applications
- **Role:** Trace agent executions, track LLM calls, collect user feedback, host automated LLM-as-Judge evaluation scores (correctness, hallucination, tool_efficiency, completeness)
- **Why chosen:** Self-hostable, comprehensive tracing, feedback management, cost tracking, score API for automated evaluation
- **Deployment:** Self-hosted using Podman with shared PostgreSQL database

### 11. **Mem0 + Qdrant (Episodic Memory)**
- **What it is:** Mem0 is a memory layer for AI agents; Qdrant is a vector database
- **Role:** Extract facts from conversations, store as 768-dim embeddings, retrieve relevant context for future queries
- **Why chosen:** Automatic fact extraction, semantic search, local deployment (no external API required)
- **Stack:** Gemini 2.5 Flash (extraction), Gemini Embedding 001 (768-dim), Qdrant (on-disk vector store)

### 12. **Gemini Embedding 001 (Embeddings)**
- **What it is:** Google's text embedding model producing 768-dimensional vectors
- **Role:** Used by Mem0 for memory retrieval and by the skill retrieval system for client-side semantic ranking of skill documents
- **Why chosen:** Consistent with the Gemini model family, no additional API keys needed

---

## Component Integration

### Layer 1: User Interface → Agent Communication

**Technology:** Streamlit → FastAPI (HTTP/REST)

```python
# Streamlit (Client)
response = requests.post(
    f"{api_url}/v1/stream",
    json={
        "message": user_message,
        "thread_id": thread_id,
        "session_id": session_id,
        "user_id": user_id,
        "stream_tokens": stream_tokens,
    },
    stream=True,
    headers={"Accept": "text/event-stream"},
)

# FastAPI Agent (Server)
@router.post("/v1/stream")
async def stream(request: StreamRequest):
    return StreamingResponse(
        message_generator(request), media_type="text/event-stream"
    )
```

**Integration Points:**
- **Protocol:** HTTP POST with Server-Sent Events (SSE) for streaming
- **Authentication:** Optional SSO bearer token
- **Data Format:** JSON request, newline-delimited JSON events in response
- **Thread Management:** Thread IDs for conversation continuity, session IDs for Langfuse grouping
- **User Tracking:** User IDs for per-user memory retrieval

### Layer 2: Agent → MCP Server Communication

**Technology:** LangGraph → FastMCP (MCP Protocol over HTTP)

```python
# Agent discovers available tools via MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "psap-mcp-server": {
        "url": "http://psap-mcp-server:5001/mcp/",
        "transport": "streamable_http",
        "headers": {"Authorization": f"Bearer {sso_token}"} if sso_token else {},
    },
})
tools = await client.get_tools()
# Returns ~31 LangChain-compatible tools, ready for create_react_agent()
```

**Integration Points:**
- **Protocol:** MCP over Streamable HTTP transport
- **Discovery:** Dynamic tool discovery via `get_tools()` (auto-converts to LangChain tools)
- **Invocation:** LangGraph handles tool calling automatically within the agent loop
- **Session Management:** `MultiServerMCPClient` manages MCP sessions internally

### Layer 3: MCP Server → Data Sources

**Technology:** FastMCP → Pandas/Grafana API

```python
# MCP Tool Definition (FastMCP)
@mcp.tool()
async def query_performance_metrics(
    model: str,
    version: str,
    accelerator: str,
    profile: str
) -> Dict[str, Any]:
    # Load data via Pandas
    df = load_rhaiis_data()
    
    # Filter and process
    result = df[
        (df["model"] == model) &
        (df["version"] == version) &
        (df["accelerator"] == accelerator)
    ]
    
    # Query Grafana if UUID available
    if uuid and has_grafana_data:
        grafana_metrics = await query_grafana_api(uuid)
    
    return result
```

**Integration Points:**
- **CSV Loading:** Pandas reads from filesystem
- **Grafana API:** HTTP requests to Prometheus data source
- **Caching:** In-memory caching of CSV data
- **Error Handling:** Graceful degradation if data unavailable

---

## Data Flow Architecture

### Complete Request Flow

```
User Query: "Compare the performance of DeepSeek-R1 between RHAIIS-3.3 and RHAIIS-3.4-EA1 on H200"

1. STREAMLIT UI
   ├─ User types query
   ├─ Creates thread_id (or reuses existing)
   └─ HTTP POST → /v1/stream
        │
        │ {"message": "Compare...", "thread_id": "abc123",
        │  "session_id": "...", "user_id": "user@example.com"}
        │
        ▼

2. MEMORY RETRIEVAL (before agent runs)
   ├─ Mem0 searches Qdrant for relevant past facts
   │   (relevance ≥ 0.6, max 5 memories)
   ├─ Skill store: fetch all skills, embed with Gemini,
   │   rank by cosine similarity, return top 1 if score ≥ 0.5
   └─ Memory context injected into user message:
        "user query\n\n[SYSTEM - Memory Context ...]\n..."
        │
        ▼

3. LANGGRAPH AGENT (with safety rails)
   ├─ Loads conversation state from PostgreSQL
   ├─ Gemini analyzes query + memory context (may follow skill recipe)
   ├─ Invokes tools via MCP: discover_configurations → compare_configurations
   │   → compare_grafana_metrics → compare_vllm_logs → generate_dashboard_url
   │    │
   │    │ MCP Protocol (Streamable HTTP)
   │    ▼
   │
   │ MCP SERVER
   │    ├─ Receives tool calls, validates parameters
   │    ├─ Executes against CSV data, Grafana API, S3
   │    └─ Returns structured JSON results
   │         │
   │         ▼
   │
   ├─ Gemini formats response with tool data
   └─ Response BUFFERED (not yet delivered to user)
        │
        ▼

4. REFLECTION (inline, before delivery)
   ├─ Critic checks for hallucinated URLs, unsupported claims
   ├─ Sees tool evidence from ALL passes (cross-pass context)
   │
   ├─ Pass → response delivered to user
   └─ Revise → targeted edit prompt, loop (up to 2 iterations)
        │
        ▼

5. RESPONSE DELIVERED TO USER
   └─ Only the final, critic-approved response is streamed
        │
        ▼ (background task, non-blocking)

6. POST-PROCESSING
   ├─ Langfuse: flush traces, resolve trace ID
   ├─ LLM-as-Judge: score correctness, hallucination,
   │   tool_efficiency, completeness (posted to Langfuse)
   ├─ Memory gate: store in Mem0 if hallucination > 0.5
   │   AND correctness > 0.5
   └─ Skill gate: generate skill doc if tool_efficiency = 1.0
       AND interaction was complex (≥ 5 tool calls)
```

---

## Model Context Protocol (MCP)

### Why MCP?

Traditional AI agents face several challenges:

**Problem 1: Tight Coupling**
```python
# Without MCP - Tightly coupled
class Agent:
    def query_data(self):
        # Agent code directly calls data functions
        result = query_database(...)
        result2 = query_api(...)
        # Adding new tools requires modifying agent code
```

**Problem 2: No Standardization**
```python
# Each tool has different calling conventions
tool1.execute(param1, param2)
tool2.run({"key": "value"})
tool3.invoke(args=[1, 2, 3])
```

**Problem 3: Limited Discovery**
```python
# Agent needs to know all tools at build time
AVAILABLE_TOOLS = [tool1, tool2, tool3]  # Hardcoded
```

### MCP Solution

**Standardized Protocol:**
```json
// Tool Discovery
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "query_performance_metrics",
        "description": "Query performance metrics for models",
        "inputSchema": {
          "type": "object",
          "properties": {
            "model": {"type": "string"},
            "version": {"type": "string"}
          }
        }
      }
    ]
  }
}

// Tool Invocation
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "query_performance_metrics",
    "arguments": {
      "model": "Llama-3.3-70B",
      "version": "RHAIIS-3.2.4"
    }
  },
  "id": 2
}
```

### FastMCP Implementation

**Server-Side:**
```python
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    name="Performance Analysis Server",
    version="1.0.0"
)

# Register tool with decorator
@mcp.tool()
async def query_performance_metrics(
    model: str,
    version: str,
    accelerator: str,
    profile: str
) -> Dict[str, Any]:
    """Query performance metrics for a specific configuration."""
    # Tool implementation
    result = await fetch_data(model, version, accelerator, profile)
    return result

# FastAPI integration
app.mount("/mcp/", mcp.get_sse_handler())
```

**Client-Side (Agent):**
```python
from mcp import ClientSession

# Connect to MCP server
session = ClientSession(
    read_url="http://localhost:5001/mcp/",
    write_url="http://localhost:5001/mcp/"
)

# Initialize session
await session.initialize()

# Dynamic tool discovery
tools_response = await session.list_tools()
available_tools = tools_response.tools

# Convert to LangChain tools
from langchain.tools import StructuredTool

langchain_tools = []
for tool in available_tools:
    langchain_tool = StructuredTool(
        name=tool.name,
        description=tool.description,
        func=lambda **kwargs: session.call_tool(tool.name, kwargs)
    )
    langchain_tools.append(langchain_tool)

# Now agent can use tools via LangChain's standard interface
```

---

## Tool Calling Mechanism

### LangChain Tool Integration

Tools are discovered dynamically from the MCP server and bound to the selected chat model automatically by `create_react_agent`. No manual tool schema definitions are needed.

**1. LLM Initialization (multi-model):**
```python
from psap_agent.src.core.model_factory import create_chat_model

llm = create_chat_model(
    model_name,  # e.g. "gemini-3.8-flash", "openai:<model-id>", or "claude-opus-4-6"
    temperature=0,
)
```

**2. Tool Discovery + Agent Creation:**
```python
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({...}) as client:
    tools = client.get_tools()  # ~31 tools from FastMCP

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,   # AsyncPostgresCheckpoint
        store=store,                  # AsyncPostgresStore (skills)
        prompt=SYSTEM_PROMPT,
    )
```

**3. Agent Execution:**
```python
async for event in agent.astream_events(
    {"messages": [HumanMessage(content=enriched_query)]},
    config={"configurable": {"thread_id": thread_id}},
    version="v2",
):
    # LangGraph autonomously loops: reason → call tools → reason
    # until the selected model produces a final text response
    pass
```

### Model Function Calling Format

When the selected model decides to use a tool, LangChain normalizes its function call for the LangGraph agent:

```json
{
  "function_call": {
    "name": "query_performance_metrics",
    "args": {
      "model": "Llama-3.3-70B",
      "version": "RHAIIS-3.2.4",
      "accelerator": "H200",
      "profile": "1k/1k"
    }
  }
}
```

The LangGraph ReAct agent then:
1. Extracts the function call(s) -- models can emit multiple parallel calls
2. Invokes the MCP tools via `MultiServerMCPClient`
3. Formats the results as `FunctionMessage` entries
4. Sends the result back to the selected model for the next reasoning step or final response

---

## AI Agent Architecture

### LangGraph State Machine

The agent uses LangGraph's prebuilt `create_react_agent`, which implements a ReAct (Reason + Act) loop automatically. State management and tool routing are handled by LangGraph internals.

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
store = AsyncPostgresStore.from_conn_string(DATABASE_URL)

agent = create_react_agent(
    model=llm,              # ChatGoogleGenerativeAI
    tools=mcp_tools + [load_skill],  # MCP tools plus local curated-skill loader
    checkpointer=checkpointer,
    store=store,            # skill documents (key-value)
    prompt=SYSTEM_PROMPT,
)
```

The ReAct loop: Gemini reasons about the query → calls tools → inspects results → reasons again → calls more tools or emits final response. LangGraph persists the full message history to PostgreSQL after each step for crash recovery.

### System Prompt Engineering

The agent uses a concise, stable policy prompt plus reviewed workflow skills that
it loads on demand with a local `load_skill` tool. The stable prompt retains the
rules that apply to every response:

**Global Integrity Rules** -- prevent hallucination and fabrication:
```
## Anti-Fabrication
- NEVER state unverified facts as verified.
- ZERO-URL POLICY: Every URL MUST come from a tool call result.
- EMPTY RESULTS RULE: When a tool returns zero results, say so.
  Do NOT fill the gap with training data.

## Certainty Calibration
- Tool-returned data → state as fact
- Training knowledge → hedge explicitly
- Observation vs. Causal Conclusion → require profiling evidence
```

Detailed instructions for data-backed clarification and fair comparisons,
benchmark and cost analysis, Grafana metrics, deep profiling, PyTorch traces,
kernel/source attribution, vLLM logs, and vLLM performance triage live in
`psap_agent/src/core/curated_skill_documents/`. The model loads only the
relevant instructions for a specialized request; simple discovery requests use
the MCP tools directly. Policy-coverage tests protect the migrated legacy
guardrails from being silently removed during future prompt changes.

The self-improvement system's generated skill documents remain a separate,
advisory source of prior tool-call recipes. They do not override global policy,
curated workflow instructions, or current tool evidence.

The prompt is injected via `create_react_agent(prompt=...)`, which prepends it as a system message to every Gemini call.

For the self-improvement system (reflection, evaluation, memory, skills), see [SELF_IMPROVEMENT_ARCHITECTURE.md](SELF_IMPROVEMENT_ARCHITECTURE.md).

---

## Communication Patterns

### Pattern 1: Request-Response (Simple Query)

```
User → Streamlit → Agent → MCP → Data → MCP → Agent → Streamlit → User
        HTTP        Process  RPC   Read  RPC   Format  HTTP
```

**Example:** "What models were tested for RHAIIS-3.2.4?"

1. Streamlit sends HTTP POST
2. Agent receives query
3. Agent calls `discover_configurations(version="RHAIIS-3.2.4")`
4. MCP server loads CSV, filters data
5. MCP returns list of models
6. Agent formats response
7. Streamlit displays list

**Timing:** ~2-5 seconds

### Pattern 2: Multi-Turn Clarification

```
User → Agent: "What's the performance of Llama?"
       ↓
Agent → MCP: discover_configurations(model="Llama")
       ↓
MCP → Agent: Found 3 Llama variants
       ↓
Agent → User: "Which Llama? (1) Llama-3.3-70B, (2) Llama-3.3-70B-FP8, (3) Llama-4-Maverick"
       ↓
User → Agent: "Llama-3.3-70B on H200 for 1k/1k on version 3.2.4"
       ↓
Agent → MCP: query_performance_metrics(...)
       ↓
MCP → Agent: Performance data
       ↓
Agent → User: Formatted results
```

**Timing:** ~5-15 seconds (depending on user response time)

### Pattern 3: Streaming Response

```python
# Agent streams response as it's generated
async def stream_chat(query: str, thread_id: str):
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}}
    ):
        if "agent" in chunk:
            # Stream tokens as they're generated
            message = chunk["agent"]["messages"][0]
            if hasattr(message, "content"):
                yield {
                    "type": "token",
                    "content": message.content
                }
        
        elif "tools" in chunk:
            # Notify about tool usage
            tool_name = chunk["tools"]["messages"][0].tool_calls[0]["name"]
            yield {
                "type": "tool_call",
                "tool": tool_name
            }
```

**User Experience:** Real-time response generation, see tool calls as they happen

### Pattern 4: Parallel Tool Calls

```python
# Agent can call multiple tools in parallel
async def parallel_comparison():
    # LLM decides to get data for multiple models
    tool_calls = [
        session.call_tool("query_performance_metrics", 
                         {"model": "Llama-3.3-70B", ...}),
        session.call_tool("query_performance_metrics",
                         {"model": "Mistral-7B", ...}),
        session.call_tool("query_performance_metrics",
                         {"model": "GPT-OSS-120B", ...})
    ]
    
    # Execute in parallel
    results = await asyncio.gather(*tool_calls)
    
    # LLM processes all results together
    return results
```

**Benefit:** 3x faster than sequential calls

---

## State Management

### PostgreSQL Checkpointing

**Why State Persistence?**
1. **Conversation Continuity:** User can return to conversation later
2. **Failure Recovery:** Agent can resume if crashed
3. **Audit Trail:** Track all interactions and tool calls
4. **Multi-Session:** Support multiple concurrent users

**Schema:**
```sql
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);
```

**Data Stored:**
```json
{
  "thread_id": "abc123",
  "checkpoint_id": "checkpoint_5",
  "checkpoint": {
    "messages": [
      {"role": "user", "content": "What's the performance of Llama?"},
      {"role": "assistant", "content": "I found 3 Llama variants..."},
      {"role": "user", "content": "Llama-3.3-70B"}
    ],
    "values": {
      "clarification_needed": false,
      "last_tool_result": {...}
    }
  },
  "metadata": {
    "step": 5,
    "source": "loop",
    "writes": {"agent": {...}}
  }
}
```

**Usage:**
```python
# Load previous conversation
config = {"configurable": {"thread_id": "abc123"}}
response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Continue from where we left off")]},
    config=config
)

# Agent automatically loads previous state from PostgreSQL
# and continues the conversation
```

### In-Memory State (Single Request)

```python
# Within a single request, LangGraph maintains state
class AgentState(TypedDict):
    messages: List[BaseMessage]  # Full conversation
    intermediate_steps: List[tuple]  # Tool calls made
    current_tool_results: dict  # Latest tool outputs
    needs_clarification: bool  # Flag for clarification
    clarification_context: dict  # What we're clarifying
```

**State Updates:**
```python
# After each node execution, state is updated
def call_tools_node(state: AgentState) -> AgentState:
    # Get tool calls from LLM
    tool_calls = state["messages"][-1].tool_calls
    
    # Execute tools
    results = []
    for tool_call in tool_calls:
        result = execute_tool(tool_call)
        results.append(result)
    
    # Update state
    return {
        **state,
        "current_tool_results": results,
        "intermediate_steps": state["intermediate_steps"] + [(tool_call, result)]
    }
```

---

## Integration Points

### 1. Streamlit ↔ Agent API

**Protocol:** HTTP REST with SSE (Server-Sent Events)

**Request:**
```python
import requests

def call_agent(message: str, thread_id: str = None, user_id: str = None):
    response = requests.post(
        "http://localhost:5002/v1/stream",
        json={
            "message": message,
            "thread_id": thread_id,
            "user_id": user_id,
            "stream_tokens": True,
        },
        stream=True,
        headers={"Accept": "text/event-stream"},
    )
    
    for line in response.iter_lines():
        if line:
            event = json.loads(line)
            yield event
```

**Response Format:**
```json
// Token streaming
{"type": "token", "content": "The performance of"}
{"type": "token", "content": " Llama-3.3-70B"}

// Tool call notification
{"type": "tool_call", "tool": "query_performance_metrics"}

// Complete message
{"type": "message", "content": "The performance of Llama-3.3-70B on H200..."}

// Error
{"type": "error", "error": "Model not found", "code": 404}
```

### 2. Agent ↔ MCP Server

**Protocol:** MCP over Streamable HTTP

**Tool Discovery + Binding:**
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "psap-mcp-server": {
        "url": "http://psap-mcp-server:5001/mcp/",
        "transport": "streamable_http",
    },
}) as client:
    tools = client.get_tools()  # ~31 LangChain-compatible tools
    agent = create_react_agent(model=llm, tools=tools, ...)
```

**Tool Invocation:**

Tool calls are handled automatically by LangGraph's ReAct loop -- the agent does not call tools directly. Gemini emits function calls, LangGraph routes them to the appropriate MCP tool via `MultiServerMCPClient`, and the results are returned as MCP `TextContent`:

```json
{
    "content": [
        {
            "type": "text",
            "text": "{\"output_toks_per_sec\": 72.4, \"concurrency\": 400, ...}"
        }
    ]
}
```

### 3. MCP Server ↔ Data Sources

**CSV Data:**
```python
import pandas as pd

# Singleton data loader
class DataLoader:
    _instance = None
    _data = None
    
    @classmethod
    def load(cls):
        if cls._data is None:
            cls._data = pd.read_csv("consolidated_dashboard.csv")
        return cls._data

# Tool uses cached data
@mcp.tool()
async def query_performance_metrics(...):
    df = DataLoader.load()
    filtered = df[df["model"] == model]
    return filtered.to_dict()
```

**Grafana API:**
```python
import aiohttp

async def query_grafana(uuid: str, start_ms: int, end_ms: int):
    # Prometheus query via Grafana
    query = f'vllm_request_success{{deployment_uuid="{uuid}"}}'
    
    params = {
        "query": query,
        "start": start_ms / 1000,
        "end": end_ms / 1000,
        "step": "60s"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{GRAFANA_URL}/api/datasources/proxy/{DATASOURCE_UID}/api/v1/query_range",
            params=params,
            headers={"Authorization": f"Bearer {GRAFANA_TOKEN}"}
        ) as response:
            data = await response.json()
            return data
```

---

## Key Design Patterns

### 1. Separation of Concerns

```
┌─────────────────┐
│  Presentation   │ ← Streamlit (UI/UX only)
├─────────────────┤
│  Orchestration  │ ← LangGraph + selected LLM (Reasoning + coordination)
├─────────────────┤
│  Tool Layer     │ ← FastMCP (Tool execution)
├─────────────────┤
│  Data Layer     │ ← Pandas/Grafana (Data access)
└─────────────────┘
```

**Benefits:**
- Each layer can be developed/tested independently
- Easy to swap implementations (e.g., select OpenAI with `openai:<model-id>`)
- Clear responsibilities

### 2. Protocol-Based Integration (MCP)

```python
# Loose coupling via standard protocol
Agent ←──MCP Protocol──→ MCP Server

# Agent doesn't need to know tool implementation
# MCP Server doesn't need to know agent logic
```

**Benefits:**
- Tools can be added without changing agent
- Multiple agents can use same MCP server
- Different programming languages can interoperate

### 3. State Machine Architecture (LangGraph)

```python
# Clear state transitions
State[t] → Node → State[t+1] → Node → State[t+2]

# vs unstructured loops
while True:
    # What state are we in?
    # What should we do next?
    # When do we stop?
```

**Benefits:**
- Predictable behavior
- Easy debugging (inspect state at each step)
- Built-in cycle detection

### 4. Async/Await Throughout

```python
# Non-blocking I/O at every layer
async def query_data():
    # Database calls don't block
    data = await db.execute(query)
    
    # API calls don't block
    metrics = await fetch_grafana_metrics()
    
    # Tool calls don't block
    result = await mcp_session.call_tool(...)
    
    return combine(data, metrics, result)
```

**Benefits:**
- High concurrency (handle many users)
- Efficient resource usage
- Better user experience (faster responses)

---

## Performance Considerations

### 1. CSV Caching

```python
# Load once, use many times
_cached_data = None

def load_rhaiis_data():
    global _cached_data
    if _cached_data is None:
        _cached_data = pd.read_csv("consolidated_dashboard.csv")
    return _cached_data
```

**Impact:** 2-3 seconds → 10ms for subsequent queries

### 2. Connection Pooling

```python
# Reuse database connections
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

**Impact:** Avoid connection overhead on every request

### 3. Streaming Responses

```python
# Stream tokens as they're generated
async def stream_response():
    async for token in llm.astream(prompt):
        yield token  # Send immediately, don't wait for full response
```

**Impact:** User sees first words in 500ms instead of waiting 5+ seconds

### 4. Parallel Tool Execution

```python
# Execute multiple tools simultaneously
results = await asyncio.gather(
    tool1.execute(),
    tool2.execute(),
    tool3.execute()
)
```

**Impact:** 3x faster than sequential execution

---

## Error Handling & Resilience

### 1. Graceful Degradation

```python
try:
    grafana_metrics = await query_grafana_metrics(uuid)
except GrafanaAPIError:
    # Continue without Grafana data
    grafana_metrics = None
    note = "GPU metrics unavailable at this time"
```

### 2. Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def query_grafana_api(uuid: str):
    # Retry on transient failures
    response = await http_client.get(url)
    return response.json()
```

### 3. Validation at Every Layer

```python
# MCP Server: Validate tool inputs
@mcp.tool()
async def query_performance_metrics(
    model: Annotated[str, "Model name"],
    version: Annotated[str, "Version identifier"],
    accelerator: Annotated[str, "Hardware type"],
    profile: Annotated[str, "ISL/OSL profile"]
):
    # FastMCP automatically validates types
    ...

# Agent: Validate tool results
result = await session.call_tool(...)
if not result.get("success"):
    # Handle error gracefully
    return "Could not retrieve data at this time"

# Streamlit: Validate user inputs
if not thread_id or not isinstance(thread_id, str):
    st.error("Invalid thread ID")
    return
```

---

## Monitoring & Observability

### 1. Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "tool_invoked",
    tool_name="query_performance_metrics",
    thread_id=thread_id,
    user_query=query,
    execution_time_ms=elapsed
)
```

### 2. Metrics Collection

```python
from prometheus_client import Counter, Histogram

tool_calls = Counter(
    "agent_tool_calls_total",
    "Number of tool calls",
    ["tool_name", "status"]
)

response_time = Histogram(
    "agent_response_time_seconds",
    "Response time distribution"
)

# Instrument code
with response_time.time():
    result = await agent.ainvoke(query)
    tool_calls.labels(tool_name="query_performance_metrics", status="success").inc()
```

### 3. Conversation Tracking

LangGraph persists the full message history (including tool calls and results) to PostgreSQL via `AsyncPostgresSaver`. Each thread is recoverable by `thread_id`.

### 4. Langfuse Tracing

Every agent invocation is wrapped in a Langfuse trace (`session_id`, `user_id`, `thread_id`). Token-level streaming events, tool calls, and LLM generations are captured automatically via the `CallbackHandler`.

### 5. LLM-as-Judge Automated Evaluation

After each response is delivered, a background task runs four evaluation criteria against the full conversation context:

| Criterion | What it measures |
|---|---|
| **correctness** | Are claims supported by tool evidence? |
| **hallucination** | Does the response fabricate data not in tool results? |
| **tool_efficiency** | Were the minimum necessary tools used? (Discovery calls count as necessary) |
| **completeness** | Did the response fully address the user's query? |

Each score (0.0–1.0) is posted to Langfuse via `langfuse.score()`, enabling dashboards for quality tracking over time.

### 6. Quality-Gated Memory & Skill Storage

Evaluation scores gate what gets stored:
- **Mem0 memory**: stored only if `hallucination > 0.5` AND `correctness > 0.5`
- **Skill documents**: generated only if `tool_efficiency = 1.0` AND ≥ 5 tool calls

For full details on the self-improvement pipeline, see [SELF_IMPROVEMENT_ARCHITECTURE.md](SELF_IMPROVEMENT_ARCHITECTURE.md).

---

## Security Considerations

### 1. API Authentication

```python
# Bearer token authentication
@app.middleware("http")
async def verify_token(request: Request, call_next):
    token = request.headers.get("Authorization")
    if not token or not verify_jwt(token):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )
    return await call_next(request)
```

### 2. Input Sanitization

```python
# Prevent injection attacks
def sanitize_input(query: str) -> str:
    # Remove SQL injection attempts
    query = query.replace("';", "")
    query = query.replace("--", "")
    
    # Limit length
    if len(query) > 10000:
        raise ValueError("Query too long")
    
    return query
```

### 3. Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/api/v1/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    ...
```

---

## Summary: How It All Fits Together

1. **User Interface (Streamlit)** provides the interaction layer
2. **HTTP/REST + SSE** carries streaming requests to the agent
3. **LangGraph** orchestrates the agent's reasoning workflow (ReAct loop)
4. **The selected LLM** (Gemini by default; OpenAI and Claude are optional) performs natural language understanding, tool calling, and generation
5. **MCP Protocol (Streamable HTTP)** provides standardized tool communication
6. **FastMCP** implements the MCP server and tool registry
7. **~31 Specialized Tools** (across 15 modules) handle analysis, comparison, profiling, and dashboarding tasks
8. **Pandas** processes CSV data efficiently
9. **Grafana API** provides real-time metrics
10. **PostgreSQL** persists conversation state (checkpoints) and skill documents (key-value store)
11. **FastAPI** handles all HTTP communication
12. **Async/Await** enables high concurrency
13. **Self-Improvement System** -- reflection, LLM-as-Judge evaluation, Mem0 memory, and skill documents create a learning loop that improves responses over time (see [SELF_IMPROVEMENT_ARCHITECTURE.md](SELF_IMPROVEMENT_ARCHITECTURE.md))

Each technology plays a specific role, and the **MCP protocol** is the key that ties the agent to the tools without tight coupling. This architecture enables:
- Easy extension (add new tools without touching agent code)
- Independent scaling (scale agent and tools separately)
- Technology flexibility (swap LLM providers via config)
- Clear debugging (inspect state at each layer + Langfuse traces)
- Continuous improvement (quality-gated memory and skill generation)

---

**This architecture represents a modern, production-ready approach to building self-improving AI agents with reliable tool access, observability, and maintainable code.**
