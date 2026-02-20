# PSAP Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12,3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LangGraph-based AI agent for vLLM inference performance analysis. Powered by Google Gemini, it connects to the [PSAP MCP Server](../psap-mcp-server/) to query benchmark data, analyze PyTorch profiler traces, compare vLLM versions, and generate Grafana dashboard links -- all through natural language.

## Architecture

```mermaid
graph TB
    subgraph clients [Clients]
        UI[Streamlit UI]
        API[API Client]
    end

    subgraph agent [PSAP Agent]
        Stream["/v1/stream (SSE)"]
        History["/v1/history"]
        Threads["/v1/threads"]
        Feedback["/v1/feedback"]
        Health["/health"]
        Core[AgentManager + LangGraph ReAct Agent]
        Prompt[System Prompt]
    end

    subgraph external [External Services]
        MCP[PSAP MCP Server]
        Gemini[Google Gemini]
        PG[(PostgreSQL)]
        LF[Langfuse]
    end

    UI --> Stream
    API --> Stream
    UI --> History & Threads & Feedback

    Stream --> Core
    Core --> Prompt
    Core --> Gemini
    Core -->|MCP protocol| MCP
    Core --> PG
    Core --> LF
    Feedback --> LF
    History --> PG
    Threads --> PG
```

The agent uses a LangGraph ReAct loop: Gemini decides which MCP tools to call based on the user's question, the MCP server executes them and returns structured data, and Gemini synthesizes the results into a response. Conversation state is checkpointed to PostgreSQL so users can resume threads across sessions.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/stream` | POST | Stream chat responses (SSE) |
| `/v1/history/{thread_id}` | GET | Get conversation history |
| `/v1/threads/{user_id}` | GET | List user threads |
| `/v1/feedback` | POST | Record feedback (sent to Langfuse) |

### Streaming Request

```http
POST /v1/stream
Content-Type: application/json
Accept: text/event-stream
```

```json
{
  "message": "Compare vLLM 0.13.0 vs 0.11.2 for DeepSeek on H200",
  "thread_id": "thread-123",
  "session_id": "session-456",
  "user_id": "user-789",
  "stream_tokens": true
}
```

### Streaming Response

```
{"type":"message","content":{"type":"ai","content":"","tool_calls":[...]}}
{"type":"token","content":"The"}
{"type":"token","content":" throughput"}
{"type":"message","content":{"type":"ai","content":"The throughput improved by 49%..."}}
[DONE]
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `AGENT_HOST` | `0.0.0.0` | Server bind address |
| `AGENT_PORT` | `8081` | Server port |
| `MCP_SERVER_URL` | `http://localhost:5001/mcp/` | PSAP MCP Server endpoint |
| `USE_INMEMORY_SAVER` | `false` | Use in-memory storage instead of PostgreSQL |
| `POSTGRES_HOST` | `pgvector` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `pgvector` | Database name |
| `POSTGRES_USER` | `pgvector` | Database user |
| `POSTGRES_PASSWORD` | `pgvector` | Database password |
| `GOOGLE_API_KEY` | -- | Google Gemini API key |
| `LANGFUSE_PUBLIC_KEY` | -- | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | -- | Langfuse secret key |
| `LANGFUSE_HOST` | -- | Langfuse server URL |
| `PYTHON_LOG_LEVEL` | `INFO` | Logging level |

## Local Development

```bash
# Create virtual environment and install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your Google API key, Langfuse keys, etc.

# Run with in-memory storage (no PostgreSQL needed)
USE_INMEMORY_SAVER=true python -m psap_agent.src.main

# Or use the Makefile
make local
```

The agent expects the PSAP MCP Server to be running at `MCP_SERVER_URL`. If it's not available and `USE_INMEMORY_SAVER=true`, the agent starts without tools (useful for testing the streaming API).

## Testing

```bash
pytest                                          # Run all tests
pytest --cov=psap_agent.src --cov-report=html   # With coverage
pytest tests/test_prompt.py -v                  # Specific test file
```

## Project Structure

```
psap-agent/
├── psap_agent/
│   ├── src/
│   │   ├── core/
│   │   │   ├── agent.py          # Agent initialization (Gemini + MCP client + checkpointer)
│   │   │   ├── manager.py        # AgentManager: streaming, Langfuse tracing, event formatting
│   │   │   ├── prompt.py         # System prompt with tool usage instructions
│   │   │   ├── storage.py        # Global in-memory checkpoint (dev mode)
│   │   │   ├── cache_manager.py  # Gemini context caching
│   │   │   └── agent_utils.py    # Message conversion utilities
│   │   ├── routes/
│   │   │   ├── stream.py         # SSE streaming endpoint
│   │   │   ├── health.py         # Health check
│   │   │   ├── history.py        # Conversation history
│   │   │   ├── threads.py        # Thread management
│   │   │   └── feedback.py       # Feedback to Langfuse
│   │   ├── api.py                # FastAPI app with lifespan
│   │   ├── main.py               # Entry point
│   │   ├── schema.py             # Request/response models
│   │   └── settings.py           # Pydantic settings
│   └── utils/
│       └── pylogger.py           # Structured logging
├── examples/
│   └── streamlit_app.py          # Streamlit chat UI
├── tests/
├── Containerfile                 # UBI9 Python 3.12 image
├── .env.example
└── pyproject.toml
```

## License

[Apache 2.0](LICENSE)
