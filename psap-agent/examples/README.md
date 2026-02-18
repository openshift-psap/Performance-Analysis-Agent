# PSAP Agent - Streamlit Interface

This directory contains the Streamlit application for interacting with the PSAP Performance Agent. The agent provides intelligent performance analysis and insights for RHAIIS, and LLM-D benchmarks.

## 📁 Streamlit Application

### Performance Chat Interface (`streamlit_app.py`)

A full-featured chat application built with Streamlit for performance analysis:
- **Real-time chat interface** with message history and user authentication
- **Performance-focused queries** with intelligent clarification
- **Token streaming visualization** for responsive UX
- **Session management** with thread and session persistence
- **User feedback** with thumbs up/down buttons
- **Export functionality** for conversation data

**Key Features:**
- Live token streaming with visual updates
- Tool call visualization with expandable details
- API health monitoring
- Red Hat branding with logo
- Example performance queries
- Dashboard link generation
- Feedback tracking via Langfuse
- Email-based user authentication
- Session persistence across browser refreshes

**To Run:**
```bash
# Install Streamlit if not already installed
pip install streamlit requests

# Run the app
streamlit run examples/streamlit_app.py

# Open http://localhost:8501 in your browser
```

## 🔗 API Reference

### Request Format

The Streamlit app sends requests to the PSAP Agent API:

```json
{
  "message": "Compare Llama 3.3 70B performance across RHAIIS 3.2.3 and 3.2.4",
  "thread_id": "unique-thread-id",
  "session_id": "unique-session-id",
  "user_id": "user@redhat.com",
  "stream_tokens": true
}
```

### Response Format

The API returns Server-Sent Events with this format:

```json
{"type": "message", "content": {"type": "ai", "content": "I'll analyze..."}}
{"type": "token", "content": " performance"}
{"type": "message", "content": {"type": "tool", "name": "discover_configurations"}}
{"type": "message", "content": {"type": "ai", "content": "Based on the data..."}}
[DONE]
```

**Event Types:**
- `message` - Complete messages (AI responses, tool calls, tool results)
- `token` - Individual tokens for real-time streaming
- `error` - Error messages with recovery information
- `[DONE]` - Stream completion marker

### Available Endpoints

- `GET /health` - Health check
- `GET /v1/health` - Detailed health with tool count
- `POST /v1/stream` - Streaming chat endpoint
- `POST /v1/feedback` - Submit feedback
- `GET /v1/history/{thread_id}` - Get conversation history
- `GET /v1/threads` - List all threads

## 🚀 Getting Started

### Prerequisites

1. **PSAP Agent Server Running**
   ```bash
   # Start the PSAP Agent server
   cd psap-agent
   .venv/bin/python3 -m psap_agent.src.main
   ```

2. **MCP Server Running**
   ```bash
   # Start the MCP server (in a separate terminal)
   cd psap-mcp-server
   .venv/bin/python3 -m psap_mcp_server.src.main
   ```

3. **Install Streamlit Dependencies**
   ```bash
   pip install streamlit requests
   ```

### Quick Test

Test the API is working:

```bash
# Health check
curl http://localhost:5002/health

# Check agent is connected to MCP
curl http://localhost:5002/v1/health
```

## 🎯 Best Practices

### 1. Session Management
- Login with your Red Hat email to track your usage
- Each browser session maintains its own conversation history
- Session persists across browser refreshes via URL parameters

### 2. Performance Queries
- Use the example queries as templates
- The agent will ask clarifying questions if needed
- Be specific about models, versions, accelerators, and profiles

### 3. Feedback
- Use thumbs up/down buttons to rate agent responses
- Feedback is tracked in Langfuse for quality improvement
- You can only submit feedback once per response

### 4. Dashboard Links
- Agent provides direct links to the performance dashboard
- Links have pre-applied filters for the data discussed
- Click the link to visualize the performance data

## 🔧 Key Features

The PSAP Agent includes:

- **User Authentication**: Red Hat email login required
- **Langfuse Tracing**: Automatic tracing, analytics, and feedback tracking
- **PostgreSQL Persistence**: Conversation history and checkpointing
- **Context Caching**: Gemini 2.5 Flash with explicit caching for cost savings
- **MCP Tools**: 11 specialized tools for performance analysis
- **Error Monitoring**: Comprehensive error logging and recovery

## 📚 Additional Resources

- [PSAP Agent Documentation](../README.md)
- [Technical Architecture](../../TECHNICAL_ARCHITECTURE.md)
- [Deployment Guide](../../DEPLOYMENT_GUIDE.md)
- [Quick Start Guide](../../QUICK_START.md)
- [Feedback System Documentation](../../FEEDBACK_SYSTEM.md)

## 🐛 Troubleshooting

### Common Issues

**"API Unreachable" in Streamlit**
- Ensure PSAP Agent server is running on http://localhost:5002
- Ensure MCP server is running on http://localhost:5001
- Check that PostgreSQL is running (for conversation persistence)
- Verify firewall settings and port availability

**"No response received from agent"**
- Complex queries may take 2-5 minutes to process
- Check agent logs for errors
- Verify MCP server has loaded all 11 tools
- Ensure performance dashboard CSV data is accessible

**Email Login Not Persisting**
- Check browser cookies/local storage settings
- Try refreshing the page after login
- URL parameters store the session (don't modify the URL)

**Feedback Buttons Not Working**
- Ensure Langfuse is running on http://localhost:3000
- Check that Langfuse API keys are configured in `.env`
- Verify PostgreSQL is running for Langfuse database

**Tool Errors**
- Check that `consolidated_dashboard.csv` exists in performance-dashboard folder
- Verify Grafana API credentials if querying Grafana metrics
- Check MCP server logs for detailed error messages

### Debug Mode

Enable detailed logging:

```bash
# Check agent logs
cd psap-agent
.venv/bin/python3 -m psap_agent.src.main

# Check MCP logs
cd psap-mcp-server
.venv/bin/python3 -m psap_mcp_server.src.main

# Streamlit error details
# Error details are shown automatically in the UI
```

### Starting All Services

Use the automated script:

```bash
cd /Users/haumesh/Desktop/mcp
./start_all_services.sh
```

Or start manually (see [START_SERVICES.md](../../START_SERVICES.md) for details).

For more help, check the [Deployment Guide](../../DEPLOYMENT_GUIDE.md) or main project documentation.
