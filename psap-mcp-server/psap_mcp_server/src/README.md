# MCP Server Core Implementation

This directory contains the core MCP server implementation following a **tools-first architecture** for maximum agent compatibility.

## 📁 Directory Structure

```
src/
├── main.py              # Server entry point
├── api.py               # FastAPI application setup
├── mcp.py               # MCP server implementation & tool registration
├── settings.py          # Configuration management
├── tools/               # 🎯 ALL MCP capabilities as tools
└── assets/              # 📁 Static assets accessed by tools
```

## 🎯 **Tools-First Philosophy**

**Everything is a tool** - prompts, resources, and business logic are all implemented as MCP tools for:
- ✅ Universal compatibility (LangGraph, CrewAI, Claude Desktop, etc.)
- ✅ Consistent interface and error handling
- ✅ Better agent understanding with structured documentation
- ✅ Simplified development and testing

## 🚀 **Quick Start**

1. **Add your tools** → `tools/your_domain_tool.py`
2. **Add static assets** → `assets/your_file.png`
3. **Register tools** → Update `mcp.py`
4. **Test** → `pytest tests/test_tools.py`

## 📝 **Next Steps**

- See `tools/README.md` for tool development guidelines
- See `assets/README.md` for asset management
- Check `.cursor/mcp-rules.md` for detailed patterns
