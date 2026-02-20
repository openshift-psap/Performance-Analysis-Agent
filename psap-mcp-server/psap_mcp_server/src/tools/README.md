# MCP Tools Directory

All MCP server capabilities are implemented as **tools** for maximum compatibility with AI agents.

## 🔧 **Tool Development Guidelines**

### **Required Tool Documentation Format**

**CRITICAL**: All tools MUST use this structured format for agent compatibility:

```python
def your_tool_function(
    input_param: str,
    optional_param: str = "default"
) -> Dict[str, Any]:
    """
    TOOL_NAME=your_tool_function
    DISPLAY_NAME=Human-Readable Tool Name
    USECASE=When/why to use this tool (specific scenarios)
    INSTRUCTIONS=Step-by-step usage guide for agents
    INPUT_DESCRIPTION=Expected data format with examples
    OUTPUT_DESCRIPTION=What format you'll receive back
    EXAMPLES=your_tool_function("example_input", "optional_value")
    PREREQUISITES=What to do first (workflow sequence)
    RELATED_TOOLS=Other tools to use with this one

    Traditional docstring for developers goes here...
    """
    try:
        # Input validation
        if not input_param:
            raise ValueError("input_param is required")

        # Your business logic here
        result = process_input(input_param, optional_param)

        return {
            "status": "success",
            "operation": "your_operation",
            "result": result,
            "message": "Operation completed successfully"
        }

    except Exception as e:
        return {
            "status": "error",
            "operation": "your_operation",
            "error": str(e),
            "message": "Operation failed"
        }
```

### **Tool Registration**

Add your tool to `../mcp.py`:

```python
# Import your tool
from psap_mcp_server.src.tools.your_tool import your_tool_function

# Register in _register_mcp_tools()
self.mcp.tool()(your_tool_function)
```

## 📋 **Current Tools**

- `query_performance_tool.py` - Query AI model performance metrics (throughput, latency, TTFT, ITL)
- `compare_performance_tool.py` - Compare configurations and versions with geometric means
- `cost_efficiency_tool.py` - Calculate cost per million tokens analysis
- `regression_analysis_tool.py` - Version-to-version performance change analysis
- `dataset_metadata_tool.py` - Discover available models, accelerators, profiles, and versions
- `discover_configurations_tool.py` - Find what was actually tested with specific filters
- `grafana_metrics_tool.py` - Query GPU and vLLM metrics from Grafana
- `compare_grafana_metrics_tool.py` - Compare Grafana metrics between two benchmark runs
- `generate_dashboard_url_tool.py` - Generate dashboard URLs with pre-applied filters
- `generate_grafana_url_tool.py` - Generate Grafana dashboard URLs for specific runs
- `vllm_release_notes_tool.py` - Fetch vLLM release notes, compare versions, inspect PRs
- `pytorch_profile_tool.py` - Analyze and compare PyTorch profiler traces across versions
- `kernel_code_mapper_tool.py` - Map kernel names to vLLM source, fetch code and diffs from GitHub

## ✅ **Best Practices**

1. **Consistent Returns**: Always return `Dict[str, Any]` with `status` field
2. **Error Handling**: Wrap in try/catch, return structured errors
3. **Input Validation**: Validate all inputs before processing
4. **Logging**: Use `from psap_mcp_server.utils.pylogger import get_python_logger`
5. **Testing**: Add tests to `../../tests/test_tools.py`

## 🎯 **Agent-Friendly Tips**

- Use **clear, action-oriented names** (`generate_report` not `report_generator`)
- Include **concrete examples** in EXAMPLES field
- Specify **prerequisites** for workflow guidance
- List **related tools** to help agents chain operations
- Keep **error messages** descriptive but concise
