"""PSAP MCP Server implementation.

This module contains the main PSAP MCP Server class that provides
tools for MCP clients. It uses FastMCP to register and manage MCP capabilities.
"""

from fastmcp import FastMCP

from psap_mcp_server.src.settings import settings

# Import performance analysis tools
from psap_mcp_server.src.tools.query_performance_tool import (
    query_performance_metrics,
)
from psap_mcp_server.src.tools.compare_performance_tool import (
    compare_configurations,
    compare_versions_comprehensive,
)
from psap_mcp_server.src.tools.cost_efficiency_tool import (
    calculate_cost_efficiency,
)
from psap_mcp_server.src.tools.energy_efficiency_tool import (
    calculate_energy_efficiency,
)
from psap_mcp_server.src.tools.regression_analysis_tool import (
    analyze_regression,
)
from psap_mcp_server.src.tools.dataset_metadata_tool import (
    get_dataset_metadata,
)
from psap_mcp_server.src.tools.discover_configurations_tool import (
    discover_configurations,
)
from psap_mcp_server.src.tools.grafana_metrics_tool import (
    query_grafana_metrics,
)
from psap_mcp_server.src.tools.compare_grafana_metrics_tool import (
    compare_grafana_metrics,
)
from psap_mcp_server.src.tools.generate_dashboard_url_tool import (
    generate_dashboard_url,
)
from psap_mcp_server.src.tools.generate_grafana_url_tool import (
    generate_grafana_url,
)
from psap_mcp_server.src.tools.vllm_release_notes_tool import (
    get_vllm_release_notes,
    compare_vllm_versions,
    get_version_mappings,
    get_vllm_pull_request,
)
from psap_mcp_server.src.tools.pytorch_profile_tool import (
    analyze_pytorch_profile,
    compare_pytorch_profiles,
    list_available_profiles,
    check_profile_status,
    analyze_performance_insights,
    analyze_trace_structure,
    compare_trace_structures,
    get_kernel_call_stacks,
)
from psap_mcp_server.src.tools.kernel_code_mapper_tool import (
    map_kernel_to_vllm_code,
    get_kernel_categories,
    correlate_kernel_with_changes,
    fetch_vllm_source,
    get_vllm_code_diff,
)
from psap_mcp_server.src.tools.vllm_log_tool import (
    fetch_vllm_logs,
    compare_vllm_logs,
)
from psap_mcp_server.src.tools.vllm_performance_triage_tool import (
    get_vllm_performance_triage_guide,
)
from psap_mcp_server.utils.pylogger import (
    force_reconfigure_all_loggers,
    get_python_logger,
)

logger = get_python_logger()


class PSAPMCPServer:
    """Main PSAP MCP Server implementation following tools-first architecture.

    This server provides only tools, not resources or prompts, adhering to
    the tools-first architectural pattern for MCP servers.
    """

    def __init__(self):
        """Initialize the MCP server with PSAP tools following tools-first architecture."""
        try:
            # Initialize FastMCP server
            self.mcp = FastMCP("psap")

            # Force reconfigure all loggers after FastMCP initialization to ensure structured logging
            force_reconfigure_all_loggers(settings.PYTHON_LOG_LEVEL)

            self._register_mcp_tools()

            logger.info("PSAP MCP Server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PSAP MCP Server: {e}")
            raise

    def _register_mcp_tools(self) -> None:
        """Register MCP tools for RHAIIS performance analysis (tools-first architecture).

        Registers all available tools with the FastMCP server instance.
        In tools-first architecture, the server only provides tools.
        Currently includes:
        - query_performance_metrics: Query AI model performance data
        - compare_configurations: Compare performance across configurations
        - compare_versions_comprehensive: Compare versions with peak, mean, median, and geometric mean
        - calculate_cost_efficiency: Calculate cost per million tokens analysis
        - calculate_energy_efficiency: Calculate GPU energy consumption and energy per million tokens
        - analyze_regression: Version-to-version performance change analysis
        - get_dataset_metadata: Discover all available models, accelerators, profiles, and versions
        - discover_configurations: Discover what was actually tested with specific filters
        - query_grafana_metrics: Query GPU and vLLM metrics from Grafana for specific benchmark runs
        - compare_grafana_metrics: Compare vLLM and DCGM metrics between two benchmark runs
        - generate_dashboard_url: Generate URLs to dashboard with pre-applied filters for visualization
        - generate_grafana_url: Generate Grafana dashboard URLs for viewing GPU metrics of specific runs
        - get_vllm_release_notes: Fetch vLLM release notes from GitHub to understand version changes
        - compare_vllm_versions: Compare changelog between vLLM versions to explain performance changes
        - get_version_mappings: View RHAIIS to vLLM version mappings for version resolution
        - get_vllm_pull_request: Get details about a specific vLLM PR to understand code changes
        - analyze_pytorch_profile: Analyze PyTorch profiler traces for kernel-level performance
        - compare_pytorch_profiles: Compare profiler traces between vLLM versions
        - list_available_profiles: List available PyTorch profile traces
        - get_kernel_call_stacks: Get real Python/C++ call stacks for a kernel from profile traces
        - analyze_trace_structure: Block-level structural analysis with median block, streams, overhead
        - compare_trace_structures: Compare trace structures between versions with root-cause detection
        - map_kernel_to_vllm_code: Map kernel names to vLLM source code locations
        - get_kernel_categories: Get information about kernel categories
        - correlate_kernel_with_changes: Correlate kernel performance with code changes
        - fetch_vllm_source: Fetch source code from any public GitHub repo (defaults to vLLM) at a specific version
        - get_vllm_code_diff: Fetch code diff between two vLLM versions for specific files
        - fetch_vllm_logs: Fetch and parse vLLM server logs for a specific version
        - compare_vllm_logs: Compare vLLM server logs between two versions (auto-enriches with ALL pinned dependency versions from requirements/cuda.txt)
        - get_vllm_performance_triage_guide: Retrieve vLLM performance triage guidance (5-step diagnostic workflow)
        """
        # Register performance analysis tools
        self.mcp.tool()(query_performance_metrics)
        self.mcp.tool()(compare_configurations)
        self.mcp.tool()(compare_versions_comprehensive)
        self.mcp.tool()(calculate_cost_efficiency)
        self.mcp.tool()(calculate_energy_efficiency)
        self.mcp.tool()(analyze_regression)
        self.mcp.tool()(get_dataset_metadata)
        self.mcp.tool()(discover_configurations)
        self.mcp.tool()(query_grafana_metrics)
        self.mcp.tool()(compare_grafana_metrics)
        self.mcp.tool()(generate_dashboard_url)
        self.mcp.tool()(generate_grafana_url)
        # Register vLLM release notes tools (supports RHAIIS version mapping)
        self.mcp.tool()(get_vllm_release_notes)
        self.mcp.tool()(compare_vllm_versions)
        self.mcp.tool()(get_version_mappings)
        self.mcp.tool()(get_vllm_pull_request)
        # Register PyTorch profile analysis tools
        self.mcp.tool()(analyze_pytorch_profile)
        self.mcp.tool()(compare_pytorch_profiles)
        self.mcp.tool()(list_available_profiles)
        self.mcp.tool()(check_profile_status)
        self.mcp.tool()(analyze_performance_insights)
        self.mcp.tool()(get_kernel_call_stacks)
        # Register trace structure analysis tools (block segmentation, median block, streams, overhead)
        self.mcp.tool()(analyze_trace_structure)
        self.mcp.tool()(compare_trace_structures)
        # Register kernel-to-code mapping tools
        self.mcp.tool()(map_kernel_to_vllm_code)
        self.mcp.tool()(get_kernel_categories)
        self.mcp.tool()(correlate_kernel_with_changes)
        # Register vLLM source code access tools
        self.mcp.tool()(fetch_vllm_source)
        self.mcp.tool()(get_vllm_code_diff)
        # Register vLLM log analysis tools
        self.mcp.tool()(fetch_vllm_logs)
        self.mcp.tool()(compare_vllm_logs)
        # Register vLLM performance triage guide
        self.mcp.tool()(get_vllm_performance_triage_guide)