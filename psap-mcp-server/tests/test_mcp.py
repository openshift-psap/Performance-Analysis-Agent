"""Tests for the MCP server module."""

from unittest.mock import Mock, patch

import pytest

from psap_mcp_server.src.mcp import PSAPMCPServer
from psap_mcp_server.src.tools.downstream_vllm_comparison_tool import (
    compare_downstream_vllm_releases,
)
from psap_mcp_server.src.tools.kernel_code_mapper_tool import (
    get_cross_repo_vllm_diff,
    get_downstream_vllm_tags,
)


class TestPSAPMCPServer:
    """Test the PSAPMCPServer class."""

    @patch("psap_mcp_server.src.mcp.force_reconfigure_all_loggers")
    @patch("psap_mcp_server.src.mcp.settings")
    @patch("psap_mcp_server.src.mcp.FastMCP")
    @patch("psap_mcp_server.src.mcp.logger")
    def test_init_success(
        self, mock_logger, mock_fastmcp, mock_settings, mock_force_reconfigure
    ):
        """Test successful initialization of PSAPMCPServer."""
        # Arrange
        mock_mcp = Mock()
        mock_fastmcp.return_value = mock_mcp
        mock_settings.PYTHON_LOG_LEVEL = "INFO"

        # Act
        server = PSAPMCPServer()

        # Assert
        assert server.mcp == mock_mcp
        mock_logger.info.assert_called_with(
            "PSAP MCP Server initialized successfully"
        )
        # In tools-first architecture, we only register tools
        mock_mcp.tool.assert_called()

    @patch("psap_mcp_server.src.mcp.force_reconfigure_all_loggers")
    @patch("psap_mcp_server.src.mcp.settings")
    @patch("psap_mcp_server.src.mcp.FastMCP")
    @patch("psap_mcp_server.src.mcp.logger")
    def test_init_failure(
        self, mock_logger, mock_fastmcp, mock_settings, mock_force_reconfigure
    ):
        """Test initialization failure handling."""
        # Arrange
        mock_fastmcp.side_effect = Exception("Test error")
        mock_settings.PYTHON_LOG_LEVEL = "INFO"

        # Act & Assert
        with pytest.raises(Exception, match="Test error"):
            PSAPMCPServer()

        mock_logger.error.assert_called_with(
            "Failed to initialize PSAP MCP Server: Test error"
        )

    @patch("psap_mcp_server.src.mcp.force_reconfigure_all_loggers")
    @patch("psap_mcp_server.src.mcp.settings")
    @patch("psap_mcp_server.src.mcp.FastMCP")
    def test_register_mcp_tools(
        self, mock_fastmcp, mock_settings, mock_force_reconfigure
    ):
        """Test MCP tools registration."""
        # Arrange
        mock_mcp = Mock()
        mock_fastmcp.return_value = mock_mcp
        mock_settings.PYTHON_LOG_LEVEL = "INFO"
        server = PSAPMCPServer()

        # Act
        server._register_mcp_tools()

        # Assert
        mock_mcp.tool.assert_called()

    @patch("psap_mcp_server.src.mcp.force_reconfigure_all_loggers")
    @patch("psap_mcp_server.src.mcp.settings")
    @patch("psap_mcp_server.src.mcp.FastMCP")
    def test_register_mcp_tools_functionality(
        self, mock_fastmcp, mock_settings, mock_force_reconfigure
    ):
        """Test that MCP tools registration includes all expected tools."""
        # Arrange
        mock_mcp = Mock()
        mock_fastmcp.return_value = mock_mcp
        mock_settings.PYTHON_LOG_LEVEL = "INFO"
        server = PSAPMCPServer()

        # Act
        server._register_mcp_tools()

        # Assert
        # Verify that tool() was called multiple times (once for each tool)
        assert mock_mcp.tool.call_count >= 3
        registered_tools = [
            call.args[0] for call in mock_mcp.tool.return_value.call_args_list
        ]
        assert get_downstream_vllm_tags in registered_tools
        assert get_cross_repo_vllm_diff in registered_tools
        assert compare_downstream_vllm_releases in registered_tools

    def test_server_attributes(self):
        """Test that server has required attributes for tools-first architecture."""
        # Arrange & Act
        with (
            patch("psap_mcp_server.src.mcp.settings") as mock_settings,
            patch("psap_mcp_server.src.mcp.FastMCP"),
            patch("psap_mcp_server.src.mcp.force_reconfigure_all_loggers"),
        ):
            mock_settings.PYTHON_LOG_LEVEL = "INFO"
            server = PSAPMCPServer()

        # Assert
        assert hasattr(server, "mcp")
        assert hasattr(server, "_register_mcp_tools")

    def test_tools_first_architecture_compliance(self):
        """Test that server adheres to tools-first architecture by not having resource/prompt methods."""
        # Arrange & Act
        with (
            patch("psap_mcp_server.src.mcp.settings") as mock_settings,
            patch("psap_mcp_server.src.mcp.FastMCP"),
            patch("psap_mcp_server.src.mcp.force_reconfigure_all_loggers"),
        ):
            mock_settings.PYTHON_LOG_LEVEL = "INFO"
            server = PSAPMCPServer()

        # Assert - These methods should NOT exist in tools-first architecture
        assert not hasattr(server, "_register_mcp_resources"), (
            "_register_mcp_resources should not exist in tools-first architecture"
        )
        assert not hasattr(server, "_register_mcp_prompts"), (
            "_register_mcp_prompts should not exist in tools-first architecture"
        )
