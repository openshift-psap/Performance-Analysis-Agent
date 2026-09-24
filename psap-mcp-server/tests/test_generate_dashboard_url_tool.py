"""Tests for dashboard URL configuration behavior."""

import pytest

from psap_mcp_server.src.settings import settings
from psap_mcp_server.src.tools.generate_dashboard_url_tool import (
    generate_dashboard_url,
)


@pytest.mark.asyncio
async def test_generate_dashboard_url_uses_configured_base_url(monkeypatch):
    """Dashboard links use the configured URL and normalize a trailing slash."""
    monkeypatch.setattr(settings, "DASHBOARD_BASE_URL", "https://dashboard.example.com/")

    result = await generate_dashboard_url(
        models=["example/model"],
        section="cost_analysis",
    )

    assert result.startswith("Dashboard URL: https://dashboard.example.com/?")


@pytest.mark.asyncio
async def test_generate_dashboard_url_requires_configuration(monkeypatch):
    """Dashboard links fail clearly when no dashboard URL is configured."""
    monkeypatch.setattr(settings, "DASHBOARD_BASE_URL", None)

    result = await generate_dashboard_url(models=["example/model"])

    assert result == (
        "Dashboard URL unavailable: DASHBOARD_BASE_URL is not configured. "
        "Set it in the MCP server environment to enable dashboard links."
    )
