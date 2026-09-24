"""Tests for the prompt module."""

from unittest.mock import patch

from psap_agent.src.core.prompt import get_current_date, get_system_prompt


class TestPrompt:
    """Test cases for prompt functions."""

    def test_get_current_date(self):
        """Test get_current_date returns formatted date string."""
        date_str = get_current_date()
        assert isinstance(date_str, str)
        # Should be in format "Month Day, Year" (e.g., "December 25, 2024")
        assert len(date_str.split()) == 3

    def test_get_system_prompt(self):
        """Test get_system_prompt returns non-empty string."""
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "PSAP Agent" in prompt
        assert "Today's date is" in prompt
        assert "load_skill" in prompt
        assert "benchmark_analysis" in prompt
        assert "clarification_and_comparison" in prompt
        assert "vllm_performance_triage" in prompt
        assert "DEEP PERFORMANCE ANALYSIS WORKFLOW" not in prompt

    @patch("psap_agent.src.core.prompt.get_current_date")
    def test_get_system_prompt_includes_date(self, mock_get_date):
        """Test that get_system_prompt includes the current date."""
        mock_get_date.return_value = "December 25, 2024"
        prompt = get_system_prompt()
        assert "Today's date is December 25, 2024" in prompt
