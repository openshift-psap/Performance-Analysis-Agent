"""Model-specific response timeout policy tests."""

from psap_agent.src.core.response_timeout import get_agent_response_timeout_seconds


def test_argon_uses_at_least_fifteen_minutes():
    assert get_agent_response_timeout_seconds("gemini-4-argon", 600, 900) == 900
    assert get_agent_response_timeout_seconds("Gemini-4-Argon-preview", 600, 900) == 900


def test_non_argon_models_keep_the_configured_timeout():
    assert get_agent_response_timeout_seconds("gemini-3.8-flash", 600, 900) == 600
    assert get_agent_response_timeout_seconds("openai:gpt-6-luna", 600, 900) == 600


def test_larger_global_timeout_is_not_reduced_for_argon():
    assert get_agent_response_timeout_seconds("gemini-4-argon", 1200, 900) == 1200
