"""Tests for provider-aware chat model routing."""

import sys
from types import ModuleType

import pytest

from psap_agent.src.core.exceptions.exceptions import AppException
from psap_agent.src.core import model_factory


def test_get_model_provider_preserves_existing_model_names():
    assert model_factory.get_model_provider("gemini-3.8-flash") == "gemini"
    assert model_factory.get_model_provider("claude-sonnet-4-6") == "claude"


def test_get_model_provider_detects_openai_prefix_and_common_model_ids():
    assert model_factory.get_model_provider("openai:gpt-5.6") == "openai"
    assert model_factory.get_model_provider("gpt-5.6") == "openai"
    assert model_factory.get_model_provider("o4-mini") == "openai"


def test_openai_model_requires_api_key(monkeypatch):
    monkeypatch.setattr(model_factory.settings, "OPENAI_API_KEY", None)

    with pytest.raises(AppException, match="OPENAI_API_KEY"):
        model_factory.create_chat_model("openai:gpt-5.6")


def test_openai_model_strips_provider_prefix(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(model_factory.settings, "OPENAI_API_KEY", "test-openai-key")

    model = model_factory.create_chat_model("openai:gpt-5.6", temperature=0.0)

    assert model.kwargs == {"model": "gpt-5.6", "api_key": "test-openai-key"}


def test_gpt6_openai_model_uses_responses_api_with_reasoning_effort(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(model_factory.settings, "OPENAI_API_KEY", "test-openai-key")

    model = model_factory.create_chat_model(
        "openai:gpt-6-luna", reasoning_effort="xhigh"
    )

    assert model.kwargs == {
        "model": "gpt-6-luna",
        "api_key": "test-openai-key",
        "reasoning": {"effort": "xhigh"},
        "use_responses_api": True,
    }


def test_gpt6_openai_model_uses_responses_api_at_default_effort(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(model_factory.settings, "OPENAI_API_KEY", "test-openai-key")

    model = model_factory.create_chat_model("openai:gpt-6-sol")

    assert model.kwargs == {
        "model": "gpt-6-sol",
        "api_key": "test-openai-key",
        "use_responses_api": True,
    }


def test_openai_model_requires_a_model_id(monkeypatch):
    monkeypatch.setattr(model_factory.settings, "OPENAI_API_KEY", "test-openai-key")

    with pytest.raises(AppException, match="model ID"):
        model_factory.create_chat_model("openai:")
