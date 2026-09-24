"""Tests for provider-aware conversation handling in the Streamlit UI."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


APP_PATH = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
SPEC = importlib.util.spec_from_file_location("streamlit_app_under_test", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
streamlit_app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(streamlit_app)


def test_provider_for_model_recognizes_openai_gemini_and_claude():
    assert streamlit_app._provider_for_model("openai:gpt-6-luna") == "openai"
    assert streamlit_app._provider_for_model("o4-mini") == "openai"
    assert streamlit_app._provider_for_model("gemini-3.8-flash") == "gemini"
    assert streamlit_app._provider_for_model("claude-opus-4-6") == "claude"


def test_start_new_conversation_creates_thread_and_keeps_prior_messages():
    old_thread_id = "old-thread"
    messages = [{"role": "user", "content": "previous prompt"}]
    streamlit_app.st = SimpleNamespace(
        session_state=SimpleNamespace(
            messages=messages,
            thread_id=old_thread_id,
            last_message_time=123.0,
            inactivity_dismissed=False,
        )
    )

    streamlit_app._start_new_conversation()

    assert streamlit_app.st.session_state.thread_id != old_thread_id
    assert messages == [
        {"role": "user", "content": "previous prompt"},
        {"role": "divider"},
    ]
    assert streamlit_app.st.session_state.last_message_time is None
    assert streamlit_app.st.session_state.inactivity_dismissed is True


def test_start_new_conversation_does_not_duplicate_divider():
    messages = [{"role": "divider"}]
    streamlit_app.st = SimpleNamespace(
        session_state=SimpleNamespace(
            messages=messages,
            thread_id="old-thread",
            last_message_time=None,
            inactivity_dismissed=True,
        )
    )

    streamlit_app._start_new_conversation()

    assert messages == [{"role": "divider"}]
