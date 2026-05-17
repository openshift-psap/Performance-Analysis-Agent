"""Streamlit App for RHAIIS Performance Analysis Agent.

This application demonstrates how to integrate with the RHAIIS Performance Analysis Agent's
simplified streaming API in a Streamlit application. It provides a clean
chat interface with real-time token streaming and message handling.

To run this app: streamlit run examples/streamlit_app.py

Make sure the RHAIIS Performance Analysis Agent server is running on http://localhost:5002
"""

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st


def get_logo_base64():
    """Load and encode the Red Hat logo as base64.

    Returns:
        Base64 encoded string of the logo image, or None if file not found.
    """
    logo_path = Path(__file__).parent / "assets" / "RedHat-logo.png"
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()

    if "pending_negative_feedback" not in st.session_state:
        st.session_state.pending_negative_feedback = None

    if "last_message_time" not in st.session_state:
        st.session_state.last_message_time = None

    if "inactivity_dismissed" not in st.session_state:
        st.session_state.inactivity_dismissed = False

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False


def apply_custom_css():
    """Inject custom CSS for button styling and optional dark mode."""
    dark = st.session_state.get("dark_mode", False)

    # Button styling (applies in both themes)
    button_css = """
    /* Style primary buttons (New Conversation) */
    button[kind="primary"] {
        border-radius: 1.5rem;
        font-weight: 600;
        padding: 0.4rem 1.2rem;
        transition: all 0.2s ease;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(204, 0, 0, 0.3);
    }
    /* Animated turn hint */
    @keyframes turnHintPulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    @keyframes turnHintSlide {
        0% { transform: translateX(0); }
        50% { transform: translateX(4px); }
        100% { transform: translateX(0); }
    }
    .turn-hint {
        animation: turnHintPulse 2.5s ease-in-out infinite;
        font-size: 0.85rem;
        color: #cc0000;
        font-weight: 500;
        text-align: right;
        padding-top: 0.5rem;
    }
    .turn-hint .arrow {
        display: inline-block;
        animation: turnHintSlide 1.5s ease-in-out infinite;
    }
    """

    # Force-override system/browser dark mode when light is selected
    if dark:
        theme_css = """
        .stApp {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stSidebar"] {
            background-color: #262730 !important;
        }
        [data-testid="stHeader"] {
            background-color: #0e1117 !important;
        }
        .stChatMessage {
            background-color: #1a1c23 !important;
        }
        .stTextInput > div > div > input,
        .stSelectbox > div > div {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        [data-testid="stChatInput"] textarea {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        .stMarkdown, .stText, .stCaption, p, span, label, h1, h2, h3, h4 {
            color: #fafafa !important;
        }
        [data-testid="stExpander"] {
            background-color: #1a1c23 !important;
            border-color: #3a3c47 !important;
        }
        button[kind="secondary"] {
            color: #fafafa !important;
            border-color: #3a3c47 !important;
        }
        /* Form submit buttons (login "Continue") */
        button[kind="secondaryFormSubmit"],
        [data-testid="stFormSubmitButton"] button,
        .stForm button {
            background-color: #cc0000 !important;
            color: #ffffff !important;
            border-color: #cc0000 !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
        }
        button[kind="secondaryFormSubmit"]:hover,
        [data-testid="stFormSubmitButton"] button:hover,
        .stForm button:hover {
            background-color: #a30000 !important;
            border-color: #a30000 !important;
        }
        """
    else:
        # Explicit light mode — overrides browser/system dark mode preference
        theme_css = """
        /* Root overrides to defeat prefers-color-scheme: dark */
        :root {
            color-scheme: light !important;
        }

        .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"] {
            background-color: #ffffff !important;
            color: #31333F !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {
            background-color: #f0f2f6 !important;
            color: #31333F !important;
        }
        [data-testid="stHeader"] {
            background-color: #ffffff !important;
        }
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background-color: #ffffff !important;
        }

        /* Text everywhere */
        .stMarkdown, .stText, .stCaption,
        p, span, label, li,
        h1, h2, h3, h4, h5, h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #31333F !important;
        }

        /* Text inputs */
        .stTextInput > div > div > input,
        .stTextInput input {
            background-color: #ffffff !important;
            color: #31333F !important;
            border-color: #d6d8de !important;
            caret-color: #31333F !important;
        }
        .stSelectbox > div > div,
        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #31333F !important;
        }
        /* Dropdown popover / menu list */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [data-baseweb="menu"] ul,
        [data-baseweb="menu"] li,
        [role="listbox"],
        [role="listbox"] li,
        [role="option"] {
            background-color: #ffffff !important;
            color: #31333F !important;
        }
        [data-baseweb="menu"] li:hover,
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background-color: #f0f2f6 !important;
        }

        /* Chat input bar */
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div {
            background-color: #ffffff !important;
            border-color: #d6d8de !important;
        }
        [data-testid="stChatInput"] textarea {
            background-color: #ffffff !important;
            color: #31333F !important;
            border-color: #d6d8de !important;
            caret-color: #31333F !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #9ca3af !important;
        }
        /* Chat input send button */
        [data-testid="stChatInput"] button,
        [data-testid="stChatInputSubmitButton"],
        [data-testid="stChatInput"] button[kind="secondary"],
        [data-testid="stChatInput"] button svg {
            color: #cc0000 !important;
            fill: #cc0000 !important;
            background-color: transparent !important;
            opacity: 1 !important;
        }

        /* Buttons — secondary (all normal buttons) */
        button[kind="secondary"],
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #ffffff !important;
            color: #31333F !important;
            border-color: #d6d8de !important;
        }
        button[kind="secondary"]:hover {
            background-color: #f0f2f6 !important;
            border-color: #b0b3ba !important;
        }
        /* Form submit buttons (login "Continue") */
        button[kind="secondaryFormSubmit"],
        [data-testid="stFormSubmitButton"] button,
        .stForm button {
            background-color: #cc0000 !important;
            color: #ffffff !important;
            border-color: #cc0000 !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
        }
        button[kind="secondaryFormSubmit"]:hover,
        [data-testid="stFormSubmitButton"] button:hover,
        .stForm button:hover {
            background-color: #a30000 !important;
            border-color: #a30000 !important;
        }

        /* Chat messages */
        .stChatMessage,
        [data-testid="stChatMessage"] {
            background-color: #f9f9fb !important;
        }

        /* Expanders */
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] div,
        [data-testid="stExpander"] pre,
        [data-testid="stExpander"] code {
            background-color: #f9f9fb !important;
            border-color: #e0e2e8 !important;
            color: #31333F !important;
        }
        /* JSON viewer inside expanders */
        [data-testid="stJson"],
        [data-testid="stJson"] > div {
            background-color: #f9f9fb !important;
        }

        /* Containers, alerts, info boxes */
        .stAlert, [data-testid="stAlert"] {
            background-color: #f0f7ff !important;
            color: #31333F !important;
        }

        /* Scrollable containers (example queries) */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            border-color: #e0e2e8 !important;
        }

        /* Checkbox */
        .stCheckbox label span {
            color: #31333F !important;
        }

        /* Code blocks and syntax highlighting */
        pre, code,
        .stMarkdown pre, .stMarkdown code,
        [data-testid="stChatMessage"] pre,
        [data-testid="stChatMessage"] code,
        .stCodeBlock, [data-testid="stCodeBlock"],
        .stCodeBlock > div, [data-testid="stCodeBlock"] > div,
        [data-testid="stCode"], [data-testid="stCode"] > div {
            background-color: #f5f5f5 !important;
            color: #31333F !important;
            border-color: #e0e2e8 !important;
        }
        /* Copy button inside code blocks */
        .stCodeBlock button, [data-testid="stCodeBlock"] button {
            color: #31333F !important;
        }

        /* Dividers */
        hr {
            border-color: #e0e2e8 !important;
        }

        /* Tooltips */
        [data-testid="stTooltipContent"],
        [data-testid="stTooltipContent"] p,
        [data-testid="stTooltipContent"] span,
        div[data-baseweb="tooltip"] > div,
        div[data-baseweb="tooltip"] > div > div {
            background-color: #ffffff !important;
            color: #31333F !important;
            border: 1px solid #d6d8de !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        }
        div[data-baseweb="tooltip"] [data-popper-arrow] {
            display: none !important;
        }
        """

    st.markdown(f"<style>{button_css}{theme_css}</style>", unsafe_allow_html=True)


def show_login_screen():
    """Display login screen to collect user's Red Hat email."""
    app_title = os.getenv("APP_TITLE", "RHAIIS Performance Analysis Agent")
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        # Display logo if available
        logo_base64 = get_logo_base64()
        if logo_base64:
            st.markdown(
                f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="200"></div>',
                unsafe_allow_html=True,
            )
        
        st.markdown(f"<h2 style='text-align: center;'>{app_title}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Please sign in with your Red Hat email to continue</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Email input form
        with st.form("login_form"):
            email = st.text_input(
                "Red Hat Email Address",
                placeholder="your.name@redhat.com",
                help="Enter your Red Hat email address"
            )
            
            submit = st.form_submit_button("Continue", use_container_width=True)
            
            if submit:
                if not email:
                    st.error("Please enter your email address")
                elif not email.endswith("@redhat.com"):
                    st.error("Please use your Red Hat email address (@redhat.com)")
                elif "@" not in email or len(email.split("@")[0]) < 1:
                    st.error("Please enter a valid email address")
                else:
                    st.session_state.user_email = email
                    # Save email to query params for persistence across refreshes
                    st.query_params["user_email"] = email
                    st.success(f"Welcome, {email.split('@')[0]}!")
                    st.rerun()


def stream_agent_response(
    message: str,
    thread_id: str,
    session_id: str,
    user_id: str,
    stream_tokens: bool = True,
    api_url: str = "http://localhost:8081",
    model: str | None = None,
    status_callback=None,
) -> tuple[str, List[Dict[str, Any]]]:
    """Stream response from the PSAP Agent using the simplified API.

    Args:
        message: User's input message
        thread_id: Conversation thread identifier
        session_id: Session identifier
        user_id: User identifier
        stream_tokens: Whether to stream individual tokens
        api_url: Base URL of the PSAP Agent API
        model: Optional Gemini model override
        status_callback: Optional callback(content_dict) called on each status event

    Returns:
        Tuple of (final_response, all_messages)
    """
    # Prepare request data
    request_data = {
        "message": message,
        "thread_id": thread_id,
        "session_id": session_id,
        "user_id": user_id,
        "stream_tokens": stream_tokens,
    }
    if model:
        request_data["model"] = model

    full_response = ""
    all_messages = []
    status_updates = []

    try:
        # Make streaming request to the simplified API
        response = requests.post(
            f"{api_url}/v1/stream",
            json=request_data,
            stream=True,
            timeout=300,
            headers={"Accept": "text/event-stream"},
        )
        response.raise_for_status()

        # Process the streaming response
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue

            # Check for completion marker
            if line.strip() == "[DONE]":
                break

            try:
                # Parse the event
                event = json.loads(line)
                event_type = event.get("type")
                content = event.get("content")

                if event_type == "token" and isinstance(content, str):
                    full_response += content

                elif event_type == "status" and isinstance(content, dict):
                    status_updates.append(content)
                    if status_callback:
                        status_callback(content)

                elif event_type == "message" and isinstance(content, dict):
                    all_messages.append(content)

                    if content.get("type") == "ai" and content.get("content"):
                        if not full_response:
                            full_response = content["content"]

                elif event_type == "error":
                    st.error(f"Agent Error: {content.get('message', 'Unknown error')}")
                    break

            except json.JSONDecodeError:
                st.warning(f"Failed to parse response line: {line[:100]}...")
                continue

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to agent: {e}")
        return "", []

    return full_response, all_messages


def send_feedback(
    run_id: str,
    score: float,
    user_query: str,
    assistant_response: str,
    api_url: str = "http://localhost:5002",
    comment: str = "",
) -> bool:
    """Send feedback for an agent response to the API.
    
    Args:
        run_id: The run_id of the agent response
        score: 1.0 for thumbs up, 0.0 for thumbs down
        user_query: The user's original question
        assistant_response: The agent's response
        api_url: Base URL of the PSAP Agent API
        comment: Optional free-text comment from the user
        
    Returns:
        True if feedback was successfully sent, False otherwise
    """
    try:
        kwargs = {
            "user_query": user_query,
            "assistant_response": assistant_response[:500],
            "timestamp": str(st.session_state.get("session_id", "")),
        }
        if comment:
            kwargs["user_comment"] = comment

        feedback_data = {
            "run_id": run_id,
            "key": "user-feedback",
            "score": score,
            "kwargs": kwargs,
        }
        
        response = requests.post(
            f"{api_url}/v1/feedback",
            json=feedback_data,
            timeout=5
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to send feedback: {e}")
        return False


def display_message(message: Dict[str, Any], role: str):
    """Display a message in the chat interface (content only, no wrapper)."""
    content = message.get("content", "")

    # Display the main content
    if content:
        st.write(content)

    # Display tool calls if present
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        with st.expander("🔧 Tool Calls", expanded=False):
            for i, tool_call in enumerate(tool_calls):
                st.json(
                    {
                        "tool": tool_call.get("name", "unknown"),
                        "args": tool_call.get("args", {}),
                        "id": tool_call.get("id", ""),
                    }
                )

    # Display metadata if present
    metadata = message.get("response_metadata", {})
    if metadata:
        with st.expander("📊 Metadata", expanded=False):
            st.json(metadata)


def main():
    """Main Streamlit application."""
    app_title = os.getenv("APP_TITLE", "RHAIIS Performance Analysis Agent")
    st.set_page_config(page_title=app_title, page_icon="📊", layout="wide")

    # Initialize session state (before any UI so dark_mode is available)
    initialize_session_state()

    # Display title with theme toggle in top-right corner
    logo_base64 = get_logo_base64()
    title_col, toggle_col = st.columns([8, 1])
    with title_col:
        if logo_base64:
            logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 40px; vertical-align: middle; margin-right: 10px;">'
            st.markdown(f'{logo_html}<span style="font-size: 2.0rem; font-weight: 600;">{app_title}</span>', unsafe_allow_html=True)
        else:
            st.title(f"📊 {app_title}")
    with toggle_col:
        theme_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
        if st.session_state.dark_mode:
            btn_style = "background-color: #ffffff; color: #31333F; border: 1px solid #d6d8de;"
        else:
            btn_style = "background-color: #1a1c23; color: #ffffff; border: 1px solid #1a1c23;"
        st.markdown(
            f'<style>div[data-testid="column"]:last-child .stButton button '
            f'{{ {btn_style} border-radius: 0.5rem !important; font-weight: 600 !important; }}</style>',
            unsafe_allow_html=True,
        )
        if st.button(theme_label, key="theme_toggle", use_container_width=True, disabled=st.session_state.is_streaming):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("**AI-powered performance analysis for RHAIIS benchmarking data** • Compare models, versions, and accelerators with intelligent insights")

    apply_custom_css()

    # Try to load stored email from localStorage on first load
    if not st.session_state.user_email and "email_check_done" not in st.session_state:
        # Use query params as a workaround to persist email
        try:
            query_params = st.query_params
            if "user_email" in query_params:
                stored_email = query_params["user_email"]
                if stored_email and stored_email.endswith("@redhat.com"):
                    st.session_state.user_email = stored_email
        except:
            pass
        st.session_state.email_check_done = True

    # Check if user is logged in
    if not st.session_state.user_email:
        show_login_screen()
        return

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Display logged-in user with Red Hat logo
        if logo_base64:
            st.markdown(
                f'<div style="background-color: rgba(49, 51, 63, 0.1); border-radius: 0.5rem; padding: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">'
                f'<img src="data:image/png;base64,{logo_base64}" style="height: 24px;">'
                f'<strong>{st.session_state.user_email}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(f"**{st.session_state.user_email}**")
        
        # Logout button
        if st.button("🚪 Sign Out", use_container_width=True, disabled=st.session_state.is_streaming):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.query_params.clear()
            st.rerun()
        
        st.divider()

        # Get API URL from environment variable (for containers) or use localhost as default
        default_api_url = os.getenv("AGENT_API_URL", "http://localhost:5002")

        model_options = {
            "⚡ Gemini 3 Flash (default) ▾": "gemini-3-flash-preview",
            "🧠 Gemini 3.1 Pro ▾": "gemini-3.1-pro-preview",
            "🟣 Claude Opus 4.6 (Vertex AI) ▾": "claude-opus-4-6",
            "🔵 Claude Sonnet 4.6 (Vertex AI) ▾": "claude-sonnet-4-6",
        }
        selected_label = st.selectbox(
            "🤖 LLM Model",
            options=list(model_options.keys()),
            index=0,
            help="Select the LLM model. Claude models require Vertex AI credentials.",
        )
        selected_model = model_options[selected_label]

        if selected_model == "gemini-3.1-pro-preview":
            st.warning(
                "**Gemini 3 Flash** is the default and recommended model for most use cases. "
                "Use **Pro** only when you need deeper reasoning or higher quality output. "
                "Pro has significantly higher latency and cost, please use it mindfully."
            )
        elif selected_model.startswith("claude-"):
            st.info(
                "**Claude models** run via Google Vertex AI "
                "Opus is best for complex architectural analysis; "
                "Sonnet is a good balance of quality and speed. Claude Models have higher cost, please use them mindfully."
            )

        # API status (cached to avoid flaky checks on every Streamlit re-render)
        def _check_api_health(url: str) -> bool:
            try:
                return requests.get(f"{url}/health", timeout=5).status_code == 200
            except Exception:
                return False

        cache_key = "_api_health_cache"
        now = time.time()
        cached = st.session_state.get(cache_key, {})
        if now - cached.get("ts", 0) > 15 or cached.get("url") != default_api_url:
            healthy = _check_api_health(default_api_url)
            st.session_state[cache_key] = {"ts": now, "url": default_api_url, "ok": healthy}
        else:
            healthy = cached["ok"]

        if healthy:
            st.success("API Status: ✅ Connected")
        else:
            st.error("API Status: ❌ Unreachable")

        st.divider()
        
        # Example queries
        st.subheader("💡 Example Queries")
        
        example_queries = [
            "Do we have any data for mistral models?",
            "What models were tested for vLLM-0.18.0 version?",
            "What vllm configs should a customer use for deploying gpt-oss-120b model on Nvidia gpu's?",
            "What's the cost per million tokens of Llama-3.3-70B-Instruct-FP8-dynamic on H200 using RHAIIS-3.3?",
            "What was the performance difference for deepseek-R1 model going from RHAIIS-3.2.2 to RHAIIS-3.2.3 on H200?",
            "Compare Llama-4-Maverick base fp16 model with the quantized fp8 model on H200 for 1k/1k profile using RHAIIS-3.2.3 version?",
            "Show me the GPU utilization for GPT-OSS-120B model on RHAIIS-3.2.4 for H200 with 1k/1k profile",
        ]
        
        # Use a scrollable container - height shows ~4 buttons
        with st.container(height=380):
            for i, query in enumerate(example_queries, 1):
                # Show the actual query text in the button
                if st.button(query, key=f"example_{i}", use_container_width=True, disabled=st.session_state.is_streaming):
                    st.session_state.example_query = query
        
        st.divider()

        api_url = st.text_input(
            "API URL",
            value=default_api_url,
            help="Base URL of the Performance Agent API",
        )

        stream_tokens = st.checkbox(
            "Stream Tokens",
            value=True,
            help="Enable real-time token streaming for faster response display",
        )

        st.divider()

        # Session information
        st.subheader("Session Info")
        st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        st.text(f"User ID: {st.session_state.user_email}")

        if st.button("🔄 New Conversation", use_container_width=True, key="new_conv_sidebar", type="primary", disabled=st.session_state.is_streaming):
            if st.session_state.messages:
                st.session_state.messages.append({"role": "divider"})
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.last_message_time = None
            st.session_state.inactivity_dismissed = True
            st.rerun()

        st.divider()
        
        # Advanced features in sidebar expander
        with st.expander("🔧 Advanced Features", expanded=False):
            st.text("Session State:")
            st.json(
                {
                    "thread_id": st.session_state.thread_id,
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_email,
                    "message_count": len(st.session_state.messages),
                }
            )

            st.text("Last Message:")
            if st.session_state.messages:
                st.json(st.session_state.messages[-1])

            # Export conversation
            if st.button("Export Conversation", use_container_width=True, disabled=st.session_state.is_streaming):
                conversation_data =                 {
                    "thread_id": st.session_state.thread_id,
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_email,
                    "messages": st.session_state.messages,
                    "export_timestamp": str(uuid.uuid4()),
                }

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(conversation_data, indent=2),
                    file_name=f"conversation_{st.session_state.thread_id[:8]}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    # Main chat interface
    st.subheader("Chat")

    # Add helpful info banner for empty chat
    if len(st.session_state.messages) == 0:
        st.info("""
        **🎯 What can I help you with?**
        
        • **Compare** models across versions, accelerators, or configurations  
        • **Analyze** cost efficiency and performance metrics  
        • **Discover** available models, versions, and test configurations  
        • **Detect** performance regressions between versions
        
        💬 Ask your question in natural language below!
        """)

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "divider":
            st.divider()
            st.caption("New conversation started")
            continue
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif isinstance(message.get("content"), dict) and message["content"].get("type") == "error":
            with st.chat_message("assistant"):
                st.error(message["content"]["content"])
        else:
            # For agent messages, display the structured content
            with st.chat_message("assistant"):
                content = message["content"]
                
                # Display the main response
                display_message(content, "assistant")
                
                # Add feedback buttons below the message
                run_id = content.get("run_id")
                
                if run_id:
                    feedback_key = f"feedback_{idx}_{run_id}"
                    pending = st.session_state.pending_negative_feedback

                    if feedback_key in st.session_state.feedback_given:
                        st.caption("✓ Feedback submitted")

                    elif pending and pending.get("feedback_key") == feedback_key:
                        st.markdown("**What could be improved?**")
                        comment = st.text_area(
                            "Your feedback (optional)",
                            key=f"comment_{idx}",
                            placeholder="e.g. The data was incorrect, missed the comparison I asked for...",
                            max_chars=500,
                        )
                        c1, c2, _ = st.columns([1, 1, 6])
                        with c1:
                            if st.button("Submit", key=f"submit_fb_{idx}", use_container_width=True, disabled=st.session_state.is_streaming):
                                send_feedback(
                                    pending["run_id"], 0.0,
                                    pending["user_query"],
                                    pending["assistant_response"],
                                    api_url,
                                    comment=comment,
                                )
                                st.session_state.feedback_given.add(feedback_key)
                                st.session_state.pending_negative_feedback = None
                                st.rerun()
                        with c2:
                            if st.button("Skip", key=f"skip_fb_{idx}", use_container_width=True, disabled=st.session_state.is_streaming):
                                send_feedback(
                                    pending["run_id"], 0.0,
                                    pending["user_query"],
                                    pending["assistant_response"],
                                    api_url,
                                )
                                st.session_state.feedback_given.add(feedback_key)
                                st.session_state.pending_negative_feedback = None
                                st.rerun()

                    else:
                        col1, col2, col3 = st.columns([1, 1, 10])
                        with col1:
                            thumbs_up = st.button("👍", key=f"thumbs_up_{idx}", help="Good response", disabled=st.session_state.is_streaming)
                        with col2:
                            thumbs_down = st.button("👎", key=f"thumbs_down_{idx}", help="Poor response", disabled=st.session_state.is_streaming)

                        if thumbs_up:
                            user_query = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                            assistant_response = content.get("content", "")
                            if send_feedback(run_id, 1.0, user_query, assistant_response, api_url):
                                st.session_state.feedback_given.add(feedback_key)
                                st.rerun()

                        if thumbs_down:
                            user_query = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                            assistant_response = content.get("content", "")
                            st.session_state.pending_negative_feedback = {
                                "feedback_key": feedback_key,
                                "run_id": run_id,
                                "user_query": user_query,
                                "assistant_response": assistant_response,
                            }
                            st.rerun()

                st.caption("Your responses are used to improve our Performance Agent.")

    # Turn count and new conversation button above the input
    turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    if turn_count > 0:
        tc_col1, tc_col2, tc_col3 = st.columns([3, 5, 3])
        with tc_col2:
            st.markdown(
                f'<div class="turn-hint">Turn {turn_count} · Switching topics? A new conversation gives the agent fresh context for more accurate answers <span class="arrow">→</span></div>',
                unsafe_allow_html=True,
            )
        with tc_col3:
            if st.button("🔄 New Conversation", key="new_conv_main", use_container_width=True, type="primary", disabled=st.session_state.is_streaming):
                if st.session_state.messages:
                    st.session_state.messages.append({"role": "divider"})
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.last_message_time = None
                st.session_state.inactivity_dismissed = True
                st.rerun()

    # Inactivity prompt - show if returning after 10+ minutes of silence
    inactivity_blocking = False
    if (turn_count > 0
            and st.session_state.last_message_time is not None
            and not st.session_state.inactivity_dismissed):
        elapsed = time.time() - st.session_state.last_message_time
        if elapsed > 600:
            inactivity_blocking = True
            minutes = int(elapsed // 60)
            with st.container(border=True):
                st.markdown(
                    f"**Welcome back!** It's been **{minutes} minutes** since your last message. "
                    "Starting fresh gives the agent clean context for more accurate answers."
                )
                c1, c2, _ = st.columns([1, 1, 4])
                with c1:
                    if st.button("Continue", key="continue_conv", use_container_width=True):
                        st.session_state.inactivity_dismissed = True
                        st.rerun()
                with c2:
                    if st.button("Start Fresh", key="start_fresh", use_container_width=True, type="primary"):
                        if st.session_state.messages:
                            st.session_state.messages.append({"role": "divider"})
                        st.session_state.thread_id = str(uuid.uuid4())
                        st.session_state.last_message_time = None
                        st.session_state.inactivity_dismissed = True
                        st.rerun()

    # Always show chat input (disabled while agent is working)
    prompt = st.chat_input(
        "Ask about RHAIIS performance, models, or configurations...",
        disabled=st.session_state.is_streaming,
    )

    # Check if an example query was clicked (takes priority over chat input)
    example_query = st.session_state.get("example_query")
    if example_query:
        st.session_state.example_query = None
        prompt = example_query

    # Block message processing while inactivity prompt is showing — save prompt so it survives the rerun
    if inactivity_blocking and prompt:
        st.session_state.pending_prompt = prompt
        st.info("Please choose **Continue** or **Start Fresh** above. Your message will be sent after.")
        st.stop()

    # New prompt OR saved prompt from inactivity block: set streaming flag, rerun
    active_new_prompt = prompt or (st.session_state.pending_prompt if not st.session_state.is_streaming else None)
    if active_new_prompt and not st.session_state.is_streaming:
        if not st.session_state.pending_prompt:
            st.session_state.pending_prompt = active_new_prompt
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != active_new_prompt or st.session_state.messages[-1].get("role") != "user":
            st.session_state.messages.append({"role": "user", "content": active_new_prompt})
        st.session_state.last_message_time = time.time()
        st.session_state.inactivity_dismissed = False
        st.session_state.is_streaming = True
        st.rerun()

    # Streaming run: buttons are already disabled, safe to do the blocking call
    if st.session_state.is_streaming and st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt

        _STATUS_LABELS = {
            "memory": ("🧠", "Recalling past interactions"),
            "thinking": ("🔍", "Analyzing your question"),
            "tool_call": ("🔧", "Calling tool: {detail}"),
            "tool_result": ("📊", "Received results from: {detail}"),
            "critic": ("🔎", "Quality check — reviewing response"),
            "critic_pass": ("✅", "Quality check passed"),
            "revising": ("✏️", "Improving response — {detail}"),
        }

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            status_placeholder = st.empty()

            def _on_status(content: dict):
                step = content.get("step", "")
                detail = content.get("detail", "")
                icon, template = _STATUS_LABELS.get(step, ("⏳", step))
                label = template.format(detail=detail) if detail else template.replace(" — {detail}", "").replace(": {detail}", "")
                status_placeholder.markdown(
                    f'<div style="display: flex; align-items: center; gap: 10px; padding: 8px 0;">'
                    f'<div class="red-spinner"></div>'
                    f'<span style="color: #555; font-size: 1.05em; font-weight: 500;">{icon} {label}</span>'
                    f'</div>'
                    f'<style>'
                    f'@keyframes red-spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}'
                    f'.red-spinner {{ width: 18px; height: 18px; border: 3px solid #f0f0f0; '
                    f'border-top: 3px solid #cc0000; border-radius: 50%; '
                    f'animation: red-spin 0.8s linear infinite; flex-shrink: 0; }}'
                    f'</style>',
                    unsafe_allow_html=True,
                )

            _on_status({"step": "thinking"})

            full_response, all_messages = stream_agent_response(
                message=active_prompt,
                thread_id=st.session_state.thread_id,
                session_id=st.session_state.session_id,
                user_id=st.session_state.user_email,
                stream_tokens=stream_tokens,
                api_url=api_url,
                model=selected_model,
                status_callback=_on_status,
            )

            status_placeholder.empty()

            if full_response:
                response_placeholder.write(full_response)

                run_id = None
                for msg in reversed(all_messages):
                    if msg.get("type") == "ai" and msg.get("run_id"):
                        run_id = msg["run_id"]
                        break

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": {
                            "type": "ai",
                            "content": full_response,
                            "run_id": run_id,
                            "messages": all_messages,
                        },
                    }
                )

            st.session_state.pending_prompt = None
            st.session_state.is_streaming = False

            if not full_response:
                st.session_state.messages.append(
                    {"role": "assistant", "content": {"type": "error", "content": "No response received from agent"}}
                )

            st.rerun()


if __name__ == "__main__":
    main()
