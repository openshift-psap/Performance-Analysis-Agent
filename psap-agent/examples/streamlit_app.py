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


def show_login_screen():
    """Display login screen to collect user's Red Hat email."""
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
        
        st.markdown("<h2 style='text-align: center;'>RHAIIS Performance Analysis Agent</h2>", unsafe_allow_html=True)
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
                    # Accumulate tokens for real-time display
                    full_response += content

                elif event_type == "message" and isinstance(content, dict):
                    # Store complete messages
                    all_messages.append(content)

                    # If this is the final AI message, use it as the response
                    if content.get("type") == "ai" and content.get("content"):
                        # If we haven't accumulated tokens, use the message content
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
    st.set_page_config(page_title="RHAIIS Performance Analysis Agent", page_icon="📊", layout="wide")

    # Display title with Red Hat logo
    logo_base64 = get_logo_base64()
    if logo_base64:
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 40px; vertical-align: middle; margin-right: 10px;">'
        st.markdown(f'{logo_html}<span style="font-size: 2.0rem; font-weight: 600;">RHAIIS Performance Analysis Agent</span>', unsafe_allow_html=True)
    else:
        st.title("📊 RHAIIS Performance Analysis Agent")
    
    st.markdown("**AI-powered performance analysis for RHAIIS benchmarking data** • Compare models, versions, and accelerators with intelligent insights")

    # Initialize session state
    initialize_session_state()

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
        
        # Display logged-in user
        st.info(f"👤 Signed in as:\n\n**{st.session_state.user_email}**")
        
        # Logout button
        if st.button("🚪 Sign Out", use_container_width=True):
            # Clear session state and query params to log out
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Clear email from query params
            st.query_params.clear()
            st.rerun()
        
        st.divider()

        # Get API URL from environment variable (for containers) or use localhost as default
        default_api_url = os.getenv("AGENT_API_URL", "http://localhost:5002")
        
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

        model_options = {
            "Gemini 3 Flash (default)": "gemini-3-flash-preview",
            "Gemini 3.1 Pro": "gemini-3.1-pro-preview",
        }
        selected_label = st.selectbox(
            "LLM Model",
            options=list(model_options.keys()),
            index=0,
            help="Select the Gemini model. Pro offers higher quality at higher latency/cost.",
        )
        selected_model = model_options[selected_label]

        if selected_model == "gemini-3.1-pro-preview":
            st.warning(
                "**Gemini 3 Flash** is the default and recommended model for most use cases. "
                "Use **Pro** only when you need deeper reasoning or higher quality output. "
                "Pro has significantly higher latency and cost, please use it mindfully."
            )

        # API test
        st.subheader("API Status")
        try:
            health_response = requests.get(f"{api_url}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error(f"❌ API Error: {health_response.status_code}")
        except Exception as e:
            st.error(f"❌ API Unreachable: {e}")

        st.divider()
        
        # Example queries
        st.subheader("💡 Example Queries")
        
        example_queries = [
            "Do we have any data for mistral models?",
            "What models were tested for RHAIIS-3.2.5 version?",
            "What's the cost per million tokens of Llama-3.3-70B-Instruct-FP8-dynamic on H200?",
            "What was the performance difference for deepseek model going from RHAIIS-3.2.2 to RHAIIS-3.2.3 on H200?",
            "Compare all common models across RHAIIS-3.2.2 and RHAIIS-3.2.3 on H200 for 512/2048 isl/osl profile",
            "Compare the performance of common models across RHAIIS-3.2.5 and sglang-0.5.5 on H200 for 1k/1k isl/osl profile",
            "Compare Llama-4-Maverick base fp16 model with the quantized fp8 model on H200 for 1k/1k profile",
            "Show me GPU utilization for GPT-OSS-120B model on RHAIIS-3.2.4 for H200 with 1k/1k profile",
        ]
        
        # Use a scrollable container - height shows ~4 buttons
        with st.container(height=380):
            for i, query in enumerate(example_queries, 1):
                # Show the actual query text in the button
                if st.button(query, key=f"example_{i}", use_container_width=True):
                    st.session_state.example_query = query
        
        st.divider()

        # Session information
        st.subheader("Session Info")
        st.text(f"Thread ID: {st.session_state.thread_id[:8]}...")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        st.text(f"User ID: {st.session_state.user_email}")

        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
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
            if st.button("Export Conversation", use_container_width=True):
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
    
    # Add helpful info banner at the top of chat
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
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
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
                            if st.button("Submit", key=f"submit_fb_{idx}", use_container_width=True):
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
                            if st.button("Skip", key=f"skip_fb_{idx}", use_container_width=True):
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
                            thumbs_up = st.button("👍", key=f"thumbs_up_{idx}", help="Good response")
                        with col2:
                            thumbs_down = st.button("👎", key=f"thumbs_down_{idx}", help="Poor response")

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

    # Always show chat input
    prompt = st.chat_input("Ask about RHAIIS performance, models, or configurations...")
    
    # Check if an example query was clicked (takes priority over chat input)
    example_query = st.session_state.get("example_query")
    if example_query:
        st.session_state.example_query = None  # Clear it
        prompt = example_query  # Override prompt with example query

    # Process the prompt (whether from example or chat input)
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Stream agent response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            # Show loading spinner
            with st.spinner("🔍 Analyzing performance data..."):
                # Stream the response
                full_response, all_messages = stream_agent_response(
                    message=prompt,
                    thread_id=st.session_state.thread_id,
                    session_id=st.session_state.session_id,
                    user_id=st.session_state.user_email,
                    stream_tokens=stream_tokens,
                    api_url=api_url,
                    model=selected_model,
                )

            # Display the final response
            if full_response:
                response_placeholder.write(full_response)

                # Extract run_id from the last AI message
                run_id = None
                for msg in reversed(all_messages):
                    if msg.get("type") == "ai" and msg.get("run_id"):
                        run_id = msg["run_id"]
                        break

                # Add to chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": {
                            "type": "ai",
                            "content": full_response,
                            "run_id": run_id,  # Store run_id for feedback
                            "messages": all_messages,  # Store all messages for debugging
                        },
                    }
                )
                
                # Force a rerun to display feedback buttons via chat history
                st.rerun()
            else:
                response_placeholder.error("No response received from agent")


if __name__ == "__main__":
    main()
