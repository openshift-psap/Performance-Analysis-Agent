"""Agent Manager for the PSAP agent system.

This module provides the AgentManager class that orchestrates agent operations,
handles streaming responses, and manages the conversion between LangGraph events
and simplified streaming.
"""

import asyncio
import inspect
from collections.abc import AsyncGenerator
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt

from psap_agent.src.core.agent import get_memory_manager, get_psap_agent
from psap_agent.src.core.agent_utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from psap_agent.src.core.evaluator import evaluate_response
from psap_agent.src.core.reflection import run_critic
from psap_agent.src.core.storage import register_thread
from psap_agent.src.schema import StreamRequest
from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

Langfuse()

langfuse_handler = CallbackHandler()

app_logger = get_python_logger(settings.PYTHON_LOG_LEVEL)


class AgentManager:
    """Manager class for handling agent operations and streaming responses.

    This class provides a simplified interface for agent interactions while
    preserving all enterprise features like authentication, tracing, and
    error handling from the original implementation.
    """

    def __init__(self, redhat_sso_token: str | None = None):
        """Initialize the AgentManager.

        Args:
            redhat_sso_token: Optional SSO token for enterprise authentication.
        """
        self.redhat_sso_token = redhat_sso_token
        self._agent: Pregel | None = None
        self._current_tool_call_id: str | None = None  # Track current active tool call

    async def stream_response(
        self, request: StreamRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response with simplified event structure.

        This method provides streaming functionality while ensuring that conversation
        state is saved only once at the end, not during intermediate streaming.

        Args:
            request: The streaming request containing user input and configuration.

        Yields:
            Simplified event dictionaries with 'type' and 'content' fields.
        """
        async with get_psap_agent(
            self.redhat_sso_token,
            enable_checkpointing=True,
            model_name=request.model,
        ) as persistent_agent:
            try:
                kwargs, run_id, thread_id, memory_context = await self._handle_input(
                    request, persistent_agent
                )

                app_logger.info(
                    f"AgentManager streaming response for run_id: {run_id}, thread_id: {thread_id}"
                )

                self._current_tool_call_id = None
                effective_session_id = request.session_id or thread_id
                effective_user_id = request.user_id or "anonymous"

                def _make_status(step: str, detail: str = ""):
                    return {"type": "status", "content": {"step": step, "detail": detail}}

                async def _run_agent_pass(agent_kwargs: dict, out: dict):
                    """Async generator: yields status events in real-time, stores results in out."""
                    out["msgs"] = []
                    out["events"] = []
                    async for stream_event in persistent_agent.astream(
                        **agent_kwargs, stream_mode=["updates", "messages", "custom"]
                    ):
                        if not isinstance(stream_event, tuple):
                            continue
                        sm, ev = stream_event
                        self._update_tool_call_tracking(sm, ev)
                        if sm == "updates":
                            for _node, updates in ev.items():
                                if updates and "messages" in updates:
                                    for msg in updates["messages"]:
                                        out["msgs"].append(msg)
                                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                                            for tc in msg.tool_calls:
                                                yield _make_status("tool_call", tc.get("name", "tool"))
                                        elif getattr(msg, "type", "") == "tool":
                                            yield _make_status("tool_result", getattr(msg, "name", "tool"))
                        formatted = self._format_events(
                            sm, ev, request.stream_tokens,
                            run_id, thread_id, effective_session_id,
                        )
                        for fe in formatted:
                            if fe:
                                out["events"].append(fe)

                if memory_context:
                    yield _make_status("memory", "Retrieved relevant context from past interactions")

                yield _make_status("thinking", "Analyzing your question")

                # --- First pass: buffer response, don't stream yet ---
                result = {}
                async for status_evt in _run_agent_pass(kwargs, result):
                    yield status_evt
                collected_messages = result["msgs"]
                buffered_events = result["events"]
                # Accumulate tool-related messages across all passes for the evaluator.
                # Revisions may reuse tool results from the thread without re-calling,
                # so the evaluator needs to see tool calls from every pass.
                all_tool_messages: list = [
                    m for m in collected_messages
                    if (hasattr(m, "tool_calls") and m.tool_calls)
                    or getattr(m, "type", "") == "tool"
                ]

                # --- Reflection loop (Tier 2): critic reviews before delivery ---
                revision_count = 0
                max_revisions = settings.MAX_REFLECTION_ITERATIONS if settings.ENABLE_REFLECTION else 0

                while revision_count < max_revisions and collected_messages:
                    try:
                        yield _make_status("critic", "Reviewing response for quality")

                        critic_result = await run_critic(
                            collected_messages, request.message,
                            memory_context=memory_context,
                            trace_id=run_id,
                        )
                        if critic_result["verdict"] == "pass":
                            app_logger.info(f"Critic passed on attempt {revision_count + 1}")
                            yield _make_status("critic_pass", "Quality check passed")
                            break

                        revision_count += 1
                        issues = critic_result.get("issues", [])
                        instructions = critic_result.get("revision_instructions", "")
                        app_logger.info(
                            f"Critic requested revision {revision_count}/{max_revisions}: {issues}"
                        )

                        yield _make_status("revising", f"Improving response (revision {revision_count})")

                        # Build a tool results summary so the revision has
                        # the actual data inline rather than relying on
                        # thread history where its own fabrications live.
                        tool_results_block = ""
                        if all_tool_messages:
                            parts = []
                            pending: dict[str, str] = {}
                            for m in all_tool_messages:
                                if hasattr(m, "tool_calls") and m.tool_calls:
                                    for tc in m.tool_calls:
                                        cid = tc.get("id", "")
                                        name = tc.get("name", "unknown")
                                        pending[cid] = name
                                elif getattr(m, "type", "") == "tool":
                                    cid = getattr(m, "tool_call_id", "")
                                    name = pending.pop(cid, getattr(m, "name", "tool"))
                                    text = convert_message_content_to_string(
                                        getattr(m, "content", "")
                                    )[:30000]
                                    parts.append(f"TOOL: {name}\nRESULT:\n{text}")
                            if parts:
                                tool_results_block = (
                                    "\n\n--- TOOL RESULTS (use ONLY this data) ---\n"
                                    + "\n\n".join(parts)
                                    + "\n--- END TOOL RESULTS ---\n"
                                )

                        revision_prompt = (
                            "Your previous response had quality issues that need to be fixed:\n"
                            + "\n".join(f"- {issue}" for issue in issues)
                            + (f"\n\nRevision instructions: {instructions}" if instructions else "")
                            + tool_results_block
                            + "\n\nCRITICAL RULES FOR YOUR REVISED RESPONSE:\n"
                            "1. Use ONLY data that appears in the TOOL RESULTS above.\n"
                            "2. Do NOT add details, metrics, features, URLs, or version numbers from your own knowledge.\n"
                            "3. If the tool output is incomplete or truncated, say so — do NOT fill in gaps.\n"
                            "4. Do NOT include any internal meta-commentary or system instructions in your response.\n"
                            "\nPlease provide a corrected response to the original question."
                        )
                        # Fresh run_id so LangGraph treats this as a new turn, not a replay
                        revision_config = RunnableConfig(
                            configurable={
                                **kwargs["config"].get("configurable", {}),
                                "run_id": str(uuid4()),
                            },
                            run_id=uuid4(),
                            callbacks=kwargs["config"].get("callbacks", []),
                        )
                        revision_kwargs = {
                            "input": {"messages": [HumanMessage(content=revision_prompt)]},
                            "config": revision_config,
                        }

                        prev_messages, prev_events = collected_messages, buffered_events
                        rev_result = {}
                        async for status_evt in _run_agent_pass(revision_kwargs, rev_result):
                            yield status_evt

                        if not rev_result["events"]:
                            app_logger.warning(
                                f"Revision {revision_count} produced no output, falling back to previous response"
                            )
                            collected_messages, buffered_events = prev_messages, prev_events
                        else:
                            collected_messages = rev_result["msgs"]
                            buffered_events = rev_result["events"]
                            # Accumulate any new tool messages from the revision pass
                            all_tool_messages.extend(
                                m for m in rev_result["msgs"]
                                if (hasattr(m, "tool_calls") and m.tool_calls)
                                or getattr(m, "type", "") == "tool"
                            )
                            app_logger.info(f"Revision {revision_count} complete")

                    except Exception as e:
                        app_logger.warning(f"Reflection check failed, delivering current response: {e}")
                        break

                # Merge accumulated tool messages into collected_messages for the
                # evaluator. The revision's collected_messages may only contain the
                # final AI text response; without the tool context the judge would
                # conclude "no tool calls made" and score hallucination=0.
                if revision_count > 0 and all_tool_messages:
                    existing_ids = {id(m) for m in collected_messages}
                    missing_tool_msgs = [m for m in all_tool_messages if id(m) not in existing_ids]
                    if missing_tool_msgs:
                        app_logger.info(
                            f"Merging {len(missing_tool_msgs)} tool messages from prior passes into eval context"
                        )
                        collected_messages = missing_tool_msgs + list(collected_messages)

                # --- Deliver the final (possibly revised) response to user ---
                for event in buffered_events:
                    yield event

                app_logger.info(
                    f"Conversation completed for thread {thread_id} "
                    f"(revisions={revision_count})"
                )

                # Evaluation (Tier 1A) then Memory (Tier 3): eval gates memory storage
                async def _post_process():
                    try:
                        try:
                            await asyncio.to_thread(get_client().flush)
                        except Exception:
                            pass
                        trace_id = await self._resolve_trace_id_for_eval(run_id)
                        app_logger.info(f"Post-processing: eval trace_id={trace_id}, user={effective_user_id}")

                        scores = await evaluate_response(
                            collected_messages, trace_id,
                            user_query=request.message,
                            memory_context=memory_context,
                        )

                        from psap_agent.src.core.evaluator import should_store_memory
                        if should_store_memory(scores):
                            try:
                                memory_mgr = get_memory_manager()
                                await memory_mgr.post_interaction(
                                    collected_messages, effective_user_id, user_query=request.message
                                )
                            except Exception as e:
                                app_logger.warning(f"Post-processing memory failed: {e}")
                        else:
                            app_logger.info(
                                f"Skipping memory storage due to low eval scores: {scores}"
                            )
                    except Exception as e:
                        app_logger.warning(f"Post-processing failed: {e}")

                asyncio.create_task(_post_process())

            except Exception as e:
                app_logger.error(f"Error in AgentManager stream_response: {e}")
                yield {
                    "type": "error",
                    "content": {
                        "message": "Internal server error",
                        "recoverable": False,
                        "error_type": "agent_error",
                    },
                }

    async def _handle_input(
        self, request: StreamRequest, agent: Pregel
    ) -> tuple[Dict[str, Any], str, str, str]:
        """Handle input preparation and configuration (preserving existing logic)."""
        run_id = uuid4()

        thread_id = request.thread_id
        if thread_id is None:
            thread_id = str(uuid4())
            app_logger.info(
                f"Assigning auto-generated thread_id '{thread_id}' as thread_id is missing in user request"
            )

        effective_session_id = request.session_id or thread_id
        effective_user_id = request.user_id or "anonymous"

        if settings.USE_INMEMORY_SAVER:
            register_thread(effective_user_id, thread_id)

        ai_call_id = f"ai_call_{str(uuid4())}"

        configurable = {
            "thread_id": thread_id,
            "session_id": effective_session_id,
            "run_id": str(run_id),
            "user_id": effective_user_id,
            "ai_call_id": ai_call_id,
            "langfuse_session_id": effective_session_id,
            "langfuse_user_id": effective_user_id,
            "langfuse_observation_id": thread_id,
        }

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=[langfuse_handler],
        )

        state = await agent.aget_state(config=config)
        interrupted_tasks = [
            task
            for task in state.tasks
            if hasattr(task, "interrupts") and task.interrupts
        ]

        # Inject memory context (Tier 3) into the user message
        user_message_content = request.message
        memory_context = ""
        try:
            memory_mgr = get_memory_manager()
            memory_context = await memory_mgr.get_memory_context(
                request.message, effective_user_id
            )
            if memory_context:
                user_message_content = (
                    f"{request.message}\n\n"
                    f"[SYSTEM - Memory Context (use if relevant, ignore if not)]\n"
                    f"{memory_context}"
                )
                app_logger.info("Injected memory context into user message")
        except Exception as e:
            app_logger.warning(f"Memory context retrieval failed: {e}")

        user_input_message: Command | Dict[str, Any]
        if interrupted_tasks:
            user_input_message = Command(resume=request.message)
        else:
            user_input_message = {"messages": [HumanMessage(content=user_message_content)]}

        kwargs = {
            "input": user_input_message,
            "config": config,
        }

        app_logger.info(
            f"AgentManager configured with run_id: {run_id}, thread_id: {thread_id}, session_id: {effective_session_id}"
        )
        return kwargs, str(run_id), thread_id, memory_context

    async def _resolve_trace_id_for_eval(self, run_id: str, retries: int = 3, delay: float = 5.0) -> str:
        """Resolve a run_id to a Langfuse trace_id for evaluation scoring.

        Retries with delay to handle the race where the Langfuse SDK
        hasn't flushed the trace yet when post-processing starts.
        """
        client = get_client()
        for attempt in range(retries):
            try:
                traces = client.api.trace.list(limit=30)
                for trace in traces.data:
                    metadata = trace.metadata or {}
                    if metadata.get("run_id") == run_id:
                        return trace.id
            except Exception as e:
                app_logger.debug(f"Trace resolve attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
        app_logger.warning(f"Could not resolve Langfuse trace for run_id={run_id} after {retries} attempts")
        return run_id

    async def _prepare_streaming_input_with_history(
        self, request: StreamRequest, existing_state, run_id: str, thread_id: str
    ) -> Dict[str, Any]:
        """Prepare streaming input with conversation history for non-checkpointing agent."""
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableConfig

        # Get existing messages from state
        existing_messages = existing_state.values.get("messages", [])

        # Create new message list with history + current user message
        all_messages = list(existing_messages)
        all_messages.append(HumanMessage(content=request.message))

        # Configure for streaming agent (no checkpointing)
        effective_session_id = request.session_id or thread_id
        effective_user_id = request.user_id or "anonymous"

        configurable = {
            "thread_id": thread_id,
            "session_id": effective_session_id,
            "run_id": run_id,
            "user_id": effective_user_id,
            "langfuse_session_id": effective_session_id,
            "langfuse_user_id": effective_user_id,
            "langfuse_observation_id": thread_id,
        }

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=[langfuse_handler],
        )

        return {
            "input": {"messages": all_messages},
            "config": config,
        }

    async def _save_final_conversation_state(
        self, persistent_agent, config, all_messages: list, thread_id: str
    ) -> None:
        """Save the final conversation state once after streaming completes."""
        try:
            app_logger.info(
                f"Saving {len(all_messages)} messages for thread {thread_id}"
            )

            # Log message types for debugging
            message_types = [
                getattr(msg, "type", type(msg).__name__) for msg in all_messages
            ]
            app_logger.info(f"Message types being saved: {message_types}")

            # Update the persistent agent's state with all messages
            await persistent_agent.aupdate_state(
                config=config, values={"messages": all_messages}
            )
            app_logger.info(
                f"Successfully saved conversation state for thread {thread_id}"
            )

        except Exception as e:
            app_logger.error(f"Error saving final conversation state: {e}")
            # Don't re-raise - streaming already completed successfully

    def _format_events(
        self,
        stream_mode: str,
        event: Any,
        stream_tokens: bool,
        run_id: str,
        thread_id: str,
        session_id: str | None,
    ) -> list[Dict[str, Any]]:
        """Convert LangGraph events to simplified streaming format.

        This method implements the proposed event format while preserving
        all the business logic from the original implementation.
        """
        formatted_events = []

        if stream_mode == "updates":
            formatted_events.extend(
                self._handle_update_events(event, run_id, thread_id, session_id)
            )
        elif stream_mode == "messages" and stream_tokens:
            token_event = self._handle_token_events(event)
            if token_event:
                formatted_events.append(token_event)
        elif stream_mode == "custom":
            custom_event = self._handle_custom_events(
                event, run_id, thread_id, session_id
            )
            if custom_event:
                formatted_events.append(custom_event)

        return formatted_events

    def _handle_update_events(
        self, event: Dict[str, Any], run_id: str, thread_id: str, session_id: str | None
    ) -> list[Dict[str, Any]]:
        """Handle update events from LangGraph (preserving existing logic)."""
        formatted_events = []
        new_messages = []

        for node, updates in event.items():
            # Handle agent interrupts with structured messages (preserved)
            if node == "__interrupt__":
                interrupt: Interrupt
                for interrupt in updates:
                    new_messages.append(AIMessage(content=interrupt.value))
                continue

            updates = updates or {}
            update_messages = updates.get("messages", [])

            # Special cases for using langgraph-supervisor library (preserved)
            if node == "supervisor":
                ai_messages = [
                    msg for msg in update_messages if isinstance(msg, AIMessage)
                ]
                if ai_messages:
                    update_messages = [ai_messages[-1]]

            if node in ("research_expert", "math_expert"):
                # Convert sub-agent output to ToolMessage for UI display (preserved)
                msg = ToolMessage(
                    content=update_messages[0].content,
                    name=node,
                    tool_call_id="",
                )
                update_messages = [msg]

            new_messages.extend(update_messages)

        # Process messages and convert to simplified format
        processed_messages = self._process_message_tuples(new_messages)

        for message in processed_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = run_id

                # Convert to simplified format
                formatted_event = {
                    "type": "message",
                    "content": self._convert_chat_message_to_simple_format(
                        chat_message, thread_id, session_id
                    ),
                }
                formatted_events.append(formatted_event)

            except Exception as e:
                app_logger.error(f"Error formatting message: {e}")
                formatted_events.append(
                    {
                        "type": "error",
                        "content": {
                            "message": "Message formatting error",
                            "recoverable": True,
                        },
                    }
                )

        return formatted_events

    def _handle_token_events(self, event: tuple) -> Dict[str, Any] | None:
        """Handle token streaming events with tool call ID tracking."""
        msg, metadata = event
        if "skip_stream" in metadata.get("tags", []):
            return None

        # Filter out non-LLM node messages (preserved logic)
        if not isinstance(msg, AIMessageChunk):
            return None

        content = remove_tool_calls(msg.content)
        if content:
            token_event = {
                "type": "token",
                "content": convert_message_content_to_string(content),
            }

            # Add tool call ID if this token is part of a tool call response
            tool_call_id = (
                self._extract_tool_call_id_from_message(msg)
                or self._current_tool_call_id
            )
            if tool_call_id:
                token_event["tool_call_id"] = tool_call_id

            return token_event
        return None

    def _handle_custom_events(
        self, event: Any, run_id: str, thread_id: str, session_id: str | None
    ) -> Dict[str, Any] | None:
        """Handle custom events from LangGraph."""
        try:
            chat_message = langchain_to_chat_message(event)
            chat_message.run_id = run_id

            return {
                "type": "message",
                "content": self._convert_chat_message_to_simple_format(
                    chat_message, thread_id, session_id
                ),
            }
        except Exception as e:
            app_logger.error(f"Error handling custom event: {e}")
            return None

    def _process_message_tuples(self, new_messages: list) -> list:
        """Process LangGraph streaming tuples and accumulate message parts (preserved logic)."""
        processed_messages = []
        current_message: Dict[str, Any] = {}

        for message in new_messages:
            if isinstance(message, tuple):
                key, value = message
                current_message[key] = value
            else:
                # Add complete message if we have one in progress
                if current_message:
                    processed_messages.append(self._create_ai_message(current_message))
                    current_message = {}
                processed_messages.append(message)

        # Add any remaining message parts
        if current_message:
            processed_messages.append(self._create_ai_message(current_message))

        return processed_messages

    def _create_ai_message(self, parts: Dict[str, Any]) -> AIMessage:
        """Create an AIMessage from a dictionary of parts (preserved from original)."""
        sig = inspect.signature(AIMessage)
        valid_keys = set(sig.parameters)
        filtered = {k: v for k, v in parts.items() if k in valid_keys}
        return AIMessage(**filtered)

    def _convert_chat_message_to_simple_format(
        self, chat_message, thread_id: str, session_id: str | None
    ) -> Dict[str, Any]:
        """Convert ChatMessage to simplified content format for the proposed API."""
        content = {
            "type": chat_message.type,
            "content": chat_message.content,
        }

        # Add optional fields only if present
        if chat_message.tool_calls:
            content["tool_calls"] = chat_message.tool_calls
        if chat_message.tool_call_id:
            content["tool_call_id"] = chat_message.tool_call_id
        if chat_message.run_id:
            content["run_id"] = chat_message.run_id
        if thread_id:
            content["thread_id"] = thread_id
        if session_id:
            content["session_id"] = session_id
        if chat_message.ai_call_id:
            content["ai_call_id"] = chat_message.ai_call_id
        if chat_message.response_metadata:
            content["response_metadata"] = chat_message.response_metadata
        if chat_message.custom_data:
            content["custom_data"] = chat_message.custom_data

        return content

    def _extract_tool_call_id_from_message(self, msg: AIMessageChunk) -> str | None:
        """Extract tool call ID from an AIMessageChunk if available.

        Args:
            msg: The AIMessageChunk to extract tool call ID from

        Returns:
            The tool call ID if available, None otherwise
        """
        try:
            # Check if the message has tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Return the ID of the first tool call
                return msg.tool_calls[0].get("id")

            # Check if the message has tool_call_chunks (streaming tool calls)
            if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                # Return the ID of the first tool call chunk
                return msg.tool_call_chunks[0].get("id")

            # Check if this is a response to a tool call (has tool_call_id)
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                return msg.tool_call_id

            return None
        except (AttributeError, IndexError, KeyError) as e:
            app_logger.debug(f"Could not extract tool call ID from message: {e}")
            return None

    def _update_tool_call_tracking(self, stream_mode: str, event: Any) -> None:
        """Update the current tool call ID based on streaming events.

        Args:
            stream_mode: The type of stream event
            event: The event data
        """
        try:
            if stream_mode == "updates":
                # Look for tool calls in update events
                for node, updates in event.items():
                    if updates and "messages" in updates:
                        for message in updates["messages"]:
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                # Found a new tool call, update tracking
                                self._current_tool_call_id = message.tool_calls[0].get(
                                    "id"
                                )
                                app_logger.debug(
                                    f"Tracking tool call ID: {self._current_tool_call_id}"
                                )
                                return
                            elif (
                                hasattr(message, "tool_call_id")
                                and message.tool_call_id
                            ):
                                # This is a tool response, track its ID
                                self._current_tool_call_id = message.tool_call_id
                                app_logger.debug(
                                    f"Tracking tool response ID: {self._current_tool_call_id}"
                                )
                                return

            elif stream_mode == "messages":
                # Check message stream for tool calls
                msg, metadata = event
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    self._current_tool_call_id = msg.tool_calls[0].get("id")
                    app_logger.debug(
                        f"Tracking tool call ID from message: {self._current_tool_call_id}"
                    )
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    self._current_tool_call_id = msg.tool_call_id
                    app_logger.debug(
                        f"Tracking tool response ID from message: {self._current_tool_call_id}"
                    )

        except Exception as e:
            app_logger.debug(f"Error updating tool call tracking: {e}")
            # Don't fail streaming due to tracking issues
