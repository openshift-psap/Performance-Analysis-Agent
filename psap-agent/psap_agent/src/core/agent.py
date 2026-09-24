"""Agent implementation for the PSAP agent system.

This module provides the core agent functionality for the PSAP agent,
including initialization, configuration, and agent creation utilities.
Integrates reflection (Tier 2) and memory (Tier 3) when enabled.
"""

from contextlib import asynccontextmanager
from typing import Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.postgres import AsyncPostgresStore

from psap_agent.src.core.cache_manager import get_cache_manager
from psap_agent.src.core.curated_skills import create_load_skill_tool
from psap_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from psap_agent.src.core.memory import MemoryManager
from psap_agent.src.core.model_factory import create_chat_model, get_model_provider, is_gemini_model
from psap_agent.src.core.prompt import get_system_prompt
from psap_agent.src.core.storage import get_global_checkpoint
from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

# Each LangGraph "step" is one node invocation (LLM call or tool execution).
# A single tool-use round-trip = 2 steps (LLM decides + tool runs).
# 50 steps ≈ 25 tool calls max, which is generous for any legitimate query.
_RECURSION_LIMIT = 50

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(store=None) -> MemoryManager:
    """Get or create the global MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(store=store)
    return _memory_manager


@asynccontextmanager
async def get_psap_agent(
    sso_token: Optional[str] = None,
    enable_checkpointing: bool = True,
    model_name: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
):
    """Get a fully initialized PSAP agent.

    This function creates and configures a PSAP agent with the necessary
    tools, model, and database connections. It uses an async context manager
    to ensure proper resource cleanup.

    Args:
        sso_token: Optional access token for authentication. If provided,
            it will be used for authorization headers in MCP client requests.
        enable_checkpointing: Whether to enable checkpointing/persistence.
            Set to False for streaming-only operations that shouldn't save to DB.
        model_name: Optional model name override. Falls back to settings.GEMINI_MODEL.
        reasoning_effort: Optional OpenAI reasoning effort override. Ignored by
            Gemini and Claude models.

    Yields:
        The initialized PSAP agent instance.

    Raises:
        Exception: If there are issues with database connections or agent setup.
    """
    # Initialize MCP client and get tools (optional for local development)
    tools = []
    try:
        client = MultiServerMCPClient(
            {
                "psap-mcp-server": {
                    "url": settings.MCP_SERVER_URL,
                    "transport": "streamable_http",
                    "headers": {"Authorization": f"Bearer {sso_token}"}
                    if sso_token
                    else {},
                },
            }
        )
        tools = await client.get_tools()
        logger.info(
            f"Successfully connected to MCP server and loaded {len(tools)} tools"
        )
    except Exception as e:
        if settings.USE_INMEMORY_SAVER:
            logger.warning(f"Could not connect to MCP server: {e}")
            logger.info("Running in local development mode without MCP tools")
            tools = []  # No tools for local development
        else:
            logger.error(f"Failed to connect to MCP server in production mode: {e}")
            raise AppException(
                "Failed to connect to MCP server in production mode",
                AppExceptionCode.PRODUCTION_MCP_CONNECTION_ERROR,
            )

    tools.append(create_load_skill_tool())
    logger.info("Curated workflow skill loader enabled")

    # Resolve which model to use (client override vs server default)
    effective_model = model_name or settings.GEMINI_MODEL
    logger.info(f"🤖 Using model: {effective_model}")
    if reasoning_effort:
        logger.info(f"🧠 OpenAI reasoning effort: {reasoning_effort}")

    # Initialize the language model
    provider = get_model_provider(effective_model)
    if is_gemini_model(effective_model) and settings.ENABLE_PROMPT_CACHING and not tools:
        # Gemini-specific: cached_content not supported with tools/system_instruction
        try:
            cache_manager = get_cache_manager()
            cache = await cache_manager.create_or_get_cache(ttl_hours=settings.CACHE_TTL_HOURS)
            cache_name = cache.name
            cache_stats = cache_manager.get_cache_stats()
            logger.info(f"✅ Caching enabled: {cache_stats.get('token_count', 0)} tokens cached")
            logger.info(f"💰 Cache expires in: {cache_stats.get('time_remaining_human', 'unknown')}")
            
            model = create_chat_model(
                effective_model,
                temperature=0.3,
                max_output_tokens=16384,
                cached_content=cache_name,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize caching: {e}. Falling back to non-cached mode.")
            model = create_chat_model(
                effective_model,
                temperature=0.3,
                max_output_tokens=16384,
                reasoning_effort=reasoning_effort,
            )
    else:
        if provider == "openai":
            logger.info("Using OpenAI model; Gemini prompt caching is unavailable")
        elif provider == "claude":
            logger.info("Using Claude model via Vertex AI")
        elif tools:
            logger.info("Caching disabled: Gemini API doesn't support cached_content with tools")
        else:
            logger.info("Caching disabled by configuration")
        model = create_chat_model(
            effective_model,
            temperature=0.3,
            max_output_tokens=16384,
            reasoning_effort=reasoning_effort,
        )

    if not enable_checkpointing:
        logger.info(
            "Creating agent without checkpointing for streaming-only operations"
        )
        get_memory_manager(store=None)
        agent_redhat = create_react_agent(
            model=model,
            prompt=get_system_prompt(),
            tools=tools,
        ).with_config(recursion_limit=_RECURSION_LIMIT)
        logger.info("PSAP agent initialized successfully without checkpointing")
        yield agent_redhat
    elif settings.USE_INMEMORY_SAVER:
        logger.info("Using single global checkpoint for local development")
        from langgraph.store.memory import InMemoryStore
        checkpoint = get_global_checkpoint()
        skill_store = InMemoryStore()
        get_memory_manager(store=skill_store)
        agent_redhat = create_react_agent(
            model=model,
            prompt=get_system_prompt(),
            tools=tools,
            checkpointer=checkpoint,
            store=skill_store,
        ).with_config(recursion_limit=_RECURSION_LIMIT)
        logger.info(
            "PSAP agent initialized successfully with single global checkpoint"
        )
        yield agent_redhat
    else:
        logger.info("Using PostgreSQL checkpoint for production")
        async with AsyncPostgresSaver.from_conn_string(
            settings.database_uri
        ) as checkpoint:
            if hasattr(checkpoint, "setup"):
                await checkpoint.setup()

            async with AsyncPostgresStore.from_conn_string(
                settings.database_uri
            ) as skill_store:
                await skill_store.setup()

                get_memory_manager(store=skill_store)
                agent_redhat = create_react_agent(
                    model=model,
                    prompt=get_system_prompt(),
                    tools=tools,
                    checkpointer=checkpoint,
                    store=skill_store,
                ).with_config(recursion_limit=_RECURSION_LIMIT)

                logger.info(
                    "PSAP agent initialized successfully with PostgreSQL checkpoint and store"
                )
                yield agent_redhat
