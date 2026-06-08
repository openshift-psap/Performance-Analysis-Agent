"""Memory layer for the PSAP agent.

Provides persistent memory across analysis runs via PostgreSQL/SQLite,
including fact storage, memory injection, and red flags working memory.
"""

from psap_agent.src.core.memory.db import get_connection, ensure_schema, is_postgres
from psap_agent.src.core.memory.queries import (
    get_config_facts,
    get_model_summary,
    get_version_summary,
    upsert_config_finding,
    upsert_model_summary,
    upsert_version_summary,
    store_report,
)
from psap_agent.src.core.memory.injection import build_memory_context
from psap_agent.src.core.memory.red_flags import RedFlagsManager

__all__ = [
    "get_connection",
    "ensure_schema",
    "get_config_facts",
    "get_model_summary",
    "get_version_summary",
    "upsert_config_finding",
    "upsert_model_summary",
    "upsert_version_summary",
    "store_report",
    "build_memory_context",
    "RedFlagsManager",
]
