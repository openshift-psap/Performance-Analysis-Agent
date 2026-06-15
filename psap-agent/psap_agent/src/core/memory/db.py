"""Database connection management for the PSAP agent memory layer.

Supports both PostgreSQL (production) and SQLite (local development).
Reuses the existing database_uri from settings for PostgreSQL connections.
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# ---------------------------------------------------------------------------
# DDL — 5-table schema
# ---------------------------------------------------------------------------

DDL_CONFIG_FINDINGS = """
CREATE TABLE IF NOT EXISTS config_findings (
    id              TEXT PRIMARY KEY,
    fact_text       TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'metrics_comparison',
    rhaiis_version  TEXT NOT NULL,
    baseline_version TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_family    TEXT,
    accelerator     TEXT NOT NULL,
    tp_config       INTEGER,
    profile         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    issue_thread_id TEXT,
    root_cause      TEXT,
    related_prs     TEXT,
    related_kernels TEXT,
    confidence      TEXT NOT NULL DEFAULT 'high',
    source          TEXT NOT NULL DEFAULT 'agent_analysis',
    run_id          TEXT,
    content_hash    TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cf_version ON config_findings (rhaiis_version);
CREATE INDEX IF NOT EXISTS idx_cf_model ON config_findings (model_name);
CREATE INDEX IF NOT EXISTS idx_cf_accel ON config_findings (accelerator);
CREATE INDEX IF NOT EXISTS idx_cf_dedup ON config_findings (rhaiis_version, model_name, accelerator, profile, category);
CREATE INDEX IF NOT EXISTS idx_cf_status ON config_findings (status);
CREATE INDEX IF NOT EXISTS idx_cf_thread ON config_findings (issue_thread_id);
"""

DDL_MODEL_SUMMARIES = """
CREATE TABLE IF NOT EXISTS model_summaries (
    id              TEXT PRIMARY KEY,
    summary_text    TEXT NOT NULL,
    rhaiis_version  TEXT NOT NULL,
    baseline_version TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_family    TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    issue_thread_id TEXT,
    confidence      TEXT NOT NULL DEFAULT 'high',
    source          TEXT NOT NULL DEFAULT 'agent_analysis',
    run_id          TEXT,
    content_hash    TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ms_version ON model_summaries (rhaiis_version);
CREATE INDEX IF NOT EXISTS idx_ms_dedup ON model_summaries (rhaiis_version, model_name);
"""

DDL_VERSION_SUMMARIES = """
CREATE TABLE IF NOT EXISTS version_summaries (
    id              TEXT PRIMARY KEY,
    summary_text    TEXT NOT NULL,
    rhaiis_version  TEXT NOT NULL,
    baseline_version TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    confidence      TEXT NOT NULL DEFAULT 'high',
    source          TEXT NOT NULL DEFAULT 'agent_analysis',
    run_id          TEXT,
    content_hash    TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_vs_version ON version_summaries (rhaiis_version);
"""

DDL_ANALYSIS_REPORTS = """
CREATE TABLE IF NOT EXISTS analysis_reports (
    id              TEXT PRIMARY KEY,
    analysis_level  TEXT NOT NULL,
    report_text     TEXT NOT NULL,
    rhaiis_version  TEXT NOT NULL,
    baseline_version TEXT NOT NULL,
    model_name      TEXT,
    accelerator     TEXT,
    tp_config       INTEGER,
    profile         TEXT,
    run_id          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ar_version ON analysis_reports (rhaiis_version);
CREATE INDEX IF NOT EXISTS idx_ar_level ON analysis_reports (analysis_level);
CREATE INDEX IF NOT EXISTS idx_ar_model ON analysis_reports (model_name);
"""

DDL_CONSOLIDATION_LOG = """
CREATE TABLE IF NOT EXISTS consolidation_log (
    id              TEXT PRIMARY KEY,
    rhaiis_version  TEXT NOT NULL,
    facts_pruned    INTEGER DEFAULT 0,
    facts_merged    INTEGER DEFAULT 0,
    red_flags_folded INTEGER DEFAULT 0,
    summary_text    TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cl_version ON consolidation_log (rhaiis_version);
"""

ALL_DDL_PG = [
    DDL_CONFIG_FINDINGS,
    DDL_MODEL_SUMMARIES,
    DDL_VERSION_SUMMARIES,
    DDL_ANALYSIS_REPORTS,
    DDL_CONSOLIDATION_LOG,
]


def _to_sqlite(ddl: str) -> str:
    return ddl.replace("TIMESTAMPTZ", "TEXT").replace("NOW()", "(datetime('now'))")


ALL_DDL_SQLITE = [_to_sqlite(d) for d in ALL_DDL_PG]

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_LOCAL_DB_PATH = Path("memory.db")
_sqlite_lock = threading.Lock()
_sqlite_conn: Optional[sqlite3.Connection] = None


def _use_postgres() -> bool:
    return bool(
        getattr(settings, "POSTGRES_HOST", None)
        and getattr(settings, "POSTGRES_DB", None)
        and settings.POSTGRES_HOST not in ("pgvector", "")
    )


def _get_sqlite() -> sqlite3.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        _sqlite_conn = sqlite3.connect(str(_LOCAL_DB_PATH), check_same_thread=False)
        _sqlite_conn.row_factory = sqlite3.Row
        for ddl in ALL_DDL_SQLITE:
            _sqlite_conn.executescript(ddl)
    return _sqlite_conn


async def get_connection():
    """Get an asyncpg connection to PostgreSQL."""
    import asyncpg

    return await asyncpg.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


async def ensure_schema() -> None:
    """Create all memory tables if they don't exist."""
    if _use_postgres():
        conn = await get_connection()
        try:
            for ddl in ALL_DDL_PG:
                await conn.execute(ddl)
            logger.info("5-table memory schema ensured in PostgreSQL")
        finally:
            await conn.close()
    else:
        with _sqlite_lock:
            _get_sqlite()
            logger.info(f"5-table memory schema ensured in SQLite ({_LOCAL_DB_PATH})")


def content_hash(text: str) -> str:
    """Compute MD5 hash for content-based deduplication."""
    return hashlib.md5(text.encode()).hexdigest()


def is_postgres() -> bool:
    """Public accessor for backend check."""
    return _use_postgres()


def get_sqlite_conn_and_lock() -> tuple[sqlite3.Connection, threading.Lock]:
    """Return (connection, lock) for callers that need transactional access.

    The caller is responsible for acquiring the lock before using the connection.
    The connection returned is lazily initialized on first call.
    """
    return _get_sqlite(), _sqlite_lock
