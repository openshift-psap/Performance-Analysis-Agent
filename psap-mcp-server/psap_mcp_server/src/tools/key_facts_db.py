"""Database layer for performance memory — 5-table design (MCP server copy).

Identical schema to the agent-side memory/. This copy exists because the
MCP server has its own settings module and runs in a separate process.

Tables: config_findings, model_summaries, version_summaries,
        analysis_reports, consolidation_log.
"""

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from psap_mcp_server.src.settings import settings
from psap_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL_CONFIG_FINDINGS_PG = """
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

DDL_MODEL_SUMMARIES_PG = """
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

DDL_VERSION_SUMMARIES_PG = """
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

DDL_ANALYSIS_REPORTS_PG = """
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

DDL_CONSOLIDATION_LOG_PG = """
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


def _to_sqlite(ddl: str) -> str:
    return ddl.replace("TIMESTAMPTZ", "TEXT").replace("NOW()", "(datetime('now'))")


DDL_CONFIG_FINDINGS_SQLITE = _to_sqlite(DDL_CONFIG_FINDINGS_PG)
DDL_MODEL_SUMMARIES_SQLITE = _to_sqlite(DDL_MODEL_SUMMARIES_PG)
DDL_VERSION_SUMMARIES_SQLITE = _to_sqlite(DDL_VERSION_SUMMARIES_PG)
DDL_ANALYSIS_REPORTS_SQLITE = _to_sqlite(DDL_ANALYSIS_REPORTS_PG)
DDL_CONSOLIDATION_LOG_SQLITE = _to_sqlite(DDL_CONSOLIDATION_LOG_PG)

_LOCAL_DB_PATH = Path("key_facts.db")
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
        _sqlite_conn.executescript(DDL_CONFIG_FINDINGS_SQLITE)
        _sqlite_conn.executescript(DDL_MODEL_SUMMARIES_SQLITE)
        _sqlite_conn.executescript(DDL_VERSION_SUMMARIES_SQLITE)
        _sqlite_conn.executescript(DDL_ANALYSIS_REPORTS_SQLITE)
        _sqlite_conn.executescript(DDL_CONSOLIDATION_LOG_SQLITE)
    return _sqlite_conn


async def _pg_connect():
    import asyncpg
    return await asyncpg.connect(
        host=settings.POSTGRES_HOST, port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB, user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


async def ensure_schema() -> None:
    if _use_postgres():
        conn = await _pg_connect()
        try:
            await conn.execute(DDL_CONFIG_FINDINGS_PG)
            await conn.execute(DDL_MODEL_SUMMARIES_PG)
            await conn.execute(DDL_VERSION_SUMMARIES_PG)
            await conn.execute(DDL_ANALYSIS_REPORTS_PG)
            await conn.execute(DDL_CONSOLIDATION_LOG_PG)
            logger.info("5-table schema ensured in PostgreSQL")
        finally:
            await conn.close()
    else:
        with _sqlite_lock:
            _get_sqlite()
            logger.info(f"5-table schema ensured in SQLite ({_LOCAL_DB_PATH})")


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def upsert_config_finding(finding: dict) -> dict:
    """Insert or update a config-level finding."""
    import uuid
    text = finding.get("fact_text", finding.get("text", ""))
    if not text:
        return {"id": None, "action": "skipped", "reason": "empty text"}

    h = _content_hash(text)
    now = datetime.now(timezone.utc)

    if _use_postgres():
        conn = await _pg_connect()
        try:
            existing = await conn.fetchrow(
                """SELECT id, content_hash FROM config_findings
                WHERE rhaiis_version=$1 AND model_name=$2 AND accelerator=$3
                  AND profile=$4 AND category=$5 LIMIT 1""",
                finding["rhaiis_version"], finding["model_name"],
                finding["accelerator"], finding["profile"],
                finding.get("category", "metrics_comparison"),
            )
            if existing:
                if existing["content_hash"] == h:
                    return {"id": existing["id"], "action": "noop"}
                await conn.execute(
                    """UPDATE config_findings SET fact_text=$1, root_cause=$2, related_prs=$3,
                    related_kernels=$4, confidence=$5, content_hash=$6, updated_at=$7,
                    run_id=$8, baseline_version=$9, model_family=$10, tp_config=$11,
                    status=$12, issue_thread_id=$13
                    WHERE id=$14""",
                    text, finding.get("root_cause"), finding.get("related_prs"),
                    finding.get("related_kernels"), finding.get("confidence", "high"),
                    h, now, finding.get("run_id"), finding["baseline_version"],
                    finding.get("model_family"), finding.get("tp_config"),
                    finding.get("status", "active"), finding.get("issue_thread_id"),
                    existing["id"],
                )
                return {"id": existing["id"], "action": "updated"}

            fid = str(uuid.uuid4())
            await conn.execute(
                """INSERT INTO config_findings (id, fact_text, category, rhaiis_version,
                baseline_version, model_name, model_family, accelerator, tp_config, profile,
                status, issue_thread_id,
                root_cause, related_prs, related_kernels, confidence, source, run_id,
                content_hash, created_at, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)""",
                fid, text, finding.get("category", "metrics_comparison"),
                finding["rhaiis_version"], finding["baseline_version"],
                finding["model_name"], finding.get("model_family"),
                finding["accelerator"], finding.get("tp_config"), finding["profile"],
                finding.get("status", "active"), finding.get("issue_thread_id"),
                finding.get("root_cause"), finding.get("related_prs"),
                finding.get("related_kernels"), finding.get("confidence", "high"),
                finding.get("source", "agent_analysis"), finding.get("run_id"),
                h, now, now,
            )
            return {"id": fid, "action": "inserted"}
        finally:
            await conn.close()
    else:
        now_s = now.isoformat()
        fid = str(uuid.uuid4())
        with _sqlite_lock:
            c = _get_sqlite()
            row = c.execute(
                """SELECT id, content_hash FROM config_findings
                WHERE rhaiis_version=? AND model_name=? AND accelerator=?
                  AND profile=? AND category=? LIMIT 1""",
                (finding["rhaiis_version"], finding["model_name"],
                 finding["accelerator"], finding["profile"],
                 finding.get("category", "metrics_comparison")),
            ).fetchone()
            if row:
                if row["content_hash"] == h:
                    return {"id": row["id"], "action": "noop"}
                c.execute(
                    """UPDATE config_findings SET fact_text=?, root_cause=?, related_prs=?,
                    related_kernels=?, confidence=?, content_hash=?, updated_at=?,
                    run_id=?, baseline_version=?, model_family=?, tp_config=?,
                    status=?, issue_thread_id=? WHERE id=?""",
                    (text, finding.get("root_cause"), finding.get("related_prs"),
                     finding.get("related_kernels"), finding.get("confidence", "high"),
                     h, now_s, finding.get("run_id"), finding["baseline_version"],
                     finding.get("model_family"), finding.get("tp_config"),
                     finding.get("status", "active"), finding.get("issue_thread_id"),
                     row["id"]),
                )
                c.commit()
                return {"id": row["id"], "action": "updated"}
            c.execute(
                """INSERT INTO config_findings (id, fact_text, category, rhaiis_version,
                baseline_version, model_name, model_family, accelerator, tp_config, profile,
                status, issue_thread_id,
                root_cause, related_prs, related_kernels, confidence, source, run_id,
                content_hash, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (fid, text, finding.get("category", "metrics_comparison"),
                 finding["rhaiis_version"], finding["baseline_version"],
                 finding["model_name"], finding.get("model_family"),
                 finding["accelerator"], finding.get("tp_config"), finding["profile"],
                 finding.get("status", "active"), finding.get("issue_thread_id"),
                 finding.get("root_cause"), finding.get("related_prs"),
                 finding.get("related_kernels"), finding.get("confidence", "high"),
                 finding.get("source", "agent_analysis"), finding.get("run_id"),
                 h, now_s, now_s),
            )
            c.commit()
            return {"id": fid, "action": "inserted"}


async def query_config_findings(
    rhaiis_version: Optional[str] = None, model_name: Optional[str] = None,
    accelerator: Optional[str] = None, profile: Optional[str] = None,
    category: Optional[str] = None, limit: int = 50,
) -> list[dict]:
    return await _query_table("config_findings", {
        "rhaiis_version": rhaiis_version, "model_name": model_name,
        "accelerator": accelerator, "profile": profile, "category": category,
    }, limit)


async def query_model_summaries(
    rhaiis_version: Optional[str] = None, model_name: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    return await _query_table("model_summaries", {
        "rhaiis_version": rhaiis_version, "model_name": model_name,
    }, limit)


async def query_version_summaries(
    rhaiis_version: Optional[str] = None, limit: int = 10,
) -> list[dict]:
    return await _query_table("version_summaries", {
        "rhaiis_version": rhaiis_version,
    }, limit)


async def _query_table(
    table: str, filters: dict[str, Optional[str]], limit: int
) -> list[dict]:
    active_filters = {k: v for k, v in filters.items() if v is not None}

    if _use_postgres():
        conditions = []
        params: list[Any] = []
        idx = 1
        for col, val in active_filters.items():
            conditions.append(f"{col} ILIKE ${idx}")
            params.append(f"%{val}%")
            idx += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM {table} {where} ORDER BY updated_at DESC LIMIT ${idx}"
        params.append(limit)

        conn = await _pg_connect()
        try:
            rows = await conn.fetch(query, *params)
            return [dict(r) for r in rows]
        finally:
            await conn.close()
    else:
        conditions = []
        params_list: list[Any] = []
        for col, val in active_filters.items():
            conditions.append(f"{col} LIKE ?")
            params_list.append(f"%{val}%")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM {table} {where} ORDER BY updated_at DESC LIMIT ?"
        params_list.append(limit)
        with _sqlite_lock:
            c = _get_sqlite()
            cursor = c.execute(query, params_list)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
