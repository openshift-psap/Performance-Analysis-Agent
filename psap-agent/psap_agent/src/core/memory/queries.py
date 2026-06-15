"""Query functions for the PSAP agent memory layer.

All functions support both PostgreSQL (production) and SQLite (local dev).
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from psap_agent.src.core.memory.db import (
    content_hash,
    get_connection,
    get_sqlite_conn_and_lock,
    is_postgres,
)
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


# ---------------------------------------------------------------------------
# Read queries
# ---------------------------------------------------------------------------


async def get_config_facts(
    rhaiis_version: str,
    model_name: Optional[str] = None,
    accelerator: Optional[str] = None,
    profile: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Retrieve config-level findings with optional filters."""
    filters = {
        "rhaiis_version": rhaiis_version,
        "model_name": model_name,
        "accelerator": accelerator,
        "profile": profile,
        "category": category,
        "status": status,
    }
    return await _query_table("config_findings", filters, limit)


async def get_model_summary(
    rhaiis_version: str,
    model_name: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Retrieve model-level summaries."""
    filters = {
        "rhaiis_version": rhaiis_version,
        "model_name": model_name,
    }
    return await _query_table("model_summaries", filters, limit)


async def get_version_summary(
    rhaiis_version: str,
    limit: int = 10,
) -> list[dict]:
    """Retrieve version-level summaries."""
    return await _query_table(
        "version_summaries", {"rhaiis_version": rhaiis_version}, limit
    )


# ---------------------------------------------------------------------------
# Upsert operations
# ---------------------------------------------------------------------------


async def upsert_config_finding(finding: dict) -> dict:
    """Insert or update a config-level finding.

    Dedup is based on content_hash: identical text for the same
    (rhaiis_version, model_name, accelerator, profile) is a noop.
    Different text always inserts a new row, even under the same category.
    Semantic dedup is handled upstream by the LLM duplicate-check gate.
    """
    text = finding.get("fact_text", finding.get("text", ""))
    if not text:
        return {"id": None, "action": "skipped", "reason": "empty text"}

    h = content_hash(text)
    now = datetime.now(timezone.utc)
    dedup_key = {
        "rhaiis_version": finding["rhaiis_version"],
        "model_name": finding["model_name"],
        "accelerator": finding["accelerator"],
        "profile": finding["profile"],
    }

    if is_postgres():
        return await _upsert_config_finding_pg(finding, text, h, now, dedup_key)
    return _upsert_config_finding_sqlite(finding, text, h, now, dedup_key)


async def upsert_model_summary(summary: dict) -> dict:
    """Insert or update a model-level summary with dedup on
    (rhaiis_version, model_name)."""
    text = summary.get("summary_text", "")
    if not text:
        return {"id": None, "action": "skipped", "reason": "empty text"}

    h = content_hash(text)
    now = datetime.now(timezone.utc)

    if is_postgres():
        conn = await get_connection()
        try:
            existing = await conn.fetchrow(
                """SELECT id, content_hash FROM model_summaries
                WHERE rhaiis_version=$1 AND model_name=$2 LIMIT 1""",
                summary["rhaiis_version"],
                summary["model_name"],
            )
            if existing:
                if existing["content_hash"] == h:
                    return {"id": existing["id"], "action": "noop"}
                await conn.execute(
                    """UPDATE model_summaries SET summary_text=$1, confidence=$2,
                    content_hash=$3, updated_at=$4, run_id=$5, baseline_version=$6,
                    model_family=$7, status=$8, issue_thread_id=$9
                    WHERE id=$10""",
                    text,
                    summary.get("confidence", "high"),
                    h,
                    now,
                    summary.get("run_id"),
                    summary["baseline_version"],
                    summary.get("model_family"),
                    summary.get("status", "active"),
                    summary.get("issue_thread_id"),
                    existing["id"],
                )
                return {"id": existing["id"], "action": "updated"}
            sid = str(uuid.uuid4())
            await conn.execute(
                """INSERT INTO model_summaries (id, summary_text, rhaiis_version,
                baseline_version, model_name, model_family, status, issue_thread_id,
                confidence, source, run_id, content_hash, created_at, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
                sid,
                text,
                summary["rhaiis_version"],
                summary["baseline_version"],
                summary["model_name"],
                summary.get("model_family"),
                summary.get("status", "active"),
                summary.get("issue_thread_id"),
                summary.get("confidence", "high"),
                summary.get("source", "agent_analysis"),
                summary.get("run_id"),
                h,
                now,
                now,
            )
            return {"id": sid, "action": "inserted"}
        finally:
            await conn.close()
    else:
        now_s = now.isoformat()
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            row = c.execute(
                """SELECT id, content_hash FROM model_summaries
                WHERE rhaiis_version=? AND model_name=? LIMIT 1""",
                (summary["rhaiis_version"], summary["model_name"]),
            ).fetchone()
            if row:
                if row["content_hash"] == h:
                    return {"id": row["id"], "action": "noop"}
                c.execute(
                    """UPDATE model_summaries SET summary_text=?, confidence=?,
                    content_hash=?, updated_at=?, run_id=?, baseline_version=?,
                    model_family=?, status=?, issue_thread_id=? WHERE id=?""",
                    (
                        text,
                        summary.get("confidence", "high"),
                        h,
                        now_s,
                        summary.get("run_id"),
                        summary["baseline_version"],
                        summary.get("model_family"),
                        summary.get("status", "active"),
                        summary.get("issue_thread_id"),
                        row["id"],
                    ),
                )
                c.commit()
                return {"id": row["id"], "action": "updated"}
            sid = str(uuid.uuid4())
            c.execute(
                """INSERT INTO model_summaries (id, summary_text, rhaiis_version,
                baseline_version, model_name, model_family, status, issue_thread_id,
                confidence, source, run_id, content_hash, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sid,
                    text,
                    summary["rhaiis_version"],
                    summary["baseline_version"],
                    summary["model_name"],
                    summary.get("model_family"),
                    summary.get("status", "active"),
                    summary.get("issue_thread_id"),
                    summary.get("confidence", "high"),
                    summary.get("source", "agent_analysis"),
                    summary.get("run_id"),
                    h,
                    now_s,
                    now_s,
                ),
            )
            c.commit()
            return {"id": sid, "action": "inserted"}


async def upsert_version_summary(summary: dict) -> dict:
    """Insert or update a version-level summary with dedup on (rhaiis_version)."""
    text = summary.get("summary_text", "")
    if not text:
        return {"id": None, "action": "skipped", "reason": "empty text"}

    h = content_hash(text)
    now = datetime.now(timezone.utc)

    if is_postgres():
        conn = await get_connection()
        try:
            existing = await conn.fetchrow(
                """SELECT id, content_hash FROM version_summaries
                WHERE rhaiis_version=$1 LIMIT 1""",
                summary["rhaiis_version"],
            )
            if existing:
                if existing["content_hash"] == h:
                    return {"id": existing["id"], "action": "noop"}
                await conn.execute(
                    """UPDATE version_summaries SET summary_text=$1, confidence=$2,
                    content_hash=$3, updated_at=$4, run_id=$5, baseline_version=$6,
                    status=$7
                    WHERE id=$8""",
                    text,
                    summary.get("confidence", "high"),
                    h,
                    now,
                    summary.get("run_id"),
                    summary["baseline_version"],
                    summary.get("status", "active"),
                    existing["id"],
                )
                return {"id": existing["id"], "action": "updated"}
            sid = str(uuid.uuid4())
            await conn.execute(
                """INSERT INTO version_summaries (id, summary_text, rhaiis_version,
                baseline_version, status, confidence, source, run_id,
                content_hash, created_at, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                sid,
                text,
                summary["rhaiis_version"],
                summary["baseline_version"],
                summary.get("status", "active"),
                summary.get("confidence", "high"),
                summary.get("source", "agent_analysis"),
                summary.get("run_id"),
                h,
                now,
                now,
            )
            return {"id": sid, "action": "inserted"}
        finally:
            await conn.close()
    else:
        now_s = now.isoformat()
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            row = c.execute(
                """SELECT id, content_hash FROM version_summaries
                WHERE rhaiis_version=? LIMIT 1""",
                (summary["rhaiis_version"],),
            ).fetchone()
            if row:
                if row["content_hash"] == h:
                    return {"id": row["id"], "action": "noop"}
                c.execute(
                    """UPDATE version_summaries SET summary_text=?, confidence=?,
                    content_hash=?, updated_at=?, run_id=?, baseline_version=?,
                    status=? WHERE id=?""",
                    (
                        text,
                        summary.get("confidence", "high"),
                        h,
                        now_s,
                        summary.get("run_id"),
                        summary["baseline_version"],
                        summary.get("status", "active"),
                        row["id"],
                    ),
                )
                c.commit()
                return {"id": row["id"], "action": "updated"}
            sid = str(uuid.uuid4())
            c.execute(
                """INSERT INTO version_summaries (id, summary_text, rhaiis_version,
                baseline_version, status, confidence, source, run_id,
                content_hash, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sid,
                    text,
                    summary["rhaiis_version"],
                    summary["baseline_version"],
                    summary.get("status", "active"),
                    summary.get("confidence", "high"),
                    summary.get("source", "agent_analysis"),
                    summary.get("run_id"),
                    h,
                    now_s,
                    now_s,
                ),
            )
            c.commit()
            return {"id": sid, "action": "inserted"}


async def store_report(report: dict) -> dict:
    """Store a full analysis report."""
    text = report.get("report_text", "")
    if not text:
        return {"id": None, "action": "skipped", "reason": "empty text"}

    rid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    if is_postgres():
        conn = await get_connection()
        try:
            await conn.execute(
                """INSERT INTO analysis_reports (id, analysis_level, report_text,
                rhaiis_version, baseline_version, model_name, accelerator,
                tp_config, profile, run_id, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                rid,
                report["analysis_level"],
                text,
                report["rhaiis_version"],
                report["baseline_version"],
                report.get("model_name"),
                report.get("accelerator"),
                report.get("tp_config"),
                report.get("profile"),
                report.get("run_id"),
                now,
            )
            return {"id": rid, "action": "inserted"}
        finally:
            await conn.close()
    else:
        now_s = now.isoformat()
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            c.execute(
                """INSERT INTO analysis_reports (id, analysis_level, report_text,
                rhaiis_version, baseline_version, model_name, accelerator,
                tp_config, profile, run_id, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rid,
                    report["analysis_level"],
                    text,
                    report["rhaiis_version"],
                    report["baseline_version"],
                    report.get("model_name"),
                    report.get("accelerator"),
                    report.get("tp_config"),
                    report.get("profile"),
                    report.get("run_id"),
                    now_s,
                ),
            )
            c.commit()
            return {"id": rid, "action": "inserted"}


# ---------------------------------------------------------------------------
# Bulk / status update helpers
# ---------------------------------------------------------------------------


async def update_fact_status(
    fact_id: str, new_status: str, table: str = "config_findings"
) -> bool:
    """Update the status of a fact (active -> resolved -> stale)."""
    now = datetime.now(timezone.utc)
    if is_postgres():
        conn = await get_connection()
        try:
            result = await conn.execute(
                f"UPDATE {table} SET status=$1, updated_at=$2 WHERE id=$3",
                new_status,
                now,
                fact_id,
            )
            return result == "UPDATE 1"
        finally:
            await conn.close()
    else:
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            cur = c.execute(
                f"UPDATE {table} SET status=?, updated_at=? WHERE id=?",
                (new_status, now.isoformat(), fact_id),
            )
            c.commit()
            return cur.rowcount > 0


async def delete_stale_facts(table: str = "config_findings") -> int:
    """Delete facts marked as stale. Returns count of deleted rows."""
    if is_postgres():
        conn = await get_connection()
        try:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE status='stale'"
            )
            count = int(result.split()[-1]) if result else 0
            return count
        finally:
            await conn.close()
    else:
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            cur = c.execute(f"DELETE FROM {table} WHERE status='stale'")
            count = cur.rowcount
            c.commit()
            return count


# ---------------------------------------------------------------------------
# Generic query helper
# ---------------------------------------------------------------------------


async def _query_table(
    table: str, filters: dict[str, Optional[str]], limit: int
) -> list[dict]:
    active_filters = {k: v for k, v in filters.items() if v is not None}

    if is_postgres():
        conditions = []
        params: list[Any] = []
        idx = 1
        for col, val in active_filters.items():
            if col == "status":
                conditions.append(f"{col} = ${idx}")
                params.append(val)
            else:
                conditions.append(f"{col} = ${idx}")
                params.append(val)
            idx += 1
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM {table} {where} ORDER BY updated_at DESC LIMIT ${idx}"
        params.append(limit)

        conn = await get_connection()
        try:
            rows = await conn.fetch(query, *params)
            return [dict(r) for r in rows]
        finally:
            await conn.close()
    else:
        conditions = []
        params_list: list[Any] = []
        for col, val in active_filters.items():
            conditions.append(f"{col} = ?")
            params_list.append(val)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM {table} {where} ORDER BY updated_at DESC LIMIT ?"
        params_list.append(limit)
        c, lock = get_sqlite_conn_and_lock()
        with lock:
            cursor = c.execute(query, params_list)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Private helpers for upsert_config_finding (split by backend)
# ---------------------------------------------------------------------------


async def _upsert_config_finding_pg(
    finding: dict, text: str, h: str, now: datetime, dedup_key: dict
) -> dict:
    conn = await get_connection()
    try:
        existing = await conn.fetchrow(
            """SELECT id FROM config_findings
            WHERE rhaiis_version=$1 AND model_name=$2 AND accelerator=$3
              AND profile=$4 AND content_hash=$5 LIMIT 1""",
            dedup_key["rhaiis_version"],
            dedup_key["model_name"],
            dedup_key["accelerator"],
            dedup_key["profile"],
            h,
        )
        if existing:
            return {"id": existing["id"], "action": "noop"}

        fid = str(uuid.uuid4())
        await conn.execute(
            """INSERT INTO config_findings (id, fact_text, category, rhaiis_version,
            baseline_version, model_name, model_family, accelerator, tp_config, profile,
            status, issue_thread_id,
            root_cause, related_prs, related_kernels, confidence, source, run_id,
            content_hash, created_at, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)""",
            fid,
            text,
            finding.get("category", "metrics_comparison"),
            dedup_key["rhaiis_version"],
            finding.get("baseline_version", ""),
            dedup_key["model_name"],
            finding.get("model_family"),
            dedup_key["accelerator"],
            finding.get("tp_config"),
            dedup_key["profile"],
            finding.get("status", "active"),
            finding.get("issue_thread_id"),
            finding.get("root_cause"),
            finding.get("related_prs"),
            finding.get("related_kernels"),
            finding.get("confidence", "high"),
            finding.get("source", "agent_analysis"),
            finding.get("run_id"),
            h,
            now,
            now,
        )
        return {"id": fid, "action": "inserted"}
    finally:
        await conn.close()


def _upsert_config_finding_sqlite(
    finding: dict, text: str, h: str, now: datetime, dedup_key: dict
) -> dict:
    now_s = now.isoformat()
    c, lock = get_sqlite_conn_and_lock()
    with lock:
        row = c.execute(
            """SELECT id FROM config_findings
            WHERE rhaiis_version=? AND model_name=? AND accelerator=?
              AND profile=? AND content_hash=? LIMIT 1""",
            (
                dedup_key["rhaiis_version"],
                dedup_key["model_name"],
                dedup_key["accelerator"],
                dedup_key["profile"],
                h,
            ),
        ).fetchone()
        if row:
            return {"id": row["id"], "action": "noop"}

        fid = str(uuid.uuid4())
        c.execute(
            """INSERT INTO config_findings (id, fact_text, category, rhaiis_version,
            baseline_version, model_name, model_family, accelerator, tp_config, profile,
            status, issue_thread_id,
            root_cause, related_prs, related_kernels, confidence, source, run_id,
            content_hash, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fid,
                text,
                finding.get("category", "metrics_comparison"),
                dedup_key["rhaiis_version"],
                finding.get("baseline_version", ""),
                dedup_key["model_name"],
                finding.get("model_family"),
                dedup_key["accelerator"],
                finding.get("tp_config"),
                dedup_key["profile"],
                finding.get("status", "active"),
                finding.get("issue_thread_id"),
                finding.get("root_cause"),
                finding.get("related_prs"),
                finding.get("related_kernels"),
                finding.get("confidence", "high"),
                finding.get("source", "agent_analysis"),
                finding.get("run_id"),
                h,
                now_s,
                now_s,
            ),
        )
        c.commit()
        return {"id": fid, "action": "inserted"}
