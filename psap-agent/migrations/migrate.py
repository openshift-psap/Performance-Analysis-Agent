#!/usr/bin/env python3
"""Run database migrations for the PSAP agent memory layer.

Usage:
    python migrations/migrate.py --backend postgres   # production
    python migrations/migrate.py --backend sqlite     # local dev
    python migrations/migrate.py                      # auto-detect from settings
"""

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MIGRATION_DIR = Path(__file__).parent


async def run_postgres_migration():
    """Apply the PostgreSQL migration."""
    from psap_agent.src.settings import settings

    import asyncpg

    sql_path = MIGRATION_DIR / "001_memory_layer_phase1.sql"
    sql = sql_path.read_text()

    conn = await asyncpg.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    try:
        await conn.execute(sql)
        print("PostgreSQL migration applied successfully.")
    finally:
        await conn.close()


def run_sqlite_migration(db_path: str = "memory.db"):
    """Apply the SQLite migration (tolerates 'duplicate column' errors)."""
    sql_path = MIGRATION_DIR / "001_memory_layer_phase1_sqlite.sql"
    statements = sql_path.read_text().split(";")

    conn = sqlite3.connect(db_path)
    applied = 0
    skipped = 0
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt or stmt.startswith("--"):
            continue
        try:
            conn.execute(stmt)
            applied += 1
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                skipped += 1
            else:
                raise
    conn.commit()
    conn.close()
    print(f"SQLite migration: {applied} applied, {skipped} skipped (already exist).")


def main():
    parser = argparse.ArgumentParser(description="Run memory layer migrations")
    parser.add_argument(
        "--backend",
        choices=["postgres", "sqlite", "auto"],
        default="auto",
        help="Database backend to migrate (default: auto-detect)",
    )
    parser.add_argument(
        "--sqlite-path",
        default="memory.db",
        help="Path to SQLite database file (default: memory.db)",
    )
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        try:
            from psap_agent.src.settings import settings

            if (
                settings.POSTGRES_HOST
                and settings.POSTGRES_DB
                and settings.POSTGRES_HOST not in ("pgvector", "")
            ):
                backend = "postgres"
            else:
                backend = "sqlite"
        except Exception:
            backend = "sqlite"

    if backend == "postgres":
        asyncio.run(run_postgres_migration())
    else:
        run_sqlite_migration(args.sqlite_path)


if __name__ == "__main__":
    main()
