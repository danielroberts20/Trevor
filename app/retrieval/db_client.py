"""
Direct read-only interface to travel.db (TravelNet's SQLite database).

Security model:
  - Only SELECT statements are permitted, enforced at two layers:
      1. This module rejects any non-SELECT statement before execution.
      2. The system prompt instructs the LLM to only generate SELECT queries.
  - WAL mode is set on every connection to allow concurrent reads while
    TravelNet's ingest service may be writing.

Decision log: Trevor queries the DB directly rather than going through
TravelNet's FastAPI ingest service, which is write-facing and not designed
as a query API. This also means Trevor can run independently of TravelNet.
"""

import sqlite3
from config import settings


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def query(sql: str, params: tuple = ()) -> list[dict]:
    """
    Execute a SELECT query and return results as a list of dicts.
    Raises ValueError if the statement is not a SELECT.
    """
    normalised = sql.strip().upper()
    if not normalised.startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are permitted. Got: {sql[:80]!r}")

    conn = _get_conn()
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_schema() -> list[dict]:
    """
    Return the DB schema (table names + CREATE statements).
    Used to build the system prompt so the LLM knows what tables exist.
    """
    return query(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
