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
import sqlparse
import logging
from config import settings


logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    # Open read-only using SQLite URI — no WAL sidecar files needed
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def _is_select(sql: str) -> bool:
    # Filter out empty tokens (trailing semicolons produce a None-type statement)
    parsed = [s for s in sqlparse.parse(sql.strip()) if s.get_type() is not None]
    if len(parsed) != 1:
        return False
    return parsed[0].get_type() == "SELECT"


def query(sql: str, params: tuple = (), row_limit: int = 100) -> dict:
    if not _is_select(sql):
        raise ValueError(f"Only SELECT queries are permitted. Got: {sql[:80]!r}")
    conn = _get_conn()
    try:
        cursor = conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(row_limit + 1)
        truncated = len(rows) > row_limit
        rows = rows[:row_limit]
        return {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }
    except Exception as e:
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": str(e),
        }
    finally:
        conn.close()

def get_schema() -> str:
    try:
        result = query(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        columns = result["columns"]  # ["name", "sql"]
        name_idx = columns.index("name")
        sql_idx = columns.index("sql")

        schema = "\n\n".join(
            f"-- {row[name_idx]}\n{row[sql_idx]}"
            for row in result["rows"]
            if row[sql_idx] and not row[name_idx].startswith("sqlite_")
        )
        logger.info(f"Schema loaded: {len(result['rows'])} tables")
        return schema
    except Exception as e:
        logger.warning(f"Schema load failed: {e}")
        return ""