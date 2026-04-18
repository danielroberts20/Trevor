"""
test_db_client.py — Unit tests for retrieval.db_client.get_schema().

Covers:
  - Returns a non-empty string when the DB is reachable
  - Returns an empty string when the DB is unreachable — does not raise
  - Returned string contains table names from the DB
  - Does not include tables whose sql column is NULL (e.g. sqlite_sequence)
  - Format: each table entry is preceded by a "-- tablename" comment
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from retrieval.db_client import get_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema_db():
    """In-memory DB for schema-load tests: two real tables + sqlite_sequence."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # AUTOINCREMENT forces sqlite_sequence to appear in sqlite_master with sql=NULL
    conn.executescript("""
        CREATE TABLE journeys (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            country     TEXT NOT NULL,
            arrived_at  TEXT
        );
        CREATE TABLE waypoints (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            journey_id INTEGER NOT NULL,
            latitude   REAL,
            longitude  REAL
        );
        INSERT INTO journeys(country, arrived_at) VALUES ('Australia', '2026-09-12');
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

class TestGetSchemaReachability:

    def test_get_schema_returns_nonempty_string_when_db_reachable(self):
        with patch("retrieval.db_client._get_conn", return_value=_make_schema_db()):
            result = get_schema()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_schema_returns_empty_string_when_db_unreachable(self):
        # _get_conn raising simulates a missing or inaccessible DB file
        with patch("retrieval.db_client._get_conn", side_effect=Exception("no such file")):
            result = get_schema()
        assert result == ""

    def test_get_schema_does_not_raise_when_db_unreachable(self):
        with patch("retrieval.db_client._get_conn", side_effect=Exception("no such file")):
            result = get_schema()  # must not raise
        assert result == ""


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

class TestGetSchemaContent:

    def test_get_schema_contains_table_names(self):
        with patch("retrieval.db_client._get_conn", return_value=_make_schema_db()):
            result = get_schema()
        assert "journeys" in result
        assert "waypoints" in result

    def test_get_schema_excludes_null_sql_tables(self):
        # sqlite_sequence has sql=NULL in sqlite_master — must be filtered out
        with patch("retrieval.db_client._get_conn", return_value=_make_schema_db()):
            result = get_schema()
        assert "sqlite_sequence" not in result

    def test_get_schema_format_has_comment_before_each_table(self):
        with patch("retrieval.db_client._get_conn", return_value=_make_schema_db()):
            result = get_schema()
        # Each table should be preceded by "-- <tablename>"
        assert "-- journeys" in result
        assert "-- waypoints" in result

    def test_get_schema_comment_precedes_create_statement(self):
        with patch("retrieval.db_client._get_conn", return_value=_make_schema_db()):
            result = get_schema()
        # The comment line must appear before the CREATE TABLE for each table
        journeys_comment_pos = result.index("-- journeys")
        journeys_create_pos  = result.index("CREATE TABLE journeys")
        assert journeys_comment_pos < journeys_create_pos
