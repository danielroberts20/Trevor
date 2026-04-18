"""
test_query_db.py — Unit tests for retrieval.db_client.query().

Covers:
  - Return structure: columns (list[str]), rows (list[list]), row_count (int), truncated (bool)
  - row_count equals len(rows) after truncation, not the total DB count
  - rows contains at most row_limit rows
  - Fetches row_limit+1 to detect truncation; row_count is row_limit when truncated
  - truncated is True when results are capped, False otherwise
  - ValueError raised (not caught) for non-SELECT statements: INSERT, UPDATE, DELETE, DROP
  - Multi-statement strings rejected even when a SELECT is present
  - Empty result sets: rows=[], row_count=0, truncated=False; columns populated from description
  - SQL errors returned as {"error": "..."} — not raised as exceptions
"""

import pytest
from unittest.mock import patch

from retrieval.db_client import query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patched_conn(mem_db):
    """Context manager: patch _get_conn to return the in-memory DB."""
    return patch("retrieval.db_client._get_conn", return_value=mem_db)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestQueryDbReturnStructure:

    def test_query_db_returns_expected_keys(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id, destination FROM trips")
        assert set(result.keys()) >= {"columns", "rows", "row_count", "truncated"}

    def test_query_db_columns_is_list_of_strings(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id, destination FROM trips")
        assert isinstance(result["columns"], list)
        assert all(isinstance(c, str) for c in result["columns"])

    def test_query_db_columns_match_selected_fields(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id, destination FROM trips")
        assert result["columns"] == ["id", "destination"]

    def test_query_db_rows_is_list_of_lists(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips")
        assert isinstance(result["rows"], list)
        assert all(isinstance(r, list) for r in result["rows"])

    def test_query_db_row_count_matches_len_of_rows(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips")
        assert result["row_count"] == len(result["rows"])


# ---------------------------------------------------------------------------
# Truncation behaviour
# ---------------------------------------------------------------------------

class TestQueryDbTruncation:

    def test_query_db_truncated_false_when_results_fit(self, mem_db):
        # 3 rows in trips; limit=10 — nothing truncated
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips", row_limit=10)
        assert result["truncated"] is False

    def test_query_db_truncated_true_when_results_capped(self, mem_db):
        # 3 rows in trips; limit=2 — should truncate
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips", row_limit=2)
        assert result["truncated"] is True

    def test_query_db_rows_at_most_row_limit(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips", row_limit=2)
        assert len(result["rows"]) <= 2

    def test_query_db_row_count_is_row_limit_not_row_limit_plus_one_when_truncated(self, mem_db):
        # The implementation fetches row_limit+1 to detect truncation, then
        # slices back. row_count must reflect what was returned, not the sentinel.
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips", row_limit=2)
        assert result["row_count"] == 2

    def test_query_db_row_count_equals_actual_count_when_not_truncated(self, mem_db):
        # 3 rows, limit=100 — row_count should be 3
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips", row_limit=100)
        assert result["row_count"] == 3


# ---------------------------------------------------------------------------
# SELECT-only enforcement
# ---------------------------------------------------------------------------

class TestQueryDbSelectOnlyEnforcement:

    def test_query_db_raises_value_error_for_insert(self, mem_db):
        with _patched_conn(mem_db):
            with pytest.raises(ValueError):
                query("INSERT INTO trips(id, destination) VALUES (99, 'Paris')")

    def test_query_db_raises_value_error_for_update(self, mem_db):
        with _patched_conn(mem_db):
            with pytest.raises(ValueError):
                query("UPDATE trips SET destination='London' WHERE id=1")

    def test_query_db_raises_value_error_for_delete(self, mem_db):
        with _patched_conn(mem_db):
            with pytest.raises(ValueError):
                query("DELETE FROM trips WHERE id=1")

    def test_query_db_raises_value_error_for_drop(self, mem_db):
        with _patched_conn(mem_db):
            with pytest.raises(ValueError):
                query("DROP TABLE trips")

    def test_query_db_raises_value_error_for_multi_statement_with_select(self, mem_db):
        # A SELECT followed by a second statement must still be rejected
        with _patched_conn(mem_db):
            with pytest.raises(ValueError):
                query("SELECT id FROM trips; DELETE FROM trips")


# ---------------------------------------------------------------------------
# Empty result sets
# ---------------------------------------------------------------------------

class TestQueryDbEmptyResults:

    def test_query_db_empty_result_set_rows_is_empty_list(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips WHERE id = 9999")
        assert result["rows"] == []

    def test_query_db_empty_result_set_row_count_is_zero(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips WHERE id = 9999")
        assert result["row_count"] == 0

    def test_query_db_empty_result_set_truncated_is_false(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM trips WHERE id = 9999")
        assert result["truncated"] is False

    def test_query_db_empty_result_set_columns_from_description(self, mem_db):
        # cursor.description is NOT None on an empty result — columns must be
        # populated from the schema, not left empty just because no rows matched
        with _patched_conn(mem_db):
            result = query("SELECT id, destination FROM trips WHERE id = 9999")
        assert result["columns"] == ["id", "destination"]


# ---------------------------------------------------------------------------
# SQL error handling
# ---------------------------------------------------------------------------

class TestQueryDbSqlErrors:

    def test_query_db_sql_error_returns_error_key(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM nonexistent_table")
        assert "error" in result

    def test_query_db_sql_error_does_not_raise(self, mem_db):
        # Bad table name must be caught and returned as a dict, not raised
        with _patched_conn(mem_db):
            result = query("SELECT id FROM nonexistent_table")
        assert isinstance(result, dict)

    def test_query_db_syntax_error_returns_error_key(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT FROM WHERE")
        assert "error" in result

    def test_query_db_sql_error_error_value_is_string(self, mem_db):
        with _patched_conn(mem_db):
            result = query("SELECT id FROM nonexistent_table")
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
