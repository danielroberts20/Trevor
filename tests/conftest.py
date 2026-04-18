"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Shared fixtures for the Trevor test suite.

main.py calls configure_logging() at module level; this is harmless in tests
(it only sets up console handlers, no file I/O) so no import-time patch is
needed. The lifespan side effects (DB checks, schema load, background thread)
are suppressed inside the client fixture by patching before TestClient enters
the app's lifespan context.
"""

import sqlite3
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from main import app


TEST_API_KEY = "test-trevor-key"

TEST_SCHEMA = (
    "-- trips\n"
    "CREATE TABLE trips(id INTEGER PRIMARY KEY, destination TEXT, start_date TEXT);\n\n"
    "-- expenses\n"
    "CREATE TABLE expenses(id INTEGER PRIMARY KEY, trip_id INTEGER, amount REAL, category TEXT);"
)

_DB_DDL = """
CREATE TABLE trips (
    id          INTEGER PRIMARY KEY,
    destination TEXT    NOT NULL,
    start_date  TEXT
);
CREATE TABLE expenses (
    id       INTEGER PRIMARY KEY,
    trip_id  INTEGER NOT NULL,
    amount   REAL    NOT NULL,
    category TEXT
);
"""


@pytest.fixture
def mem_db():
    """
    In-memory SQLite database with two travel-themed tables and a handful of rows.
    Used to drive query_db and get_schema tests without touching the real DB.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DB_DDL)
    conn.executemany(
        "INSERT INTO trips(id, destination, start_date) VALUES (?, ?, ?)",
        [
            (1, "Melbourne", "2026-09-12"),
            (2, "Auckland",  "2027-06-01"),
            (3, "Bangkok",   "2027-12-01"),
        ],
    )
    conn.executemany(
        "INSERT INTO expenses(id, trip_id, amount, category) VALUES (?, ?, ?, ?)",
        [
            (1, 1, 45.00,  "food"),
            (2, 1, 120.00, "accommodation"),
            (3, 2, 30.00,  "transport"),
        ],
    )
    conn.commit()
    return conn


@pytest.fixture
def client():
    """
    TestClient for the Trevor FastAPI app.

    Patches lifespan dependencies so the app starts cleanly without Docker:
      - DB and Chroma health checks return stub OK responses
      - get_schema() returns TEST_SCHEMA instead of querying the real DB
      - start_background_tasks() is suppressed (no SSH polling thread)

    Patches api.chat.settings so the API key comparison uses TEST_API_KEY.
    """
    settings_mock = MagicMock()
    settings_mock.trevor_api_key = TEST_API_KEY

    with patch("api.chat.settings", settings_mock), \
         patch("main._check_db",       return_value={"status": "ok", "tables": 2}), \
         patch("main._check_chroma",   return_value={"status": "ok", "collections": 0}), \
         patch("main.get_schema",      return_value=TEST_SCHEMA), \
         patch("main.start_background_tasks"):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def mock_provider():
    """
    Configurable mock LLM provider.

    Defaults to a single 'stop' response with a fixed content string.
    Tests that need different behaviour can override mock_provider.chat.return_value
    or set mock_provider.chat.side_effect.
    """
    provider = MagicMock()
    provider.chat = AsyncMock(return_value={
        "content": "Hello from mock.",
        "finish_reason": "stop",
        "tool_calls": [],
        "assistant_message": {"role": "assistant", "content": "Hello from mock."},
    })
    return provider
