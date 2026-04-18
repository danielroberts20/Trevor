"""
Tool: query_db

Executes a SELECT query against travel.db and returns results.
The LLM generates the SQL; this module enforces SELECT-only at the code layer.

Security: dual-layer enforcement
  Layer 1 — system prompt instructs LLM to only write SELECT statements.
  Layer 2 — db_client.query() rejects anything that isn't SELECT before execution.
"""

from retrieval.db_client import query as db_query, get_schema

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "query_db",
        "description": (
            "Execute a SELECT SQL query against the TravelNet database to retrieve "
            "structured telemetry data: GPS location history, health metrics (steps, activity), "
            "financial records, HMM travel segments, DBSCAN location clusters, and anomaly flags. "
            "Only SELECT statements are permitted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQLite SELECT statement.",
                },
            },
            "required": ["sql"],
        },
    },
}


async def run(sql: str) -> list[dict]:
    """
    Execute the query_db tool.
    db_client.query() enforces SELECT-only; raises ValueError otherwise.
    """
    return db_query(sql)
