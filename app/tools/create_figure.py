"""
Tool: create_figure

Executes a SELECT query against travel.db and returns results.
The LLM generates the SQL; this module enforces SELECT-only at the code layer.

Security: dual-layer enforcement
  Layer 1 — system prompt instructs LLM to only write SELECT statements.
  Layer 2 — db_client.query() rejects anything that isn't SELECT before execution.
"""

from retrieval.db_client import query


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "create_figure",
        "description": (
            "Generates a data-driven chart from the TravelNet database. "
            "The tool executes a SELECT-only SQL query, then transforms the resulting data "
            "into a Plotly-compatible figure specification for visualisation in the client UI.\n\n"

            "Use this tool when the user requests trends, comparisons, distributions, time series, "
            "or any form of data visualisation (e.g. spending over time, location heat, activity breakdown).\n\n"

            "Trevor is responsible for generating a valid SQLite SELECT query that returns "
            "aggregated and chart-ready data (e.g. grouped or time-bucketed results).\n\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "description": (
                        "Type of chart to generate. Must be one of: "
                        "'bar', 'line', 'scatter', 'pie', 'histogram'."
                    ),
                },
                "sql": {
                    "type": "string",
                    "description": (
                        "A valid SQLite SELECT statement that returns chart-ready data. "
                        "Must include aggregation where appropriate (e.g. GROUP BY for bar/pie, "
                        "time bucketing for line charts)."
                    ),
                },
                "x_column": {
                    "type": "string",
                    "description": "Column name used for x-axis (categorical or time).",
                },
                "y_column": {
                    "type": "string",
                    "description": "Column name used for y-axis (numeric metric).",
                },
                "series_column": {
                    "type": "string",
                    "description": (
                        "Optional column for grouping multiple series (used in multi-line or grouped bar charts)."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Human-readable title for the chart.",
                }
            },
            "required": ["chart_type", "sql", "x_column", "y_column", "title"],
        },
    },
}

def run(chart_type: str,
              sql: str,
              x_column: str,
              y_column: str,
              title: str,
              series_column: str = None) -> list[dict]:
    """
    Execute the query_db tool.
    db_client.query() enforces SELECT-only; raises ValueError otherwise.
    """
    data = query(sql)
    return {
        "figure": {"plotly": "data"},
        "data_summary": data,
        "meta": {
            "sql": sql,
            "row_count": data["row_count"]
        }
    }