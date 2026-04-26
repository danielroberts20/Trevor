"""
Tool: search_chrome_docs

Given a natural language query, embeds it and retrieves the most semantically
similar Chrome extension documentation entries from the Chrome docs vector store.

The store is populated from doc page entries scraped from the Chrome Web Store /
developer docs. Each entry corresponds to a section or page of documentation.
"""

# Tool definition in OpenAI function-calling format.
# Normalised to Anthropic's input_schema format in the Anthropic provider if needed.
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_chrome_docs",
        "description": (
            "Search the Chrome extension developer documentation for content semantically "
            "related to a query. Use for questions about Chrome extension APIs, manifest "
            "fields, permissions, event lifecycle, or any Chrome Web Store / developer "
            "docs topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language search query.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of documentation entries to retrieve (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def run(query: str, n_results: int = 5) -> list[dict]:
    """
    Execute the Chrome docs search tool.
    Returns a list of relevant documentation chunks with metadata.
    """
    # TODO: wire up once the Chrome docs vector store is built
    # from retrieval.chroma_client import search_collection
    # embedding = embed(query)
    # return search_collection("chrome_docs", embedding, n_results=n_results)
    raise NotImplementedError
