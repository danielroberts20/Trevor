"""
Tool: search_docs

Given a natural language query, embeds it and retrieves the most semantically
similar documentation entries from the docs Chroma vector store.

The store is populated from doc page entries scraped and indexed at build time.
Each entry corresponds to a section or page of documentation.
"""

# Tool definition in OpenAI function-calling format.
# Normalised to Anthropic's input_schema format in the Anthropic provider if needed.
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": (
            "Search indexed documentation entries for content semantically related to a query. "
            "Use for questions about APIs, configuration, concepts, or any topic covered "
            "in the documentation pages stored in the Chroma vector store."
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


async def run(query: str, n_results: int = 5) -> list[dict]:
    """
    Execute the docs search tool.
    Returns a list of relevant documentation chunks with metadata.
    """
    from llm.provider import get_provider
    from retrieval.chroma_client import search, Collection
    provider = get_provider()
    embedding = await provider.embed(query)
    raise NotImplementedError
    return search(Collection.DOCS, embedding, n_results=n_results)
