"""
Tool: search_journal

Given a natural language query, embeds it and retrieves the most semantically
similar journal entries from Chroma.

This is one of the two tools available to the LLM via function/tool calling.
The LLM decides when to call it based on the user's question.
"""

# Tool definition in OpenAI function-calling format.
# Normalised to Anthropic's input_schema format in the Anthropic provider if needed.
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_journal",
        "description": (
            "Search Dan's travel journal entries for content semantically related to a query. "
            "Use for questions about experiences, feelings, reflections, or events described in the journal."
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
                    "description": "Number of journal entries to retrieve (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


async def run(query: str, n_results: int = 5) -> list[dict]:
    from llm.provider import get_provider
    from retrieval.chroma_client import search, Collection, get_collection

    # Return early if collection is empty — avoids Chroma error when
    # n_results exceeds document count
    collection = get_collection(Collection.JOURNAL)
    count = collection.count()
    if count == 0:
        return [{"text": "No journal entries have been ingested yet.", "metadata": {}}]

    provider = get_provider()
    embedding = await provider.embed(query)
    return search(Collection.JOURNAL, embedding, n_results=min(n_results, count))
    
