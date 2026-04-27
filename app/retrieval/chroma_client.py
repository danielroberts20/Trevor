"""
Chroma client — wraps ChromaDB for journal entry and doc page retrieval.

Decision log: Chroma chosen over Pinecone for zero idle cost.
Persistent client with a named volume ensures embeddings survive container restarts.

Collection schema:
  - id: filename-based (deduplication key)
  - document: journal entry text (full entry, treated as atomic chunk)
  - metadata:
      date: str (ISO)
      location: str
      lat: float | None
      lon: float | None
      mood: str | None
      mood_score: float | None
"""

from enum import Enum

import chromadb #type: ignore
from config import settings

COLLECTION_NAME = "journal_entries"

class Collection(Enum):
    JOURNAL = "journal_entries"
    DOCS = "doc_pages"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_path)


def get_collection(name: Collection, client: chromadb.PersistentClient | None = None):
    c = client or get_client()
    return c.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def search(name: Collection, query_embedding: list[float], n_results: int = 5) -> list[dict]:
    """
    Return the top-N journal chunks most similar to the query embedding.
    Each result includes the document text and its metadata.
    """
    # TODO: implement
    # collection = get_collection()
    # results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    # return [{"text": doc, "metadata": meta}
    #         for doc, meta in zip(results["documents"][0], results["metadatas"][0])]
    client = get_client()
    collection = get_collection(name, client)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return [{"text": doc, "metadata": meta} 
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])]