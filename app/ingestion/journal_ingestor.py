"""
Journal ingestion script — run offline, not as a service.

Takes Apple Journal HTML exports, parses entries, embeds them,
and populates the Chroma vector store.

Usage:
    python -m ingestion.journal_ingestor --input /path/to/journal/export

Design decisions:
  - Entries are treated as atomic chunks (one Chroma document per journal entry).
    Rationale: entries are already natural semantic units; sub-entry chunking
    risks splitting context that the LLM needs together.
  - Deduplication: filename-based. Re-running the script on the same export
    is safe — existing entries are skipped.
  - Filtering: entries before TRAVEL_START_DATE are excluded.
    Mechanism for filtering non-travel personal entries: TBD.
  - Structured fields from the resource sidecar JSON (lat, lon, place name,
    mood, mood_score) are stored as Chroma metadata for filtered retrieval.
"""

import argparse
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def parse_entry(html_path: Path) -> dict | None:
    """
    Parse a single Apple Journal HTML export file.

    Returns a dict with:
        id: str           — filename stem (deduplication key)
        text: str         — plain text content of the entry
        date: str         — ISO date string
        location: str     — place name from sidecar JSON, if available
        lat: float | None
        lon: float | None
        mood: str | None  — requires Health Auto Export for valence data
        mood_score: float | None

    Returns None if the entry should be excluded (pre-travel, or filtered).
    """
    # TODO: implement HTML parsing + sidecar JSON loading
    raise NotImplementedError


def ingest(input_dir: Path) -> None:
    """
    Walk input_dir, parse each entry, embed it, and upsert into Chroma.
    Skips entries that already exist in the collection (filename-based dedupe).
    """
    from retrieval.chroma_client import get_collection
    # TODO: initialise LLM provider for embeddings
    # from llm.provider import get_provider

    collection = get_collection()
    existing_ids = set(collection.get()["ids"])

    html_files = sorted(input_dir.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files in {input_dir}")

    added, skipped, excluded = 0, 0, 0

    for html_path in html_files:
        entry_id = html_path.stem

        if entry_id in existing_ids:
            skipped += 1
            continue

        entry = parse_entry(html_path)
        if entry is None:
            excluded += 1
            continue

        # TODO: embed entry["text"] and upsert into collection
        # embedding = await provider.embed(entry["text"])
        # collection.upsert(
        #     ids=[entry_id],
        #     documents=[entry["text"]],
        #     embeddings=[embedding],
        #     metadatas=[{k: v for k, v in entry.items()
        #                 if k not in ("id", "text") and v is not None}],
        # )
        added += 1

    logger.info(f"Ingestion complete: {added} added, {skipped} skipped, {excluded} excluded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest Apple Journal entries into Trevor's vector store.")
    parser.add_argument("--input", required=True, help="Path to journal export directory")
    args = parser.parse_args()
    ingest(Path(args.input))
