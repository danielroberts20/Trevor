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

from ingestion.journal.parse import JournalEntry, parse_entry
from config import settings

logger = logging.getLogger(__name__)


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

    entries_dir = input_dir / "Entries"
    resources_dir = input_dir / "Resources"

    html_files = sorted(entries_dir.glob("*.html"))
    if not html_files:
        raise FileNotFoundError(f"No .html files found in {entries_dir}")

    new_entries: list[JournalEntry] = []

    # Find new entries to add, and parse them into JournalEntry objects.
    for html_path in html_files:
        entry_id = html_path.stem

        if entry_id in existing_ids:
            skipped += 1
            continue

        try:
            new_entries.append(parse_entry(html_path, resources_dir))
        except ValueError as e:
            logger.info(f"SKIP {html_path.name}: {e}")
            excluded += 1
        except Exception as e:
            logger.error(f"ERROR {html_path.name}: {type(e).__name__}: {e}")
            excluded += 1

    new_entries.sort(key=lambda e: e.timestamp_utc, reverse=True)

    for entry in new_entries:
        # TODO: embed entry["journal_prose"] and upsert into collection
        # embedding = await provider.embed(entry["journal_prose"])
        # collection.upsert(
        #     ids=[entry_id],
        #     documents=[entry["journal_prose"]],
        #     embeddings=[embedding],
        #     metadatas=[{k: v for k, v in entry.items()
        #                 if k not in ("id", "journal_prose") and v is not None}],
        # )
        added += 1

    logger.info(f"Ingestion complete: {added} added, {skipped} skipped, {excluded} excluded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest Apple Journal entries into Trevor's vector store.")
    parser.add_argument("--input", required=True, help="Path to journal export directory")
    args = parser.parse_args()
    ingest(Path(args.input))
