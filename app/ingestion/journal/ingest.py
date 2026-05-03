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
import asyncio
import logging
import tempfile
import zipfile
from pathlib import Path

from ingestion.journal.parse import JournalEntry, parse_entry
from config import settings

logger = logging.getLogger(__name__)


def _build_metadata(entry: JournalEntry) -> dict:
    """
    Flatten JournalEntry into a Chroma metadata dict.
    Chroma metadata values must be str, int, float, or bool — no None, no lists.
    Lists are joined as comma-separated strings.
    """
    meta = {
        "date":       entry.timestamp_utc.date().isoformat(),
        "utc_offset": entry.utc_offset_str,
    }

    if entry.location:
        if entry.location.town:
            meta["town"] = entry.location.town
        if entry.location.city:
            meta["city"] = entry.location.city
        if entry.location.country:
            meta["country"] = entry.location.country
        if entry.location.latitude is not None:
            meta["lat"] = entry.location.latitude
        if entry.location.longitude is not None:
            meta["lon"] = entry.location.longitude

    if entry.mood:
        if entry.mood.labels:
            meta["mood_labels"] = ", ".join(entry.mood.labels)
        if entry.mood.associations:
            meta["mood_associations"] = ", ".join(entry.mood.associations)
        if entry.mood.background_color:
            meta["mood_color"] = entry.mood.background_color

    if entry.energy is not None:
        meta["energy"] = entry.energy
    if entry.day_rating_raw is not None:
        meta["day_rating"] = entry.day_rating_raw
    if entry.tags:
        meta["tags"] = ", ".join(entry.tags)
    if entry.best_moment:
        meta["best_moment"] = entry.best_moment
    if entry.low_moment:
        meta["low_moment"] = entry.low_moment
    if entry.who_with:
        meta["who_with"] = entry.who_with

    return meta

async def ingest_from_zip(zip_path: Path) -> None:
    """Extract zip to a temp directory and ingest. Cleans up after."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        await ingest(Path(tmp))


async def ingest(input_dir: Path) -> None:
    from retrieval.chroma_client import get_collection, Collection
    from llm.provider import get_provider

    collection = get_collection(Collection.JOURNAL)
    existing_ids = set(collection.get()["ids"])

    # Support both structured (Entries/ subdir) and flat layouts
    entries_dir = input_dir / "Entries"
    if not entries_dir.exists():
        entries_dir = input_dir  # flat zip — all files in root

    resources_dir = input_dir / "Resources"
    if not resources_dir.exists():
        resources_dir = input_dir  # flat zip — JSON sidecars alongside HTML

    html_files = sorted(entries_dir.glob("*.html"))
    # Exclude index.html which Apple Journal includes as a TOC
    html_files = [f for f in html_files if f.stem != "index"]

    if not html_files:
        raise FileNotFoundError(f"No .html files found in {entries_dir}")

    added = skipped = excluded = 0
    new_entries: list[tuple[str, JournalEntry]] = []

    for html_path in html_files:
        entry_id = html_path.stem
        if entry_id in existing_ids:
            skipped += 1
            continue

        # Filter entries before travel start date
        try:
            entry = parse_entry(html_path, resources_dir if resources_dir.exists() else None)
        except ValueError as e:
            logger.info(f"SKIP {html_path.name}: {e}")
            excluded += 1
            continue
        except Exception as e:
            logger.error(f"ERROR {html_path.name}: {type(e).__name__}: {e}")
            excluded += 1
            continue

        entry_date = entry.timestamp_utc.date().isoformat()
        if entry_date < settings.travel_start_date:
            logger.debug(f"SKIP {html_path.name}: before travel start ({entry_date})")
            excluded += 1
            continue

        if not entry.journal_prose:
            logger.info(f"SKIP {html_path.name}: no journal prose")
            excluded += 1
            continue

        new_entries.append((entry_id, entry))

    new_entries.sort(key=lambda x: x[1].timestamp_utc, reverse=True)
    logger.info(
        f"Found {len(new_entries)} new entries to ingest "
        f"({skipped} skipped, {excluded} excluded)"
    )

    provider = get_provider()

    async def _ingest_entries():
        nonlocal added
        for entry_id, entry in new_entries:
            try:
                embedding = await provider.embed(entry.journal_prose)
                metadata = _build_metadata(entry)
                collection.upsert(
                    ids=[entry_id],
                    documents=[entry.journal_prose],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )
                added += 1
                logger.info(f"Ingested {entry_id} ({metadata['date']})")
            except Exception as e:
                logger.error(f"Failed to ingest {entry_id}: {e}")
                excluded += 1

    await _ingest_entries()
    logger.info(
        f"Ingestion complete: {added} added, {skipped} skipped, {excluded} excluded"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Ingest Apple Journal entries into Trevor's vector store."
    )
    parser.add_argument("--input", required=True,
        help="Path to journal export directory (containing Entries/ and Resources/) "
             "or use --zip for a zip file")
    parser.add_argument("--zip", dest="use_zip", action="store_true",
        help="Treat --input as a zip file path rather than a directory")
    args = parser.parse_args()

    if args.use_zip:
        asyncio.run(ingest_from_zip(Path(args.input)))
    else:
        asyncio.run(ingest(Path(args.input)))