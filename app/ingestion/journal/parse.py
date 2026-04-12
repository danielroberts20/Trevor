"""
parse_journal.py — Apple Journal HTML entry parser for Trevor ingestion pipeline.

Parses the export format produced by Apple Journal (iOS 17+).

Export layout:
    Archive/
    ├── index.html              TOC — not used by this parser
    ├── Entries/
    │   └── YYYY-MM-DD_YYYY-MM-DD-DDD_HH_MM_SS_+HHMM.html   (one file per entry)
    └── Resources/
        ├── <UUID>.heic / .mp4 / .gif                         (media assets)
        └── <UUID>.json                                        (asset sidecar)

Filename encoding:
    The filename is the authoritative timestamp source. The UTC offset is
    encoded explicitly (e.g. +0100), enabling exact UTC conversion regardless
    of where in the world the entry was written. The pageHeader and title divs
    are display text only.

    Pattern: YYYY-MM-DD_YYYY-MM-DD-DDD_HH_MM_SS_+HHMM.html
    Example: 2026-04-12_2026-04-12-Sun_17_57_32_+0100.html

Resource sidecar schemas (Resources/<UUID>.json):
    State of mind:  {"labels": "Content, Satisfied", "associations": "Hobbies"}
    Place visit:    {"visits": [{"placeName": ..., "latitude": ..., "longitude": ...}]}
    Date-only:      {"date": <Core Data epoch — seconds since 2001-01-01>}

Entry template (enforced via iOS Shortcut — all fields present in every entry):
    📍 LOCATION     Town, Lat, Lon, Country
    🌤 WEATHER      Temp/Max/Min °C, Condition
    📅 DATE & TIME  Redundant with filename — parsed but not stored
    😶 MOOD         Reminder string — ignored (mood comes from stateOfMind asset)
    ⚡ ENERGY       Level: 1–5
    ⭐️ DAY RATING   Level: 1–10  (raw; z-score normalisation done at ingest time)
    🔖 Tags         Comma-separated
    ✨ Best moment  One-liner
    😓 Low moment   One-liner
    👥 Who with     Freeform
    🚗 Transport    Freeform
    💸 Rough spend  Freeform
    📝 JOURNAL      Free prose — everything after this header paragraph

Design decisions:
    - Sections identified by keyword in first line, not raw emoji bytes.
      Emoji matching is fragile due to variation selectors (U+FE0F); keywords
      are stable.
    - Mood labels/associations read from the HTML overlay (gridItemOverlayHeader /
      gridItemOverlayFooter) rather than the JSON sidecar. Both contain the same
      data; the HTML is already loaded so the sidecar read is avoided.
    - Freeform prose fields ("none", "None") are kept as the literal string.
      Semantic interpretation is left to the LLM. Only numeric fields (energy,
      day_rating_raw) return Python None on missing/invalid values.
    - day_rating_z (rolling z-score normalisation) is NOT computed here.
      It requires the full dataset window and is computed by the ingestion
      orchestrator (ingest_journal.py) after all entries are parsed.
    - asset_uuids lists all resource UUIDs referenced in the entry, enabling
      the ingestion layer to load photo visit data for location enrichment if needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MoodData:
    labels: list[str]           # e.g. ["Content", "Satisfied"]
    associations: list[str]     # e.g. ["Hobbies"]
    background_color: Optional[str] = None  # CSS hex from stateOfMind div style


@dataclass
class LocationData:
    town: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None


@dataclass
class WeatherData:
    temp_c: Optional[float] = None
    max_c: Optional[float] = None
    min_c: Optional[float] = None
    condition: Optional[str] = None


@dataclass
class JournalEntry:
    # --- Identity ---
    filename: str                       # original filename, no path
    timestamp_local: datetime           # naive datetime in local wall-clock time
    timestamp_utc: datetime             # UTC-aware datetime
    utc_offset_str: str                 # e.g. "+0100"

    # --- Mood / wellbeing ---
    mood: Optional[MoodData]
    energy: Optional[int]               # 1–5
    day_rating_raw: Optional[int]       # 1–10, as written; z-score added at ingest time

    # --- Structured template fields ---
    location: Optional[LocationData]
    weather: Optional[WeatherData]
    tags: Optional[list[str]]
    best_moment: Optional[str]
    low_moment: Optional[str]
    who_with: Optional[str]
    transport: Optional[str]
    rough_spend: Optional[str]

    # --- Content ---
    journal_prose: Optional[str]        # free prose after the 📝 JOURNAL header

    # --- Asset references ---
    asset_uuids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Matches: 2026-04-12_2026-04-12-Sun_17_57_32_+0100.html
_FILENAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})"            # group 1: date YYYY-MM-DD
    r"_\d{4}-\d{2}-\d{2}-[A-Za-z]+"   # _YYYY-MM-DD-DDD (day-of-week, discarded)
    r"_(\d{2})_(\d{2})_(\d{2})"        # groups 2-4: HH MM SS
    r"_([+-]\d{4})"                     # group 5: UTC offset e.g. +0100
    r"\.html$"
)


def _parse_filename(filename: str) -> tuple[datetime, datetime, str]:
    """
    Returns (timestamp_local, timestamp_utc, utc_offset_str).
    Raises ValueError if the filename does not match the expected pattern.
    """
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Filename does not match expected Apple Journal export pattern: {filename!r}\n"
            f"Expected: YYYY-MM-DD_YYYY-MM-DD-DDD_HH_MM_SS_+HHMM.html"
        )

    date_str, hh, mm, ss, offset_str = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)

    # Parse naive local datetime
    local_dt = datetime.strptime(f"{date_str} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")

    # Parse UTC offset string → timedelta
    sign = 1 if offset_str[0] == "+" else -1
    offset_hours = int(offset_str[1:3])
    offset_mins = int(offset_str[3:5])
    utc_offset = timedelta(hours=sign * offset_hours, minutes=sign * offset_mins)

    # Convert to UTC-aware datetime
    utc_dt = (local_dt - utc_offset).replace(tzinfo=timezone.utc)

    return local_dt, utc_dt, offset_str


# ---------------------------------------------------------------------------
# Asset grid parsing
# ---------------------------------------------------------------------------

def _load_mood_sidecar(uuid: str, resources_dir: Optional[Path]) -> tuple[list[str], list[str]]:
    """
    Loads the full, untruncated labels and associations from the JSON sidecar
    for a stateOfMind asset.

    The HTML overlay (gridItemOverlayHeader/Footer) truncates long lists with
    "and more" — the sidecar always contains the complete data.

    Falls back to empty lists if resources_dir is None or the file is missing.
    """
    if not resources_dir:
        return [], []

    sidecar_path = resources_dir / f"{uuid}.json"
    if not sidecar_path.exists():
        return [], []

    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        labels_raw = data.get("labels", "")
        assoc_raw = data.get("associations", "")
        labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
        associations = [a.strip() for a in assoc_raw.split(",") if a.strip()]
        return labels, associations
    except (json.JSONDecodeError, OSError):
        return [], []


def _parse_asset_grid(
    soup: BeautifulSoup,
    resources_dir: Optional[Path] = None,
) -> tuple[Optional[MoodData], list[str]]:
    """
    Extracts stateOfMind data and all asset UUIDs from the assetGrid div.
    Returns (mood_data_or_None, list_of_all_uuids).

    If resources_dir is provided, mood labels and associations are loaded from
    the JSON sidecar (complete list). Otherwise falls back to the HTML overlay
    text, which may be truncated for entries with many labels.

    The background_color hex is always sourced from the HTML inline style —
    it encodes Apple's valence slider position (Very Unpleasant → Very Pleasant),
    set independently of which labels were chosen.
    """
    mood: Optional[MoodData] = None
    uuids: list[str] = []

    grid = soup.find("div", class_="assetGrid")
    if not grid:
        return mood, uuids

    for item in grid.find_all("div", class_="gridItem"):
        uuid = item.get("id", "").strip()
        if uuid:
            uuids.append(uuid)

        classes = item.get("class", [])
        if "assetType_stateOfMind" in classes:
            style = item.get("style", "")

            # Background colour encodes the valence slider position
            bg_match = re.search(r"background-color:\s*(#[0-9A-Fa-f]+)", style)
            bg_color = bg_match.group(1) if bg_match else None

            # Prefer sidecar for complete label list; fall back to HTML overlay
            if resources_dir and uuid:
                labels, associations = _load_mood_sidecar(uuid, resources_dir)
            else:
                header = item.find("div", class_="gridItemOverlayHeader")
                footer = item.find("div", class_="gridItemOverlayFooter")
                labels_raw = header.get_text(strip=True) if header else ""
                assoc_raw = footer.get_text(strip=True) if footer else ""
                labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
                associations = [a.strip() for a in assoc_raw.split(",") if a.strip()]

            mood = MoodData(
                labels=labels,
                associations=associations,
                background_color=bg_color,
            )

    return mood, uuids


# ---------------------------------------------------------------------------
# Structured field parsing
# ---------------------------------------------------------------------------

def _first_line_upper(text: str) -> str:
    """Returns the first line of a paragraph, uppercased, for section identification."""
    return text.split("\n")[0].strip().upper()


def _after_colon(text: str) -> Optional[str]:
    """Returns the text after the first colon, or None if no colon or empty."""
    if ":" not in text:
        return None
    value = text.split(":", 1)[1].strip()
    return value if value else None


def _parse_int_field(text: str) -> Optional[int]:
    """Parses 'Level: 3' style lines. Returns int or None."""
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("LEVEL:"):
            raw = line.split(":", 1)[1].strip()
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def _parse_location(text: str) -> LocationData:
    loc = LocationData()
    for line in text.split("\n")[1:]:  # skip the section header line
        line = line.strip()
        if line.startswith("Town:"):
            loc.town = line.split(":", 1)[1].strip() or None
        elif line.startswith("City:"):
            loc.city = line.split(":", 1)[1].strip() or None
        elif line.startswith("Lat:"):
            try:
                loc.latitude = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Lon:"):
            try:
                loc.longitude = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Country:"):
            loc.country = line.split(":", 1)[1].strip() or None
    return loc


def _parse_weather(text: str) -> WeatherData:
    w = WeatherData()
    for line in text.split("\n")[1:]:
        line = line.strip()
        if "|" in line:
            # "Temp: 8°C | Max: 10°C | Min: 4°C"
            for part in line.split("|"):
                part = part.strip()
                m = re.match(r"([A-Za-z]+):\s*(-?\d+(?:\.\d+)?)°C", part)
                if m:
                    key, val = m.group(1).lower(), float(m.group(2))
                    if key == "temp":
                        w.temp_c = val
                    elif key == "max":
                        w.max_c = val
                    elif key == "min":
                        w.min_c = val
        elif line.startswith("Condition:"):
            w.condition = line.split(":", 1)[1].strip() or None
    return w


def _parse_structured_paragraphs(paragraphs: list[str]) -> dict:
    """
    Walks p.p2 paragraph texts and extracts all structured template fields.
    Returns a dict ready to unpack into JournalEntry kwargs.
    """
    result: dict = {
        "location": None,
        "weather": None,
        "tags": [],
        "energy": None,
        "day_rating_raw": None,
        "best_moment": None,
        "low_moment": None,
        "who_with": None,
        "transport": None,
        "rough_spend": None,
        "journal_prose": None,
    }

    for i, text in enumerate(paragraphs):
        first = _first_line_upper(text)

        # Sections are identified by keyword in their first line, not raw emoji,
        # because emoji bytes are fragile (variation selectors, font encoding).

        if "LOCATION" in first:
            result["location"] = _parse_location(text)

        elif "WEATHER" in first:
            result["weather"] = _parse_weather(text)

        elif "ENERGY" in first:
            result["energy"] = _parse_int_field(text)

        elif "DAY RATING" in first:
            result["day_rating_raw"] = _parse_int_field(text)

        elif "TAGS" in first:
            tags_str = _after_colon(text)
            if tags_str:
                result["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]

        elif "BEST MOMENT" in first:
            result["best_moment"] = _after_colon(text)

        elif "LOW MOMENT" in first:
            result["low_moment"] = _after_colon(text)

        elif "WHO I SPENT" in first or "WHO WITH" in first:
            result["who_with"] = _after_colon(text)

        elif "TRANSPORT" in first:
            result["transport"] = _after_colon(text)

        elif "ROUGH SPEND" in first:
            result["rough_spend"] = _after_colon(text)

        elif "JOURNAL" in first and "📝" in text or first.startswith("📝"):
            # Everything after this paragraph is free prose.
            # Skip the header paragraph itself (which contains the journaling prompt).
            prose_parts = [p.strip() for p in paragraphs[i + 1:] if p.strip()]
            result["journal_prose"] = "\n\n".join(prose_parts) or None
            break  # nothing after the journal prose

        # Deliberately skipped:
        #   📅 DATE & TIME  — redundant with filename
        #   😶 MOOD         — reminder string; mood comes from stateOfMind asset

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_entry(html_path: Path, resources_dir: Optional[Path] = None) -> JournalEntry:
    """
    Parse a single Apple Journal HTML entry file.

    Args:
        html_path:     Path to the .html file inside Entries/.
        resources_dir: Path to the Resources/ directory alongside Entries/.
                       Optional — only needed if you want to load additional
                       sidecar data not already in the HTML (e.g. photo visits).
                       Mood data is read from the HTML and does not require this.

    Returns:
        JournalEntry dataclass with all available fields populated.

    Raises:
        ValueError: if the filename does not match the expected pattern.
        FileNotFoundError: if html_path does not exist.
    """
    if not html_path.exists():
        raise FileNotFoundError(f"Entry file not found: {html_path}")

    filename = html_path.name
    local_dt, utc_dt, offset_str = _parse_filename(filename)

    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    mood, asset_uuids = _parse_asset_grid(soup, resources_dir)

    # Collect all p.p2 paragraph texts (the structured template lives here)
    paragraphs = [p.get_text(separator="\n") for p in soup.find_all("p", class_="p2")]


    fields = _parse_structured_paragraphs(paragraphs)

    return JournalEntry(
        filename=filename,
        timestamp_local=local_dt,
        timestamp_utc=utc_dt,
        utc_offset_str=offset_str,
        mood=mood,
        asset_uuids=asset_uuids,
        **fields,
    )


def parse_export(entries_dir: Path, resources_dir: Optional[Path] = None) -> list[JournalEntry]:
    """
    Parse all .html files in an Entries/ directory.
    Files that fail to parse are logged and skipped (non-fatal).

    Returns entries sorted by timestamp_utc ascending.
    """
    entries: list[JournalEntry] = []

    html_files = sorted(entries_dir.glob("*.html"))
    if not html_files:
        raise FileNotFoundError(f"No .html files found in {entries_dir}")

    for html_path in html_files:
        try:
            entries.append(parse_entry(html_path, resources_dir))
        except ValueError as e:
            print(f"[parse_journal] SKIP {html_path.name}: {e}")
        except Exception as e:
            print(f"[parse_journal] ERROR {html_path.name}: {type(e).__name__}: {e}")

    entries.sort(key=lambda e: e.timestamp_utc)
    return entries


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pprint import pprint

    entries_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Entries")
    resources_dir = entries_dir.parent / "Resources"

    entries = parse_export(entries_dir, resources_dir if resources_dir.exists() else None)

    print(f"\nParsed {len(entries)} entries\n")
    for e in entries:
        print(f"  {e.timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')}  "
              f"[{e.utc_offset_str}]  "
              f"energy={e.energy}  rating={e.day_rating_raw}  "
              f"mood={e.mood.labels if e.mood else None}  "
              f"tags={e.tags}")

    if entries:
        print("\n=== Latest entry detail ===")
        latest = entries[-1]
        pprint({
            "filename": latest.filename,
            "timestamp_utc": latest.timestamp_utc.isoformat(),
            "location": latest.location,
            "weather": latest.weather,
            "mood": latest.mood,
            "energy": latest.energy,
            "day_rating_raw": latest.day_rating_raw,
            "tags": latest.tags,
            "best_moment": latest.best_moment,
            "low_moment": latest.low_moment,
            "who_with": latest.who_with,
            "transport": latest.transport,
            "rough_spend": latest.rough_spend,
            "journal_prose": latest.journal_prose[:100] + "..." if latest.journal_prose else None,
            "asset_uuids": latest.asset_uuids,
        })