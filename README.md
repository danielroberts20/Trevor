# Trevor

> A RAG-based conversational AI for querying three years of personal travel data: GPS tracks, health metrics, spending logs, and daily journal entries across ~9 countries.

**[Live Demo (coming soon)](#)** · **[TravelNet Repository](https://github.com/danielroberts20/TravelNet)** · **[TravelNet Demo Site](https://travelnet.dev)**

---

## What is Trevor?

Trevor is a conversational AI assistant built on top of [TravelNet](https://travelnet.dev) — a personal data platform collecting real GPS, health, and financial telemetry during a multi-year journey. Where TravelNet surfaces data visually through a dashboard, Trevor makes it queryable through natural language.

Ask Trevor things like:

- *"Summarise how I felt during my time in Vietnam."*
- *"What was I spending most on in the weeks I was least active?"*
- *"Did my mood improve after I left Southeast Asia?"*
- *"What happened around the spending spike in February?"*

Trevor draws on two data sources: structured telemetry from TravelNet's SQLite database, and unstructured journal entries written throughout the trip. The combination enables a class of query that neither source alone could answer — correlating what the data shows with what was actually happening on the ground.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Query                        │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   FastAPI API   │
              │  /chat endpoint │
              └────────┬────────┘
                       │
         ┌─────────────▼──────────────┐
         │      LLM + Tool Router     │
         │  (decides what to query)   │
         └──┬──────────────────────┬──┘
            │                      │
   ┌────────▼────────┐   ┌─────────▼────────┐
   │  Chroma Vector  │   │  TravelNet DB    │
   │  Store          │   │  (SQLite)        │
   │ (journal chunks │   │  structured      │
   │  + embeddings)  │   │  telemetry +     │
   └────────┬────────┘   │  mood data       │
            │            └─────────┬────────┘
            │                      │
         ┌──▼──────────────────────▼──┐
         │     Retrieved Context      │
         └──────────────┬─────────────┘
                        │
              ┌─────────▼──────────┐
              │   LLM Response     │
              │ (grounded in data) │
              └────────────────────┘
```

Trevor uses a **tool-calling** retrieval strategy rather than a hardcoded router. The LLM is given two tools — `search_journal` (semantic search over journal embeddings) and `query_db` (structured SQL against TravelNet's database) — and decides which to invoke, or both, based on what the question requires. This handles cross-stream queries naturally without explicit routing logic.

---

## Data Sources

### Journal Entries (Apple Journal)

Daily journal entries written throughout the trip using Apple Journal with a Shortcuts-generated template. Each entry captures a narrative account of the day alongside automatically attached metadata: timestamp (with timezone), location, and state-of-mind check-ins recorded via Apple HealthKit.

Apple Journal exports all entries as a zip archive containing:

- `Entries/` — one HTML file per journal entry
- `Resources/` — UUID-named asset pairs (`.heic` visual + `.json` sidecar) for each attached asset (mood graphics, location pins, photos)

The resource JSON sidecars contain structured data extracted per-asset:
- **Mood assets** — emotion labels and life-area associations
- **Location assets** — precise latitude/longitude, place name, place type
- **Photo/media assets** — capture timestamp, place name

### Mood Data (Apple HealthKit via Health Auto Export)

Apple Journal's state-of-mind check-ins are stored in HealthKit and exported separately via [Health Auto Export](https://www.healthautoexport.com/). Each mood record contains:

| Field | Description |
|---|---|
| `id` | HealthKit UUID |
| `start` | Precise UTC timestamp of when mood was logged |
| `valence` | Float from -1.0 (very unpleasant) to 1.0 (very pleasant) |
| `valenceClassification` | Discretised band: `very_unpleasant` → `very_pleasant` |
| `labels` | Emotion tags selected by user (e.g. `joyful`, `anxious`) |
| `associations` | Life areas linked to mood (e.g. `work`, `travel`, `health`) |

`valence` and `valenceClassification` are **not present in the Journal export** — they exist only in the HealthKit export. This makes Health Auto Export a required data dependency, not an optional enrichment.

Mood data is ingested into TravelNet's database where it is also available for TravelNet's ML analysis. Trevor queries TravelNet's mood table at journal ingestion time to enrich each entry with its corresponding valence score.

### TravelNet Telemetry

Structured data collected via Raspberry Pi throughout the trip:

- **GPS / location** — continuous movement tracks, country and city transitions, meaningful place discovery via DBSCAN clustering
- **Health** — step count, activity levels, sleep, heart rate
- **Financial** — spending logs categorised by type and country
- **ML model outputs** — HMM travel segments, DBSCAN clusters, anomaly flags — precomputed and persisted to the database rather than called at query time

Trevor queries this data directly against TravelNet's SQLite database.

---

## Ingestion Pipeline

Journal ingestion is an **offline script** (`ingestion/journal_ingestor.py`), not a live service. It is run manually against an Apple Journal export, and is safe to re-run — existing entries are skipped.

```
Apple Journal Export (zip)
        │
        ▼
Parse index.html → enumerate all entries
        │
        ▼
For each entry HTML:
  - Extract body text
  - Parse entry timestamp from filename
    (Shortcut format encodes explicit timezone offset)
  - Extract asset UUIDs → look up resource sidecar JSONs
  - Separate mood, location, and media assets
        │
        ▼
Filter: skip entries before TRAVEL_START_DATE
        │
        ▼
Idempotency check: skip already-processed filenames
        │
        ▼
For new entries:
  - Query TravelNet mood table for matching HealthKit records
  - Attach enriched metadata
  - Embed entry text → upsert into Chroma
```

### Mood–Entry Joining

Mood records are joined to journal entries at ingestion time using a timestamp-window approach:

1. Parse entry creation time `T=0` from the Shortcut filename (includes explicit timezone offset; converted to UTC)
2. Query TravelNet's mood table for records where `start` falls within `[T, T+10min]`
3. Confirm match using label and association string overlap against the entry's resource sidecar mood JSON
4. If multiple records fall within the window, prefer the closest timestamp; use label overlap as a tiebreaker

The **forward-only window** (`[T, T+10min]`) reflects the Shortcut workflow: the journal entry is created first, then the mood is logged manually — always after `T=0`, never before.

**Rationale for ingestion-time joining (vs query-time):** joining at ingestion means the full enriched record — text, location metadata, and mood valence — is stored together in Chroma. This keeps retrieval simple and removes a runtime dependency on TravelNet being live during every query. The accepted tradeoff is that if mood data arrives after ingestion, it won't be reflected until the next ingestion run.

### Chunking Strategy

Journal entries are treated as **atomic chunks** — one entry produces one Chroma document. Entries written via the Shortcut template are typically a few hundred words, well within embedding model token limits. Splitting at sub-entry granularity would produce fragments lacking the context needed for coherent retrieval.

If an entry exceeds the embedding model's token limit, it is split at paragraph boundaries with entry-level metadata attached to every child chunk.

**Metadata stored per vector (not embedded — used for filtered retrieval):**

| Field | Source |
|---|---|
| `entry_date` | Filename |
| `entry_timestamp_utc` | Filename (converted from local + offset) |
| `place_name` | Resource sidecar (location asset) |
| `latitude`, `longitude` | Resource sidecar (location asset) |
| `place_type` | Resource sidecar (location asset) |
| `valence` | TravelNet mood table (via join) |
| `valence_classification` | TravelNet mood table (via join) |
| `mood_labels` | TravelNet mood table (via join) |
| `mood_associations` | TravelNet mood table (via join) |

---

## Retrieval

Incoming queries are routed by the LLM to one or more tools:

**`search_journal`** — for journal queries, reflective questions, or anything requiring narrative context. The query is embedded and compared against journal chunk vectors in Chroma. Top-N results are returned with their metadata.

**`query_db`** — for structured telemetry queries requiring precision: spending totals, step counts, date-specific lookups, country-level aggregations. The LLM generates a `SELECT` query against TravelNet's SQLite database. Only `SELECT` statements are permitted, enforced at two layers: system prompt instruction and code-level validation.

**Hybrid** — for cross-stream queries, both tools are invoked within a single LLM turn and results are merged into a unified context before response generation.

---

## Anomaly Explainer

The Anomaly Explainer is Trevor's most sophisticated feature — a direct integration between TravelNet's ML layer and Trevor's LLM layer.

TravelNet continuously monitors telemetry streams for anomalies: unusual spending spikes, atypical movement patterns, significant deviations from baseline health metrics. When an anomaly is detected, TravelNet calls Trevor's `/explain` endpoint with the anomaly metadata (type, timestamp, magnitude, affected data stream).

Trevor then:
1. Retrieves journal entries from the surrounding time window via semantic search
2. Fetches the relevant telemetry context from TravelNet's database
3. Constructs a prompt grounding the LLM in both the anomaly data and the journal narrative
4. Returns a plain-English explanation of what likely caused the anomaly

This pattern — ML detection piped into LLM interpretation — mirrors a common production use case: automated monitoring systems that explain their own alerts in human language.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Startup check — reports DB and vector store status |
| `/chat` | POST | Main conversational interface. Accepts message + conversation history; returns grounded response and updated history |
| `/explain` | POST | Called by TravelNet on anomaly detection. Returns plain-English explanation with journal context |

---

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Backend | FastAPI | Consistent with TravelNet stack; async-native |
| LLM (dev) | Ollama / phi3:mini | Free, local, no API cost during development |
| LLM (prod) | OpenAI / Anthropic | Configurable via env var — swap without code changes |
| Embedding model | nomic-embed-text (Ollama) | Local; note: Anthropic has no embedding API |
| Vector store | Chroma | Zero idle cost vs Pinecone; appropriate for intermittent portfolio use |
| Database | SQLite (TravelNet's travel.db) | Read-only access; WAL mode enabled for concurrent reads |
| Container runtime | Docker + Docker Compose | Consistent with TravelNet deployment |

---

## Infrastructure

Trevor runs as a Docker container on a Raspberry Pi alongside TravelNet, joined to the same Docker network. LLM inference runs on a separate Windows PC (RTX 3080) accessible over Tailscale.

```
Raspberry Pi (always-on)
├── TravelNet containers  (travelnet Docker network)
└── Trevor container      (joins travelnet network; port 8300)
      │
      ├── reads /data/travel.db  (shared read-only volume)
      └── reads/writes /chroma   (own named volume: trevor-chroma)
      │
      └──[Tailscale]──▶  Windows PC (RTX 3080)
                          └── Ollama inference (port 8100)
```

---

## Repository Structure

```
trevor/
├── app/
│   ├── main.py                  # FastAPI entrypoint; /health with startup validation
│   ├── config.py                # All env vars via pydantic-settings
│   ├── api/
│   │   └── chat.py              # POST /chat — stateless; client manages history
│   ├── llm/
│   │   ├── base.py              # Abstract interface: chat() and embed()
│   │   ├── provider.py          # get_provider() — selects backend from LLM_PROVIDER
│   │   ├── ollama.py            # Ollama via local FastAPI wrapper
│   │   ├── openai.py            # OpenAI SDK
│   │   └── anthropic.py        # Anthropic SDK
│   ├── retrieval/
│   │   ├── chroma_client.py     # ChromaDB interface; persistent volume
│   │   └── db_client.py         # Direct SQLite reads; SELECT-only enforced
│   ├── tools/
│   │   ├── search_journal.py    # Tool: semantic journal search
│   │   └── query_db.py          # Tool: structured telemetry SQL queries
│   └── ingestion/
│       └── journal_ingestor.py  # Offline script: parse, enrich, embed, upsert
├── evals/                       # Evaluation framework (approach TBD)
├── CLAUDE.md                    # Architecture decisions; context for AI coding tools
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running Locally

```bash
# 1. Copy and configure environment
cp .env.example .env

# 2. Ensure the travelnet Docker network exists
docker network create travelnet   # skip if TravelNet is already running

# 3. Build and start
docker compose up --build

# 4. Verify
curl http://localhost:8300/health
```

A healthy response:

```json
{
  "status": "ok",
  "provider": "ollama",
  "checks": {
    "db": {"status": "ok", "tables": 28},
    "chroma": {"status": "ok", "collections": 0}
  }
}
```

`collections: 0` is expected before ingestion has been run.

### Ingesting journal entries

```bash
docker compose exec trevor python -m ingestion.journal_ingestor --input /path/to/export
```

Ingestion is idempotent — re-running on the same export skips existing entries.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` |
| `OLLAMA_BASE_URL` | — | URL of the Ollama FastAPI wrapper |
| `OLLAMA_MODEL` | `phi3:mini` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OPENAI_API_KEY` | — | Required if provider is `openai` |
| `ANTHROPIC_API_KEY` | — | Required if provider is `anthropic` |
| `DB_PATH` | `/data/travel.db` | Path to TravelNet SQLite database (inside container) |
| `CHROMA_PATH` | `/chroma` | Path to Chroma persistent storage (inside container) |
| `TRAVEL_START_DATE` | `2025-06-01` | Entries before this date are excluded from ingestion |
| `TREVOR_API_KEY` | `changeme` | Bearer token for `/chat` and `/explain` endpoints |
| `ANONYMIZED_TELEMETRY` | — | Set to `False` to silence Chroma telemetry logs |

---

## Design Decisions

A full log of architectural decisions and their rationale is maintained in [`CLAUDE.md`](./CLAUDE.md). Key decisions:

- **Tool-calling over hardcoded routing** — the LLM decides which data source to query, handling cross-stream queries naturally without a classifier-based router
- **Chroma over Pinecone** — zero idle cost; appropriate for a portfolio project with intermittent use
- **Atomic journal chunking** — entries are not split below entry level; paragraph-level splitting applied only when entries exceed token limits
- **Ingestion-time mood joining** — mood valence is joined at ingestion rather than query time, keeping retrieval simple at the cost of a small staleness window
- **Forward-only mood join window** — the Shortcut workflow guarantees mood is logged after journal creation, so the join window is `[T, T+10min]`
- **Filename-based idempotency** — Apple Journal generates deterministic filenames from entry timestamps, making them stable across repeated exports
- **Direct DB access** — Trevor reads `travel.db` directly (read-only volume) rather than through TravelNet's write-facing API; WAL mode for concurrent read safety

---

## Evaluation

An evaluation framework is planned — a suite of test questions with expected answer shapes, re-run whenever chunking strategy, retrieval parameters, or prompt templates change. See [`evals/README.md`](evals/README.md).

---

## Privacy

Raw financial, health, and journal data is not exposed through the public demo. The public deployment uses a curated subset of data with sensitive fields aggregated or redacted — mirroring the data scoping decisions that arise in any production RAG system handling personal or proprietary data.

---

## Project Status

Trevor is under active development alongside TravelNet. The data collection stage is underway — journal entries and telemetry are being collected throughout the trip. The ingestion pipeline, retrieval layer, and full API are in active development.

---

## Related

- [TravelNet](https://github.com/danielroberts20/TravelNet) — the data platform Trevor queries
- [TravelNet Demo Site](https://travelnet.dev) — live dashboard where Trevor will be embedded

---

*Built as part of a portfolio demonstrating end-to-end data engineering and LLM integration: real data collection → ML analysis → conversational AI query layer → deployed product.*