# Trevor

A RAG-based conversational AI that provides a natural language query interface over [TravelNet](https://github.com/YOUR_USERNAME/travelnet) — real personal telemetry and journal data collected across a multi-year trip through ~9 countries.

Ask Trevor questions like *"how did I feel in Vietnam?"*, *"what was my average daily spend in Japan?"*, or *"did my step count drop in weeks I overspent?"* and get answers grounded in actual data.

> **Portfolio context:** Trevor is one half of a two-project portfolio targeting AI Engineer roles. TravelNet is the data platform (GPS, health, financial telemetry via Raspberry Pi); Trevor is the LLM query layer on top of it. Together they demonstrate the full arc: collect real data → analyse it with ML → make it conversational with LLMs → ship it.

---

## What it does

Trevor has access to two data sources:

**Apple Journal entries** — unstructured prose written throughout the trip, embedded into a vector store and retrieved semantically. Enables questions about experiences, feelings, reflections, and events.

**TravelNet telemetry** — structured GPS, health, and financial data stored in SQLite, queried directly via generated SQL. Enables aggregate questions about movement, spending, and activity.

The LLM decides which source (or both) to query for each question using tool-calling, rather than a hardcoded router. This handles cross-stream queries naturally — *"were you happier in countries where you spent less?"* retrieves journal sentiment and spending data in a single response.

---

## Architecture

```
User
 │
 ▼
FastAPI /chat
 │
 ▼
LLM (tool-calling loop)
 ├── search_journal()  →  Chroma vector store  →  journal chunks + metadata
 └── query_db()        →  SQLite (travel.db)   →  structured telemetry rows
 │
 ▼
Grounded response
```

### Key design decisions

**Tool-calling over hardcoded routing.** The LLM is given two tools and decides which to invoke. This makes cross-stream queries trivial and is more architecturally honest than a classifier-based router.

**Entries as atomic chunks.** Journal entries are not split further — each entry is one Chroma document. Entries are natural semantic units; sub-entry chunking risks splitting context the LLM needs together. Structured fields (mood, location, coordinates, timestamps) are stored as Chroma metadata for filtered retrieval.

**Direct DB access.** Trevor reads `travel.db` directly via a read-only volume mount rather than going through TravelNet's API, which is write-facing. WAL mode is enabled on every connection to allow concurrent reads while TravelNet's ingest service writes. Only `SELECT` statements are permitted, enforced at two layers: system prompt instruction and code-level validation in `db_client.py`.

**Configurable LLM provider.** All LLM calls go through a provider abstraction (`llm/provider.py`). Swap between Ollama (local), OpenAI, and Anthropic via a single environment variable. Ollama on a local GPU is used during development; a cloud provider is used for demos where model quality matters.

**Stateless `/chat` endpoint.** Conversation history is returned with every response and managed client-side. Keeps the backend simple and mirrors standard chat API patterns.

**Reasoning Freedom.** A per-conversation dial (0–10) controls temperature and system prompt verbosity. Speculative content is always labelled `<SPEC>`; citations use `<CITE:chunk_id>` tags resolved in post-processing.

---

## Stack

| Component | Choice | Rationale |
|---|---|---|
| API framework | FastAPI | Async, typed, matches TravelNet stack |
| Vector store | Chroma | Zero idle cost vs Pinecone; good for intermittent portfolio use |
| LLM (dev) | Ollama / phi3:mini | Free, local, no API cost during development |
| LLM (prod) | OpenAI / Anthropic | Configurable; swap via env var |
| Embedding model | nomic-embed-text (Ollama) | Local; note: Anthropic has no embedding API |
| Database | SQLite (TravelNet's travel.db) | Read-only access; WAL mode for concurrency |
| Container runtime | Docker + Docker Compose | Consistent with TravelNet deployment |

---

## Repository structure

```
trevor/
├── app/
│   ├── main.py                  # FastAPI entrypoint; /health check with startup validation
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
│       └── journal_ingestor.py  # Offline script: parses, embeds, and populates Chroma
├── evals/                       # Evaluation framework
├── CLAUDE.md                    # Architecture decisions; context for AI coding tools
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Infrastructure

Trevor runs as a Docker container on a Raspberry Pi alongside TravelNet, joined to the same Docker network. LLM inference runs on a separate Windows PC (RTX 3080) accessible over Tailscale, exposed via a FastAPI wrapper. The LLM compute node is a swappable dependency — Trevor degrades gracefully if it is unreachable (PC asleep).

```
Raspberry Pi (always-on)
├── TravelNet containers  (travelnet network)
└── Trevor container      (joins travelnet network; port 8300)
      │
      └── reads /data/travel.db  (shared read-only volume)
      └── reads/writes /chroma   (own named volume)
      │
      └──[Tailscale]──▶  Windows PC (RTX 3080)
                          └── Ollama inference (port 8100)
```

---

## Running locally

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

A healthy response looks like:

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

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` |
| `OLLAMA_BASE_URL` | — | URL of the Ollama FastAPI wrapper |
| `OLLAMA_MODEL` | `phi3:mini` | Chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OPENAI_API_KEY` | — | Required if provider is `openai` |
| `ANTHROPIC_API_KEY` | — | Required if provider is `anthropic` |
| `DB_PATH` | `/data/travel.db` | Path to TravelNet SQLite database |
| `CHROMA_PATH` | `/chroma` | Path to Chroma persistent storage |
| `TRAVEL_START_DATE` | `2025-06-01` | Entries before this date are excluded from ingestion |
| `TREVOR_API_KEY` | `changeme` | Bearer token for `/chat` endpoint |
| `ANONYMIZED_TELEMETRY` | — | Set to `False` to silence Chroma telemetry logs |

---

## Evaluation

An evaluation framework is planned. See [`evals/README.md`](evals/README.md).

---

## Related

- [TravelNet](https://github.com/YOUR_USERNAME/travelnet) — the data platform Trevor queries