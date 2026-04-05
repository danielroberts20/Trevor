"""
Trevor — conversational AI query interface over TravelNet data.
FastAPI entrypoint.
"""

import sqlite3
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_db() -> dict:
    """Verify travel.db is reachable and returns a row count as a sanity check."""
    try:
        conn = sqlite3.connect(settings.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")  # enable WAL for concurrent reads
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
        table_count = cursor.fetchone()[0]
        conn.close()
        return {"status": "ok", "tables": table_count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_chroma() -> dict:
    """Verify Chroma is reachable and the client can be instantiated."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_path)
        collections = client.list_collections()
        return {"status": "ok", "collections": len(collections)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks; log results but don't block startup on warnings."""
    logger.info("Trevor starting up...")
    logger.info(f"LLM provider: {settings.llm_provider}")
    logger.info(f"DB path: {settings.db_path}")
    logger.info(f"Chroma path: {settings.chroma_path}")

    db_status = _check_db()
    logger.info(f"DB check: {db_status}")

    chroma_status = _check_chroma()
    logger.info(f"Chroma check: {chroma_status}")

    app.state.startup_checks = {
        "db": db_status,
        "chroma": chroma_status,
    }

    yield

    logger.info("Trevor shutting down.")


app = FastAPI(
    title="Trevor",
    description="Conversational AI query interface over TravelNet data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/chat", tags=["chat"])


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns Trevor's status plus results of startup dependency checks.
    Useful for verifying the container is wired up correctly.
    """
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "checks": app.state.startup_checks,
    }
