"""
Trevor — conversational AI query interface over TravelNet data.
FastAPI entrypoint.
"""

import sqlite3
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI #type: ignore
from fastapi.middleware.cors import CORSMiddleware #type: ignore

from api.chat import router as chat_router
from compute.manager import start_background_tasks
from retrieval.db_client import get_schema
from config import settings
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def _check_db() -> dict:
    """Verify travel.db is reachable and returns a row count as a sanity check."""
    try:
        conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
        table_count = cursor.fetchone()[0]
        conn.close()
        return {"status": "ok", "tables": table_count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_chroma() -> dict:
    """Verify Chroma is reachable and the client can be instantiated."""
    try:
        import chromadb #type: ignore
        client = chromadb.PersistentClient(path=settings.chroma_path)
        collections = client.list_collections()
        return {"status": "ok", "collections": len(collections)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks and start background tasks."""
    logger.info("Trevor starting up...")
    logger.info(f"LLM provider: {settings.llm_provider}")
    logger.info(f"DB path: {settings.db_path}")
    logger.info(f"Chroma path: {settings.chroma_path}")

    db_status = _check_db()
    logger.info(f"DB check: {db_status}")

    chroma_status = _check_chroma()
    logger.info(f"Chroma check: {chroma_status}")

    # Load schema once; stored on app.state for the lifetime of the process
    app.state.db_schema = get_schema()

    app.state.startup_checks = {
        "db": db_status,
        "chroma": chroma_status,
    }

    start_background_tasks()

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
    """
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "checks": app.state.startup_checks,
        "schema_loaded": bool(app.state.db_schema),
    }