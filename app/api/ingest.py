"""
/ingest/journal endpoint.

Accepts a path to a pre-saved journal export zip on the shared /data volume.
Runs ingestion as a background task — returns immediately.
Called internally by TravelNet after saving an uploaded export.
"""
import logging
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
from config import settings
from ingestion.journal.ingest import ingest_from_zip

logger = logging.getLogger(__name__)
router = APIRouter()


class IngestRequest(BaseModel):
    zip_path: str  # absolute path inside the container (shared /data volume)

def _run_ingestion(zip_path: Path) -> None:
    import asyncio
    try:
        logger.info("Starting journal ingestion from %s", zip_path)
        asyncio.run(ingest_from_zip(zip_path))
        logger.info("Journal ingestion complete")
    except Exception as e:
        logger.error("Journal ingestion failed: %s", e, exc_info=True)


@router.post("/journal")
async def ingest_journal(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default=""),
):
    if x_api_key != settings.trevor_api_key:
        raise HTTPException(status_code=401, detail="Unauthorised")

    zip_path = Path(request.zip_path)

    if not zip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Zip file not found at {zip_path}"
        )

    if not zip_path.suffix == ".zip":
        raise HTTPException(status_code=400, detail="Path must point to a .zip file")

    background_tasks.add_task(_run_ingestion, zip_path)

    return {
        "status": "accepted",
        "zip_path": str(zip_path),
        "message": "Ingestion started in background",
    }