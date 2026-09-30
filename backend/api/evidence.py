"""
api/evidence.py — FastAPI router for the evidence vault.

Endpoints:
    GET  /api/v1/evidence                    — List archived clips (pagination + filters)
    GET  /api/v1/evidence/stats               — Vault statistics
    GET  /api/v1/evidence/{event_id}         — Get single event metadata
    GET  /api/v1/evidence/{event_id}/video   — Stream the MP4 clip
    DELETE /api/v1/evidence/{event_id}        — Purge clip + metadata
    POST /api/v1/evidence/{event_id}/bookmark — Toggle bookmark flag
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_archiver import get_evidence_archiver
from storage_purge import get_purge_daemon

router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])

BACKEND_DIR = Path(__file__).resolve().parent.parent
VAULT_DIR = BACKEND_DIR / "evidence_vault"
CLIPS_DIR = VAULT_DIR / "clips"
METADATA_DIR = VAULT_DIR / "metadata"


# ---------------------------------------------------------------------------
# List evidence (must be first — empty path beats /{event_id})
# ---------------------------------------------------------------------------

@router.get("")
async def list_evidence(
    malpractice_type: Optional[str] = Query(None, description="Filter by type: HEAD_TURNING, PEEKING, NOTE_PASSING"),
    track_id: Optional[int] = Query(None, description="Filter by ByteTrack track_id"),
    date_from: Optional[str] = Query(None, description="ISO 8601 start date"),
    date_to: Optional[str] = Query(None, description="ISO 8601 end date"),
    limit: int = Query(50, ge=1, le=200, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Fetch list of archived evidence clips with pagination and filter params.

    Returns:
        {
            "events": [...],
            "total": int,
            "limit": int,
            "offset": int
        }
    """
    archiver = get_evidence_archiver()
    events = archiver.list_events(
        malpractice_type=malpractice_type,
        track_id=track_id,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    # Get total count (without pagination)
    all_events = archiver.list_events(
        malpractice_type=malpractice_type,
        track_id=track_id,
        date_from=date_from,
        date_to=date_to,
        limit=99999,
    )
    return {
        "events": events,
        "total": len(all_events),
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# Vault statistics (must be before /{event_id} to avoid path param match)
# ---------------------------------------------------------------------------

@router.get("/stats")
async def vault_stats():
    """Vault statistics: size, clip count, retention config, recent purges."""
    daemon = get_purge_daemon()
    return daemon.get_stats()


# ---------------------------------------------------------------------------
# Get single event
# ---------------------------------------------------------------------------

@router.get("/{event_id}")
async def get_event(event_id: str):
    """Retrieve a single event's metadata by event_id."""
    archiver = get_evidence_archiver()
    event = archiver.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")
    return event


# ---------------------------------------------------------------------------
# Stream video clip
# ---------------------------------------------------------------------------

@router.get("/{event_id}/video")
async def stream_video(event_id: str):
    """
    Stream the specific archived MP4 evidence clip to the frontend.

    Returns a FileResponse with the correct MIME type for browser playback.
    """
    clip_path = CLIPS_DIR / f"{event_id}.mp4"
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"clip {event_id} not found")

    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=f"{event_id}.mp4",
    )


# ---------------------------------------------------------------------------
# Delete evidence
# ---------------------------------------------------------------------------

@router.delete("/{event_id}")
async def delete_evidence(event_id: str):
    """
    Purge specific evidence clip and metadata manually.

    Deletes both the .mp4 clip and .json metadata sidecar.
    """
    archiver = get_evidence_archiver()
    deleted = archiver.delete_event(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")
    return {"deleted": True, "event_id": event_id}


# ---------------------------------------------------------------------------
# Bookmark toggle
# ---------------------------------------------------------------------------

@router.post("/{event_id}/bookmark")
async def toggle_bookmark(event_id: str):
    """
    Toggle the bookmark flag on an evidence clip.

    Bookmarked clips are preserved during disk pressure purges
    and age-based retention sweeps.
    """
    meta_path = METADATA_DIR / f"{event_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")

    try:
        meta = json.loads(meta_path.read_text())
        meta["bookmarked"] = not meta.get("bookmarked", False)
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        return {"event_id": event_id, "bookmarked": meta["bookmarked"]}
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
