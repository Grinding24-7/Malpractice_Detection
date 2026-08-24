"""
fastapi_main.py — FastAPI application entry point (Week 6-7).

Sets up:
    - CORS middleware (allows Vite dev server + production same-origin)
    - WebSocket + REST router mounting from api/ package
    - Startup/shutdown lifecycle:
        * StreamingEngine producer thread (Week 6)
        * EvidenceArchiver vault initialization (Week 7)
        * StoragePurgeDaemon background cleanup (Week 7)
    - Static file serving for the React SPA (production build)

Run:
    cd backend && uvicorn fastapi_main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.stream import router as stream_router
from api.endpoints import router as endpoints_router
from api.evidence import router as evidence_router
from streaming_backend import get_streaming_engine
from evidence_archiver import get_evidence_archiver
from storage_purge import get_purge_daemon

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
BACKEND_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for streaming engine + evidence subsystem."""
    # Week 6: streaming engine
    engine = get_streaming_engine()
    loop = asyncio.get_event_loop()
    engine.start(loop)
    print("[fastapi] streaming engine started", flush=True)

    # Week 7: evidence vault + purge daemon
    archiver = get_evidence_archiver()
    archiver.start()
    purge = get_purge_daemon()
    purge.start()
    print("[fastapi] evidence archiver + purge daemon started", flush=True)

    yield

    # Shutdown
    purge.stop()
    archiver.stop()
    engine.stop()
    print("[fastapi] all subsystems stopped", flush=True)


app = FastAPI(
    title="Malpractice Detection API",
    version="1.0.0",
    description="Week 6 — Real-time streaming dashboard API for intelligent exam malpractice detection.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow Vite dev server (port 5173) and production same-origin
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
        # Production: same-origin, no origin header needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------
app.include_router(stream_router)
app.include_router(endpoints_router)
app.include_router(evidence_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
async def health():
    from buffer_manager import get_buffer_manager
    bm = get_buffer_manager()
    return {
        "status": "ok",
        "version": "2.0.0",
        "streaming": True,
        "evidence_vault": True,
        "active_tracks": len(bm.active_ids()),
        "buffer_frames": bm.total_frames(),
        "buffer_memory_mb": round(bm.estimated_memory_mb(), 2),
    }


# ---------------------------------------------------------------------------
# Serve React SPA (production build)
# ---------------------------------------------------------------------------
dist_dir = FRONTEND_DIR / "dist"
if dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="spa")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
