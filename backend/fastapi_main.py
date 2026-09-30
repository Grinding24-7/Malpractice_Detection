"""
fastapi_main.py — FastAPI application entry point (Week 6-8).

Sets up:
    - CORS middleware (allows Vite dev server + production same-origin)
    - WebSocket + REST router mounting from api/ package
    - Startup/shutdown lifecycle:
        * StreamPipeline 3-stage async pipeline (Week 8)
        * EvidenceArchiver vault initialization (Week 7)
        * StoragePurgeDaemon background cleanup (Week 7)
    - Prometheus metrics endpoint (GET /metrics)
    - Static file serving for the React SPA (production build)

Run:
    cd backend && uvicorn fastapi_main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.stream import router as stream_router
from api.ws_router import router as ws_v2_router
from api.endpoints import router as endpoints_router
from api.evidence import router as evidence_router
from stream_pipeline import get_stream_pipeline
from evidence_archiver import get_evidence_archiver
from storage_purge import get_purge_daemon
from metrics import get_metrics

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
BACKEND_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for streaming pipeline + evidence subsystem."""
    # Week 8: 3-stage async streaming pipeline
    pipeline = get_stream_pipeline()
    await pipeline.start()
    print("[fastapi] stream pipeline started (3-stage)", flush=True)

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
    await pipeline.stop()
    print("[fastapi] all subsystems stopped", flush=True)


app = FastAPI(
    title="Malpractice Detection API",
    version="3.0.0",
    description="Week 8 — Hardened 3-stage async streaming pipeline for intelligent exam malpractice detection.",
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
app.include_router(ws_v2_router)
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
        "version": "3.0.0",
        "streaming": True,
        "pipeline": "3-stage-async",
        "evidence_vault": True,
        "active_tracks": len(bm.active_ids()),
        "buffer_frames": bm.total_frames(),
        "buffer_memory_mb": round(bm.estimated_memory_mb(), 2),
    }


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
@app.get("/metrics")
async def metrics():
    """Prometheus text-format metrics endpoint."""
    m = get_metrics()
    return Response(
        content=m.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/api/v1/metrics")
async def metrics_json():
    """JSON metrics snapshot for dashboards."""
    m = get_metrics()
    return m.snapshot()


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
