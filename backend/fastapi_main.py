"""
fastapi_main.py — FastAPI application entry point for Week 6 dashboard.

Sets up:
    - CORS middleware (allows Vite dev server + production same-origin)
    - WebSocket + REST router mounting from api/ package
    - Startup/shutdown lifecycle for the StreamingEngine producer thread
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
from streaming_backend import get_streaming_engine

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
BACKEND_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the async streaming engine."""
    engine = get_streaming_engine()
    loop = asyncio.get_event_loop()
    engine.start(loop)
    print("[fastapi] streaming engine started", flush=True)
    yield
    engine.stop()
    print("[fastapi] streaming engine stopped", flush=True)


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


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "streaming": True}


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
