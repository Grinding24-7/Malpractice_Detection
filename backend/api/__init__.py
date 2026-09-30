"""
api package — FastAPI routers for Week 6 real-time streaming dashboard.

Routes:
    /api/v1/stream/mjpeg   — MJPEG frame generator (GET)
    /api/v1/stream/ws      — WebSocket telemetry + compressed frames (WS)
    /api/v1/upload-test-video — Multipart upload for offline analysis (POST)
    /api/v1/job-status/{id}  — Pollable progress endpoint (GET)
    /api/v1/webcam/toggle    — Local webcam ingestion control (POST)
"""
