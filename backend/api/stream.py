"""
stream.py — FastAPI router for real-time video streaming.

Endpoints:
    GET  /api/v1/stream/mjpeg  — MJPEG frame generator with ByteTrack overlays
    WS   /api/v1/stream/ws    — High-frequency WebSocket: JSON telemetry +
                                 compressed base64 JPEG frames at 30 FPS

Backpressure:
    Frame-skipping is handled by StreamingEngine.consume() which drops
    intermediate frames when the queue depth exceeds the threshold.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming_backend import get_streaming_engine, FrameTelemetry

router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])


# ---------------------------------------------------------------------------
# MJPEG streaming endpoint
# ---------------------------------------------------------------------------

@router.get("/mjpeg")
async def mjpeg_stream():
    """
    MJPEG frame generator streaming processed OpenCV frames with ByteTrack
    overlays.  Uses multipart/x-mixed-replace for browser-compatible
    progressive rendering.
    """
    engine = get_streaming_engine()

    async def frame_generator():
        while True:
            telemetry = await engine.consume()
            if telemetry is None or telemetry.annotated_frame is None:
                continue
            ret, jpeg = cv2.imencode(
                ".jpg", telemetry.annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_stream(ws: WebSocket):
    """
    High-frequency WebSocket yielding synchronized JSON telemetry +
    compressed base64 JPEG frames at 30 FPS.

    Payload format:
    {
        "frame_id": 1042,
        "timestamp": 34.73,
        "active_tracks": 4,
        "students": [
            {
                "track_id": 3,
                "bbox": [140, 90, 280, 320],
                "prediction": "HEAD_TURNING",
                "confidence": 0.92,
                "keypoints": [[160, 105], [162, 101], ...],
                "velocity_spike": 0.68
            }
        ],
        "frame_jpeg": "<base64-encoded JPEG>"
    }
    """
    await ws.accept()
    engine = get_streaming_engine()
    send_interval = 1.0 / 30  # 30 FPS target
    last_send = time.monotonic()

    try:
        while True:
            telemetry = await engine.consume()
            if telemetry is None:
                continue

            now = time.monotonic()
            # Frame pacing: skip if sending too fast
            if now - last_send < send_interval * 0.8:
                continue
            last_send = now

            # Encode frame to base64 JPEG
            frame_b64 = ""
            if telemetry.annotated_frame is not None:
                ret, jpeg = cv2.imencode(
                    ".jpg", telemetry.annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 72],
                )
                if ret:
                    frame_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

            # Build telemetry payload
            students_data = [
                {
                    "track_id": s.track_id,
                    "bbox": s.bbox,
                    "prediction": s.prediction,
                    "confidence": s.confidence,
                    "keypoints": s.keypoints,
                    "velocity_spike": s.velocity_spike,
                    "ear_ratio": s.ear_ratio,
                    "norm_vertical_drop": s.norm_vertical_drop,
                }
                for s in telemetry.students
            ]

            payload = {
                "frame_id": telemetry.frame_id,
                "timestamp": telemetry.timestamp,
                "active_tracks": telemetry.active_tracks,
                "students": students_data,
                "frame_jpeg": frame_b64,
            }

            await ws.send_json(payload)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] client error: {e}", flush=True)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
