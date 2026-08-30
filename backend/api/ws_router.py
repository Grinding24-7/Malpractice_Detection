"""
ws_router.py — Week 8: Hardened WebSocket streaming router.

Endpoints:
    WS   /api/v1/stream/v2/ws   — Hardened WebSocket with heartbeat,
                                  reconnection state, adaptive quality.

This replaces the unhardened /api/v1/stream/ws from Week 6 with:
    * Heartbeat ping/pong for proxy-compatible keep-alive.
    * Reconnection state tokens for seamless frontend recovery.
    * Per-connection health tracking.
    * Binary wire protocol (4-byte header len + JSON header + JPEG).

Backward-compatible: the old /api/v1/stream/ws endpoint still works
via api/stream.py.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, WebSocket

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stream_pipeline import get_stream_pipeline
from api.harden import HardenedWebSocket

router = APIRouter(prefix="/api/v1/stream", tags=["streaming-v2"])


@router.websocket("/v2/ws")
async def hardened_websocket_stream(ws: WebSocket):
    """
    Hardened WebSocket endpoint.

    Protocol:
        Server → Client (binary):
            [4 bytes: header length (big-endian uint32)]
            [N bytes: JSON header]
            [remaining: raw JPEG bytes]

        Server → Client (JSON):
            {"type": "ping", "ts": <float>}
            {"type": "reconnection_state", "token_id": "...", ...}

        Client → Server (JSON):
            {"type": "pong"}
            {"type": "request_state"}
            {"type": "ack_reconnection", "token_id": "..."}
    """
    pipeline = get_stream_pipeline()
    hardened = HardenedWebSocket(
        ws=ws,
        pipeline=pipeline,
        heartbeat_interval=15.0,
        heartbeat_timeout=10.0,
    )
    await hardened.run()
