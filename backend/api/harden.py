"""
harden.py — Week 8: WebSocket connection hardening middleware.

Provides:
    * HeartbeatPingPong — periodic ``{"type":"ping"}`` keep-alive through
      Nginx / Cloudflare proxy gates.  Client must respond with
      ``{"type":"pong"}`` within the timeout window.
    * ReconnectionState — issues one-time state-initialisation tokens so
      the frontend can seamlessly resume tracking state after a reconnect.
    * ConnectionHealth — per-connection health tracker recording latency,
      missed heartbeats, and last-active timestamp.

Usage (in ws_router.py):
    from api.harden import HardenedWebSocket

    ws = HardenedWebSocket(websocket, pipeline, heartbeat_interval=15)
    await ws.run()   # blocks until disconnect
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect


# ---------------------------------------------------------------------------
# Reconnection state token
# ---------------------------------------------------------------------------

@dataclass
class ReconnectionToken:
    """One-time token carrying tracking state for seamless reconnection."""
    token_id: str
    issued_at: float
    active_tracks: int
    last_frame_id: int
    # Per-track summary: {track_id: {prediction, bbox, ...}}
    track_state: dict[int, dict[str, Any]] = field(default_factory=dict)
    expires_in: float = 30.0  # seconds

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.issued_at) > self.expires_in

    def to_dict(self) -> dict:
        return {
            "type": "reconnection_state",
            "token_id": self.token_id,
            "active_tracks": self.active_tracks,
            "last_frame_id": self.last_frame_id,
            "track_state": {str(k): v for k, v in self.track_state.items()},
        }


# ---------------------------------------------------------------------------
# Connection health tracker
# ---------------------------------------------------------------------------

@dataclass
class ConnectionHealth:
    """Per-connection health metrics."""
    client_id: str
    connected_at: float = field(default_factory=time.monotonic)
    last_pong: float = field(default_factory=time.monotonic)
    last_frame_sent: float = 0.0
    missed_pongs: int = 0
    total_pongs: int = 0
    total_pings: int = 0
    total_frames: int = 0
    rtt_ms: float = 0.0  # round-trip time of last heartbeat

    @property
    def is_healthy(self) -> bool:
        return self.missed_pongs < 3

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.connected_at

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "uptime_s": round(self.uptime, 1),
            "missed_pongs": self.missed_pongs,
            "total_pongs": self.total_pongs,
            "total_frames": self.total_frames,
            "rtt_ms": round(self.rtt_ms, 1),
        }


# ---------------------------------------------------------------------------
# Hardened WebSocket wrapper
# ---------------------------------------------------------------------------

class HardenedWebSocket:
    """
    Wraps a raw FastAPI WebSocket with heartbeat, reconnection state,
    and health monitoring.

    Call ``run()`` to enter the main loop — it will:
      1. Accept the connection and send an initial state token.
      2. Start a heartbeat ping task.
      3. Bridge incoming client messages (pong) and outgoing frames.
      4. Handle disconnects gracefully.
    """

    def __init__(
        self,
        ws: WebSocket,
        pipeline: Any,  # StreamPipeline
        client_id: Optional[str] = None,
        heartbeat_interval: float = 15.0,
        heartbeat_timeout: float = 10.0,
    ) -> None:
        self._ws = ws
        self._pipeline = pipeline
        self._client_id = client_id or uuid.uuid4().hex[:8]
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout

        self.health = ConnectionHealth(client_id=self._client_id)
        self._reconnection_token: Optional[ReconnectionToken] = None
        self._running = False
        self._ping_sent_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: accept, heartbeat, bridge frames, disconnect."""
        await self._ws.accept()
        self._running = True

        # Register with pipeline
        _, self._frame_queue = await self._pipeline.register_ws_client()

        # Issue initial reconnection state
        await self._send_reconnection_state()

        # Launch heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            await asyncio.gather(
                heartbeat_task,
                self._frame_bridge(),
                self._message_reader(),
            )
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            self._running = False
            await self._pipeline.unregister_ws_client(self._client_id)
            heartbeat_task.cancel()
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Heartbeat ping/pong
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings and check for pong responses."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)

            if not self.health.is_healthy:
                print(
                    f"[harden] client {self._client_id} unhealthy "
                    f"(missed {self.health.missed_pongs} pongs), closing",
                    flush=True,
                )
                break

            # Send ping
            self._ping_sent_at = time.monotonic()
            self.health.total_pings += 1
            try:
                await self._ws.send_json({"type": "ping", "ts": self._ping_sent_at})
            except Exception:
                break

            # Wait for pong (with timeout)
            try:
                await asyncio.wait_for(self._wait_for_pong(), timeout=self._heartbeat_timeout)
            except asyncio.TimeoutError:
                self.health.missed_pongs += 1
                print(
                    f"[harden] client {self._client_id} missed pong "
                    f"({self.health.missed_pongs}/3)",
                    flush=True,
                )

    async def _wait_for_pong(self) -> None:
        """Block until a pong message arrives or timeout."""
        while self._running:
            try:
                data = await asyncio.wait_for(
                    self._ws.receive_json(), timeout=self._heartbeat_timeout,
                )
                if isinstance(data, dict) and data.get("type") == "pong":
                    self.health.total_pongs += 1
                    self.health.missed_pongs = 0
                    self.health.last_pong = time.monotonic()
                    if self._ping_sent_at > 0:
                        self.health.rtt_ms = (
                            (time.monotonic() - self._ping_sent_at) * 1000
                        )
                    return
            except asyncio.TimeoutError:
                raise
            except Exception:
                return

    # ------------------------------------------------------------------
    # Message reader (client → server)
    # ------------------------------------------------------------------

    async def _message_reader(self) -> None:
        """Read incoming client messages (pong, ack, etc.)."""
        while self._running:
            try:
                data = await self._ws.receive_json()
            except (WebSocketDisconnect, asyncio.CancelledError):
                break
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")

            if msg_type == "pong":
                self.health.total_pongs += 1
                self.health.missed_pongs = 0
                self.health.last_pong = time.monotonic()
                if self._ping_sent_at > 0:
                    self.health.rtt_ms = (
                        (time.monotonic() - self._ping_sent_at) * 1000
                    )

            elif msg_type == "request_state":
                # Client explicitly requests fresh reconnection state
                await self._send_reconnection_state()

            elif msg_type == "ack_reconnection":
                # Client confirms it applied the reconnection state
                token_id = data.get("token_id")
                if (
                    self._reconnection_token
                    and token_id == self._reconnection_token.token_id
                ):
                    self._reconnection_token = None  # consumed

    # ------------------------------------------------------------------
    # Frame bridge (server → client)
    # ------------------------------------------------------------------

    async def _frame_bridge(self) -> None:
        """Forward pipeline frames to the WebSocket client."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            if msg is None:
                break

            try:
                await self._ws.send_bytes(msg)
                self.health.total_frames += 1
                self.health.last_frame_sent = time.monotonic()
            except Exception:
                break

    # ------------------------------------------------------------------
    # Reconnection state
    # ------------------------------------------------------------------

    async def _send_reconnection_state(self) -> None:
        """Issue a reconnection state token and send it to the client."""
        snapshot = self._pipeline._metrics.snapshot() if hasattr(self._pipeline, '_metrics') else {}
        last_frame = snapshot.get("frames_consumed", 0)

        self._reconnection_token = ReconnectionToken(
            token_id=uuid.uuid4().hex[:12],
            issued_at=time.monotonic(),
            active_tracks=self._pipeline._current_active.__len__() if hasattr(self._pipeline, '_current_active') else 0,
            last_frame_id=last_frame,
            track_state={
                cid: {
                    "prediction": self._pipeline._last_predictions.get(cid, "NORMAL"),
                }
                for cid in (
                    self._pipeline._current_active
                    if hasattr(self._pipeline, '_current_active')
                    else set()
                )
            },
        )

        try:
            await self._ws.send_json(self._reconnection_token.to_dict())
        except Exception:
            pass
