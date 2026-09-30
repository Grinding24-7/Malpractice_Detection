"""
system_integrator.py — Week 9: Unified Pipeline Orchestrator.

Central ``ExamSurveillanceEngine`` class that binds all system modules
into a single managed lifecycle with error-boundary isolation.

Subsystems managed:
    1. Frame Ingestion (video source → raw frames)
    2. Pose Tracking (YOLO11-pose + ByteTrack)
    3. Feature Buffers (per-track ring buffers)
    4. Temporal Inference (LSTM/GRU sequence classifier)
    5. Evidence Archiving (pre-roll + post-roll MP4 export)
    6. Storage Purge (disk pressure + retention daemon)
    7. WebSocket Broadcasting (adaptive JPEG + fan-out)

Error Boundary:
    Each subsystem runs inside an ``ErrorBoundary`` wrapper.  An
    exception in one candidate track or subsystem is logged and
    isolated — the core engine continues processing.

Usage:
    engine = ExamSurveillanceEngine(video_source="sample_exam.mp4")
    await engine.start()
    ...
    status = engine.status()
    await engine.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import PoseDetector, InferenceResult
from feature_extractor import extract_normalized_pose_features
from buffer_manager import BufferManager, get_buffer_manager
from evidence_archiver import EvidenceArchiver, get_evidence_archiver
from storage_purge import StoragePurgeDaemon, get_purge_daemon
from temporal_features import (
    HeuristicBaseline,
    PoseWindowManager,
    TemporalFeatureExtractor,
    normalize_keypoints,
)
from metrics import get_metrics, read_gpu_memory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("system_integrator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
VIDEO_SOURCE = __import__("os").environ.get(
    "VIDEO_SOURCE", str(BACKEND_DIR / "sample_exam.mp4")
)
CAM_FPS = 30
SEQUENCE_LEN = 30
INFERENCE_INTERVAL = 3  # full inference every k frames

HEAD_TURN_EAR_RATIO_MIN = 0.70
HEAD_TURN_EAR_RATIO_MAX = 1.40
PITCH_LEAN_NORM_DROP = 0.90

# Anomaly must persist this many frames within a 30-frame window
# to trigger an evidence export (reduces false positives).
ANOMALY_PERSISTENCE_THRESHOLD = 12

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ---------------------------------------------------------------------------
# Engine states
# ---------------------------------------------------------------------------

class EngineState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Error boundary
# ---------------------------------------------------------------------------

class ErrorBoundary:
    """
    Wraps a callable so that exceptions are caught, logged, and counted
    instead of propagating up and crashing the engine.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.last_error_time: float = 0.0

    def __call__(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_error_time = time.monotonic()
            logger.error(
                f"[boundary:{self.name}] error #{self.error_count}: {exc}",
                exc_info=True,
            )
            return None

    @property
    def healthy(self) -> bool:
        if self.error_count == 0:
            return True
        # Allow recovery: if no error in last 30s, consider healthy
        return (time.monotonic() - self.last_error_time) > 30.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "healthy": self.healthy,
        }


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Per-candidate tracking + classification state."""
    track_id: int
    anomaly_frames: int = 0       # frames with anomaly in sliding window
    total_frames: int = 0         # total frames seen
    last_prediction: str = "NORMAL"
    last_confidence: float = 0.0
    last_features: Optional[np.ndarray] = None
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    evidence_triggered: bool = False

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.first_seen

    @property
    def persistence_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.anomaly_frames / self.total_frames


# ---------------------------------------------------------------------------
# ExamSurveillanceEngine
# ---------------------------------------------------------------------------

class ExamSurveillanceEngine:
    """
    Unified orchestrator binding all subsystems with error isolation.

    Lifecycle:
        1. ``start()`` — initialises subsystems, launches pipeline tasks
        2. ``status()`` — returns consolidated health + telemetry
        3. ``stop()`` — graceful shutdown of all tasks and threads
    """

    def __init__(
        self,
        video_source: str | Path = VIDEO_SOURCE,
        inference_interval: int = INFERENCE_INTERVAL,
        max_tracks: int = 64,
        enable_evidence: bool = True,
        enable_purge: bool = True,
    ) -> None:
        self.video_source = str(video_source)
        self.inference_interval = inference_interval
        self.max_tracks = max_tracks
        self.enable_evidence = enable_evidence
        self.enable_purge = enable_purge

        # --- Error boundaries (one per subsystem) ---
        self._boundaries: dict[str, ErrorBoundary] = {
            "ingestion": ErrorBoundary("ingestion"),
            "tracking": ErrorBoundary("tracking"),
            "features": ErrorBoundary("features"),
            "inference": ErrorBoundary("inference"),
            "evidence": ErrorBoundary("evidence"),
            "broadcast": ErrorBoundary("broadcast"),
            "purge": ErrorBoundary("purge"),
        }

        # --- Subsystem references (lazy-init) ---
        self._detector: Optional[PoseDetector] = None
        self._pose_windows = PoseWindowManager(window_size=SEQUENCE_LEN)
        self._temporal_extractor = TemporalFeatureExtractor(window_size=SEQUENCE_LEN)
        self._temporal_baseline = HeuristicBaseline()
        self._buffer_manager: Optional[BufferManager] = None
        self._archiver: Optional[EvidenceArchiver] = None
        self._purge_daemon: Optional[StoragePurgeDaemon] = None
        self._metrics = get_metrics()

        # --- Per-track state ---
        self._tracks: dict[int, TrackState] = {}

        # --- Queues ---
        self._q_frames: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(maxsize=5)
        self._q_broadcast: asyncio.Queue[Optional[dict]] = asyncio.Queue(maxsize=10)

        # --- Tasks ---
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._state = EngineState.IDLE

        # --- Telemetry ---
        self._start_time: float = 0.0
        self._frame_counter = 0
        self._total_inference_runs = 0
        self._total_evidence_exports = 0

        # --- WebSocket fan-out ---
        self._ws_clients: dict[str, asyncio.Queue[bytes]] = {}
        self._ws_lock = asyncio.Lock()

        # --- Candidate body state (for velocity spike) ---
        self._candidate_body: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all subsystems and pipeline tasks."""
        if self._running:
            return

        logger.info("[engine] starting ExamSurveillanceEngine...")
        self._start_time = time.monotonic()
        self._running = True
        self._state = EngineState.RUNNING

        # Init subsystems
        self._detector = PoseDetector()
        self._buffer_manager = get_buffer_manager()

        if self.enable_evidence:
            self._archiver = get_evidence_archiver()
            self._archiver.start()

        if self.enable_purge:
            self._purge_daemon = get_purge_daemon()
            self._purge_daemon.start()

        # Launch pipeline tasks
        self._tasks.append(asyncio.create_task(self._stage_ingest()))
        self._tasks.append(asyncio.create_task(self._stage_process()))
        self._tasks.append(asyncio.create_task(self._stage_broadcast()))
        self._tasks.append(asyncio.create_task(self._health_monitor()))

        logger.info("[engine] all subsystems started")

    async def stop(self) -> None:
        """Graceful shutdown of all subsystems and tasks."""
        if not self._running:
            return

        logger.info("[engine] shutting down...")
        self._running = False
        self._state = EngineState.STOPPED

        # Cancel pipeline tasks
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Stop subsystems in reverse order
        if self._purge_daemon:
            self._purge_daemon.stop()
        if self._archiver:
            self._archiver.stop()

        # Close WS clients
        async with self._ws_lock:
            for q in self._ws_clients.values():
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            self._ws_clients.clear()

        logger.info("[engine] all subsystems stopped")

    # ------------------------------------------------------------------
    # Stage 1: Frame Ingestion
    # ------------------------------------------------------------------

    async def _stage_ingest(self) -> None:
        """Read frames from video source in a daemon thread, push to queue."""
        loop = asyncio.get_event_loop()

        def _reader() -> None:
            boundary = self._boundaries["ingestion"]
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                logger.error(f"[ingestion] cannot open {self.video_source}")
                self._running = False
                return

            interval = 1.0 / CAM_FPS
            start = time.monotonic()

            try:
                while self._running:
                    t0 = time.monotonic()
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    self._frame_counter += 1
                    self._metrics.inc_frames_produced()

                    # Non-blocking push
                    try:
                        loop.call_soon_threadsafe(self._q_frames.put_nowait, frame)
                    except asyncio.QueueFull:
                        try:
                            loop.call_soon_threadsafe(self._q_frames.get_nowait)
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            loop.call_soon_threadsafe(self._q_frames.put_nowait, frame)
                        except asyncio.QueueFull:
                            self._metrics.inc_frames_dropped()

                    elapsed = time.monotonic() - t0
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                cap.release()

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        while self._running:
            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Stage 2: Processing (Tracking + Features + Inference)
    # ------------------------------------------------------------------

    async def _stage_process(self) -> None:
        """Pull frames, run tracking, extract features, classify."""
        k = self.inference_interval

        while self._running:
            try:
                frame: Optional[np.ndarray] = await asyncio.wait_for(
                    self._q_frames.get(), timeout=1.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            if frame is None:
                break

            is_inference_frame = (self._frame_counter % k) == 0

            # --- Tracking (with error boundary) ---
            result: Optional[InferenceResult] = None
            if is_inference_frame:
                result = self._boundaries["tracking"](
                    self._detector.track, frame
                )
                if result is None:
                    self._state = EngineState.DEGRADED
                    continue
                self._total_inference_runs += 1
                self._metrics.inc_inference_runs()

            # --- Feature extraction + classification per track ---
            predictions: dict[int, str] = {}
            students: list[dict] = []

            if result is not None:
                active_ids = set(int(i) for i in result.tracker_ids.tolist())

                for idx in range(len(result.tracker_ids)):
                    cid = int(result.tracker_ids[idx])
                    person_kpts = result.keypoints[idx]
                    box = result.boxes[idx]

                    # --- Feature extraction (with error boundary) ---
                    feats = self._boundaries["features"](
                        extract_normalized_pose_features, person_kpts
                    )
                    if feats is None:
                        continue

                    # --- Classification (with error boundary) ---
                    pred = self._boundaries["inference"](
                        self._classify_track, cid, feats, frame, person_kpts
                    ) or "NORMAL"

                    predictions[cid] = pred

                    # --- Update track state ---
                    track = self._tracks.get(cid)
                    if track is None:
                        track = TrackState(track_id=cid)
                        self._tracks[cid] = track

                    track.total_frames += 1
                    track.last_seen = time.monotonic()
                    track.last_prediction = pred
                    track.last_confidence = float(feats[4])  # nose_conf
                    track.last_features = feats

                    is_anomaly = pred != "NORMAL"
                    if is_anomaly:
                        track.anomaly_frames += 1
                    else:
                        track.anomaly_frames = max(0, track.anomaly_frames - 1)

                    # --- Velocity spike ---
                    nose = person_kpts[0]
                    shoulder_w = np.hypot(
                        person_kpts[5][0] - person_kpts[6][0],
                        person_kpts[5][1] - person_kpts[6][1],
                    ) or 40.0
                    prev = self._candidate_body.get(cid)
                    if prev is not None:
                        dist = np.hypot(
                            nose[0] - prev["nose"][0],
                            nose[1] - prev["nose"][1],
                        )
                        vel = min(dist / max(shoulder_w, 1e-3), 2.0)
                    else:
                        vel = 0.0
                    self._candidate_body[cid] = {
                        "nose": (float(nose[0]), float(nose[1])),
                        "shoulder_w": float(shoulder_w),
                    }

                    kp_list = [
                        [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                        for x, y, c in person_kpts
                    ]

                    students.append({
                        "track_id": cid,
                        "bbox": [round(float(v), 1) for v in box],
                        "prediction": pred,
                        "confidence": round(float(feats[4]), 3),
                        "keypoints": kp_list,
                        "velocity_spike": round(vel, 3),
                        "ear_ratio": round(float(feats[0]), 4),
                        "norm_vertical_drop": round(float(feats[1], ), 4),
                    })

                    # --- Push to buffer manager ---
                    if self._buffer_manager is not None:
                        self._boundaries["features"](
                            self._buffer_manager.push_frame,
                            cid, frame, self._frame_counter, kp_list,
                        )

                    # --- Evidence export (with persistence check) ---
                    if (
                        self._archiver is not None
                        and is_anomaly
                        and not track.evidence_triggered
                        and track.anomaly_frames >= ANOMALY_PERSISTENCE_THRESHOLD
                    ):
                        self._boundaries["evidence"](
                            self._trigger_evidence, cid, pred, track
                        )

                # --- Prune stale tracks ---
                stale_ids = [
                    tid for tid, t in self._tracks.items()
                    if tid not in active_ids
                    and (time.monotonic() - t.last_seen) > 5.0
                ]
                for tid in stale_ids:
                    del self._tracks[tid]
                    self._candidate_body.pop(tid, None)
                self._pose_windows.prune(active_ids)

                if self._buffer_manager is not None:
                    self._buffer_manager.prune_stale()

            # --- Broadcast telemetry ---
            telemetry = {
                "frame_id": self._frame_counter,
                "timestamp": round(time.monotonic() - self._start_time, 3),
                "active_tracks": len(students),
                "students": students,
                "engine_state": self._state.value,
            }

            try:
                self._q_broadcast.put_nowait(telemetry)
            except asyncio.QueueFull:
                try:
                    self._q_broadcast.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._q_broadcast.put_nowait(telemetry)
                except asyncio.QueueFull:
                    pass

    # ------------------------------------------------------------------
    # Stage 3: WebSocket Broadcast
    # ------------------------------------------------------------------

    async def _stage_broadcast(self) -> None:
        """Fan-out telemetry to all connected WS clients."""
        while self._running:
            try:
                telemetry: Optional[dict] = await asyncio.wait_for(
                    self._q_broadcast.get(), timeout=1.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            if telemetry is None:
                break

            payload = self._boundaries["broadcast"](
                self._encode_broadcast_payload, telemetry
            )
            if payload is None:
                continue

            async with self._ws_lock:
                dead: list[str] = []
                for cid, q in self._ws_clients.items():
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            q.put_nowait(payload)
                        except asyncio.QueueFull:
                            dead.append(cid)
                for cid in dead:
                    del self._ws_clients[cid]

            self._metrics.inc_frames_consumed()
            self._metrics.set_queue_depth(self._q_broadcast.qsize())

    def _encode_broadcast_payload(self, telemetry: dict) -> bytes:
        """Encode telemetry dict as JSON bytes."""
        import json
        return json.dumps(telemetry, separators=(",", ":")).encode("ascii")

    # ------------------------------------------------------------------
    # Classification (with persistence rules)
    # ------------------------------------------------------------------

    def _classify_track(
        self,
        track_id: int,
        features: np.ndarray,
        frame: np.ndarray,
        person_kpts: np.ndarray,
    ) -> str:
        """Classify a single candidate's behavior."""
        ear_ratio = float(features[0])
        norm_vertical_drop = float(features[1])
        nose_conf = float(features[4])
        l_ear_conf = float(features[5])
        r_ear_conf = float(features[6])

        # Low-confidence skip
        if nose_conf < 0.35 or l_ear_conf < 0.2 or r_ear_conf < 0.2:
            return "NORMAL"

        # Evaluate heuristic
        is_anomaly = False
        if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
            is_anomaly = True
        if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
            is_anomaly = True

        # Temporal window
        xy = normalize_keypoints(person_kpts, frame.shape[1], frame.shape[0])
        if xy is not None:
            self._pose_windows.push(track_id, xy)
            if self._pose_windows.is_ready(track_id):
                seq = self._pose_windows.window(track_id)
                temporal_flags = self._temporal_baseline.evaluate(seq)
                if temporal_flags.get("anomalous", False):
                    is_anomaly = True

        if not is_anomaly:
            return "NORMAL"

        # Classify type
        if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
            return "HEAD_TURNING"
        if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
            return "PEEKING"
        return "SUSPICIOUS"

    # ------------------------------------------------------------------
    # Evidence trigger
    # ------------------------------------------------------------------

    def _trigger_evidence(self, track_id: int, pred: str, track: TrackState) -> None:
        """Export evidence clip for sustained anomaly."""
        event_id = self._archiver.trigger_export(
            track_id=track_id,
            malpractice_type=pred,
            confidence=track.last_confidence,
            frame_id=self._frame_counter,
            keypoints=None,
        )
        if event_id:
            track.evidence_triggered = True
            self._total_evidence_exports += 1
            logger.info(
                f"[evidence] triggered for track {track_id}: "
                f"{pred} (conf={track.last_confidence:.3f})"
            )

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor(self) -> None:
        """Periodically update GPU memory metrics and check subsystem health."""
        while self._running:
            await asyncio.sleep(2.0)
            alloc, reserved = read_gpu_memory()
            self._metrics.set_gpu_memory(alloc, reserved)

            # Check if any boundary is unhealthy
            unhealthy = [
                name for name, b in self._boundaries.items()
                if not b.healthy
            ]
            if unhealthy and self._state == EngineState.RUNNING:
                self._state = EngineState.DEGRADED
                logger.warning(f"[engine] degraded: unhealthy boundaries: {unhealthy}")
            elif not unhealthy and self._state == EngineState.DEGRADED:
                self._state = EngineState.RUNNING
                logger.info("[engine] recovered to running state")

    # ------------------------------------------------------------------
    # WebSocket client management
    # ------------------------------------------------------------------

    async def register_ws_client(self) -> tuple[str, asyncio.Queue[bytes]]:
        cid = __import__("uuid").uuid4().hex[:8]
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)
        async with self._ws_lock:
            self._ws_clients[cid] = q
        self._metrics.set_active_connections(len(self._ws_clients))
        return cid, q

    async def unregister_ws_client(self, client_id: str) -> None:
        async with self._ws_lock:
            self._ws_clients.pop(client_id, None)
        self._metrics.set_active_connections(len(self._ws_clients))

    # ------------------------------------------------------------------
    # Status / telemetry
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Consolidated engine status snapshot."""
        now = time.monotonic()
        return {
            "state": self._state.value,
            "uptime_s": round(now - self._start_time, 1) if self._start_time else 0,
            "frame_counter": self._frame_counter,
            "total_inference_runs": self._total_inference_runs,
            "total_evidence_exports": self._total_evidence_exports,
            "active_tracks": len(self._tracks),
            "max_tracks": self.max_tracks,
            "ws_clients": len(self._ws_clients),
            "queue_depth": self._q_broadcast.qsize(),
            "boundaries": {
                name: b.to_dict() for name, b in self._boundaries.items()
            },
            "subsystems": {
                "detector": self._detector is not None,
                "buffer_manager": self._buffer_manager is not None,
                "archiver": self._archiver is not None and self.enable_evidence,
                "purge_daemon": self._purge_daemon is not None and self.enable_purge,
            },
            "metrics": self._metrics.snapshot(),
            "track_details": {
                tid: {
                    "age_s": round(t.age_seconds, 1),
                    "total_frames": t.total_frames,
                    "anomaly_frames": t.anomaly_frames,
                    "persistence_ratio": round(t.persistence_ratio, 3),
                    "last_prediction": t.last_prediction,
                    "evidence_triggered": t.evidence_triggered,
                }
                for tid, t in self._tracks.items()
            },
        }

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        predictions: dict[int, str],
    ) -> np.ndarray:
        """Draw ByteTrack boxes, COCO-17 skeletons, and status tags."""
        annotated = frame.copy()
        for idx in range(len(result.tracker_ids)):
            cid = int(result.tracker_ids[idx])
            kpts = result.keypoints[idx]
            box = result.boxes[idx]
            pred = predictions.get(cid, "NORMAL")

            x1, y1, x2, y2 = (int(v) for v in box)
            if pred == "NORMAL":
                color = (34, 197, 94)
            elif pred == "HEAD_TURNING":
                color = (245, 158, 11)
            else:
                color = (239, 68, 68)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = f"ID:{cid} {pred}"
            cv2.putText(
                annotated, tag, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

            for a, b in SKELETON:
                pa, pb = kpts[a], kpts[b]
                if pa[2] > 0.3 and pb[2] > 0.3:
                    cv2.line(
                        annotated,
                        (int(pa[0]), int(pa[1])),
                        (int(pb[0]), int(pb[1])),
                        (255, 0, 255), 2,
                    )
            for x, y, conf in kpts:
                if conf > 0.3:
                    cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 255), -1)

        return annotated


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[ExamSurveillanceEngine] = None


def get_exam_engine() -> ExamSurveillanceEngine:
    global _engine
    if _engine is None:
        _engine = ExamSurveillanceEngine()
    return _engine
