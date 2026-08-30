"""
stream_pipeline.py — Week 8: Three-stage async streaming pipeline.

Architecture
============
    Stage 1 (Frame Ingestion):
        Reads RTSP / MP4 / Webcam frames in a dedicated daemon thread.
        Pushes raw frames into ``_q_ingest`` (bounded, maxsize=5).

    Stage 2 (Inference & Tracking):
        Consumes raw frames, runs YOLO11-pose + ByteTrack **every k-th
        frame** (adaptive k, default 3).  Intermediate frames reuse the
        most recent trajectory predictions — zero GPU cost.  Pushes
        annotated ``FramePayload`` into ``_q_broadcast``.

    Stage 3 (WebSocket Broadcast):
        Pulls annotated payloads, encodes adaptive-quality JPEG, and
        distributes to all connected WebSocket clients via a fan-out
        broadcast set.

Backpressure
============
    * Both queues are bounded (maxsize=5).  When the consumer is slower
      than the producer, stale intermediate frames are dropped first
      (they carry no new inference data).
    * JPEG quality adapts: 85 → 60 when broadcast queue depth > 3.

Synthetic Benchmark
===================
    Run ``python stream_pipeline.py`` to execute a load test with
    synthetic frames and N concurrent WebSocket clients.

Integration
===========
    Reuses detector.py (PoseDetector), feature_extractor.py,
    temporal_features.py, and metrics.py from the same package.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import PoseDetector, InferenceResult
from feature_extractor import extract_normalized_pose_features
from temporal_features import (
    HeuristicBaseline,
    PoseWindowManager,
    TemporalFeatureExtractor,
    normalize_keypoints,
)
from metrics import get_metrics, read_gpu_memory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
VIDEO_SOURCE = __import__("os").environ.get(
    "VIDEO_SOURCE", str(BACKEND_DIR / "sample_exam.mp4")
)
CAM_FPS = 30
SEQUENCE_LEN = 30

HEAD_TURN_EAR_RATIO_MIN = 0.70
HEAD_TURN_EAR_RATIO_MAX = 1.40
PITCH_LEAN_NORM_DROP = 0.90

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

LABEL_NAMES = {0: "Normal", 1: "Head Turn", 2: "Leaning", 3: "Passing"}

# Adaptive frame-skip: full inference every k frames.
DEFAULT_INFERENCE_INTERVAL = 3
MIN_INFERENCE_INTERVAL = 1
MAX_INFERENCE_INTERVAL = 8

# Queue bounds
INGEST_QUEUE_MAX = 5
BROADCAST_QUEUE_MAX = 5

# JPEG quality bounds
JPEG_QUALITY_HIGH = 85
JPEG_QUALITY_LOW = 60
JPEG_QUALITY_THRESHOLD = 3  # queue depth above which we reduce quality


# ---------------------------------------------------------------------------
# Data payloads flowing through the pipeline
# ---------------------------------------------------------------------------

@dataclass
class RawFrame:
    """Stage 1 → Stage 2: raw frame from video source."""
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class StudentTelemetry:
    track_id: int
    bbox: list[float]
    prediction: str
    confidence: float
    keypoints: list[list[float]]
    velocity_spike: float
    ear_ratio: float
    norm_vertical_drop: float


@dataclass
class FramePayload:
    """Stage 2 → Stage 3: annotated frame + telemetry."""
    frame_id: int
    timestamp: float
    active_tracks: int
    students: list[StudentTelemetry]
    annotated_frame: np.ndarray
    inference_frame: bool  # True if this frame ran full inference


# ---------------------------------------------------------------------------
# Three-Stage Streaming Pipeline
# ---------------------------------------------------------------------------

class StreamPipeline:
    """
    Non-blocking async pipeline with three decoupled stages.

    Stages communicate exclusively through bounded ``asyncio.Queue``
    instances.  Each stage runs as an independent asyncio task.
    """

    def __init__(
        self,
        video_source: str | Path = VIDEO_SOURCE,
        inference_interval: int = DEFAULT_INFERENCE_INTERVAL,
    ) -> None:
        self.video_source = str(video_source)
        self._inference_interval = inference_interval

        # --- Queues (bounded for backpressure) ---
        self._q_ingest: asyncio.Queue[RawFrame | None] = asyncio.Queue(
            maxsize=INGEST_QUEUE_MAX,
        )
        self._q_broadcast: asyncio.Queue[FramePayload | None] = asyncio.Queue(
            maxsize=BROADCAST_QUEUE_MAX,
        )

        # --- State ---
        self._detector: Optional[PoseDetector] = None
        self._pose_windows = PoseWindowManager(window_size=SEQUENCE_LEN)
        self._temporal_extractor = TemporalFeatureExtractor(window_size=SEQUENCE_LEN)
        self._temporal_baseline = HeuristicBaseline()

        self._candidate_features: dict[int, dict] = {}
        self._candidate_cheating: dict[int, bool] = {}
        self._candidate_body: dict[int, dict] = {}
        self._current_active: set[int] = set()

        # Last full inference result (for intermediate-frame reuse)
        self._last_predictions: dict[int, str] = {}
        self._last_students: list[StudentTelemetry] = []

        # Adaptive JPEG quality
        self._jpeg_quality = JPEG_QUALITY_HIGH

        # Tasks
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._producer_thread: Optional[threading.Thread] = None
        self._frame_counter = 0

        # WebSocket fan-out
        self._ws_clients: dict[str, asyncio.Queue[bytes]] = {}
        self._ws_lock = asyncio.Lock()

        # Metrics
        self._metrics = get_metrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch all three pipeline stages + broadcast fan-out."""
        if self._running:
            return
        self._running = True

        # Stage 1: frame ingestion (producer thread → async queue bridge)
        self._tasks.append(asyncio.create_task(self._stage1_ingest()))

        # Stage 2: inference & tracking
        self._tasks.append(asyncio.create_task(self._stage2_inference()))

        # Stage 3: WebSocket broadcast consumer
        self._tasks.append(asyncio.create_task(self._stage3_broadcast()))

        # Adaptive quality monitor
        self._tasks.append(asyncio.create_task(self._adaptive_quality_monitor()))

        # GPU memory reporter
        self._tasks.append(asyncio.create_task(self._gpu_reporter()))

        print("[pipeline] all stages started", flush=True)

    async def stop(self) -> None:
        """Gracefully shut down all stages."""
        self._running = False

        # Sentinel for stage 1 queue
        try:
            self._q_ingest.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Sentinel for broadcast queue
        try:
            self._q_broadcast.put_nowait(None)
        except asyncio.QueueFull:
            pass

        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close all WS client queues
        async with self._ws_lock:
            for q in self._ws_clients.values():
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            self._ws_clients.clear()

        print("[pipeline] all stages stopped", flush=True)

    # ------------------------------------------------------------------
    # Stage 1 — Frame Ingestion Worker
    # ------------------------------------------------------------------

    async def _stage1_ingest(self) -> None:
        """
        Read frames from video source in a daemon thread and bridge
        them into the async ``_q_ingest`` queue.

        The actual I/O (OpenCV VideoCapture) is blocking, so it runs
        in a dedicated thread.  The thread pushes ``RawFrame`` objects
        and the async bridge feeds the queue.
        """
        loop = asyncio.get_event_loop()
        _queue_put = loop.call_soon_threadsafe

        def _reader_thread() -> None:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                print(f"[pipeline:ingest] ERROR: cannot open {self.video_source}", flush=True)
                self._running = False
                return

            frame_interval = 1.0 / CAM_FPS
            start = time.monotonic()

            try:
                while self._running:
                    t0 = time.monotonic()
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    self._frame_counter += 1
                    raw = RawFrame(
                        frame_id=self._frame_counter,
                        timestamp=round(time.monotonic() - start, 3),
                        frame=frame,
                    )

                    # Non-blocking push; drop oldest if full
                    try:
                        _queue_put(self._q_ingest.put_nowait, raw)
                    except asyncio.QueueFull:
                        try:
                            _queue_put(self._q_ingest.get_nowait)
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            _queue_put(self._q_ingest.put_nowait, raw)
                        except asyncio.QueueFull:
                            self._metrics.inc_frames_dropped()

                    self._metrics.inc_frames_produced()
                    elapsed = time.monotonic() - t0
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                cap.release()
                print("[pipeline:ingest] reader thread stopped", flush=True)

        self._producer_thread = threading.Thread(target=_reader_thread, daemon=True)
        self._producer_thread.start()

        # Keep the coroutine alive while the thread runs
        while self._running:
            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Stage 2 — Inference & Tracking Worker
    # ------------------------------------------------------------------

    async def _stage2_inference(self) -> None:
        """
        Pull raw frames, run full YOLO11-pose + ByteTrack every k-th
        frame, reuse trajectory predictions for intermediate frames.

        Adaptive k: increases when GPU is saturated, decreases when
        idle, clamped to [MIN_INFERENCE_INTERVAL, MAX_INFERENCE_INTERVAL].
        """
        if self._detector is None:
            self._detector = PoseDetector()

        k = self._inference_interval
        consecutive_inference_times: deque[float] = deque(maxlen=20)

        while self._running:
            try:
                raw: RawFrame | None = await asyncio.wait_for(
                    self._q_ingest.get(), timeout=1.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            if raw is None:
                # Sentinel: propagate and stop
                try:
                    self._q_broadcast.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                break

            is_inference_frame = (raw.frame_id % k) == 0

            if is_inference_frame:
                t0 = time.monotonic()
                result = self._detector.track(raw.frame)
                inference_time = time.monotonic() - t0

                consecutive_inference_times.append(inference_time)
                self._metrics.inc_inference_runs()
                self._metrics.observe_inference_latency(inference_time)

                # Adaptive k: slow down if inference > 80ms avg
                if len(consecutive_inference_times) >= 5:
                    avg = sum(consecutive_inference_times) / len(consecutive_inference_times)
                    if avg > 0.08 and k < MAX_INFERENCE_INTERVAL:
                        k += 1
                        self._metrics.set_frame_skip_k(k)
                    elif avg < 0.03 and k > MIN_INFERENCE_INTERVAL:
                        k -= 1
                        self._metrics.set_frame_skip_k(k)

                self._current_active = set(
                    int(i) for i in result.tracker_ids.tolist()
                )

                predictions: dict[int, str] = {}
                students: list[StudentTelemetry] = []

                for idx in range(len(result.tracker_ids)):
                    cid = int(result.tracker_ids[idx])
                    person_kpts = result.keypoints[idx]
                    box = result.boxes[idx]
                    conf = (
                        float(result.confidences[idx])
                        if idx < len(result.confidences)
                        else float(person_kpts[:, 2].max()) if person_kpts.size else 1.0
                    )

                    feats = extract_normalized_pose_features(person_kpts)
                    is_anomaly = self._evaluate_candidate(feats)
                    pred = self._classify(feats, is_anomaly)

                    # Temporal pose window
                    xy = normalize_keypoints(
                        person_kpts, raw.frame.shape[1], raw.frame.shape[0],
                    )
                    if xy is not None:
                        self._pose_windows.push(cid, xy)
                        if self._pose_windows.is_ready(cid):
                            seq = self._pose_windows.window(cid)
                            temporal_flags = self._temporal_baseline.evaluate(seq)
                            if temporal_flags["anomalous"]:
                                is_anomaly = True
                                pred = self._classify(feats, True)

                    predictions[cid] = pred
                    self._candidate_cheating[cid] = is_anomaly
                    self._candidate_features[cid] = {
                        "ear_ratio": float(feats[0]),
                        "norm_vertical_drop": float(feats[1]),
                    }

                    # Velocity spike
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
                    students.append(StudentTelemetry(
                        track_id=cid,
                        bbox=[round(float(v), 1) for v in box],
                        prediction=pred,
                        confidence=round(conf, 3),
                        keypoints=kp_list,
                        velocity_spike=round(vel, 3),
                        ear_ratio=round(float(feats[0]), 4),
                        norm_vertical_drop=round(float(feats[1]), 4),
                    ))

                annotated = self._draw_overlay(raw.frame, result, predictions)

                # Cache for intermediate frames
                self._last_predictions = predictions
                self._last_students = students

            else:
                # Intermediate frame: reuse cached overlay
                annotated = self._draw_cached_overlay(raw.frame)
                predictions = self._last_predictions
                students = self._last_students

            payload = FramePayload(
                frame_id=raw.frame_id,
                timestamp=raw.timestamp,
                active_tracks=len(students),
                students=students,
                annotated_frame=annotated,
                inference_frame=is_inference_frame,
            )

            # Non-blocking push; drop stale frames
            try:
                self._q_broadcast.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    stale = self._q_broadcast.get_nowait()
                    self._metrics.inc_frames_dropped()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._q_broadcast.put_nowait(payload)
                except asyncio.QueueFull:
                    self._metrics.inc_frames_dropped()

    # ------------------------------------------------------------------
    # Stage 3 — WebSocket Broadcast Consumer
    # ------------------------------------------------------------------

    async def _stage3_broadcast(self) -> None:
        """
        Pull annotated payloads from the broadcast queue, encode
        adaptive-quality JPEG, and fan out to all connected WS clients.
        """
        while self._running:
            try:
                payload: FramePayload | None = await asyncio.wait_for(
                    self._q_broadcast.get(), timeout=1.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            if payload is None:
                break

            # Encode JPEG with adaptive quality
            ret, jpeg = cv2.imencode(
                ".jpg",
                payload.annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if not ret:
                continue

            jpeg_bytes = jpeg.tobytes()

            # Build telemetry (without frame_jpeg — sent separately)
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
                for s in payload.students
            ]

            header = json.dumps({
                "frame_id": payload.frame_id,
                "timestamp": payload.timestamp,
                "active_tracks": payload.active_tracks,
                "students": students_data,
                "inference_frame": payload.inference_frame,
            }).encode("ascii")

            # Wire format: 4-byte big-endian header length + header + jpeg
            msg = len(header).to_bytes(4, "big") + header + jpeg_bytes

            # Fan-out to all clients
            async with self._ws_lock:
                dead: list[str] = []
                for cid, q in self._ws_clients.items():
                    try:
                        q.put_nowait(msg)
                    except asyncio.QueueFull:
                        # Drop oldest frame in client buffer
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            q.put_nowait(msg)
                        except asyncio.QueueFull:
                            dead.append(cid)
                for cid in dead:
                    del self._ws_clients[cid]

            self._metrics.inc_frames_consumed()
            self._metrics.set_queue_depth(self._q_broadcast.qsize())

    # ------------------------------------------------------------------
    # Adaptive quality monitor
    # ------------------------------------------------------------------

    async def _adaptive_quality_monitor(self) -> None:
        """Periodically adjust JPEG quality based on broadcast queue depth."""
        while self._running:
            await asyncio.sleep(0.5)
            depth = self._q_broadcast.qsize()
            if depth > JPEG_QUALITY_THRESHOLD:
                self._jpeg_quality = max(
                    JPEG_QUALITY_LOW,
                    self._jpeg_quality - 2,
                )
            else:
                self._jpeg_quality = min(
                    JPEG_QUALITY_HIGH,
                    self._jpeg_quality + 1,
                )
            self._metrics.set_jpeg_quality(self._jpeg_quality)

    # ------------------------------------------------------------------
    # GPU memory reporter
    # ------------------------------------------------------------------

    async def _gpu_reporter(self) -> None:
        """Push GPU memory stats to metrics every 2 seconds."""
        while self._running:
            await asyncio.sleep(2.0)
            alloc, reserved = read_gpu_memory()
            self._metrics.set_gpu_memory(alloc, reserved)

    # ------------------------------------------------------------------
    # WebSocket client registration
    # ------------------------------------------------------------------

    async def register_ws_client(self) -> tuple[str, asyncio.Queue[bytes]]:
        """Register a new WS client; returns (client_id, receive_queue)."""
        cid = uuid.uuid4().hex[:8]
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=3)
        async with self._ws_lock:
            self._ws_clients[cid] = q
        self._metrics.set_active_connections(len(self._ws_clients))
        return cid, q

    async def unregister_ws_client(self, client_id: str) -> None:
        async with self._ws_lock:
            self._ws_clients.pop(client_id, None)
        self._metrics.set_active_connections(len(self._ws_clients))

    # ------------------------------------------------------------------
    # Inference helpers (ported from StreamingEngine)
    # ------------------------------------------------------------------

    def _evaluate_candidate(self, features: np.ndarray) -> bool:
        ear_ratio = float(features[0])
        norm_vertical_drop = float(features[1])
        nose_conf = float(features[4])
        l_ear_conf = float(features[5])
        r_ear_conf = float(features[6])
        if nose_conf < 0.35 or l_ear_conf < 0.2 or r_ear_conf < 0.2:
            return False
        if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
            return True
        if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
            return True
        return False

    def _classify(self, features: np.ndarray, is_anomaly: bool) -> str:
        if not is_anomaly:
            return "NORMAL"
        ear_ratio = float(features[0])
        norm_vertical_drop = float(features[1])
        if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
            return "HEAD_TURNING"
        if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
            return "PEEKING"
        return "SUSPICIOUS"

    def _draw_overlay(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        predictions: dict[int, str],
    ) -> np.ndarray:
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

    def _draw_cached_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw cached bounding boxes on an intermediate (non-inference) frame."""
        annotated = frame.copy()
        for cid, pred in self._last_predictions.items():
            # Find matching student for bbox
            for s in self._last_students:
                if s.track_id == cid:
                    x1, y1, x2, y2 = (int(v) for v in s.bbox)
                    if pred == "NORMAL":
                        color = (34, 197, 94)
                    elif pred == "HEAD_TURNING":
                        color = (245, 158, 11)
                    else:
                        color = (239, 68, 68)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                    tag = f"ID:{cid}"
                    cv2.putText(
                        annotated, tag, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    )
                    break
        return annotated


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pipeline: Optional[StreamPipeline] = None


def get_stream_pipeline() -> StreamPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = StreamPipeline()
    return _pipeline


# ===================================================================
# Synthetic Benchmark — run with: python stream_pipeline.py
# ===================================================================

async def _benchmark() -> None:
    """
    Simulate N concurrent WebSocket clients consuming from the pipeline
    under synthetic load.  Measures throughput, latency, and drop rate.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Week 8 pipeline benchmark")
    parser.add_argument("--clients", type=int, default=4, help="Simulated WS clients")
    parser.add_argument("--duration", type=int, default=10, help="Benchmark duration (s)")
    parser.add_argument("--source", type=str, default=VIDEO_SOURCE, help="Video source")
    args = parser.parse_args()

    pipeline = StreamPipeline(
        video_source=args.source,
        inference_interval=DEFAULT_INFERENCE_INTERVAL,
    )

    print(f"\n{'='*60}")
    print(f"  Week 8 Pipeline Benchmark")
    print(f"  Clients: {args.clients}  |  Duration: {args.duration}s")
    print(f"{'='*60}\n")

    await pipeline.start()
    await asyncio.sleep(1)  # let pipeline warm up

    # Register simulated clients
    clients: list[tuple[str, asyncio.Queue[bytes]]] = []
    for _ in range(args.clients):
        cid, q = await pipeline.register_ws_client()
        clients.append((cid, q))

    # Consumer tasks
    stats = {
        "total_frames": 0,
        "total_bytes": 0,
        "latencies": [],
        "start": time.monotonic(),
    }

    async def _consumer(cid: str, q: asyncio.Queue[bytes]) -> None:
        while time.monotonic() - stats["start"] < args.duration:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue
            if msg is None:
                break
            stats["total_frames"] += 1
            stats["total_bytes"] += len(msg)

    tasks = [
        asyncio.create_task(_consumer(cid, q))
        for cid, q in clients
    ]

    await asyncio.sleep(args.duration)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await pipeline.stop()

    elapsed = time.monotonic() - stats["start"]
    m = pipeline._metrics.snapshot()

    print(f"\n{'='*60}")
    print(f"  Benchmark Results")
    print(f"{'='*60}")
    print(f"  Duration:           {elapsed:.1f}s")
    print(f"  Clients:            {args.clients}")
    print(f"  Total frames:       {stats['total_frames']}")
    print(f"  Throughput:         {stats['total_frames']/elapsed:.1f} frames/s")
    print(f"  Data transferred:   {stats['total_bytes']/1024/1024:.2f} MB")
    print(f"  Producer FPS:       {m['producer_fps']}")
    print(f"  Consumer FPS:       {m['consumer_fps']}")
    print(f"  Frames produced:    {m['frames_produced']}")
    print(f"  Frames consumed:    {m['frames_consumed']}")
    print(f"  Frames dropped:     {m['frames_dropped']}")
    print(f"  Inference runs:     {m['inference_runs']}")
    print(f"  JPEG quality:       {m['jpeg_quality']}")
    print(f"  Frame skip k:       {m['frame_skip_k']}")
    print(f"  GPU memory (MB):    {m['gpu_memory_mb']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(_benchmark())
