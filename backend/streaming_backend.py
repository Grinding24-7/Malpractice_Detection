"""
streaming_backend.py — Async video frame generator and queue pipeline.

Architecture:
    Producer-consumer pattern using asyncio.Queue.

    Producer (sync thread): reads frames from OpenCV VideoCapture,
    runs ByteTrack inference at 5 FPS (sub-sampled), and pushes
    annotated frames + telemetry payloads into an asyncio.Queue.

    Consumer (async): yields frames from the queue to WebSocket /
    MJPEG clients, implementing backpressure via frame-skipping
    when the queue depth exceeds a threshold.

Integration:
    Reuses existing detector.py (PoseDetector), temporal_features.py
    (PoseWindowManager, HeuristicBaseline, TemporalFeatureExtractor),
    and feature_extractor.py (extract_normalized_pose_features).
"""

from __future__ import annotations

import asyncio
import base64
import time
import threading
from collections import defaultdict, deque
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
VIDEO_SOURCE = __import__("os").environ.get(
    "VIDEO_SOURCE", str(BACKEND_DIR / "sample_exam.mp4")
)
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
SUBSAMPLE = 6  # every 6th frame → 5 FPS AI inference
ANOMALY_THRESHOLD = 10
SEQUENCE_LEN = 30

HEAD_TURN_EAR_RATIO_MIN = 0.70
HEAD_TURN_EAR_RATIO_MAX = 1.40
PITCH_LEAN_NORM_DROP = 0.90

LABEL_NAMES = {0: "Normal", 1: "Head Turn", 2: "Leaning", 3: "Passing"}

# Queue config: max depth before frame-skipping kicks in.
MAX_QUEUE_DEPTH = 4
# Target WebSocket FPS (matched to browser refresh rate).
TARGET_WS_FPS = 30

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ---------------------------------------------------------------------------
# Telemetry payload (sent over WebSocket)
# ---------------------------------------------------------------------------
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
class FrameTelemetry:
    frame_id: int
    timestamp: float
    active_tracks: int
    students: list[StudentTelemetry]
    annotated_frame: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------------------
class StreamingEngine:
    """
    Async video frame generator with producer-consumer queue pipeline.

    The producer runs in a dedicated daemon thread, reading frames from
    OpenCV and pushing FrameTelemetry objects into an asyncio.Queue.
    Consumers (WebSocket/MJPEG handlers) pull from the queue with
    backpressure-aware frame-skipping.
    """

    def __init__(
        self,
        video_source: str | Path = VIDEO_SOURCE,
        target_fps: int = TARGET_WS_FPS,
        max_queue: int = MAX_QUEUE_DEPTH,
    ) -> None:
        self.video_source = str(video_source)
        self.target_fps = target_fps
        self.max_queue = max_queue
        self._queue: asyncio.Queue[FrameTelemetry | None] = asyncio.Queue(
            maxsize=max_queue * 2
        )
        self._detector: Optional[PoseDetector] = None
        self._pose_windows = PoseWindowManager(window_size=SEQUENCE_LEN)
        self._temporal_extractor = TemporalFeatureExtractor(window_size=SEQUENCE_LEN)
        self._temporal_baseline = HeuristicBaseline()
        self._frame_counter = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-candidate state (mirrors app.py pattern)
        self._candidate_features: dict[int, dict] = {}
        self._candidate_cheating: dict[int, bool] = {}
        self._candidate_body: dict[int, dict] = {}
        self._current_active: set[int] = set()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the producer thread. Must be called from the async event loop."""
        if self._running:
            return
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._produce_frames, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the producer to stop and push a sentinel."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, None
            )

    async def consume(self) -> Optional[FrameTelemetry]:
        """
        Pull the next frame from the queue with backpressure handling.

        If the queue is deeper than max_queue, skips intermediate frames
        to prevent memory buildup and latency drift.
        """
        # Backpressure: skip frames if queue is backing up
        while self._queue.qsize() > self.max_queue:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None

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
        """Draw ByteTrack boxes, COCO-17 skeletons, and status tags."""
        annotated = frame.copy()
        for idx in range(len(result.tracker_ids)):
            cid = int(result.tracker_ids[idx])
            kpts = result.keypoints[idx]
            box = result.boxes[idx]
            pred = predictions.get(cid, "NORMAL")

            x1, y1, x2, y2 = (int(v) for v in box)
            if pred == "NORMAL":
                color = (34, 197, 94)   # #22c55e green
            elif pred == "HEAD_TURNING":
                color = (245, 158, 11)  # #f59e0b amber
            else:
                color = (239, 68, 68)   # #ef4444 red

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

    def _produce_frames(self) -> None:
        """Producer thread: read frames, run inference, push to queue."""
        if self._detector is None:
            self._detector = PoseDetector()

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[streaming] ERROR: cannot open {self.video_source}", flush=True)
            self._running = False
            return

        start_time = time.monotonic()
        frames_read = 0
        frame_interval = 1.0 / CAM_FPS

        try:
            while self._running:
                loop_start = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frames_read += 1
                self._frame_counter += 1
                process_this = (self._frame_counter % SUBSAMPLE) == 0

                predictions: dict[int, str] = {}
                students: list[StudentTelemetry] = []

                if process_this:
                    result = self._detector.track(frame)
                    self._current_active = set(int(i) for i in result.tracker_ids.tolist())

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
                        xy = normalize_keypoints(person_kpts, frame.shape[1], frame.shape[0])
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

                    annotated_frame = self._draw_overlay(frame, result, predictions)
                else:
                    annotated_frame = frame

                elapsed = time.monotonic() - start_time
                telemetry = FrameTelemetry(
                    frame_id=self._frame_counter,
                    timestamp=round(elapsed, 3),
                    active_tracks=len(students),
                    students=students,
                    annotated_frame=annotated_frame,
                )

                # Non-blocking push; drop frame if queue full (backpressure)
                try:
                    self._queue.put_nowait(telemetry)
                except asyncio.QueueFull:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self._queue.put_nowait(telemetry)
                    except asyncio.QueueFull:
                        pass

                # Frame pacing: sleep to maintain target FPS
                elapsed_frame = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed_frame
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()
            print("[streaming] producer thread stopped", flush=True)


# ---------------------------------------------------------------------------
# Singleton for app-wide access
# ---------------------------------------------------------------------------
_engine: Optional[StreamingEngine] = None


def get_streaming_engine() -> StreamingEngine:
    global _engine
    if _engine is None:
        _engine = StreamingEngine()
    return _engine
