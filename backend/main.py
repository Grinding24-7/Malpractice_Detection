"""
main.py — Orchestrator pipeline loop.

Architecture:
    - VideoStreamReader (thread):  reads frames at 30 FPS, pushes to
      CircularBuffer, and passes a copy to detector_queue.
    - DetectorRunner (thread):  pulls frames from detector_queue, runs
      PoseDetector.process() at 5 FPS, evaluates heuristics, logs JSON
      alerts, and triggers ExporterThread jobs.
    - ExporterThread:  daemon thread consuming ExportJobs (instantiated
      in exporter.py).

Constraints enforced:
    - Non-blocking:  video reader never waits for inference.
    - Low-latency:   detector_queue bounded to 2 frames (back-pressure).
    - Memory safety: gc.collect() after each export; @torch.no_grad.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from buffer import CircularBuffer
from detector import PoseDetector, InferenceResult
from exporter import ExporterThread, ExportJob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_SOURCE: str = os.environ.get("VIDEO_SOURCE", "sample_exam.mp4")  # 0 = webcam, or path via env var
INPUT_FPS: int = 30
DETECTOR_QUEUE_MAXSIZE: int = 2  # back-pressure to avoid unbounded memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Alert schema
# ---------------------------------------------------------------------------
ALERT_SEVERITY_MAP = {
    "head_down": "low",
    "body_turn": "medium",
    "excessive_lean": "medium",
    "multi_person": "high",
}


def build_alert(
    alert_id: str, flags: dict[str, bool], inference_ts: float
) -> dict:
    active = [k for k, v in flags.items() if v]
    severities = [ALERT_SEVERITY_MAP.get(k, "info") for k in active]
    return {
        "alert_id": alert_id,
        "timestamp": inference_ts,
        "anomalies": active,
        "severities": severities,
        "max_severity": max(severities)
        if severities
        else "none",
    }


# ---------------------------------------------------------------------------
# Thread: Video Stream Reader
# ---------------------------------------------------------------------------

@dataclass
class VideoStreamReader:
    """
    Reads frames from source at native FPS.
    Pushes to CircularBuffer; forwards copy to detector_queue.

    Detector queue is bounded (size 2) so the reader is never blocked
    by slow inference — if the queue is full, frames are dropped.
    """

    source: str = VIDEO_SOURCE
    buffer: CircularBuffer = field(default_factory=CircularBuffer)
    detector_queue: "queue.Queue[np.ndarray]" = field(default_factory=lambda: __import__("queue").Queue(maxsize=DETECTOR_QUEUE_MAXSIZE))  # type: ignore
    _cap: Optional[cv2.VideoCapture] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name="VideoReader")
        t.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        logger.info("Video reader started: %s", self.source)
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()  # type: ignore
            if not ret:
                logger.info("Video stream ended (%s).", self.source)
                break

            # 1. Write every frame to the RAM buffer (30 FPS ingest).
            self.buffer.push(frame)

            # 2. Forward a *copy* to the detector if queue has room.
            #    Non-blocking: if queue is full, we skip (back-pressure).
            try:
                self.detector_queue.put_nowait(frame.copy())
            except __import__("queue").Full:
                pass

        if self._cap is not None:
            self._cap.release()
        logger.info("Video reader stopped.")


# ---------------------------------------------------------------------------
# Thread: Detector Runner
# ---------------------------------------------------------------------------

@dataclass
class DetectorRunner:
    detector: PoseDetector = field(default_factory=PoseDetector)
    detector_queue: Optional["queue.Queue[np.ndarray]"] = None
    exporter: Optional[ExporterThread] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)

    ALERT_COOLDOWN: float = 5.0  # seconds between same-type alerts
    _last_alert_time: float = 0.0

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name="Detector")
        t.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        logger.info("Detector runner started.")
        while not self._stop_event.is_set():
            if self.detector_queue is None:
                time.sleep(0.1)
                continue

            try:
                frame = self.detector_queue.get(timeout=0.5)
            except __import__("queue").Empty:
                continue

            # Time the inference to verify < 15 ms constraint.
            t0 = time.perf_counter()
            result: Optional[InferenceResult] = self.detector.process(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if result is None:
                continue  # sub-sampled frame, skip

            if elapsed_ms > 15.0:
                logger.warning(
                    "Inference took %.2f ms (exceeds 15 ms budget)", elapsed_ms
                )

            # Evaluate anomaly flags
            active_flags = [k for k, v in result.anomaly_flags.items() if v]
            if not active_flags:
                continue

            # Cooldown check
            now = time.time()
            if now - self._last_alert_time < self.ALERT_COOLDOWN:
                continue
            self._last_alert_time = now

            # Build & log JSON alert
            alert_id = str(uuid.uuid4())[:8]
            alert = build_alert(alert_id, result.anomaly_flags, result.timestamps)
            logger.info("ALERT: %s", json.dumps(alert))

            # Trigger export
            if self.exporter is not None:
                self.exporter.submit(
                    ExportJob(
                        alert_id=alert_id,
                        alert_type="_".join(active_flags),
                        timestamp=result.timestamps,
                    )
                )

        logger.info("Detector runner stopped.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting Intelligent Offline Exam Malpractice Detection System")

    # 1. Shared RAM buffer
    buffer = CircularBuffer()

    # 2. Exporter daemon
    exporter = ExporterThread()
    exporter.attach_buffer(buffer)
    exporter.start()

    # 3. Detector (PoseDetector is lazy-loaded on first process() call)
    detector_queue: "queue.Queue[np.ndarray]" = __import__("queue").Queue(
        maxsize=DETECTOR_QUEUE_MAXSIZE
    )
    detector = PoseDetector()
    detector_runner = DetectorRunner(
        detector=detector,
        detector_queue=detector_queue,
        exporter=exporter,
    )
    detector_runner.start()

    # 4. Video reader
    reader = VideoStreamReader(
        source=VIDEO_SOURCE,
        buffer=buffer,
        detector_queue=detector_queue,
    )
    reader.open()
    reader.start()

    # 5. Keep main thread alive until Ctrl+C
    try:
        while True:
            time.sleep(1.0)
            # Periodically collect garbage to prevent cruft accumulation.
            gc.collect()
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        reader.stop()
        detector_runner.stop()
        exporter.stop()
        logger.info("System halted. Goodbye.")


if __name__ == "__main__":
    main()