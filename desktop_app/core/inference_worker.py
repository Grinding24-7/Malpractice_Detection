"""
inference_worker.py — QThread that samples frames from the CameraThread's
lossy RAM buffer, runs the P01-P40 classifier + HUD renderer, and emits
processed results back to the UI.

Architecture:
    InferenceThread pulls the latest frame from CameraThread's single-frame
    lossy buffer via ``camera.get_latest_frame()``.  Stale frames are
    automatically dropped (buffer is ``deque(maxlen=1)``).  Inference runs
    asynchronously without blocking the 60 FPS UI rendering pipeline.

Dual FPS:
    - Inference FPS: throughput of the classifier + HUD pipeline.
    - Capture FPS: counted by receiving raw_frame signals from CameraThread.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from desktop_app.core.camera_thread import CameraThread
from desktop_app.core.pose_classifier import POSE_MAP, PoseClassifier
from desktop_app.core.hud_renderer import render_hud
from desktop_app.core.dataset_logger import DatasetLogger
from desktop_app.core import perf_config

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Samples frames from CameraThread's lossy buffer and runs inference."""

    frame_processed = pyqtSignal(object)   # annotated BGR frame with HUD
    result_ready = pyqtSignal(dict)        # P01-P40 classification dict
    fps_updated = pyqtSignal(float)        # inference FPS
    capture_fps_updated = pyqtSignal(float)  # capture/display FPS
    head_state_changed = pyqtSignal(str, float, float, float)  # (pose_id, yaw, pitch, roll)

    def __init__(self, camera: CameraThread, parent=None) -> None:
        super().__init__(parent)
        self._running = True
        self._camera = camera
        self._classifier: PoseClassifier | None = None
        self._dataset_logger: DatasetLogger | None = None
        self._inference_fps = _FpsCounter()
        self._capture_fps = _FpsCounter()
        self._last_pose_id = "P01"
        self._recording = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        self.wait(5000)
        if self._dataset_logger is not None:
            self._dataset_logger.close()

    def tick_capture_fps(self) -> None:
        self._capture_fps.tick()

    def set_recording(self, enabled: bool) -> None:
        """Enable or disable dataset recording."""
        self._recording = enabled
        if self._dataset_logger is None:
            self._dataset_logger = DatasetLogger()
        if enabled:
            self._dataset_logger.start_recording()
        else:
            self._dataset_logger.stop_recording()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._classifier = PoseClassifier()
        self._dataset_logger = DatasetLogger()
        logger.info("Inference worker started")

        # Rate limiting: without a cap this loop runs classify() flat out,
        # saturating the CPU and starving the UI thread.  The camera buffer is
        # a lossy single-frame deque, so any surplus frames are discarded
        # regardless — capping the rate costs nothing and gives the CPU back.
        min_interval = (
            1.0 / perf_config.TARGET_INFERENCE_FPS
            if perf_config.TARGET_INFERENCE_FPS > 0
            else 0.0
        )
        next_due = time.monotonic()

        while self._running:
            frame = self._camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Sleep until the next scheduled inference slot.  Sleeping first
            # avoids classifying a frame we are about to throw away anyway.
            if min_interval > 0.0:
                remaining = next_due - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                    continue
            next_due = time.monotonic() + min_interval

            # The capture thread withholds frames once it has confirmed the
            # source is dead, which leaves the previous good frame sitting in
            # the buffer.  Classifying that repeatedly would display a frozen
            # pose as if it were live, so report the fault instead.
            if self._camera.source_dead:
                self._emit_no_signal(frame)
                continue

            try:
                result = self._classifier.classify(frame)

                # Attach recording state for HUD indicator
                result["recording"] = self._recording

                inf_fps = self._inference_fps.tick()
                self.fps_updated.emit(inf_fps)
                cap_fps = self._capture_fps.current_fps()
                self.capture_fps_updated.emit(cap_fps)

                frame_with_hud = render_hud(frame, result, cap_fps, inf_fps)
                self.result_ready.emit(result)
                self.frame_processed.emit(frame_with_hud)

                # Log to dataset if recording
                if self._recording and self._dataset_logger is not None:
                    temporal = result.get("temporal_metrics", {})
                    self._dataset_logger.log_frame(
                        result,
                        motion_delta=temporal.get("bbox_scale_delta", 0.0),
                    )

                # Emit head state changes for console logging
                pose_id = result.get("pose_id", "P01")
                hp = result.get("head_pose", {})
                if pose_id != self._last_pose_id:
                    self.head_state_changed.emit(
                        pose_id,
                        hp.get("yaw", 0.0),
                        hp.get("pitch", 0.0),
                        hp.get("roll", 0.0),
                    )
                    self._last_pose_id = pose_id

            except Exception as exc:
                logger.error("Inference error: %s", exc, exc_info=True)
                self.frame_processed.emit(frame)

        logger.info("Inference worker stopped")

    def _emit_no_signal(self, frame: np.ndarray) -> None:
        """Report a confirmed dead source instead of a proctoring finding.

        The room is not being monitored, and saying so is the whole point: a
        frozen pose or a spurious "Absent Seat" would imply a student is being
        proctored when in fact nothing is being seen.
        """
        result = {
            "pose_id": "NO_SIGNAL",
            "pose_name": POSE_MAP["NO_SIGNAL"],
            "confidence": 0.0,
            "bounding_boxes": [],
            "keypoints": [],
            "landmarks": [],
            "gaze_endpoint": None,
            "head_pose": {},
            "spatial_features": {},
            "temporal_metrics": {},
            "objects_detected": [],
            "recording": self._recording,
            "source_dead": True,
        }
        self.fps_updated.emit(self._inference_fps.tick())
        self.capture_fps_updated.emit(self._capture_fps.current_fps())
        self.result_ready.emit(result)
        self.frame_processed.emit(
            render_hud(frame, result, self._capture_fps.current_fps(),
                       self._inference_fps.tick())
        )


class _FpsCounter:
    """Simple rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> float:
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        return self.current_fps()

    def current_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
