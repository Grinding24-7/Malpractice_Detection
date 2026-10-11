"""
inference_worker.py — QThread that samples frames from the CameraThread's
lossy RAM buffer, runs the P01-P40 classifier + HUD renderer, and emits
processed results back to the UI.

Architecture:
    InferenceThread pulls the latest frame from CameraThread's single-frame
    lossy buffer via ``camera.get_latest_frame()``.  Stale frames are
    automatically dropped (buffer is ``deque(maxlen=1)``).  Inference runs
    asynchronously without blocking the 60 FPS UI rendering pipeline.

    DeskProximityThread runs the custom desk-object YOLO model concurrently
    on candidate bounding-box crops, measures wrist-to-object distances, and
    publishes an immutable snapshot the classifier reads on its own thread
    (see PoseClassifier.set_proximity).

Dual FPS:
    - Inference FPS: throughput of the classifier + HUD pipeline.
    - Capture FPS: counted by receiving raw_frame signals from CameraThread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from desktop_app.core.camera_thread import CameraThread
from desktop_app.core.pose_classifier import (
    POSE_MAP,
    PoseClassifier,
)
from desktop_app.core.hud_renderer import render_hud
from desktop_app.core.dataset_logger import DatasetLogger
from desktop_app.core.object_context import MP_LEFT_WRIST, MP_RIGHT_WRIST
from desktop_app.core import perf_config

logger = logging.getLogger(__name__)


class DeskProximityThread(QThread):
    """Concurrent desk-object detection on candidate crops.

    Runs its OWN YOLO model instance: ultralytics inference is not safe to
    call from two threads on one model, and the classifier's ``_obj_ctx``
    must stay untouched -- the e2e suite asserts that context's
    ``detect_counter`` equals the number of gate-approved calls, so a second
    user of it would break that accounting.

    Loop per iteration:
      1. read the latest frame from the camera buffer and the latest
         (bbox, wrists) candidate geometry under a lock;
      2. crop the candidate box expanded by ``PROX_CROP_EXPAND``;
      3. run desk YOLO on the crop (throttled to 1 pass per
         ``PROX_EVERY_N_FRAMES``; detections persist between passes);
      4. compute Euclidean wrist -> bbox-centre distances;
      5. publish a fresh, never-mutated snapshot dict under the lock.

    Consumers call ``snapshot()`` and receive the dict by reference; it is
    replaced (never modified) on each publish, so a reader on another thread
    always sees a consistent result without copying.
    """

    def __init__(self, camera: CameraThread, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._running = True
        self._camera = camera
        self._lock = threading.Lock()
        self._bbox: list[int] | None = None
        self._wrists: tuple[tuple[float, float], ...] = ()
        self._snapshot: dict[str, Any] = {
            "objects": [], "phone": None, "ts": 0.0,
        }
        self._model: Any = None
        self._class_map: dict[int, str] = {}
        self._frame_count = 0
        # Detections persist between passes so the throttle costs latency,
        # not recall; aged out after a few passes without a sighting.
        self._tracks: list[dict[str, Any]] = []
        self._track_ttl = max(1, perf_config.PROX_EVERY_N_FRAMES * 3)

    # ------------------------------------------------------------------
    # Input / output (thread-safe)
    # ------------------------------------------------------------------

    def update_candidate(
        self,
        bbox: list[int] | None,
        wrists: tuple[tuple[float, float], ...],
    ) -> None:
        """Publish this frame's candidate box and visible wrist positions.

        Called from the inference thread after each classify(); the frame
        itself comes from the camera's own locked buffer, so only the small
        geometry tuple needs handing over here.
        """
        with self._lock:
            self._bbox = bbox
            self._wrists = wrists

    def snapshot(self) -> dict[str, Any]:
        """Return the latest immutable detection snapshot (never None)."""
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        self.wait(5000)

    def _load_model(self) -> None:
        # Import here: object_context pulls in perf_config and numpy only,
        # but keeping it out of module import keeps startup lighter.
        from desktop_app.core.object_context import (
            adopt_custom_classes,
            resolve_object_model,
        )
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("ultralytics unavailable for proximity thread: %s", exc)
            return

        resolved = resolve_object_model(
            perf_config.OBJ_CUSTOM_MODEL_FILE, perf_config.OBJ_MODEL_DIRS
        ) or resolve_object_model(
            perf_config.OBJ_FALLBACK_MODEL_FILE, perf_config.OBJ_MODEL_DIRS
        )
        if resolved is None:
            logger.info(
                "Proximity thread: no desk-object weights found; "
                "wrist-phone rule disabled"
            )
            return
        path, is_custom = resolved
        try:
            self._model = YOLO(str(path))
            warm = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self._model(warm, verbose=False)
        except Exception as exc:
            logger.warning("Proximity thread: failed to load %s: %s", path, exc)
            self._model = None
            return

        if is_custom:
            self._class_map = adopt_custom_classes(self._model)
        else:
            from desktop_app.core.object_context import OBJ_CLASS_MAP
            self._class_map = dict(OBJ_CLASS_MAP)
        logger.info(
            "Proximity thread: desk-object model %s (%d classes)",
            path.name, len(self._class_map),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._load_model()
        logger.info("Desk proximity thread started")
        while self._running:
            frame = self._camera.get_latest_frame()
            if frame is None or self._model is None:
                time.sleep(0.02)
                continue

            with self._lock:
                bbox = list(self._bbox) if self._bbox else None
                wrists = self._wrists
            if bbox is None:
                # No candidate this frame: an empty snapshot is published so
                # the classifier cannot act on a phone seen beside the last
                # person who was in frame.
                self._publish([], None)
                time.sleep(0.02)
                continue

            self._frame_count += 1
            due = (
                self._frame_count % max(1, perf_config.PROX_EVERY_N_FRAMES) == 1
            )
            if due:
                self._detect(frame, bbox, wrists)
            else:
                # Refresh the phone entry against the CURRENT wrists even
                # between passes: distance is cheap, detection is not.
                with self._lock:
                    objects = list(self._snapshot.get("objects") or [])
                phone = self._nearest_phone(objects, wrists)
                self._publish(objects, phone)

            # Bound the loop to roughly the inference rate so this thread
            # cannot spin flat-out when the camera buffer always has a frame.
            time.sleep(0.01)

        logger.info("Desk proximity thread stopped")

    def _detect(
        self,
        frame: np.ndarray,
        bbox: list[int],
        wrists: tuple[tuple[float, float], ...],
    ) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * perf_config.PROX_CROP_EXPAND)
        pad_y = int((y2 - y1) * perf_config.PROX_CROP_EXPAND)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return

        try:
            results = self._model(
                crop,
                imgsz=perf_config.OBJ_IMGSZ,
                conf=perf_config.PROX_OBJ_CONF,
                verbose=False,
            )
            r = results[0]
        except Exception as exc:
            logger.debug("Proximity detection failed: %s", exc)
            return

        fresh: list[dict[str, Any]] = []
        if r.boxes is not None:
            for det_box, det_cls, det_conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
            ):
                if det_cls not in self._class_map:
                    continue
                fresh.append({
                    "label": self._class_map[det_cls],
                    "confidence": round(float(det_conf), 3),
                    # Offset crop coordinates back to the full frame.
                    "box": [
                        int(det_box[0]) + cx1,
                        int(det_box[1]) + cy1,
                        int(det_box[2]) + cx1,
                        int(det_box[3]) + cy1,
                    ],
                })

        objects = self._merge_tracks(fresh)
        phone = self._nearest_phone(objects, wrists)
        self._publish(objects, phone)

    def _merge_tracks(self, fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Carry detections across throttled passes, aged by TTL.

        Every retained track ages one pass per invocation; a track that has
        not been re-sighted within ``_track_ttl`` passes is dropped, and a
        fresh detection within IoU 0.10 of a same-label track replaces it
        (resetting its age) rather than being reported as a second object.
        """
        for track in self._tracks:
            track["_age"] = track.get("_age", 0) + 1
        self._tracks = [
            t for t in self._tracks if t["_age"] <= self._track_ttl
        ]

        matched: set[int] = set()
        for det in fresh:
            best_i, best_iou = -1, 0.0
            for i, track in enumerate(self._tracks):
                if i in matched or track["label"] != det["label"]:
                    continue
                iou = _box_iou(track["box"], det["box"])
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0 and best_iou >= 0.10:
                self._tracks[best_i] = {**det, "_age": 0}
                matched.add(best_i)
            else:
                self._tracks.append({**det, "_age": 0})

        return [
            {k: v for k, v in t.items() if k != "_age"} for t in self._tracks
        ]

    @staticmethod
    def _nearest_phone(
        objects: list[dict[str, Any]],
        wrists: tuple[tuple[float, float], ...],
    ) -> dict[str, Any] | None:
        """Strongest phone-device detection measured against the wrists.

        Mirrors PoseClassifier._phone_proximity_hit so a snapshot consumed
        without the classifier's landmark context still carries the same
        wrist-distance evidence.
        """
        device_labels = {"cell_phone", "device", "phone"}
        best: dict[str, Any] | None = None
        best_dist = float("inf")
        for obj in objects:
            if obj.get("label") not in device_labels:
                continue
            box = obj.get("box")
            if not box or len(box) != 4:
                continue
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            dist = min(
                (np.hypot(wx - cx, wy - cy) for wx, wy in wrists),
                default=float("inf"),
            )
            if dist < best_dist:
                best_dist = float(dist)
                best = dict(obj)
                best["wrist_dist"] = round(float(dist), 1)
        if best is None:
            return None
        if (
            best_dist >= perf_config.PHONE_WRIST_DIST_PX
            or float(best.get("confidence", 0.0)) <= perf_config.PHONE_MIN_CONF
        ):
            return None
        return best

    def _publish(
        self,
        objects: list[dict[str, Any]],
        phone: dict[str, Any] | None,
    ) -> None:
        # A fresh dict every publish: readers holding the old reference keep
        # a consistent view, so no reader-side copy is needed.
        new_snapshot = {"objects": objects, "phone": phone, "ts": time.time()}
        with self._lock:
            self._snapshot = new_snapshot


def _box_iou(a: list[int], b: list[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


class InferenceWorker(QThread):
    """Samples frames from CameraThread's lossy buffer and runs inference."""

    frame_processed = pyqtSignal(object)   # annotated BGR frame with HUD
    result_ready = pyqtSignal(dict)        # P01-P40 classification dict
    fps_updated = pyqtSignal(float)        # inference FPS
    capture_fps_updated = pyqtSignal(float)  # capture/display FPS
    head_state_changed = pyqtSignal(str, float, float, float)  # (pose_id, yaw, pitch, roll)
    # (neutral_state, neutral_label) -- emitted only on change, so the UI
    # can log "[P00] Candidate #1: Writing (Neutral)" once per transition
    # rather than every frame.
    neutral_state_changed = pyqtSignal(str, str)

    def __init__(self, camera: CameraThread, parent=None) -> None:
        super().__init__(parent)
        self._running = True
        self._camera = camera
        self._classifier: PoseClassifier | None = None
        self._dataset_logger: DatasetLogger | None = None
        self._inference_fps = _FpsCounter()
        self._capture_fps = _FpsCounter()
        self._last_pose_id = "P01"
        self._last_neutral_state = ""
        self._recording = False
        self._proximity: DeskProximityThread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        self.wait(5000)
        # Stop the concurrent detector only after this thread has drained:
        # classify() reads its snapshot on this thread, so shutting it down
        # first would hand the last frames a dead source.
        if self._proximity is not None:
            self._proximity.stop()
            self._proximity = None
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

        # Concurrent desk-object pass.  Started here (not in __init__)
        # because QThread may only be started once per instance and the
        # worker is constructed before its camera source is live.
        self._proximity = DeskProximityThread(self._camera)
        self._proximity.start()

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
                # Hand the concurrent detector's latest snapshot to the
                # classifier on THIS thread: the snapshot reference was
                # copied out under the detector's lock, so classify() never
                # touches shared mutable state.
                if self._proximity is not None:
                    self._classifier.set_proximity(self._proximity.snapshot())

                result = self._classifier.classify(frame)

                # Feed this frame's candidate box + wrists back to the
                # detector so its next pass crops the right region.
                if self._proximity is not None:
                    boxes = result.get("bounding_boxes") or []
                    bbox = boxes[0] if boxes else None
                    wrists = self._visible_wrists(result)
                    self._proximity.update_candidate(bbox, wrists)

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

                # Neutral state transitions only -- the label is resolved by
                # the classifier, so the UI just logs what it is given.
                neutral = result.get("neutral_state") or ""
                if neutral != self._last_neutral_state:
                    self.neutral_state_changed.emit(
                        neutral, result.get("neutral_label") or ""
                    )
                    self._last_neutral_state = neutral

            except Exception as exc:
                logger.error("Inference error: %s", exc, exc_info=True)
                self.frame_processed.emit(frame)

        logger.info("Inference worker stopped")

    @staticmethod
    def _visible_wrists(result: dict[str, Any]) -> tuple[tuple[float, float], ...]:
        """Visible primary-candidate wrists from the just-finished frame."""
        landmarks = result.get("landmarks") or []
        lms = landmarks[0] if landmarks else []
        wrists: list[tuple[float, float]] = []
        for idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
            if len(lms) <= idx:
                continue
            x, y, vis = lms[idx]
            if vis >= 0.5 and x == x and y == y:
                wrists.append((float(x), float(y)))
        return tuple(wrists)

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
            # No image means no posture: the neutral filter must report
            # nothing rather than carry the last live state over a fault.
            "neutral_state": None,
            "neutral_label": "",
            "neutral_suppressed": False,
            "phone_proximity": None,
        }
        self.fps_updated.emit(self._inference_fps.tick())
        self.capture_fps_updated.emit(self._capture_fps.current_fps())
        # A dead source also ends any neutral display: the student is no
        # longer being seen at all.
        if self._last_neutral_state != "":
            self.neutral_state_changed.emit("", "")
            self._last_neutral_state = ""
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
