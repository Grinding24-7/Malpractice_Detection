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
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from desktop_app.core.camera_thread import CameraThread
from desktop_app.core.atomic_snapshot import AtomicSnapshot
from desktop_app.core.pose_classifier import (
    POSE_MAP,
    PoseClassifier,
)
from desktop_app.core.hud_renderer import render_hud
from desktop_app.core.dataset_logger import DatasetLogger
from desktop_app.core.object_context import (
    MP_LEFT_SHOULDER,
    MP_LEFT_WRIST,
    MP_RIGHT_SHOULDER,
    MP_RIGHT_WRIST,
    desk_region_for,
)
from desktop_app.core.seat_anchor import HeadAnchorInitializer
from desktop_app.core.torch_compile_utils import install_safe_compile
from desktop_app.core.vision_utils import (
    LetterboxBuffer,
    configure_opencv_threads,
    letterbox_square,
    unletterbox_box,
)
from desktop_app.core import perf_config

try:
    from backend.runtime import (
        empty_cache,
        optimize_cuda,
        resolve_device,
        yolo_device,
    )
except ImportError:  # pragma: no cover - direct module execution
    from runtime import (  # type: ignore[no-redef]
        empty_cache,
        optimize_cuda,
        resolve_device,
        yolo_device,
    )

logger = logging.getLogger(__name__)


class DeskProximityThread(QThread):
    """Concurrent desk-object detection on a throttled desk ROI.

    Runs its OWN YOLO model instance: ultralytics inference is not safe to
    call from two threads on one model, and the classifier's ``_obj_ctx``
    must stay untouched -- the e2e suite asserts that context's
    ``detect_counter`` equals the number of gate-approved calls, so a second
    user of it would break that accounting.

    Loop per iteration:
      1. read the latest frame from the camera buffer and the latest
         (bbox, wrists, shoulder width) candidate geometry under a lock;
      2. derive the desk ROI in front of that candidate (lower frame band,
         narrowed around the person, bottom expanded past the box);
      3. run desk YOLO on the ROI at ``OBJ_IMGSZ`` (416 px), throttled to
         1 pass per ``PROX_EVERY_N_FRAMES`` (1-in-8 by default); detections
         persist between passes and carry a ``hits`` count;
      4. compute raw Euclidean wrist -> bbox-centre distances and mark a
         device ``in_hand`` only inside ``PHONE_WRIST_DIST_PX``;
      5. publish a fresh, never-mutated snapshot dict under the lock.

    Consumers call ``snapshot()`` and receive the dict by reference; it is
    replaced (never modified) on each publish, so a reader on another thread
    always sees a consistent result without copying.
    """

    def __init__(self, camera: CameraThread, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._running = True
        self._camera = camera
        # Lock-free telemetry slots.  The producer (this thread) publishes a
        # complete immutable object; readers never take a lock, so a stalled
        # UI can never block inference and a slow detection pass can never
        # block the UI.
        self._bbox: list[int] | None = None
        self._wrists: tuple[tuple[float, float], ...] = ()
        self._shoulder_w: float = 0.0
        self._candidate = AtomicSnapshot(
            (None, (), 0.0, 0)
        )
        self._snapshot = AtomicSnapshot({
            "objects": [], "phone": None, "ts": 0.0,
        })
        self._model: Any = None
        # Set once _load_model() has returned, whether weights loaded or not.
        # Startup warmup uses this to distinguish "still loading" from "no
        # desk model available", so a missing model cannot cost a timeout.
        self._model_ready = threading.Event()
        self._class_map: dict[int, str] = {}
        self._device = resolve_device(perf_config.INFERENCE_DEVICE)
        self._device_arg = yolo_device(self._device)
        self._half = bool(perf_config.INFERENCE_HALF) and self._device.type == "cuda"
        # Throttle on candidate frames, not on this thread's loop iterations.
        # The loop spins much faster than inference (it sleeps 10 ms), so an
        # iteration counter made "1-in-8" mean ~12 model passes per second
        # instead of one pass per eight classified frames.  That competed
        # with the pose pipeline for CPU without adding evidence: the camera
        # buffer had not produced a new frame for most of those passes.
        self._candidate_seq = 0
        self._last_detected_seq = 0
        # Incremented only after a real model pass publishes.  Startup warmup
        # waits on this, not on snapshot timestamps: the thread publishes an
        # empty snapshot before the model loads, which would otherwise look
        # like a completed warmup pass.
        self._pass_count = AtomicSnapshot(0)
        # Detections persist between passes so the throttle costs latency,
        # not recall; aged out after a few passes without a sighting.
        self._tracks: list[dict[str, Any]] = []
        self._track_ttl = max(1, perf_config.PROX_EVERY_N_FRAMES * 3)
        configure_opencv_threads()
        self._letterbox_buffer = LetterboxBuffer(perf_config.OBJ_IMGSZ)

    # ------------------------------------------------------------------
    # Input / output (thread-safe)
    # ------------------------------------------------------------------

    def update_candidate(
        self,
        bbox: list[int] | None,
        wrists: tuple[tuple[float, float], ...],
        shoulder_w: float = 0.0,
    ) -> None:
        """Publish this frame's candidate box and visible wrist positions.

        Called from the inference thread after each classify(); the frame
        itself comes from the camera's own locked buffer, so only the small
        geometry tuple needs handing over here.  ``shoulder_w`` is optional
        so existing callers/tests keep working; when supplied it widens the
        desk ROI to the student's actual scale instead of the box width.
        """
        bbox_copy = list(bbox) if bbox else None
        wrists_copy = tuple(wrists)
        shoulder_width = float(shoulder_w or 0.0)
        self._candidate_seq += 1
        self._bbox = bbox_copy
        self._wrists = wrists_copy
        self._shoulder_w = shoulder_width
        self._candidate.publish(
            (bbox_copy, wrists_copy, shoulder_width, self._candidate_seq)
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the latest immutable detection snapshot (never None)."""
        return self._snapshot.get()

    @property
    def model_ready(self) -> bool:
        """True once model loading has finished (successfully or not)."""
        return self._model_ready.is_set()

    @property
    def model_loaded(self) -> bool:
        """True when a desk-object model is available for inference."""
        return self._model is not None

    @property
    def pass_count(self) -> int:
        """Number of completed model passes (safe to read from any thread)."""
        return self._pass_count.get()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        # Only release the model once the loop has actually exited.  Clearing
        # it after a timed-out wait would pull the model out from under a
        # pass that is still running on the proximity thread.
        if not self.wait(10000):
            logger.warning(
                "Desk proximity thread did not stop within 10 s; leaving "
                "its model to the thread"
            )
            return
        self._model = None
        empty_cache(self._device)

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
            optimize_cuda(self._device)
            self._model = YOLO(str(path))
            warm = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self._model(
                warm,
                imgsz=perf_config.OBJ_IMGSZ,
                device=self._device_arg,
                quantize=16 if self._half else None,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("Proximity thread: failed to load %s: %s", path, exc)
            self._model = None
            return

        if is_custom:
            self._class_map = adopt_custom_classes(self._model)
        else:
            from desktop_app.core.object_context import OBJ_CLASS_MAP
            self._class_map = dict(OBJ_CLASS_MAP)
        install_safe_compile(self._model, "desk-proximity")
        logger.info(
            "Proximity thread: desk-object model %s (%d classes)",
            path.name, len(self._class_map),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        # Publish a valid empty snapshot before the model finishes loading, so
        # a consumer never sees the ts=0 sentinel while startup is in progress.
        self._publish([], None)
        self._load_model()
        self._model_ready.set()
        logger.info("Desk proximity thread started")
        while self._running:
            frame = self._camera.get_latest_frame()
            if frame is None or self._model is None:
                time.sleep(0.02)
                continue

            bbox, wrists, shoulder_w, seq = self._candidate.get()
            if bbox is None:
                # No candidate this frame: an empty snapshot is published so
                # the classifier cannot act on a phone seen beside the last
                # person who was in frame.
                self._publish([], None)
                time.sleep(0.02)
                continue

            # One pass per PROX_EVERY_N_FRAMES candidate frames.  Guarding on
            # the sequence number also prevents re-detecting the same frame
            # while the loop waits for the next classify() to publish one.
            due = (
                seq != self._last_detected_seq
                and seq % max(1, perf_config.PROX_EVERY_N_FRAMES) == 1
            )
            if due:
                self._last_detected_seq = seq
                self._detect(frame, bbox, wrists, shoulder_w)
            else:
                # Refresh the phone entry against the CURRENT wrists even
                # between passes: distance is cheap, detection is not.
                objects = list(self._snapshot.get().get("objects") or [])
                phone = self._nearest_phone(objects, wrists)
                self._publish(objects, phone)

            # Bound the loop to roughly the inference rate so this thread
            # cannot spin flat-out when the camera buffer always has a frame.
            time.sleep(0.01)

        logger.info("Desk proximity thread stopped")

    @staticmethod
    def _desk_roi(
        frame_shape: tuple[int, ...],
        bbox: list[int],
        shoulder_w: float,
    ) -> tuple[int, int, int, int]:
        """The candidate's desk band: lower frame, narrow around the person.

        The ROI starts at the configured desk height (not at the torso, where
        a held phone would be), runs laterally past the candidate, and stops
        just below their box so the floor beyond the desk is excluded.
        """
        frame_h, frame_w = frame_shape[:2]
        rx1, ry1, rx2, _ = desk_region_for(
            frame_shape,
            bbox,
            shoulder_w,
            top_ratio=perf_config.PROX_DESK_TOP_RATIO,
            lateral_shoulder_widths=(
                perf_config.PROX_DESK_LATERAL_SHOULDER_WIDTHS
            ),
            force_narrow=True,
        )
        box_h = max(1, int(bbox[3] - bbox[1]))
        ry2 = min(
            frame_h,
            max(ry1 + 1, int(bbox[3] + perf_config.PROX_DESK_BOTTOM_EXPAND * box_h)),
        )
        rx1 = max(0, min(rx1, frame_w - 1))
        rx2 = max(rx1 + 1, min(rx2, frame_w))
        return rx1, ry1, rx2, ry2

    def _detect(
        self,
        frame: np.ndarray,
        bbox: list[int],
        wrists: tuple[tuple[float, float], ...],
        shoulder_w: float = 0.0,
    ) -> None:
        cx1, cy1, cx2, cy2 = self._desk_roi(
            frame.shape, bbox, shoulder_w
        )
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return

        canvas, scale, pad_x, pad_y = self._letterbox_buffer.apply(crop)

        try:
            with torch.inference_mode():
                results = self._model(
                    canvas,
                    imgsz=perf_config.OBJ_IMGSZ,
                    conf=perf_config.PROX_OBJ_CONF,
                    device=self._device_arg,
                    quantize=16 if self._half else None,
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
                    "box": unletterbox_box(
                        det_box,
                        scale,
                        pad_x,
                        pad_y,
                        offset_x=cx1,
                        offset_y=cy1,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    ),
                })

        objects = self._merge_tracks(fresh)
        self._annotate_objects(objects, wrists)
        phone = self._nearest_phone(objects, wrists)
        self._publish(objects, phone)
        self._pass_count.publish(self._pass_count.get() + 1)

    def _merge_tracks(self, fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Carry detections across throttled passes, aged by TTL.

        Every retained track ages one pass per invocation; a track that has
        not been re-sighted within ``_track_ttl`` passes is dropped.  A fresh
        detection matches a same-label track on overlap or on a two-axis
        centre shift; the track's ``hits`` counter then records how many
        throttled passes have re-sighted it, which is the temporal
        persistence gate the phone verdict reads.
        """
        for track in self._tracks:
            track["_age"] = track.get("_age", 0) + 1
        self._tracks = [
            t for t in self._tracks if t["_age"] <= self._track_ttl
        ]

        matched: set[int] = set()
        for det in fresh:
            best_i, best_score = -1, 0.0
            for i, track in enumerate(self._tracks):
                if i in matched or track["label"] != det["label"]:
                    continue
                iou = _box_iou(track["box"], det["box"])
                dx = abs(
                    (track["box"][0] + track["box"][2]) / 2.0
                    - (det["box"][0] + det["box"][2]) / 2.0
                )
                dy = abs(
                    (track["box"][1] + track["box"][3]) / 2.0
                    - (det["box"][1] + det["box"][3]) / 2.0
                )
                # Scale the centre-shift tolerance by the tracked object's
                # own size: a phone's box is small, so a fixed pixel gate
                # would be wrong at different camera distances.
                tbox = track["box"]
                scale = max(
                    20.0,
                    0.75 * max(tbox[2] - tbox[0], tbox[3] - tbox[1]),
                )
                if iou >= 0.10 or (dx <= scale and dy <= scale):
                    score = iou + 1.0 / (1.0 + dx + dy)
                    if score > best_score:
                        best_i, best_score = i, score
            if best_i >= 0:
                previous = self._tracks[best_i]
                self._tracks[best_i] = {
                    **det,
                    "_age": 0,
                    "hits": int(previous.get("hits", 0)) + 1,
                }
                matched.add(best_i)
            else:
                self._tracks.append({**det, "_age": 0, "hits": 1})

        return [
            {k: v for k, v in t.items() if k != "_age"} for t in self._tracks
        ]

    @staticmethod
    def _annotate_objects(
        objects: list[dict[str, Any]],
        wrists: tuple[tuple[float, float], ...],
    ) -> None:
        """Attach wrist distance and a coarse relation to each object.

        The relation is deliberately conservative: only an object whose
        centre is within the phone-use radius of a visible wrist is marked
        ``in_hand``.  Everything else stays ``on_desk`` when it is below the
        wrist line, or ``away`` -- enough for the shared escalation gate to
        decide, without pretending the proximity thread knows the full pose.
        """
        device_labels = {
            "cell_phone", "phone", "device", "laptop", "tablet", "headphones",
        }
        for obj in objects:
            box = obj.get("box")
            if not box or len(box) != 4:
                continue
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            dist = min(
                (float(np.hypot(wx - cx, wy - cy)) for wx, wy in wrists),
                default=float("inf"),
            )
            if dist == float("inf"):
                obj["wrist_dist_px"] = None
                obj.setdefault("relation", "away")
                obj.setdefault("in_hand", False)
                obj.setdefault("reachable", False)
                continue
            obj["wrist_dist_px"] = round(dist, 1)
            if obj.get("label") in device_labels:
                if dist < perf_config.PHONE_WRIST_DIST_PX:
                    obj["relation"] = "in_hand"
                    obj["in_hand"] = True
                    obj["reachable"] = True
                elif dist < perf_config.PHONE_WRIST_DIST_PX * 3.0:
                    obj["relation"] = "on_desk"
                    obj["in_hand"] = False
                    obj["reachable"] = True
                else:
                    obj["relation"] = "away"
                    obj["in_hand"] = False
                    obj["reachable"] = False
            else:
                obj.setdefault("relation", "on_desk")
                obj.setdefault("in_hand", False)
                obj.setdefault("reachable", True)

    @staticmethod
    def _nearest_phone(
        objects: list[dict[str, Any]],
        wrists: tuple[tuple[float, float], ...],
    ) -> dict[str, Any] | None:
        """Strongest phone-device detection measured against the wrists.

        Mirrors PoseClassifier._phone_proximity_hit so a snapshot consumed
        without the classifier's landmark context still carries the same
        wrist-distance evidence.  A phone must be re-sighted on
        ``PROX_MIN_HITS`` throttled passes and clear both the confidence and
        the 120 px wrist gates before it is reported as usable.
        """
        device_labels = {"cell_phone", "device", "phone"}
        best: dict[str, Any] | None = None
        best_dist = float("inf")
        for obj in objects:
            if obj.get("label") not in device_labels:
                continue
            if int(obj.get("hits", 0) or 0) < perf_config.PROX_MIN_HITS:
                continue
            if float(obj.get("confidence", 0.0)) <= perf_config.PHONE_MIN_CONF:
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
                best["wrist_dist_px"] = round(float(dist), 1)
                best["persistent"] = True
        if best is None or best_dist >= perf_config.PHONE_WRIST_DIST_PX:
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
        self._snapshot.publish(new_snapshot)


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
    # Emitted once, after the SCUT-HEAD initialiser has frozen the seat
    # anchors.  Payload is HeadAnchorInitializer.as_info(); the HUD gets the
    # same data on every result, this signal is for the console/log.
    classroom_initialized = pyqtSignal(dict)
    # Event-driven console telemetry.  Emitted only when the *published*
    # verdict changes (the stability gate has already applied the 12-frame
    # requirement), including a clear back to P01 and a NO_SIGNAL fault.
    # This is deliberately not a per-frame signal: the console drawer must
    # not be flooded with movement or frame-delta updates.
    verdict_changed = pyqtSignal(dict)

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
        # Last *published* verdict, after the stability gate.  Starts at P01
        # so a session does not open with a redundant "cleared" line.
        self._last_verdict_id = "P01"
        self._recording = False
        # Retained after the 30-frame countdown.  The unified cycle runs this
        # same SCUT-HEAD instance before YOLO-Pose on every live frame.
        self._head_initializer: HeadAnchorInitializer | None = None
        self._live_heads: list[dict[str, Any]] = []
        self._live_head_counter = 0
        # The UI thread toggles recording; the worker thread reads the flag and
        # writes rows.  The logger itself is locked too, but this lock keeps
        # creation/teardown from racing a toggle.
        self._recording_lock = threading.Lock()
        self._proximity: DeskProximityThread | None = None
        # Latest complete result for the UI's lock-free poller.  Published
        # once per frame; the UI never waits on the inference thread.
        self._latest_result = AtomicSnapshot({})
        self._classroom_info: dict[str, Any] = {
            "locked": False,
            "seated_count": 0,
            "frames_seen": 0,
            "frames_target": perf_config.CLASSROOM_INIT_FRAMES,
            "in_progress": False,
            "error": "",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Ask the worker to finish and wait for its own cleanup.

        Resource teardown happens in ``run()``'s ``finally`` block, on the
        worker thread.  Releasing the classifier or stopping the proximity
        thread here while ``classify()`` is still executing would pull state
        out from under a live frame.  If the wait times out, the worker still
        owns those objects and will release them when it exits; touching them
        from the UI thread is the one thing that must not happen.
        """
        self._running = False
        if not self.wait(15000):
            logger.warning(
                "Inference worker did not stop within 15 s; cleanup will run "
                "on the worker thread when it exits"
            )

    def tick_capture_fps(self) -> None:
        self._capture_fps.tick()

    def latest_result(self) -> dict[str, Any]:
        """Return the most recent complete result without blocking."""
        return self._latest_result.get()

    def set_recording(self, enabled: bool) -> None:
        """Enable or disable dataset recording (called from the UI thread)."""
        with self._recording_lock:
            if not self._running:
                logger.info(
                    "Ignoring recording toggle: inference worker is stopping"
                )
                return
            self._recording = enabled
            if self._dataset_logger is None:
                self._dataset_logger = DatasetLogger()
            dataset_logger = self._dataset_logger
        if enabled:
            dataset_logger.start_recording()
        else:
            dataset_logger.stop_recording()

    # ------------------------------------------------------------------
    # Classroom initialisation
    # ------------------------------------------------------------------

    def _initialise_classroom(self) -> dict[str, Any]:
        """Count seated heads for the first N frames and lock seat anchors.

        The initialiser loads SCUT-HEAD lazily and observes the camera's
        lossy buffer until it has seen ``CLASSROOM_INIT_FRAMES`` frames or
        ``CLASSROOM_INIT_MAX_SEC`` elapses.  A dead camera therefore cannot
        hold the worker in startup: the initialiser finishes with an empty,
        unlocked registry and the pose pipeline runs without anchors.
        """
        init = HeadAnchorInitializer()
        # Keep the initialiser (and its loaded SCUT-HEAD weights) alive for
        # the continuous single-mode cycle.  Seat identity stays locked; the
        # same detector supplies the live head boxes used to gate skeletons.
        self._head_initializer = init
        self._live_heads = []
        self._live_head_counter = 0
        interval = 1.0 / max(1.0, perf_config.TARGET_INFERENCE_FPS)
        image_replay = bool(
            getattr(self._camera, "is_image_replay", False)
        )
        if image_replay:
            interval = max(interval, 1.0 / max(perf_config.REPLAY_FPS, 0.1))

        # The initialiser enforces its own deadline inside observe(), but a
        # camera that never delivers a frame never reaches observe().  Check
        # the ceiling here as well, so a dead source cannot hold the worker
        # (and therefore the first result) in startup forever.
        started = time.monotonic()
        while self._running and not init.complete:
            if time.monotonic() - started >= perf_config.CLASSROOM_INIT_MAX_SEC:
                logger.warning(
                    "Classroom initialisation timed out after %.1f s with "
                    "%d/%d frames; continuing without seat anchors",
                    perf_config.CLASSROOM_INIT_MAX_SEC,
                    init.frames_seen,
                    init.frames_target,
                )
                break
            frame = self._camera.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            init.observe(frame)
            if init.complete:
                break
            time.sleep(interval)

        registry = init.finish()
        if self._classifier is not None:
            self._classifier.set_seat_anchors(registry)
        info = init.as_info()
        self._classroom_info = info
        if info.get("locked"):
            logger.info(
                "[INIT] Classroom Seated Count: %d Students Locked",
                int(info.get("seated_count", 0)),
            )
        else:
            logger.warning(
                "Classroom initialisation did not lock any seats: %s",
                info.get("error") or "no stable heads",
            )
        self.classroom_initialized.emit(dict(info))
        return info

    def _classify_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Run one unified cycle: SCUT-HEAD first, then pose classification.

        The head pass is the spatial anchor for the pose pass, so the order
        is part of the contract: heads are detected (or the cached boxes from
        the previous pass are reused when throttled), then the classifier is
        handed those boxes and validates every body hypothesis against them
        before spending MediaPipe inference.  Kept as its own method so the
        ordering can be tested without starting a camera/UI loop.
        """
        with torch.inference_mode():
            head_boxes: list[dict[str, Any]] = []
            if (
                perf_config.ENABLE_LIVE_HEAD_ANCHORS
                and self._head_initializer is not None
                and self._head_initializer.loaded
            ):
                every = max(1, perf_config.LIVE_HEAD_EVERY_N_FRAMES)
                if self._live_head_counter % every == 0:
                    self._live_heads = self._head_initializer.detect(frame)
                self._live_head_counter += 1
                head_boxes = self._live_heads
            return self._classifier.classify(frame, head_boxes=head_boxes)

    def _warm_detectors(self, frame: np.ndarray) -> None:
        """Warm the anomaly and proximity desk detectors on one real frame.

        Both use separate YOLO instances on the same device.  Their first
        simultaneous inference can take seconds (CUDA/driver warmup), which
        used to show up as a frozen HUD shortly after the first result.  This
        runs one classify() and one forced object pass while the proximity
        thread is also given a candidate, then waits for the proximity pass to
        publish.  Nothing is emitted: the first result the operator sees is
        already on the steady-state path.
        """
        try:
            result = self._classify_frame(frame)
        except Exception as exc:
            logger.warning("Startup classifier warmup failed: %s", exc)
            return

        boxes = result.get("bounding_boxes") or []
        bbox = boxes[0] if boxes else None
        wrists = self._visible_wrists(result)
        shoulder_w = self._primary_shoulder_width(result)

        before_passes = 0
        if self._proximity is not None:
            before_passes = self._proximity.pass_count
            if bbox is None:
                # No person yet: still give the proximity model a real crop so
                # its first pass is warm before someone walks into frame.
                frame_h, frame_w = frame.shape[:2]
                bbox = [
                    frame_w // 4, frame_h // 4,
                    3 * frame_w // 4, 3 * frame_h // 4,
                ]
            self._proximity.update_candidate(bbox, wrists, shoulder_w)

        try:
            with torch.inference_mode():
                self._classifier.warm_object_detector(frame, boxes)
        except Exception as exc:
            logger.warning("Startup object-detector warmup failed: %s", exc)

        if self._proximity is None:
            return
        deadline = time.monotonic() + 15.0
        while self._running and time.monotonic() < deadline:
            if self._proximity.model_ready and not self._proximity.model_loaded:
                logger.info(
                    "Proximity warmup skipped: no desk-object weights"
                )
                return
            if self._proximity.pass_count > before_passes:
                break
            time.sleep(0.05)
        else:
            if self._running:
                logger.warning(
                    "Proximity warmup did not publish within 15 s; "
                    "continuing without blocking startup"
                )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._classifier = PoseClassifier()
            if not self._running:
                return
            # Reuse a logger created by an early set_recording() call rather
            # than replacing it and losing the recording toggle.
            with self._recording_lock:
                if self._dataset_logger is None:
                    self._dataset_logger = DatasetLogger()

            # Concurrent desk-object pass.  Started here (not in __init__)
            # because QThread may only be started once per instance and the
            # worker is constructed before its camera source is live.
            self._proximity = DeskProximityThread(self._camera)
            self._proximity.start()

            # SCUT-HEAD classroom initialisation.  Runs here, on the worker
            # thread, so the 30-frame head count never touches the UI thread.
            # The registry it produces is handed to the classifier before the
            # first classify() call.
            self._initialise_classroom()
            if not self._running:
                return

            # Pay the first-use cost of both desk detectors before the first
            # HUD result.  Their first simultaneous CUDA inference is several
            # seconds of driver/model warmup on some systems; doing it here
            # keeps that stall out of the live feed.
            warm_frame = self._camera.get_latest_frame()
            if warm_frame is not None and not self._camera.source_dead:
                self._warm_detectors(warm_frame)
            if not self._running:
                return

            logger.info("Inference worker started")

            # Rate limiting: without a cap this loop runs classify() flat out,
            # saturating the CPU and starving the UI thread.  The camera buffer
            # is a lossy single-frame deque, so any surplus frames are
            # discarded regardless — capping the rate costs nothing and gives
            # the CPU back.
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

                # Sleep until the next scheduled inference slot.  Sleeping
                # first avoids classifying a frame we are about to throw away
                # anyway.
                if min_interval > 0.0:
                    remaining = next_due - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
                        continue
                next_due = time.monotonic() + min_interval

                # The capture thread withholds frames once it has confirmed
                # the source is dead, which leaves the previous good frame
                # sitting in the buffer.  Classifying that repeatedly would
                # display a frozen pose as if it were live, so report the
                # fault instead.
                if self._camera.source_dead:
                    self._emit_no_signal(frame)
                    continue

                try:
                    # Hand the concurrent detector's latest snapshot to the
                    # classifier on THIS thread: the snapshot reference was
                    # copied out under the detector's lock, so classify()
                    # never touches shared mutable state.
                    if self._proximity is not None:
                        self._classifier.set_proximity(
                            self._proximity.snapshot()
                        )

                    # Unified sequential cycle: SCUT-HEAD first, then
                    # YOLO11-Pose on the boxes those heads anchor.  The head
                    # pass is throttled only when configured; cached boxes
                    # bridge the skipped frame(s), never a failed pass.
                    result = self._classify_frame(frame)

                    # Feed this frame's candidate box + wrists back to the
                    # detector so its next pass crops the right region.
                    if self._proximity is not None:
                        boxes = result.get("bounding_boxes") or []
                        bbox = boxes[0] if boxes else None
                        wrists = self._visible_wrists(result)
                        shoulder_w = self._primary_shoulder_width(result)
                        self._proximity.update_candidate(
                            bbox, wrists, shoulder_w
                        )

                    # Attach recording state and classroom-init state for HUD.
                    with self._recording_lock:
                        recording = self._recording
                        dataset_logger = self._dataset_logger
                    result["recording"] = recording
                    result["classroom"] = dict(self._classroom_info)
                    # Publish before emitting: a UI timer that polls this slot
                    # must never observe an older frame than the signal path.
                    self._latest_result.publish(result)

                    inf_fps = self._inference_fps.tick()
                    self.fps_updated.emit(inf_fps)
                    cap_fps = self._capture_fps.current_fps()
                    self.capture_fps_updated.emit(cap_fps)

                    frame_with_hud = render_hud(
                        frame, result, cap_fps, inf_fps
                    )
                    self.result_ready.emit(result)
                    self.frame_processed.emit(frame_with_hud)
                    # One console event per published transition, never per
                    # frame.  The classifier has already applied the 12-frame
                    # streak and confidence gate at this point.
                    self._publish_verdict(result)

                    # Log to dataset if recording.  DatasetLogger owns its own
                    # lock; the recording flag and logger reference were read
                    # under ours above.
                    if recording and dataset_logger is not None:
                        temporal = result.get("temporal_metrics", {})
                        dataset_logger.log_frame(
                            result,
                            motion_delta=temporal.get(
                                "bbox_scale_delta", 0.0
                            ),
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

                    # Neutral state transitions only -- the label is resolved
                    # by the classifier, so the UI just logs what it is given.
                    neutral = result.get("neutral_state") or ""
                    if neutral != self._last_neutral_state:
                        self.neutral_state_changed.emit(
                            neutral, result.get("neutral_label") or ""
                        )
                        self._last_neutral_state = neutral

                except Exception as exc:
                    logger.error("Inference error: %s", exc, exc_info=True)
                    self.frame_processed.emit(frame)

        finally:
            # All teardown runs on the worker thread, after the loop can no
            # longer classify.  The proximity detector is stopped first
            # because classify() reads its snapshot; the classifier is
            # released only after that.
            if self._proximity is not None:
                self._proximity.stop()
                self._proximity = None
            if self._classifier is not None:
                self._classifier.release()
                self._classifier = None
            with self._recording_lock:
                dataset_logger = self._dataset_logger
                self._dataset_logger = None
            if dataset_logger is not None:
                dataset_logger.close()
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

    @staticmethod
    def _primary_shoulder_width(result: dict[str, Any]) -> float:
        """Shoulder width of the primary candidate, for desk-ROI scaling."""
        landmarks = result.get("landmarks") or []
        lms = landmarks[0] if landmarks else []
        if len(lms) <= MP_RIGHT_SHOULDER:
            return 0.0
        lx, _, lv = lms[MP_LEFT_SHOULDER]
        rx, _, rv = lms[MP_RIGHT_SHOULDER]
        if lv < 0.5 or rv < 0.5:
            return 0.0
        return abs(float(lx) - float(rx))

    def _publish_verdict(self, result: dict[str, Any]) -> None:
        """Emit one ``verdict_changed`` event per published transition.

        ``classify()`` only publishes a finding after the 12-frame stability
        gate, so comparing against the last published id is exactly the
        "verdict changed or cleared the gate" contract.  Initialising and
        repeated frames are silent; a clear back to P01 and a NO_SIGNAL fault
        are each emitted once.
        """
        pose_id = str(result.get("pose_id") or "")
        if not pose_id or pose_id == "INITIALIZING":
            return
        previous = self._last_verdict_id
        if pose_id == previous:
            return
        self._last_verdict_id = pose_id
        seat_ids = result.get("seat_ids") or []
        seat_id = next((s for s in seat_ids if s), "")
        escalation = result.get("object_escalation")
        self.verdict_changed.emit({
            "pose_id": pose_id,
            "pose_name": result.get("pose_name", ""),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "previous_pose_id": previous,
            "cleared": pose_id == "P01",
            "neutral_state": result.get("neutral_state"),
            "neutral_label": result.get("neutral_label", ""),
            "seat_id": seat_id,
            "object_escalation": (
                dict(escalation) if isinstance(escalation, dict) else None
            ),
            "candidate_depth_m": result.get("candidate_depth_m"),
            "source_dead": bool(result.get("source_dead", False)),
        })

    def _emit_no_signal(self, frame: np.ndarray) -> None:
        """Report a confirmed dead source instead of a proctoring finding.

        The room is not being monitored, and saying so is the whole point: a
        frozen pose or a spurious "Absent Seat" would imply a student is being
        proctored when in fact nothing is being seen.
        """
        with self._recording_lock:
            recording = self._recording
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
            "recording": recording,
            "source_dead": True,
            # No image means no posture: the neutral filter must report
            # nothing rather than carry the last live state over a fault.
            "neutral_state": None,
            "neutral_label": "",
            "neutral_suppressed": False,
            "phone_proximity": None,
            # Seat/continuity context survives a dead feed so the HUD header
            # still shows how many students the session locked at startup.
            "track_ids": [],
            "seat_ids": [],
            "seat_anchors": (
                self._classifier.seat_anchor_info()["seat_anchors"]
                if self._classifier is not None else []
            ),
            "seated_count": int(self._classroom_info.get("seated_count", 0)),
            "seats_locked": bool(self._classroom_info.get("locked", False)),
            "candidate_count": 0,
            "presence_held": False,
            "object_escalation": None,
            "classroom": dict(self._classroom_info),
        }
        self.fps_updated.emit(self._inference_fps.tick())
        self.capture_fps_updated.emit(self._capture_fps.current_fps())
        # A dead source also ends any neutral display: the student is no
        # longer being seen at all.
        if self._last_neutral_state != "":
            self.neutral_state_changed.emit("", "")
            self._last_neutral_state = ""
        self._latest_result.publish(result)
        self.result_ready.emit(result)
        self._publish_verdict(result)
        self.frame_processed.emit(
            render_hud(frame, result, self._capture_fps.current_fps(),
                       self._inference_fps.tick())
        )


class _FpsCounter:
    """Simple rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._timestamps: list[float] = []
        # The capture counter is ticked from the camera thread (raw_frame is
        # emitted there) while the worker reads it for the HUD.  The window
        # list must not be appended to and indexed concurrently.
        self._lock = threading.Lock()

    def tick(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            if len(self._timestamps) > self._window:
                self._timestamps.pop(0)
            return self._current_fps_locked()

    def current_fps(self) -> float:
        with self._lock:
            return self._current_fps_locked()

    def _current_fps_locked(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
