"""
detector.py — Pose estimation + per-candidate heuristics + research modules.

Pipeline:
    1. YOLO11n-pose runs at 5 FPS (sub-sampled from 30 FPS stream) in
       Ultralytics ByteTrack tracking mode, so every detected person keeps a
       stable candidate_id across frames.
    2. Keypoint heuristics flag posture anomalies (head-down, body-turn, lean).
    3. PnP 3D head pose estimation for gaze direction (via HeadPoseEstimator).
    4. ST-GCN for multi-person interaction detection.
    5. Object detection (phone/book) triggered on anomaly (reduce false positives).
    6. @torch.no_grad wrappers + explicit gc prevent memory leaks.

Sub-sampling: every 6th frame (30 FPS / 5 FPS = 6).
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

# The backend is launched as a script (python backend/main.py), where
# sys.path[0] is backend/ and the project root is absent.  The shared object
# context lives in desktop_app.core, so the root is pinned here exactly once
# rather than relying on how the process happened to be started.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.object_context import ObjectContext  # noqa: E402
from backend.runtime import (  # noqa: E402
    optimize_cuda,
    resolve_device,
    yolo_device,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — YOLO11n-pose keypoint indices (COCO 17-keypoint skeleton)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# COCO-17 -> the shared MediaPipe-33 layout the object context reads:
# nose, both shoulders (the scale reference) and both wrists (the spatial
# relation).  object_context defines the target layout once for both
# deployments; this map is the backend's half of that contract.
_COCO_TO_MP: dict[int, int] = {
    NOSE: 0,
    LEFT_SHOULDER: 11,
    RIGHT_SHOULDER: 12,
    LEFT_WRIST: 15,
    RIGHT_WRIST: 16,
}


def coco_to_mp_skeleton(person_kpts: np.ndarray) -> list[tuple[float, float, float]]:
    """Convert one COCO-17 person to the shared MediaPipe-33 layout.

    Only the five landmarks the object context reads are filled.  Everything
    else is emitted as (0, 0, 0) -- zero visibility, which the shared code
    treats as "not usable" rather than as a point at the origin, so an
    unfilled landmark can never look like a visible joint at (0,0).
    """
    mp: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 33
    for coco_idx, mp_idx in _COCO_TO_MP.items():
        if coco_idx >= len(person_kpts):
            continue
        x, y, conf = person_kpts[coco_idx]
        mp[mp_idx] = (float(x), float(y), float(conf))
    return mp

# Suspicious-item class ids, the desk region, spatial relations, persistence
# and the escalation map all live in desktop_app.core.object_context, shared
# with the desktop app so a rule change cannot land in one deployment and not
# the other.  Nothing about objects is decided here.


@dataclass
class InferenceResult:
    keypoints: np.ndarray  # shape (N, 17, 3) — x, y, confidence
    boxes: np.ndarray      # shape (N, 4) — xyxy
    timestamps: float      # time.monotonic() of inference
    # Defaults keep callers that do not run ByteTrack (stubbed tests, plain
    # `model()` inference) constructible without them.
    confidences: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    tracker_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    anomaly_flags: dict[str, bool] = field(default_factory=dict)
    head_pose: dict[str, float] = field(default_factory=dict)  # yaw, pitch, roll
    interaction: str = "normal"  # ST-GCN output label
    objects_detected: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PoseDetector:
    """
    Wraps YOLO11n-pose with ByteTrack tracking, sub-sampling, heuristics,
    PnP head pose, ST-GCN interaction detection, and object detection.

    Latency target: < 15 ms per inference call on CPU (WSL).
    """

    model: Any = None
    obj_model: Any = None  # YOLO11n for object detection
    _device: Any = field(default=None, repr=False)
    _device_arg: str = "cpu"
    _half: bool = False
    _frame_counter: int = 0
    subsample_rate: int = 6  # process every 6th frame (30 / 5)

    # Lazy-loaded modules (imported at runtime to avoid circular imports)
    _head_pose_estimator: Any = field(default=None, repr=False)
    _stgcn_classifier: Any = field(default=None, repr=False)
    _skeleton_buffer: Any = field(default=None, repr=False)
    # Shared desk-object component (region, spatial relations, escalation).
    _obj_ctx: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._load_models()

    def _load_models(self) -> None:
        """Load YOLO pose model, YOLO object model, and research modules."""
        # --- YOLO11n-pose (person detection + keypoints) ---
        try:
            from ultralytics import YOLO
            self._device = resolve_device(perf_config.INFERENCE_DEVICE)
            self._device_arg = yolo_device(self._device)
            self._half = (
                bool(perf_config.INFERENCE_HALF)
                and self._device.type == "cuda"
            )
            optimize_cuda(self._device)
            model_path = Path(__file__).resolve().parent / "yolo11n-pose.pt"
            # Prefer a checked-out copy next to this module; fall back to the
            # bare asset name so a fresh clone still triggers the download.
            self.model = YOLO(str(model_path if model_path.exists() else "yolo11n-pose.pt"))
            _ = self.model(
                np.zeros((640, 640, 3), dtype=np.uint8),
                imgsz=perf_config.POSE_IMGSZ,
                device=self._device_arg,
                quantize=16 if self._half else None,
                verbose=False,
            )
            logger.info(
                "YOLO11n-pose loaded from %s (%s%s)",
                model_path,
                self._device.type,
                ", FP16" if self._half else "",
            )
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: uv pip install ultralytics")

        # --- Desk-object detector (shared with the desktop app) ---
        # ObjectContext.load searches perf_config.OBJ_MODEL_DIRS for a custom
        # desk model first, falls back to the stock COCO weights, adopts the
        # loaded model's own class names, and logs loudly when the fallback
        # cannot see paper/pen/notebook.  Loading it here rather than a bare
        # YOLO("yolo11n.pt") is what makes the two deployments behave the same.
        try:
            self._obj_ctx = ObjectContext()
            self.obj_model = self._obj_ctx.load(
                perf_config.OBJ_CUSTOM_MODEL_FILE,
                perf_config.OBJ_FALLBACK_MODEL_FILE,
                perf_config.OBJ_MODEL_DIRS,
            )
        except Exception as exc:
            logger.warning("Failed to load object detector: %s", exc)
            self._obj_ctx = None
            self.obj_model = None

        # --- Head pose estimator (PnP) ---
        try:
            from desktop_app.core.head_pose import HeadPoseEstimator
            self._head_pose_estimator = HeadPoseEstimator()
            logger.info("HeadPoseEstimator loaded")
        except Exception as exc:
            logger.warning("Failed to load HeadPoseEstimator: %s", exc)

        # --- ST-GCN interaction detector ---
        try:
            from stgcn import STGCNClassifier
            self._stgcn_classifier = STGCNClassifier()
            logger.info("STGCN classifier loaded")
        except Exception as exc:
            logger.warning("Failed to load STGCN: %s", exc)

        # --- Skeleton buffer for temporal windowing ---
        try:
            from skeleton_buffer import SkeletonBuffer
            self._skeleton_buffer = SkeletonBuffer(window_size=30)
            logger.info("SkeletonBuffer initialized (window=30)")
        except Exception as exc:
            logger.warning("Failed to load SkeletonBuffer: %s", exc)

    @torch.inference_mode()
    def _infer(self, frame: np.ndarray) -> InferenceResult:
        """Run YOLO11n-pose in ByteTrack tracking mode.  Inference mode
        prevents autograd graph accumulation (memory leak prevention).
        `persist=True` keeps candidate_ids stable across frames."""
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=perf_config.POSE_IMGSZ,
            max_det=perf_config.POSE_MAX_DET,
            device=self._device_arg,
            quantize=16 if self._half else None,
            verbose=False,
        )
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            tracker_ids = r.boxes.id.cpu().numpy().astype(np.int64)
        else:
            boxes = np.empty((0, 4))
            tracker_ids = np.empty((0,), dtype=np.int64)
        confidences = (
            r.boxes.conf.cpu().numpy()
            if r.boxes is not None and r.boxes.conf is not None
            else np.empty((0,))
        )
        kpts = (
            r.keypoints.data.cpu().numpy()
            if r.keypoints is not None
            else np.empty((0, 17, 3))
        )

        return InferenceResult(
            keypoints=kpts,
            boxes=boxes,
            confidences=confidences,
            tracker_ids=tracker_ids,
            timestamps=time.monotonic(),
        )

    def should_process(self) -> bool:
        """Sub-sample: return True only every `subsample_rate`-th call."""
        self._frame_counter += 1
        return (self._frame_counter % self.subsample_rate) == 0

    def track(self, frame: np.ndarray) -> InferenceResult:
        """
        Run tracked inference + heuristics on a single frame.

        Unlike `process()`, this always runs (no sub-sampling) so the caller
        (e.g. app.py) controls its own sampling cadence.
        """
        result = self._infer(frame)
        result.anomaly_flags = self._run_heuristics(result)
        return result

    def process(self, frame: np.ndarray) -> Optional[InferenceResult]:
        """
        Run the full detection pipeline on a single frame.

        Steps:
            1. YOLO pose detection
            2. Keypoint heuristics (head-down, body-turn, lean)
            3. PnP 3D head pose (yaw/pitch/roll + gaze)
            4. Update skeleton buffer → ST-GCN interaction classification
            5. Object detection (only when anomaly detected)

        Returns None if the frame is skipped (sub-sampling).
        """
        if not self.should_process():
            return None

        result = self._infer(frame)

        # --- 1. Baseline heuristics ---
        flags = self._run_heuristics(result)
        result.anomaly_flags = flags

        # --- 2. PnP 3D head pose ---
        head_pose = self._estimate_head_pose(result, frame.shape)
        result.head_pose = head_pose

        # Enhance heuristics with 3D head pose
        yaw = head_pose.get("yaw", 0.0)
        pitch = head_pose.get("pitch", 0.0)
        if abs(yaw) > 35.0:
            flags["body_turn"] = True
        if pitch < -25.0:
            flags["head_down"] = True

        # --- 3. ST-GCN interaction detection ---
        interaction = self._detect_interaction(result)
        result.interaction = interaction
        if interaction in ("interaction", "paper_pass", "phone_pass"):
            flags["multi_person"] = True

        # --- 4. Object detection (only on anomaly) ---
        # The track clock advances once per processed frame, not once per
        # anomaly: track expiry (OBJ_PERSIST_FRAMES) is measured against it,
        # so tying it to anomalies would keep stale objects alive whenever
        # the room was quiet.
        if self._obj_ctx is not None:
            self._obj_ctx.frames_seen += 1

        has_anomaly = any(flags.values())
        if has_anomaly and self._obj_ctx is not None and self.obj_model is not None:
            objects = self._detect_objects(frame, result)
            result.objects_detected = objects
            # A HIGH alert requires the object to clear the spatial gate AND
            # to be a label configured to escalate -- the same decision the
            # desktop makes.  Existence alone no longer raises anything: a
            # proctor's laptop on the far desk is reported in the payload for
            # context but does not accuse the student.
            if ObjectContext.escalation(objects) is not None:
                flags["object_detected"] = True

        return result

    # ------------------------------------------------------------------
    # Baseline heuristics
    # ------------------------------------------------------------------

    def _run_heuristics(self, result: InferenceResult) -> dict[str, bool]:
        """Evaluate lightweight posture rules."""
        flags: dict[str, bool] = {
            "head_down": False,
            "body_turn": False,
            "excessive_lean": False,
            "multi_person": False,
        }

        if result.keypoints.shape[0] == 0:
            return flags

        # --- Head-down detection ---
        for person in result.keypoints:
            nose_conf = person[NOSE, 2]
            ls_conf = person[LEFT_SHOULDER, 2]
            rs_conf = person[RIGHT_SHOULDER, 2]
            if nose_conf > 0.5 and ls_conf > 0.3 and rs_conf > 0.3:
                nose_y = person[NOSE, 1]
                shoulder_y = (person[LEFT_SHOULDER, 1] + person[RIGHT_SHOULDER, 1]) / 2.0
                if nose_y > shoulder_y + 20.0:
                    flags["head_down"] = True

        # --- Body-turn detection ---
        for person in result.keypoints:
            nose_conf = person[NOSE, 2]
            ls_conf = person[LEFT_SHOULDER, 2]
            rs_conf = person[RIGHT_SHOULDER, 2]
            if nose_conf > 0.5 and ls_conf > 0.3 and rs_conf > 0.3:
                nose_x = person[NOSE, 0]
                shoulder_mid_x = (person[LEFT_SHOULDER, 0] + person[RIGHT_SHOULDER, 0]) / 2.0
                if abs(nose_x - shoulder_mid_x) > 40.0:
                    flags["body_turn"] = True

        # --- Excessive lean ---
        for person in result.keypoints:
            ls_conf = person[LEFT_SHOULDER, 2]
            rs_conf = person[RIGHT_SHOULDER, 2]
            lh_conf = person[LEFT_HIP, 2]
            rh_conf = person[RIGHT_HIP, 2]
            if all(c > 0.3 for c in (ls_conf, rs_conf, lh_conf, rh_conf)):
                shoulder_mid = (
                    (person[LEFT_SHOULDER, 0] + person[RIGHT_SHOULDER, 0]) / 2.0,
                    (person[LEFT_SHOULDER, 1] + person[RIGHT_SHOULDER, 1]) / 2.0,
                )
                hip_mid = (
                    (person[LEFT_HIP, 0] + person[RIGHT_HIP, 0]) / 2.0,
                    (person[LEFT_HIP, 1] + person[RIGHT_HIP, 1]) / 2.0,
                )
                dy = shoulder_mid[1] - hip_mid[1]
                dx = shoulder_mid[0] - hip_mid[0]
                if dy > 0 and abs(dx / dy) > 0.3:
                    flags["excessive_lean"] = True

        # --- Multi-person ---
        if result.keypoints.shape[0] > 1:
            flags["multi_person"] = True

        return flags

    # ------------------------------------------------------------------
    # PnP 3D Head Pose Estimation
    # ------------------------------------------------------------------

    def _estimate_head_pose(
        self, result: InferenceResult, frame_shape: tuple[int, ...]
    ) -> dict[str, float]:
        """Run PnP-based 3D head pose estimation on the primary person.

        Uses HeadPoseEstimator which maps MediaPipe-like 6-point facial
        landmarks to a 3D face model via solvePnP.
        """
        if self._head_pose_estimator is None:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        if result.keypoints.shape[0] == 0:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Use COCO keypoints to approximate facial landmarks for PnP
        # Map COCO keypoints to the 6-point format expected by HeadPoseEstimator
        person = result.keypoints[0]  # primary person
        h, w = frame_shape[:2]

        # Build image_points array: [nose, chin, left_eye, right_eye, left_mouth, right_mouth]
        # Approximate from COCO keypoints (nose=0, left_eye=1, right_eye=2)
        image_points = np.array([
            [person[NOSE, 0], person[NOSE, 1]],              # nose
            [person[NOSE, 0], person[NOSE, 1] + 50],         # chin (approximated below nose)
            [person[LEFT_EYE, 0], person[LEFT_EYE, 1]],      # left eye outer
            [person[RIGHT_EYE, 0], person[RIGHT_EYE, 1]],    # right eye outer
            [person[NOSE, 0] - 20, person[NOSE, 1] + 20],    # left mouth corner (approx)
            [person[NOSE, 0] + 20, person[NOSE, 1] + 20],    # right mouth corner (approx)
        ], dtype=np.float64)

        # Build fake landmarks list (33 entries, fill with zeros)
        landmarks: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 33
        landmarks[0] = (person[NOSE, 0], person[NOSE, 1], float(person[NOSE, 2]))
        landmarks[3] = (person[LEFT_EYE, 0], person[LEFT_EYE, 1], float(person[LEFT_EYE, 2]))
        landmarks[6] = (person[RIGHT_EYE, 0], person[RIGHT_EYE, 1], float(person[RIGHT_EYE, 2]))
        landmarks[9] = (image_points[4][0], image_points[4][1], 0.5)
        landmarks[10] = (image_points[5][0], image_points[5][1], 0.5)

        result = self._head_pose_estimator.estimate(landmarks, image_shape=(h, w))
        return {
            "yaw": result.get("yaw", 0.0),
            "pitch": result.get("pitch", 0.0),
            "roll": result.get("roll", 0.0),
        }

    # ------------------------------------------------------------------
    # ST-GCN Interaction Detection
    # ------------------------------------------------------------------

    def _detect_interaction(self, result: InferenceResult) -> str:
        """Feed keypoints into SkeletonBuffer and run ST-GCN classification.

        Returns one of: "normal", "interaction", "paper_pass", "phone_pass".
        """
        if self._skeleton_buffer is None or self._stgcn_classifier is None:
            return "normal"

        # Append current frame's primary person keypoints
        if result.keypoints.shape[0] > 0:
            self._skeleton_buffer.append(result.keypoints[0])
        else:
            self._skeleton_buffer.append(None)

        # Only classify when buffer is full
        if not self._skeleton_buffer.is_full:
            return "normal"

        seq = self._skeleton_buffer.get_sequence()
        if seq is None:
            return "normal"

        label, confidence = self._stgcn_classifier.predict(seq)

        # Only flag if confidence exceeds threshold
        if confidence > 0.6:
            return label
        return "normal"

    # ------------------------------------------------------------------
    # Object detection (shared desk-object context)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _detect_objects(
        self, frame: np.ndarray, result: InferenceResult
    ) -> list[dict[str, Any]]:
        """Detect desk objects via the shared ObjectContext.

        Everything the previous implementation got wrong now lives in the
        shared module instead of here: it cropped the bottom 60% of each
        *person box* (so anything beside the student was unreachable by
        construction), had no spatial relation, no nearest-person attribution,
        no persistence across throttled passes, and no escalation map -- which
        is how a proctor's laptop on the far desk became a HIGH alert.

        This method only converts the backend's COCO keypoints into the
        shared skeleton layout and hands the frame over; region, throttle,
        tracking, attribution and escalation are desktop_app.core's rules.
        """
        ctx = self._obj_ctx
        if ctx is None or ctx.model is None:
            return []
        if result.keypoints.shape[0] == 0 or result.boxes.shape[0] == 0:
            return []

        skeletons = [coco_to_mp_skeleton(p) for p in result.keypoints]
        people = ObjectContext.capture_persons(skeletons, result.boxes)
        ctx.set_frame_context(
            frame.shape,
            people,
            list(map(int, result.boxes[0])),
            skeletons[0],
            people[0]["shoulder_w"],
        )
        return ctx.detect(frame)
