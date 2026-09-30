"""
pose_classifier.py — P01-P40 pose classification engine.

Combines YOLO11n-pose (person detection + COCO-17 keypoints) with
MediaPipe PoseLandmarker Task API (33-point skeleton landmarks) to
classify every detected person into one of 40 exam-malpractice poses.

Pipeline per frame:
    1. YOLO -> person bounding boxes
    2. MediaPipe PoseLandmarker -> 33-point skeleton inside each box
    3. 3D Head Pose via dedicated HeadPoseEstimator (PnP solver)
    4. Geometric feature engineering (joint angles + normalised distances)
    5. Temporal sliding window (1-sec motion history, sustained anomaly)
    6. Spatial bounding-box vectoring (coverage, proximity, gaze)
    7. Rule engine evaluates P01-P40 in priority order
    8. Optional: scikit-learn RF classifier for supervised prediction
    9. Returns standardised dict
"""

from __future__ import annotations

import collections
import logging
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from desktop_app.core import perf_config
from desktop_app.core.head_pose import HeadPoseEstimator
from desktop_app.core.object_context import (
    MP_LEFT_SHOULDER,
    MP_LEFT_WRIST,
    MP_NOSE,
    MP_RIGHT_SHOULDER,
    MP_RIGHT_WRIST,
    OBJ_CLASS_MAP,
    SYSTEM_STATES,
    ObjectContext,
    adopt_custom_classes,
    box_iou,
    resolve_object_model,
    shoulder_width_of,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSETS_DIR = _PROJECT_ROOT / "assets"
_BACKEND_DIR = _PROJECT_ROOT / "backend"

# MediaPipe PoseLandmarker model variants (float16 weights).
# The variant is chosen by perf_config.MP_MODEL_VARIANT so the accuracy /
# latency trade-off can be tuned without editing this module.
_MP_MODEL_URL_TEMPLATE = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_{variant}/float16/latest/"
    "pose_landmarker_{variant}.task"
)
_MP_MODEL_VARIANTS = ("lite", "full", "heavy")

_FOCAL = 700.0

# Landmark indices in MediaPipe 33-point skeleton.  The five the shared
# object context also needs (nose, shoulders, wrists) are imported above from
# object_context so there is exactly one definition of the layout; the rest
# are desktop-only.
MP_LEFT_EYE_INNER = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_THUMB = 21
MP_RIGHT_THUMB = 22
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8

# Pose ID -> human-readable name
POSE_MAP: dict[str, str] = {
    # Not a proctoring finding.  Reported when the capture source is confirmed
    # to be producing no image, so a dead camera is never rendered as a
    # student who is absent from the room.
    "NO_SIGNAL": "Camera Not Sending Image",
    # Also not a finding.  Reported while the models are still settling and no
    # verdict about the room can honestly be made yet.  "Absent Seat" on the
    # first frame is a serious false accusation -- it bypasses the stability
    # gate because it is treated as a fact about the room -- so the room is
    # explicitly marked as not-yet-assessed instead.
    "INITIALIZING": "Starting Up - Room Not Yet Assessed",
    "P01": "Normal Sitting",
    "P02": "Head Turn Left 30",
    "P03": "Head Turn Right 30",
    "P04": "Head Turn Left 60",
    "P05": "Head Turn Right 60",
    "P06": "Head Down",
    "P07": "Head Up",
    "P08": "Lean Left",
    "P09": "Lean Right",
    "P10": "Body Turn Left",
    "P11": "Body Turn Right",
    "P12": "Looking Away",
    "P13": "Note Peeking",
    "P14": "Talking / Whispering",
    "P15": "Standing Up",
    "P16": "Absent Seat",
    "P17": "Absent + Multiple",
    "P18": "Head on Desk",
    "P19": "Hand to Mouth",
    "P20": "Looking at Neighbor",
    "P21": "Writing Under Desk",
    "P22": "Raising Hand",
    "P23": "Excessive Fidgeting",
    "P24": "Second Screen Gaze",
    "P25": "Hidden Face",
    "P26": "Earbuds Detected",
    "P27": "Object Transfer",
    "P28": "Frequent Glancing",
    "P29": "Chin Rest",
    "P30": "Crossed Arms",
    "P31": "Extreme Side Lean",
    "P32": "Backward Lean",
    "P33": "Forward Hunch",
    "P34": "Eye Gaze Averted",
    "P35": "Multiple Persons",
    "P36": "Head Oscillation",
    "P37": "Torso Rotation",
    "P38": "Unusual Object",
    "P39": "Collaboration",
    "P40": "Other Anomaly",
}

# Ordered feature names for scikit-learn classifier input vector
FEATURE_NAMES: list[str] = [
    "elbow_flexion_left",
    "elbow_flexion_right",
    "neck_tilt_angle",
    "wrist_to_eye_ratio",
    "wrist_to_ear_ratio",
    "eye_to_shoulder_ratio",
    "hand_mouth_distance",
    "shoulder_hip_offset_ratio",
    "torso_leg_ratio",
    "shoulder_width_ratio",
    "hip_width_ratio",
    "torso_asymmetry",
]

# Features are NaN when the underlying landmarks are not visible enough to
# measure.  These helpers keep rule evaluation honest about that.


def _feature_or_zero(features: dict[str, float], name: str) -> float:
    """Return a feature value, coalescing 'not measured' (NaN) to 0.0."""
    value = features.get(name)
    if value is None or value != value:  # missing, or NaN
        return 0.0
    return value


def _feature_lt(features: dict[str, float], name: str, threshold: float) -> bool:
    """True only if the feature was measured AND is below *threshold*."""
    value = features.get(name)
    return value is not None and value == value and value < threshold


def _feature_gt(features: dict[str, float], name: str, threshold: float) -> bool:
    """True only if the feature was measured AND is above *threshold*."""
    value = features.get(name)
    return value is not None and value == value and value > threshold


# COCO class ids for objects that indicate possible misconduct live in
# object_context.OBJ_CLASS_MAP, because the CCTV backend shares them.  The
# ids are verified against the ultralytics COCO names list: 63=laptop,
# 67=cell phone, 73=book.  Not present in COCO-80, and therefore unreachable
# until a custom model is trained: paper, pen, pencil, notebook, exam sheet,
# desk -- the objects that actually matter for a written exam, which is the
# gap the custom desk model is meant to close.

# Poses that report on the state of the *system*, not on a student.  They are
# exempt from the stability gate and the confidence floor, and are styled
# differently in the HUD, because a camera fault rendered as a pose verdict is
# a false statement about a person.  Defined in object_context (shared with
# the backend) and re-exported here under the name this module's gate uses.
_SYSTEM_STATES = SYSTEM_STATES

# Poses that describe who or how many are present rather than what one student
# is doing.  They are reported immediately and cannot be escalated by
# desk-object evidence: "there is no student" is not made either truer or
# falser by a phone lying on the empty desk.  They are also excluded from
# object detection, which would otherwise spend a second model pass on a
# verdict that no object can change.
#
# Derived from perf_config rather than restated, so the two definitions cannot
# drift apart.
_PRESENCE_POSES = perf_config.IMMEDIATE_POSES


def frame_has_usable_image(frame: np.ndarray | None) -> bool:
    """True when *frame* carries enough image to support a claim about a room.

    A frame that is valid in every structural sense -- correct size, correct
    dtype -- but essentially black cannot show a person, so nothing may be
    concluded from it.  This is what separates "the seat is empty" from "the
    camera is not sending anything".

    The check runs on a 64x36 downscale: at 720p that samples ~9k pixels
    instead of 921k, which keeps it off the per-frame critical path.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()) >= (
        perf_config.BLACK_LUMA_THRESHOLD
    )


# Temporal window config
_TEMPORAL_WINDOW = 30   # 1 second at 30 FPS
_SUSTAINED_THRESHOLD = 15  # frames of sustained anomaly before flagging


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at vertex *b* formed by segments ba and bc (degrees)."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _dist_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class _TemporalBuffer:
    """Sliding-window motion history buffer for a single candidate track."""

    __slots__ = (
        "_yaw_hist", "_pitch_hist", "_bbox_hist", "_anomaly_streak",
        "_pose_streak_id", "_pose_streak_len",
        "_pitch_baseline", "_yaw_baseline",
        "_turn_baseline", "_turn_mad",
    )

    def __init__(self) -> None:
        self._yaw_hist: collections.deque[float] = collections.deque(maxlen=_TEMPORAL_WINDOW)
        self._pitch_hist: collections.deque[float] = collections.deque(maxlen=_TEMPORAL_WINDOW)
        self._bbox_hist: collections.deque[tuple[float, float, float, float]] = collections.deque(
            maxlen=_TEMPORAL_WINDOW
        )
        self._anomaly_streak: int = 0
        # Per-pose-id consecutive counter, used to gate noisy single frames.
        self._pose_streak_id: str = ""
        self._pose_streak_len: int = 0
        # Running baselines, so "looking down" means a deviation from how this
        # student normally sits rather than from an absolute zero point.
        self._pitch_baseline: float | None = None
        self._yaw_baseline: float | None = None
        # Body-turn ratio cannot be thresholded on an absolute value (see
        # _turn_deviation_z), so it gets its own baseline and dispersion.
        self._turn_baseline: float | None = None
        self._turn_mad: float | None = None

    # ── Baseline tracking ──────────────────────────────────────────────

    def update_baseline(self, yaw: float, pitch: float) -> None:
        """Track a slow baseline of neutral head orientation.

        A long exponential average is used rather than a windowed mean so a
        sustained, genuine deviation does not become the new normal.
        """
        if self._pitch_baseline is None:
            self._pitch_baseline = pitch
            self._yaw_baseline = yaw
            return
        # ~0.5 response per 30 frames: slow enough that a deliberate head
        # movement cannot pull the baseline with it.
        alpha = 1.0 / 60.0
        self._pitch_baseline += alpha * (pitch - self._pitch_baseline)
        self._yaw_baseline += alpha * (yaw - self._yaw_baseline)

    @property
    def pitch_baseline(self) -> float:
        return self._pitch_baseline if self._pitch_baseline is not None else 0.0

    @property
    def yaw_baseline(self) -> float:
        return self._yaw_baseline if self._yaw_baseline is not None else 0.0

    def update_turn(self, ratio: float) -> None:
        """Track this student's normal body-turn ratio and its dispersion.

        A self-scaling ratio cannot be thresholded: turning the torso
        foreshortens the shoulders, so the denominator shrinks at exactly the
        moment the numerator grows.  Measured on a known-normal clip the ratio
        spanned -0.86..+0.56, and raising the absolute cut from 0.20 to 0.50
        barely moved the false-positive rate (41.7% -> 33.3%).  The cut is
        therefore a robust z-score against the student's own normal.
        """
        if self._turn_baseline is None:
            self._turn_baseline = ratio
            self._turn_mad = 0.0
            return
        alpha = 1.0 / 60.0
        self._turn_baseline += alpha * (ratio - self._turn_baseline)
        deviation = abs(ratio - self._turn_baseline)
        if self._turn_mad is None:
            self._turn_mad = deviation
        else:
            self._turn_mad += alpha * (deviation - self._turn_mad)

    def turn_deviation_z(self, ratio: float) -> float:
        """Robust z-score of *ratio* against this student's own normal.

        Returns 0.0 until there is enough history to judge, which keeps the
        opening seconds of a session from producing spurious body-turn events.
        """
        if self._turn_baseline is None or len(self._bbox_hist) < 20:
            return 0.0
        # Floor on dispersion so a very still student cannot turn every
        # sub-pixel wiggle into a 6-sigma event.
        spread = max(self._turn_mad or 0.0, 0.05)
        return abs(ratio - self._turn_baseline) / spread

    # ── Per-pose streak ────────────────────────────────────────────────

    def note_pose(self, pose_id: str) -> int:
        """Record the raw rule verdict and return its consecutive-frame count.

        The count resets whenever the rule output changes, so a pose that
        flickers between two ids can never accumulate a streak.
        """
        if pose_id == self._pose_streak_id:
            self._pose_streak_len += 1
        else:
            self._pose_streak_id = pose_id
            self._pose_streak_len = 1
        return self._pose_streak_len

    def update(
        self, yaw: float, pitch: float, bbox: list[int] | None
    ) -> dict[str, float]:
        """Append current frame data and return temporal metrics."""
        self._yaw_hist.append(yaw)
        self._pitch_hist.append(pitch)
        if bbox:
            self._bbox_hist.append(tuple(bbox))

        metrics: dict[str, float] = {
            "yaw_variance": 0.0,
            "pitch_variance": 0.0,
            "bbox_scale_delta": 0.0,
            "sustained_anomaly_frames": 0.0,
        }

        # Yaw / pitch variance (detects persistent glance vs rapid neck turn)
        if len(self._yaw_hist) >= 5:
            metrics["yaw_variance"] = float(np.var(list(self._yaw_hist)))
            metrics["pitch_variance"] = float(np.var(list(self._pitch_hist)))

        # Bounding box scale delta (detects leaning out / seat departure)
        if len(self._bbox_hist) >= 2:
            prev = self._bbox_hist[-2]
            curr = self._bbox_hist[-1]
            prev_area = max(1, (prev[2] - prev[0]) * (prev[3] - prev[1]))
            curr_area = max(1, (curr[2] - curr[0]) * (curr[3] - curr[1]))
            metrics["bbox_scale_delta"] = (curr_area - prev_area) / prev_area

        # Sustained anomaly tracking
        metrics["sustained_anomaly_frames"] = float(self._anomaly_streak)
        return metrics

    def record_anomaly(self) -> None:
        """Increment the sustained anomaly streak counter."""
        self._anomaly_streak += 1

    def clear_anomaly(self) -> None:
        """Reset the sustained anomaly streak counter."""
        self._anomaly_streak = 0

    @property
    def is_sustained_anomaly(self) -> bool:
        """True if anomaly has been active for > _SUSTAINED_THRESHOLD frames."""
        return self._anomaly_streak > _SUSTAINED_THRESHOLD


class PoseClassifier:
    """P01-P40 pose classification engine.

    Uses YOLO for person detection, MediaPipe Task API for fine-grained
    33-point pose landmarks, dedicated HeadPoseEstimator for 3D angles,
    temporal sliding window for motion tracking, spatial bounding-box
    vectoring for multi-candidate scenarios, and YOLO object detection
    for phone/book/laptop detection (triggered on anomaly only).
    """

    def __init__(self) -> None:
        self._yolo: Any = None
        # Desk-object detector: a custom desk model when one is present,
        # otherwise the COCO-80 fallback.  `_obj_class_map` is what is
        # actually filtered on, so a custom model's classes apply without
        # touching the detection code.
        self._obj_model: Any = None
        self._obj_class_map: dict[int, str] = dict(OBJ_CLASS_MAP)
        self._obj_custom: bool = False
        self._obj_model_path: str = ""
        # Per-person body context for object attribution (see
        # _capture_person_context).  The singular fields above remain for the
        # desk region and single-subject rules.
        self._last_persons: list[dict[str, Any]] = []
        self._pose_landmarker: Any = None
        self._mp_image_cls: Any = None
        self._mp_image_format: Any = None
        self._head_pose_estimator = HeadPoseEstimator()
        self._temporal_buffer = _TemporalBuffer()
        self._last_features: dict[str, float] = {}
        self._ml_classifier: Any = None
        self._slow_frame_counter = 0
        # Frames seen since construction.  Used to hold back the first
        # verdicts while the models and the temporal baselines settle.
        self._frames_seen = 0
        # Most recent frame, kept so rules can ask whether it carries an image
        # before asserting anything about who is (or is not) in the room.
        self._last_frame: np.ndarray | None = None
        # Context the desk-region and spatial-relation code needs, captured
        # once per frame so those helpers stay free of parameters.
        self._last_frame_shape: tuple[int, ...] = (0, 0)
        self._last_frame_width: int = 0
        self._last_landmarks: list[tuple[float, float, float]] | None = None
        self._last_person_bbox: list[int] | None = None
        self._last_shoulder_width: float = 0.0
        self._init_yolo()
        self._init_object_detector()
        self._init_mediapipe()
        self._load_ml_classifier()
        logger.info(perf_config.describe())

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_yolo(self) -> None:
        try:
            from ultralytics import YOLO

            model_path = _BACKEND_DIR / "yolo11n-pose.pt"
            self._yolo = YOLO(str(model_path))
            _ = self._yolo(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            logger.info("YOLO11n-pose loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load YOLO: %s", exc)
            self._yolo = None

    def _ensure_obj_context(self) -> ObjectContext:
        """The shared desk-object component (created on first use).

        Lazy so partially constructed instances -- ``__new__``-built objects
        in tests -- still behave, and so desktop and backend cannot end up
        with different object rules: the decisions all live on the context.
        """
        ctx = getattr(self, "_obj_ctx", None)
        if ctx is None:
            ctx = ObjectContext()
            self._obj_ctx = ctx
        return ctx

    def _sync_obj_context(self) -> ObjectContext:
        """Copy this frame's body/model state into the shared context.

        The classifier keeps the ``_last_*`` fields because the rest of the
        rule engine reads them; the object context reads its own copies, so
        every object-facing method funnels through here first.
        """
        ctx = self._ensure_obj_context()
        shape = getattr(self, "_last_frame_shape", None)
        if shape is not None and len(shape) > 1:
            ctx.frame_shape = shape
            ctx.frame_width = int(shape[1])
        ctx.persons = getattr(self, "_last_persons", []) or []
        ctx.person_bbox = getattr(self, "_last_person_bbox", None)
        ctx.landmarks = getattr(self, "_last_landmarks", None)
        ctx.shoulder_w = getattr(self, "_last_shoulder_width", 0.0) or 0.0
        ctx.frames_seen = getattr(self, "_frames_seen", 0)
        ctx.model = getattr(self, "_obj_model", None)
        ctx.class_map = getattr(self, "_obj_class_map", dict(OBJ_CLASS_MAP))
        ctx.is_custom = getattr(self, "_obj_custom", False)
        return ctx

    def _init_object_detector(self) -> None:
        """Load the desk-object detector via the shared context.

        A custom desk model wins over the stock COCO weights when one is
        present, and its own class names become the label set, so training a
        model on paper/pen/notebook needs no code change here.  Falling back to
        COCO-80 leaves those classes genuinely undetectable, which is why the
        failure mode is logged rather than silently accepted.
        """
        ctx = self._ensure_obj_context()
        model = ctx.load(
            perf_config.OBJ_CUSTOM_MODEL_FILE,
            perf_config.OBJ_FALLBACK_MODEL_FILE,
            perf_config.OBJ_MODEL_DIRS,
        )
        self._obj_model = model
        self._obj_model_path = ctx.model_path
        self._obj_class_map = ctx.class_map
        self._obj_custom = ctx.is_custom

    @staticmethod
    def _resolve_object_model(filename: str) -> tuple[Path, bool] | None:
        """Find `filename` in the configured search directories."""
        return resolve_object_model(filename, perf_config.OBJ_MODEL_DIRS)

    def _adopt_custom_object_classes(self, model: Any) -> None:
        """Use a custom model's own class names as the label set.

        Everything it can detect is treated as reportable, because the whole
        point of a desk model is that its classes were chosen for this task.
        Class 0 ("person") is excluded: a custom model is trained on desk
        objects, and a person box carries no evidence about the desk.
        """
        ctx = self._ensure_obj_context()
        ctx.class_map = adopt_custom_classes(model)
        ctx.is_custom = True
        self._obj_class_map = ctx.class_map
        self._obj_custom = ctx.is_custom


    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp

            self._mp_image_cls = mp.Image
            self._mp_image_format = mp.ImageFormat.SRGB

            variant = perf_config.MP_MODEL_VARIANT
            if variant not in _MP_MODEL_VARIANTS:
                logger.warning(
                    "Unknown MediaPipe variant %r — falling back to 'full'",
                    variant,
                )
                variant = "full"

            model_path = _ASSETS_DIR / f"pose_landmarker_{variant}.task"
            if not model_path.exists():
                self._download_model(model_path, variant)

            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

            base_opts = BaseOptions(model_asset_path=str(model_path))
            pose_opts = PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=perf_config.MP_NUM_POSES,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_landmarker = PoseLandmarker.create_from_options(pose_opts)
            logger.info("MediaPipe PoseLandmarker (%s) loaded from %s", variant, model_path)
        except Exception as exc:
            logger.error("Failed to load MediaPipe PoseLandmarker: %s", exc)
            self._pose_landmarker = None

    def _load_ml_classifier(self) -> None:
        model_path = _PROJECT_ROOT / "models" / "pose_rf_classifier.pkl"
        if not model_path.exists():
            logger.info("No ML classifier found at %s — using rules only", model_path)
            return
        try:
            import joblib
            self._ml_classifier = joblib.load(str(model_path))
            logger.info("ML classifier loaded from %s", model_path)
        except ImportError:
            logger.warning("scikit-learn/joblib not installed — ML classifier unavailable")
        except Exception as exc:
            logger.warning("Failed to load ML classifier: %s", exc)

    @staticmethod
    def _download_model(dest: Path, variant: str = "heavy") -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _MP_MODEL_URL_TEMPLATE.format(variant=variant)
        logger.info("Downloading MediaPipe model (%s) -> %s", variant, dest)
        urllib.request.urlretrieve(url, str(dest))
        logger.info("Download complete: %s", dest)

    # ------------------------------------------------------------------
    # Public classify entry-point
    # ------------------------------------------------------------------

    def _object_context_wanted(self, raw_pose_id: str, warming_up: bool) -> bool:
        """True when desk-object detection could change the verdict.

        The rules (no anomaly pass, no system states, no presence verdicts,
        no warmup, no student) live on ObjectContext so the backend asks the
        identical question; this wrapper only supplies this frame's inputs.
        """
        ctx = self._ensure_obj_context()
        ctx.landmarks = self._last_landmarks
        return ObjectContext.wanted(
            raw_pose_id,
            warming_up,
            model_loaded=self._obj_model is not None,
            has_landmarks=self._last_landmarks is not None,
        )

    def classify(self, frame: np.ndarray) -> dict[str, Any]:
        """Classify the frame and return a standardised result dict.

        Returns
        -------
        dict with keys:
            pose_id, pose_name, confidence, bounding_boxes,
            keypoints, landmarks, head_pose, gaze_vector,
            geometric_features, temporal_metrics, spatial_features,
            anomalies, anomaly_flags
        """
        t0 = time.perf_counter()
        self._last_frame = frame
        h, w = frame.shape[:2]
        self._last_frame_shape = frame.shape
        self._last_frame_width = w

        yolo_boxes, yolo_kpts = self._yolo_detect(frame)
        all_landmarks = self._mediapipe_landmarks(frame, yolo_boxes)

        # Capture the body context the desk-region and spatial-relation code
        # needs.  Shoulder width is the normalising scale throughout, so a
        # measurement is only made when the shoulders are actually usable.
        #
        # Every person is captured, not just the first: a desk object must be
        # attributed to the student nearest it, or a phone on a shared desk
        # belongs to whoever happened to be detected first.  `_last_persons`
        # is the list the object-relation code iterates; the singular
        # `_last_landmarks` / `_last_person_bbox` / `_last_shoulder_width`
        # keep feeding the desk region and the existing single-subject rules.
        self._last_persons = self._capture_person_context(all_landmarks, yolo_boxes)
        self._last_landmarks = all_landmarks[0] if all_landmarks else None
        self._last_person_bbox = list(yolo_boxes[0]) if yolo_boxes else None
        self._last_shoulder_width = (
            self._last_persons[0]["shoulder_w"] if self._last_persons else 0.0
        )

        # ── 3D Head Pose (dedicated estimator) ───────────────────────────
        head_pose_result = self._head_pose_estimator.estimate(
            all_landmarks[0] if all_landmarks else [],
            image_shape=(h, w),
        )
        head_pose = {
            "yaw": head_pose_result["yaw"],
            "pitch": head_pose_result["pitch"],
            "roll": head_pose_result["roll"],
        }
        gaze_vector = head_pose_result["gaze_vector"]
        gaze_endpoint = self._head_pose_estimator.get_gaze_endpoint(
            all_landmarks[0] if all_landmarks else [], (h, w), length=150.0
        )

        # ── Geometric features ───────────────────────────────────────────
        features = self._compute_geometric_features(
            all_landmarks[0] if all_landmarks else []
        )
        self._last_features = features

        # ── Temporal sliding window ──────────────────────────────────────
        primary_bbox = yolo_boxes[0] if yolo_boxes else None
        temporal_metrics = self._temporal_buffer.update(
            head_pose["yaw"], head_pose["pitch"], primary_bbox
        )
        # Baselines are read BEFORE this frame is folded in, so a single frame
        # cannot shift the neutral reference it is compared against.  They are
        # updated after the rule engine returns.
        pitch_baseline = self._temporal_buffer.pitch_baseline
        yaw_baseline = self._temporal_buffer.yaw_baseline

        # Body-turn baseline.  The ratio is recomputed here from the same
        # landmarks the rule engine uses, and fed to the buffer before the
        # rule engine reads its z-score, so the two can never disagree about
        # what the ratio was this frame.
        turn_ratio = self._compute_turn_ratio(
            all_landmarks[0] if all_landmarks else []
        )
        if turn_ratio is not None:
            self._temporal_buffer.update_turn(turn_ratio)

        # ── Spatial features ─────────────────────────────────────────────
        spatial = self._compute_spatial_features(yolo_boxes, frame.shape)

        # ── Multi-candidate proximity vectoring ──────────────────────────
        person_count = len(yolo_boxes)
        anomalies_from_head = head_pose_result.get("anomalies", [])

        # ── Rule engine ──────────────────────────────────────────────────
        raw_pose_id, raw_confidence = self._evaluate_rules(
            all_landmarks, head_pose, person_count, features,
            anomalies_from_head, temporal_metrics, spatial,
            pitch_baseline=pitch_baseline, yaw_baseline=yaw_baseline,
            turn_ratio=turn_ratio,
        )

        # ── Object detection (only when anomaly detected) ────────────────
        # Gated on the RAW rule verdict so an anomaly that is still building a
        # streak still gets its object context, and so warmup frames do not pay
        # for a second YOLO pass.
        #
        # This runs BEFORE the stability gate on purpose.  Escalating after
        # the gate would let a single-frame object detection report P38/P13
        # immediately, which is precisely the unfiltered single-frame finding
        # the gate exists to remove.
        objects_detected: list[dict[str, Any]] = []
        warming_up = self._frames_seen + 1 <= perf_config.WARMUP_FRAMES
        pose_id, confidence = raw_pose_id, raw_confidence
        if self._object_context_wanted(raw_pose_id, warming_up):
            objects_detected = self._detect_objects(frame, yolo_boxes)
            # Escalate to P13 Note Peeking / P38 Unusual Object, but only for
            # objects the student could actually be using.
            pose_id, confidence = self._escalate_from_objects(
                pose_id, confidence, objects_detected
            )

        # ── Temporal / confidence gate ───────────────────────────────────
        # Applied to the FINAL verdict, so an object-driven escalation is held
        # to the same persistence requirement as a pose-driven one.
        pose_id, confidence, suppressed = self._apply_stability_gate(
            pose_id, confidence, warming_up=warming_up
        )

        # ── Startup warmup ───────────────────────────────────────────────
        # The first frames of a session are not evidence.  Models are still
        # warming, the student may still be sitting down or reaching for the
        # keyboard, and the head-pitch/turn baselines have not been
        # established.  Reporting "Absent Seat" on frame 1 -- which is exactly
        # what happened, and which bypasses the gate because presence is
        # treated as a fact about the room -- accuses a student who is simply
        # not settled yet.  Nothing is asserted until warmup completes.
        self._frames_seen += 1
        if warming_up:
            pose_id, confidence = "INITIALIZING", 0.0
            temporal_metrics["warmup"] = 1.0

        if pose_id not in ("P01", "INITIALIZING"):
            temporal_metrics["suppressed_anomaly"] = 1.0 if suppressed else 0.0
            if suppressed:
                # A real but not-yet-convincing finding: keep it in the
                # metrics/logging path without publishing it as an event.
                logger.debug(
                    "suppressed %s (streak/confidence below gate)", pose_id
                )

        # Now that this frame's verdict is final, fold it into the baseline.
        self._temporal_buffer.update_baseline(head_pose["yaw"], head_pose["pitch"])

        # ── Sustained anomaly tracking ───────────────────────────────────
        if pose_id not in ("P01", "INITIALIZING"):
            self._temporal_buffer.record_anomaly()
        else:
            self._temporal_buffer.clear_anomaly()

        if self._temporal_buffer.is_sustained_anomaly:
            # Escalate confidence for sustained anomalies
            confidence = min(confidence + 0.10, 0.99)
            temporal_metrics["sustained_anomaly_frames"] = float(
                self._temporal_buffer._anomaly_streak
            )

        # ── Optional ML classifier override ──────────────────────────────
        if self._ml_classifier is not None and features and not warming_up:
            ml_pose_id, ml_conf = self._ml_predict(features)
            if ml_conf > 0.80:
                pose_id, confidence = ml_pose_id, ml_conf

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > perf_config.WARN_THRESHOLD_MS:
            # Rate-limit: warn periodically rather than on every frame, so a
            # slow pipeline does not flood the console drawer.
            self._slow_frame_counter += 1
            if self._slow_frame_counter % perf_config.WARN_RATE_LIMIT_FRAMES == 1:
                logger.warning(
                    "classify() took %.1f ms (over %.0f ms budget)",
                    elapsed_ms,
                    perf_config.WARN_THRESHOLD_MS,
                )

        return {
            "pose_id": pose_id,
            "pose_name": POSE_MAP.get(pose_id, "Unknown"),
            "confidence": round(confidence, 3),
            "bounding_boxes": [list(map(int, b)) for b in yolo_boxes],
            "keypoints": yolo_kpts,
            "landmarks": all_landmarks,
            "head_pose": head_pose,
            "gaze_vector": gaze_vector,
            "gaze_endpoint": gaze_endpoint,
            "geometric_features": features,
            "temporal_metrics": temporal_metrics,
            "spatial_features": spatial,
            "anomalies": anomalies_from_head,
            "anomaly_flags": self._build_flags(pose_id),
            "objects_detected": objects_detected,
        }

    # ------------------------------------------------------------------
    # YOLO detection
    # ------------------------------------------------------------------

    def _yolo_detect(
        self, frame: np.ndarray
    ) -> tuple[list[list[int]], list[np.ndarray]]:
        if self._yolo is None:
            return ([], [])
        try:
            results = self._yolo(
                frame,
                imgsz=perf_config.POSE_IMGSZ,
                max_det=perf_config.POSE_MAX_DET,
                verbose=False,
            )
            r = results[0]
            boxes = (
                r.boxes.xyxy.cpu().numpy().tolist()
                if r.boxes is not None
                else []
            )
            kpts = (
                r.keypoints.data.cpu().numpy()
                if r.keypoints is not None
                else np.empty((0, 17, 3))
            )
            return boxes, list(kpts) if kpts.ndim == 3 else []
        except Exception as exc:
            logger.error("YOLO detection failed: %s", exc)
            return ([], [])

    # ------------------------------------------------------------------
    # MediaPipe PoseLandmarker (Task API)
    # ------------------------------------------------------------------

    def _mediapipe_landmarks(
        self,
        frame: np.ndarray,
        boxes: list[list[int]],
    ) -> list[list[tuple[float, float, float]]]:
        if self._pose_landmarker is None or self._mp_image_cls is None:
            return []
        if not boxes:
            return []

        all_landmarks: list[list[tuple[float, float, float]]] = []
        frame_h, frame_w = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Convert only the crop to RGB.  Converting the whole frame first
            # (as this previously did) is pure waste: on a 1280x720 CCTV feed
            # that is ~2.7 MB of extra conversion per frame, per person.
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            if crop.size == 0:
                continue

            # MediaPipe requires C-contiguous input.  Feeding it a strided
            # numpy view makes it silently return zero poses (no error, no
            # warning), which quietly disables the entire P01-P40 rule
            # engine downstream.  Guarantee contiguity explicitly.
            if not crop.flags["C_CONTIGUOUS"]:
                crop = np.ascontiguousarray(crop)

            mp_img = self._mp_image_cls(
                image_format=self._mp_image_format, data=crop
            )
            result = self._pose_landmarker.detect(mp_img)

            if result.pose_landmarks:
                person_lms = []
                for lm in result.pose_landmarks[0]:
                    ax = lm.x * (x2 - x1) + x1
                    ay = lm.y * (y2 - y1) + y1
                    person_lms.append((ax, ay, lm.visibility))
                all_landmarks.append(person_lms)

        return all_landmarks

    # ------------------------------------------------------------------
    # Stability gate
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_turn_ratio(
        landmarks: list[tuple[float, float, float]],
    ) -> float | None:
        """Nose offset from the shoulder midpoint, in shoulder-widths.

        Returns None when the landmarks needed for the measurement are missing
        or the shoulders are too narrow to divide by, so callers can tell
        "not measurable this frame" apart from "measured zero".
        """
        if len(landmarks) <= max(MP_NOSE, MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER):
            return None
        if (min(landmarks[MP_NOSE][2], landmarks[MP_LEFT_SHOULDER][2],
                landmarks[MP_RIGHT_SHOULDER][2]) <= 0.5):
            return None
        shoulder_w = abs(
            landmarks[MP_LEFT_SHOULDER][0] - landmarks[MP_RIGHT_SHOULDER][0]
        )
        if shoulder_w < perf_config.MIN_SCALE_PX:
            return None
        shoulder_mid_x = (
            landmarks[MP_LEFT_SHOULDER][0] + landmarks[MP_RIGHT_SHOULDER][0]
        ) / 2.0
        return (landmarks[MP_NOSE][0] - shoulder_mid_x) / shoulder_w

    def _apply_stability_gate(
        self, pose_id: str, confidence: float, warming_up: bool = False
    ) -> tuple[str, float, bool]:
        """Suppress anomalies that are too brief or too weak to be real.

        Single-frame threshold crossings used to be reported as proctoring
        events, which produced a 50% false-positive rate on a known-normal
        clip.  A finding must now hold the same pose id for
        ``ANOMALY_MIN_STREAK_FRAMES`` consecutive frames and reach
        ``ANOMALY_MIN_MARGIN`` confidence before it is published.

        Presence/absence poses (absent student, extra person) bypass the gate:
        those are facts about the room rather than inferences about a person,
        and delaying them would be harmful.

        While ``warming_up``, no streak is recorded and no finding is
        published.  Streaks accumulated during warmup would leak: the models
        and baselines are not ready, the student may still be sitting down,
        and an anomaly "observed" before the room was ever assessed would let
        the very first post-warmup frame publish immediately, defeating the
        warmup entirely.  The streak therefore starts at zero when warmup
        ends.

        Returns ``(pose_id, confidence, suppressed)``.
        """
        if pose_id in _SYSTEM_STATES:
            # System states are not findings and must never be gated, ranked or
            # downgraded: a camera fault reported as "P01 Normal Sitting" is
            # exactly the false reassurance this whole mechanism exists to
            # prevent.
            return pose_id, confidence, False

        if pose_id in perf_config.IMMEDIATE_POSES:
            return pose_id, confidence, False

        if warming_up:
            # Record nothing.  Returning the raw verdict is safe because
            # classify() replaces it with INITIALIZING, but clearing here too
            # keeps the contract local: during warmup the gate has no opinion.
            return pose_id, confidence, False

        # P01 is recorded so that a return to normal behaviour CLEARS the
        # streak.  Returning early here instead would let an old streak
        # survive any number of normal frames, so the next single noisy frame
        # would publish an anomaly instantly.
        streak = self._temporal_buffer.note_pose(pose_id)

        if pose_id == "P01":
            return pose_id, confidence, False

        if streak < perf_config.ANOMALY_MIN_STREAK_FRAMES:
            return "P01", 0.0, True

        if confidence < perf_config.ANOMALY_MIN_MARGIN:
            return "P01", 0.0, True

        # Once an anomaly has earned its place, report the strength of the
        # accumulated evidence rather than that of a single frame.
        return pose_id, min(0.99, confidence + 0.05), False

    # ------------------------------------------------------------------
    # Geometric feature engineering
    # ------------------------------------------------------------------

    def _compute_geometric_features(
        self, landmarks: list[tuple[float, float, float]]
    ) -> dict[str, float]:
        # NaN means "could not be measured" (landmark below the visibility
        # threshold).  It must NOT be 0.0: 0.0 is a meaningful value for these
        # ratios, and rules that fire on "too small" (e.g. hand-to-mouth)
        # would then trigger on every unmeasurable frame.  That previously
        # mislabelled the majority of frames as P19.
        features: dict[str, float] = {name: float("nan") for name in FEATURE_NAMES}

        if not landmarks or len(landmarks) < 33:
            return features

        def _pt(idx: int) -> tuple[float, float]:
            return (landmarks[idx][0], landmarks[idx][1])

        def _vis(idx: int, thr: float = 0.5) -> bool:
            return landmarks[idx][2] > thr

        nose = _pt(MP_NOSE)
        shoulder_mid = (
            (landmarks[MP_LEFT_SHOULDER][0] + landmarks[MP_RIGHT_SHOULDER][0]) / 2,
            (landmarks[MP_LEFT_SHOULDER][1] + landmarks[MP_RIGHT_SHOULDER][1]) / 2,
        )
        hip_mid = (
            (landmarks[MP_LEFT_HIP][0] + landmarks[MP_RIGHT_HIP][0]) / 2,
            (landmarks[MP_LEFT_HIP][1] + landmarks[MP_RIGHT_HIP][1]) / 2,
        )
        torso_len = _dist_2d(nose, hip_mid)
        if torso_len < 1.0:
            torso_len = 1.0

        # Joint angles
        if _vis(MP_LEFT_SHOULDER) and _vis(MP_LEFT_ELBOW) and _vis(MP_LEFT_WRIST):
            features["elbow_flexion_left"] = _angle_between(
                np.array(_pt(MP_LEFT_SHOULDER)),
                np.array(_pt(MP_LEFT_ELBOW)),
                np.array(_pt(MP_LEFT_WRIST)),
            )
        if _vis(MP_RIGHT_SHOULDER) and _vis(MP_RIGHT_ELBOW) and _vis(MP_RIGHT_WRIST):
            features["elbow_flexion_right"] = _angle_between(
                np.array(_pt(MP_RIGHT_SHOULDER)),
                np.array(_pt(MP_RIGHT_ELBOW)),
                np.array(_pt(MP_RIGHT_WRIST)),
            )

        spine_vec = (hip_mid[0] - nose[0], hip_mid[1] - nose[1])
        spine_len = math.sqrt(spine_vec[0] ** 2 + spine_vec[1] ** 2)
        if spine_len > 1.0:
            cos_angle = spine_vec[1] / spine_len
            features["neck_tilt_angle"] = float(
                np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            )

        # Normalised distance vectors
        left_eye = _pt(MP_LEFT_EYE_OUTER)
        right_eye = _pt(MP_RIGHT_EYE_OUTER)

        wrist_eye_dists = []
        wrist_ear_dists = []
        for wrist_idx in [MP_LEFT_WRIST, MP_RIGHT_WRIST]:
            if _vis(wrist_idx):
                wrist = _pt(wrist_idx)
                wrist_eye_dists.extend(_dist_2d(wrist, eye) for eye in [left_eye, right_eye])
                for ear_idx in [MP_LEFT_EAR, MP_RIGHT_EAR]:
                    if _vis(ear_idx):
                        wrist_ear_dists.append(_dist_2d(wrist, _pt(ear_idx)))
        if wrist_eye_dists:
            features["wrist_to_eye_ratio"] = min(wrist_eye_dists) / torso_len
        if wrist_ear_dists:
            features["wrist_to_ear_ratio"] = min(wrist_ear_dists) / torso_len

        if _vis(MP_LEFT_SHOULDER) and _vis(MP_RIGHT_SHOULDER):
            eye_y = (left_eye[1] + right_eye[1]) / 2
            features["eye_to_shoulder_ratio"] = abs(eye_y - shoulder_mid[1]) / torso_len

        hand_mouth_dists = []
        for wrist_idx in [MP_LEFT_WRIST, MP_RIGHT_WRIST]:
            if _vis(wrist_idx):
                hand_mouth_dists.append(_dist_2d(_pt(wrist_idx), nose))
        if hand_mouth_dists:
            features["hand_mouth_distance"] = min(hand_mouth_dists) / torso_len

        features["shoulder_hip_offset_ratio"] = abs(shoulder_mid[0] - hip_mid[0]) / torso_len

        if _vis(MP_LEFT_ANKLE) and _vis(MP_LEFT_HIP):
            leg_len = _dist_2d(_pt(MP_LEFT_HIP), _pt(MP_LEFT_ANKLE))
            if leg_len > 1.0:
                features["torso_leg_ratio"] = torso_len / leg_len

        if _vis(MP_LEFT_SHOULDER) and _vis(MP_RIGHT_SHOULDER):
            features["shoulder_width_ratio"] = _dist_2d(_pt(MP_LEFT_SHOULDER), _pt(MP_RIGHT_SHOULDER)) / torso_len

        if _vis(MP_LEFT_HIP) and _vis(MP_RIGHT_HIP):
            features["hip_width_ratio"] = _dist_2d(_pt(MP_LEFT_HIP), _pt(MP_RIGHT_HIP)) / torso_len

        if all(_vis(i) for i in [MP_LEFT_SHOULDER, MP_LEFT_HIP, MP_RIGHT_SHOULDER, MP_RIGHT_HIP]):
            lt = _dist_2d(_pt(MP_LEFT_SHOULDER), _pt(MP_LEFT_HIP))
            rt = _dist_2d(_pt(MP_RIGHT_SHOULDER), _pt(MP_RIGHT_HIP))
            total = lt + rt
            if total > 1.0:
                features["torso_asymmetry"] = abs(lt - rt) / total

        return features

    # ------------------------------------------------------------------
    # Spatial features (bounding-box vectoring)
    # ------------------------------------------------------------------

    def _compute_spatial_features(
        self, boxes: list[list[int]], frame_shape: tuple[int, ...]
    ) -> dict[str, float]:
        """Compute spatial features from bounding boxes.

        Returns dict with keys:
            frame_center_offset_x, frame_center_offset_y,
            bbox_coverage_ratio, bbox_aspect_ratio, person_count
        """
        fh, fw = frame_shape[:2]
        spatial: dict[str, float] = {
            "frame_center_offset_x": 0.0,
            "frame_center_offset_y": 0.0,
            "bbox_coverage_ratio": 0.0,
            "bbox_aspect_ratio": 0.0,
            "person_count": float(len(boxes)),
        }

        if not boxes:
            return spatial

        # Primary candidate (largest bbox)
        primary = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cx = (primary[0] + primary[2]) / 2.0
        cy = (primary[1] + primary[3]) / 2.0
        spatial["frame_center_offset_x"] = (cx - fw / 2.0) / fw
        spatial["frame_center_offset_y"] = (cy - fh / 2.0) / fh

        bbox_area = (primary[2] - primary[0]) * (primary[3] - primary[1])
        spatial["bbox_coverage_ratio"] = bbox_area / (fw * fh)

        bw = max(1, primary[2] - primary[0])
        bh = max(1, primary[3] - primary[1])
        spatial["bbox_aspect_ratio"] = bw / bh

        return spatial

    # ------------------------------------------------------------------
    # scikit-learn ML prediction
    # ------------------------------------------------------------------

    def _ml_predict(self, features: dict[str, float]) -> tuple[str, float]:
        try:
            # NaN (unmeasured) is coalesced to 0.0 here: the sklearn pipeline
            # was fitted on fully-measured rows and has no NaN handling.
            vector = np.array(
                [[_feature_or_zero(features, name) for name in FEATURE_NAMES]]
            )
            prediction = self._ml_classifier.predict(vector)[0]
            pose_id = str(prediction)
            confidence = 0.85
            if hasattr(self._ml_classifier, "predict_proba"):
                proba = self._ml_classifier.predict_proba(vector)[0]
                confidence = float(np.max(proba))
            return pose_id, confidence
        except Exception as exc:
            logger.debug("ML prediction failed: %s", exc)
            return ("P01", 0.0)

    # ------------------------------------------------------------------
    # Object detection (phone / book / laptop)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Object detection
    # ------------------------------------------------------------------

    @staticmethod
    def _shoulder_width_of(
        landmarks: list[tuple[float, float, float]] | None,
    ) -> float:
        """Shoulder width as the scale reference, or 0.0 when unusable."""
        return shoulder_width_of(landmarks)

    def _capture_person_context(
        self,
        all_landmarks: list[list[tuple[float, float, float]]],
        boxes: list[list[int]] | list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Per-person scale context (see ObjectContext.capture_persons)."""
        return ObjectContext.capture_persons(all_landmarks, boxes)

    def _desk_region(
        self, frame_shape: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
        """Region of the frame searched for desk objects (shared rules)."""
        ctx = self._sync_obj_context()
        return ctx.desk_region(frame_shape)

    def _classify_object_relation(
        self,
        obj_box: list[int],
        landmarks: list[tuple[float, float, float]] | None,
        shoulder_w: float | None = None,
    ) -> dict[str, Any]:
        """Spatial relation of an object to one person (shared rules)."""
        ctx = self._sync_obj_context()
        return ctx.classify_relation(
            obj_box, landmarks, shoulder_w,
            frame_shape=ctx.frame_shape,
        )

    @staticmethod
    def _box_iou(a: list[int], b: list[int]) -> float:
        """Intersection-over-union of two xyxy boxes."""
        return box_iou(a, b)

    def _relation_to_nearest_person(
        self, obj_box: list[int]
    ) -> dict[str, Any]:
        """Spatial relation of one object against the closest person."""
        ctx = self._sync_obj_context()
        return ctx.relation_to_nearest_person(obj_box)

    def _merge_persistent_objects(
        self, fresh: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Carry detections forward across throttled passes (shared rules)."""
        ctx = self._sync_obj_context()
        return ctx.merge_tracks(fresh)

    def _flatten_object_tracks(self) -> list[dict[str, Any]]:
        """Project internal tracks to the outward-facing detection shape."""
        return self._ensure_obj_context()._flatten_tracks()

    def _detect_objects(
        self,
        frame: np.ndarray,
        boxes: list[list[int]],
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Detect desk objects and describe their position relative to people.

        Region, throttle, persistence and spatial attribution all live on the
        shared ObjectContext so the CCTV backend behaves identically;
        ``boxes`` is retained only for call-site compatibility (the region is
        no longer derived per person).
        """
        ctx = self._sync_obj_context()
        return ctx.detect(frame, force=force)

    def _escalate_from_objects(
        self, pose_id: str, confidence: float,
        objects: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """Escalate a verdict using detected objects, with spatial agreement.

        Delegates to ObjectContext.escalate: which labels escalate and where
        the spatial gate sits are configuration, shared with the backend.
        """
        return ObjectContext.escalate(pose_id, confidence, objects)

    # ------------------------------------------------------------------
    # P01-P40 rule engine
    # ------------------------------------------------------------------

    def _evaluate_rules(
        self,
        all_landmarks: list[list[tuple[float, float, float]]],
        head_pose: dict[str, float],
        person_count: int,
        features: dict[str, float],
        head_anomalies: list[str],
        temporal: dict[str, float],
        spatial: dict[str, float],
        pitch_baseline: float | None = None,
        yaw_baseline: float | None = None,
        turn_ratio: float | None = None,
    ) -> tuple[str, float]:
        yaw = head_pose["yaw"]
        pitch = head_pose["pitch"]
        # Neutral head orientation for this particular student, so posture
        # rules read as deviations from their normal rather than from zero.
        pitch_ref = pitch_baseline if pitch_baseline is not None else 0.0
        yaw_ref = yaw_baseline if yaw_baseline is not None else 0.0
        pitch_delta = pitch - pitch_ref
        yaw_delta = yaw - yaw_ref

        # No persons detected.
        # Guarded by a luminance check on the frame itself.  "Absent Seat"
        # asserts something about the room, so it bypasses the stability gate
        # -- which means it must never be derived from a frame that carries no
        # image.  A dead camera produces exactly that: structurally valid,
        # entirely black, zero detections.  The capture watchdog also catches
        # this, but only after BLACK_STREAK_FRAMES dark frames, during which
        # the classifier was otherwise reporting "Absent Seat" against an
        # unplugged camera.
        if person_count == 0:
            if not frame_has_usable_image(self._last_frame):
                return ("NO_SIGNAL", 0.0)
            return ("P16", 0.95)

        # Multiple persons (P35)
        if person_count > 1:
            return ("P35", 0.92)

        lms = all_landmarks[0] if all_landmarks else []
        if len(lms) < 33:
            return ("P01", 0.70)

        def _lm(idx: int) -> tuple[float, float, float]:
            return lms[idx]

        def _visible(idx: int, thr: float = 0.5) -> bool:
            return lms[idx][2] > thr

        nose_y = _lm(MP_NOSE)[1]
        ls_y = _lm(MP_LEFT_SHOULDER)[1]
        rs_y = _lm(MP_RIGHT_SHOULDER)[1]
        shoulder_mid_y = (ls_y + rs_y) / 2.0
        shoulder_mid_x = (_lm(MP_LEFT_SHOULDER)[0] + _lm(MP_RIGHT_SHOULDER)[0]) / 2.0
        hip_mid_x = (_lm(MP_LEFT_HIP)[0] + _lm(MP_RIGHT_HIP)[0]) / 2.0
        hip_mid_y = (_lm(MP_LEFT_HIP)[1] + _lm(MP_RIGHT_HIP)[1]) / 2.0

        # Scale reference for image-plane offsets.  Shoulder width is used
        # (rather than torso length) because the shoulders are reliably
        # visible while MediaPipe frequently extrapolates the hips far
        # outside the frame.  Every horizontal/vertical pixel threshold below
        # is expressed as a fraction of this, so the result holds at any
        # resolution, subject distance or camera mounting height.
        shoulder_w = abs(_lm(MP_LEFT_SHOULDER)[0] - _lm(MP_RIGHT_SHOULDER)[0])
        have_scale = shoulder_w >= perf_config.MIN_SCALE_PX

        # ── 3D Head Pose anomalies (from HeadPoseEstimator) ──────────────
        if "P12_LOOKING_AWAY" in head_anomalies:
            return ("P12", 0.88)
        if "P14_LOOKING_DOWN" in head_anomalies:
            return ("P14", 0.82)

        # ── Head turns (yaw-based) ───────────────────────────────────────
        # Measured against this student's neutral yaw.  A camera that is not
        # centred on the seat produces a constant yaw offset, and an absolute
        # cut would either miss real turns or flag every frame.
        if yaw_delta < -60:
            return ("P04", 0.88)
        if yaw_delta > 60:
            return ("P05", 0.88)
        if yaw_delta < -30:
            return ("P02", 0.82)
        if yaw_delta > 30:
            return ("P03", 0.82)

        # ── Head down / on-desk / up (pitch) ─────────────────────────────
        # Normalised by torso length (nose -> hip) so the result does not
        # depend on resolution, subject distance or camera mounting height.
        # Raw pixel thresholds here flagged 100% of live frames as P07.
        if _visible(MP_NOSE) and _visible(MP_LEFT_SHOULDER) and _visible(MP_RIGHT_SHOULDER):
            torso_ref = abs(hip_mid_y - nose_y)
            if torso_ref >= perf_config.MIN_TORSO_PX:
                pitch_ratio = (nose_y - shoulder_mid_y) / torso_ref

                # Head on desk is the most severe, so it is tested first.
                # (Previously P18 sat behind P06's looser test and could
                #  never be reached.)
                if pitch_ratio > perf_config.HEAD_ON_DESK_RATIO:
                    return ("P18", 0.85)
                if pitch_ratio > perf_config.HEAD_DOWN_RATIO:
                    return ("P06", 0.82)
                if pitch_ratio < perf_config.HEAD_UP_RATIO:
                    return ("P07", 0.78)

        # ── Hand to mouth (geometric) ────────────────────────────────────
        # Guarded: an unmeasurable hand must not read as "distance 0".
        if _feature_lt(features, "hand_mouth_distance", 0.12):
            return ("P19", 0.78)

        # ── Lean left / right (geometric neck_tilt_angle) ────────────────
        neck_tilt = features.get("neck_tilt_angle")
        neck_tilt = 0.0 if neck_tilt is None or neck_tilt != neck_tilt else neck_tilt
        if neck_tilt > 15.0:
            if _visible(MP_LEFT_SHOULDER) and _visible(MP_RIGHT_SHOULDER) and _visible(MP_LEFT_HIP) and _visible(MP_RIGHT_HIP):
                dy = shoulder_mid_y - hip_mid_y
                dx = shoulder_mid_x - hip_mid_x
                if dy > 0:
                    lean_ratio = dx / dy
                    if abs(lean_ratio) > 0.6:
                        return ("P31", 0.85)
                    if lean_ratio < -0.35:
                        return ("P08", 0.78)
                    if lean_ratio > 0.35:
                        return ("P09", 0.78)

        # ── Extreme side lean ────────────────────────────────────────────
        if _feature_gt(features, "shoulder_hip_offset_ratio", 0.35):
            return ("P31", 0.85)

        # ── Body turn ────────────────────────────────────────────────────
        # Nose offset from the shoulder midpoint, in shoulder-widths.  This
        # ratio is deliberately NOT thresholded: turning the torso foreshortens
        # the shoulders, so the denominator shrinks just as the numerator
        # grows, and the measured normal band spanned -0.86..+0.56.  The cut is
        # a robust z-score against the student's own normal instead.  `turn_ratio`
        # was already computed in classify() and fed to the buffer, so the two
        # readings cannot disagree.
        if turn_ratio is not None:
            turn_z = self._temporal_buffer.turn_deviation_z(turn_ratio)
            if turn_z > perf_config.BODY_TURN_Z:
                # Nose to the image-right is the student's right in a
                # non-mirrored view, so a positive ratio is a right turn.
                return ("P11" if turn_ratio > 0 else "P10", 0.80)

        # ── Forward hunch / backward lean ────────────────────────────────
        # Relative to the student's own neutral pitch.  The absolute 25 deg
        # cut sat inside the normal band (median 21 deg, p95 31 deg), so
        # ordinary posture was reported as "Forward Hunch" ~11% of the time.
        if pitch_delta > perf_config.HUNCH_PITCH_DEG:
            return ("P33", 0.75)
        if pitch_delta < perf_config.LEAN_BACK_PITCH_DEG:
            return ("P32", 0.75)

        # ── Standing up ──────────────────────────────────────────────────
        # Single normalised test.  A second, hand-rolled version of the same
        # torso/leg ratio used to follow this one.
        if _feature_gt(features, "torso_leg_ratio", 1.5):
            return ("P15", 0.82)

        # ── Hand to mouth (pixel fallback) ───────────────────────────────
        # REMOVED.  This was a second, absolute-pixel test for P19 that sat
        # below the normalised `hand_mouth_distance` rule above, so it only
        # ever added resolution-dependent false positives.  The feature-based
        # rule already covers this case with a torso-normalised threshold.

        # ── Raising hand ─────────────────────────────────────────────────
        # How far the wrist sits above the shoulder, in shoulder-widths.
        if have_scale:
            for wrist_idx, sh_idx in ((MP_LEFT_WRIST, MP_LEFT_SHOULDER),
                                      (MP_RIGHT_WRIST, MP_RIGHT_SHOULDER)):
                if _visible(wrist_idx) and _visible(sh_idx):
                    rise = (_lm(sh_idx)[1] - _lm(wrist_idx)[1]) / shoulder_w
                    if rise > perf_config.RAISE_HAND_RATIO:
                        return ("P22", 0.78)

        # ── Writing under desk ───────────────────────────────────────────
        # How far the wrist sits below the hip, in shoulder-widths.
        if have_scale:
            for wrist_idx, hip_idx in ((MP_LEFT_WRIST, MP_LEFT_HIP),
                                       (MP_RIGHT_WRIST, MP_RIGHT_HIP)):
                if _visible(wrist_idx) and _visible(hip_idx):
                    drop = (_lm(wrist_idx)[1] - _lm(hip_idx)[1]) / shoulder_w
                    if drop > perf_config.HAND_UNDER_DESK_RATIO:
                        return ("P21", 0.70)

        # ── Crossed arms ─────────────────────────────────────────────────
        if (_visible(MP_LEFT_WRIST) and _visible(MP_RIGHT_WRIST)
                and _visible(MP_LEFT_SHOULDER) and _visible(MP_RIGHT_SHOULDER)):
            if _lm(MP_LEFT_WRIST)[0] > _lm(MP_RIGHT_SHOULDER)[0] and _lm(MP_RIGHT_WRIST)[0] < _lm(MP_LEFT_SHOULDER)[0]:
                return ("P30", 0.75)

        # ── Head oscillation (yaw history) ───────────────────────────────
        if temporal.get("yaw_variance", 0) > 400:
            return ("P36", 0.78)

        # ── Looking at neighbor ──────────────────────────────────────────
        # Relative to the student's neutral yaw, and deliberately low
        # confidence: this is a weak signal that must not survive the gate on
        # its own unless it persists.
        if abs(yaw_delta) > 20:
            return ("P20", 0.65)

        # ── Default: normal sitting ──────────────────────────────────────
        return ("P01", 0.95)

    # ------------------------------------------------------------------
    # Flag builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_flags(pose_id: str) -> dict[str, bool]:
        flag_map = {
            "head_down": {"P06", "P18"},
            "body_turn": {"P10", "P11", "P37"},
            "excessive_lean": {"P08", "P09", "P31"},
            "multi_person": {"P35", "P39"},
            "phone_detected": {"P12"},
            "note_peeking": {"P13"},
            "talking": {"P14"},
            "standing": {"P15"},
            "absent": {"P16"},
            "hand_to_mouth": {"P19"},
            "hand_under_table": {"P21"},
            "raised_hand": {"P22"},
            "crossed_arms": {"P30"},
            "head_oscillation": {"P36"},
            "looking_away": {"P12"},
            "looking_down": {"P14"},
        }
        return {flag: (pose_id in poses) for flag, poses in flag_map.items()}
