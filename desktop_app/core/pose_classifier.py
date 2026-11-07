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

from desktop_app.core import metric_depth, perf_config
from desktop_app.core.feature_schema import (
    FEATURE_NAMES as _SCHEMA_FEATURE_NAMES,
    GEOMETRIC_FEATURES,
    feature_dict_from_components,
    validate_feature_names,
    vector_from_dict,
)
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
from desktop_app.core.seat_anchor import (
    BoundaryClipGuard,
    CandidateTracker,
    SeatAnchorRegistry,
)
from desktop_app.core.torch_compile_utils import install_safe_compile
from desktop_app.core.vision_utils import configure_opencv_threads

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
    # Produced by HeadPoseEstimator's PITCH_LOOKING_DOWN_THRESHOLD anomaly
    # (head_pose.py), NOT by any talking/whispering test -- the old
    # "Talking / Whispering" label misdescribed every alert this id ever
    # produced.  See _evaluate_rules, which returns P14 for
    # "P14_LOOKING_DOWN" and for nothing else.
    "P14": "Looking Down",
    "P15": "Standing Up",
    "P16": "Absent Seat",
    "P18": "Head on Desk",
    "P19": "Hand to Mouth",
    "P20": "Looking at Neighbor",
    "P21": "Writing Under Desk",
    "P22": "Raising Hand",
    "P23": "Excessive Fidgeting",
    "P25": "Hidden Face",
    "P26": "Earbuds Detected",
    "P28": "Frequent Glancing",
    "P29": "Chin Rest",
    "P30": "Crossed Arms",
    "P31": "Extreme Side Lean",
    # Wrist-to-phone proximity verdict from the concurrent desk-object pass
    # (see set_proximity / _phone_proximity_hit).  Deliberately a separate id
    # from P31 (side lean) and P38 (object merely present with spatial
    # agreement): this one asserts use, and only fires when the wrist is at
    # the phone.
    "P31_PHONE_USABILITY": "Phone Usability",
    "P32": "Backward Lean",
    "P33": "Forward Hunch",
    "P35": "Multiple Persons",
    "P36": "Head Oscillation",
    "P38": "Unusual Object",
}

# Ordered feature names for the scikit-learn classifier input vector.
# The canonical definition lives in feature_schema so the logger, trainer and
# runtime cannot drift apart.  Previously this list held only the 12 geometric
# features while the trainer used 50, which made every trained model fail at
# prediction time and silently fall back to the rule engine.
FEATURE_NAMES: list[str] = list(_SCHEMA_FEATURE_NAMES)
GEOMETRIC_FEATURE_NAMES: list[str] = list(GEOMETRIC_FEATURES)

# Verdicts that describe an object or the room rather than the pose.  A
# pose-only classifier must never invent these; they stay with the rule and
# desk-object engines.
_ML_OBJECT_VERDICTS = frozenset(
    {"P13", "P16", "P26", "P31_PHONE_USABILITY", "P35", "P38"}
)

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


# ---------------------------------------------------------------------------
# Positive neutral posture filter
# ---------------------------------------------------------------------------
# Ordinary exam behaviour that shares a signature with a threat: writing
# looks like looking down, a hand at the chin looks like phone use, a glance
# at the clock looks like looking away.  Each state, when active, suppresses
# the specific verdicts it would otherwise produce, so a live demo of a
# student simply taking an exam does not light the HUD up in red.

NEUTRAL_WRITING = "NEUTRAL_WRITING"
NEUTRAL_THINKING = "NEUTRAL_THINKING"
NEUTRAL_GLANCE = "NEUTRAL_GLANCE"

NEUTRAL_LABELS: dict[str, str] = {
    NEUTRAL_WRITING: "Writing",
    NEUTRAL_THINKING: "Thinking",
    NEUTRAL_GLANCE: "Glancing Up",
}

# Threat verdicts a neutral state overrides.  P14 is the Looking Down
# verdict produced by HeadPoseEstimator's pitch test; P31_PHONE_USABILITY is
# the wrist-proximity phone verdict.  Nothing else is suppressed -- a neutral
# posture must never hide a second person, a stand-up, or an absent seat.
_NEUTRAL_SUPPRESSES: frozenset[str] = frozenset({
    "P14", "P31_PHONE_USABILITY",
})

# Object labels that make a hand-at-cheek read as device use rather than
# thinking.  'cell_phone' and 'device' as specified; 'phone' is the COCO
# fallback's spelling of the same class.
_NEUTRAL_DEVICE_LABELS: frozenset[str] = frozenset({
    "cell_phone", "device", "phone",
})

# Device labels whose P38 escalation must also clear the metric 0.15 m
# wrist-distance gate in the live path.  The pixel threshold remains only as
# a legacy fallback for callers without solvePnP depth.  Earbuds are
# deliberately excluded: P26 is an ear-proximity verdict, not a wrist one.
_DEVICE_WRIST_GATED: frozenset[str] = frozenset({
    "cell_phone", "phone", "device", "laptop", "tablet", "headphones",
})


def _visible_xy(
    keypoints: list[tuple[float, float, float]] | None, idx: int
) -> tuple[float, float] | None:
    """Landmark *idx* as (x, y) when it is visible enough to reason about."""
    if not keypoints or len(keypoints) <= idx:
        return None
    x, y, vis = keypoints[idx]
    if vis is not None and vis < 0.5:
        return None
    if x != x or y != y:  # NaN
        return None
    return (float(x), float(y))


def check_neutral_alignment(
    keypoints: list[tuple[float, float, float]] | None,
    yaw: float,
    pitch: float,
    roll: float,
    desk_objects: list[dict[str, Any]] | None,
    bbox: list[int] | None = None,
    temporal: "_TemporalBuffer | None" = None,
    depth_info: dict[str, Any] | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> str | None:
    """Classify the current frame as a benign exam state, or ``None``.

    Parameters
    ----------
    keypoints
        MediaPipe 33-point landmarks of the primary candidate as
        ``(x, y, visibility)`` tuples.  Wrists, mouth and nose are the
        landmarks the three tests read.
    yaw, pitch, roll
        3D head pose in degrees from ``HeadPoseEstimator``.  Only *pitch* is
        read today; yaw and roll are part of the contract so a neutral
        sideways-glance test can be added without changing callers.
    desk_objects
        detections with ``label``/``box``/``confidence`` from the concurrent
        desk-object pass and/or the anomaly-triggered desk-region pass.
    bbox
        Candidate bounding box, needed for the "wrist in the lower 40%"
        writing test.
    temporal
        Sliding window that carries wrist history (micro-oscillation) and
        the glance clock.  Without it the stateful tests cannot run and
        only the stateless parts are evaluated.
    depth_info
        ``metric_depth.estimate_candidate_depth()`` output.  When present,
        the "no device near the hand" test uses the 0.15 m metric radius
        instead of the legacy pixel radius, so thinking is only declared
        when the hand is genuinely far from a phone.
    frame_shape
        Video frame shape, needed to scale the PnP intrinsics for the metric
        device test.

    States, in priority order:

    ``NEUTRAL_WRITING``
        pitch in [-35, -15] degrees (head down onto the page), wrist inside
        the lower 40% of the candidate box, and the wrist trace showing
        micro-oscillation (pen strokes).

    ``NEUTRAL_THINKING``
        a visible wrist within 50 px of the cheek/chin while no device-class
        object sits within 50 px of that wrist.

    ``NEUTRAL_GLANCE``
        pitch in [+10, +30] degrees continuously for under 5 seconds.
    """
    lms = keypoints or []
    has_lms = len(lms) >= 33

    # Both clocks tick every frame, in or out of any neutral band, so an
    # early return below can never freeze the glance timer or mix pre-band
    # wrist motion into the current oscillation window.
    if temporal is not None:
        temporal.note_wrist(*_wrist_or_none(lms))
        glance_elapsed = temporal.update_glance(
            perf_config.NEUTRAL_GLANCE_PITCH_MIN
            <= pitch
            <= perf_config.NEUTRAL_GLANCE_PITCH_MAX,
            time.monotonic(),
        )
    else:
        glance_elapsed = None

    # ── NEUTRAL_WRITING ────────────────────────────────────────────────
    writing_band = (
        perf_config.NEUTRAL_WRITING_PITCH_MIN
        <= pitch
        <= perf_config.NEUTRAL_WRITING_PITCH_MAX
    )
    if writing_band and temporal is not None and bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        band_top = y1 + (1.0 - perf_config.NEUTRAL_WRIST_LOWER_RATIO) * (y2 - y1)
        for wrist_idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
            wrist = _visible_xy(lms, wrist_idx)
            if wrist is None:
                continue
            if band_top <= wrist[1] <= y2 and temporal.wrist_micro_oscillation():
                return NEUTRAL_WRITING

    if not has_lms:
        return None

    # ── NEUTRAL_THINKING ───────────────────────────────────────────────
    cheek = _cheek_anchor(lms)
    if cheek is not None:
        for wrist_idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
            wrist = _visible_xy(lms, wrist_idx)
            if wrist is None:
                continue
            if _dist_2d(wrist, cheek) > perf_config.NEUTRAL_THINK_CHEEK_PX:
                continue
            # Hand at the cheek is only innocent when nothing the hand could
            # be holding is anywhere near it.  In the live path this is a
            # metric 15 cm test; the pixel radius remains only as a fallback
            # for callers without solvePnP depth.
            if not any(
                obj.get("label") in _NEUTRAL_DEVICE_LABELS
                and _device_near_wrist(
                    obj, wrist, lms, depth_info, frame_shape
                )
                for obj in (desk_objects or [])
            ):
                return NEUTRAL_THINKING

    # ── NEUTRAL_GLANCE ─────────────────────────────────────────────────
    if glance_elapsed is not None and (
        glance_elapsed < perf_config.NEUTRAL_GLANCE_MAX_SEC
    ):
        return NEUTRAL_GLANCE

    return None


def _wrist_or_none(
    lms: list[tuple[float, float, float]]
) -> tuple[float | None, float | None]:
    """The most visible wrist, for the oscillation history."""
    best: tuple[float, float] | None = None
    best_vis = -1.0
    for idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
        if len(lms) <= idx:
            continue
        x, y, vis = lms[idx]
        if vis is not None and vis > best_vis and x == x and y == y:
            best_vis = vis
            best = (float(x), float(y))
    if best is None or best_vis < 0.5:
        return (None, None)
    return best


def _cheek_anchor(
    lms: list[tuple[float, float, float]]
) -> tuple[float, float] | None:
    """Cheek/chin position: mouth midpoint, nose as a fallback.

    MediaPipe's 33-point pose has no explicit cheek or chin landmark, so the
    mouth corners are the closest measured proxy; the nose is used when the
    mouth is occluded by the very hand we are testing for.
    """
    left = _visible_xy(lms, MP_MOUTH_LEFT)
    right = _visible_xy(lms, MP_MOUTH_RIGHT)
    if left is not None and right is not None:
        return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    return _visible_xy(lms, MP_NOSE)


def _obj_within(
    obj_box: Any, point: tuple[float, float], max_px: float
) -> bool:
    """True when *point* is within *max_px* of box *obj_box* (or inside it)."""
    if not obj_box or len(obj_box) != 4:
        return False
    ox1, oy1, ox2, oy2 = (float(v) for v in obj_box)
    px, py = point
    dx = max(ox1 - px, 0.0, px - ox2)
    dy = max(oy1 - py, 0.0, py - oy2)
    return math.sqrt(dx * dx + dy * dy) <= max_px


def _metric_box_distance_m(
    point: tuple[float, float],
    obj_box: Any,
    depth_info: dict[str, Any] | None,
    frame_shape: tuple[int, ...] | None,
) -> float | None:
    """Metric distance from a pixel point to a box centre on the depth plane.

    Returns None when the solvePnP depth is unavailable; callers must fail
    closed rather than silently substitute a pixel threshold in the live
    path.
    """
    if not depth_info or depth_info.get("depth_m") is None:
        return None
    if not obj_box or len(obj_box) != 4:
        return None
    depth_m = float(depth_info["depth_m"])
    intrinsics = depth_info.get("camera_matrix")
    object_point = metric_depth.box_center_point_m(
        obj_box, depth_m, intrinsics, frame_shape
    )
    if object_point is None:
        return None
    wrist_point = metric_depth.pixel_to_point_m(
        point[0], point[1], depth_m, intrinsics, frame_shape
    )
    return float(np.linalg.norm(wrist_point - object_point))


def _device_near_wrist(
    obj: dict[str, Any],
    wrist: tuple[float, float],
    landmarks: list[tuple[float, float, float]] | None,
    depth_info: dict[str, Any] | None,
    frame_shape: tuple[int, ...] | None,
) -> bool:
    """True when a device is within the neutral-thinking exclusion radius.

    The live measurement is metric (``NEUTRAL_THINK_DEVICE_M``, 15 cm).  The
    pixel fallback exists only for callers that have no depth solve -- the
    backend and unit tests -- and is never used when a metric distance was
    already measured for the object.
    """
    if str(obj.get("label", "")).lower() not in _NEUTRAL_DEVICE_LABELS:
        return False
    metric = obj.get("wrist_dist_m")
    if metric is not None:
        try:
            return float(metric) <= perf_config.NEUTRAL_THINK_DEVICE_M
        except (TypeError, ValueError):
            pass
    measured = _metric_box_distance_m(
        wrist, obj.get("box"), depth_info, frame_shape
    )
    if measured is not None:
        return measured <= perf_config.NEUTRAL_THINK_DEVICE_M
    return _obj_within(
        obj.get("box"), wrist, perf_config.NEUTRAL_THINK_NO_OBJ_PX
    )


def _cheek_contact_active(
    keypoints: list[tuple[float, float, float]] | None,
    desk_objects: list[dict[str, Any]] | None,
    depth_info: dict[str, Any] | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> bool:
    """True when a visible wrist rests at the cheek with no device near it.

    The same geometry as the NEUTRAL_THINKING test, exposed separately so
    the P29 chin-rest clock ticks on the identical measurement the neutral
    filter uses -- the two can never disagree about what "hand at cheek"
    means.
    """
    lms = keypoints or []
    if len(lms) < 33:
        return False
    cheek = _cheek_anchor(lms)
    if cheek is None:
        return False
    for wrist_idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
        wrist = _visible_xy(lms, wrist_idx)
        if wrist is None:
            continue
        if _dist_2d(wrist, cheek) > perf_config.NEUTRAL_THINK_CHEEK_PX:
            continue
        if not any(
            _device_near_wrist(
                obj, wrist, lms, depth_info, frame_shape
            )
            for obj in (desk_objects or [])
        ):
            return True
    return False


def _merge_detections(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    iou_floor: float = 0.30,
) -> list[dict[str, Any]]:
    """Union two detection lists without double-counting the same object.

    The anomaly-triggered desk-region pass and the concurrent candidate-crop
    pass can both see the same phone; reporting it twice would corrupt the
    wrist-distance comparison (a duplicate would always be "nearest").  A
    same-label pair overlapping above *iou_floor* is one object -- the
    higher-confidence box wins, since it is the better localisation.
    """
    merged: list[dict[str, Any]] = [dict(d) for d in primary]
    for det in secondary:
        dup_idx = None
        for i, existing in enumerate(merged):
            if existing.get("label") != det.get("label"):
                continue
            if box_iou(existing.get("box") or [], det.get("box") or []) >= iou_floor:
                dup_idx = i
                break
        if dup_idx is None:
            merged.append(dict(det))
        elif float(det.get("confidence", 0.0)) > float(
            merged[dup_idx].get("confidence", 0.0)
        ):
            merged[dup_idx] = dict(det)
    return merged


def _anchor_items(
    scut_head_anchors: Any,
) -> list[tuple[list[int], str | None]]:
    """Normalise a registry / dict list / anchor list to ``(box, seat_id)``.

    The seat id is carried through so a validated person box can be attributed
    to its locked seat without re-deriving it from a mutable box.  Plain lists
    of boxes (the unit-test shape) produce items with ``seat_id=None``.
    """
    if scut_head_anchors is None:
        return []
    anchors = getattr(scut_head_anchors, "anchors", None)
    if anchors is None and isinstance(scut_head_anchors, dict):
        anchors = scut_head_anchors.get("anchors")
    if anchors is None:
        anchors = scut_head_anchors
    items: list[tuple[list[int], str | None]] = []
    try:
        for anchor in anchors:
            box = getattr(anchor, "box", None)
            seat_id = getattr(anchor, "seat_id", None)
            if isinstance(anchor, dict):
                box = anchor.get("box")
                seat_id = anchor.get("seat_id")
            elif box is None:
                # A bare [x1, y1, x2, y2] entry.
                box = anchor
            if box is not None and len(box) == 4:
                items.append((
                    [int(v) for v in box],
                    str(seat_id) if seat_id is not None else None,
                ))
    except TypeError:
        return []
    return items


def _anchor_boxes(
    scut_head_anchors: Any,
) -> list[list[int]]:
    """Normalise a registry / dict list / anchor list to xyxy head boxes."""
    return [box for box, _seat_id in _anchor_items(scut_head_anchors)]


def _point_box_distance(
    point: tuple[float, float], box: list[int] | list[float]
) -> float:
    """Euclidean distance from *point* to *box* (0 when inside)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    px, py = point
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.sqrt(dx * dx + dy * dy)


def mask_face_outside_head(
    keypoints: Any,
    head_box: list[int] | list[float] | None,
    margin_ratio: float | None = None,
) -> int:
    """Blank face keypoints that cannot belong to the anchored head.

    Continuous SCUT-HEAD boxes are the spatial anchor for the skeleton.  If a
    person box overlaps a neighbouring student, MediaPipe can place the face
    landmarks on the wrong head; those landmarks then poison head-pose and
    neutral-posture rules.  Face points outside the matched head box (expanded
    by ``HEAD_KEYPOINT_MARGIN_RATIO``) have their visibility set to zero, so
    downstream rules treat them as unmeasured while body keypoints remain
    untouched.
    """
    if not head_box or len(head_box) != 4 or keypoints is None:
        return 0
    try:
        hx1, hy1, hx2, hy2 = (float(v) for v in head_box)
    except (TypeError, ValueError):
        return 0
    if hx2 <= hx1 or hy2 <= hy1:
        return 0
    ratio = (
        perf_config.HEAD_KEYPOINT_MARGIN_RATIO
        if margin_ratio is None else float(margin_ratio)
    )
    margin = max(hx2 - hx1, hy2 - hy1) * max(0.0, ratio)
    x1, y1, x2, y2 = hx1 - margin, hy1 - margin, hx2 + margin, hy2 + margin
    face_indices = range(11) if len(keypoints) >= 33 else range(5)
    masked = 0
    for idx in face_indices:
        if idx >= len(keypoints):
            break
        point = keypoints[idx]
        try:
            px, py = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            try:
                point[2] = 0.0
            except (TypeError, IndexError):
                keypoints[idx] = (px, py, 0.0)
            masked += 1
    return masked


def _keypoint_xy_vis(
    keypoints: Any, idx: int
) -> tuple[float, float, float] | None:
    """Return ``(x, y, visibility)`` for a keypoint, tolerating numpy rows."""
    if keypoints is None:
        return None
    try:
        if len(keypoints) <= idx:
            return None
        point = keypoints[idx]
        x, y = float(point[0]), float(point[1])
        vis = float(point[2]) if len(point) > 2 else 1.0
    except (TypeError, ValueError, IndexError):
        return None
    if x != x or y != y or vis != vis:
        return None
    return x, y, vis


def validate_candidate_boundaries(
    person_box: list[int] | None,
    keypoints: Any,
    scut_head_anchors: Any,
) -> dict[str, Any]:
    """Validate a person box against the classroom's head/seat anchors.

    A candidate must contain a head in the upper 30% of its box.  The head
    evidence is either a SCUT-HEAD anchor whose centre falls in that band or,
    for a person who entered after initialisation, a visible nose keypoint in
    the same band.  Boxes without either are clipped torso/leg fragments and
    are rejected, which is what stops a pseudo-clip from being counted as an
    extra person (P35) or as a new candidate.

    Limbs that extend outside the box are still attributed to the candidate
    while they are within ``LIMB_ATTRIBUTION_SHOULDER_WIDTHS`` shoulder-widths
    of its edge.  The function reports how many were attributed/rejected so
    callers can decide whether a landmark is safe to use in a rule.

    Returns a dict with ``valid``, ``head_found``, ``seat_id``,
    ``head_box``, ``upper_region``, ``limbs_attributed``, ``limbs_rejected``
    and ``reason``.  ``head_box`` is the matched SCUT-HEAD box (live or
    initialised) that the caller must use to restrict face landmarks.
    """
    result: dict[str, Any] = {
        "valid": False,
        "head_found": False,
        "seat_id": None,
        "head_box": None,
        "upper_region": None,
        "limbs_attributed": 0,
        "limbs_rejected": 0,
        "reason": "",
    }
    if person_box is None or len(person_box) != 4:
        result["reason"] = "no_box"
        return result
    x1, y1, x2, y2 = (float(v) for v in person_box)
    if x2 <= x1 or y2 <= y1:
        result["reason"] = "degenerate_box"
        return result

    upper_bottom = y1 + perf_config.HEAD_UPPER_BODY_RATIO * (y2 - y1)
    upper = [x1, y1, x2, upper_bottom]
    result["upper_region"] = [int(v) for v in upper]

    anchors = _anchor_items(scut_head_anchors)
    if anchors:
        best_seat: list[int] | None = None
        best_seat_id: str | None = None
        best_score = -1.0
        for box, seat_id in anchors:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            inside = x1 <= cx <= x2 and y1 <= cy <= upper_bottom
            overlap = box_iou(box, [int(v) for v in upper])
            if not inside and overlap <= 0.0:
                continue
            # A head centre inside the upper band is the strongest evidence;
            # overlap is the fallback when the box is partly clipped.
            score = (2.0 if inside else 0.0) + overlap
            if score > best_score:
                best_seat, best_seat_id = box, seat_id
                best_score = score
        if best_seat is not None:
            result["head_found"] = True
            result["head_box"] = list(best_seat)
            result["seat_id"] = best_seat_id
            # Resolve the seat id through the registry when the caller passed
            # a registry rather than a live-head list.
            if result["seat_id"] is None and isinstance(
                scut_head_anchors, SeatAnchorRegistry
            ):
                for anchor in scut_head_anchors.anchors:
                    if list(anchor.box) == list(best_seat):
                        result["seat_id"] = anchor.seat_id
                        break
            result["reason"] = "head_anchor"
        else:
            # A person who entered after initialisation has no anchor yet.
            # A visible nose in the upper band is still real head evidence.
            nose = _keypoint_xy_vis(keypoints, MP_NOSE)
            if nose is not None and nose[2] > 0.5:
                nx, ny = nose[0], nose[1]
                if x1 <= nx <= x2 and y1 <= ny <= upper_bottom:
                    result["head_found"] = True
                    result["reason"] = "keypoint_nose"
            if not result["head_found"]:
                result["reason"] = "no_head_in_upper_band"
                return result
    else:
        # No anchors yet (initialisation unavailable): do not invent a
        # rejection rule.  The legacy pipeline had no head evidence at all.
        result["head_found"] = True
        result["reason"] = "no_anchors"

    # Limb attribution.  Use MediaPipe's 33-point layout when available,
    # otherwise COCO-17.  The wrist/elbow points are the ones that decide
    # reach rules, so a point far outside the box must not be silently
    # attributed to this candidate.
    shoulder_w = 0.0
    shoulder_points: list[tuple[float, float]] = []
    limb_indices: tuple[int, ...]
    if keypoints is not None and len(keypoints) >= 33:
        left = _keypoint_xy_vis(keypoints, MP_LEFT_SHOULDER)
        right = _keypoint_xy_vis(keypoints, MP_RIGHT_SHOULDER)
        if left and right and left[2] > 0.5 and right[2] > 0.5:
            shoulder_w = abs(left[0] - right[0])
            shoulder_points = [(left[0], left[1]), (right[0], right[1])]
        limb_indices = (
            MP_LEFT_WRIST, MP_RIGHT_WRIST, MP_LEFT_ELBOW, MP_RIGHT_ELBOW,
        )
    elif keypoints is not None and len(keypoints) >= 17:
        left = _keypoint_xy_vis(keypoints, 5)
        right = _keypoint_xy_vis(keypoints, 6)
        if left and right and left[2] > 0.5 and right[2] > 0.5:
            shoulder_w = abs(left[0] - right[0])
            shoulder_points = [(left[0], left[1]), (right[0], right[1])]
        limb_indices = (9, 10, 7, 8)  # COCO wrists/elbows
    else:
        limb_indices = ()

    margin = (
        perf_config.LIMB_ATTRIBUTION_SHOULDER_WIDTHS * shoulder_w
        if shoulder_w > 0 else 0.0
    )
    # Only the candidate's *matched* head anchor counts.  Using every
    # classroom anchor would attribute a wrist to this candidate merely
    # because a neighbouring student's head is nearby, which is exactly the
    # false multi-person attribution this function exists to prevent.
    matched_head = result.get("head_box")
    anchor_boxes = [list(matched_head)] if matched_head else []
    for idx in limb_indices:
        point = _keypoint_xy_vis(keypoints, idx)
        if point is None or point[2] <= 0.5:
            continue
        px, py = point[0], point[1]
        if x1 <= px <= x2 and y1 <= py <= y2:
            continue
        # Attribute a limb by its distance to the nearest SCUT-HEAD anchor
        # or shoulder joint, exactly as the requirement specifies.  A wrist
        # reaching out of the person box is still this candidate's hand when
        # it is within 1.2 shoulder-widths of the head/shoulder.
        distances = [_point_box_distance((px, py), box) for box in anchor_boxes]
        distances.extend(
            math.hypot(px - sx, py - sy) for sx, sy in shoulder_points
        )
        if distances:
            dist = min(distances)
        else:
            # No anchor evidence at all: preserve the legacy box-edge rule
            # for callers running without SCUT-HEAD.
            dist = _point_box_distance((px, py), [x1, y1, x2, y2])
        if dist <= margin:
            result["limbs_attributed"] += 1
        else:
            result["limbs_rejected"] += 1

    result["valid"] = True
    return result


class _TemporalBuffer:
    """Sliding-window motion history buffer for a single candidate track."""

    __slots__ = (
        "_yaw_hist", "_pitch_hist", "_bbox_hist", "_anomaly_streak",
        "_pose_streak_id", "_pose_streak_len",
        "_pitch_baseline", "_yaw_baseline",
        "_turn_baseline", "_turn_mad",
        "_wrist_hist", "_glance_start",
        "_torso_hist", "_glance_events", "_yaw_averted_prev",
        "_cheek_contact_start",
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
        # Wrist positions for the neutral-writing micro-oscillation test, and
        # the start time of a continuous in-band positive pitch for the
        # neutral-glance test.  Both are written by the neutral posture
        # engine each frame (see PoseClassifier.classify).
        self._wrist_hist: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=max(2, perf_config.NEUTRAL_WRIST_OSC_WINDOW)
        )
        self._glance_start: float | None = None
        # Torso-centre trace (normalised by shoulder width) for the P23
        # fidget test; a rolling window of averted-yaw event timestamps for
        # P28; a clock for continuous cheek contact for P29.
        self._torso_hist: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=perf_config.FIDGET_WINDOW
        )
        self._glance_events: collections.deque[float] = collections.deque(
            maxlen=64
        )
        self._yaw_averted_prev: bool = False
        self._cheek_contact_start: float | None = None

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

    # ── Neutral posture helpers ────────────────────────────────────────

    def note_wrist(self, x: float | None, y: float | None) -> None:
        """Record this frame's wrist position for the oscillation test.

        ``None`` clears the history: a wrist that leaves the frame mid-stroke
        must not have its pre-gap motion counted as current oscillation.
        """
        if x is None or y is None or x != x or y != y:
            self._wrist_hist.clear()
            return
        self._wrist_hist.append((x, y))

    def wrist_micro_oscillation(self) -> bool:
        """True when the recorded wrist trace shows pen-stroke-scale motion.

        A writing hand alternates direction every few frames over a small
        distance; a hand resting on the cheek barely moves, and a hand
        reaching for a phone travels one way and stops.  Both conditions are
        required -- enough peak-to-peak travel to be a real movement, and at
        least two direction reversals to distinguish oscillation from a
        single reach.
        """
        hist = list(self._wrist_hist)
        if len(hist) < 4:
            return False
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        travel = max(xs) - min(xs)
        # Vertical travel counts too: small downward-tilted writing strokes
        # are often mostly vertical in the image plane.
        travel = max(travel, max(ys) - min(ys))
        if travel < perf_config.NEUTRAL_WRIST_OSC_PX:
            return False

        reversals = 0
        prev_dir = 0
        for i in range(1, len(hist)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            # Project onto the axis carrying most of the motion.
            if abs(dx) >= abs(dy):
                step = 1 if dx > 0 else (-1 if dx < 0 else 0)
            else:
                step = 1 if dy > 0 else (-1 if dy < 0 else 0)
            if step == 0:
                continue
            if prev_dir != 0 and step != prev_dir:
                reversals += 1
            prev_dir = step
        return reversals >= 2

    def update_glance(self, pitch_in_band: bool, now: float) -> float | None:
        """Track how long a positive-pitch glance has been held (seconds).

        Returns ``None`` whenever the pitch leaves the neutral-glance band,
        and ``0.0`` on the first in-band frame, so the caller can tell "no
        glance" from "glance just began".  Only one continuous upward look
        is ever measured: leaving the band resets the clock.
        """
        if not pitch_in_band:
            self._glance_start = None
            return None
        if self._glance_start is None:
            self._glance_start = now
            return 0.0
        return max(0.0, now - self._glance_start)

    # ── P23 / P28 / P29 helpers ────────────────────────────────────────

    def note_torso(self, cx: float, cy: float) -> None:
        """Record the torso centre in shoulder-width units for P23."""
        self._torso_hist.append((cx, cy))

    def fidget_displacement(self) -> float:
        """Mean per-frame torso-centre displacement over the window.

        Expressed in shoulder-width units (the caller normalises before
        recording).  Returns 0.0 when there is too little history to judge,
        so the opening seconds of a session cannot produce a fidget verdict.
        """
        hist = list(self._torso_hist)
        if len(hist) < perf_config.FIDGET_MIN_FRAMES:
            return 0.0
        dists = [
            math.hypot(hist[i][0] - hist[i - 1][0], hist[i][1] - hist[i - 1][1])
            for i in range(1, len(hist))
        ]
        return float(sum(dists) / len(dists))

    def note_yaw_averted(self, averted: bool, now: float) -> None:
        """Count entries into the averted-gaze band for P28.

        Only the rising edge counts: holding one turn for ten seconds is a
        single event, while four separate looks across the same window is a
        pattern.  Events older than GLANCE_WINDOW_SEC are pruned on read.
        """
        if averted and not self._yaw_averted_prev:
            self._glance_events.append(now)
        self._yaw_averted_prev = averted

    def glance_event_count(self, now: float | None = None) -> int:
        """Averted-gance events inside the rolling window."""
        if now is None:
            now = time.monotonic()
        cutoff = now - perf_config.GLANCE_WINDOW_SEC
        while self._glance_events and self._glance_events[0] < cutoff:
            self._glance_events.popleft()
        return len(self._glance_events)

    def update_cheek_contact(self, in_contact: bool, now: float) -> float | None:
        """Track how long cheek contact has been held continuously (seconds).

        Returns ``None`` whenever contact breaks, and ``0.0`` on the first
        in-contact frame -- the same contract as ``update_glance``.
        """
        if not in_contact:
            self._cheek_contact_start = None
            return None
        if self._cheek_contact_start is None:
            self._cheek_contact_start = now
            return 0.0
        return max(0.0, now - self._cheek_contact_start)

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
        configure_opencv_threads()
        self._yolo: Any = None
        self._device = resolve_device(perf_config.INFERENCE_DEVICE)
        self._device_arg = yolo_device(self._device)
        self._half = bool(perf_config.INFERENCE_HALF) and self._device.type == "cuda"
        optimize_cuda(self._device)
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
        # Reusable BGR->RGB crop buffers, keyed by crop width.  A fixed-width
        # buffer keeps each row contiguous while allowing the crop height to
        # vary with the candidate box; MediaPipe silently returns no pose for
        # non-contiguous input, so this must never be a strided row view.
        self._mp_rgb_buffers: dict[int, np.ndarray] = {}
        self._head_pose_estimator = HeadPoseEstimator()
        self._temporal_buffer = _TemporalBuffer()
        self._last_features: dict[str, float] = {}
        self._ml_classifier: Any = None
        self._ml_feature_names: list[str] = []
        self._ml_metadata: dict[str, Any] = {}
        self._slow_frame_counter = 0
        # Frames the current raw verdict has held (see _apply_stability_gate),
        # published in temporal_metrics for the HUD's alert-sustain tier.
        self._last_gate_streak: int = 0
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
        # Latest snapshot from the concurrent desk-proximity thread (see
        # set_proximity).  Written only from the inference thread, between
        # frames; never shared with the detection thread directly.
        self._proximity: dict[str, Any] | None = None

        # Classroom continuity: SCUT-HEAD seat anchors (set by the inference
        # worker after initialisation), ByteTrack ids from the pose detector,
        # and the fallback tracker / boundary guard used when those ids are
        # unavailable or a candidate is clipped by the frame edge.
        self._seat_anchors: SeatAnchorRegistry | None = None
        self._last_track_ids: list[int | None] = []
        self._last_seat_ids: list[str | None] = []
        self._last_candidate_count: int = 0
        self._presence_held: bool = False
        # Continuous-SCUT head evidence for the last frame: the raw live
        # boxes, the boundary validation result per surviving candidate, and
        # the primary candidate's solvePnP depth.  Written only on the
        # inference thread.
        self._last_live_head_boxes: list[dict[str, Any]] = []
        self._last_validations: list[dict[str, Any]] = []
        self._last_depth_info: dict[str, Any] = {
            "success": False,
            "depth_m": None,
            "position_m": None,
            "camera_matrix": None,
            "method": "",
            "reprojection_error_px": None,
        }
        self._candidate_tracker = CandidateTracker()
        self._clip_guard = BoundaryClipGuard()
        self._bytetrack_failed = False
        self._object_escalation: dict[str, Any] | None = None
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
            _ = self._yolo(
                np.zeros((640, 640, 3), dtype=np.uint8),
                imgsz=perf_config.POSE_IMGSZ,
                device=self._device_arg,
                quantize=16 if self._half else None,
                verbose=False,
            )
            install_safe_compile(self._yolo, "pose")
            logger.info(
                "YOLO11n-pose loaded from %s (%s%s)",
                model_path,
                self._device.type,
                ", FP16" if self._half else "",
            )
        except Exception as exc:
            logger.error("Failed to load YOLO: %s", exc)
            self._yolo = None

    def release(self) -> None:
        """Drop model references so a mode switch can reclaim VRAM.

        Called by the inference thread on shutdown.  Without it the YOLO and
        MediaPipe objects stay reachable from the stopped thread until the
        next GC pass, and the CUDA caching allocator keeps their blocks
        reserved while the head-occupancy mode loads another model.
        """
        self._yolo = None
        self._obj_model = None
        ctx = getattr(self, "_obj_ctx", None)
        if ctx is not None:
            ctx.model = None
        landmarker = getattr(self, "_pose_landmarker", None)
        if landmarker is not None:
            close = getattr(landmarker, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # pragma: no cover - native teardown
                    logger.debug("MediaPipe landmarker close failed: %s", exc)
        self._pose_landmarker = None
        empty_cache(getattr(self, "_device", None))

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
            logger.info(
                "No ML classifier found at %s - using rules only", model_path
            )
            return
        try:
            import joblib

            artifact = joblib.load(str(model_path))
            if isinstance(artifact, dict):
                model = artifact.get("model")
                names = artifact.get("feature_names")
                metadata = {
                    key: value
                    for key, value in artifact.items()
                    if key not in ("model", "feature_names")
                }
            else:
                model = artifact
                names = None
                metadata = {}

            if model is None:
                logger.error(
                    "ML artifact %s has no 'model' entry - using rules only",
                    model_path,
                )
                return

            if names is None:
                # A bare estimator can only be accepted when its input width
                # matches the canonical vector.  Its column order is assumed,
                # so every model written by train_classifier.py carries names.
                width = getattr(model, "n_features_in_", None)
                if width != len(FEATURE_NAMES):
                    logger.error(
                        "ML classifier expects %s features but the runtime "
                        "contract is %d - refusing to load %s",
                        width,
                        len(FEATURE_NAMES),
                        model_path,
                    )
                    return
                names = list(FEATURE_NAMES)
                logger.warning(
                    "ML artifact has no feature_names; assuming canonical order"
                )

            if not validate_feature_names(names):
                logger.error(
                    "ML classifier feature contract mismatch: artifact has %d "
                    "columns, runtime expects %d (%s) - refusing to load",
                    len(list(names)),
                    len(FEATURE_NAMES),
                    model_path,
                )
                return

            self._ml_classifier = model
            self._ml_feature_names = list(names)
            self._ml_metadata = metadata
            accuracy = metadata.get("test_accuracy")
            if accuracy is not None:
                logger.info(
                    "ML classifier loaded from %s (%d features, test accuracy %.2f%%)",
                    model_path,
                    len(self._ml_feature_names),
                    float(accuracy) * 100.0,
                )
            else:
                logger.info(
                    "ML classifier loaded from %s (%d features)",
                    model_path,
                    len(self._ml_feature_names),
                )
        except ImportError:
            logger.warning(
                "scikit-learn/joblib not installed - ML classifier unavailable"
            )
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

    def set_proximity(self, snapshot: dict[str, Any] | None) -> None:
        """Publish the latest desk-proximity snapshot for the next frame.

        Called by InferenceWorker immediately before ``classify()`` on the
        same thread (the concurrent detection thread writes its own snapshot
        under a lock; the worker copies it out and hands it over here), so
        the classifier itself is never touched from a second thread.  The
        snapshot shape is ``{"objects": [...], "phone": {...} | None,
        "ts": float}`` -- see DeskProximityThread.snapshot().
        """
        self._proximity = snapshot if isinstance(snapshot, dict) else None

    def set_seat_anchors(
        self, anchors: SeatAnchorRegistry | list[dict[str, Any]] | None
    ) -> int:
        """Install the classroom seat anchors established during startup.

        Accepts the registry built by ``HeadAnchorInitializer``, a plain list
        of anchor dicts (for tests / external callers), or None.  Returns the
        number of locked seats.
        """
        if anchors is None:
            self._seat_anchors = None
            return 0
        if isinstance(anchors, SeatAnchorRegistry):
            self._seat_anchors = anchors
        else:
            from desktop_app.core.seat_anchor import SeatAnchor

            built: list[SeatAnchor] = []
            for i, item in enumerate(anchors):
                if not isinstance(item, dict):
                    continue
                box = item.get("box") or []
                if len(box) != 4:
                    continue
                built.append(SeatAnchor(
                    seat_id=str(item.get("seat_id") or f"Seat_{i + 1:02d}"),
                    box=[int(v) for v in box],
                    confidence=float(item.get("confidence", 0.0)),
                    hits=int(item.get("hits", 0)),
                    track_id=item.get("track_id"),
                ))
            self._seat_anchors = SeatAnchorRegistry(
                anchors=built, locked=bool(built)
            )
        return self._seat_anchors.seat_count

    def seat_anchor_info(self) -> dict[str, Any]:
        registry = self._seat_anchors
        return {
            "seated_count": registry.seat_count if registry else 0,
            "seats_locked": bool(registry and registry.locked),
            "seat_anchors": registry.as_list() if registry else [],
        }

    def warm_object_detector(
        self,
        frame: np.ndarray,
        boxes: list[list[int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one forced desk-object pass to pay first-use model warmup.

        The anomaly-triggered desk pass and the concurrent proximity pass use
        different YOLO instances.  Their first simultaneous inference can
        spend seconds in CUDA/driver warmup, which would otherwise appear as a
        frozen HUD during the first live frames.  ``force=True`` does not
        advance ``ObjectContext.detect_counter``, so this startup-only pass
        cannot disturb the live gate accounting.
        """
        self._last_frame = frame
        self._last_frame_shape = frame.shape
        self._last_frame_width = frame.shape[1]
        return self._detect_objects(frame, boxes or [], force=True)

    def _phone_proximity_hit(
        self, objects: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """The strongest wrist-at-phone evidence in *objects*, if any.

        The live decision is metric: the wrist and the detection's bbox
        centre are back-projected onto the candidate's solvePnP depth plane,
        and the verdict requires a 3D Euclidean distance under
        ``PHONE_WRIST_DIST_M`` (15 cm).  A pixel fallback is used only when
        the depth solve failed entirely; when depth is available but an
        object could not be measured, the object is not evidence (fail
        closed).  The confidence floor and temporal persistence gate apply
        to both paths.
        """
        if not objects:
            return None
        wrists = []
        for idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
            pt = _visible_xy(self._last_landmarks, idx)
            if pt is not None:
                wrists.append(pt)
        if not wrists:
            return None

        depth_ok = bool(
            self._last_depth_info
            and self._last_depth_info.get("depth_m") is not None
        )
        best: dict[str, Any] | None = None
        best_key: tuple[int, float] | None = None
        for obj in objects:
            if obj.get("label") not in _NEUTRAL_DEVICE_LABELS:
                continue
            if float(obj.get("confidence", 0.0)) <= perf_config.PHONE_MIN_CONF:
                continue
            # Temporal persistence: the concurrent pass throttles detection,
            # so a phone must be re-sighted on PROX_MIN_HITS passes.  An
            # anomaly-path object instead carries ``persistent`` from the
            # shared track merger.
            if "hits" in obj:
                if int(obj.get("hits", 0) or 0) < perf_config.PROX_MIN_HITS:
                    continue
            elif not obj.get("persistent"):
                continue
            box = obj.get("box")
            if not box or len(box) != 4:
                continue

            metric = obj.get("wrist_dist_m")
            if metric is not None:
                try:
                    dist_m = float(metric)
                except (TypeError, ValueError):
                    continue
                if dist_m >= perf_config.PHONE_WRIST_DIST_M:
                    continue
                key = (0, dist_m)
            elif not depth_ok:
                # Legacy path only: no depth solve for this frame at all.
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                dist_px = min(_dist_2d(w, (cx, cy)) for w in wrists)
                if dist_px >= perf_config.PHONE_WRIST_DIST_PX:
                    continue
                key = (1, dist_px)
            else:
                # Depth exists: an unmeasured object is not usable evidence.
                continue

            if best_key is None or key < best_key:
                best_key = key
                best = dict(obj)
                if key[0] == 0:
                    best["wrist_dist_m"] = round(key[1], 4)
                else:
                    best["wrist_dist"] = round(key[1], 1)
        return best

    @staticmethod
    def _proximity_snapshot_phone_usable(
        phone: dict[str, Any] | None,
    ) -> bool:
        """True when a snapshot phone has a measured metric wrist distance.

        The proximity thread detects the object; the depth solve happens on
        the inference thread when the lists are merged.  A snapshot entry
        that never received that metric annotation cannot establish 15 cm
        proximity, so it is not usable evidence.
        """
        if not isinstance(phone, dict):
            return False
        if float(phone.get("confidence", 0.0)) <= perf_config.PHONE_MIN_CONF:
            return False
        hits = int(phone.get("hits", 0) or 0)
        if "hits" in phone and hits < perf_config.PROX_MIN_HITS:
            return False
        dist = phone.get("wrist_dist_m")
        if dist is None:
            return False
        try:
            return float(dist) < perf_config.PHONE_WRIST_DIST_M
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _device_evidence_ok(
        obj: dict[str, Any],
        require_wrist: bool = False,
    ) -> bool:
        """Live-path metric gate for device P38 escalations.

        The shared spatial relation answers "can the student reach it"; the
        phone/P38 rule additionally requires the hand to be on the device.
        ``require_wrist=True`` is the live classify() contract: a device with
        no measured metric distance is not evidence.  The default stays
        lenient for the backend and unit-level escalation tests, which
        construct objects without pose landmarks and historically relied on
        the pixel fallback.
        """
        label = str(obj.get("label", "")).lower()
        if label not in _DEVICE_WRIST_GATED:
            return True
        if float(obj.get("confidence", 0.0)) <= perf_config.PHONE_MIN_CONF:
            return False
        metric = obj.get("wrist_dist_m")
        if metric is not None:
            try:
                return float(metric) < perf_config.PHONE_WRIST_DIST_M
            except (TypeError, ValueError):
                return False
        if require_wrist:
            return False
        dist = obj.get("wrist_dist_px")
        if dist is None:
            dist = obj.get("wrist_dist")
        if dist is None:
            return True
        if float(dist) >= perf_config.PHONE_WRIST_DIST_PX:
            return False
        return True

    def _earbuds_worn_hit(
        self, objects: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """The strongest ear-level earbud evidence in *objects*, if any.

        Mirrors the phone-usability design: the concurrent proximity pass
        runs on every frame, so this works even when the raw verdict is P01
        and the anomaly-triggered object pass is correctly skipped.  An
        earbud bbox centred within EARBUDS_EAR_DIST_PX of a visible ear
        landmark is "worn"; the same object lying on the desk is not.
        """
        if not objects:
            return None
        ears = []
        for idx in (MP_LEFT_EAR, MP_RIGHT_EAR):
            pt = _visible_xy(self._last_landmarks, idx)
            if pt is not None:
                ears.append(pt)
        if not ears:
            return None
        best: dict[str, Any] | None = None
        best_dist = math.inf
        for obj in objects:
            label = str(obj.get("label", "")).lower()
            if label != "earbuds":
                continue
            if float(obj.get("confidence", 0.0)) < perf_config.PHONE_MIN_CONF:
                continue
            box = obj.get("box")
            if not box or len(box) != 4:
                continue
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            dist = min(_dist_2d(e, (cx, cy)) for e in ears)
            if dist < best_dist:
                best_dist = dist
                best = dict(obj)
                best["ear_dist"] = round(dist, 1)
        if best is None or best_dist >= perf_config.EARBUDS_EAR_DIST_PX:
            return None
        return best

    def classify(
        self,
        frame: np.ndarray,
        head_boxes: list[dict[str, Any]] | list[list[int]] | None = None,
    ) -> dict[str, Any]:
        """Classify the frame and return a standardised result dict.

        ``head_boxes`` is the current SCUT-HEAD output from the unified
        worker cycle.  It is used *before* MediaPipe: body hypotheses with no
        head anchor in their upper 30% are rejected, so a clipped torso or a
        duplicate fragment never becomes a candidate or a P35 finding.
        Surviving candidates have their face landmarks restricted to the
        matched head box.

        Returns
        -------
        dict with keys:
            pose_id, pose_name, confidence, bounding_boxes,
            keypoints, landmarks, head_pose, gaze_vector,
            geometric_features, temporal_metrics, spatial_features,
            anomalies, anomaly_flags, candidate_depth_m, live_head_boxes
        """
        t0 = time.perf_counter()
        self._last_frame = frame
        h, w = frame.shape[:2]
        self._last_frame_shape = frame.shape
        self._last_frame_width = w
        self._object_escalation = None

        # ── Continuous SCUT-HEAD anchors ─────────────────────────────────
        # Live heads move the locked anchors but never create, remove or
        # renumber a seat.  The validation source merges the live boxes with
        # any locked seat the current head pass missed, so a one-frame head
        # miss cannot reject a genuine student.
        live_heads = list(head_boxes or [])
        self._last_live_head_boxes = [
            dict(head) if isinstance(head, dict) else {"box": list(head)}
            for head in live_heads
        ]
        live_seat_ids: list[str | None] = []
        validation_heads: Any = self._seat_anchors
        if live_heads and perf_config.ENABLE_LIVE_HEAD_ANCHORS:
            if self._seat_anchors is not None:
                live_seat_ids = self._seat_anchors.observe_live_heads(
                    live_heads, self._frames_seen
                )
            validation_heads = []
            matched_seats: set[str] = set()
            for i, head in enumerate(live_heads):
                box = head.get("box") if isinstance(head, dict) else head
                if not box or len(box) != 4:
                    continue
                seat_id = live_seat_ids[i] if i < len(live_seat_ids) else None
                if seat_id:
                    matched_seats.add(seat_id)
                validation_heads.append({
                    "box": [int(v) for v in box],
                    "seat_id": seat_id,
                })
            if self._seat_anchors is not None:
                for anchor in self._seat_anchors.anchors:
                    if anchor.seat_id not in matched_seats:
                        validation_heads.append({
                            "box": list(anchor.box),
                            "seat_id": anchor.seat_id,
                        })

        # ByteTrack ids are a side effect of _yolo_detect: the method keeps
        # its (boxes, keypoints) contract so existing stubs/tests stay valid.
        self._last_track_ids = []
        raw_boxes, raw_kpts = self._yolo_detect(frame)
        track_ids = list(self._last_track_ids)
        if len(track_ids) != len(raw_boxes):
            # ByteTrack unavailable (or the detector was stubbed): the small
            # IoU tracker still gives every candidate a stable id.
            track_ids = self._candidate_tracker.update(raw_boxes, frame.shape)
            self._last_track_ids = list(track_ids)

        # Record presence BEFORE boundary validation.  A clipped candidate
        # whose box is later rejected must still count as recently seen, or
        # the next frame would falsely report P16 Absent Seat.
        clip_info = self._clip_guard.update(
            frame.shape, raw_boxes, track_ids, self._frames_seen
        )

        # Validate body hypotheses against the head anchors BEFORE spending
        # MediaPipe inference on them.  This is the unified-cycle contract:
        # SCUT-HEAD boxes gate the skeleton pass, not the other way round.
        valid_indices: list[int] = []
        validations: list[dict[str, Any]] = []
        for i, box in enumerate(raw_boxes):
            kp = raw_kpts[i] if i < len(raw_kpts) else None
            validation = validate_candidate_boundaries(
                box, kp, validation_heads
            )
            if validation["valid"]:
                valid_indices.append(i)
                validations.append(validation)

        if len(valid_indices) != len(raw_boxes):
            yolo_boxes = [raw_boxes[i] for i in valid_indices]
            yolo_kpts = [raw_kpts[i] for i in valid_indices if i < len(raw_kpts)]
            track_ids = [track_ids[i] for i in valid_indices]
        else:
            yolo_boxes, yolo_kpts = raw_boxes, raw_kpts
        self._last_validations = validations

        # MediaPipe runs only on the head-validated boxes.  When it returns
        # one landmark set per box, mask face points that fall outside the
        # matched SCUT head -- those belong to a neighbouring student.
        all_landmarks = self._mediapipe_landmarks(frame, yolo_boxes)
        if len(all_landmarks) == len(yolo_boxes):
            for landmarks, validation in zip(all_landmarks, validations):
                head_box = validation.get("head_box")
                if head_box:
                    mask_face_outside_head(landmarks, head_box)

        # Bind the surviving boxes to the classroom seats.  This is what
        # keeps a partly clipped student attached to Seat_03 instead of being
        # re-counted as a new candidate.
        if self._seat_anchors is not None:
            self._last_seat_ids = self._seat_anchors.bind(
                yolo_boxes, track_ids, self._frames_seen
            )
        else:
            self._last_seat_ids = [None] * len(yolo_boxes)
        self._last_track_ids = list(track_ids)

        # A genuine second student is still P35.  The same ByteTrack id, the
        # same seat, or a near-identical duplicate box is one person.
        person_count = self._clip_guard.distinct_person_count(
            yolo_boxes, track_ids, self._last_seat_ids
        )
        presence_held = False
        if (
            person_count == 0
            and self._clip_guard.presence_held(self._frames_seen)
            and frame_has_usable_image(frame)
        ):
            # The candidate was here moments ago and is likely clipped out
            # of the frame.  Hold the seat: P16 is an accusation, and it must
            # not be manufactured by a boundary trim.
            presence_held = True
            person_count = 1
        self._presence_held = presence_held
        self._last_candidate_count = person_count

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

        # Metric depth for the primary candidate.  A failed solve is recorded
        # explicitly rather than silently falling back to pixels: the live
        # phone/reach verdicts below fail closed when ``depth_m`` is None.
        self._last_depth_info = metric_depth.estimate_candidate_depth(
            self._last_landmarks, frame.shape
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

        # ── P23 / P28 temporal inputs ────────────────────────────────────
        # Torso centre normalised by shoulder width (P23 fidget), and
        # rising-edge entries into the averted-gaze band (P28 glancing).
        # Fed to the buffer before the rule engine reads them, matching the
        # body-turn ratio contract above so the two can never disagree.
        lms0 = all_landmarks[0] if all_landmarks else []
        if (
            self._last_shoulder_width >= perf_config.MIN_SCALE_PX
            and len(lms0) >= 33
        ):
            ls = _visible_xy(lms0, MP_LEFT_SHOULDER)
            rs = _visible_xy(lms0, MP_RIGHT_SHOULDER)
            lh = _visible_xy(lms0, MP_LEFT_HIP)
            rh = _visible_xy(lms0, MP_RIGHT_HIP)
            if ls and rs and lh and rh:
                sw = max(self._last_shoulder_width, 1e-6)
                cx = (ls[0] + rs[0] + lh[0] + rh[0]) / (4.0 * sw)
                cy = (ls[1] + rs[1] + lh[1] + rh[1]) / (4.0 * sw)
                self._temporal_buffer.note_torso(cx, cy)
        self._temporal_buffer.note_yaw_averted(
            abs(head_pose["yaw"] - yaw_baseline) > perf_config.GLANCE_YAW_DEG,
            time.monotonic(),
        )
        temporal_metrics["fidget_disp"] = (
            self._temporal_buffer.fidget_displacement()
        )
        temporal_metrics["glance_events"] = float(
            self._temporal_buffer.glance_event_count()
        )

        # ── Spatial features ─────────────────────────────────────────────
        spatial = self._compute_spatial_features(yolo_boxes, frame.shape)

        # ── Multi-candidate proximity vectoring ──────────────────────────
        # Distinct candidates after ByteTrack/seat/duplicate collapse, not
        # raw boxes: a clipped duplicate must not become P35 Multiple Persons.
        person_count = self._last_candidate_count
        temporal_metrics["candidate_count"] = float(person_count)
        temporal_metrics["presence_held"] = 1.0 if self._presence_held else 0.0
        temporal_metrics["clip_guard"] = 1.0 if clip_info["any_clipped"] else 0.0
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

        # ── Concurrent desk-object proximity fusion ─────────────────────
        # The DeskProximityThread runs its own desk YOLO pass on a throttled
        # 416 px desk ROI in parallel with this loop.  Its detections are
        # merged with any the anomaly-triggered pass above produced, so the
        # phone/object rules see one list regardless of which pass saw the
        # object first.  A proximity-only detection can escalate on a P01
        # frame -- that is the whole point of running it concurrently.
        proximity_snapshot = self._proximity
        phone_hit: dict[str, Any] | None = None
        if proximity_snapshot:
            merged_objects = _merge_detections(
                objects_detected, proximity_snapshot.get("objects") or []
            )
            if merged_objects is not objects_detected:
                objects_detected = merged_objects

        # Convert every device detection into metric camera coordinates before
        # any rule reads it.  The annotation rewrites ``relation``/``in_hand``
        # from pixels, so escalation and the neutral "device at jaw" test all
        # agree on the 15 cm physical radius.
        if objects_detected:
            metric_depth.annotate_metric_distances(
                objects_detected,
                self._last_landmarks,
                self._last_depth_info,
                frame.shape,
            )

        if objects_detected and not warming_up:
            # Devices must clear the metric 0.15 m wrist gate (confidence and
            # temporal persistence included) before they can escalate.  The
            # shared ObjectContext still decides reachability and the
            # authorized/unauthorized document zones.
            escalatable = [
                obj for obj in objects_detected
                if self._device_evidence_ok(obj, require_wrist=True)
            ]
            pose_id, confidence = self._escalate_from_objects(
                pose_id, confidence, escalatable,
                person_bbox=primary_bbox,
                frame_shape=frame.shape,
            )

        if proximity_snapshot:
            phone_hit = self._phone_proximity_hit(objects_detected)
            snapshot_phone = proximity_snapshot.get("phone")
            if phone_hit is None and self._proximity_snapshot_phone_usable(
                snapshot_phone
            ):
                phone_hit = dict(snapshot_phone)  # type: ignore[arg-type]
        if phone_hit is not None:
            # Usability, not mere presence: the wrist must be AT the phone
            # and the detection must be confident.  Held to the stability
            # gate like every other anomaly below.
            pose_id = "P31_PHONE_USABILITY"
            confidence = max(confidence, 0.85)
            self._object_escalation = {
                "pose_id": "P31_PHONE_USABILITY",
                "confidence": confidence,
                "label": phone_hit.get("label"),
                "reason": "wrist_proximity",
            }
        elif self._earbuds_worn_hit(objects_detected):
            # Worn, not merely present: the earbud bbox must sit at an ear
            # landmark.  Same design as the phone-usability rule -- the
            # concurrent proximity pass runs on every frame, so a student
            # sitting normally (P01) while wearing earbuds is still caught,
            # which the anomaly-triggered escalation path alone would miss.
            pose_id = "P26"
            confidence = max(confidence, 0.78)

        # ── P29 chin-rest clock ──────────────────────────────────────────
        # Ticks every frame on the same cheek-contact geometry the neutral
        # filter uses, so the two can never disagree.  Contact held past
        # CHIN_REST_SUSTAINED_SEC stops being "thinking" and becomes P29;
        # the neutral filter does not suppress it (it is not in
        # _NEUTRAL_SUPPRESSES) -- the first minute is benign, the next hour
        # is disengagement, and both are real findings.
        cheek_elapsed = self._temporal_buffer.update_cheek_contact(
            _cheek_contact_active(
                self._last_landmarks,
                objects_detected,
                self._last_depth_info,
                frame.shape,
            ),
            time.monotonic(),
        )

        # ── Positive neutral posture filter ─────────────────────────────
        # Evaluated before the gate so a suppressed threat never accumulates
        # a streak: while the student is demonstrably writing / thinking /
        # glancing, the gate sees P01 and the streak resets.  When the
        # neutral state ends, a real threat must re-earn its streak from
        # zero rather than resuming an old one.
        neutral_state = None
        if not warming_up:
            neutral_state = check_neutral_alignment(
                self._last_landmarks,
                head_pose["yaw"], head_pose["pitch"], head_pose["roll"],
                objects_detected,
                bbox=list(primary_bbox) if primary_bbox else None,
                temporal=self._temporal_buffer,
                depth_info=self._last_depth_info,
                frame_shape=frame.shape,
            )
        if neutral_state is not None and pose_id in _NEUTRAL_SUPPRESSES:
            temporal_metrics["neutral_suppressed"] = 1.0
            logger.debug(
                "neutral %s suppresses %s", neutral_state, pose_id
            )
            pose_id, confidence = "P01", 0.95

        # ── P29 chin rest (sustained cheek contact) ──────────────────────
        # Applied after the neutral filter so a suppressed P14/P31 cannot
        # mask it, and only over benign verdicts -- a real anomaly already
        # standing (P16, P35, P15 ...) outranks a posture observation.
        if (
            cheek_elapsed is not None
            and cheek_elapsed >= perf_config.CHIN_REST_SUSTAINED_SEC
            and pose_id in ("P01",)
        ):
            temporal_metrics["chin_rest_sec"] = cheek_elapsed
            pose_id, confidence = "P29", 0.70

        # ── Temporal / confidence gate ───────────────────────────────────
        # Applied to the FINAL verdict, so an object-driven escalation is held
        # to the same persistence requirement as a pose-driven one.
        pose_id, confidence, suppressed = self._apply_stability_gate(
            pose_id, confidence, warming_up=warming_up
        )

        # The raw streak counts frames the rule has been firing (including
        # ones the gate still withholds); the sustained one counts published
        # anomaly frames.  The HUD's alert tiers key off the larger of the
        # two so a threat reaches AMBER at frame 30 of continuous behaviour,
        # not frame 42.  Both are published once per frame, below.
        temporal_metrics["anomaly_streak"] = float(
            0 if warming_up else self._last_gate_streak
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

        # Published after record_anomaly so this frame's verdict is counted.
        temporal_metrics["sustained_anomaly_frames"] = float(
            self._temporal_buffer._anomaly_streak
        )

        if self._temporal_buffer.is_sustained_anomaly:
            # Escalate confidence for sustained anomalies
            confidence = min(confidence + 0.10, 0.99)

        # ── Optional ML classifier override ──────────────────────────────
        # ---- Optional ML classifier override ----
        # The model consumes the same 54-D vector that DatasetLogger writes.
        # A missing keypoint stays NaN so the saved imputer sees the same
        # distribution it saw during training.
        if self._ml_classifier is not None and features and not warming_up:
            # The rule engine treats a failed PnP solve as zero-angle, but the
            # training CSV stores those angles as NaN.  Feed the model the
            # training distribution; leave ``head_pose`` itself untouched so
            # rule/temporal behaviour does not change.
            ml_head_pose = head_pose
            if head_pose_result.get("rvec") is None:
                ml_head_pose = {
                    "yaw": float("nan"),
                    "pitch": float("nan"),
                    "roll": float("nan"),
                }
            ml_features = feature_dict_from_components(
                keypoints=yolo_kpts[0] if yolo_kpts else None,
                bbox=yolo_boxes[0] if yolo_boxes else None,
                head_pose=ml_head_pose,
                geometric=features,
                spatial=spatial,
                motion_delta=temporal_metrics.get("bbox_scale_delta", 0.0),
            )
            ml_pose_id, ml_conf = self._ml_predict(ml_features)
            if ml_conf > 0.80:
                # A pose-only classifier must not overturn room-level or
                # object-level facts (absence, occupancy, phone, documents).
                if pose_id in _PRESENCE_POSES or ml_pose_id in _ML_OBJECT_VERDICTS:
                    temporal_metrics["ml_blocked"] = 1.0
                # The neutral filter is a promise that these specific alerts
                # stay down while benign behaviour is being demonstrated; an
                # ML override must not be able to sneak them back in.
                elif neutral_state is not None and ml_pose_id in _NEUTRAL_SUPPRESSES:
                    temporal_metrics["neutral_suppressed"] = 1.0
                else:
                    pose_id, confidence = ml_pose_id, ml_conf

        seat_info = self.seat_anchor_info()
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
            # solvePnP camera-frame estimate for the primary candidate.  A
            # None depth means no metric verdict this frame (never pixels).
            "candidate_depth_m": self._last_depth_info.get("depth_m"),
            "candidate_position_m": self._last_depth_info.get("position_m"),
            "depth_method": self._last_depth_info.get("method", ""),
            "depth_reprojection_px": self._last_depth_info.get(
                "reprojection_error_px"
            ),
            # Live SCUT-HEAD boxes used to gate this frame's skeletons, plus
            # the per-candidate boundary decision (head anchor, seat, limb
            # attribution) for the sidebar/debug overlay.
            "live_head_boxes": list(self._last_live_head_boxes),
            "head_anchor_validations": [dict(v) for v in validations],
            # Classroom continuity: the ByteTrack ids, the seat each candidate
            # resolved to, and the locked seat inventory established during
            # startup.  The HUD draws one label per box from these.
            "track_ids": list(self._last_track_ids),
            "seat_ids": list(self._last_seat_ids),
            "seat_anchors": seat_info["seat_anchors"],
            "seated_count": seat_info["seated_count"],
            "seats_locked": seat_info["seats_locked"],
            "candidate_count": int(self._last_candidate_count),
            "presence_held": bool(self._presence_held),
            # Which object (if any) escalated the verdict, for the sidebar and
            # dataset logs.  None on frames with no object evidence.
            "object_escalation": (
                dict(self._object_escalation)
                if self._object_escalation is not None
                else None
            ),
            # Positive neutral posture filter output.  The label is resolved
            # here so the HUD and sidebar never have to import NEUTRAL_LABELS
            # to render a state.
            "neutral_state": neutral_state,
            "neutral_label": NEUTRAL_LABELS.get(neutral_state or "", ""),
            "neutral_suppressed": bool(
                temporal_metrics.get("neutral_suppressed")
            ),
            # Wrist-to-phone evidence for telemetry/debug overlays.
            "phone_proximity": (
                {
                    "label": phone_hit.get("label"),
                    "confidence": phone_hit.get("confidence"),
                    "wrist_dist_m": phone_hit.get("wrist_dist_m"),
                    # Retained for the HUD's legacy debug read; metric code
                    # must use wrist_dist_m.
                    "wrist_dist": phone_hit.get("wrist_dist"),
                }
                if phone_hit is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # YOLO detection
    # ------------------------------------------------------------------

    def _yolo_detect(
        self, frame: np.ndarray
    ) -> tuple[list[list[int]], list[np.ndarray]]:
        """Run pose detection and, when available, ByteTrack id assignment.

        The return contract stays ``(boxes, keypoints)`` because callers and
        tests stub this method with that shape.  Track ids are published as a
        side effect in ``self._last_track_ids``; classify() falls back to its
        own IoU tracker when they are missing or misaligned.
        """
        if self._yolo is None:
            self._last_track_ids = []
            return ([], [])

        use_track = (
            bool(perf_config.ENABLE_BYTETRACK) and not self._bytetrack_failed
        )
        kwargs = dict(
            imgsz=perf_config.POSE_IMGSZ,
            max_det=perf_config.POSE_MAX_DET,
            device=self._device_arg,
            quantize=16 if self._half else None,
            verbose=False,
        )
        try:
            if use_track:
                try:
                    results = self._yolo.track(
                        frame,
                        persist=True,
                        tracker=perf_config.BYTETRACK_CONFIG,
                        **kwargs,
                    )
                except Exception as exc:
                    # A missing/invalid tracker config must not disable pose
                    # detection: remember the failure and use the plain
                    # detector plus the IoU fallback tracker from now on.
                    self._bytetrack_failed = True
                    logger.warning(
                        "ByteTrack unavailable (%s); using IoU tracker", exc
                    )
                    results = self._yolo(frame, **kwargs)
            else:
                results = self._yolo(frame, **kwargs)
            r = results[0]
            boxes = (
                r.boxes.xyxy.cpu().numpy().tolist()
                if r.boxes is not None
                else []
            )
            ids: list[int] = []
            if use_track and r.boxes is not None:
                raw_ids = getattr(r.boxes, "id", None)
                if raw_ids is not None:
                    try:
                        ids = [
                            int(v) for v in raw_ids.cpu().numpy().tolist()
                        ]
                    except (AttributeError, TypeError, ValueError):
                        ids = []
            self._last_track_ids = (
                ids if ids and len(ids) == len(boxes) else []
            )
            kpts = (
                r.keypoints.data.cpu().numpy()
                if r.keypoints is not None
                else np.empty((0, 17, 3))
            )
            return boxes, list(kpts) if kpts.ndim == 3 else []
        except Exception as exc:
            logger.error("YOLO detection failed: %s", exc)
            self._last_track_ids = []
            return ([], [])

    # ------------------------------------------------------------------
    # MediaPipe PoseLandmarker (Task API)
    # ------------------------------------------------------------------

    def _mp_rgb_crop(self, crop: np.ndarray) -> np.ndarray | None:
        """BGR crop -> reusable contiguous RGB buffer (or None if empty)."""
        if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
            return None
        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return None
        buffer = self._mp_rgb_buffers.get(width)
        if buffer is None or buffer.shape[0] < height:
            # Keep the cache bounded.  Widths come from candidate boxes; a
            # handful of stable widths covers a classroom, and evicting the
            # oldest entry prevents a moving camera from growing it forever.
            if len(self._mp_rgb_buffers) >= 8:
                self._mp_rgb_buffers.pop(next(iter(self._mp_rgb_buffers)))
            buffer = np.empty(
                (max(height, 64), width, 3), dtype=np.uint8
            )
            self._mp_rgb_buffers[width] = buffer
        # Width matches the buffer exactly, so this row-prefix view is
        # C-contiguous (height may be smaller than the allocation).
        view = buffer[:height]
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB, dst=view)
        return view

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
            #
            # The destination is a reusable, exactly-width buffer.  MediaPipe
            # requires C-contiguous input; a strided view makes it silently
            # return zero poses (no error, no warning), which quietly disables
            # the entire P01-P40 rule engine downstream.
            crop = self._mp_rgb_crop(frame[y1:y2, x1:x2])
            if crop is None or crop.size == 0:
                continue

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
            self._last_gate_streak = 0
            return pose_id, confidence, False

        if pose_id in perf_config.IMMEDIATE_POSES:
            # Immediate poses never accumulate a streak (by design), so the
            # published one must not carry over from the pose before it.
            self._last_gate_streak = 0
            return pose_id, confidence, False

        if warming_up:
            # Record nothing.  Returning the raw verdict is safe because
            # classify() replaces it with INITIALIZING, but clearing here too
            # keeps the contract local: during warmup the gate has no opinion.
            self._last_gate_streak = 0
            return pose_id, confidence, False

        # P01 is recorded so that a return to normal behaviour CLEARS the
        # streak.  Returning early here instead would let an old streak
        # survive any number of normal frames, so the next single noisy frame
        # would publish an anomaly instantly.
        streak = self._temporal_buffer.note_pose(pose_id)
        # Published for the HUD's alert-sustain tier: how long the raw rule
        # has been firing at this id, including frames still below the gate.
        self._last_gate_streak = streak

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
        features: dict[str, float] = {
            name: float("nan") for name in GEOMETRIC_FEATURE_NAMES
        }

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

    def _ml_predict(
        self, features: dict[str, float] | np.ndarray
    ) -> tuple[str, float]:
        """Predict a verdict from the canonical 54-D feature vector.

        Missing values stay NaN: the saved pipeline owns a median imputer, and
        imputing here with 0.0 would silently disagree with training.
        """
        try:
            if isinstance(features, dict):
                vector = vector_from_dict(features).reshape(1, -1)
            else:
                vector = np.asarray(features, dtype=np.float64).reshape(1, -1)

            expected = len(self._ml_feature_names or FEATURE_NAMES)
            if vector.shape[1] != expected:
                raise ValueError(
                    f"expected {expected} features, got {vector.shape[1]}"
                )

            # One traversal, not two.  Calling predict() and then
            # predict_proba() walked all 200 trees twice; at 10 FPS that was
            # ~90 ms of avoidable per-frame latency.  Derive the label from
            # the probability vector when the estimator supports it.
            if hasattr(self._ml_classifier, "predict_proba"):
                proba = np.asarray(
                    self._ml_classifier.predict_proba(vector)
                ).reshape(-1)
                if not proba.size:
                    raise ValueError("empty predict_proba output")
                classes = getattr(self._ml_classifier, "classes_", None)
                if classes is None and hasattr(
                    self._ml_classifier, "named_steps"
                ):
                    final = list(self._ml_classifier.named_steps.values())[-1]
                    classes = getattr(final, "classes_", None)
                best = int(np.argmax(proba))
                if classes is not None and best < len(classes):
                    pose_id = str(np.asarray(classes)[best])
                else:
                    pose_id = str(self._ml_classifier.predict(vector)[0])
                confidence = float(proba[best])
            else:
                pose_id = str(self._ml_classifier.predict(vector)[0])
                confidence = 0.85

            if pose_id not in POSE_MAP and pose_id != "P00":
                logger.debug("ML predicted unknown verdict %r", pose_id)
                return ("P01", 0.0)
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
        person_bbox: list[int] | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> tuple[str, float]:
        """Escalate a verdict using detected objects, with spatial agreement.

        Delegates to ObjectContext.escalation_detail: which labels escalate,
        where the spatial gate sits, and which documents are authorized are
        configuration, shared with the backend.  ``person_bbox``/``frame_shape``
        are optional so the legacy three-argument calls (and their tests)
        keep working; the live path supplies them so unauthorized documents
        can be tested against the candidate's designated desk zone.
        """
        detail = ObjectContext.escalation_detail(
            objects, person_bbox, frame_shape
        )
        if detail is None:
            return pose_id, confidence
        self._object_escalation = dict(detail)
        return detail["pose_id"], max(confidence, float(detail["confidence"]))

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

        # ── P25 Hidden Face ──────────────────────────────────────────────
        # Face landmarks (nose / eyes / ears) invisible while both shoulders
        # stay clearly visible: the student is in frame but their face is
        # occluded -- hand, hair, mask, coat collar, head buried.  Sits
        # BEFORE every head-pose and yaw/pitch rule because solvePnP on an
        # occluded face produces confident garbage, so the rules below must
        # not be trusted once the face is gone.
        face_idx = (
            MP_NOSE, MP_LEFT_EYE_INNER, MP_LEFT_EYE_OUTER,
            MP_RIGHT_EYE_INNER, MP_RIGHT_EYE_OUTER, MP_LEFT_EAR, MP_RIGHT_EAR,
        )
        face_vis_vals = [
            lms[i][2] for i in face_idx
            if len(lms) > i and lms[i][2] is not None
        ]
        if face_vis_vals and _visible(MP_LEFT_SHOULDER,
                                      perf_config.HIDDEN_FACE_MIN_SHOULDER_VIS) \
                and _visible(MP_RIGHT_SHOULDER,
                             perf_config.HIDDEN_FACE_MIN_SHOULDER_VIS):
            face_vis = sum(face_vis_vals) / len(face_vis_vals)
            if face_vis < perf_config.HIDDEN_FACE_MAX_VIS:
                return ("P25", 0.75)

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

        # ── P28 Frequent Glancing ────────────────────────────────────────
        # Rising-edge entries into the averted-gaze band (|yaw delta| >
        # GLANCE_YAW_DEG) inside the rolling window.  A single look is P20
        # below; four separate looks in ten seconds is a pattern.  Counts
        # events, not frames, so one held turn cannot inflate the total.
        if temporal.get("glance_events", 0) >= float(
            perf_config.GLANCE_MIN_EVENTS
        ):
            return ("P28", 0.72)

        # ── P23 Excessive Fidgeting ──────────────────────────────────────
        # Mean per-frame torso-centre displacement in shoulder-width units,
        # over the rolling window.  A seated student writing or typing moves
        # well under FIDGET_MIN_DISP per frame; shifting, rocking and
        # re-settling clear it easily, and the stability gate requires the
        # motion to persist before it is published.
        if temporal.get("fidget_disp", 0.0) > perf_config.FIDGET_MIN_DISP:
            return ("P23", 0.72)

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
            "body_turn": {"P10", "P11"},
            "excessive_lean": {"P08", "P09", "P31"},
            "multi_person": {"P35"},
            "phone_detected": {"P12", "P31_PHONE_USABILITY"},
            "note_peeking": {"P13"},
            # "talking" deliberately maps to nothing: no rule in this engine
            # produces a talking verdict (P14 is Looking Down), and mapping
            # it there made every looking-down alert also claim whispering.
            "talking": set(),
            "standing": {"P15"},
            "absent": {"P16"},
            "hand_to_mouth": {"P19"},
            "hand_under_table": {"P21"},
            "raised_hand": {"P22"},
            "crossed_arms": {"P30"},
            "head_oscillation": {"P36"},
            "looking_away": {"P12"},
            "looking_down": {"P14"},
            "hidden_face": {"P25"},
            "fidgeting": {"P23"},
            "frequent_glancing": {"P28"},
            "chin_rest": {"P29"},
            "earbuds": {"P26"},
        }
        return {flag: (pose_id in poses) for flag, poses in flag_map.items()}
