"""Single source of truth for the proctoring feature vector.

The live classifier, the CSV logger and ``train_classifier.py`` must agree on
the exact column order.  When they do not, scikit-learn either raises a shape
error (which the runtime catches and silently turns into "rules only") or,
worse, accepts a same-width vector whose columns mean something else.

The vector is 54-dimensional::

    34  COCO-17 keypoints, x then y, normalised to the candidate bbox
     3  head pose (yaw, pitch, roll), degrees
    12  scale-normalised geometric features
     4  spatial features (frame position and bbox shape)
     1  temporal motion delta

Keypoints are deliberately bbox-normalised rather than raw pixels.  A model
trained on 1280x720 classroom frames must not learn that a nose is at x=640;
normalising by the candidate box removes resolution and camera-distance
dependence.  Spatial features retain the absolute frame position separately.

Missing measurements are ``NaN``, never ``0.0``.  Zero is a real value for
every ratio here (a perfectly centred hand has distance 0), so substituting it
for "not measured" teaches the model a false distribution.  The training
pipeline imputes inside cross-validation and the runtime feeds the same NaNs
to the fitted pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

KEYPOINT_FEATURES: list[str] = [
    *[f"kpt_x{i}" for i in range(17)],
    *[f"kpt_y{i}" for i in range(17)],
]

HEAD_POSE_FEATURES: list[str] = ["yaw", "pitch", "roll"]

GEOMETRIC_FEATURES: list[str] = [
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

SPATIAL_FEATURES: list[str] = [
    "frame_center_offset_x",
    "frame_center_offset_y",
    "bbox_coverage_ratio",
    "bbox_aspect_ratio",
]

TEMPORAL_FEATURES: list[str] = ["motion_delta"]

FEATURE_NAMES: list[str] = [
    *KEYPOINT_FEATURES,
    *HEAD_POSE_FEATURES,
    *GEOMETRIC_FEATURES,
    *SPATIAL_FEATURES,
    *TEMPORAL_FEATURES,
]

FEATURE_COUNT = len(FEATURE_NAMES)
assert FEATURE_COUNT == 54, FEATURE_COUNT

# Metadata columns are not model inputs.
METADATA_COLUMNS: list[str] = ["timestamp", "pose_id", "confidence"]
CSV_COLUMNS: list[str] = [*METADATA_COLUMNS, *FEATURE_NAMES, "source"]

# Minimum YOLO keypoint confidence accepted as a measurement.  Below this the
# coordinate is guessed, and a guessed coordinate is worse than a missing one.
MIN_KEYPOINT_CONFIDENCE = 0.25


def _finite(value: Any) -> float:
    """Coerce *value* to a finite float, or NaN when it is not a measurement."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def normalize_keypoints(
    keypoints: Sequence[Sequence[float]] | np.ndarray | None,
    bbox: Sequence[float] | None,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> np.ndarray:
    """Return a ``(17, 2)`` bbox-normalised keypoint array, NaN where missing.

    ``keypoints`` may be COCO ``(17, 2)`` or ``(17, 3)`` rows (x, y, conf).
    A keypoint outside its own person box is kept, not clipped: a raised hand
    genuinely lies outside the detector box and that is useful signal.  Only
    the confidence test and finite-coordinate test can mark it missing.
    """
    result = np.full((17, 2), np.nan, dtype=np.float64)
    if keypoints is None or bbox is None or len(bbox) < 4:
        return result

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return result
    width = x2 - x1
    height = y2 - y1
    if not (math.isfinite(width) and math.isfinite(height)):
        return result
    if width <= 1.0 or height <= 1.0:
        return result

    rows = np.asarray(keypoints, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return result

    for index in range(min(17, rows.shape[0])):
        row = rows[index]
        if row.size < 2:
            continue
        x = _finite(row[0])
        y = _finite(row[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if row.size >= 3:
            confidence = _finite(row[2])
            if not math.isfinite(confidence) or confidence < min_confidence:
                continue
        result[index, 0] = (x - x1) / width
        result[index, 1] = (y - y1) / height
    return result


def feature_dict_from_components(
    keypoints: Sequence[Sequence[float]] | np.ndarray | None = None,
    bbox: Sequence[float] | None = None,
    head_pose: Mapping[str, Any] | None = None,
    geometric: Mapping[str, Any] | None = None,
    spatial: Mapping[str, Any] | None = None,
    motion_delta: float | None = None,
) -> dict[str, float]:
    """Build the canonical 54-key feature dict.

    Every key is always present.  Values that were not measured are NaN so a
    caller cannot accidentally train on zero-filled holes.
    """
    features: dict[str, float] = {name: float("nan") for name in FEATURE_NAMES}

    normalised = normalize_keypoints(keypoints, bbox)
    for index in range(17):
        features[f"kpt_x{index}"] = float(normalised[index, 0])
        features[f"kpt_y{index}"] = float(normalised[index, 1])

    head = head_pose or {}
    for name in HEAD_POSE_FEATURES:
        features[name] = _finite(head.get(name))

    geom = geometric or {}
    for name in GEOMETRIC_FEATURES:
        features[name] = _finite(geom.get(name))

    space = spatial or {}
    for name in SPATIAL_FEATURES:
        features[name] = _finite(space.get(name))

    features["motion_delta"] = _finite(motion_delta)
    return features


def vector_from_dict(features: Mapping[str, Any]) -> np.ndarray:
    """Return the canonical 54-element float64 vector, NaN for missing keys."""
    return np.asarray(
        [_finite(features.get(name)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def validate_feature_names(names: Iterable[str] | None) -> bool:
    """True when *names* exactly matches the current canonical order."""
    if names is None:
        return False
    return list(names) == FEATURE_NAMES


__all__ = [
    "CSV_COLUMNS",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "GEOMETRIC_FEATURES",
    "HEAD_POSE_FEATURES",
    "KEYPOINT_FEATURES",
    "METADATA_COLUMNS",
    "MIN_KEYPOINT_CONFIDENCE",
    "SPATIAL_FEATURES",
    "TEMPORAL_FEATURES",
    "feature_dict_from_components",
    "normalize_keypoints",
    "validate_feature_names",
    "vector_from_dict",
]
