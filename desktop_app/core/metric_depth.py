"""Metric 3D depth perception from monocular landmarks via ``cv2.solvePnP``.

Why this module exists
----------------------
The pixel-domain rules in the desktop app used raw distances such as
``wrist < 120 px from phone``.  A pixel is not a length: the same 120 px is
15 cm on a close webcam and 60 cm on a classroom CCTV feed.  Phone-usability
is a physical claim ("the hand is on the device"), so it needs metres.

The estimator here solves the Perspective-n-Point problem against a generic
3D face model (with a torso fallback) using the standard HD intrinsic matrix
required by the project::

    fx = 1000, fy = 1000, cx = 640, cy = 360

Those values describe a 1280x720 camera.  For any other frame size the
matrix is scaled proportionally, so a 768x432 clip gets fx=600, cx=384 and
the solve remains valid.  The translation vector ``tvec`` is then in metres
because the 3D model points are metres.

The depth is used to back-project 2D pixels into camera coordinates::

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth from solvePnP

A wrist and a desk object are both projected onto the candidate's depth
plane, which is the correct monocular approximation when the object sits on
the same desk as the student.  The resulting Euclidean distance is a metric
quantity, so the P31 threshold can be stated as 15 cm instead of pixels.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from desktop_app.core import perf_config

# MediaPipe 33-point indices used by the depth models.  Re-declared locally
# so this module has no import cycle with pose_classifier/object_context.
MP_NOSE = 0
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_OUTER = 6
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24

# Generic adult face model in METRES, OpenCV camera axes (x right, y down,
# z into the scene).  Values are the standard anthropometric averages used
# by the existing head-pose estimator, converted from millimetres.
FACE_MODEL_3D_M = np.array(
    [
        (0.0000, 0.0000, 0.0000),    # nose tip (origin)
        (0.0000, 0.0636, -0.0125),   # chin (below the nose)
        (-0.0433, -0.0327, -0.0260), # left eye outer corner
        (0.0433, -0.0327, -0.0260),  # right eye outer corner
        (-0.0289, 0.0289, -0.0241),  # left mouth corner
        (0.0289, 0.0289, -0.0241),   # right mouth corner
    ],
    dtype=np.float64,
)

# The PnP model uses only the five landmarks MediaPipe actually measures for
# this solve: nose tip, both outer eye corners and both mouth corners.  The
# chin is deliberately excluded -- MediaPipe has no chin point, and the old
# `nose + 2.2*(mouth_mid - nose)` extrapolation did not correspond to the 3D
# chin model, which biased the recovered depth.  The nose is out of the
# eye/mouth plane, so five non-coplanar points are enough for an unambiguous
# solve.
FACE_PNP_MODEL_3D_M = FACE_MODEL_3D_M[[0, 2, 3, 4, 5]]

# Torso fallback.  Used when the face is occluded but both shoulders, both
# hips and the nose are visible.  The points are deliberately non-coplanar
# (the nose sits forward of the shoulder/hip plane) because solvePnP cannot
# disambiguate a planar target.
TORSO_MODEL_3D_M = np.array(
    [
        (-0.1800, 0.0000, 0.0200),  # left shoulder
        (0.1800, 0.0000, 0.0200),   # right shoulder
        (-0.1200, 0.3000, 0.0000),  # left hip
        (0.1200, 0.3000, 0.0000),   # right hip
        (0.0000, -0.1200, -0.0800), # nose, forward and above shoulders
    ],
    dtype=np.float64,
)


def camera_matrix(frame_shape: tuple[int, ...] | None) -> np.ndarray:
    """Intrinsic matrix for *frame_shape*, scaled from the 1280x720 nominal.

    The project contract fixes fx=fy=1000, cx=640, cy=360 at 1280x720.  A
    different resolution must scale the whole matrix, otherwise cx=640 lies
    outside a 768-pixel-wide frame and every back-projection is nonsense.
    """
    if frame_shape is not None and len(frame_shape) >= 2:
        height, width = int(frame_shape[0]), int(frame_shape[1])
    else:
        height, width = (
            perf_config.METRIC_NOMINAL_HEIGHT,
            perf_config.METRIC_NOMINAL_WIDTH,
        )
    width = max(1, width)
    height = max(1, height)
    sx = width / float(perf_config.METRIC_NOMINAL_WIDTH)
    sy = height / float(perf_config.METRIC_NOMINAL_HEIGHT)
    return np.array(
        [
            [perf_config.METRIC_FX * sx, 0.0, perf_config.METRIC_CX * sx],
            [0.0, perf_config.METRIC_FY * sy, perf_config.METRIC_CY * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _landmark_xy(
    landmarks: list[tuple[float, float, float]] | None,
    idx: int,
) -> np.ndarray | None:
    """Visible landmark *idx* as a 2D float64 point, or None."""
    if not landmarks or len(landmarks) <= idx:
        return None
    try:
        x, y, vis = landmarks[idx]
        x, y, vis = float(x), float(y), float(vis)
    except (TypeError, ValueError, IndexError):
        return None
    if vis < perf_config.METRIC_MIN_VISIBILITY:
        return None
    if x != x or y != y:
        return None
    return np.array([x, y], dtype=np.float64)


def _solve_pnp(
    model_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Solve PnP, refine, and reject implausible or badly-fit solutions."""
    try:
        ok, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            intrinsics,
            None,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                model_points,
                image_points,
                intrinsics,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            return None
        # EPNP is robust to the minimal point set but can be slightly off;
        # Levenberg-Marquardt refinement is cheap and makes the depth stable.
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                model_points, image_points, intrinsics, None, rvec, tvec
            )
        except (cv2.error, AttributeError):  # pragma: no cover - OpenCV build
            pass
        projected, _ = cv2.projectPoints(
            model_points, rvec, tvec, intrinsics, None
        )
        residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        reprojection_px = float(np.mean(np.linalg.norm(residual, axis=1)))
        # OpenCV returns tvec as a (3, 1) array.  NumPy 2.x refuses
        # float(array([z])) ("only 0-dimensional arrays can be converted"),
        # so flatten before extracting Z.
        depth = float(np.asarray(tvec).reshape(-1)[2])
        if not math.isfinite(depth) or not math.isfinite(reprojection_px):
            return None
        if not (
            perf_config.METRIC_MIN_DEPTH_M
            <= depth
            <= perf_config.METRIC_MAX_DEPTH_M
        ):
            return None
        if reprojection_px > perf_config.METRIC_MAX_REPROJECTION_PX:
            return None
        return rvec, tvec, reprojection_px
    except Exception:
        return None


def estimate_candidate_depth(
    landmarks: list[tuple[float, float, float]] | None,
    frame_shape: tuple[int, ...] | None,
) -> dict[str, Any]:
    """Estimate the candidate's camera-frame pose and Z depth in metres.

    Returns a dict with ``success``, ``depth_m``, ``position_m``, ``rvec``,
    ``tvec``, ``camera_matrix``, ``method`` and ``reprojection_error_px``.
    On failure ``success`` is False and the numeric fields are None; callers
    must fail closed rather than fall back to pixels for a metric verdict.
    """
    result: dict[str, Any] = {
        "success": False,
        "depth_m": None,
        "position_m": None,
        "rvec": None,
        "tvec": None,
        "camera_matrix": camera_matrix(frame_shape),
        "method": "",
        "reprojection_error_px": None,
    }
    intrinsics = result["camera_matrix"]

    # ---- Face model -----------------------------------------------------
    nose = _landmark_xy(landmarks, MP_NOSE)
    left_eye = _landmark_xy(landmarks, MP_LEFT_EYE_OUTER)
    right_eye = _landmark_xy(landmarks, MP_RIGHT_EYE_OUTER)
    mouth_left = _landmark_xy(landmarks, MP_MOUTH_LEFT)
    mouth_right = _landmark_xy(landmarks, MP_MOUTH_RIGHT)
    if all(
        point is not None
        for point in (nose, left_eye, right_eye, mouth_left, mouth_right)
    ):
        image_points = np.array(
            [nose, left_eye, right_eye, mouth_left, mouth_right],
            dtype=np.float64,
        )
        solved = _solve_pnp(FACE_PNP_MODEL_3D_M, image_points, intrinsics)
        if solved is not None:
            rvec, tvec, reprojection = solved
            result.update({
                "success": True,
                "depth_m": round(float(np.asarray(tvec).reshape(-1)[2]), 4),
                "position_m": tuple(
                    round(float(v), 4)
                    for v in np.asarray(tvec).reshape(-1)
                ),
                "rvec": rvec,
                "tvec": tvec,
                "method": "face",
                "reprojection_error_px": round(reprojection, 3),
            })
            return result

    # ---- Torso fallback -------------------------------------------------
    left_shoulder = _landmark_xy(landmarks, MP_LEFT_SHOULDER)
    right_shoulder = _landmark_xy(landmarks, MP_RIGHT_SHOULDER)
    left_hip = _landmark_xy(landmarks, MP_LEFT_HIP)
    right_hip = _landmark_xy(landmarks, MP_RIGHT_HIP)
    if all(
        point is not None
        for point in (
            left_shoulder, right_shoulder, left_hip, right_hip, nose,
        )
    ):
        image_points = np.array(
            [left_shoulder, right_shoulder, left_hip, right_hip, nose],
            dtype=np.float64,
        )
        solved = _solve_pnp(TORSO_MODEL_3D_M, image_points, intrinsics)
        if solved is not None:
            rvec, tvec, reprojection = solved
            result.update({
                "success": True,
                "depth_m": round(float(np.asarray(tvec).reshape(-1)[2]), 4),
                "position_m": tuple(
                    round(float(v), 4)
                    for v in np.asarray(tvec).reshape(-1)
                ),
                "rvec": rvec,
                "tvec": tvec,
                "method": "torso",
                "reprojection_error_px": round(reprojection, 3),
            })
    return result


def pixel_to_point_m(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: np.ndarray | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Back-project pixel ``(u, v)`` onto the plane Z=*depth_m*."""
    matrix = (
        intrinsics if intrinsics is not None else camera_matrix(frame_shape)
    )
    z = float(depth_m)
    x = (float(u) - matrix[0, 2]) * z / max(1e-8, matrix[0, 0])
    y = (float(v) - matrix[1, 2]) * z / max(1e-8, matrix[1, 1])
    return np.array([x, y, z], dtype=np.float64)


def box_center_point_m(
    box: list[int] | list[float],
    depth_m: float,
    intrinsics: np.ndarray | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    """3D point at the centre of an xyxy box on the candidate depth plane."""
    if not box or len(box) != 4:
        return None
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return pixel_to_point_m(cx, cy, depth_m, intrinsics, frame_shape)


def wrist_points_m(
    landmarks: list[tuple[float, float, float]] | None,
    depth_m: float,
    intrinsics: np.ndarray | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> list[np.ndarray]:
    """Visible wrist keypoints back-projected onto the candidate depth plane."""
    points: list[np.ndarray] = []
    for idx in (MP_LEFT_WRIST, MP_RIGHT_WRIST):
        wrist = _landmark_xy(landmarks, idx)
        if wrist is None:
            continue
        points.append(
            pixel_to_point_m(
                wrist[0], wrist[1], depth_m, intrinsics, frame_shape
            )
        )
    return points


def wrist_object_distance_m(
    obj_box: list[int] | list[float],
    landmarks: list[tuple[float, float, float]] | None,
    depth_m: float | None,
    intrinsics: np.ndarray | None = None,
    frame_shape: tuple[int, ...] | None = None,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Closest wrist-to-object-centre distance in metres.

    Returns ``(distance_m, wrist_point, object_point)`` or None when the
    depth or a visible wrist is unavailable.  Callers must treat None as
    "not measured", never as "far away" or "close".
    """
    if depth_m is None or not landmarks:
        return None
    object_point = box_center_point_m(
        obj_box, depth_m, intrinsics, frame_shape
    )
    if object_point is None:
        return None
    wrists = wrist_points_m(landmarks, depth_m, intrinsics, frame_shape)
    if not wrists:
        return None
    best_wrist = min(
        wrists,
        key=lambda point: float(np.linalg.norm(point - object_point)),
    )
    distance = float(np.linalg.norm(best_wrist - object_point))
    return distance, best_wrist, object_point


def annotate_metric_distances(
    objects: list[dict[str, Any]],
    landmarks: list[tuple[float, float, float]] | None,
    depth_info: dict[str, Any] | None,
    frame_shape: tuple[int, ...] | None,
) -> None:
    """Attach ``wrist_dist_m`` / ``object_xyz_m`` to device detections.

    Runs in place on the classifier's own merged detection list before any
    escalation decides.  Metric annotation is only applied to objects that
    can be reached from a wrist; documents still use the desk-zone overlap
    rule, which is a 2D footprint question, not a reach question.
    """
    depth_m = (depth_info or {}).get("depth_m")
    intrinsics = (depth_info or {}).get("camera_matrix")
    if depth_m is None:
        return
    for obj in objects:
        label = str(obj.get("label", "")).lower()
        if label not in perf_config.OBJ_DEVICE_CLASSES:
            continue
        box = obj.get("box")
        if not box or len(box) != 4:
            continue
        measured = wrist_object_distance_m(
            box, landmarks, float(depth_m), intrinsics, frame_shape
        )
        if measured is None:
            obj["wrist_dist_m"] = None
            continue
        distance, wrist_point, object_point = measured
        obj["wrist_dist_m"] = round(distance, 4)
        obj["wrist_xyz_m"] = tuple(round(float(v), 4) for v in wrist_point)
        obj["object_xyz_m"] = tuple(round(float(v), 4) for v in object_point)
        # Replace the pixel-domain relation with the metric one.  The
        # shared escalation gate reads these fields.
        if distance < perf_config.PHONE_WRIST_DIST_M:
            obj["relation"] = "in_hand"
            obj["in_hand"] = True
            obj["reachable"] = True
        elif distance < perf_config.PHONE_WRIST_DIST_M * 3.0:
            obj["relation"] = "on_desk"
            obj["in_hand"] = False
            obj["reachable"] = True
        else:
            obj["relation"] = "away"
            obj["in_hand"] = False
            obj["reachable"] = False
