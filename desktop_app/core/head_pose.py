"""
head_pose.py — 3D Head Pose Estimation via cv2.solvePnP.

Maps 2D facial landmarks (Nose tip, Chin, Left/Right Eye corners,
Left/Right Mouth corners) to a standard 3D face model, solves for
rotation + translation vectors, and extracts Euler angles (Yaw, Pitch, Roll).

Provides anomaly classification:
    - P12_LOOKING_AWAY: |Yaw| > 35 degrees
    - P14_LOOKING_DOWN: Pitch < -25 degrees

Also computes a gaze direction projection vector for HUD rendering.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# ── Standard 3D facial model points (millimetres, generic average) ────────
# Order: Nose tip, Chin, Left eye outer corner, Right eye outer corner,
#         Left mouth corner, Right mouth corner
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),         # Nose tip (origin)
        (0.0, -63.6, -12.5),     # Chin
        (-43.3, 32.7, -26.0),    # Left eye outer corner
        (43.3, 32.7, -26.0),     # Right eye outer corner
        (-28.9, -28.9, -24.1),   # Left mouth corner
        (28.9, -28.9, -24.1),    # Right mouth corner
    ],
    dtype=np.float64,
)

# Landmark indices in MediaPipe 33-point skeleton
MP_NOSE = 0
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_OUTER = 6
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10

# Anomaly thresholds (degrees)
YAW_LOOKING_AWAY_THRESHOLD = 35.0
PITCH_LOOKING_DOWN_THRESHOLD = -25.0


class HeadPoseEstimator:
    """3D head pose estimation via Perspective-n-Point.

    Produces Euler angles (Yaw, Pitch, Roll) and a gaze direction vector
    from MediaPipe 33-point landmarks mapped to a generic 3D face model.
    """

    def __init__(self) -> None:
        # Focal length expressed as a fraction of image WIDTH, not as a fixed
        # pixel count.  A hardcoded 700 px is only correct for one resolution:
        # at 1920 px wide the true focal is ~2.5x larger, so the PnP solve
        # returned a badly wrong pitch, which the rule engine then read as
        # "head down" (P32/P33) on high-resolution CCTV feeds.
        # 700/768 ~= 0.91, i.e. a ~58 degree horizontal field of view.
        self._focal_ratio = 700.0 / 768.0
        self._last_result: dict[str, Any] = {}

    def _focal_px(self, width: int) -> float:
        """Focal length in pixels for an image of the given width."""
        return max(1.0, self._focal_ratio * float(width))

    def estimate(
        self,
        landmarks: list[tuple[float, float, float]],
        image_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Estimate 3D head pose from MediaPipe landmarks.

        Parameters
        ----------
        landmarks : list of (x, y, visibility)
            33-point MediaPipe pose landmarks in pixel coordinates.
        image_shape : (height, width) or None
            Frame dimensions.  Used to set camera intrinsic matrix.

        Returns
        -------
        dict with keys:
            yaw, pitch, roll (float, degrees),
            gaze_vector (tuple[float, float, float]),
            rvec, tvec (np.ndarray or None),
            anomalies (list[str])
        """
        result: dict[str, Any] = {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "gaze_vector": (0.0, 0.0, 1.0),
            "rvec": None,
            "tvec": None,
            "anomalies": [],
        }

        if not landmarks or len(landmarks) < 33:
            return result

        # Every point below must actually be measured.  Solving PnP with
        # unmeasured landmarks returns a plausible-looking but meaningless
        # rotation (it was producing roll values near 180 degrees).
        required = (
            MP_NOSE,
            MP_LEFT_EYE_OUTER,
            MP_RIGHT_EYE_OUTER,
            MP_MOUTH_LEFT,
            MP_MOUTH_RIGHT,
        )
        if any(landmarks[i][2] < 0.5 for i in required):
            return result

        # ── Map 2D landmarks to image points ─────────────────────────────
        # Model point 1 is the CHIN, which MediaPipe does not provide.  It is
        # approximated by extrapolating from nose to mouth-centre: the model
        # places the nose at y=0, the mouth corners at y=-28.9 and the chin at
        # y=-63.6, so the chin sits ~2.2x further along that line.
        #
        # This previously supplied MP_MOUTH_LEFT here instead, which both
        # discarded the chin entirely and double-fitted the left mouth corner.
        nose_pt = self._pt(landmarks, MP_NOSE)
        mouth_mid = (
            self._pt(landmarks, MP_MOUTH_LEFT)
            + self._pt(landmarks, MP_MOUTH_RIGHT)
        ) / 2.0
        chin_pt = nose_pt + 2.2 * (mouth_mid - nose_pt)

        image_points = np.array(
            [
                nose_pt,
                chin_pt,  # Chin (approximated)
                self._pt(landmarks, MP_LEFT_EYE_OUTER),
                self._pt(landmarks, MP_RIGHT_EYE_OUTER),
                self._pt(landmarks, MP_MOUTH_LEFT),
                self._pt(landmarks, MP_MOUTH_RIGHT),
            ],
            dtype=np.float64,
        )

        # ── Camera intrinsic matrix ──────────────────────────────────────
        if image_shape is not None:
            h, w = image_shape
        else:
            h, w = 480, 640

        cam_mat = np.array(
            [
                [self._focal_px(w), 0, w / 2.0],
                [0, self._focal_px(w), h / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # ── Solve PnP ────────────────────────────────────────────────────
        # EPNP is the robust choice for a 6-point set; ITERATIVE is kept as a
        # fallback.  Without a valid initial guess ITERATIVE converged to
        # mirrored solutions (roll near 180 degrees).
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            cam_mat,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points,
                cam_mat,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not success:
            return result

        result["rvec"] = rvec
        result["tvec"] = tvec

        # ── Extract Euler angles ─────────────────────────────────────────
        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._rotation_matrix_to_euler(R)
        result["yaw"] = round(yaw, 2)
        result["pitch"] = round(pitch, 2)
        result["roll"] = round(roll, 2)

        # ── Compute gaze direction vector ────────────────────────────────
        result["gaze_vector"] = self._compute_gaze_vector(R, tvec)

        # ── Classify anomalies ───────────────────────────────────────────
        result["anomalies"] = self._classify_anomalies(
            result["yaw"], result["pitch"]
        )

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pt(
        landmarks: list[tuple[float, float, float]], idx: int
    ) -> np.ndarray:
        """Extract a 2D point from landmarks as float64 array."""
        return np.array([landmarks[idx][0], landmarks[idx][1]], dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
        """Decompose rotation matrix into Euler angles (degrees).

        Convention: Yaw (Y-axis), Pitch (X-axis), Roll (Z-axis).
        """
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy < 1e-6:
            yaw = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
            pitch = math.degrees(math.atan2(-R[2, 0], sy))
            roll = 0.0
        else:
            yaw = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            pitch = math.degrees(math.atan2(-R[2, 0], sy))
            roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return yaw, pitch, roll

    @staticmethod
    def _compute_gaze_vector(
        R: np.ndarray, tvec: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute the gaze direction vector in world coordinates.

        The gaze direction is the negative Z-axis of the head's local
        coordinate frame, transformed by the rotation matrix.
        """
        # Local forward direction (negative Z of the head frame)
        local_forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gaze = R @ local_forward
        # Normalise
        norm = np.linalg.norm(gaze)
        if norm > 1e-8:
            gaze = gaze / norm
        return (float(gaze[0]), float(gaze[1]), float(gaze[2]))

    @staticmethod
    def _classify_anomalies(yaw: float, pitch: float) -> list[str]:
        """Classify head pose into anomaly codes based on Euler angles."""
        anomalies: list[str] = []
        if abs(yaw) > YAW_LOOKING_AWAY_THRESHOLD:
            anomalies.append("P12_LOOKING_AWAY")
        if pitch < PITCH_LOOKING_DOWN_THRESHOLD:
            anomalies.append("P14_LOOKING_DOWN")
        return anomalies

    def get_gaze_endpoint(
        self,
        landmarks: list[tuple[float, float, float]],
        image_shape: tuple[int, int],
        length: float = 150.0,
    ) -> tuple[int, int] | None:
        """Project the 3D gaze direction onto the 2D image plane.

        Returns the (x, y) endpoint of the gaze ray on the image,
        or None if estimation is unavailable.
        """
        result = self._last_result
        rvec = result.get("rvec")
        tvec = result.get("tvec")
        if rvec is None or tvec is None:
            return None
        if not landmarks or len(landmarks) < 33:
            return None

        # Nose tip in 3D
        nose_3d = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        # Gaze endpoint in 3D (forward along local Z)
        gaze_3d = nose_3d + np.array(result["gaze_vector"]) * length

        # Project to 2D
        h, w = image_shape
        focal = self._focal_px(w)
        cam_mat = np.array(
            [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            pts_3d = np.array([nose_3d, gaze_3d], dtype=np.float64).reshape(-1, 1, 3)
            pts_2d, _ = cv2.projectPoints(
                pts_3d, rvec, tvec, cam_mat, dist_coeffs
            )
            end_pt = pts_2d[1].flatten()
            return (int(end_pt[0]), int(end_pt[1]))
        except Exception:
            return None
