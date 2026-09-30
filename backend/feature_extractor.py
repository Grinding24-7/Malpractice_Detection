"""
feature_extractor.py — Week 2: pose feature extraction for dataset collection.

Turns a single person's COCO-17 pose keypoints into a compact 1D feature
vector consumed by the ML dataset pipeline.

Keypoint indices (COCO 17-keypoint skeleton):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear,
    4: right_ear, 5: left_shoulder, 6: right_shoulder
"""

from __future__ import annotations

import numpy as np

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def extract_pose_features(kpts: np.ndarray) -> np.ndarray:
    """
    Compute a 7-element feature vector from one person's keypoints.

    Args:
        kpts: (17, 3) NumPy array of keypoints in COCO order, columns
            are [x, y, confidence].

    Returns:
        float32 1D array:
            [ear_ratio, vertical_drop, shoulder_angle,
             nose_conf, l_ear_conf, r_ear_conf, shoulder_width]
    """
    kpts = np.asarray(kpts, dtype=np.float64)

    def pt(idx: int) -> tuple[float, float]:
        return float(kpts[idx, 0]), float(kpts[idx, 1])

    nose_x, nose_y = pt(NOSE)
    l_ear_x, l_ear_y = pt(LEFT_EAR)
    r_ear_x, r_ear_y = pt(RIGHT_EAR)
    l_sh_x, l_sh_y = pt(LEFT_SHOULDER)
    r_sh_x, r_sh_y = pt(RIGHT_SHOULDER)

    nose_to_left_ear = np.hypot(nose_x - l_ear_x, nose_y - l_ear_y)
    nose_to_right_ear = np.hypot(nose_x - r_ear_x, nose_y - r_ear_y)
    ear_ratio = nose_to_left_ear / (nose_to_right_ear + 1e-6)

    shoulders_mid_y = (l_sh_y + r_sh_y) / 2.0
    eyes_mid_y = (l_ear_y + r_ear_y) / 2.0
    vertical_drop = shoulders_mid_y - eyes_mid_y

    shoulder_angle = np.arctan2(r_sh_y - l_sh_y, r_sh_x - l_sh_x)

    shoulder_width = np.hypot(l_sh_x - r_sh_x, l_sh_y - r_sh_y)

    return np.asarray(
        [
            ear_ratio,
            vertical_drop,
            shoulder_angle,
            float(kpts[NOSE, 2]),
            float(kpts[LEFT_EAR, 2]),
            float(kpts[RIGHT_EAR, 2]),
            shoulder_width,
        ],
        dtype=np.float32,
    )


def extract_normalized_pose_features(kpts: np.ndarray) -> np.ndarray:
    """
    Compute a scale-invariant 7-element feature vector from one person's
    keypoints.  Distances are normalised by shoulder width so the features
    generalise across camera zoom / subject distance (typical of CCTV feeds).

    Args:
        kpts: (17, 3) NumPy array of keypoints in COCO order, columns
            are [x, y, confidence].

    Returns:
        float32 1D array:
            [ear_ratio, norm_vertical_drop, shoulder_angle,
             norm_nose_ear_drop, nose_conf, l_ear_conf, r_ear_conf]
    """
    kpts = np.asarray(kpts, dtype=np.float64)

    def pt(idx: int) -> tuple[float, float]:
        return float(kpts[idx, 0]), float(kpts[idx, 1])

    nose_x, nose_y = pt(NOSE)
    l_ear_x, l_ear_y = pt(LEFT_EAR)
    r_ear_x, r_ear_y = pt(RIGHT_EAR)
    l_eye_x, l_eye_y = pt(LEFT_EYE)
    r_eye_x, r_eye_y = pt(RIGHT_EYE)
    l_sh_x, l_sh_y = pt(LEFT_SHOULDER)
    r_sh_x, r_sh_y = pt(RIGHT_SHOULDER)

    # Scaling factor (in pixels) — +1e-6 guards against a degenerate zero width.
    shoulder_width = np.hypot(l_sh_x - r_sh_x, l_sh_y - r_sh_y) + 1e-6

    # (a) Head yaw proxy: how asymmetric the nose-to-ear distances are.
    nose_to_left_ear = np.hypot(nose_x - l_ear_x, nose_y - l_ear_y)
    nose_to_right_ear = np.hypot(nose_x - r_ear_x, nose_y - r_ear_y)
    ear_ratio = nose_to_left_ear / (nose_to_right_ear + 1e-6)

    # (b) Pitch / leaning proxy: vertical drop from eye line to shoulders.
    eyes_mid_y = (l_eye_y + r_eye_y) / 2.0
    shoulders_mid_y = (l_sh_y + r_sh_y) / 2.0
    norm_vertical_drop = (shoulders_mid_y - eyes_mid_y) / shoulder_width

    # (c) Body tilt proxy: shoulder line angle relative to the image x-axis.
    shoulder_angle = np.arctan2(r_sh_y - l_sh_y, r_sh_x - l_sh_x)

    # (d) Downward-tilt metric: how far the nose sinks below the ear line.
    ears_mid_y = (l_ear_y + r_ear_y) / 2.0
    norm_nose_ear_drop = (nose_y - ears_mid_y) / shoulder_width

    return np.asarray(
        [
            ear_ratio,
            norm_vertical_drop,
            shoulder_angle,
            norm_nose_ear_drop,
            float(kpts[NOSE, 2]),
            float(kpts[LEFT_EAR, 2]),
            float(kpts[RIGHT_EAR, 2]),
        ],
        dtype=np.float32,
    )
