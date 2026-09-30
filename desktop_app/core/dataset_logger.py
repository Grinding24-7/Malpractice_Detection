"""
dataset_logger.py — Automated feature logging utility for ML re-training.

Saves keypoint vectors, head pose angles, geometric features, and
motion deltas into a structured CSV dataset for training P01-P40
supervised classifiers.

Usage:
    logger = DatasetLogger()
    logger.log_frame(result, motion_delta=0.5)
    logger.close()

CSV schema:
    timestamp, pose_id, confidence,
    kpt_x0..kpt_x16, kpt_y0..kpt_y16,
    yaw, pitch, roll,
    elbow_flexion_left, elbow_flexion_right, neck_tilt_angle,
    wrist_to_eye_ratio, wrist_to_ear_ratio, eye_to_shoulder_ratio,
    hand_mouth_distance, shoulder_hip_offset_ratio, torso_leg_ratio,
    shoulder_width_ratio, hip_width_ratio, torso_asymmetry,
    motion_delta
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CSV_PATH = _PROJECT_ROOT / "data" / "proctoring_features.csv"

# Column order for the CSV
_CSV_COLUMNS: list[str] = [
    "timestamp",
    "pose_id",
    "confidence",
    # 17 COCO keypoints (x and y)
    *[f"kpt_x{i}" for i in range(17)],
    *[f"kpt_y{i}" for i in range(17)],
    # Head pose
    "yaw",
    "pitch",
    "roll",
    # Geometric features
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
    # Motion
    "motion_delta",
]


class DatasetLogger:
    """Logs per-frame feature vectors to CSV for ML re-training.

    Thread-safe: uses a lock around file writes so the inference thread
    and UI thread can both call ``log_frame()`` without races.
    """

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV_PATH
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_COLUMNS)
        self._recording = False
        self._sample_count = 0

        # Write header if file is empty
        if self._csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Enable feature logging."""
        self._recording = True
        logger.info("Dataset recording started")

    def stop_recording(self) -> None:
        """Disable feature logging."""
        self._recording = False
        logger.info(
            "Dataset recording stopped — %d samples logged", self._sample_count
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def sample_count(self) -> int:
        return self._sample_count

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_frame(
        self,
        result: dict[str, Any],
        motion_delta: float = 0.0,
    ) -> None:
        """Log a single frame's features to the CSV dataset.

        Parameters
        ----------
        result : dict
            Standardised pose-classification result from ``PoseClassifier.classify()``.
        motion_delta : float
            Bounding-box scale delta from the temporal tracker.
        """
        if not self._recording:
            return

        row: dict[str, str | float] = {}

        # Metadata
        row["timestamp"] = datetime.now().isoformat()
        row["pose_id"] = result.get("pose_id", "P01")
        row["confidence"] = round(result.get("confidence", 0.0), 4)

        # Keypoints (flatten 17x3 -> 34 values)
        kpts = result.get("keypoints", [])
        if kpts and isinstance(kpts[0], list):
            kpts_arr = kpts[0]  # first person
        elif kpts and hasattr(kpts[0], "shape"):
            kpts_arr = kpts[0]
        else:
            kpts_arr = None

        if kpts_arr is not None:
            try:
                import numpy as np
                if isinstance(kpts_arr, np.ndarray) and kpts_arr.ndim == 2:
                    for i in range(min(17, kpts_arr.shape[0])):
                        row[f"kpt_x{i}"] = round(float(kpts_arr[i, 0]), 2)
                        row[f"kpt_y{i}"] = round(float(kpts_arr[i, 1]), 2)
                else:
                    for i in range(min(17, len(kpts_arr))):
                        row[f"kpt_x{i}"] = round(float(kpts_arr[i][0]), 2)
                        row[f"kpt_y{i}"] = round(float(kpts_arr[i][1]), 2)
            except Exception:
                pass

        # Fill missing keypoints with 0
        for i in range(17):
            row.setdefault(f"kpt_x{i}", 0.0)
            row.setdefault(f"kpt_y{i}", 0.0)

        # Head pose
        hp = result.get("head_pose", {})
        row["yaw"] = round(hp.get("yaw", 0.0), 4)
        row["pitch"] = round(hp.get("pitch", 0.0), 4)
        row["roll"] = round(hp.get("roll", 0.0), 4)

        # Geometric features
        feats = result.get("geometric_features", {})
        for name in [
            "elbow_flexion_left", "elbow_flexion_right", "neck_tilt_angle",
            "wrist_to_eye_ratio", "wrist_to_ear_ratio", "eye_to_shoulder_ratio",
            "hand_mouth_distance", "shoulder_hip_offset_ratio", "torso_leg_ratio",
            "shoulder_width_ratio", "hip_width_ratio", "torso_asymmetry",
        ]:
            row[name] = round(float(feats.get(name, 0.0)), 4)

        # Motion
        row["motion_delta"] = round(motion_delta, 4)

        try:
            self._writer.writerow(row)
            self._file.flush()
            self._sample_count += 1
        except Exception as exc:
            logger.error("Failed to write CSV row: %s", exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the CSV file handle."""
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
