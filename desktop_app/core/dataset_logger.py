"""Automated feature logging for ML re-training.

Writes the canonical 54-dimensional vector defined by
``desktop_app.core.feature_schema`` plus metadata and source provenance.

Key changes from the previous logger:

* keypoints are normalised to the candidate bounding box, so a model trained
  on one camera resolution transfers to another;
* missing measurements are written as empty cells (``NaN`` on read), not
  ``0.0``;
* the four spatial features and ``motion_delta`` complete the 54-D contract;
* every row records where it came from (``live``, ``kaggle_exam``, ...).

Usage:
    logger = DatasetLogger()
    logger.start_recording()
    logger.log_frame(result, motion_delta=0.5)
    logger.close()
"""

from __future__ import annotations

import csv
import logging
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from desktop_app.core.feature_schema import (
    CSV_COLUMNS,
    FEATURE_NAMES,
    feature_dict_from_components,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CSV_PATH = _PROJECT_ROOT / "data" / "proctoring_features.csv"

# Kept as a module-level alias for callers/tests that imported the old name.
_CSV_COLUMNS: list[str] = list(CSV_COLUMNS)


def _first(value: Any) -> Any:
    """Return the first row of a list/array of candidates, or the value."""
    if value is None:
        return None
    if hasattr(value, "ndim"):
        if value.ndim == 0:
            return value
        return value[0]
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _first_keypoints(value: Any) -> Any:
    """Extract one (17, 2/3) keypoint array from a result payload."""
    if value is None:
        return None
    # (N, 17, 3) / (N, 17, 2) -> first candidate
    if hasattr(value, "ndim") and value.ndim >= 3:
        return value[0]
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if hasattr(first, "ndim"):
            return first
        # A flat list of 17 points is also valid.
        if isinstance(first, (list, tuple)) and first:
            if isinstance(first[0], (list, tuple)):
                return first
            return value
    return value


def _csv_number(value: Any) -> str | float:
    """Render NaN as an empty CSV cell; keep finite numbers as floats."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return round(number, 6)


class DatasetLogger:
    """Thread-safe per-frame feature logger.

    The file handle, writer, recording flag and counter are shared between
    the inference thread and the UI thread, so every mutation is locked.
    """

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV_PATH
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_COLUMNS)
        self._recording = False
        self._sample_count = 0
        self._lock = threading.Lock()

        if self._csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        with self._lock:
            self._recording = True
        logger.info("Dataset recording started -> %s", self._csv_path)

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
            samples = self._sample_count
        logger.info("Dataset recording stopped - %d samples logged", samples)

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_frame(
        self,
        result: dict[str, Any],
        motion_delta: float = 0.0,
    ) -> None:
        """Log one classification result as a 54-feature CSV row.

        ``result`` is the payload returned by ``PoseClassifier.classify()`` or
        the equivalent offline extraction dictionary.  Only the first
        candidate is logged; offline ingestion calls this once per candidate.
        """
        with self._lock:
            self._log_frame_locked(result, motion_delta)

    def _log_frame_locked(
        self,
        result: dict[str, Any],
        motion_delta: float,
    ) -> None:
        if not self._recording:
            return

        bbox = _first(result.get("bounding_boxes"))
        if bbox is None:
            bbox = result.get("bbox")

        keypoints = _first_keypoints(result.get("keypoints"))
        features = feature_dict_from_components(
            keypoints=keypoints,
            bbox=bbox,
            head_pose=result.get("head_pose"),
            geometric=result.get("geometric_features"),
            spatial=result.get("spatial_features"),
            motion_delta=motion_delta,
        )

        row: dict[str, str | float] = {
            "timestamp": datetime.now().isoformat(),
            "pose_id": str(result.get("pose_id") or "P01"),
            "confidence": round(float(result.get("confidence", 0.0)), 4),
            "source": str(result.get("source") or "live"),
        }
        for name in FEATURE_NAMES:
            row[name] = _csv_number(features.get(name))

        try:
            self._writer.writerow(row)
            self._file.flush()
            self._sample_count += 1
        except Exception as exc:  # noqa: BLE001 - a log failure must not kill inference
            logger.error("Failed to write CSV row: %s", exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()
