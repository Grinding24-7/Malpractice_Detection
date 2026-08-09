"""
dataset_collector.py — Week 2: pose feature dataset collection.

Appends extracted feature vectors to a CSV file so that labelled posture
data can be used to train a classifier later.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

DATASET_FILENAME = "pose_dataset.csv"
DATASET_HEADERS = [
    "ear_ratio",
    "vertical_drop",
    "shoulder_angle",
    "nose_conf",
    "l_ear_conf",
    "r_ear_conf",
    "shoulder_width",
    "label",
]

_lock = threading.Lock()
_dataset_path: Path | None = None


def _resolve_path() -> Path:
    """Return the dataset CSV path (cwd-relative, like the video source)."""
    return Path(DATASET_FILENAME)


def set_dataset_path(path: str | Path) -> None:
    """Allow overriding the dataset CSV location (used by tests/app)."""
    global _dataset_path
    _dataset_path = Path(path)


def initialize_dataset() -> None:
    """Ensure 'pose_dataset.csv' exists with the correct headers."""
    path = _dataset_path if _dataset_path is not None else _resolve_path()
    if path.is_file() and path.stat().st_size > 0:
        return
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(DATASET_HEADERS)


def log_feature_vector(features, label: int) -> None:
    """Append a feature vector row and integer label to the dataset CSV."""
    path = _dataset_path if _dataset_path is not None else _resolve_path()
    initialize_dataset()
    row = [float(v) for v in features] + [int(label)]
    with _lock:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
