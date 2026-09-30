"""
skeleton_buffer.py — Temporal sliding window for skeleton sequences.

Maintains a fixed-length buffer of per-frame keypoint arrays so that
ST-GCN can receive a (T, V, 3) tensor representing the last T frames
of skeleton data.

Thread-safe: uses a lock so the inference thread can read while the
detection thread writes.
"""

from __future__ import annotations

import collections
import threading
from typing import Optional

import numpy as np

# Default temporal window: 30 frames = 1 second at 30 FPS
DEFAULT_WINDOW_SIZE = 30
NUM_JOINTS = 17


class SkeletonBuffer:
    """Sliding-window buffer of skeleton keypoint arrays.

    Each entry is a (V, 3) array: V joints × (x, y, confidence).
    The buffer holds up to ``window_size`` frames.
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    def append(self, keypoints: Optional[np.ndarray]) -> None:
        """Append a single frame's keypoints to the buffer.

        Parameters
        ----------
        keypoints : np.ndarray or None
            Shape (V, 3) or (V, 2) for V joints.
            If None or wrong shape, a zero array is appended to maintain
            temporal alignment.
        """
        if keypoints is None:
            zeros = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
            with self._lock:
                self._buffer.append(zeros)
            return

        kpts = np.array(keypoints, dtype=np.float32)
        if kpts.ndim == 2 and kpts.shape[0] >= NUM_JOINTS:
            # Truncate to NUM_JOINTS if more (e.g. MediaPipe 33 → COCO 17)
            kpts = kpts[:NUM_JOINTS]
            if kpts.shape[1] == 2:
                # Pad confidence column
                kpts = np.column_stack([kpts, np.ones(NUM_JOINTS, dtype=np.float32)])
        else:
            zeros = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
            kpts = zeros

        with self._lock:
            self._buffer.append(kpts)

    def get_sequence(self) -> Optional[np.ndarray]:
        """Return the current buffer as a (T, V, 3) array.

        Returns None if the buffer is empty.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return np.stack(list(self._buffer), axis=0)

    def get_tensor(self) -> Optional[np.ndarray]:
        """Return the buffer in ST-GCN input shape: (C=3, T, V).

        This is the format expected by STGCNClassifier.
        """
        seq = self.get_sequence()
        if seq is None:
            return None
        # (T, V, 3) -> (3, T, V)
        return seq.transpose(2, 0, 1)

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def is_full(self) -> bool:
        """True if the buffer has reached window_size frames."""
        with self._lock:
            return len(self._buffer) >= self._window_size

    @property
    def size(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def window_size(self) -> int:
        return self._window_size
