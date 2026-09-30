"""
buffer.py — In-memory circular RAM buffer.

Architecture:
    Single collections.deque with fixed maxlen = FPS * BUFFER_SECONDS.
    Thread-safe via threading.Lock.
    Zero continuous disk I/O by design.
    Provides window extraction for exporter.py (pre-event / post-event slices).

Frame convention: numpy.ndarray, dtype=uint8, shape=(H, W, 3), BGR order.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import numpy as np

BUFFER_FPS: int = 30
BUFFER_SECONDS: int = 30
MAXLEN: int = BUFFER_FPS * BUFFER_SECONDS  # 900 frames


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float  # time.monotonic() at capture


@dataclass
class CircularBuffer:
    _buf: deque[FramePacket] = field(
        default_factory=lambda: deque(maxlen=MAXLEN)
    )
    _lock: Lock = field(default_factory=Lock)

    def push(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        packet = FramePacket(frame=frame, timestamp=time.monotonic())
        with self._lock:
            self._buf.append(packet)

    def latest(self) -> Optional[FramePacket]:
        with self._lock:
            try:
                return self._buf[-1]
            except IndexError:
                return None

    def extract_window(self, before: float, after: float) -> list[FramePacket]:
        with self._lock:
            length = len(self._buf)
            if length == 0:
                return []
            total_needed = int(before * BUFFER_FPS)
            start_idx = max(0, length - total_needed)
            result = list(self._buf)[start_idx:]
        return result

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def is_warm(self) -> bool:
        return len(self) >= MAXLEN

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()