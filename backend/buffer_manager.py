"""
buffer_manager.py — Week 7: Memory-safe per-track RAM ring buffer with GC.

Architecture:
    Per-track_id circular buffers storing raw BGR frames and keypoint
    features.  Thread-safe via threading.Lock.  Automatic garbage collection
    prunes tracks unseen for >60 frames.  Memory guard monitors process RAM
    via psutil and flushes oldest inactive buffers when usage exceeds 80%.

Integration:
    Called by StreamingEngine (streaming_backend.py) and EvidenceArchiver
    to store/retrieve pre-roll frames around anomaly events.

Buffer convention:
    Raw frames: numpy.ndarray, dtype=uint8, shape=(H, W, 3), BGR order.
    Keypoints:  list[dict] per frame with track_id, bbox, prediction,
                confidence, keypoints, velocity_spike.
"""

from __future__ import annotations

import gc
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUFFER_FPS: int = 30
PRE_ROLL_FRAMES: int = 30        # ~1 second at 30 FPS
POST_ROLL_FRAMES: int = 30       # ~1 second at 30 FPS
MAX_BUFFER_FRAMES: int = 90      # 3 seconds per track (pre + post headroom)
STALE_THRESHOLD_FRAMES: int = 60  # prune tracks unseen for >2 seconds @ 30 FPS
MEMORY_HIGH_WATERMARK: float = 0.80  # 80% process RAM triggers flush


@dataclass
class TrackFrame:
    """Single frame stored in a per-track buffer."""
    frame: np.ndarray               # BGR uint8 (H, W, 3)
    timestamp: float                # time.monotonic()
    frame_id: int                   # global monotonic frame counter
    keypoints: Optional[list] = None  # detection telemetry for this frame


@dataclass
class TrackBuffer:
    """Circular buffer for a single tracked student."""
    track_id: int
    frames: deque[TrackFrame] = field(
        default_factory=lambda: deque(maxlen=MAX_BUFFER_FRAMES)
    )
    last_seen_frame: int = 0        # last global frame_id where this track appeared
    created_at: float = field(default_factory=time.monotonic)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class BufferManager:
    """
    Memory-safe per-track RAM ring buffer with automatic GC.

    Manages a dict of TrackBuffer instances keyed by ByteTrack track_id.
    Thread-safe for concurrent push/extract/prune operations from the
    producer thread and evidence archiver.
    """
    _buffers: dict[int, TrackBuffer] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _global_frame_counter: int = 0
    _total_bytes: int = 0           # estimated RAM usage in bytes

    # -- Push ----------------------------------------------------------------

    def push_frame(
        self,
        track_id: int,
        frame: np.ndarray,
        frame_id: int,
        keypoints: Optional[list] = None,
    ) -> None:
        """
        Append a raw frame + optional telemetry to a track's buffer.

        O(1) amortized via deque append.  Creates a new TrackBuffer for
        unseen track_ids lazily.
        """
        if frame is None:
            return
        pkt = TrackFrame(
            frame=frame,
            timestamp=time.monotonic(),
            frame_id=frame_id,
            keypoints=keypoints,
        )
        with self._lock:
            if track_id not in self._buffers:
                self._buffers[track_id] = TrackBuffer(track_id=track_id)
            buf = self._buffers[track_id]
            # Estimate memory delta: old frame leaving + new frame entering
            if len(buf.frames) == buf.frames.maxlen:
                old_frame = buf.frames[0]
                self._total_bytes -= old_frame.frame.nbytes
            buf.frames.append(pkt)
            self._total_bytes += frame.nbytes
            buf.last_seen_frame = frame_id
            self._global_frame_counter = max(self._global_frame_counter, frame_id)

    # -- Extract -------------------------------------------------------------

    def extract_pre_roll(self, track_id: int) -> list[TrackFrame]:
        """
        Return the most recent PRE_ROLL_FRAMES from a track's buffer.

        Used by EvidenceArchiver to get frames before an anomaly trigger.
        """
        with self._lock:
            buf = self._buffers.get(track_id)
            if buf is None:
                return []
            frames = list(buf.frames)
            return frames[-PRE_ROLL_FRAMES:] if len(frames) >= PRE_ROLL_FRAMES else frames

    def extract_clip(self, track_id: int) -> list[TrackFrame]:
        """
        Return the full current buffer for a track (pre-roll + any post frames).

        Called after the post-roll collection period completes.
        """
        with self._lock:
            buf = self._buffers.get(track_id)
            if buf is None:
                return []
            return list(buf.frames)

    # -- GC ------------------------------------------------------------------

    def prune_stale(self) -> int:
        """
        Drop buffers for track_ids unseen for > STALE_THRESHOLD_FRAMES.

        Must be called periodically (e.g., every inference tick).
        Returns the number of tracks pruned.
        """
        with self._lock:
            if not self._buffers:
                return 0
            current_frame = self._global_frame_counter
            stale_ids = [
                tid for tid, buf in self._buffers.items()
                if current_frame - buf.last_seen_frame > STALE_THRESHOLD_FRAMES
            ]
            for tid in stale_ids:
                buf = self._buffers.pop(tid)
                self._total_bytes -= sum(f.frame.nbytes for f in buf.frames)
            if stale_ids:
                gc.collect()
            return len(stale_ids)

    def flush_inactive(self, keep_ids: set[int] | None = None) -> int:
        """
        Flush all buffers not in keep_ids.  Used by memory guard under pressure.
        Returns the number of tracks flushed.
        """
        with self._lock:
            if keep_ids is None:
                keep_ids = set()
            inactive = [tid for tid in self._buffers if tid not in keep_ids]
            for tid in inactive:
                buf = self._buffers.pop(tid)
                self._total_bytes -= sum(f.frame.nbytes for f in buf.frames)
            if inactive:
                gc.collect()
            return len(inactive)

    # -- Memory guard --------------------------------------------------------

    def check_memory_pressure(self) -> bool:
        """
        Returns True if process RSS exceeds MEMORY_HIGH_WATERMARK of system RAM.

        Uses psutil to read current process memory and system total memory.
        """
        try:
            import psutil
            proc = psutil.Process()
            rss_bytes = proc.memory_info().rss
            total_ram = psutil.virtual_memory().total
            usage_ratio = rss_bytes / total_ram
            return usage_ratio > MEMORY_HIGH_WATERMARK
        except (ImportError, AttributeError, ZeroDivisionError):
            return False

    def memory_usage_mb(self) -> float:
        """Return current process RSS in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except (ImportError, AttributeError):
            return 0.0

    # -- Introspection -------------------------------------------------------

    def active_ids(self) -> list[int]:
        """Live track_ids currently held in memory."""
        with self._lock:
            return list(self._buffers.keys())

    def buffer_size(self, track_id: int) -> int:
        """Number of frames buffered for a specific track."""
        with self._lock:
            buf = self._buffers.get(track_id)
            return len(buf) if buf else 0

    def total_frames(self) -> int:
        """Total frames across all tracks."""
        with self._lock:
            return sum(len(buf) for buf in self._buffers.values())

    def estimated_memory_mb(self) -> float:
        """Estimated RAM used by all buffered frames (in MB)."""
        with self._lock:
            return self._total_bytes / (1024 * 1024)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffers)

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._total_bytes = 0
            gc.collect()


# ---------------------------------------------------------------------------
# Singleton for app-wide access
# ---------------------------------------------------------------------------
_manager: Optional[BufferManager] = None


def get_buffer_manager() -> BufferManager:
    global _manager
    if _manager is None:
        _manager = BufferManager()
    return _manager
