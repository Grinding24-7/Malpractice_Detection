"""
exporter.py — Asynchronous thread to export video clips from RAM buffer.

Design:
    - Runs in a dedicated daemon thread (non-blocking to the pipeline).
    - Receives export jobs via a thread-safe queue.Queue.
    - Each job: extract 15s pre + 15s post frames from CircularBuffer,
      encode to .mp4 via OpenCV VideoWriter, write to disk.
    - Cleans up its own memory after each job.
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from buffer import CircularBuffer, FramePacket

logger = logging.getLogger(__name__)

EXPORT_DIR = Path("./exports")
EXPORT_DIR.mkdir(exist_ok=True)

PRE_SECONDS: float = 15.0
POST_SECONDS: float = 15.0
OUTPUT_FPS: int = 30


@dataclass
class ExportJob:
    alert_id: str
    alert_type: str
    timestamp: float  # time.monotonic() when alert fired


@dataclass
class ExporterThread:
    """
    Daemon thread that consumes ExportJobs and writes .mp4 clips.

    Usage:
        exporter = ExporterThread()
        exporter.start()
        exporter.submit(ExportJob(...))
    """

    _queue: queue.Queue = field(default_factory=queue.Queue)
    _buffer_ref: Optional[CircularBuffer] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)

    def attach_buffer(self, buf: CircularBuffer) -> None:
        """Give the exporter a reference to the shared RAM buffer."""
        self._buffer_ref = buf

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name="Exporter")
        t.start()

    def submit(self, job: ExportJob) -> None:
        """Non-blocking enqueue.  Returns immediately."""
        self._queue.put_nowait(job)

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        logger.info("Exporter thread started.")
        while not self._stop_event.is_set():
            try:
                job: ExportJob = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._encode_clip(job)
            except Exception:
                logger.exception("Failed to export clip %s", job.alert_id)

            # Explicit garbage collection after each export to prevent
            # memory leaks from accumulating across video clips.
            gc.collect()

        logger.info("Exporter thread stopped.")

    def _encode_clip(self, job: ExportJob) -> None:
        """
        Extract frames from buffer, encode to .mp4, write to disk.

        Pre-event window:  15 s  before alert time
        Post-event window: 15 s  after alert time
        Total clip length: 30 s  (~900 frames at 30 FPS)
        """
        if self._buffer_ref is None:
            logger.warning("No buffer attached; skipping export.")
            return

        frames: list[FramePacket] = self._buffer_ref.extract_window(
            before=PRE_SECONDS + POST_SECONDS, after=0.0
        )
        if len(frames) == 0:
            logger.warning("Buffer empty; cannot export clip.")
            return

        # Build video writer
        clip_path = EXPORT_DIR / f"{job.alert_id}_{job.alert_type}.mp4"
        h, w = frames[0].frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, OUTPUT_FPS, (w, h))

        for pkt in frames:
            writer.write(pkt.frame)

        writer.release()
        logger.info("Exported clip: %s  (%d frames)", clip_path, len(frames))