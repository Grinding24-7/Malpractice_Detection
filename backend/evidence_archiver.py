"""
evidence_archiver.py — Week 7: Automated evidence clip recorder.

Responsibilities:
    1. On anomaly trigger, retrieve pre-roll frames from BufferManager.
    2. Collect post-roll frames for a configurable duration after trigger.
    3. Asynchronously write the complete clip (pre + post) to an MP4 file
       via ffmpeg subprocess (H.264, browser-compatible).
    4. Generate a JSON metadata sidecar per clip with event details and
       keypoint telemetry.

Non-blocking design:
    All file I/O and video encoding runs in a concurrent.futures.ThreadPoolExecutor
    so the real-time 30-FPS video stream is never stalled.

Clip layout:
    evidence_vault/
        clips/
            EVT_20260824_104200_3.mp4
        metadata/
            EVT_20260824_104200_3.json
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from buffer_manager import BufferManager, TrackFrame, get_buffer_manager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
VAULT_DIR = BACKEND_DIR / "evidence_vault"
CLIPS_DIR = VAULT_DIR / "clips"
METADATA_DIR = VAULT_DIR / "metadata"
OUTPUT_FPS: int = 30
POST_ROLL_COLLECT_SECONDS: float = 1.0  # collect 1 second of post-roll
MAX_EXPORT_WORKERS: int = 2  # bounded thread pool for video encoding


@dataclass
class EvidenceEvent:
    """Metadata for a single evidence clip."""
    event_id: str
    track_id: int
    malpractice_type: str
    confidence: float
    timestamp: str                # ISO 8601
    clip_path: str                # relative path to .mp4
    metadata_path: str            # relative path to .json
    pre_roll_frames: int
    post_roll_frames: int
    total_frames: int
    keypoint_telemetry: list = field(default_factory=list)
    severity: str = "medium"
    reviewed: bool = False
    bookmarked: bool = False


class EvidenceArchiver:
    """
    Asynchronous evidence clip recorder.

    Usage:
        archiver = EvidenceArchiver()
        archiver.start()
        # On anomaly trigger:
        archiver.trigger_export(
            track_id=4,
            malpractice_type="NOTE_PASSING",
            confidence=0.94,
            frame_id=1042,
        )
        # archiver collects post-roll frames, then writes clip + metadata.
    """

    def __init__(
        self,
        buffer_manager: Optional[BufferManager] = None,
        vault_dir: Path = VAULT_DIR,
        max_workers: int = MAX_EXPORT_WORKERS,
    ) -> None:
        self._bm = buffer_manager or get_buffer_manager()
        self._vault_dir = vault_dir
        self._clips_dir = vault_dir / "clips"
        self._metadata_dir = vault_dir / "metadata"
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir.mkdir(parents=True, exist_ok=True)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="evidence-export",
        )
        self._active_exports: dict[int, threading.Event] = {}  # track_id -> stop event
        self._export_lock = threading.Lock()
        self._events: list[EvidenceEvent] = []
        self._events_lock = threading.Lock()

    def start(self) -> None:
        """Initialize directories.  Called once at app startup."""
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        print("[archiver] evidence vault initialized", flush=True)

    def stop(self) -> None:
        """Cancel any pending post-roll collection and shutdown executor."""
        with self._export_lock:
            for evt in self._active_exports.values():
                evt.set()
            self._active_exports.clear()
        self._executor.shutdown(wait=False)

    # -- Trigger export ------------------------------------------------------

    def trigger_export(
        self,
        track_id: int,
        malpractice_type: str,
        confidence: float,
        frame_id: int,
        keypoints: Optional[list] = None,
    ) -> Optional[str]:
        """
        Start an evidence export for a track.

        1. Snapshot pre-roll frames from BufferManager immediately.
        2. Start a background thread to collect post-roll frames.
        3. Encode + write clip + metadata on the thread pool.

        Returns the event_id, or None if this track is already being exported.
        """
        # Prevent duplicate exports for the same track
        with self._export_lock:
            if track_id in self._active_exports:
                return None
            stop_event = threading.Event()
            self._active_exports[track_id] = stop_event

        # Generate event ID
        ts = datetime.now(timezone.utc)
        event_id = f"EVT_{ts.strftime('%Y%m%d_%H%M%S')}_{track_id}"

        # Snapshot pre-roll
        pre_frames = self._bm.extract_pre_roll(track_id)

        # Launch post-roll collection + encoding in background
        self._executor.submit(
            self._collect_and_encode,
            event_id=event_id,
            track_id=track_id,
            malpractice_type=malpractice_type,
            confidence=confidence,
            frame_id=frame_id,
            pre_frames=pre_frames,
            keypoints=keypoints or [],
            stop_event=stop_event,
        )

        print(f"[archiver] export triggered: {event_id} (track {track_id})", flush=True)
        return event_id

    def _collect_and_encode(
        self,
        event_id: str,
        track_id: int,
        malpractice_type: str,
        confidence: float,
        frame_id: int,
        pre_frames: list[TrackFrame],
        keypoints: list,
        stop_event: threading.Event,
    ) -> None:
        """
        Background task: collect post-roll frames, then encode clip + metadata.

        Polls the buffer manager every 33ms (~1 frame at 30 FPS) for
        POST_ROLL_COLLECT_SECONDS, then finalizes the clip.
        """
        try:
            post_frames: list[TrackFrame] = []
            collect_until = time.monotonic() + POST_ROLL_COLLECT_SECONDS
            while time.monotonic() < collect_until and not stop_event.is_set():
                latest = self._bm.extract_pre_roll(track_id)
                # Grab frames after the last pre-roll frame
                if latest:
                    last_pre_id = pre_frames[-1].frame_id if pre_frames else -1
                    new_post = [f for f in latest if f.frame_id > last_pre_id]
                    if new_post:
                        post_frames = new_post
                time.sleep(0.033)  # ~30 FPS polling

            # Combine pre + post
            all_frames = pre_frames + post_frames

            # Write clip
            clip_filename = f"{event_id}.mp4"
            clip_path = self._clips_dir / clip_filename
            self._write_clip(clip_path, all_frames)

            # Build keypoint telemetry from pre-roll
            kp_telemetry = []
            for f in pre_frames:
                if f.keypoints:
                    kp_telemetry.append({
                        "frame_id": f.frame_id,
                        "timestamp": f.timestamp,
                        "students": f.keypoints,
                    })

            # Write metadata sidecar
            meta = EvidenceEvent(
                event_id=event_id,
                track_id=track_id,
                malpractice_type=malpractice_type,
                confidence=round(confidence, 4),
                timestamp=datetime.now(timezone.utc).isoformat(),
                clip_path=f"evidence_vault/clips/{clip_filename}",
                metadata_path=f"evidence_vault/metadata/{event_id}.json",
                pre_roll_frames=len(pre_frames),
                post_roll_frames=len(post_frames),
                total_frames=len(all_frames),
                keypoint_telemetry=kp_telemetry,
                severity=self._classify_severity(malpractice_type, confidence),
            )
            meta_path = self._metadata_dir / f"{event_id}.json"
            meta_path.write_text(json.dumps(meta.__dict__, indent=2, default=str))

            # Store event in index
            with self._events_lock:
                self._events.append(meta)

            print(
                f"[archiver] exported {clip_filename}: "
                f"{len(all_frames)} frames ({len(pre_frames)} pre + {len(post_frames)} post)",
                flush=True,
            )

        except Exception as exc:
            print(f"[archiver] export failed for {event_id}: {exc}", flush=True)
        finally:
            with self._export_lock:
                self._active_exports.pop(track_id, None)

    def _write_clip(self, path: Path, frames: list[TrackFrame]) -> None:
        """
        Write frames to an MP4 file using ffmpeg (H.264, browser-compatible).

        Uses ffmpeg subprocess piped input instead of OpenCV VideoWriter to
        produce browser-playable .mp4 files.
        """
        if not frames:
            return

        first_frame = frames[0].frame
        h, w = first_frame.shape[:2]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{w}x{h}",
            "-framerate", str(OUTPUT_FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for pkt in frames:
                proc.stdin.write(pkt.frame.tobytes())
            proc.stdin.close()
            proc.wait(timeout=30)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            proc.kill()
            raise

    @staticmethod
    def _classify_severity(malpractice_type: str, confidence: float) -> str:
        if malpractice_type == "NOTE_PASSING":
            return "high"
        if malpractice_type == "PEEKING" and confidence > 0.9:
            return "high"
        if confidence > 0.85:
            return "medium"
        return "low"

    # -- Query ---------------------------------------------------------------

    def list_events(
        self,
        malpractice_type: Optional[str] = None,
        track_id: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """
        Query archived evidence clips with filtering and pagination.

        Returns a list of event dicts sorted by timestamp (newest first).
        """
        with self._events_lock:
            events = list(self._events)

        # Load any events from disk that aren't in memory yet
        events = self._load_all_metadata()

        # Apply filters
        if malpractice_type:
            events = [e for e in events if e.get("malpractice_type") == malpractice_type]
        if track_id is not None:
            events = [e for e in events if e.get("track_id") == track_id]
        if date_from:
            events = [e for e in events if e.get("timestamp", "") >= date_from]
        if date_to:
            events = [e for e in events if e.get("timestamp", "") <= date_to]

        # Sort newest first
        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

        # Paginate
        return events[offset:offset + limit]

    def get_event(self, event_id: str) -> Optional[dict]:
        """Retrieve a single event's metadata by event_id."""
        # Check in-memory first
        with self._events_lock:
            for ev in self._events:
                if ev.event_id == event_id:
                    return ev.__dict__
        # Fall back to disk
        meta_path = self._metadata_dir / f"{event_id}.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return None

    def delete_event(self, event_id: str) -> bool:
        """Delete an evidence clip and its metadata. Returns True if found."""
        clip_path = self._clips_dir / f"{event_id}.mp4"
        meta_path = self._metadata_dir / f"{event_id}.json"

        deleted = False
        if clip_path.exists():
            clip_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        if deleted:
            with self._events_lock:
                self._events = [e for e in self._events if e.event_id != event_id]

        return deleted

    def _load_all_metadata(self) -> list[dict]:
        """Load all JSON metadata sidecars from disk."""
        events = []
        if not self._metadata_dir.exists():
            return events
        for f in self._metadata_dir.iterdir():
            if f.suffix == ".json":
                try:
                    events.append(json.loads(f.read_text()))
                except (json.JSONDecodeError, OSError):
                    continue
        return events


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_archiver: Optional[EvidenceArchiver] = None


def get_evidence_archiver() -> EvidenceArchiver:
    global _archiver
    if _archiver is None:
        _archiver = EvidenceArchiver()
    return _archiver
