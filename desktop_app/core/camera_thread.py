"""
camera_thread.py — QThread that captures frames from a webcam, RTSP/CCTV
stream, video file, or mock source.

Architecture:
    CameraThread is a pure capture engine. It reads frames at maximum native
    camera speed, emits raw frames via Qt Signal for smooth UI rendering,
    and maintains a thread-safe single-frame lossy RAM buffer that the
    InferenceThread samples from asynchronously.

Source types:
    - USB webcam (integer device index, auto-scan 0-3)
    - RTSP/RTMP network stream (string URL, with auto-reconnection)
    - Video file (string path)
    - Image folder or image-list file (.txt/.csv) for dataset replay
    - Synthetic mock frames (fallback)

Priority on Windows: cv2.CAP_DSHOW for device indices.
Fallback chain: specified source -> webcam scan -> video file -> mock.
Never throws to the GUI; all errors are logged and recovered from.
"""

from __future__ import annotations

import collections
import csv
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from desktop_app.core import perf_config

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSETS_DIR = _PROJECT_ROOT / "assets"
_BACKEND_DIR = _PROJECT_ROOT / "backend"

_MOCK_W, _MOCK_H = 640, 480
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_LIST_EXTS = {".txt", ".csv"}

# RTSP reconnection constants
RTSP_MAX_RECONNECT_ATTEMPTS = 10
RTSP_INITIAL_BACKOFF = 1.0
RTSP_MAX_BACKOFF = 30.0
RTSP_OPEN_TIMEOUT_MS = 10_000
RTSP_READ_TIMEOUT_MS = 5_000

# ── Dead-source detection ───────────────────────────────────────────────
# A camera that is unplugged, pointed at a lens cap, or fed a corrupt file
# keeps returning "valid" frames: correct dimensions, correct dtype, all
# black.  Downstream this is indistinguishable from an empty room, so the
# classifier reports "Absent Seat" and the room looks monitored when nothing
# is being seen at all.  These constants let the capture thread notice.
#
# BLACK_LUMA_THRESHOLD lives in perf_config so the capture watchdog and the
# classifier agree on exactly what counts as "no image".  Import it rather than
# redefining it here; the two checks must never disagree.
BLACK_LUMA_THRESHOLD = perf_config.BLACK_LUMA_THRESHOLD
# Consecutive dark frames before the source is declared dead.  At 30 fps this
# is ~1 s, long enough to ride out a lamp cycling or a dark cut in footage but
# short enough that an operator notices quickly.
BLACK_STREAK_FRAMES = 30
# Consecutive readable frames required to consider the source recovered.
BLACK_RECOVERY_FRAMES = 5
# Minimum seconds between repeated camera-fault notices.  A source that flaps
# dead/live (a looping file cutting to black, a stuttering RTSP stream) would
# otherwise emit two status messages per cycle and bury everything else.
FAULT_NOTICE_COOLDOWN = 5.0


def _describe_source(source: str | int) -> str:
    """Human-readable source name for status messages."""
    if isinstance(source, int):
        return f"webcam index {source}"
    text = str(source)
    if text.startswith(("rtsp://", "rtmp://")):
        # Do not echo a URL that may embed credentials.
        return f"stream {text.split('@')[-1]}"
    return Path(text).name or text


class CameraThread(QThread):
    """Emits one BGR ``np.ndarray`` per frame at maximum capture rate.

    Two output channels:
        - ``raw_frame`` signal: emits every captured frame for UI display.
        - Lossy RAM buffer: single-frame ``deque(maxlen=1)`` that the
          ``InferenceThread`` samples from via ``get_latest_frame()``.

    Supports USB webcams (integer index), RTSP/RTMP streams (string URL),
    video file paths (string), and falls back to mock frames.
    """

    raw_frame = pyqtSignal(object)      # every captured frame for UI
    status_update = pyqtSignal(str)     # status / error messages

    def __init__(self, source: str | int = -1, parent=None) -> None:
        """
        Parameters
        ----------
        source : str | int
            Video source.  Interpretation:
              - int >= 0 : Webcam device index (auto-scan 0-3 if -1)
              - str starting with 'rtsp://' or 'rtmp://' : Network stream
              - str ending in '.mp4' / '.avi' / etc. : Video file
        """
        super().__init__(parent)
        self._running = True
        self._cap: cv2.VideoCapture | None = None
        self._mock_mode = False
        self._mock_tick = 0
        self._source = source
        self._is_network_stream = isinstance(source, str) and (
            source.startswith("rtsp://") or source.startswith("rtmp://")
        )
        self._is_image_source = self._looks_like_image_source(source)
        self._image_paths: list[Path] = []
        self._image_index = 0
        self._image_cursor = 0
        self._image_interval = 1.0 / max(perf_config.REPLAY_FPS, 0.1)
        self._actual_fps: float = 30.0

        # Thread-safe single-frame lossy buffer (InferenceThread reads this)
        self._buffer_lock = threading.Lock()
        self._frame_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=1)
        self._frame_buffer_index: int = 0
        self._frame_buffer_path: str = ""

        # Dead-source watchdog state.
        self._dark_streak: int = 0
        self._lit_streak: int = 0
        self._source_dead: bool = False
        # Frames that are structurally valid but carry no image.  Held back
        # from the pipeline so the classifier does not report "Absent Seat"
        # for a dead feed; the UI still gets them so the operator can see the
        # screen is black rather than frozen.
        self._dead_frame: np.ndarray | None = None
        # Fault-notice throttling, so a flapping source cannot flood the UI
        # status line or the console.
        self._last_fault_notice: float = 0.0
        self._suppressed_fault_notices: int = 0

    @staticmethod
    def _looks_like_image_source(source: str | int) -> bool:
        """True for a directory of images or a .txt/.csv image list."""
        if not isinstance(source, str) or not source.strip():
            return False
        path = Path(source)
        if path.is_dir():
            return True
        return path.suffix.lower() in IMAGE_LIST_EXTS

    # ------------------------------------------------------------------
    # Static device probing (called from Hub dropdown population)
    # ------------------------------------------------------------------

    @staticmethod
    def probe_devices() -> list[dict[str, str | int]]:
        """Probe webcam indices 0-3 and return list of available devices.

        Returns a list of dicts: [{"index": int, "name": str}, ...]
        Only includes devices that can actually open and read a frame.
        """
        devices: list[dict[str, str | int]] = []
        for idx in range(4):
            try:
                if sys.platform == "win32":
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        devices.append({
                            "index": idx,
                            "name": f"Camera {idx} ({w}x{h})",
                        })
                    cap.release()
            except Exception:
                continue
        return devices

    @staticmethod
    def probe_rtsp(url: str, timeout: int = 5) -> bool:
        """Test connectivity to an RTSP/RTMP stream URL.

        Opens the stream, attempts to read one frame, then releases.
        Returns True if the stream is reachable and returns frames.
        """
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MS, timeout * 1000)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            cap.release()
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def actual_fps(self) -> float:
        """Actual FPS reported by the capture device."""
        return self._actual_fps

    @property
    def image_count(self) -> int:
        """Number of images in the current replay source (0 for live video)."""
        return len(self._image_paths)

    @property
    def image_index(self) -> int:
        """Index of the most recently emitted replay image (1-based-ish)."""
        return self._image_index

    @property
    def is_image_replay(self) -> bool:
        """True when the source is an image folder/list rather than video."""
        return bool(self._image_paths)

    def get_latest_frame(self) -> np.ndarray | None:
        """Non-blocking read from the lossy RAM buffer.

        Called by InferenceThread to sample the most recent frame.
        Returns None if no frame is available yet.
        """
        with self._buffer_lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]
            return None

    def get_latest_frame_with_meta(
        self,
    ) -> tuple[np.ndarray | None, int, str]:
        """Return frame, replay index, and replay image path atomically."""
        with self._buffer_lock:
            if self._frame_buffer:
                return (
                    self._frame_buffer[-1],
                    self._frame_buffer_index,
                    self._frame_buffer_path,
                )
            return None, self._frame_buffer_index, self._frame_buffer_path

    def stop(self) -> None:
        """Gracefully stop the capture loop and release hardware."""
        self._running = False
        self.wait(5000)
        self.release()

    def release(self) -> None:
        """Release the OpenCV capture device (hardware lock)."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._open_source()
        self.status_update.emit("Camera thread started")

        while self._running:
            try:
                if self._image_paths:
                    frame = self._next_image()
                    if frame is None:
                        self.status_update.emit(
                            "Image replay source has no readable images — "
                            "switching to mock feed"
                        )
                        self._mock_mode = True
                        self._image_paths = []
                        continue
                    self._emit_frame(frame)
                    # Dataset replay is deliberately slower than inference:
                    # it is a frame-by-frame inspection source, not video.
                    time.sleep(self._image_interval)
                    continue

                if self._mock_mode:
                    frame = self._mock_frame()
                else:
                    ret, frame = self._cap.read()  # type: ignore[union-attr]
                    if not ret:
                        if self._is_network_stream:
                            frame = self._reconnect_stream()
                            if frame is not None:
                                self._emit_frame(frame)
                                continue
                        else:
                            # Loop video files
                            if self._cap is not None:
                                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.005)
                        continue

                    # Mirror USB webcams for natural self-view (not CCTV)
                    if not self._is_network_stream and isinstance(self._source, int):
                        frame = cv2.flip(frame, 1)

                if not self._is_network_stream and self._check_frame_alive(frame):
                    # Source is confirmed dead: do not feed the classifier.
                    continue

                self._emit_frame(frame)
                time.sleep(0.001)
            except Exception as exc:
                self.status_update.emit(f"[CameraThread] Error: {exc}")
                time.sleep(0.1)

    def _emit_frame(self, frame: np.ndarray) -> None:
        """Store frame in lossy buffer and emit signal for UI rendering."""
        with self._buffer_lock:
            self._frame_buffer.append(frame)
            self._frame_buffer_index = self._image_index
            if self._image_paths and 0 <= self._image_index < len(self._image_paths):
                self._frame_buffer_path = str(self._image_paths[self._image_index])
            else:
                self._frame_buffer_path = ""
        self.raw_frame.emit(frame)

    # ------------------------------------------------------------------
    # Image folder / image-list replay
    # ------------------------------------------------------------------

    def _next_image(self) -> np.ndarray | None:
        """Return the next readable replay image, looping at the end."""
        total = len(self._image_paths)
        if total == 0:
            return None
        for _ in range(total):
            index = self._image_cursor % total
            self._image_cursor = (index + 1) % total
            frame = cv2.imread(str(self._image_paths[index]))
            if frame is not None:
                self._image_index = index
                return frame
        return None

    def _open_image_source(self) -> bool:
        """Load a directory or .txt/.csv image list as a replay source."""
        source = Path(str(self._source))
        paths = self._collect_image_paths(source)
        if not paths:
            self.status_update.emit(
                f"No readable images found in {_describe_source(self._source)}"
            )
            return False
        self._image_paths = paths
        self._image_cursor = 0
        self._image_index = 0
        self._actual_fps = max(perf_config.REPLAY_FPS, 0.1)
        self.status_update.emit(
            f"Image replay: {len(paths):,} images from "
            f"{_describe_source(self._source)} "
            f"({self._actual_fps:g} fps, looping)"
        )
        return True

    def _collect_image_paths(self, source: Path) -> list[Path]:
        """Resolve a directory, text list or CSV manifest to image files."""
        candidates: list[Path] = []
        if source.is_dir():
            direct = sorted(
                p for p in source.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            # A parent folder such as datasets/SCUT_HEAD has no images at the
            # top level; fall back to a recursive scan in that case.
            candidates = direct or sorted(
                p for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        elif source.suffix.lower() == ".csv":
            candidates = self._read_csv_list(source)
        elif source.suffix.lower() == ".txt":
            candidates = self._read_text_list(source)

        seen: set[str] = set()
        unique: list[Path] = []
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            key = str(resolved).lower()
            if key in seen or not resolved.is_file():
                continue
            if resolved.suffix.lower() not in IMAGE_EXTS:
                continue
            seen.add(key)
            unique.append(resolved)
        return unique

    @staticmethod
    def _resolve_listed_path(value: str, list_path: Path) -> Path:
        raw = Path(value.strip().strip('"').strip("'"))
        if raw.is_absolute():
            return raw
        for base in (list_path.parent, _PROJECT_ROOT):
            candidate = base / raw
            if candidate.is_file():
                return candidate
        return _PROJECT_ROOT / raw

    def _read_text_list(self, list_path: Path) -> list[Path]:
        paths: list[Path] = []
        for line in list_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            # Tolerate "path,label" rows in hand-written lists.
            if "," in value:
                candidate = value.split(",", 1)[0].strip().strip('"').strip("'")
                if Path(candidate).suffix.lower() in IMAGE_EXTS:
                    value = candidate
            paths.append(self._resolve_listed_path(value, list_path))
        return paths

    def _read_csv_list(self, list_path: Path) -> list[Path]:
        paths: list[Path] = []
        with open(list_path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            field = ""
            for name in ("image", "path", "file", "filename"):
                if reader.fieldnames and name in reader.fieldnames:
                    field = name
                    break
            if not field and reader.fieldnames:
                field = reader.fieldnames[0]
            if not field:
                return paths
            for row in reader:
                value = str(row.get(field, "") or "").strip()
                if value:
                    paths.append(self._resolve_listed_path(value, list_path))
        return paths

    # ------------------------------------------------------------------
    # Dead-source watchdog
    # ------------------------------------------------------------------

    def _check_frame_alive(self, frame: np.ndarray) -> bool:
        """Return True when the source has been declared dead for this frame.

        A dead source still returns structurally valid frames, so the only
        signal is that the image carries no information.  The frame is still
        emitted to the UI (so an operator sees black rather than a frozen
        picture) but withheld from the inference buffer, because "Absent Seat"
        from a dead camera is a false accusation, not a finding.

        Hysteresis is deliberate: it takes ``BLACK_STREAK_FRAMES`` dark frames
        to declare the source dead, and ``BLACK_RECOVERY_FRAMES`` lit frames to
        clear it again, so a lamp flickering or a dark cut in the footage does
        not repeatedly trip the watchdog.
        """
        # A small downscale is enough to judge luma and keeps this off the
        # critical path: at 720p this samples ~9k pixels instead of 921k.
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        luma = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

        if luma < BLACK_LUMA_THRESHOLD:
            self._dark_streak += 1
            self._lit_streak = 0

            if self._dark_streak == BLACK_STREAK_FRAMES and not self._source_dead:
                self._source_dead = True
                self._dead_frame = frame
                # A looping or intermittently-failing source can flap between
                # dead and alive many times a minute.  The operator still
                # needs the first notice and the current state, so transitions
                # are logged but repeats inside the cooldown are counted.
                if time.monotonic() - self._last_fault_notice >= FAULT_NOTICE_COOLDOWN:
                    self._last_fault_notice = time.monotonic()
                    self.status_update.emit(
                        f"[CameraThread] No image from "
                        f"{_describe_source(self._source)} "
                        f"(luma {luma:.1f}) — analysis paused. The room is NOT "
                        f"being monitored; check the camera or feed."
                    )
                else:
                    self._suppressed_fault_notices += 1
                logger.warning(
                    "Source %s produced %d consecutive black frames (luma %.2f); "
                    "withholding frames from the classifier",
                    _describe_source(self._source), self._dark_streak, luma,
                )
        else:
            self._lit_streak += 1
            if self._lit_streak == BLACK_RECOVERY_FRAMES and self._source_dead:
                self._source_dead = False
                self._dark_streak = 0
                self._dead_frame = None
                if self._suppressed_fault_notices:
                    logger.info(
                        "Camera fault flapped %d times during cooldown; "
                        "reporting current state",
                        self._suppressed_fault_notices,
                    )
                    self._suppressed_fault_notices = 0
                if time.monotonic() - self._last_fault_notice >= FAULT_NOTICE_COOLDOWN:
                    self._last_fault_notice = time.monotonic()
                    self.status_update.emit(
                        f"[CameraThread] Image restored from "
                        f"{_describe_source(self._source)} — resuming analysis"
                    )
                else:
                    self._suppressed_fault_notices += 1
                logger.info(
                    "Source %s recovered after %d lit frames",
                    _describe_source(self._source), self._lit_streak,
                )
            if not self._source_dead:
                self._dark_streak = 0

        if self._source_dead:
            # Keep the UI live, drop the inference input.
            if self._dead_frame is None:
                self._dead_frame = frame
            self.raw_frame.emit(self._dead_frame)
            return True
        return False

    # ------------------------------------------------------------------
    # RTSP reconnection with exponential backoff
    # ------------------------------------------------------------------

    def _reconnect_stream(self) -> np.ndarray | None:
        """Attempt to reconnect to an RTSP/RTMP stream with backoff.

        Returns a frame on successful reconnection, or None if all
        attempts are exhausted (triggers fallback chain).
        """
        backoff = RTSP_INITIAL_BACKOFF
        for attempt in range(1, RTSP_MAX_RECONNECT_ATTEMPTS + 1):
            if not self._running:
                return None
            self.status_update.emit(
                f"Stream disconnected — reconnecting (attempt {attempt}"
                f"/{RTSP_MAX_RECONNECT_ATTEMPTS})..."
            )
            time.sleep(backoff)
            try:
                self.release()
                cap = cv2.VideoCapture(self._source)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MS, RTSP_READ_TIMEOUT_MS)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self._cap = cap
                        self._actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        self.status_update.emit(
                            f"Stream reconnected (fps={self._actual_fps:.0f})"
                        )
                        return frame
                cap.release()
            except Exception:
                pass
            backoff = min(backoff * 2, RTSP_MAX_BACKOFF)

        self.status_update.emit("Stream reconnection failed — falling back")
        return None

    def _open_source(self) -> None:
        """Open the specified source, falling back through the chain."""
        # ── Image folder / image list (dataset replay) ────────────────────
        # This is an explicit source: never silently fall through to a webcam
        # when the user asked to inspect a dataset folder.
        if self._is_image_source:
            if self._open_image_source():
                return
            self.status_update.emit(
                "Image replay source could not be opened — using mock feed"
            )
            self._mock_mode = True
            return

        # ── Explicit video file path ─────────────────────────────────────
        # Checked before the webcam scan.  When a caller names a file, that
        # file is what was asked for: the previous order scanned webcams
        # first, so selecting a video silently captured from webcam 0 instead.
        if isinstance(self._source, str) and not self._is_network_stream:
            if self._open_file(self._source, "requested video"):
                return

        # ── RTSP / RTMP network stream ───────────────────────────────────
        if self._is_network_stream:
            try:
                cap = cv2.VideoCapture(self._source)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MS, RTSP_READ_TIMEOUT_MS)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self._cap = cap
                        self._actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        self.status_update.emit(
                            f"Network stream opened: {self._source}"
                            f" (fps={self._actual_fps:.0f})"
                        )
                        return
                    cap.release()
            except Exception as exc:
                self.status_update.emit(f"Failed to open stream: {exc}")

            # Stream failed — fall through to webcam scan
            self.status_update.emit("Stream unavailable, scanning webcams...")

        # ── Specific webcam device index ─────────────────────────────────
        if isinstance(self._source, int) and self._source >= 0:
            if self._open_webcam(self._source):
                return

        # ── Webcam scan (indices 0-3) ────────────────────────────────────
        for idx in range(4):
            if self._open_webcam(idx):
                return

        # ── Video file fallback ──────────────────────────────────────────
        # Ordered usable-first.  `1_hour_exam_test.mp4` is a 60-minute clip
        # of an essentially black image (mean luma 1.6); it used to be tried
        # before `sample_exam.mp4`, so a dead webcam silently fell back to a
        # black feed and the classifier reported "Absent Seat" for a whole
        # session.  Each candidate is now validated before it is accepted.
        candidates = [
            _ASSETS_DIR / "test_feed.mp4",
            _BACKEND_DIR / "sample_exam.mp4",
            _BACKEND_DIR / "1_hour_exam_test.mp4",
        ]
        for path in candidates:
            if self._open_file(str(path), "fallback video"):
                return

        # ── Mock frame generator ─────────────────────────────────────────
        self._mock_mode = True
        self.status_update.emit("No camera/video found — generating mock frames")

    def _open_file(self, path: str | Path, label: str) -> bool:
        """Open a video file, but only if it actually contains an image.

        Returns True when the file is opened and committed as the live source.
        A file that cannot be opened, or that decodes to nothing but black
        frames, is rejected so the caller can try the next candidate.
        """
        file_path = Path(path)
        if not file_path.is_file():
            return False
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            self.status_update.emit(
                f"Could not open {file_path.name} — trying next source"
            )
            return False
        if not self._peek_has_image(cap, file_path):
            cap.release()
            return False
        self._cap = cap
        self._actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.status_update.emit(
            f"Using {label} {file_path.name} "
            f"(fps={self._actual_fps:.0f})"
        )
        return True

    def _peek_has_image(self, cap: cv2.VideoCapture, path: Path) -> bool:
        """Return True when the next frames of *cap* contain a real image.

        Checks several frames rather than one, so a file that opens on a black
        leader is not rejected outright.  The capture position is restored so
        the caller can play the file from the start.
        """
        start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        has_image = False
        try:
            for _ in range(10):
                ok, frame = cap.read()
                if not ok:
                    break
                small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
                luma = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
                if luma >= BLACK_LUMA_THRESHOLD:
                    has_image = True
                    break
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        if not has_image:
            self.status_update.emit(
                f"Skipping {path.name} — file contains no image (black video)"
            )
        return has_image

    def _open_webcam(self, idx: int) -> bool:
        """Try to open a webcam by index. Returns True on success."""
        try:
            if sys.platform == "win32":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 60)
                    self._cap = cap
                    self._actual_fps = 60.0
                    self.status_update.emit(
                        f"Webcam index {idx} opened"
                        f"{' (CAP_DSHOW)' if sys.platform == 'win32' else ''}"
                    )
                    return True
                cap.release()
        except Exception:
            pass
        return False

    @property
    def source_dead(self) -> bool:
        """True when the current source is confirmed to be producing no image."""
        return self._source_dead

    def _mock_frame(self) -> np.ndarray:
        """Generate a synthetic colour-cycling test frame."""
        self._mock_tick += 1
        frame = np.zeros((_MOCK_H, _MOCK_W, 3), dtype=np.uint8)

        # Horizontal gradient cycling over time
        t = (self._mock_tick % 256)
        for x in range(_MOCK_W):
            hue = (x + t) % 256
            frame[:, x] = [hue, 200, 40]

        # Overlay text
        cv2.putText(frame, "MOCK FEED", (180, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, "No Camera Detected", (160, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f"Frame #{self._mock_tick}", (200, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        return frame
