"""
head_detector.py — SCUT-HEAD head-occupancy mode for the PyQt6 desktop app.

SCUT-HEAD provides head bounding boxes only.  This worker therefore runs a
one-class YOLO head detector, draws head boxes, and reports a head count.  It
makes no P01-P38 behaviour claim, because the dataset has no behaviour labels.

The pose-rule pipeline remains available as a separate mode; this module is
the honest, dataset-backed alternative.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from desktop_app.core.camera_thread import CameraThread
from desktop_app.core import perf_config

try:
    from backend.runtime import (
        empty_cache,
        optimize_cuda,
        resolve_device,
        yolo_device,
    )
except ImportError:  # pragma: no cover - direct module execution
    from runtime import (  # type: ignore[no-redef]
        empty_cache,
        optimize_cuda,
        resolve_device,
        yolo_device,
    )

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCUT_HEAD_MANIFEST = _PROJECT_ROOT / "datasets" / "SCUT_HEAD" / "manifest.csv"


def resolve_head_model(
    filename: str | None = None,
    dirs: tuple[str, ...] | list[str] | None = None,
) -> Path | None:
    """Find the head-detection weights in the configured search paths."""
    raw = Path(filename or perf_config.HEAD_MODEL_FILE)
    if raw.is_absolute():
        return raw if raw.is_file() else None
    search = tuple(dirs or perf_config.HEAD_MODEL_DIRS)
    candidates = [Path(directory) / raw for directory in search]
    candidates += [_PROJECT_ROOT / directory / raw for directory in search]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _head_class_ids(model: Any) -> set[int]:
    names = getattr(model, "names", None) or {}
    if isinstance(names, (list, tuple)):
        names = {i: name for i, name in enumerate(names)}
    ids = {
        int(cid) for cid, name in names.items()
        if "head" in str(name).lower()
    }
    return ids or {0}


def render_head_hud(
    frame: np.ndarray,
    result: dict[str, Any],
    capture_fps: float,
    inference_fps: float,
) -> np.ndarray:
    """Draw head boxes and an occupancy panel.  Never mutates *frame*."""
    overlay = frame.copy()

    # SCUT-HEAD annotations first (thin magenta), so detector boxes drawn
    # afterwards remain readable on top.  Live sources have no GT.
    gt_heads: list[list[float]] = result.get("ground_truth_heads") or []
    for box in gt_heads:
        if len(box) != 4:
            continue
        gx1, gy1, gx2, gy2 = (int(v) for v in box)
        cv2.rectangle(
            overlay, (gx1, gy1), (gx2, gy2), (255, 0, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            overlay, "GT", (gx1 + 2, max(10, gy1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 255), 1, cv2.LINE_AA,
        )

    heads: list[dict[str, Any]] = result.get("heads") or []
    for head in heads:
        box = head.get("box") or []
        if len(box) != 4:
            continue
        confidence = float(head.get("confidence", 0.0))
        if confidence >= 0.50:
            colour = (0, 220, 90)
        elif confidence >= 0.30:
            colour = (0, 190, 255)
        else:
            colour = (60, 160, 255)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        label = f"head {confidence:.2f}"
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        ty = max(size[1] + 4, y1 - 4)
        cv2.rectangle(
            overlay, (x1, ty - size[1] - 4), (x1 + size[0] + 4, ty + 2),
            colour, -1,
        )
        cv2.putText(
            overlay, label, (x1 + 2, ty - 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (10, 10, 10), 1, cv2.LINE_AA,
        )

    count = int(result.get("head_count", len(heads)))
    gt_count = result.get("ground_truth_count")
    dead = bool(result.get("source_dead"))
    error = str(result.get("error") or "")
    image_count = int(result.get("image_count", 0) or 0)
    image_index = int(result.get("image_index", 0) or 0)

    head_line = f"Heads: {count}"
    if gt_count is not None:
        head_line += f"  |  SCUT-HEAD GT: {int(gt_count)}"
    lines = [
        ("SCUT-HEAD HEAD DETECTION", (0, 220, 255)),
        (head_line, (0, 0, 255) if dead else (0, 220, 90)),
        (
            f"Capture {capture_fps:5.1f} FPS | Inference {inference_fps:5.1f} FPS",
            (220, 220, 220),
        ),
    ]
    if gt_count is not None:
        lines.append(("Magenta boxes = SCUT-HEAD labels", (255, 0, 255)))
    if image_count:
        lines.append((f"Image {image_index + 1}/{image_count}", (180, 220, 255)))
    if error:
        lines.append((error[:70], (0, 0, 255)))
    if dead:
        lines.append(("NO SIGNAL - occupancy unknown", (0, 0, 255)))

    panel_w = max(
        300,
        max(
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            for text, _ in lines
        ) + 24,
    )
    panel_h = 14 + 23 * len(lines)
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (80, 80, 80), 1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(
            overlay, text, (20, 34 + i * 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA,
        )
    return overlay


class HeadCountWorker(QThread):
    """Runs head detection on the camera buffer and emits occupancy results."""

    frame_processed = pyqtSignal(object)
    result_ready = pyqtSignal(dict)
    fps_updated = pyqtSignal(float)
    capture_fps_updated = pyqtSignal(float)
    status_update = pyqtSignal(str)

    def __init__(
        self,
        camera: CameraThread,
        model_path: Path | str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._running = True
        self._camera = camera
        self._model_path = Path(model_path) if model_path else None
        self._model: Any = None
        self._device = resolve_device(perf_config.HEAD_DEVICE)
        self._device_arg = yolo_device(self._device)
        self._half = bool(perf_config.HEAD_HALF) and self._device.type == "cuda"
        self._head_ids: set[int] = {0}
        self._error = ""
        self._last_frame: np.ndarray | None = None
        self._gt_by_image: dict[str, list[list[float]]] | None = None
        self._gt_loaded = False
        self._inference_fps = _FpsCounter()
        self._capture_fps = _FpsCounter()

    def stop(self) -> None:
        self._running = False
        # Model loading can take tens of seconds on a cold cache (Ultralytics
        # import + warm-up); give it room so the QThread is never destroyed
        # while still running.
        self.wait(30000)
        # Return the model's cached blocks to the driver once this worker is
        # done; otherwise a mode switch leaves VRAM reserved until process exit.
        self._model = None
        empty_cache(getattr(self, "_device", None))

    def tick_capture_fps(self) -> None:
        self._capture_fps.tick()

    def run(self) -> None:
        self._load_model()
        min_interval = (
            1.0 / perf_config.HEAD_TARGET_FPS
            if perf_config.HEAD_TARGET_FPS > 0 else 0.0
        )
        next_due = time.monotonic()
        logger.info("Head-occupancy worker started")

        while self._running:
            frame, frame_index, frame_path = (
                self._camera.get_latest_frame_with_meta()
            )
            if frame is None:
                time.sleep(0.01)
                continue
            # The lossy frame buffer keeps returning the last frame until the
            # capture thread replaces it.  Without this guard, dataset replay
            # would re-run detection several times on the same still image.
            if frame is self._last_frame:
                time.sleep(0.01)
                continue
            if min_interval > 0.0:
                remaining = next_due - time.monotonic()
                if remaining > 0.0:
                    time.sleep(min(remaining, 0.05))
                    continue
            self._last_frame = frame
            next_due = time.monotonic() + min_interval
            ground_truth = self._ground_truth_for(frame_path)

            if self._camera.source_dead:
                self._emit_result(
                    frame, [], source_dead=True, image_index=frame_index
                )
                continue

            if self._model is None:
                self._emit_result(
                    frame, [], error=self._error, image_index=frame_index,
                    ground_truth=ground_truth,
                )
                continue

            try:
                heads = self._detect(frame)
                self._emit_result(
                    frame, heads, image_index=frame_index,
                    ground_truth=ground_truth,
                )
            except Exception as exc:
                logger.error("Head detection failed: %s", exc, exc_info=True)
                self._emit_result(
                    frame, [], error=f"head detection error: {exc}",
                    image_index=frame_index, ground_truth=ground_truth,
                )

        logger.info("Head-occupancy worker stopped")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        path = self._model_path or resolve_head_model()
        if path is None or not Path(path).is_file():
            self._error = "head model not found: run tools/train_head_detector.py"
            self.status_update.emit(self._error)
            logger.warning(self._error)
            return
        try:
            from ultralytics import YOLO

            optimize_cuda(self._device)
            self._model = YOLO(str(path))
            warm = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(
                warm,
                imgsz=perf_config.HEAD_IMGSZ,
                device=self._device_arg,
                quantize=16 if self._half else None,
                verbose=False,
            )
            self._head_ids = _head_class_ids(self._model)
            message = (
                f"Head detector loaded: {Path(path).name} "
                f"({self._device.type}{', FP16' if self._half else ''})"
            )
            self.status_update.emit(message)
            logger.info("%s (classes=%s)", message, sorted(self._head_ids))
        except Exception as exc:
            self._error = f"failed to load head model: {exc}"
            self.status_update.emit(self._error)
            logger.warning(self._error)
            self._model = None

    # ------------------------------------------------------------------
    # SCUT-HEAD ground truth (image replay only)
    # ------------------------------------------------------------------

    def _ground_truth_for(
        self, image_path: str
    ) -> list[list[float]] | None:
        """Return annotated head boxes for a replay image, if available."""
        if not image_path:
            return None
        mapping = self._load_ground_truth()
        if not mapping:
            return None
        try:
            key = str(Path(image_path).resolve()).lower()
        except OSError:
            return None
        return mapping.get(key)

    def _load_ground_truth(self) -> dict[str, list[list[float]]]:
        """Load the SCUT-HEAD manifest once, keyed by resolved image path."""
        if self._gt_loaded:
            return self._gt_by_image or {}
        self._gt_loaded = True
        mapping: dict[str, list[list[float]]] = {}
        if not _SCUT_HEAD_MANIFEST.is_file():
            self._gt_by_image = mapping
            return mapping
        try:
            with open(
                _SCUT_HEAD_MANIFEST, newline="", encoding="utf-8"
            ) as handle:
                for row in csv.DictReader(handle):
                    image = str(row.get("image") or "").strip()
                    if not image:
                        continue
                    raw = Path(image)
                    if not raw.is_absolute():
                        raw = _PROJECT_ROOT / raw
                    try:
                        key = str(raw.resolve()).lower()
                    except OSError:
                        continue
                    try:
                        boxes = json.loads(row.get("boxes") or "[]")
                    except json.JSONDecodeError:
                        boxes = []
                    mapping[key] = boxes
        except OSError as exc:
            logger.warning(
                "Could not read SCUT-HEAD manifest for GT overlay: %s", exc
            )
            mapping = {}
        self._gt_by_image = mapping
        if mapping:
            logger.info(
                "SCUT-HEAD ground truth available for %d replay images",
                len(mapping),
            )
        return mapping

    def _detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        with torch.inference_mode():
            results = self._model(
                frame,
                imgsz=perf_config.HEAD_IMGSZ,
                conf=perf_config.HEAD_CONF,
                iou=perf_config.HEAD_NMS_IOU,
                max_det=perf_config.HEAD_MAX_DET,
                classes=sorted(self._head_ids),
                device=self._device_arg,
                quantize=16 if self._half else None,
                verbose=False,
            )
        result = results[0]
        heads: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(np.int32)
            # The class filter above already runs in NMS; keep a vectorised
            # guard so a model with differently named classes cannot leak a
            # non-head box into the occupancy count.
            keep = np.isin(classes, list(self._head_ids))
            if not keep.all():
                boxes, scores = boxes[keep], scores[keep]
            heads = [
                {
                    "box": [int(v) for v in box],
                    "confidence": round(float(score), 3),
                }
                for box, score in zip(boxes, scores)
            ]
        heads.sort(key=lambda item: item["confidence"], reverse=True)
        return heads

    # ------------------------------------------------------------------
    # Result / HUD
    # ------------------------------------------------------------------

    def _emit_result(
        self,
        frame: np.ndarray,
        heads: list[dict[str, Any]],
        source_dead: bool = False,
        error: str = "",
        image_index: int | None = None,
        ground_truth: list[list[float]] | None = None,
    ) -> None:
        count = len(heads)
        mean_conf = (
            sum(float(h.get("confidence", 0.0)) for h in heads) / count
            if count else 0.0
        )
        if image_index is None:
            image_index = getattr(self._camera, "image_index", 0)
        gt_heads = [
            [int(v) for v in box]
            for box in (ground_truth or [])
            if len(box) == 4
        ]
        result = {
            "mode": "head",
            "pose_id": "NO_SIGNAL" if source_dead else "HEAD",
            "pose_name": (
                "NO SIGNAL" if source_dead
                else f"{count} head{'s' if count != 1 else ''} detected"
            ),
            "confidence": mean_conf,
            "head_count": count,
            "heads": heads,
            "ground_truth_heads": gt_heads,
            "ground_truth_count": (
                len(gt_heads) if ground_truth is not None else None
            ),
            "bounding_boxes": [h["box"] for h in heads],
            "keypoints": [],
            "landmarks": [],
            "gaze_endpoint": None,
            "head_pose": {},
            "spatial_features": {},
            "temporal_metrics": {},
            "objects_detected": [],
            "recording": False,
            "source_dead": source_dead,
            "error": error,
            "image_index": int(image_index),
            "image_count": getattr(self._camera, "image_count", 0),
            "neutral_state": None,
            "neutral_label": "",
            "neutral_suppressed": False,
            "phone_proximity": None,
        }
        inf_fps = self._inference_fps.tick()
        cap_fps = self._capture_fps.current_fps()
        self.fps_updated.emit(inf_fps)
        self.capture_fps_updated.emit(cap_fps)
        self.result_ready.emit(result)
        self.frame_processed.emit(
            render_head_hud(frame, result, cap_fps, inf_fps)
        )


class _FpsCounter:
    """Simple rolling-average FPS counter (same policy as InferenceWorker)."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> float:
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        return self.current_fps()

    def current_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
