"""SCUT-HEAD classroom seat anchors and boundary-continuity helpers.

The pose pipeline and the head-occupancy mode answer different questions, but
they share one operational problem: a student who is partly outside the frame
must not become a different student (or an absent seat) just because their
person box was clipped for a few frames.

This module owns that continuity layer:

* :class:`HeadAnchorInitializer` runs SCUT-HEAD on the first frames of a
  session and turns stable head detections into ``Seat_01..Seat_N`` anchors.
* :class:`SeatAnchorRegistry` binds those anchors to ByteTrack ids and answers
  "which seat is this person box?".
* :class:`CandidateTracker` is a small IoU tracker used when ByteTrack ids are
  unavailable (or when a unit test stubs the detector).
* :class:`BoundaryClipGuard` holds a recently seen seat through a short
  boundary-clipped gap and de-duplicates the same detection reported twice,
  which is what suppresses false P16/P35 alerts.

Nothing here imports Qt or ultralytics at module import time.  The initializer
loads its model lazily inside the inference worker thread.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from desktop_app.core import perf_config
from desktop_app.core.torch_compile_utils import install_safe_compile
from desktop_app.core.vision_utils import (
    LetterboxBuffer,
    configure_opencv_threads,
)

logger = logging.getLogger(__name__)


def _iou(a: list[int] | list[float], b: list[int] | list[float]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _center(box: list[int] | list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _area(box: list[int] | list[float]) -> float:
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1]))


@dataclass
class SeatAnchor:
    """One classroom seat, represented by its SCUT-HEAD head box."""

    seat_id: str
    box: list[int]
    confidence: float
    hits: int = 0
    track_id: int | None = None
    last_seen: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return _center(self.box)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "box": list(self.box),
            "confidence": round(float(self.confidence), 3),
            "hits": int(self.hits),
            "track_id": self.track_id,
            "last_seen": int(self.last_seen),
        }


@dataclass
class SeatAnchorRegistry:
    """Seat anchors plus their current ByteTrack binding."""

    anchors: list[SeatAnchor] = field(default_factory=list)
    locked: bool = False

    @property
    def seat_count(self) -> int:
        return len(self.anchors)

    def as_list(self) -> list[dict[str, Any]]:
        return [anchor.as_dict() for anchor in self.anchors]

    def anchor_boxes(self) -> list[list[int]]:
        return [list(anchor.box) for anchor in self.anchors]

    def bind(
        self,
        person_boxes: list[list[int]] | list[np.ndarray],
        track_ids: list[int | None] | None = None,
        frame_index: int = 0,
    ) -> list[str | None]:
        """Return the seat id for each person box (``None`` when unbound).

        A head anchor belongs to a person box when the anchor's centre sits in
        the box's upper body region (the same 30% band candidate validation
        uses) or when the anchor box overlaps that region.  Multiple boxes may
        resolve to the same seat: that is exactly the clipped-duplicate case
        the P35 guard needs to collapse.
        """
        if not self.anchors or not person_boxes:
            return [None] * len(person_boxes)

        ids = list(track_ids or [])
        result: list[str | None] = []
        for i, raw_box in enumerate(person_boxes):
            box = [float(v) for v in raw_box[:4]]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                result.append(None)
                continue
            upper_bottom = box[1] + perf_config.HEAD_UPPER_BODY_RATIO * (
                box[3] - box[1]
            )
            upper = [box[0], box[1], box[2], upper_bottom]
            best: SeatAnchor | None = None
            best_score = 0.0
            for anchor in self.anchors:
                cx, cy = anchor.center
                inside = (
                    box[0] <= cx <= box[2]
                    and box[1] <= cy <= upper_bottom
                )
                overlap = _iou(anchor.box, upper)
                score = (1.0 + overlap) if inside else overlap
                if score > best_score:
                    best, best_score = anchor, score
            if best is None or best_score <= 0.0:
                result.append(None)
                continue
            if i < len(ids) and ids[i] is not None:
                best.track_id = int(ids[i])
            best.last_seen = frame_index
            best.hits += 1
            result.append(best.seat_id)
        return result

    def seat_for_box(self, box: list[int]) -> SeatAnchor | None:
        """Return the anchor matching one box, if any."""
        seat_ids = self.bind([box], [None], frame_index=0)
        if not seat_ids or seat_ids[0] is None:
            return None
        for anchor in self.anchors:
            if anchor.seat_id == seat_ids[0]:
                return anchor
        return None

    def observe_live_heads(
        self,
        heads: list[dict[str, Any]] | list[list[int]] | None,
        frame_index: int = 0,
        smoothing: float | None = None,
    ) -> list[str | None]:
        """Match continuous SCUT-HEAD detections to the locked seat anchors.

        Seat identity is fixed at initialisation: a live head can move the
        anchor's box so candidate validation tracks the student, but it can
        never create a new seat or renumber an existing one.  A head with no
        anchor (a late entrant or a background head) returns ``None`` -- the
        person box can still be validated by that live head, but it has no
        seat id and therefore still counts as a distinct candidate.
        """
        if not self.anchors or not heads:
            return [None] * len(heads or [])
        alpha = (
            perf_config.SEAT_ANCHOR_LIVE_SMOOTHING
            if smoothing is None else float(smoothing)
        )
        alpha = min(1.0, max(0.0, alpha))

        result: list[str | None] = []
        for head in heads:
            if isinstance(head, dict):
                box = head.get("box")
                confidence = float(head.get("confidence", 0.0))
            else:
                box = head
                confidence = 0.0
            if not box or len(box) != 4:
                result.append(None)
                continue
            head_box = [float(v) for v in box[:4]]
            if head_box[2] <= head_box[0] or head_box[3] <= head_box[1]:
                result.append(None)
                continue

            best: SeatAnchor | None = None
            best_score = 0.0
            hcx = (head_box[0] + head_box[2]) / 2.0
            hcy = (head_box[1] + head_box[3]) / 2.0
            for anchor in self.anchors:
                iou = _iou(anchor.box, head_box)
                acx, acy = anchor.center
                scale = max(
                    8.0,
                    float(anchor.box[2] - anchor.box[0]),
                    float(anchor.box[3] - anchor.box[1]),
                )
                distance = ((hcx - acx) ** 2 + (hcy - acy) ** 2) ** 0.5
                # A head is the same seat when the boxes overlap, or when its
                # centre is within one anchor-size of the anchor centre.
                if iou <= 0.0 and distance > scale:
                    continue
                score = iou + 1.0 / (1.0 + distance / scale)
                if score > best_score:
                    best, best_score = anchor, score

            if best is None:
                result.append(None)
                continue

            old = [float(v) for v in best.box]
            best.box = [
                int(round(old[i] + alpha * (head_box[i] - old[i])))
                for i in range(4)
            ]
            best.confidence = max(best.confidence, confidence)
            best.hits += 1
            best.last_seen = int(frame_index)
            result.append(best.seat_id)
        return result


class CandidateTracker:
    """Small IoU tracker used when ByteTrack ids are unavailable.

    ByteTrack is the primary source of continuity.  This fallback exists for
    two cases: ``ENABLE_BYTETRACK`` disabled, and unit tests that stub the
    detector without a real ultralytics tracker.  It deliberately does not try
    to be ByteTrack -- it only keeps an id attached across a short gap.
    """

    def __init__(
        self,
        max_age: int | None = None,
        min_iou: float | None = None,
        max_shift_ratio: float | None = None,
    ) -> None:
        self._max_age = (
            perf_config.TRACK_MAX_AGE_FRAMES if max_age is None else max_age
        )
        self._min_iou = perf_config.TRACK_MIN_IOU if min_iou is None else min_iou
        self._max_shift = (
            perf_config.TRACK_MAX_CENTER_SHIFT_RATIO
            if max_shift_ratio is None
            else max_shift_ratio
        )
        self._next_id = 1
        self._tracks: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(
        self,
        boxes: list[list[int]] | list[np.ndarray],
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Return one persistent id per box, in input order."""
        for track in self._tracks:
            track["age"] = int(track.get("age", 0)) + 1
        self._tracks = [
            t for t in self._tracks if t["age"] <= self._max_age
        ]

        frame_h = int(frame_shape[0]) if frame_shape and len(frame_shape) > 0 else 0
        frame_w = int(frame_shape[1]) if frame_shape and len(frame_shape) > 1 else 0
        tol_x = self._max_shift * frame_w if frame_w else 1e9
        tol_y = self._max_shift * frame_h if frame_h else 1e9

        pairs: list[tuple[float, int, int]] = []
        for di, raw_box in enumerate(boxes):
            box = [float(v) for v in raw_box[:4]]
            for ti, track in enumerate(self._tracks):
                iou = _iou(track["box"], box)
                tcx, tcy = _center(track["box"])
                bcx, bcy = _center(box)
                dx, dy = abs(tcx - bcx), abs(tcy - bcy)
                if iou >= self._min_iou or (dx <= tol_x and dy <= tol_y):
                    pairs.append((iou - (dx / max(1.0, frame_w or 1.0))
                                  - (dy / max(1.0, frame_h or 1.0)), di, ti))
        pairs.sort(reverse=True, key=lambda item: item[0])

        used_boxes: set[int] = set()
        used_tracks: set[int] = set()
        assigned: dict[int, int] = {}
        for _, di, ti in pairs:
            if di in used_boxes or ti in used_tracks:
                continue
            track = self._tracks[ti]
            track["box"] = [float(v) for v in boxes[di][:4]]
            track["age"] = 0
            assigned[di] = int(track["id"])
            used_boxes.add(di)
            used_tracks.add(ti)

        for di, raw_box in enumerate(boxes):
            if di in used_boxes:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks.append({
                "id": track_id,
                "box": [float(v) for v in raw_box[:4]],
                "age": 0,
            })
            assigned[di] = track_id

        return [assigned[i] for i in range(len(boxes))]


class BoundaryClipGuard:
    """Hold recently seen candidates through short boundary-clipped gaps.

    A person box that touches the frame edge is marked clipped.  If the box
    disappears entirely for a few frames -- the common failure when a student
    leans out of a tight crop or a pseudo-clip trims the frame -- the guard
    reports presence as held.  The classifier turns that into P01 rather than
    P16 Absent Seat.  It also de-duplicates overlapping boxes so a clipped
    fragment cannot manufacture P35 Multiple Persons.
    """

    def __init__(
        self,
        grace_frames: int | None = None,
        edge_margin_ratio: float | None = None,
        duplicate_iou: float | None = None,
    ) -> None:
        self._grace = (
            perf_config.CLIP_GUARD_GRACE_FRAMES
            if grace_frames is None else int(grace_frames)
        )
        self._edge_margin = (
            perf_config.CLIP_GUARD_EDGE_MARGIN_RATIO
            if edge_margin_ratio is None else float(edge_margin_ratio)
        )
        self._duplicate_iou = (
            perf_config.CLIP_GUARD_DUPLICATE_IOU
            if duplicate_iou is None else float(duplicate_iou)
        )
        self.last_presence_frame: int = -1
        self.last_clipped: bool = False
        self.last_boxes: list[list[int]] = []
        self.last_track_ids: list[int | None] = []

    def reset(self) -> None:
        self.last_presence_frame = -1
        self.last_clipped = False
        self.last_boxes = []
        self.last_track_ids = []

    def update(
        self,
        frame_shape: tuple[int, ...],
        boxes: list[list[int]] | list[np.ndarray],
        track_ids: list[int | None] | None = None,
        frame_index: int = 0,
    ) -> dict[str, Any]:
        """Record this frame's candidates and return clipping metadata."""
        frame_h = int(frame_shape[0]) if len(frame_shape) > 0 else 0
        frame_w = int(frame_shape[1]) if len(frame_shape) > 1 else 0
        mx = frame_w * self._edge_margin
        my = frame_h * self._edge_margin

        clipped: list[bool] = []
        clean_boxes: list[list[int]] = []
        for raw_box in boxes:
            if len(raw_box) < 4:
                clipped.append(False)
                continue
            x1, y1, x2, y2 = (float(v) for v in raw_box[:4])
            clean_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            clipped.append(
                x1 <= mx or y1 <= my or x2 >= frame_w - mx or y2 >= frame_h - my
            )

        if clean_boxes:
            self.last_presence_frame = int(frame_index)
            self.last_boxes = clean_boxes
            self.last_track_ids = list(track_ids or [])
        self.last_clipped = any(clipped)
        return {
            "clipped": clipped,
            "any_clipped": self.last_clipped,
            "edge_margin": (mx, my),
        }

    def presence_held(self, frame_index: int) -> bool:
        """True while a recent candidate is still inside the grace window."""
        if self.last_presence_frame < 0:
            return False
        return (int(frame_index) - self.last_presence_frame) <= self._grace

    def distinct_person_count(
        self,
        boxes: list[list[int]] | list[np.ndarray],
        track_ids: list[int | None] | None = None,
        seat_ids: list[str | None] | None = None,
    ) -> int:
        """Count distinct candidates after track/seat/duplicate collapse.

        Two boxes with the same ByteTrack id or the same seat anchor are one
        person.  Two nearly identical boxes are one detection reported twice.
        Anything else is a distinct candidate, so a genuine second student is
        still reported as P35.
        """
        ids = list(track_ids or [])
        seats = list(seat_ids or [])
        count = 0
        seen_keys: set[tuple[str, Any]] = set()
        kept_boxes: list[list[float]] = []

        for i, raw_box in enumerate(boxes):
            box = [float(v) for v in raw_box[:4]]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            if any(_iou(box, other) >= self._duplicate_iou
                   for other in kept_boxes):
                continue
            key: tuple[str, Any] | None = None
            if i < len(ids) and ids[i] is not None:
                key = ("track", int(ids[i]))
            elif i < len(seats) and seats[i]:
                key = ("seat", seats[i])
            if key is not None and key in seen_keys:
                continue
            if key is not None:
                seen_keys.add(key)
            kept_boxes.append(box)
            count += 1
        return count


class HeadAnchorInitializer:
    """Count seated heads with SCUT-HEAD and build the seat registry.

    Usage (inside the inference worker thread)::

        init = HeadAnchorInitializer()
        init.load()
        while not init.complete:
            init.observe(camera.get_latest_frame())
        registry = init.finish()

    The initializer never touches the UI thread.  If weights are unavailable
    or the camera never produces frames, it finishes with an empty, unlocked
    registry rather than holding the pipeline forever.
    """

    def __init__(
        self,
        frames: int | None = None,
        max_seconds: float | None = None,
        model_path: str | None = None,
    ) -> None:
        self._target = (
            perf_config.CLASSROOM_INIT_FRAMES if frames is None else int(frames)
        )
        self._max_seconds = (
            perf_config.CLASSROOM_INIT_MAX_SEC
            if max_seconds is None else float(max_seconds)
        )
        self._model_path = model_path
        self._model: Any = None
        self._head_ids: set[int] = {0}
        self._device: Any = None
        self._device_arg: str | None = None
        self._half: bool = False
        self._frames_seen = 0
        self._clusters: list[dict[str, Any]] = []
        self._started_at = time.monotonic()
        self._done = False
        self._error = ""
        self._registry = SeatAnchorRegistry()
        configure_opencv_threads()
        self._letterbox_buffer = LetterboxBuffer(perf_config.HEAD_IMGSZ)

    @property
    def complete(self) -> bool:
        return self._done

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def frames_target(self) -> int:
        return self._target

    @property
    def error(self) -> str:
        return self._error

    @property
    def registry(self) -> SeatAnchorRegistry:
        return self._registry

    @property
    def loaded(self) -> bool:
        """True when the SCUT-HEAD weights are resident and reusable.

        The inference worker keeps this initialiser alive after the 30-frame
        countdown so the same model instance can supply continuous anchors;
        reloading a second copy would double the VRAM/CPU footprint for no
        benefit.
        """
        return self._model is not None

    def detect(self, frame: np.ndarray | None) -> list[dict[str, Any]]:
        """Run one SCUT-HEAD pass and return head boxes.

        Public because the unified worker cycle calls it on every classified
        frame after initialisation.  It shares ``load()``, the device and the
        class filter with ``observe()``, so the 30-frame count and the live
        anchors can never disagree about what a head is.
        """
        if frame is None:
            return []
        if self._model is None and not self.load():
            return []
        try:
            from desktop_app.core.head_detector import detect_heads

            return detect_heads(
                self._model,
                frame,
                head_ids=self._head_ids,
                device_arg=self._device_arg,
                half=self._half,
                letterbox_buffer=self._letterbox_buffer,
            )
        except Exception as exc:  # pragma: no cover - detector failure
            logger.debug("SCUT-HEAD live detection failed: %s", exc)
            return []

    def load(self) -> bool:
        """Load the SCUT-HEAD model.  Safe to call more than once."""
        if self._model is not None:
            return True
        try:
            from desktop_app.core.head_detector import (
                _head_class_ids,
                resolve_head_model,
            )
            from backend.runtime import (
                optimize_cuda,
                resolve_device,
                yolo_device,
            )
            from ultralytics import YOLO

            path = resolve_head_model(self._model_path)
            if path is None or not path.is_file():
                self._error = (
                    "SCUT-HEAD weights unavailable; seat anchors not locked"
                )
                self._done = True
                logger.warning(self._error)
                return False

            self._device = resolve_device(perf_config.HEAD_DEVICE)
            self._device_arg = yolo_device(self._device)
            self._half = (
                bool(perf_config.HEAD_HALF) and self._device.type == "cuda"
            )
            optimize_cuda(self._device)
            self._model = YOLO(str(path))
            self._model(
                np.zeros((640, 640, 3), dtype=np.uint8),
                imgsz=perf_config.HEAD_IMGSZ,
                device=self._device_arg,
                quantize=16 if self._half else None,
                verbose=False,
            )
            self._head_ids = _head_class_ids(self._model)
            install_safe_compile(self._model, "head-init")
            logger.info(
                "Classroom initialiser: SCUT-HEAD %s loaded (%s%s)",
                path.name,
                self._device.type,
                ", FP16" if self._half else "",
            )
            return True
        except Exception as exc:  # pragma: no cover - model/runtime failure
            self._error = f"classroom initialiser failed: {exc}"
            self._done = True
            logger.warning(self._error)
            self._model = None
            return False

    def observe(self, frame: np.ndarray | None) -> int:
        """Run head detection on one init frame; returns heads seen."""
        if self._done or frame is None:
            return 0
        if time.monotonic() - self._started_at >= self._max_seconds:
            self.finish()
            return 0
        if self._model is None and not self.load():
            return 0

        heads = self.detect(frame)

        self._frames_seen += 1
        frame_h, frame_w = frame.shape[:2]
        frame_area = max(1.0, float(frame_h * frame_w))
        accepted: list[dict[str, Any]] = []
        for head in heads:
            box = head.get("box") or []
            if len(box) != 4:
                continue
            conf = float(head.get("confidence", 0.0))
            if conf < perf_config.SEAT_ANCHOR_MIN_CONF:
                continue
            if _area(box) / frame_area > (
                perf_config.CLASSROOM_INIT_MAX_HEAD_AREA_RATIO
            ):
                continue
            accepted.append({"box": [int(v) for v in box], "confidence": conf})
        self._cluster(accepted)

        if self._frames_seen >= self._target:
            self.finish()
        return len(accepted)

    def _cluster(self, heads: list[dict[str, Any]]) -> None:
        """Associate this frame's heads with existing init clusters."""
        used: set[int] = set()
        for head in heads:
            box = head["box"]
            best_i = -1
            best_iou = 0.0
            for i, cluster in enumerate(self._clusters):
                if i in used:
                    continue
                iou = _iou(cluster["box"], box)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0 and best_iou >= perf_config.SEAT_ANCHOR_CLUSTER_IOU:
                cluster = self._clusters[best_i]
                cluster["box"] = box
                cluster["hits"] = int(cluster.get("hits", 0)) + 1
                cluster["confidence"] = max(
                    float(cluster.get("confidence", 0.0)),
                    float(head["confidence"]),
                )
                cluster["last_frame"] = self._frames_seen
                used.add(best_i)
            else:
                self._clusters.append({
                    "box": box,
                    "hits": 1,
                    "confidence": float(head["confidence"]),
                    "last_frame": self._frames_seen,
                })
                used.add(len(self._clusters) - 1)

    def finish(self) -> SeatAnchorRegistry:
        """Freeze stable head clusters into ``Seat_01..Seat_N`` anchors."""
        if self._done and self._registry.anchors:
            return self._registry
        self._done = True
        if not self._clusters:
            self._registry = SeatAnchorRegistry()
            if not self._error:
                self._error = "no stable seated heads found during initialisation"
            return self._registry

        min_hits = max(
            perf_config.CLASSROOM_INIT_MIN_HITS,
            int(self._frames_seen * perf_config.CLASSROOM_INIT_MIN_HIT_RATIO),
        )
        stable = [
            cluster for cluster in self._clusters
            if int(cluster.get("hits", 0)) >= min_hits
            and float(cluster.get("confidence", 0.0))
            >= perf_config.SEAT_ANCHOR_MIN_CONF
        ]
        # Reading order: left-to-right, then top-to-bottom.  Seat ids are
        # stable for a session because the registry is frozen here.
        stable.sort(key=lambda c: (_center(c["box"])[0], _center(c["box"])[1]))
        anchors = [
            SeatAnchor(
                seat_id=f"Seat_{i + 1:02d}",
                box=[int(v) for v in cluster["box"]],
                confidence=float(cluster["confidence"]),
                hits=int(cluster["hits"]),
            )
            for i, cluster in enumerate(stable)
        ]
        self._registry = SeatAnchorRegistry(anchors=anchors, locked=bool(anchors))
        if anchors:
            logger.info(
                "[INIT] Classroom Seated Count: %d Students Locked",
                len(anchors),
            )
        else:
            self._error = "no stable seated heads found during initialisation"
            logger.warning(self._error)
        return self._registry

    def as_info(self) -> dict[str, Any]:
        """Serialisable init state for HUD/sidebar/logging."""
        return {
            "locked": bool(self._registry.locked),
            "seated_count": self._registry.seat_count,
            "frames_seen": int(self._frames_seen),
            "frames_target": int(self._target),
            "in_progress": not self._done,
            "error": self._error,
        }
