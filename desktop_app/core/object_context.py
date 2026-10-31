"""Desk-object context: one implementation shared by desktop and backend.

Both deployments answer the same question -- "is the student using an object,
and is it theirs?" -- and both previously answered it by testing whether any
phone/book/laptop existed somewhere near a detected person.  On the desktop
that produced a 50%-false-positive posture pipeline; on the backend it raised
a HIGH-severity alert for a proctor's laptop on the far desk.  The decision
logic therefore lives here exactly once:

  * ``desk_region``       where to look (the lower frame, not the torso box)
  * ``classify_relation`` is it in a hand / reachable / away, in shoulder-width
                          units so it holds at any camera resolution
  * ``relation_to_nearest_person``  whose hands -- nearest, not detection order
  * ``merge_tracks``      persistence across throttled passes, two-axis,
                          one-to-one association
  * ``escalate``          which labels become which verdict (configured, so a
                          custom model's classes are decided, not guessed)
  * ``wanted``            whether running a second model pass could change the
                          verdict at all
  * ``load`` / ``adopt_custom_classes``  model selection, custom-model
                          class-map adoption

State (tracks, throttle counter, per-frame body context) lives on the
:class:`ObjectContext` instance; each application owns one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from desktop_app.core import perf_config

try:
    from backend.runtime import optimize_cuda, resolve_device, yolo_device
except ImportError:  # pragma: no cover - direct module execution
    from runtime import (  # type: ignore[no-redef]
        optimize_cuda,
        resolve_device,
        yolo_device,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class vocabulary
# ---------------------------------------------------------------------------

# COCO-80 ids that carry meaning for this task.  Its label set has no paper,
# pen, notebook or desk, so "nothing detected" from these weights means
# "outside COCO's vocabulary", never "the desk was clear" -- which is why a
# custom desk model is preferred whenever one exists (see ``load``).
OBJ_CLASS_MAP: dict[int, str] = {
    63: "laptop",
    67: "cell_phone",
    73: "book",
}

# Verdicts about the system rather than about a student.  A black or starting
# frame is not evidence about anyone, and running object detection on one
# spends a full model pass on noise.
SYSTEM_STATES = frozenset({"NO_SIGNAL", "INITIALIZING"})

# Shared skeleton layout.  Both deployments emit skeletons in this order: the
# desktop natively (MediaPipe Pose), the backend by converting its COCO
# keypoints into the same layout.  Defining it here is what lets the spatial
# maths below be written once for both.
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16



def presence_poses() -> frozenset[str]:
    """Poses that describe who is present rather than what they are doing.

    Derived from perf_config so the gate and the object context cannot drift:
    "Absent Seat" or "Multiple People" is not made any truer by a phone on
    the desk, and both are immediate verdicts no object could change.
    """
    return perf_config.IMMEDIATE_POSES


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def shoulder_width_of(
    landmarks: list[tuple[float, float, float]] | None,
) -> float:
    """Shoulder width as the scale reference, or 0.0 when unusable."""
    if not landmarks or len(landmarks) <= MP_RIGHT_SHOULDER:
        return 0.0
    if min(landmarks[MP_LEFT_SHOULDER][2],
          landmarks[MP_RIGHT_SHOULDER][2]) <= 0.5:
        return 0.0
    return abs(
        landmarks[MP_LEFT_SHOULDER][0] - landmarks[MP_RIGHT_SHOULDER][0]
    )


def box_iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_object_model(
    filename: str, dirs: tuple[str, ...] | list[str]
) -> tuple[Path, bool] | None:
    """Find *filename* in *dirs*; returns ``(path, is_custom)``.

    ``is_custom`` keys off the stem so a model dropped into the search path
    as ``desk_objects.pt`` is adopted as a custom desk model -- its own class
    names become the label set -- without any code change.
    """
    if not filename:
        return None
    raw = Path(filename)
    candidates = (
        [raw] if raw.is_absolute()
        else [Path(d) / raw for d in dirs]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, raw.stem.startswith("desk_")
    return None


def adopt_custom_classes(model: Any) -> dict[int, str]:
    """Use a custom model's own class names as the label set.

    Everything it can detect is reportable, because the whole point of a desk
    model is that its classes were chosen for this task.

    A class named "person" is excluded: a person box carries no evidence
    about the desk.  The exclusion is by NAME, not by id -- the stock COCO
    model happens to put person at id 0, but models/desk/desk_classes.py
    puts ``exam_paper`` there, and dropping id 0 unconditionally would have
    made the one class the whole custom model exists for undetectable while
    every log still looked healthy.
    """
    names = getattr(model, "names", None) or {}
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    class_map: dict[int, str] = {}
    for cid, name in names.items():
        label = str(name).strip().lower().replace(" ", "_")
        if label in ("person", "people"):
            continue
        class_map[int(cid)] = label
    if not class_map:
        logger.warning(
            "Custom desk model exposed no usable classes; the label set is empty"
        )
    return class_map


# ---------------------------------------------------------------------------
# The per-application object context
# ---------------------------------------------------------------------------

class ObjectContext:
    """State + decisions for desk-object detection in one application.

    One instance is owned by the desktop classifier and one by the CCTV
    backend detector, so a change to the spatial rules cannot land in one
    deployment and not the other.
    """

    def __init__(self) -> None:
        # Model selection.
        self.model: Any = None
        self.class_map: dict[int, str] = dict(OBJ_CLASS_MAP)
        self.is_custom: bool = False
        self.model_path: str = ""
        self.device = resolve_device(perf_config.INFERENCE_DEVICE)
        self.device_arg: str = yolo_device(self.device)
        self.half: bool = (
            bool(perf_config.INFERENCE_HALF) and self.device.type == "cuda"
        )

        # Persistence / throttle.
        self.tracks: dict[str, list[dict[str, Any]]] = {}
        self.detect_counter: int = 0

        # Per-frame body context (see set_frame_context).
        self.frames_seen: int = 0
        self.frame_shape: tuple[int, ...] | None = None
        self.frame_width: int = 0
        self.persons: list[dict[str, Any]] = []
        self.person_bbox: list[int] | None = None
        self.shoulder_w: float = 0.0
        self.landmarks: list[tuple[float, float, float]] | None = None

    # -- model ------------------------------------------------------------

    def load(
        self,
        custom_filename: str,
        fallback_filename: str,
        dirs: tuple[str, ...] | list[str],
    ) -> Any | None:
        """Load a custom desk model if present, else the fallback weights.

        Logs loudly when falling back to COCO-80, because the difference
        matters operationally: paper, pen, notebook and desk cannot be
        detected by the fallback, so a clear result from it must never be
        read as a clear desk.
        """
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("ultralytics unavailable for object detection: %s", exc)
            return None

        custom = resolve_object_model(custom_filename, dirs)
        stock = resolve_object_model(fallback_filename, dirs)
        if custom is not None:
            path, is_custom = custom
        elif stock is not None:
            path, is_custom = stock
        else:
            logger.info(
                "No desk-object weights found (looked for %s then %s) -- "
                "object detection disabled",
                custom_filename, fallback_filename,
            )
            return None

        try:
            optimize_cuda(self.device)
            model = YOLO(str(path))
            _ = model(
                np.zeros((640, 640, 3), dtype=np.uint8),
                imgsz=perf_config.OBJ_IMGSZ,
                device=self.device_arg,
                quantize=16 if self.half else None,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("Failed to load object detector %s: %s", path, exc)
            return None

        self.model = model
        self.model_path = str(path)
        if is_custom:
            self.class_map = adopt_custom_classes(model)
            self.is_custom = True
        else:
            self.class_map = dict(OBJ_CLASS_MAP)
            self.is_custom = False
            logger.warning(
                "Using COCO-80: paper, pen, pencil, notebook, exam sheet and "
                "desk are outside its label set and cannot be detected. "
                "Train %s to cover them.",
                custom_filename,
            )
        logger.info(
            "Object detector loaded from %s (%s), classes: %s",
            path, "custom desk model" if is_custom else "COCO-80 fallback",
            ", ".join(sorted(set(self.class_map.values()))),
        )
        return model

    # -- per-frame context ------------------------------------------------

    def set_frame_context(
        self,
        frame_shape: tuple[int, ...],
        persons: list[dict[str, Any]],
        person_bbox: list[int] | None,
        landmarks: list[tuple[float, float, float]] | None,
        shoulder_w: float,
    ) -> None:
        """Record this frame's body context for relation and region maths."""
        self.frame_shape = frame_shape
        self.frame_width = int(frame_shape[1]) if len(frame_shape) > 1 else 0
        self.persons = persons
        self.person_bbox = person_bbox
        self.landmarks = landmarks
        self.shoulder_w = shoulder_w

    @staticmethod
    def capture_persons(
        all_landmarks: list[list[tuple[float, float, float]]],
        boxes: list[Any],
    ) -> list[dict[str, Any]]:
        """Per-person scale context used to attribute a desk object.

        An object is evidence about whoever is closest to it, so each person
        needs their own shoulder width: normalising against the WRONG
        person's shoulders misjudges reach, and defaulting to person 0
        attributes everything to the first detection.
        """
        people: list[dict[str, Any]] = []
        for i, lm in enumerate(all_landmarks):
            box = list(boxes[i]) if i < len(boxes) else None
            people.append({
                "landmarks": lm,
                "shoulder_w": shoulder_width_of(lm),
                "bbox": box,
            })
        return people

    # -- verdict gate -----------------------------------------------------

    @staticmethod
    def wanted(
        raw_pose_id: str,
        warming_up: bool,
        model_loaded: bool,
        has_landmarks: bool,
    ) -> bool:
        """True when desk-object detection could change the verdict.

        Object detection only runs on the *anomaly* path, so the
        overwhelmingly common "student is sitting normally" frame costs one
        YOLO pass rather than two:

        - ``P01``: no anomaly, so an object cannot change the verdict.
          Excluding it is what keeps the cost of a normal frame unchanged;
        - system states: a black or starting frame is noise, and no student
          exists to relate an object to;
        - presence poses: who is present is not changed by what is on the
          desk, and those verdicts are immediate;
        - warmup: models and baselines are still forming.
        """
        if warming_up or not model_loaded:
            return False
        if raw_pose_id == "P01":
            return False
        if raw_pose_id in SYSTEM_STATES or raw_pose_id in presence_poses():
            return False
        return has_landmarks

    # -- spatial relations -------------------------------------------------

    def desk_region(
        self, frame_shape: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
        """Region of the frame searched for desk objects (x1, y1, x2, y2).

        The previous region was the person's own box from 40% down: about 20%
        of the frame width, excluding the 27% to the left and the 33% to the
        right of the torso.  Anything beside the student -- a phone next to
        the keyboard, an object on a shared desk -- was unreachable by
        construction rather than missed by the model.

        Now: full frame width from a fixed height to the bottom edge, which
        is where a desk in front of a seated student appears in a typical
        overhead-mount CCTV view, narrowing to a band around the subject only
        when a very wide scene would otherwise waste detection on empty floor.
        """
        frame_h, frame_w = frame_shape[:2]
        y1 = int(frame_h * perf_config.OBJ_REGION_TOP_RATIO)
        x1, x2 = 0, frame_w

        box = self.person_bbox
        if box is not None:
            shoulder_w = self.shoulder_w
            if shoulder_w and shoulder_w > 0:
                bx1, _, bx2, _ = box
                centre = (bx1 + bx2) / 2.0
                half = max(
                    perf_config.OBJ_LATERAL_SHOULDER_WIDTHS * shoulder_w,
                    (bx2 - bx1) / 2.0,
                )
                narrowed_x1 = int(max(0.0, centre - half))
                narrowed_x2 = int(min(float(frame_w), centre + half))
                # Only adopt the narrower band if it keeps most of the frame.
                if (narrowed_x2 - narrowed_x1) > 0.75 * frame_w:
                    x1, x2 = narrowed_x1, narrowed_x2

        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(y1 + 1, frame_h)
        return x1, y1, x2, y2

    def classify_relation(
        self,
        obj_box: list[int],
        landmarks: list[tuple[float, float, float]] | None,
        shoulder_w: float | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Classify one object's position relative to one person's body.

        Returns ``relation`` (``in_hand`` / ``on_desk`` / ``reachable`` /
        ``lap`` / ``away``), the booleans the escalation gate reads, and the
        normalised ``distance`` from the nearest wrist.

        Distances are divided by shoulder width so the same threshold holds
        for a 720p webcam and a 4K ceiling mount.  An object merely existing
        in the room says nothing about the student; whether their hands are
        on it is the evidence.
        """
        shape = frame_shape or self.frame_shape
        if shape is None:
            return {
                "relation": "away", "in_hand": False,
                "reachable": False, "distance": float("inf"),
            }
        frame_h, frame_w = shape[:2]
        cx = (obj_box[0] + obj_box[2]) / 2.0
        cy = (obj_box[1] + obj_box[3]) / 2.0

        result: dict[str, Any] = {
            "relation": "away",
            "in_hand": False,
            "reachable": False,
            "distance": float("inf"),
        }

        if not landmarks or len(landmarks) <= MP_RIGHT_WRIST:
            return result

        if shoulder_w is None:
            shoulder_w = self.shoulder_w or 0.0
        if shoulder_w < perf_config.MIN_SCALE_PX:
            return result

        best: tuple[float, str] | None = None
        for idx, side in (
            (MP_LEFT_WRIST, "left"),
            (MP_RIGHT_WRIST, "right"),
        ):
            wx, wy, vis = landmarks[idx]
            if vis <= 0.5:
                continue
            dist = ((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5 / shoulder_w
            if best is None or dist < best[0]:
                best = (dist, side)

        if best is None:
            return result

        dist, side = best
        result["distance"] = round(dist, 3)
        result["wrist"] = side

        if dist <= perf_config.OBJ_IN_HAND_SHOULDER_WIDTHS:
            result["relation"] = "in_hand"
            result["in_hand"] = True
            result["reachable"] = True
            return result

        if dist <= perf_config.OBJ_REACHABLE_SHOULDER_WIDTHS:
            # Reachable but not held: still in front of the student, e.g. a
            # phone lying on the desk within arm's length.
            result["relation"] = "on_desk" if cy > frame_h * 0.5 else "reachable"
            result["reachable"] = True
            return result

        # Beyond arm's reach: attribute by height, since a lap-held object
        # sits above the hip line and a floor-level object does not.
        if cy > frame_h * 0.75:
            result["relation"] = "lap"
        return result

    def relation_to_nearest_person(self, obj_box: list[int]) -> dict[str, Any]:
        """Spatial relation of one object against the closest person.

        Falls back to the singular primary-person context when nothing
        better is available, and returns ``away`` when nobody has a usable
        scale.  "Nearest" is decided by normalised wrist distance -- the same
        quantity escalation reasons about -- so the person selected here is
        the one the verdict will be about.
        """
        people = self.persons or []
        if not people:
            return self.classify_relation(obj_box, self.landmarks)

        best: dict[str, Any] | None = None
        best_index = 0
        for i, person in enumerate(people):
            rel = self.classify_relation(
                obj_box, person["landmarks"], person["shoulder_w"]
            )
            if best is None or rel["distance"] < best["distance"]:
                best, best_index = rel, i
                # An object literally in someone's hand stops the search.
                if rel["in_hand"]:
                    break
        if best is None:
            return {
                "relation": "away", "in_hand": False,
                "reachable": False, "distance": float("inf"),
            }
        best["person_index"] = best_index
        return best

    # -- persistence -------------------------------------------------------

    def merge_tracks(self, fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Carry detections forward across throttled detection passes.

        Object detection runs on 1 frame in N, so a naive read of the result
        would make objects appear and vanish eight times a second.

        Association is two-phase and one-to-one:

        1. pairs with box overlap (IoU) match first, best overlap first,
           re-acquiring a static object on the next throttled pass;
        2. remaining pairs match on centre distance within the configured
           tolerance in BOTH axes.

        Matching on the horizontal centre alone -- the previous rule --
        merged two same-class objects stacked vertically into one track and
        let two fresh detections claim one track, silently dropping the
        second object.  One-to-one assignment prevents both.
        """
        now = self.frames_seen
        tracks = self.tracks

        # Drop tracks that have aged out.
        for label in list(tracks):
            tracks[label] = [t for t in tracks[label] if now - t["last_seen"] <= (
                perf_config.OBJ_PERSIST_FRAMES
            )]
            if not tracks[label]:
                del tracks[label]

        if not fresh:
            return self._flatten_tracks()

        frame_w = max(1, self.frame_width or 1)
        frame_h = max(1, self.frame_shape[0] if self.frame_shape else 1)
        tol_x = perf_config.OBJ_TRACK_MAX_SHIFT_X * frame_w
        tol_y = perf_config.OBJ_TRACK_MAX_SHIFT_Y * frame_h
        min_iou = perf_config.OBJ_TRACK_MIN_IOU

        # Candidate pairs: same label, matching on overlap or on both axes.
        pairs: list[tuple[float, int, int]] = []  # (score, det_idx, track_idx)
        for di, det in enumerate(fresh):
            dbox = det["box"]
            dcx, dcy = (dbox[0] + dbox[2]) / 2.0, (dbox[1] + dbox[3]) / 2.0
            for ti, track in enumerate(tracks.get(det["label"], [])):
                tbox = track["box"]
                iou = box_iou(tbox, dbox)
                dx = abs((tbox[0] + tbox[2]) / 2.0 - dcx)
                dy = abs((tbox[1] + tbox[3]) / 2.0 - dcy)
                if iou >= min_iou:
                    # Higher IoU is better; normalised distance breaks ties so
                    # the nearest centre wins when boxes barely touch.
                    pairs.append((10.0 + iou - (dx / frame_w + dy / frame_h),
                                  di, ti))
                elif dx <= tol_x and dy <= tol_y:
                    pairs.append((1.0 - (dx / frame_w + dy / frame_h), di, ti))

        # Greedy one-to-one assignment, best candidate first.
        pairs.sort(reverse=True, key=lambda p: p[0])
        used_dets: set[int] = set()
        used_tracks: set[tuple[str, int]] = set()
        for _, di, ti in pairs:
            det = fresh[di]
            label = det["label"]
            if di in used_dets or (label, ti) in used_tracks:
                continue
            track = tracks[label][ti]
            track.update({
                "box": det["box"],
                "confidence": max(track["confidence"], det["confidence"]),
                "last_seen": now,
                "relation": det.get("relation", "away"),
                "in_hand": det.get("in_hand", False),
                "reachable": det.get("reachable", False),
                "persistent": True,
            })
            used_dets.add(di)
            used_tracks.add((label, ti))

        # Unmatched detections become new tracks.
        for di, det in enumerate(fresh):
            if di in used_dets:
                continue
            tracks.setdefault(det["label"], []).append({
                "label": det["label"],
                "confidence": det["confidence"],
                "box": det["box"],
                "relation": det.get("relation", "away"),
                "in_hand": det.get("in_hand", False),
                "reachable": det.get("reachable", False),
                "last_seen": now,
                "persistent": False,
            })

        return self._flatten_tracks()

    def _flatten_tracks(self) -> list[dict[str, Any]]:
        """Project internal tracks to the outward-facing detection shape."""
        now = self.frames_seen
        out: list[dict[str, Any]] = []
        for label_tracks in self.tracks.values():
            for track in label_tracks:
                out.append({
                    "label": track["label"],
                    "confidence": round(track["confidence"], 3),
                    "box": track["box"],
                    "relation": track["relation"],
                    "in_hand": track["in_hand"],
                    "reachable": track["reachable"],
                    # Frames the object has been absent for, so a caller can
                    # tell a fresh detection from a remembered one.
                    "age_frames": now - track["last_seen"],
                    "persistent": track["persistent"],
                })
        return out

    # -- detection ---------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        force: bool = False,
        imgsz: int | None = None,
        conf: float | None = None,
    ) -> list[dict[str, Any]]:
        """Detect desk objects and describe their position relative to people.

        Throttled to 1 frame in ``OBJ_DETECT_EVERY_N_FRAMES`` because this is
        a second full model pass; detections persist between passes, so
        throttling costs latency rather than recall.  ``force=True`` runs on
        this frame regardless (used by the offline evaluation harness).

        One pass over the whole desk region, rather than one per person: a
        phone on a shared desk has no single owner, and N passes over
        overlapping crops both cost time and duplicate the same object.
        """
        if self.model is None:
            return []

        if not force:
            self.detect_counter += 1
            if self.detect_counter % perf_config.OBJ_DETECT_EVERY_N_FRAMES != 1:
                return self.merge_tracks([])

        if self.frame_shape is None:
            return self.merge_tracks([])

        region = self.desk_region(self.frame_shape)
        rx1, ry1, rx2, ry2 = region
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return self.merge_tracks([])

        fresh: list[dict[str, Any]] = []
        try:
            results = self.model(
                crop,
                imgsz=imgsz if imgsz is not None else perf_config.OBJ_IMGSZ,
                conf=conf if conf is not None else perf_config.OBJ_CONF_THRESHOLD,
                device=self.device_arg,
                quantize=16 if self.half else None,
                verbose=False,
            )
            r = results[0]
            if r.boxes is not None:
                for det_box, det_cls, det_conf in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int),
                    r.boxes.conf.cpu().numpy(),
                ):
                    if det_cls not in self.class_map:
                        continue
                    # Offset crop coordinates back to full frame.
                    full_box = [
                        int(det_box[0]) + rx1,
                        int(det_box[1]) + ry1,
                        int(det_box[2]) + rx1,
                        int(det_box[3]) + ry1,
                    ]
                    det = {
                        "label": self.class_map[det_cls],
                        "confidence": float(det_conf),
                        "box": full_box,
                    }
                    # Attribute to the nearest person, not to whoever was
                    # detected first: on a shared desk the nearest student is
                    # the one whose hands could plausibly be on the object.
                    det.update(self.relation_to_nearest_person(full_box))
                    fresh.append(det)
        except Exception as exc:
            logger.debug("Object detection failed: %s", exc)

        return self.merge_tracks(fresh)

    # -- escalation --------------------------------------------------------

    @staticmethod
    def escalation(
        objects: list[dict[str, Any]]
    ) -> tuple[str, float] | None:
        """``(verdict, confidence_floor)`` if *objects* warrant a finding.

        Returns None when nothing in the list clears the spatial gate and the
        escalation map -- which is the overwhelmingly common case, and the
        whole point of the gate: a phone across the room, or exam paper in
        hand, is not a finding.

        Exposed separately from :meth:`escalate` because the backend raises a
        flag rather than reporting a pose id, and needs the same decision
        without inventing a sentinel verdict to compare against.
        """
        if not objects:
            return None

        require_spatial = perf_config.OBJ_REQUIRE_SPATIAL_AGREEMENT

        for obj in objects:
            conf = obj["confidence"]
            if conf < perf_config.OBJ_SPATIAL_MIN_CONFIDENCE:
                continue

            if require_spatial:
                # Either the student is holding it, or it is within reach.
                # Both are necessary: without reach, a phone across the room
                # says nothing; without confidence, a fuzzy box near a wrist
                # says nothing.
                if not (obj.get("in_hand") or obj.get("reachable")):
                    continue

            escalation = perf_config.OBJ_ESCALATE_TO.get(obj["label"])
            if escalation is None:
                continue
            return escalation

        return None

    @classmethod
    def escalate(
        cls, pose_id: str, confidence: float, objects: list[dict[str, Any]]
    ) -> tuple[str, float]:
        """Escalate a verdict using detected objects, with spatial agreement.

        A detection must clear the spatial gate (unless
        ``OBJ_REQUIRE_SPATIAL_AGREEMENT`` is disabled for comparison) AND be a
        label configured to escalate.  Which labels escalate is configuration
        rather than code, because a custom desk model names 13 classes and
        each one needs a decision; anything unmapped (paper, pen, calculator)
        is displayed but never escalates -- holding exam paper is an exam,
        not a finding.
        """
        hit = cls.escalation(objects)
        if hit is None:
            return pose_id, confidence
        target, floor = hit
        return target, max(confidence, floor)
