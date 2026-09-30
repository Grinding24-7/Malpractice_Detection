"""Verify the CCTV backend uses the SAME desk-object rules as the desktop.

The backend previously had its own object logic: it cropped the bottom 60%
of each *person box* (so anything beside the student was unreachable), had
no spatial relation, no nearest-person attribution, no persistence, and no
escalation map -- which is how a proctor's laptop on the far desk became a
HIGH-severity alert accusing the student.

These tests pin the fix at its three seams:

  1. the COCO-17 -> shared MediaPipe-33 keypoint conversion, because a wrong
     mapping silently mis-measures shoulders (scale) and wrists (relation);
  2. the backend really is wired to the shared ObjectContext, not to a
     private copy that will drift;
  3. end-to-end through _detect_objects with a stub model: an object far
     from every student must NOT raise object_detected, while the same
     object at a wrist must.

No YOLO weights are needed -- the detector model is stubbed -- so this runs
fast and offline.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "backend"))

logging.disable(logging.WARNING)

import detector as backend_detector  # noqa: E402
from detector import (  # noqa: E402
    InferenceResult, PoseDetector, coco_to_mp_skeleton,
)
from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.object_context import (  # noqa: E402
    MP_LEFT_SHOULDER, MP_LEFT_WRIST, MP_NOSE, MP_RIGHT_SHOULDER,
    MP_RIGHT_WRIST, ObjectContext, shoulder_width_of,
)

FRAME_W, FRAME_H = 1280, 720


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return condition


# ---------------------------------------------------------------------------
# Stub model: serves fixed detections in CROP coordinates, shaped exactly as
# ctx.detect() reads them (xyxy/cls/conf with .cpu().numpy()).
# ---------------------------------------------------------------------------

class _T:
    def __init__(self, arr) -> None:
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _StubModel:
    """Returns each queued detection once, then nothing."""

    def __init__(self, boxes, cls, conf) -> None:
        self._boxes = boxes
        self._cls = cls
        self._conf = conf
        self.calls = 0

    def __call__(self, crop, **kwargs):
        self.calls += 1

        class _B:
            xyxy = _T(self._boxes)
            cls = _T(self._cls)
            conf = _T(self._conf)

        class _R:
            boxes = _B()

        return [_R()]


def make_coco_person(
    shoulder_l=(560.0, 260.0), shoulder_r=(720.0, 260.0),
    wrist_l=(560.0, 420.0), wrist_r=(720.0, 420.0),
    nose=(640.0, 200.0),
) -> np.ndarray:
    """One COCO-17 person with the five landmarks the converter reads set."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = (*nose, 1.0)              # NOSE
    kpts[5] = (*shoulder_l, 1.0)        # LEFT_SHOULDER
    kpts[6] = (*shoulder_r, 1.0)        # RIGHT_SHOULDER
    kpts[9] = (*wrist_l, 1.0)           # LEFT_WRIST
    kpts[10] = (*wrist_r, 1.0)          # RIGHT_WRIST
    return kpts


def test_keypoint_conversion() -> bool:
    """1. COCO-17 -> shared MediaPipe-33 layout"""
    print("1. COCO-17 keypoints convert to the shared MediaPipe layout")
    kpts = make_coco_person()
    mp = coco_to_mp_skeleton(kpts)

    ok = check(f"33 landmarks emitted ({len(mp)})", len(mp) == 33)
    ok &= check("nose -> index 0",
                mp[MP_NOSE][:2] == (640.0, 200.0), str(mp[MP_NOSE]))
    ok &= check("left shoulder -> index 11",
                mp[MP_LEFT_SHOULDER][:2] == (560.0, 260.0))
    ok &= check("right shoulder -> index 12",
                mp[MP_RIGHT_SHOULDER][:2] == (720.0, 260.0))
    ok &= check("left wrist -> index 15",
                mp[MP_LEFT_WRIST][:2] == (560.0, 420.0))
    ok &= check("right wrist -> index 16",
                mp[MP_RIGHT_WRIST][:2] == (720.0, 420.0))

    # Shoulder width is the normalising scale for every distance in the
    # shared code: if the conversion is wrong, reach is misjudged silently.
    width = shoulder_width_of(mp)
    ok &= check(f"shoulder width derives correctly ({width:.0f}px, want 160)",
                abs(width - 160.0) < 0.01, f"got {width}")

    # Unfilled landmarks must be INVISIBLE, not visible points at the origin
    # -- a (0,0) joint with confidence would corrupt any distance read.
    ok &= check(
        "unfilled landmarks are invisible (conf 0), not points at (0,0)",
        mp[20][2] == 0.0 and mp[20][:2] == (0.0, 0.0),
        str(mp[20]),
    )
    return ok


def test_shared_module_identity() -> bool:
    """2. backend must import the ONE shared implementation, not a copy."""
    print()
    print("2. backend is wired to the shared ObjectContext")
    ok = check(
        "backend.detector.ObjectContext is the desktop's class",
        backend_detector.ObjectContext is ObjectContext,
    )
    ok &= check(
        "backend imports the same perf_config the desktop tunes",
        backend_detector.perf_config is perf_config,
    )
    return ok


def test_escalation_decision() -> bool:
    """3. the flag decision: existence never escalates, spatial agreement does."""
    print()
    print("3. escalation requires spatial agreement (backend decision path)")
    far_phone = {
        "label": "cell_phone", "confidence": 0.9,
        "box": [100, 400, 160, 460],
        "relation": "away", "in_hand": False, "reachable": False,
    }
    held_phone = dict(far_phone, relation="in_hand",
                      in_hand=True, reachable=True, box=[690, 400, 750, 440])
    paper = {
        "label": "exam_paper", "confidence": 0.95,
        "box": [690, 400, 750, 440],
        "relation": "in_hand", "in_hand": True, "reachable": True,
    }

    ok = check(
        "phone across the room -> no escalation (was a false HIGH)",
        ObjectContext.escalation([far_phone]) is None,
    )
    ok &= check(
        "phone at the wrist -> escalates",
        ObjectContext.escalation([held_phone]) is not None,
        str(ObjectContext.escalation([held_phone])),
    )
    ok &= check(
        "exam paper in hand -> deliberately no escalation",
        ObjectContext.escalation([paper]) is None,
    )
    ok &= check("no objects -> no escalation",
                ObjectContext.escalation([]) is None)
    return ok


def _run_detect(stub, detections_in_full_frame, result) -> list:
    """Run PoseDetector._detect_objects end-to-end with a stubbed model.

    ctx.detect() crops the desk region first and offsets detections back to
    full frame, so the stub must serve boxes in *crop* coordinates.  The
    context is pre-seeded with exactly the state _detect_objects will set,
    which makes the crop origin used here identical to the one it uses --
    otherwise a narrowing desk region would shift every box by the difference.
    """
    det = PoseDetector.__new__(PoseDetector)   # no model loading
    ctx = ObjectContext()
    ctx.model = stub
    det._obj_ctx = ctx

    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    skeletons = [coco_to_mp_skeleton(p) for p in result.keypoints]
    people = ObjectContext.capture_persons(skeletons, result.boxes)
    ctx.set_frame_context(
        frame.shape,
        people,
        list(map(int, result.boxes[0])),
        skeletons[0],
        people[0]["shoulder_w"],
    )

    rx1, ry1, _, _ = ctx.desk_region(frame.shape)
    stub._boxes = np.asarray(
        [[b[0] - rx1, b[1] - ry1, b[2] - rx1, b[3] - ry1]
         for b in detections_in_full_frame],
        dtype=np.float32,
    )
    stub._cls = np.asarray(
        [67] * len(detections_in_full_frame), dtype=np.int64)  # cell_phone
    stub._conf = np.asarray(
        [0.9] * len(detections_in_full_frame), dtype=np.float32)

    return det._detect_objects(frame, result)


def test_end_to_end_flag() -> bool:
    """4. through _detect_objects: far object must not raise the flag."""
    print()
    print("4. _detect_objects end-to-end (stubbed model, no YOLO needed)")
    ok = True

    person = make_coco_person()
    result = InferenceResult(
        keypoints=np.stack([person]),
        boxes=np.array([[560, 120, 720, 700]], dtype=np.float32),
        timestamps=0.0,
    )

    # 4a. A phone beside the student, far from both wrists: reported, but
    #     must NOT raise object_detected.
    stub = _StubModel([], [], [])
    dets = _run_detect(stub, [[100, 400, 160, 460]], result)
    ok &= check(f"detection is reported in the payload ({len(dets)})",
                len(dets) == 1, str(dets[:1]))
    if dets:
        ok &= check(
            f"attributed as {dets[0]['relation']} (not in_hand)",
            not dets[0]["in_hand"] and not dets[0]["reachable"],
        )
        ok &= check(
            "far object does NOT raise object_detected",
            ObjectContext.escalation(dets) is None,
        )

    # 4b. The same phone at the student's right wrist: escalates.
    stub2 = _StubModel([], [], [])
    dets2 = _run_detect(stub2, [[690, 400, 750, 440]], result)
    ok &= check(f"phone at wrist reported ({len(dets2)})", len(dets2) == 1)
    if dets2:
        ok &= check(
            f"attributed as {dets2[0]['relation']}",
            dets2[0]["in_hand"] and dets2[0]["reachable"],
        )
        hit = ObjectContext.escalation(dets2)
        ok &= check(
            "phone at wrist DOES raise object_detected",
            hit is not None and hit[0] == "P38",
            str(hit),
        )

    # 4c. The region searched must be the shared desk region, not the old
    #     bottom-60%-of-person-box crop that could not see beside the student.
    ctx = ObjectContext()
    ctx.person_bbox = [560, 120, 720, 700]
    ctx.shoulder_w = 160.0
    x1, y1, x2, y2 = ctx.desk_region((FRAME_H, FRAME_W, 3))
    ok &= check(
        f"desk region spans {(x2 - x1) / FRAME_W:.0%} of the frame "
        "(old crop: 12.5%)",
        (x2 - x1) / FRAME_W > 0.75,
        f"x {x1}..{x2}",
    )
    return ok


def main() -> int:
    ok = True
    ok &= test_keypoint_conversion()
    ok &= test_shared_module_identity()
    ok &= test_escalation_decision()
    ok &= test_end_to_end_flag()
    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
