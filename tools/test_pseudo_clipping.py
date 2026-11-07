"""Pseudo-clipping harness: boundary trims, seat anchors, and false P16/P35.

A student who is partly outside a tight crop must not become a different
student, an absent seat, or a second person.  This harness simulates the three
ways a frame gets trimmed in practice -- cropping, masking, and overlaying a
border -- and checks the continuity layer end to end:

1. ``HeadAnchorInitializer`` turns synthetic SCUT-HEAD detections into
   ``Seat_01..Seat_N`` anchors (no model weights needed).
2. ``validate_candidate_boundaries`` rejects torso/leg fragments that have no
   head in the upper 30%, while still attributing limbs within 1.2 shoulder
   widths of the box edge.
3. ``SeatAnchorRegistry.bind`` keeps a shifted/clipped box on the same seat;
   ``CandidateTracker`` keeps the same id when ByteTrack ids are unavailable.
4. ``BoundaryClipGuard`` holds presence through the grace window, so the
   classifier publishes P01 rather than a false P16 Absent Seat, and collapses
   duplicate/overlaid boxes so P35 Multiple Persons is not manufactured.
5. A genuine second person (different track id and seat) still raises P35.
6. ``render_hud`` accepts the new seat/init fields and the neutral P00 state.

Run:  python tools/test_pseudo_clipping.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.head_detector import detect_heads  # noqa: E402
from desktop_app.core.hud_renderer import render_hud  # noqa: E402
from desktop_app.core.pose_classifier import (  # noqa: E402
    NEUTRAL_WRITING,
    PoseClassifier,
    validate_candidate_boundaries,
)
from desktop_app.core.seat_anchor import (  # noqa: E402
    BoundaryClipGuard,
    CandidateTracker,
    HeadAnchorInitializer,
    SeatAnchor,
    SeatAnchorRegistry,
)

MP_NOSE, MP_L_SHOULDER, MP_R_SHOULDER = 0, 11, 12
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_ELBOW, MP_R_ELBOW = 13, 14

FRAME_H, FRAME_W = 720, 1280
FRAME = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)

# Three seated candidates in a 1280x720 classroom view.  Each person box has
# the head anchor in its upper 30% (HEAD_UPPER_BODY_RATIO = 0.30).
PERSON_BOXES: list[list[int]] = [
    [80, 180, 360, 640],
    [440, 170, 740, 650],
    [820, 180, 1120, 660],
]
HEAD_BOXES: list[list[int]] = [
    [170, 200, 270, 300],
    [540, 190, 640, 290],
    [920, 200, 1020, 300],
]


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def classroom_frame() -> np.ndarray:
    """A simple three-student classroom scene for the visual trim helpers."""
    frame = np.full((FRAME_H, FRAME_W, 3), 185, dtype=np.uint8)
    cv2.rectangle(frame, (0, 430), (FRAME_W, FRAME_H), (120, 110, 100), -1)
    for (x1, y1, x2, y2), (hx1, hy1, hx2, hy2) in zip(PERSON_BOXES, HEAD_BOXES):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 90, 150), -1)
        cv2.circle(
            frame,
            ((hx1 + hx2) // 2, (hy1 + hy2) // 2),
            max(12, (hx2 - hx1) // 2),
            (150, 170, 210),
            -1,
        )
    return frame


def crop_frame(frame: np.ndarray, box: list[int]) -> np.ndarray:
    """Boundary trim by cropping the frame to *box*."""
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2].copy()


def mask_frame(frame: np.ndarray, box: list[int]) -> np.ndarray:
    """Boundary trim by blacking out everything outside *box*."""
    x1, y1, x2, y2 = box
    masked = np.zeros_like(frame)
    masked[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return masked


def overlay_frame(frame: np.ndarray, thickness: int = 40) -> np.ndarray:
    """Boundary trim by painting a border over the frame edges."""
    overlaid = frame.copy()
    overlaid[:thickness, :] = 0
    overlaid[-thickness:, :] = 0
    overlaid[:, :thickness] = 0
    overlaid[:, -thickness:] = 0
    return overlaid


def make_registry() -> SeatAnchorRegistry:
    anchors = [
        SeatAnchor(
            seat_id=f"Seat_{i + 1:02d}",
            box=list(box),
            confidence=0.92,
            hits=25,
        )
        for i, box in enumerate(HEAD_BOXES)
    ]
    return SeatAnchorRegistry(anchors=anchors, locked=True)


def landmarks33(
    left_wrist_x: float = 250.0,
    right_wrist_x: float = 310.0,
    wrist_y: float = 520.0,
) -> list[tuple[float, float, float]]:
    """A valid 33-point MediaPipe layout with visible shoulders and wrists."""
    lms = [(250.0, 150.0, 0.9)] * 33
    lms[MP_NOSE] = (250.0, 130.0, 0.9)
    lms[MP_L_SHOULDER] = (150.0, 200.0, 0.9)
    lms[MP_R_SHOULDER] = (350.0, 200.0, 0.9)
    lms[MP_L_ELBOW] = (160.0, 350.0, 0.9)
    lms[MP_R_ELBOW] = (340.0, 350.0, 0.9)
    lms[MP_L_WRIST] = (left_wrist_x, wrist_y, 0.9)
    lms[MP_R_WRIST] = (right_wrist_x, wrist_y, 0.9)
    return lms


def make_classifier() -> PoseClassifier:
    """A classifier with every heavyweight initialiser stubbed out.

    The harness is about continuity logic, not model accuracy.  Stubbing the
    three model loaders keeps the test fast and deterministic on CPU-only
    machines while leaving classify() and the real gate/guard code intact.
    """
    PoseClassifier._init_yolo = lambda self: setattr(self, "_yolo", None)
    PoseClassifier._init_object_detector = lambda self: (
        setattr(self, "_obj_model", None),
        setattr(self, "_obj_model_path", ""),
        setattr(self, "_obj_class_map", {}),
        setattr(self, "_obj_custom", False),
    )
    PoseClassifier._init_mediapipe = lambda self: (
        setattr(self, "_pose_landmarker", None),
        setattr(self, "_mp_image_cls", None),
        setattr(self, "_mp_image_format", None),
    )
    PoseClassifier._load_ml_classifier = lambda self: setattr(
        self, "_ml_classifier", None
    )

    clf = PoseClassifier()
    clf._frames_seen = perf_config.WARMUP_FRAMES
    clf._proximity = None
    clf._object_context_wanted = lambda *a, **k: False
    clf._detect_objects = lambda *a, **k: []
    clf._head_pose_estimator.estimate = lambda lms, image_shape: {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "gaze_vector": (0.0, 0.0, 1.0),
        "anomalies": [],
    }
    clf._head_pose_estimator.get_gaze_endpoint = lambda *a, **k: (0, 0)

    def rules(all_lms, head_pose, person_count, *args, **kwargs):
        if person_count <= 0:
            return "P16", 0.95
        if person_count >= 2:
            return "P35", 0.90
        return "P01", 0.95

    clf._evaluate_rules = rules
    return clf


def set_detection(
    clf: PoseClassifier,
    boxes: list[list[int]],
    track_ids: list[int] | None,
) -> None:
    """Stub the detector, including the ByteTrack side-effect channel."""
    def detect(_frame: np.ndarray):
        clf._last_track_ids = list(track_ids or [])
        return ([list(b) for b in boxes], [np.zeros((17, 3)) for _ in boxes])

    clf._yolo_detect = detect
    clf._mediapipe_landmarks = lambda _frame, bs: [
        landmarks33() for _ in bs
    ]


def run_classifier_frames(
    clf: PoseClassifier,
    frames: list[tuple[list[list[int]], list[int] | None]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for boxes, ids in frames:
        set_detection(clf, boxes, ids)
        results.append(clf.classify(FRAME))
    return results


def test_initializer() -> bool:
    print("1. SCUT-HEAD initialiser -> Seat_01..Seat_N anchors")
    init = HeadAnchorInitializer(frames=10, max_seconds=30.0)
    # Avoid loading weights: observe() only needs a truthy model plus the
    # patched detect_heads below.
    init._model = object()
    init._done = False

    import desktop_app.core.head_detector as hd

    original = hd.detect_heads
    frame_no = {"n": 0}

    def fake_detect_heads(model, frame, **kwargs):
        frame_no["n"] += 1
        jitter = frame_no["n"] % 3
        return [
            {
                "box": [b[0] + jitter, b[1], b[2] + jitter, b[3]],
                "confidence": 0.9,
            }
            for b in HEAD_BOXES
        ]

    try:
        hd.detect_heads = fake_detect_heads
        for _ in range(10):
            init.observe(classroom_frame())
        registry = init.finish()
    finally:
        hd.detect_heads = original

    ok = True
    ok &= check("three stable heads -> three seats",
                registry.seat_count == 3,
                f"got {registry.seat_count}")
    ok &= check("registry is locked", registry.locked)
    ok &= check("seat ids are stable Seat_01..Seat_03",
                [a.seat_id for a in registry.anchors]
                == ["Seat_01", "Seat_02", "Seat_03"],
                f"got {[a.seat_id for a in registry.anchors]}")
    ok &= check("init info reports the locked count",
                init.as_info()["seated_count"] == 3
                and init.as_info()["locked"] is True)
    return ok


def test_boundary_validation() -> bool:
    print()
    print("2. candidate boundary validation rejects headless fragments")
    registry = make_registry()
    ok = True

    full = validate_candidate_boundaries(PERSON_BOXES[0], None, registry)
    ok &= check("full person box is valid", full["valid"])
    ok &= check("full box resolves to Seat_01",
                full["seat_id"] == "Seat_01", f"got {full['seat_id']}")

    # Lower torso/legs only: the head anchor is above this box.
    fragment = validate_candidate_boundaries(
        [80, 400, 360, 640], None, registry
    )
    ok &= check("bottom-clipped torso fragment is rejected",
                not fragment["valid"],
                f"reason={fragment['reason']}")

    # A top trim that removes the head: anchor centre falls below the band.
    top_trim = validate_candidate_boundaries(
        [80, 0, 360, 190], None, registry
    )
    ok &= check("top-clipped headless fragment is rejected",
                not top_trim["valid"], f"reason={top_trim['reason']}")

    # Left trim that leaves the body but removes the head horizontally.
    left_trim = validate_candidate_boundaries(
        [0, 180, 150, 640], None, registry
    )
    ok &= check("left-clipped headless fragment is rejected",
                not left_trim["valid"], f"reason={left_trim['reason']}")

    # Limb attribution: shoulders 200 px apart -> 1.2 sw = 240 px margin.
    # A wrist outside the body box is still this candidate's hand when it is
    # within that margin of the matched head anchor or a shoulder joint.
    near_lms = landmarks33(
        left_wrist_x=500.0, right_wrist_x=450.0, wrist_y=300.0
    )
    near_limbs = validate_candidate_boundaries(
        PERSON_BOXES[0], near_lms, registry
    )
    ok &= check("near-outside wrist is attributed to the candidate",
                near_limbs["limbs_attributed"] >= 1,
                f"attributed={near_limbs['limbs_attributed']} "
                f"rejected={near_limbs['limbs_rejected']}")
    ok &= check("matched head anchor is reported",
                near_limbs["head_box"] == HEAD_BOXES[0],
                f"got {near_limbs['head_box']}")

    # Both wrists are >240 px from Seat_01's head/shoulders, even though
    # Seat_02's head happens to be closer to one of them.
    far_lms = landmarks33(left_wrist_x=500.0, right_wrist_x=620.0)
    far_limbs = validate_candidate_boundaries(
        PERSON_BOXES[0], far_lms, registry
    )
    ok &= check("far-outside wrist is rejected, not attributed",
                far_limbs["limbs_rejected"] >= 1,
                f"attributed={far_limbs['limbs_attributed']} "
                f"rejected={far_limbs['limbs_rejected']}")

    # Without anchors the legacy behaviour is preserved (no invented rule).
    legacy = validate_candidate_boundaries([80, 400, 360, 640], None, None)
    ok &= check("no anchors -> legacy box accepted",
                legacy["valid"] and legacy["reason"] == "no_anchors")
    return ok


def test_seat_and_track_continuity() -> bool:
    print()
    print("3. seat binding and fallback tracker keep one identity")
    registry = make_registry()
    shifted = [85, 185, 365, 645]
    ok = True

    seats = registry.bind([PERSON_BOXES[0], shifted], [11, 11], frame_index=5)
    ok &= check("shifted box stays on Seat_01",
                seats == ["Seat_01", "Seat_01"], f"got {seats}")
    ok &= check("anchor records the ByteTrack id",
                registry.anchors[0].track_id == 11)
    ok &= check("anchor hit counter advanced",
                registry.anchors[0].hits >= 2,
                f"hits={registry.anchors[0].hits}")

    tracker = CandidateTracker(max_age=10)
    first = tracker.update([PERSON_BOXES[0]], (FRAME_H, FRAME_W, 3))
    second = tracker.update([shifted], (FRAME_H, FRAME_W, 3))
    ok &= check("fallback tracker keeps the id across a small shift",
                first == second, f"{first} -> {second}")
    # Disappear for a few frames, then reappear: still the same track.
    for _ in range(3):
        tracker.update([], (FRAME_H, FRAME_W, 3))
    third = tracker.update([shifted], (FRAME_H, FRAME_W, 3))
    ok &= check("fallback tracker survives a short clipped gap",
                third == first, f"{first} -> {third}")
    return ok


def test_clip_guard_and_false_alerts() -> bool:
    print()
    print("4. pseudo-clips suppress false P16 and P35")
    ok = True

    # Duplicate/overlaid box: same track id and same seat -> one person.
    clf = make_classifier()
    clf.set_seat_anchors(make_registry())
    duplicate = [85, 185, 365, 645]
    results = run_classifier_frames(
        clf,
        [
            ([PERSON_BOXES[0]], [1]),
            ([PERSON_BOXES[0], duplicate], [1, 1]),
        ],
    )
    last = results[-1]
    ok &= check("overlaid duplicate collapses to one candidate",
                last["candidate_count"] == 1,
                f"got {last['candidate_count']}")
    ok &= check("duplicate does not raise P35",
                last["pose_id"] != "P35", f"got {last['pose_id']}")
    ok &= check("ByteTrack id survives the duplicate",
                last["track_ids"] and all(t == 1 for t in last["track_ids"]),
                f"got {last['track_ids']}")
    ok &= check("seat binding survives the duplicate",
                last["seat_ids"]
                and all(s == "Seat_01" for s in last["seat_ids"]),
                f"got {last['seat_ids']}")

    # Bottom-clipped fragment only: rejected, but presence is held.
    clf2 = make_classifier()
    clf2.set_seat_anchors(make_registry())
    results = run_classifier_frames(
        clf2,
        [
            ([PERSON_BOXES[0]], [7]),
            ([[80, 400, 360, 640]], [7]),   # torso only, no head
        ],
    )
    clipped = results[-1]
    ok &= check("headless fragment is not published as a person",
                clipped["bounding_boxes"] == [],
                f"got {clipped['bounding_boxes']}")
    ok &= check("presence is held through the clipped frame",
                clipped["presence_held"] is True)
    ok &= check("clipped frame reports P01, not P16",
                clipped["pose_id"] == "P01", f"got {clipped['pose_id']}")

    # Person disappears entirely for the grace window, then beyond it.
    clf3 = make_classifier()
    clf3.set_seat_anchors(make_registry())
    run_classifier_frames(clf3, [([PERSON_BOXES[0]], [3])])
    empty_results = run_classifier_frames(
        clf3,
        [([], None) for _ in range(perf_config.CLIP_GUARD_GRACE_FRAMES + 4)],
    )
    grace = perf_config.CLIP_GUARD_GRACE_FRAMES
    ok &= check("no P16 during the clip grace window",
                all(r["pose_id"] != "P16" for r in empty_results[:grace]),
                f"got {[r['pose_id'] for r in empty_results[:grace]]}")
    ok &= check("P16 returns after the grace window expires",
                any(r["pose_id"] == "P16" for r in empty_results[grace:]),
                f"got {[r['pose_id'] for r in empty_results[grace:]]}")

    # A genuine second student is still P35.
    clf4 = make_classifier()
    clf4.set_seat_anchors(make_registry())
    results = run_classifier_frames(
        clf4,
        [([PERSON_BOXES[0], PERSON_BOXES[1]], [1, 2])],
    )
    genuine = results[-1]
    ok &= check("genuine second person still raises P35",
                genuine["pose_id"] == "P35", f"got {genuine['pose_id']}")
    ok &= check("genuine second person counts as two candidates",
                genuine["candidate_count"] == 2,
                f"got {genuine['candidate_count']}")
    ok &= check("both seats are reported",
                genuine["seat_ids"] == ["Seat_01", "Seat_02"],
                f"got {genuine['seat_ids']}")
    return ok


def test_visual_trims() -> bool:
    print()
    print("5. crop / mask / overlay trims preserve frame geometry")
    frame = classroom_frame()
    ok = True

    cropped = crop_frame(frame, [60, 160, 380, 670])
    masked = mask_frame(frame, [60, 160, 380, 670])
    overlaid = overlay_frame(frame, thickness=40)
    ok &= check("crop produces the trimmed view",
                cropped.shape[:2] == (510, 320),
                f"got {cropped.shape[:2]}")
    ok &= check("mask keeps the frame size and blacks the outside",
                masked.shape == frame.shape
                and int(masked[0, 0].sum()) == 0)
    ok &= check("overlay paints the border but keeps the subject",
                overlaid.shape == frame.shape
                and int(overlaid[:40, :].sum()) == 0
                and int(overlaid[300, 200].sum()) > 0)

    # The same geometry rules apply to the trimmed frame's person box; the
    # anchors move into crop-local coordinates with the frame.
    crop_anchors = SeatAnchorRegistry(
        anchors=[
            SeatAnchor(
                seat_id=f"Seat_{i + 1:02d}",
                box=[b[0] - 60, b[1] - 160, b[2] - 60, b[3] - 160],
                confidence=0.9,
            )
            for i, b in enumerate(HEAD_BOXES)
        ],
        locked=True,
    )
    shifted = [20, 20, 300, 480]  # crop-local coordinates
    validation = validate_candidate_boundaries(shifted, None, crop_anchors)
    ok &= check("crop-local box still validates against a head anchor",
                validation["valid"], f"reason={validation['reason']}")
    return ok


def test_hud_and_neutral() -> bool:
    print()
    print("6. HUD accepts seat/init fields and the P00 neutral state")
    clf = make_classifier()
    clf.set_seat_anchors(make_registry())
    set_detection(clf, [PERSON_BOXES[0]], [1])
    result = clf.classify(FRAME)
    result["recording"] = False
    result["classroom"] = {
        "locked": True,
        "seated_count": 3,
        "frames_seen": 30,
        "frames_target": 30,
        "in_progress": False,
        "error": "",
    }
    result["neutral_state"] = NEUTRAL_WRITING
    result["neutral_label"] = "Writing"
    result["pose_id"] = "P01"
    result["pose_name"] = "Normal Sitting"
    result["confidence"] = 0.95

    ok = True
    try:
        hud = render_hud(FRAME.copy(), result, 30.0, 10.0)
        ok &= check("render_hud completed with the init header",
                    hud is not None and hud.shape == FRAME.shape)
    except Exception as exc:  # pragma: no cover - failure path
        ok &= check("render_hud completed with the init header", False, str(exc))

    ok &= check("result carries seat and track identity",
                result["seat_ids"] == ["Seat_01"]
                and result["track_ids"] == [1]
                and result["seats_locked"] is True
                and result["seated_count"] == 3)
    return ok


def main() -> int:
    ok = True
    ok &= test_initializer()
    ok &= test_boundary_validation()
    ok &= test_seat_and_track_continuity()
    ok &= test_clip_guard_and_false_alerts()
    ok &= test_visual_trims()
    ok &= test_hud_and_neutral()

    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
