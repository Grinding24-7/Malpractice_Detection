"""Verify the desk region, spatial relations, and object-driven escalation.

The pre-rewrite behaviour: any phone, book or laptop above 0.5 confidence
anywhere in the person's own lap crop escalated to P38/P13. Nothing about
where the object was, or whose it was, was considered, so a proctor's laptop
on the far side of the room would have been reported as the student using a
device.

The rewrite requires spatial agreement -- the object must be in a hand or
within reach -- and the search region now covers the desk instead of the
torso box. These tests pin both halves, because a detector that finds more
objects is only an improvement if the escalations get more precise too.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import (  # noqa: E402
    MP_LEFT_SHOULDER, MP_LEFT_WRIST, MP_RIGHT_SHOULDER, MP_RIGHT_WRIST,
    OBJ_CLASS_MAP, _PRESENCE_POSES, PoseClassifier,
)

FRAME_W, FRAME_H = 1280, 720


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return condition


def make_landmarks(
    nose=(640.0, 200.0),
    left_shoulder=(560.0, 260.0),
    right_shoulder=(720.0, 260.0),
    left_wrist=(560.0, 420.0),
    right_wrist=(720.0, 420.0),
) -> list[tuple[float, float, float]]:
    """33 visible landmarks with the torso/arms placed where the tests need."""
    lms = [(0.0, 0.0, 1.0)] * 33
    lms[0] = (*nose, 1.0)
    lms[MP_LEFT_SHOULDER] = (*left_shoulder, 1.0)
    lms[MP_RIGHT_SHOULDER] = (*right_shoulder, 1.0)
    lms[MP_LEFT_WRIST] = (*left_wrist, 1.0)
    lms[MP_RIGHT_WRIST] = (*right_wrist, 1.0)
    return lms


def test_desk_region_covers_frame() -> bool:
    """The region must reach the frame edges, not just the torso."""
    print("1. desk region covers the desk, not just the torso")
    c = PoseClassifier()
    c._last_frame_shape = (FRAME_H, FRAME_W, 3)
    c._last_frame_width = FRAME_W
    c._last_person_bbox = [560, 120, 720, 700]
    c._last_shoulder_width = 160.0

    x1, y1, x2, y2 = c._desk_region((FRAME_H, FRAME_W, 3))
    width_frac = (x2 - x1) / FRAME_W
    ok = check(
        f"region spans {width_frac:.0%} of frame width (was 20%)",
        width_frac > 0.75,
        f"x {x1}..{x2} of {FRAME_W}",
    )
    ok &= check(
        f"region starts at {y1/FRAME_H:.0%} of frame height",
        abs(y1 / FRAME_H - perf_config.OBJ_REGION_TOP_RATIO) < 0.02,
    )
    ok &= check("region reaches the bottom edge", y2 == FRAME_H)
    return ok


def test_desk_region_excludes_head() -> bool:
    """A phone held to the ear is a different finding; don't search the head."""
    print()
    print("2. desk region excludes the head/upper body")
    c = PoseClassifier()
    c._last_frame_shape = (FRAME_H, FRAME_W, 3)
    c._last_frame_width = FRAME_W
    c._last_person_bbox = [560, 120, 720, 700]
    c._last_shoulder_width = 160.0
    x1, y1, x2, y2 = c._desk_region((FRAME_H, FRAME_W, 3))
    return check(
        "top edge is below the shoulders",
        y1 > 120 + (700 - 120) * 0.2,
        f"y1={y1} vs person box 120..700",
    )


def test_spatial_relations() -> bool:
    """Position relative to the wrists is what distinguishes use from scenery."""
    print()
    print("3. spatial relation classification")
    c = PoseClassifier()
    c._last_frame_shape = (FRAME_H, FRAME_W, 3)
    c._last_frame_width = FRAME_W
    c._last_shoulder_width = 160.0
    c._last_landmarks = make_landmarks()
    # Right wrist is at (720, 420).
    ok = True

    at_wrist = c._classify_object_relation([690, 400, 750, 440], c._last_landmarks)
    ok &= check(
        f"object at the wrist -> {at_wrist['relation']}",
        at_wrist["relation"] == "in_hand" and at_wrist["in_hand"],
    )

    near = c._classify_object_relation([790, 430, 850, 470], c._last_landmarks)
    ok &= check(
        f"object just past the wrist -> {near['relation']} "
        f"(reachable={near['reachable']})",
        near["reachable"] and not near["in_hand"],
    )

    far = c._classify_object_relation([100, 400, 160, 460], c._last_landmarks)
    ok &= check(
        f"object across the room -> {far['relation']} "
        f"(reachable={far['reachable']})",
        not far["reachable"] and not far["in_hand"],
    )

    floor = c._classify_object_relation([1000, 640, 1080, 700], c._last_landmarks)
    ok &= check(f"object at floor level -> {floor['relation']}",
                floor["relation"] in ("away", "lap"))
    return ok


def test_escalation_requires_spatial_agreement() -> bool:
    """The core fix: an object far from the student must not escalate."""
    print()
    print("4. escalation requires spatial agreement")
    c = PoseClassifier()
    ok = True

    far_phone = {
        "label": "cell_phone", "confidence": 0.9, "box": [100, 400, 160, 460],
        "relation": "away", "in_hand": False, "reachable": False,
    }
    pose, conf = c._escalate_from_objects("P01", 0.0, [far_phone])
    ok &= check(
        f"distant phone -> {pose} (was P38)",
        pose == "P01",
    )

    held_phone = dict(far_phone, relation="in_hand", in_hand=True, reachable=True)
    pose, conf = c._escalate_from_objects("P01", 0.0, [held_phone])
    ok &= check(f"phone in hand -> {pose}", pose == "P38")

    weak = dict(held_phone, confidence=0.20)
    pose, conf = c._escalate_from_objects("P01", 0.0, [weak])
    ok &= check(
        f"low-confidence phone in hand -> {pose} (below "
        f"{perf_config.OBJ_SPATIAL_MIN_CONFIDENCE})",
        pose == "P01",
    )

    held_book = {
        "label": "book", "confidence": 0.8, "box": [690, 400, 750, 440],
        "relation": "in_hand", "in_hand": True, "reachable": True,
    }
    pose, conf = c._escalate_from_objects("P01", 0.0, [held_book])
    ok &= check(f"book in hand -> {pose}", pose == "P13")

    no_objects = c._escalate_from_objects("P14", 0.82, [])
    ok &= check("no objects leaves the verdict untouched",
                no_objects == ("P14", 0.82))

    # Custom-model classes are reported but never escalate: holding exam
    # paper is what exam-taking looks like, and flagging it would accuse
    # every student in every exam.
    paper = {
        "label": "exam_paper", "confidence": 0.95,
        "box": [690, 400, 750, 440],
        "relation": "in_hand", "in_hand": True, "reachable": True,
    }
    pose, conf = c._escalate_from_objects("P01", 0.0, [paper])
    ok &= check(
        f"paper in hand -> {pose} (deliberately not escalated)",
        pose == "P01",
    )

    tablet = dict(held_phone, label="tablet")  # custom-model device class
    pose, conf = c._escalate_from_objects("P01", 0.0, [tablet])
    ok &= check(f"tablet in hand -> {pose} (custom device class)",
                pose == "P38")
    return ok


def test_persistence_across_throttle() -> bool:
    """Objects must not flicker in and out between throttled passes."""
    print()
    print("5. detections persist across throttled passes")
    c = PoseClassifier()
    c._last_frame_shape = (FRAME_H, FRAME_W, 3)
    c._last_frame_width = FRAME_W
    c._last_shoulder_width = 160.0
    c._last_landmarks = make_landmarks()
    c._frames_seen = 100

    fresh = [{
        "label": "cell_phone", "confidence": 0.7,
        "box": [690, 400, 750, 440],
        "relation": "in_hand", "in_hand": True, "reachable": True,
    }]
    first = c._merge_persistent_objects(fresh)
    ok = check("first pass reports the object", len(first) == 1)
    ok &= check("marked as fresh (not persistent)",
                first and not first[0]["persistent"])

    # A throttled pass adds nothing, but the object must survive.
    c._frames_seen = 104
    second = c._merge_persistent_objects([])
    ok &= check("survives a pass with no new detections", len(second) == 1)
    ok &= check("age_frames counted", second[0]["age_frames"] == 4,
                str(second[0]["age_frames"]))

    # Refresh it.
    c._frames_seen = 108
    third = c._merge_persistent_objects(fresh)
    ok &= check("refresh resets the age", third[0]["age_frames"] == 0)
    ok &= check("now marked persistent", third[0]["persistent"])

    # A different object far away must be a separate track, not a move.
    c._frames_seen = 112
    second_phone = [{
        "label": "cell_phone", "confidence": 0.6,
        "box": [100, 400, 160, 460],
        "relation": "away", "in_hand": False, "reachable": False,
    }]
    both = c._merge_persistent_objects(second_phone)
    ok &= check(f"two phones tracked separately ({len(both)})", len(both) == 2)

    # Ageing out.  Measured from when the track was last refreshed (112),
    # not from an earlier point, otherwise this would pass for the wrong
    # reason.
    last_seen = 112
    c._frames_seen = last_seen + perf_config.OBJ_PERSIST_FRAMES + 1
    gone = c._merge_persistent_objects([])
    ok &= check(f"tracks expire {perf_config.OBJ_PERSIST_FRAMES} frames after "
                f"last seen ({len(gone)} left)", len(gone) == 0)

    # One frame short of expiry must still be remembered.  Refresh both
    # tracks at the same instant first, so the boundary is tested rather
    # than the difference between two different refresh times.
    c._frames_seen = 112
    c._merge_persistent_objects(fresh)
    c._merge_persistent_objects(second_phone)
    c._frames_seen = 112 + perf_config.OBJ_PERSIST_FRAMES
    still = c._merge_persistent_objects([])
    ok &= check(f"both remembered at exactly {perf_config.OBJ_PERSIST_FRAMES} "
                f"frames ({len(still)})", len(still) == 2)
    c._frames_seen = 112 + perf_config.OBJ_PERSIST_FRAMES + 1
    gone2 = c._merge_persistent_objects([])
    ok &= check(f"both dropped one frame later ({len(gone2)})", len(gone2) == 0)
    return ok


def test_track_association_geometry() -> bool:
    """Association must consider both axes, one-to-one.

    Matching on the horizontal centre alone merged two same-class objects
    stacked vertically into one track, and let two fresh detections claim a
    single track, silently dropping the second object.
    """
    print()
    print("6. track association: both axes, one-to-one")
    ok = True

    def fresh_classifier() -> PoseClassifier:
        c = PoseClassifier()
        c._last_frame_shape = (FRAME_H, FRAME_W, 3)
        c._last_frame_width = FRAME_W
        c._frames_seen = 200
        return c

    high = [690, 400, 750, 440]   # centre (720, 420)
    low = [690, 620, 750, 660]    # centre (720, 640): same x, 220px lower
    det_high = {
        "label": "cell_phone", "confidence": 0.7, "box": high,
        "relation": "in_hand", "in_hand": True, "reachable": True,
    }
    det_low = dict(
        det_high, box=low, relation="lap", in_hand=False, reachable=False
    )

    c = fresh_classifier()
    c._merge_persistent_objects([det_high])
    stacked = c._merge_persistent_objects([det_low])
    ok &= check(
        f"stacked same-class phone stays a separate track "
        f"({len(stacked)} tracks)",
        len(stacked) == 2,
    )
    ok &= check(
        "vertical separation respected "
        f"(dy={220} vs max {perf_config.OBJ_TRACK_MAX_SHIFT_Y * FRAME_H:.0f}px)",
        len(stacked) == 2,
    )

    # Two fresh detections around ONE existing track: the second must not be
    # swallowed by the first's match.
    c = fresh_classifier()
    c._merge_persistent_objects([det_high])
    near_high = dict(det_high, box=[700, 410, 760, 450])
    both = c._merge_persistent_objects([det_high, near_high])
    ok &= check(
        f"two overlapping fresh detections keep both objects ({len(both)})",
        len(both) == 2,
    )

    # Re-detection of a static object re-acquires its track instead of
    # growing the list every throttled pass.
    c = fresh_classifier()
    c._merge_persistent_objects([det_high])
    for step in range(1, 6):
        c._frames_seen = 200 + step
        c._merge_persistent_objects([dict(det_high, confidence=0.65)])
    reacquired = c._merge_persistent_objects([])
    ok &= check(
        f"static object re-acquired across 5 throttled passes "
        f"({len(reacquired)} track)",
        len(reacquired) == 1,
    )
    ok &= check("re-acquired track marked persistent",
                reacquired and reacquired[0]["persistent"])
    return ok


def test_nearest_person_attribution() -> bool:
    """An object must be judged against the person nearest it.

    Attributing to whoever was detected first made a phone in the right-hand
    student's hand invisible if another student was detected first, and
    normalised the distance by the wrong person's shoulder width.
    """
    print()
    print("7. objects attribute to the nearest person")
    c = PoseClassifier()
    c._last_frame_shape = (FRAME_H, FRAME_W, 3)
    c._last_frame_width = FRAME_W

    far_left = make_landmarks(
        nose=(100.0, 200.0),
        left_shoulder=(60.0, 260.0), right_shoulder=(140.0, 260.0),
        left_wrist=(60.0, 420.0), right_wrist=(140.0, 420.0),
    )
    right_person = make_landmarks()  # wrists at (560,420)/(720,420)

    c._last_persons = c._capture_person_context(
        [far_left, right_person],
        [[40, 120, 160, 700], [560, 120, 720, 700]],
    )
    # Singular context still points at person 0, which is what the pre-fix
    # code used for every object.
    c._last_landmarks = far_left
    c._last_shoulder_width = c._shoulder_width_of(far_left)
    c._last_person_bbox = [40, 120, 160, 700]

    held = [690, 400, 750, 440]  # at person 2's right wrist
    rel = c._relation_to_nearest_person(held)
    ok = check(
        f"attributed to the nearest person (person_index="
        f"{rel.get('person_index')})",
        rel.get("person_index") == 1,
        str(rel.get("person_index")),
    )
    ok &= check(
        f"object at their wrist -> {rel['relation']}",
        rel["relation"] == "in_hand" and rel["in_hand"],
    )

    # The distance must be normalised by the NEAREST person's shoulders,
    # not person 0's: person 1's shoulder width is 160, person 0's is 80.
    expected = 0.0  # wrist-to-centre distance is zero here
    ok &= check(
        f"normalised by their own scale (distance={rel['distance']})",
        abs(rel["distance"] - expected) < 0.05,
        f"got {rel['distance']}",
    )

    # An object genuinely far from both stays away, so the nearest-person
    # rule cannot be used to escalate background clutter.
    clutter = [1100, 640, 1180, 700]
    rel2 = c._relation_to_nearest_person(clutter)
    ok &= check(
        f"far clutter stays {rel2['relation']}, never in_hand",
        not rel2["in_hand"] and not rel2["reachable"],
    )
    return ok


def test_class_map() -> bool:
    """COCO ids must be right; a wrong id silently detects the wrong class."""
    print()
    print("8. COCO class ids")
    ok = check("63=laptop", OBJ_CLASS_MAP[63] == "laptop")
    ok &= check("67=cell_phone", OBJ_CLASS_MAP[67] == "cell_phone")
    ok &= check("73=book", OBJ_CLASS_MAP[73] == "book")
    print("       note: paper/pen/desk are not in COCO-80 and remain")
    print("             unreachable until a custom model is trained.")
    return ok


class _FakeModel:
    """Stands in for a trained desk model to test class-map adoption."""

    names = {0: "person", 1: "exam_paper", 2: "pen", 3: "notebook"}


def test_custom_model_class_adoption() -> bool:
    """A custom model's own class names must become the label set.

    This is the whole point of the custom-model path: paper, pen and notebook
    are exactly the classes COCO-80 lacks, so a model trained on them has to
    work without editing the detection code.
    """
    print()
    print("9. custom desk model wiring")
    ok = True

    c = PoseClassifier.__new__(PoseClassifier)
    c._obj_class_map = dict(OBJ_CLASS_MAP)
    c._obj_custom = False
    c._adopt_custom_object_classes(_FakeModel())

    ok &= check("class names normalised", c._obj_class_map[1] == "exam_paper")
    ok &= check("multiword name becomes snake_case",
                c._obj_class_map.get(3) == "notebook")
    ok &= check("person excluded -- a person box is not desk "
                "evidence", 0 not in c._obj_class_map)
    ok &= check("previously unreachable COCO class now mapped",
                "paper" in c._obj_class_map.values()
                or "exam_paper" in c._obj_class_map.values())
    ok &= check("marked as custom", c._obj_custom is True)

    # Exclusion is by NAME, not by id.  desk_classes.py puts exam_paper at
    # id 0, so an id-based rule would have dropped the one class the custom
    # model exists for -- silently, while every log still looked healthy.
    c3 = PoseClassifier.__new__(PoseClassifier)
    c3._obj_class_map = dict(OBJ_CLASS_MAP)
    c3._obj_custom = False
    c3._adopt_custom_object_classes(
        type("M", (), {"names": dict(enumerate(
            ["exam_paper", "answer_sheet", "notebook", "pen", "person"]))})()
    )
    ok &= check(
        "id 0 kept when it is a desk class (exam_paper)",
        c3._obj_class_map.get(0) == "exam_paper",
        str(c3._obj_class_map),
    )
    ok &= check(
        "a person class at ANY id is excluded",
        4 not in c3._obj_class_map
        and "person" not in c3._obj_class_map.values(),
        str(c3._obj_class_map),
    )
    ok &= check("the rest of the desk classes survive",
                len(c3._obj_class_map) == 4,
                str(sorted(c3._obj_class_map)))

    # An empty names mapping must not leave a model that reports nothing while
    # appearing healthy.
    c2 = PoseClassifier.__new__(PoseClassifier)
    c2._obj_class_map = dict(OBJ_CLASS_MAP)
    c2._obj_custom = False
    c2._adopt_custom_object_classes(type("M", (), {"names": {}})())
    ok &= check("no classes -> empty label set, not a stale COCO map",
                c2._obj_class_map == {})

    # Precedence: a custom desk model must win over the stock weights.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        custom = pathlib.Path(tmp) / perf_config.OBJ_CUSTOM_MODEL_FILE
        custom.write_bytes(b"not a real checkpoint")
        stock = pathlib.Path(tmp) / perf_config.OBJ_FALLBACK_MODEL_FILE
        stock.write_bytes(b"not a real checkpoint")

        saved = perf_config.OBJ_MODEL_DIRS
        try:
            perf_config.OBJ_MODEL_DIRS = (tmp,)
            found = PoseClassifier._resolve_object_model(
                perf_config.OBJ_CUSTOM_MODEL_FILE
            )
            ok &= check("custom desk model found", found is not None)
            ok &= check("custom model flagged as custom", bool(found and found[1]))

            missing = PoseClassifier._resolve_object_model("not_there.pt")
            ok &= check("absent weights resolve to None (no crash)",
                        missing is None)
        finally:
            perf_config.OBJ_MODEL_DIRS = saved

    return ok


def test_object_context_wanted() -> bool:
    """The object model must only run when it could change the verdict.

    On a black frame a second YOLO pass is pure waste, and for presence
    verdicts ("Absent Seat", "Multiple People") no object can change the
    answer.  Exercised through the real predicate used by classify(), so a
    regression in the call site is caught rather than a stale copy of the
    condition.
    """
    print()
    print("10. object context is requested only when it can change the verdict")
    ok = True

    c = PoseClassifier.__new__(PoseClassifier)
    c._obj_model = object()
    c._last_landmarks = [[0.0, 0.0, 0.9]] * 33

    wants = lambda pose, warming=False: c._object_context_wanted(pose, warming)

    for pose in ("NO_SIGNAL", "INITIALIZING"):
        ok &= check(f"{pose} (black/starting feed) -> no object pass",
                    not wants(pose))
    for pose in ("P16", "P35"):
        ok &= check(f"{pose} (presence verdict) -> no object pass",
                    not wants(pose))
    ok &= check("P01 (no anomaly) -> no object pass", not wants("P01"))
    ok &= check("P13 (note peeking) -> object pass wanted", wants("P13"))
    ok &= check("P38 (unusual object) -> object pass wanted", wants("P38"))
    ok &= check("warmup -> no object pass", not wants("P13", warming=True))

    c._last_landmarks = None
    ok &= check("no visible student -> no object pass", not wants("P13"))

    c._last_landmarks = [[0.0, 0.0, 0.9]] * 33
    c._obj_model = None
    ok &= check("no object model loaded -> no object pass", not wants("P13"))

    ok &= check("presence set matches the canonical immediate set",
                _PRESENCE_POSES == perf_config.IMMEDIATE_POSES)
    return ok


def main() -> int:
    ok = True
    ok &= test_desk_region_covers_frame()
    ok &= test_desk_region_excludes_head()
    ok &= test_spatial_relations()
    ok &= test_escalation_requires_spatial_agreement()
    ok &= test_persistence_across_throttle()
    ok &= test_track_association_geometry()
    ok &= test_nearest_person_attribution()
    ok &= test_class_map()
    ok &= test_custom_model_class_adoption()
    ok &= test_object_context_wanted()
    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
