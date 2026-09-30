"""Verify the dead-source watchdog and the fallback file ordering.

Two failure modes are covered:

1. A source that returns structurally valid but black frames. The watchdog
   must declare it dead and withhold it from the classifier, because
   "Absent Seat" from a dead camera is a false accusation.
2. The fallback chain. `1_hour_exam_test.mp4` is 60 minutes of black video
   (mean luma 1.6) and used to be preferred over the working
   `sample_exam.mp4`, so a dead webcam fell back to a black feed.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core.camera_thread import (  # noqa: E402
    BLACK_LUMA_THRESHOLD, BLACK_RECOVERY_FRAMES, BLACK_STREAK_FRAMES,
    CameraThread, _describe_source,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SAMPLE = _ROOT / "backend" / "sample_exam.mp4"
_BLACK = _ROOT / "backend" / "1_hour_exam_test.mp4"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return condition


def luma_of(path: pathlib.Path, frames: int = 10) -> float:
    cap = cv2.VideoCapture(str(path))
    total, n = 0.0, 0
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        total += float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        n += 1
    cap.release()
    return total / max(n, 1)


def test_watchdog_black() -> bool:
    """A black source must be declared dead and never reach the buffer."""
    print("1. black source is detected as dead")
    t = CameraThread(source=-1)
    t._mock_mode = False
    black = np.zeros((480, 640, 3), dtype=np.uint8)

    emitted, withheld = [], []
    t.raw_frame.connect(lambda f: emitted.append(f))
    # _emit_frame is what would feed the classifier.
    t._emit_frame = lambda f: withheld.append(f)

    for _ in range(BLACK_STREAK_FRAMES + 5):
        t._check_frame_alive(black)

    ok = check("source flagged dead", t.source_dead)
    ok &= check("no frame withheld from the classifier", len(withheld) == 0)
    ok &= check(
        "UI still received frames",
        len(emitted) > 0,
        f"{len(emitted)} emitted",
    )
    return ok


def test_watchdog_lit() -> bool:
    """A normal source must never be declared dead."""
    print()
    print("2. a real source is never declared dead")
    t = CameraThread(source=-1)
    t._mock_mode = False
    cap = cv2.VideoCapture(str(_SAMPLE))
    ok = True
    seen = 0
    while seen < BLACK_STREAK_FRAMES + 20:
        got, frame = cap.read()
        if not got:
            break
        seen += 1
        if t._check_frame_alive(frame):
            ok = False
            break
    cap.release()
    ok &= check(f"survived {seen} real frames", ok and not t.source_dead)
    return ok


def test_watchdog_hysteresis() -> bool:
    """A brief dark dip must not trip the watchdog."""
    print()
    print("3. hysteresis: a short dark dip does not trip the watchdog")
    t = CameraThread(source=-1)
    t._mock_mode = False
    cap = cv2.VideoCapture(str(_SAMPLE))
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    ok = True
    for i in range(60):
        got, frame = cap.read()
        if not got:
            break
        # 5 dark frames every 20 -- shorter than BLACK_STREAK_FRAMES.
        frame = black if (i % 20) < 5 else frame
        t._check_frame_alive(frame)
    cap.release()
    ok &= check("intermittent darkness below the threshold is tolerated",
                not t.source_dead)
    return ok


def test_watchdog_recovery() -> bool:
    """After a blackout, lit frames must clear the dead state."""
    print()
    print("4. recovery: lit frames clear the dead state")
    t = CameraThread(source=-1)
    t._mock_mode = False
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(_SAMPLE))

    for _ in range(BLACK_STREAK_FRAMES + 2):
        t._check_frame_alive(black)
    ok = check("dead before recovery", t.source_dead)

    for _ in range(BLACK_RECOVERY_FRAMES + 2):
        got, frame = cap.read()
        if not got:
            break
        t._check_frame_alive(frame)
    cap.release()
    ok &= check("alive again after lit frames", not t.source_dead)
    return ok


def test_fallback_ordering() -> bool:
    """The black file must be rejected in favour of the usable one."""
    print()
    print("5. fallback chain rejects the black file")
    t = CameraThread(source=-1)
    t._mock_mode = False

    cap = cv2.VideoCapture(str(_BLACK))
    ok = check(
        f"1_hour_exam_test.mp4 rejected (luma {luma_of(_BLACK):.1f})",
        not t._peek_has_image(cap, _BLACK),
    )
    cap.release()

    cap = cv2.VideoCapture(str(_SAMPLE))
    ok &= check(
        f"sample_exam.mp4 accepted (luma {luma_of(_SAMPLE):.1f})",
        t._peek_has_image(cap, _SAMPLE),
    )
    ok &= check(
        "capture position restored to start",
        cap.get(cv2.CAP_PROP_POS_FRAMES) < 5,
        f"pos={cap.get(cv2.CAP_PROP_POS_FRAMES):.0f}",
    )
    cap.release()

    print()
    print("6. _describe_source hides credentials in stream URLs")
    ok &= check(
        "rtsp://user:pass@host/stream -> stream host/stream",
        "pass" not in _describe_source("rtsp://user:pass@host/stream"),
        _describe_source("rtsp://user:pass@host/stream"),
    )
    return ok


def test_classifier_refuses_presence_on_black() -> bool:
    """A black frame must never become "Absent Seat".

    The capture watchdog only trips after BLACK_STREAK_FRAMES dark frames.
    During that window the classifier still sees those frames, so it has to
    refuse to assert anything about the room from them on its own.  This is
    the most dangerous single failure in the system: "Absent Seat" accuses a
    student of not being there, and it bypasses the stability gate because
    presence is treated as a fact rather than an inference.
    """
    print()
    print("7. classifier refuses presence claims on frames with no image")
    from desktop_app.core.pose_classifier import (
        PoseClassifier, frame_has_usable_image,
    )

    ok = check(
        "frame_has_usable_image rejects an all-black frame",
        not frame_has_usable_image(np.zeros((480, 640, 3), dtype=np.uint8)),
    )
    ok &= check(
        "frame_has_usable_image accepts a real frame",
        frame_has_usable_image(np.full((480, 640, 3), 150, dtype=np.uint8)),
    )
    ok &= check(
        "frame_has_usable_image rejects None",
        not frame_has_usable_image(None),
    )

    # Drive the rule engine directly with zero people on a black frame.
    c = PoseClassifier()
    c._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pose_id, conf = c._evaluate_rules(
        [], {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, 0, {}, [], {}, {},
    )
    ok &= check(
        f"black frame with no people -> {pose_id} (not P16)",
        pose_id == "NO_SIGNAL",
        f"got {pose_id} conf={conf}",
    )

    # The same input on a lit frame IS an absent seat.
    c._last_frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    pose_id, conf = c._evaluate_rules(
        [], {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, 0, {}, [], {}, {},
    )
    ok &= check(
        f"lit empty frame -> {pose_id} (genuine absent seat)",
        pose_id == "P16",
        f"got {pose_id} conf={conf}",
    )
    return ok


def main() -> int:
    print(f"BLACK_LUMA_THRESHOLD={BLACK_LUMA_THRESHOLD} "
          f"BLACK_STREAK_FRAMES={BLACK_STREAK_FRAMES} "
          f"BLACK_RECOVERY_FRAMES={BLACK_RECOVERY_FRAMES}")
    print()
    ok = True
    ok &= test_watchdog_black()
    ok &= test_watchdog_lit()
    ok &= test_watchdog_hysteresis()
    ok &= test_watchdog_recovery()
    ok &= test_fallback_ordering()
    ok &= test_classifier_refuses_presence_on_black()
    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
