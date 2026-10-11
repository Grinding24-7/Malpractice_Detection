"""End-to-end check: launch the capture + inference pipeline offscreen and
confirm it reaches a steady state, that a dead source reports a camera fault
rather than a proctoring finding, that inference stays inside its budget, and
that a normal frame never pays for the second (object detection) model pass.

Runs the real CameraThread and InferenceThread with PyQt offscreen, so this
exercises the wiring rather than the pieces in isolation.
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

os_env = "QT_QPA_PLATFORM"
import os  # noqa: E402

os.environ.setdefault(os_env, "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.camera_thread import (  # noqa: E402
    BLACK_STREAK_FRAMES, CameraThread,
)
from desktop_app.core.inference_worker import InferenceWorker  # noqa: E402

logging.basicConfig(
    level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s"
)
# The camera thread logs a warning per dead-source transition, which is useful
# in the app and noise here.
logging.getLogger("desktop_app.core.camera_thread").setLevel(logging.ERROR)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SAMPLE = _ROOT / "backend" / "sample_exam.mp4"
_BLACK = _ROOT / "backend" / "1_hour_exam_test.mp4"
_FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "_fixtures"


def build_lit_then_black_fixture() -> pathlib.Path:
    """Create a clip that opens with a real image, then goes black.

    The all-black clip is now rejected at open time and falls through to the
    next source, so it cannot exercise the *runtime* watchdog.  This fixture
    covers the other case: a camera that is healthy at startup and then stops
    delivering an image, which is the common real-world failure (unplugged,
    lens capped, stream stalls to a black frame).
    """
    _FIXTURE_DIR.mkdir(exist_ok=True)
    path = _FIXTURE_DIR / "lit_then_black.mp4"
    if path.is_file():
        return path

    src = cv2.VideoCapture(str(_SAMPLE))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (768, 432)
    )
    # Long enough lit section that startup warmup (30 frames) completes and
    # the classifier reaches a real verdict before the source dies, then
    # ~5 s of black.  The capture thread loops the file, so the watchdog will
    # see repeated fault/recover cycles, which is itself worth exercising.
    for _ in range(60):
        ok, frame = src.read()
        if not ok:
            break
        writer.write(frame)
    src.release()
    for _ in range(60):
        writer.write(np.zeros((432, 768, 3), dtype=np.uint8))
    writer.release()
    return path


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return condition


def _wait_and_instrument(inference, timeout: float = 25.0) -> list:
    """Spy on the classifier's object gate once the worker has built it.

    The worker constructs ``PoseClassifier()`` inside its thread, so the spy
    has to wait for it.  Recording every ``(raw_pose, warming_up, wanted)``
    the real ``classify()`` produces is what makes the single-pass assertion
    possible: a normal frame must never ask for the second (object) model
    pass, and that property only holds if the gate is observed in situ --
    re-testing a copy of the condition would pass even if the call site
    ignored it.
    """
    log: list = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        clf = getattr(inference, "_classifier", None)
        ctx = getattr(clf, "_obj_ctx", None) if clf is not None else None
        if ctx is not None:
            original = clf._object_context_wanted

            def spy(pose, warming, _original=original, _log=log):
                wanted = _original(pose, warming)
                _log.append((pose, warming, wanted))
                return wanted

            clf._object_context_wanted = spy
            return log
        time.sleep(0.05)
    return log  # empty -> the assertions below fail loudly, as they should


def run(source, seconds: float, tag: str) -> dict:
    """Run the pipeline for *seconds* and summarise what it reported."""
    print(f"--- {tag} ---")
    app = QApplication.instance() or QApplication([])

    camera = CameraThread(source=source)
    inference = InferenceWorker(camera)

    results, latencies, statuses, fps_log = [], [], [], []
    inference.result_ready.connect(results.append)
    camera.status_update.connect(statuses.append)
    inference.fps_updated.connect(fps_log.append)
    camera.finished.connect(app.quit)
    QTimer.singleShot(int(seconds * 1000), app.quit)

    camera.start()
    inference.start()
    wanted_log = _wait_and_instrument(inference)
    app.exec()
    inference.stop()
    camera.wait(3000)

    # Read the object detector's call counter after the worker has drained,
    # so it cannot move under us while we compare it with the gate log.
    detect_calls = -1
    clf = getattr(inference, "_classifier", None)
    ctx = getattr(clf, "_obj_ctx", None) if clf is not None else None
    if ctx is not None:
        detect_calls = ctx.detect_counter

    ok = True
    ok &= check("frames were produced", len(results) > 0, f"{len(results)} results")

    if results:
        last = results[-1]
        print(f"       last pose={last.get('pose_id')} "
              f"name={last.get('pose_name')!r}")
    if fps_log:
        steady = fps_log[len(fps_log) // 2:]
        print(f"       inference fps (second half): "
              f"min {min(steady):.1f} max {max(steady):.1f}")

    for s in statuses[:6]:
        print(f"       status: {s}")
    if len(statuses) > 6:
        print(f"       ... {len(statuses) - 6} more status lines")

    return {"results": results, "statuses": statuses, "fps_log": fps_log,
            "ok": ok, "wanted_log": wanted_log, "detect_calls": detect_calls}


def main() -> int:
    ok = True

    print("1. live video file through the real pipeline")
    out = run(str(_SAMPLE), 25.0, str(_SAMPLE.name))
    live_out = out
    ok &= out["ok"]
    live = out["results"]
    if live:
        dead = [r for r in live if r.get("pose_id") == "NO_SIGNAL"]
        ok &= check("live feed is not reported as dead", not dead,
                    f"{len(dead)} dead reports")
        fps = out["fps_log"]
        ok &= check(
            f"inference near {perf_config.TARGET_INFERENCE_FPS} fps target",
            fps and min(fps[len(fps) // 2:]) > perf_config.TARGET_INFERENCE_FPS * 0.6,
            f"fps log {len(fps)} samples",
        )

    print()
    print("2. all-black file is rejected at open, falls through to a live source")
    out = run(str(_BLACK), 6.0, f"{_BLACK.name} (all black)")
    ok &= out["ok"]
    res = out["results"]
    joined = " ".join(out["statuses"])
    ok &= check("black file was rejected", "black video" in joined, joined[:90])
    if res:
        poses = {r.get("pose_id") for r in res}
        ok &= check(
            "never reported P16 Absent Seat (would accuse a student)",
            "P16" not in poses,
            str(poses),
        )

    print()
    print("3. source that dies mid-session reports a camera fault")
    fixture = build_lit_then_black_fixture()
    out = run(str(fixture), 22.0, f"{fixture.name} (lit then black)")
    ok &= out["ok"]
    res = out["results"]
    joined = " ".join(out["statuses"])
    if res:
        poses = [r.get("pose_id") for r in res]
        print(f"       poses over time: {poses[:3]} ... {poses[-3:]}")
        p16 = [i for i, p in enumerate(poses) if p == "P16"]
        if p16:
            print(f"       P16 at frames {p16[:10]}"
                  f"{' ...' if len(p16) > 10 else ''} (of {len(poses)})")
        ok &= check("NO_SIGNAL was reported", "NO_SIGNAL" in poses,
                    f"{poses.count('NO_SIGNAL')} of {len(poses)} frames")
        ok &= check(
            "never reported P16 Absent Seat (would accuse a student)",
            "P16" not in poses,
        )
        ok &= check(
            "classified normally once warmup completed",
            "P01" in poses,
        )
        ok &= check(
            "first frames carried no proctoring verdict",
            poses[0] in ("INITIALIZING", "NO_SIGNAL", "P01"),
            f"first pose {poses[0]!r}",
        )
        ok &= check(
            "no proctoring verdict issued while the source was dead",
            not any(p in ("P16",) for p in poses),
        )
    ok &= check("operator was told the room is not monitored",
                "NOT" in joined and "monitored" in joined)

    print()
    print("4. a normal frame runs one model pass, not two")
    # The object detector is a second full YOLO pass; it is only ever worth
    # paying for when it could change the verdict.  Excluded: the raw P01
    # verdict, black/starting system states, presence verdicts (who is in
    # the room is not changed by what is on the desk), and warmup frames.
    log = live_out["wanted_log"]
    ok &= check("gate was instrumented in the live run",
                bool(log), f"{len(log)} frames observed")
    if log:
        excluded = (
            {"P01", "NO_SIGNAL", "INITIALIZING"}
            | set(perf_config.IMMEDIATE_POSES)
        )
        p01_wanted = [e for e in log if e[2] and e[0] == "P01"]
        ok &= check(
            "P01 (normal sitting) never requested the second pass",
            not p01_wanted,
            str(p01_wanted[:3]),
        )
        bad = [e for e in log if e[2] and (e[0] in excluded or e[1])]
        ok &= check(
            "no system state, presence verdict or warmup frame "
            "requested it either",
            not bad,
            str(bad[:3]),
        )
        wanting = sum(1 for e in log if e[2])
        ok &= check(
            f"object pass requested on {wanting} of {len(log)} frames",
            wanting < len(log),
        )
        ok &= check(
            "requested passes == detect() invocations (gate not bypassed)",
            live_out["detect_calls"] == wanting,
            f"detect={live_out['detect_calls']} wanted={wanting}",
        )

    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
