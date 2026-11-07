"""Verify the unified SCUT-HEAD -> YOLO11-Pose cycle in InferenceWorker.

The production loop runs both models on the worker thread.  These tests call
the extracted ``_classify_frame`` method with lightweight fakes (no camera,
no Qt event loop) to prove the ordering contract:

1. SCUT-HEAD ``detect(frame)`` runs first;
2. its boxes are passed to ``classify(frame, head_boxes=...)``;
3. throttling reuses cached boxes without running the head model again;
4. the initialiser (and therefore the loaded head weights) survives the
   30-frame initialisation instead of being discarded.

Run:  python tools/test_unified_worker.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from desktop_app.core import inference_worker as iw  # noqa: E402
from desktop_app.core import perf_config  # noqa: E402


PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[PASS] {label}")
    else:
        FAIL += 1
        suffix = f" ({detail})" if detail else ""
        print(f"[FAIL] {label}{suffix}")
    return condition


class FakeInitializer:
    def __init__(self, heads: list[dict]) -> None:
        self.heads = heads
        self.loaded = True
        self.detect_calls = 0
        self.order: list[str] = []

    def detect(self, frame: np.ndarray) -> list[dict]:
        self.detect_calls += 1
        self.order.append("detect")
        return [dict(head) for head in self.heads]


class FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def classify(self, frame: np.ndarray, head_boxes=None) -> dict:
        self.calls.append({
            "frame_shape": frame.shape,
            "head_boxes": list(head_boxes or []),
        })
        return {"pose_id": "P01", "head_boxes_seen": list(head_boxes or [])}


def make_worker(
    initializer: FakeInitializer | None,
    classifier: FakeClassifier,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        _head_initializer=initializer,
        _live_heads=[],
        _live_head_counter=0,
        _classifier=classifier,
    )


def test_detect_then_classify() -> bool:
    print()
    print("1. SCUT-HEAD detect runs before pose classify")
    heads = [{"box": [100, 100, 180, 180], "confidence": 0.9}]
    init = FakeInitializer(heads)
    classifier = FakeClassifier()
    worker = make_worker(init, classifier)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    result = iw.InferenceWorker._classify_frame(worker, frame)

    ok = True
    ok &= check("head detector called exactly once", init.detect_calls == 1,
                f"got {init.detect_calls}")
    ok &= check("classifier called exactly once", len(classifier.calls) == 1)
    ok &= check("head pass ran before the pose pass",
                init.order == ["detect"])
    ok &= check(
        "classifier received the SCUT head boxes",
        classifier.calls[0]["head_boxes"] == heads,
        f"got {classifier.calls[0]['head_boxes']}",
    )
    ok &= check("result carries the same boxes", result["head_boxes_seen"] == heads)
    return ok


def test_head_throttling() -> bool:
    print()
    print("2. throttled head pass reuses cached boxes")
    heads = [{"box": [100, 100, 180, 180], "confidence": 0.9}]
    init = FakeInitializer(heads)
    classifier = FakeClassifier()
    worker = make_worker(init, classifier)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    original = perf_config.LIVE_HEAD_EVERY_N_FRAMES
    perf_config.LIVE_HEAD_EVERY_N_FRAMES = 2
    try:
        iw.InferenceWorker._classify_frame(worker, frame)  # counter 0 -> detect
        iw.InferenceWorker._classify_frame(worker, frame)  # counter 1 -> cache
        iw.InferenceWorker._classify_frame(worker, frame)  # counter 2 -> detect
    finally:
        perf_config.LIVE_HEAD_EVERY_N_FRAMES = original

    ok = True
    ok &= check("head model ran on frames 0 and 2 only", init.detect_calls == 2,
                f"got {init.detect_calls}")
    ok &= check("all three pose passes ran", len(classifier.calls) == 3)
    ok &= check(
        "cached boxes were passed on the throttled frame",
        classifier.calls[1]["head_boxes"] == heads,
        f"got {classifier.calls[1]['head_boxes']}",
    )
    return ok


def test_no_initializer_is_legacy() -> bool:
    print()
    print("3. missing/unloaded head model keeps the legacy path")
    classifier = FakeClassifier()
    worker = make_worker(None, classifier)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    iw.InferenceWorker._classify_frame(worker, frame)

    ok = True
    ok &= check("classifier still runs", len(classifier.calls) == 1)
    ok &= check("no head boxes are invented",
                classifier.calls[0]["head_boxes"] == [])

    unloaded = FakeInitializer([])
    unloaded.loaded = False
    classifier2 = FakeClassifier()
    worker2 = make_worker(unloaded, classifier2)
    iw.InferenceWorker._classify_frame(worker2, frame)
    ok &= check("unloaded initialiser is not called", unloaded.detect_calls == 0)
    ok &= check("legacy classify still receives no boxes",
                classifier2.calls[0]["head_boxes"] == [])
    return ok


class FakeSignal:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def emit(self, payload: dict) -> None:
        self.payloads.append(payload)


class FakeRegistry:
    def __init__(self) -> None:
        self.seat_count = 1

    def as_list(self) -> list[dict]:
        return [{"box": [100, 100, 180, 180], "seat_id": "Seat_01"}]


class FakeInitForClassroom:
    complete = True
    frames_seen = perf_config.CLASSROOM_INIT_FRAMES
    frames_target = perf_config.CLASSROOM_INIT_FRAMES

    def __init__(self) -> None:
        self.registry = FakeRegistry()
        self.observe_calls = 0

    def observe(self, frame: np.ndarray) -> None:
        self.observe_calls += 1

    def finish(self) -> FakeRegistry:
        return self.registry

    def as_info(self) -> dict:
        return {
            "locked": True,
            "seated_count": 1,
            "frames_seen": self.frames_seen,
            "frames_target": self.frames_target,
            "in_progress": False,
            "error": "",
        }


class FakeSeatClassifier:
    def __init__(self) -> None:
        self.registry = None

    def set_seat_anchors(self, registry) -> int:
        self.registry = registry
        return int(getattr(registry, "seat_count", 0))


def test_initializer_retained() -> bool:
    print()
    print("4. SCUT-HEAD initialiser survives the 30-frame phase")
    fake_init = FakeInitForClassroom()
    classifier = FakeSeatClassifier()
    signal = FakeSignal()
    fake_self = types.SimpleNamespace(
        _running=True,
        _camera=types.SimpleNamespace(is_image_replay=False),
        _classifier=classifier,
        _classroom_info={},
        _head_initializer=None,
        _live_heads=[{"stale": True}],
        _live_head_counter=7,
        classroom_initialized=signal,
    )

    original = iw.HeadAnchorInitializer
    iw.HeadAnchorInitializer = lambda: fake_init  # type: ignore[assignment]
    try:
        info = iw.InferenceWorker._initialise_classroom(fake_self)
    finally:
        iw.HeadAnchorInitializer = original  # type: ignore[assignment]

    ok = True
    ok &= check("initializer instance retained on the worker",
                fake_self._head_initializer is fake_init)
    ok &= check("classifier received the locked registry",
                classifier.registry is fake_init.registry)
    ok &= check("live head cache reset after init",
                fake_self._live_heads == [] and fake_self._live_head_counter == 0)
    ok &= check("init info emitted", signal.payloads == [info])
    ok &= check("seated count reported", info.get("seated_count") == 1)
    return ok


def main() -> int:
    print("=" * 68)
    print("Unified SCUT-HEAD + pose cycle QA")
    print("=" * 68)
    ok = True
    ok &= test_detect_then_classify()
    ok &= test_head_throttling()
    ok &= test_no_initializer_is_legacy()
    ok &= test_initializer_retained()
    print()
    print("=" * 68)
    if ok and FAIL == 0:
        print(f"ALL PASS ({PASS} checks)")
        return 0
    print(f"FAILURES PRESENT ({FAIL} failed, {PASS} passed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
