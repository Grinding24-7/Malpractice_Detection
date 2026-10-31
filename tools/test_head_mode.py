"""Regression checks for the SCUT-HEAD head-occupancy desktop mode.

SCUT-HEAD has head bounding boxes, not poses or behaviour labels.  These
checks pin the parts of the head-mode integration that are easy to break:

1. replay lists resolve to the same images the GT manifest describes;
2. annotated boxes are attached only to their own image;
3. the HUD draws magenta GT boxes without mutating the frame;
4. results remain non-behavioural and carry the GT count;
5. the local one-class head checkpoint loads and reports head classes.

The dataset and trained weights are local, git-ignored assets.  Each section
skips instead of failing when they are absent, so a clean checkout can run
this script safely.

Usage:
    python tools/test_head_mode.py
"""

from __future__ import annotations

import logging
import pathlib
import sys

import cv2
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.disable(logging.WARNING)

from desktop_app.core.camera_thread import CameraThread  # noqa: E402
from desktop_app.core.head_detector import (  # noqa: E402
    _SCUT_HEAD_MANIFEST,
    HeadCountWorker,
    render_head_hud,
    resolve_head_model,
)

_REPLAY_LIST = (
    _ROOT / "datasets" / "SCUT_HEAD" / "yolo" / "replay" / "A_test.txt"
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return condition


def skip(label: str, reason: str) -> None:
    print(f"  [SKIP] {label} -- {reason}")


def magenta_pixels(frame: np.ndarray) -> int:
    return int(np.count_nonzero(
        (frame[:, :, 0] == 255)
        & (frame[:, :, 1] == 0)
        & (frame[:, :, 2] == 255)
    ))


def test_replay_gt_contract() -> bool:
    """1. replay paths from the YOLO export map to annotated manifest rows."""
    print("1. replay list -> SCUT-HEAD ground truth")
    if not _SCUT_HEAD_MANIFEST.is_file():
        skip("manifest lookup", "datasets/SCUT_HEAD/manifest.csv absent")
        return True

    worker = HeadCountWorker(camera=None)
    mapping = worker._load_ground_truth()
    ok = check(f"manifest loaded ({len(mapping)} images)", bool(mapping))
    if not mapping:
        return ok

    annotated = [
        (key, boxes) for key, boxes in mapping.items() if boxes
    ]
    ok &= check(
        "manifest contains annotated images",
        bool(annotated),
        f"{len(annotated)} annotated",
    )
    if not annotated:
        return ok

    sample_path = annotated[0][0]
    if _REPLAY_LIST.is_file():
        camera = CameraThread(source=-1)
        replay_paths = camera._collect_image_paths(_REPLAY_LIST)
        ok &= check(
            f"A_test replay list resolves ({len(replay_paths)} images)",
            len(replay_paths) > 0,
        )
        ok &= check(
            "replay list has no duplicate images",
            len(replay_paths) == len({str(p).lower() for p in replay_paths}),
        )
        if replay_paths:
            sample_path = str(replay_paths[0])
        for path in replay_paths[:5]:
            boxes = worker._ground_truth_for(str(path))
            ok &= check(
                f"replay image has GT ({path.name})",
                boxes is not None and len(boxes) > 0,
                f"{0 if boxes is None else len(boxes)} boxes",
            )
    else:
        skip("A_test replay list", "yolo/replay/A_test.txt absent")
        key, boxes = annotated[0]
        ok &= check(
            "manifest image has GT",
            worker._ground_truth_for(key) == boxes,
        )

    missing = worker._ground_truth_for(
        str(_ROOT / "datasets" / "SCUT_HEAD" / "not_an_image.jpg")
    )
    ok &= check("unknown image has no GT", missing is None)

    first_boxes = worker._ground_truth_for(sample_path)
    if first_boxes:
        ok &= check(
            "GT boxes are four-coordinate",
            all(len(box) == 4 for box in first_boxes),
        )
        ok &= check(
            "GT coordinates are non-negative",
            all(float(v) >= 0 for box in first_boxes for v in box),
        )
    return ok


def test_hud_overlay() -> bool:
    """2. the HUD draws GT boxes in magenta and leaves the input intact."""
    print()
    print("2. GT overlay rendering")
    if not _SCUT_HEAD_MANIFEST.is_file():
        skip("HUD overlay", "SCUT-HEAD manifest absent")
        return True

    worker = HeadCountWorker(camera=None)
    annotated = [
        (key, boxes)
        for key, boxes in worker._load_ground_truth().items()
        if boxes
    ]
    if not annotated:
        skip("HUD overlay", "no annotated SCUT-HEAD images")
        return True

    key, gt_boxes = annotated[0]
    frame = cv2.imread(key)
    if frame is None:
        skip("HUD overlay", f"cannot read {key}")
        return True

    before = frame.copy()
    result = {
        "heads": [{"box": [10, 10, 40, 40], "confidence": 0.8}],
        "head_count": 1,
        "ground_truth_heads": gt_boxes,
        "ground_truth_count": len(gt_boxes),
        "source_dead": False,
        "error": "",
        "image_count": 500,
        "image_index": 0,
    }
    overlay = render_head_hud(frame, result, 2.0, 5.0)
    ok = check("HUD returns a frame", overlay is not None and overlay.shape == frame.shape)
    ok &= check("HUD does not mutate its input", np.array_equal(frame, before))
    ok &= check(
        "GT boxes are drawn in magenta",
        magenta_pixels(overlay) > 0,
        f"{magenta_pixels(overlay)} magenta pixels",
    )

    no_gt = render_head_hud(
        frame,
        dict(result, ground_truth_heads=[], ground_truth_count=None),
        2.0,
        5.0,
    )
    ok &= check(
        "no magenta when a source has no GT",
        magenta_pixels(no_gt) == 0,
        f"{magenta_pixels(no_gt)} magenta pixels",
    )
    return ok


def test_result_contract() -> bool:
    """3. head results carry GT counts and make no pose/behaviour claim."""
    print()
    print("3. head result contract")
    if not _SCUT_HEAD_MANIFEST.is_file():
        skip("result contract", "SCUT-HEAD manifest absent")
        return True

    worker = HeadCountWorker(camera=None)
    annotated = [
        boxes for boxes in worker._load_ground_truth().values() if boxes
    ]
    if not annotated:
        skip("result contract", "no annotated SCUT-HEAD images")
        return True

    results: list[dict] = []
    frames: list[np.ndarray] = []
    worker.result_ready.connect(results.append)
    worker.frame_processed.connect(frames.append)

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    heads = [{"box": [20, 20, 80, 80], "confidence": 0.75}]
    gt = annotated[0][:2]
    worker._emit_result(
        frame, heads, image_index=7, ground_truth=gt,
    )

    ok = check("one result emitted", len(results) == 1)
    ok &= check("one processed frame emitted", len(frames) == 1)
    if not results:
        return ok

    result = results[0]
    ok &= check("mode is head", result.get("mode") == "head")
    ok &= check(
        "pose id is HEAD, not a P-code",
        result.get("pose_id") == "HEAD",
        str(result.get("pose_id")),
    )
    ok &= check(
        "no pose/behaviour payload",
        result.get("keypoints") == []
        and result.get("objects_detected") == []
        and result.get("head_pose") == {},
    )
    ok &= check(
        "detector and GT counts are separate",
        result.get("head_count") == 1
        and result.get("ground_truth_count") == len(gt),
        f"heads={result.get('head_count')} gt={result.get('ground_truth_count')}",
    )
    ok &= check(
        "image index is preserved",
        result.get("image_index") == 7,
        str(result.get("image_index")),
    )
    return ok


def test_head_model_loads() -> bool:
    """4. the local checkpoint resolves and loads as a head detector."""
    print()
    print("4. head checkpoint load")
    path = resolve_head_model()
    if path is None or not path.is_file():
        skip("head checkpoint", "no local head weights")
        return True

    ok = check(f"resolved weights ({path.name})", path.is_file())
    worker = HeadCountWorker(camera=None, model_path=path)
    worker._load_model()
    ok &= check("model loaded", worker._model is not None, worker._error)
    if worker._model is not None:
        ok &= check(
            "model exposes a head class",
            bool(worker._head_ids),
            str(sorted(worker._head_ids)),
        )
    return ok


def main() -> int:
    print(f"SCUT-HEAD manifest: {_SCUT_HEAD_MANIFEST}")
    print()
    ok = True
    ok &= test_replay_gt_contract()
    ok &= test_hud_overlay()
    ok &= test_result_contract()
    ok &= test_head_model_loads()
    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
