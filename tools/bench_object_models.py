"""Benchmark object-detection weight sizes: does a bigger model help here?

The user approved comparing a larger detector against yolo11n before
adopting it.  The question is not "is yolo11m more accurate in general" -- it
is whether it finds desk objects this detector currently misses, at a latency
the 10 FPS budget can absorb.

Reports, per model:
  - latency at the production imgsz
  - every class proposed at a low probe confidence
  - whether any laptop/phone/book appears

A model that is only slower is not adopted.  If neither finds objects, the
limit is the footage, and the honest conclusion is that a bigger model cannot
be justified or rejected on this clip.
"""

from __future__ import annotations

import argparse
import collections
import logging
import pathlib
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import (  # noqa: E402
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, OBJ_CLASS_MAP, PoseClassifier,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLIP = _ROOT / "backend" / "sample_exam.mp4"
_BACKEND = _ROOT / "backend"

# COCO classes that would actually indicate a problem in an exam.
_SUSPECT_IDS = set(OBJ_CLASS_MAP)


def desk_crops(clip: pathlib.Path, frames: int, stride: int):
    """Yield the desk region for each analysed frame, plus the frame itself."""
    c = PoseClassifier()
    cap = cv2.VideoCapture(str(clip))
    n = 0
    while n < frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % stride:
            continue
        boxes, _ = c._yolo_detect(frame)
        lms = c._mediapipe_landmarks(frame, boxes)
        c._last_frame_shape = frame.shape
        if lms:
            c._last_shoulder_width = abs(
                lms[0][MP_LEFT_SHOULDER][0] - lms[0][MP_RIGHT_SHOULDER][0]
            )
        c._last_person_bbox = list(boxes[0]) if boxes else None
        x1, y1, x2, y2 = c._desk_region(frame.shape)
        crop = frame[y1:y2, x1:x2]
        if crop.size:
            yield crop, frame
    cap.release()


def benchmark(model_path: pathlib.Path, crops, frames_list, probe_conf):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names

    hot = [cv2.resize(c, (perf_config.OBJ_IMGSZ, perf_config.OBJ_IMGSZ))
           for c in frames_list[:8]]
    for img in hot[:3]:  # warm up
        model(img, imgsz=perf_config.OBJ_IMGSZ, conf=0.25, verbose=False)

    latencies = []
    for crop in frames_list:
        t0 = time.perf_counter()
        model(crop, imgsz=perf_config.OBJ_IMGSZ, conf=0.25, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    classes: collections.Counter[str] = collections.Counter()
    suspect_conf: list[float] = []
    for crop, full in crops:
        r = model(crop, imgsz=perf_config.OBJ_IMGSZ, conf=probe_conf,
                  verbose=False)[0]
        if r.boxes is None:
            continue
        for cls, conf in zip(
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
        ):
            classes[names.get(cls, str(cls))] += 1
            if cls in _SUSPECT_IDS:
                suspect_conf.append(float(conf))

    return {
        "model": model_path.name,
        "size_mb": round(model_path.stat().st_size / 1e6, 1),
        "median_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0,
        "classes": dict(classes.most_common(8)),
        "suspect_proposals": len(suspect_conf),
        "suspect_max": round(max(suspect_conf), 3) if suspect_conf else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--probe-conf", type=float, default=0.01)
    parser.add_argument("--models", nargs="*", default=["yolo11n.pt"])
    args = parser.parse_args()

    if not _CLIP.is_file():
        print(f"missing {_CLIP}")
        return 1

    print(f"clip: {_CLIP.name}")
    print(f"desk region imgsz: {perf_config.OBJ_IMGSZ}   "
          f"probe conf: {args.probe_conf}")
    print()

    crops = list(desk_crops(_CLIP, args.frames, args.stride))
    if not crops:
        print("no desk crops extracted")
        return 1
    frames_list = [c for c, _ in crops]
    print(f"analysed {len(frames_list)} desk crops\n")

    results = []
    for name in args.models:
        path = _BACKEND / name
        if not path.is_file():
            print(f"[skip] {name} not present in backend/")
            continue
        print(f"running {name} ...")
        results.append(benchmark(path, crops, frames_list, args.probe_conf))

    if not results:
        print("no models available to compare")
        return 1

    print()
    print(f"{'model':<16} {'MB':>6} {'median':>8} {'p95':>8} "
          f"{'suspect':>8} {'max conf':>9}")
    print("-" * 62)
    for r in results:
        print(f"{r['model']:<16} {r['size_mb']:>6} {r['median_ms']:>7}ms "
              f"{r['p95_ms']:>7}ms {r['suspect_proposals']:>8} "
              f"{r['suspect_max']:>9}")
    print()
    for r in results:
        print(f"{r['model']} classes: {r['classes']}")
    print()

    if len(results) > 1:
        base, cand = results[0], results[-1]
        slowdown = cand["median_ms"] / max(base["median_ms"], 1e-6)
        gain = cand["suspect_proposals"] - base["suspect_proposals"]
        # What the extra cost means against the per-frame budget.
        budget_ms = 1000.0 / 10.0  # TARGET_INFERENCE_FPS = 10
        share = cand["median_ms"] / budget_ms
        print(f"{cand['model']} vs {base['model']}: "
              f"{slowdown:.1f}x the latency, {gain:+d} suspicious-class "
              f"detections")
        print(f"   at {cand['median_ms']:.0f} ms it alone consumes "
              f"{share:.0%} of the {budget_ms:.0f} ms/frame budget "
              f"(and is throttled to 1 frame in "
              f"{perf_config.OBJ_DETECT_EVERY_N_FRAMES})")
        if gain <= 0:
            print("VERDICT: no recall gain on this footage, at materially")
            print("         higher cost -- do not adopt.")
        elif slowdown > 2:
            print("VERDICT: recall gain costs more than 2x latency. Only")
            print("         worth it if desk-object recall is a hard")
            print("         requirement, and it should stay behind the")
            print("         anomaly gate rather than run every frame.")
        else:
            print("VERDICT: recall gain within budget -- adopt.")
    else:
        print("Only one model present, so no comparison is possible.")
        if results[0]["suspect_proposals"] == 0:
            print("No desk objects are detectable in this clip at any")
            print("confidence, so no model size can be justified or rejected")
            print("here.  This needs footage with a visible desk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
