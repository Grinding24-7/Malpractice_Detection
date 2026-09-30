"""Measure object detection over a clip, with spatial context.

Reports what was found, where it was relative to the student, and -- most
importantly -- how many detections could actually escalate versus how many
were merely present in the room.  A detector that finds more objects is only
an improvement if the escalations get more precise too, so both numbers are
reported side by side.

Uses `force=True` on the detector so the throttle does not hide detections
from an offline measurement; persistence is disabled for the same reason
(every frame is a fresh detection here).

Usage:
    python tools/eval_objects.py [clip] [--frames N] [--stride N] [--json out]
    python tools/eval_objects.py --compare     # old region vs new region
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import PoseClassifier  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Relations that make a detection actionable against the student.
_ACTIONABLE = {"in_hand", "on_desk", "reachable"}


def _legacy_desk_region(frame_shape, box):
    """The pre-rewrite region: the person's own box, lower 60%."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    h_person = y2 - y1
    cx1, cy1 = max(0, x1), max(0, y1 + int(h_person * 0.4))
    cx2, cy2 = min(w, x2), min(h, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def run_clip(
    clip: pathlib.Path,
    frames: int,
    stride: int,
    compare: bool = False,
) -> dict:
    c = PoseClassifier()
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counts: collections.Counter[str] = collections.Counter()
    relations: collections.Counter[str] = collections.Counter()
    actionable = 0
    escalations = 0
    region_areas: list[float] = []
    legacy_counts: collections.Counter[str] = collections.Counter()
    max_conf = 0.0
    durations: list[float] = []
    frames_with_person = 0
    classified = 0

    read = 0
    while read < frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        if (read - 1) % stride:
            continue
        classified += 1

        boxes, _ = c._yolo_detect(frame)
        lms = c._mediapipe_landmarks(frame, boxes)
        c._last_frame = frame
        c._last_frame_shape = frame.shape
        c._last_frame_width = frame.shape[1]
        c._last_landmarks = lms[0] if lms else None
        c._last_person_bbox = list(boxes[0]) if boxes else None
        c._last_shoulder_width = 0.0
        if c._last_landmarks and len(c._last_landmarks) > 12:
            c._last_shoulder_width = abs(
                c._last_landmarks[11][0] - c._last_landmarks[12][0]
            )
        if boxes:
            frames_with_person += 1

        x1, y1, x2, y2 = c._desk_region(frame.shape)
        region_areas.append(((x2 - x1) * (y2 - y1)) /
                            float(width * height))

        t0 = time.perf_counter()
        dets = c._detect_objects(frame, boxes, force=True)
        durations.append((time.perf_counter() - t0) * 1000.0)

        for d in dets:
            counts[d["label"]] += 1
            relations[d.get("relation", "?")] += 1
            max_conf = max(max_conf, d["confidence"])
            if d.get("in_hand") or d.get("reachable"):
                actionable += 1
            # What the verdict would actually become.
            escalated_pose, _ = c._escalate_from_objects("P01", 0.0, [d])
            if escalated_pose != "P01":
                escalations += 1

        if compare and boxes:
            legacy = _legacy_desk_region(frame.shape, boxes[0])
            if legacy is not None:
                cx1, cy1, cx2, cy2 = legacy
                crop = frame[cy1:cy2, cx1:cx2]
                try:
                    r = c._obj_model(
                        crop, imgsz=perf_config.OBJ_IMGSZ,
                        conf=perf_config.OBJ_CONF_THRESHOLD, verbose=False,
                    )[0]
                    if r.boxes is not None:
                        for cls in r.boxes.cls.cpu().numpy().astype(int):
                            legacy_counts[
                                {63: "laptop", 67: "cell_phone",
                                 73: "book"}.get(cls, f"class_{cls}")
                            ] += 1
                except Exception:
                    pass

    cap.release()

    return {
        "clip": clip.name,
        "resolution": f"{width}x{height}",
        "frames_classified": classified,
        "frames_with_person": frames_with_person,
        "mean_region_fraction": round(float(np.mean(region_areas)), 4)
        if region_areas else 0.0,
        "counts": dict(counts),
        "relations": dict(relations),
        "total_detections": sum(counts.values()),
        "actionable": actionable,
        "escalating": escalations,
        "max_confidence": round(max_conf, 3),
        "median_detect_ms": round(float(np.median(durations)), 1)
        if durations else 0.0,
        "legacy_counts": dict(legacy_counts),
    }


def print_report(r: dict, compare: bool) -> None:
    print(f"\n=== {r['clip']} {r['resolution']} ===")
    print(f"  frames classified   : {r['frames_classified']} "
          f"({r['frames_with_person']} with a person)")
    print(f"  desk region covers  : {r['mean_region_fraction']:.1%} of frame "
          f"(was ~20%)")
    print(f"  detections          : {r['total_detections']}")
    print(f"  actionable (in hand / within reach): {r['actionable']}")
    print(f"  would escalate      : {r['escalating']}")
    print(f"  max confidence      : {r['max_confidence']}")
    print(f"  median detect cost  : {r['median_detect_ms']} ms")
    if compare:
        print(f"  legacy region found : {sum(r['legacy_counts'].values())} "
              f"{r['legacy_counts']}")
    if r["counts"]:
        print("  by label:")
        for label, n in r["counts"].items():
            print(f"     {label:<12} {n}")
        print("  by spatial relation:")
        for rel, n in r["relations"].items():
            marker = " <-- actionable" if rel in _ACTIONABLE else ""
            print(f"     {rel:<12} {n}{marker}")
    else:
        print("  no objects detected at conf "
              f"{perf_config.OBJ_CONF_THRESHOLD}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", nargs="?", type=pathlib.Path)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--compare", action="store_true",
                        help="also run the pre-rewrite region")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    clips = [args.clip] if args.clip else [
        _ROOT / "backend" / "sample_exam.mp4",
    ]
    clips = [c for c in clips if c.is_file()]
    if not clips:
        print("no clips found")
        return 1

    results = [run_clip(c, args.frames, args.stride, args.compare)
               for c in clips]
    for r in results:
        print_report(r, args.compare)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
