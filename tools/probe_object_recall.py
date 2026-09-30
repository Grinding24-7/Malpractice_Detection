"""Probe what the object detector actually sees, at any confidence.

The desk region is now 49% of the frame (was 20%), and the old region was
mostly re-detecting the student's own torso as class 0 "person".  Neither
number tells us whether the absence of desk objects is a real property of
the footage or a detector/threshold problem, so this runs the raw model over
the desk region with the confidence floor effectively removed and reports
every class it proposes.
"""

from __future__ import annotations

import collections
import logging
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from ultralytics import YOLO  # noqa: E402

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import (  # noqa: E402
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, OBJ_CLASS_MAP, PoseClassifier,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLIP = _ROOT / "backend" / "sample_exam.mp4"
_PROBE_CONF = 0.01


def main() -> int:
    model = YOLO(str(_ROOT / "backend" / "yolo11n.pt"))
    names = model.names

    c = PoseClassifier()
    cap = cv2.VideoCapture(str(_CLIP))
    all_classes: collections.Counter[str] = collections.Counter()
    desk_classes: collections.Counter[str] = collections.Counter()
    suspect: list[tuple[str, float]] = []
    n = 0

    while n < 60:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % 2:
            continue
        boxes, _ = c._yolo_detect(frame)
        lms = c._mediapipe_landmarks(frame, boxes)
        c._last_frame_shape = frame.shape
        if lms:
            c._last_shoulder_width = abs(
                lms[0][MP_LEFT_SHOULDER][0] - lms[0][MP_RIGHT_SHOULDER][0]
            )
        if boxes:
            c._last_person_bbox = list(boxes[0])

        # Whole frame, so nothing is hidden by the region choice.
        r = model(frame, imgsz=640, conf=_PROBE_CONF, verbose=False)[0]
        if r.boxes is not None:
            for cls, conf in zip(
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
            ):
                all_classes[names.get(cls, str(cls))] += 1

        # Desk region only.
        x1, y1, x2, y2 = c._desk_region(frame.shape)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        r2 = model(crop, imgsz=perf_config.OBJ_IMGSZ, conf=_PROBE_CONF,
                   verbose=False)[0]
        if r2.boxes is not None:
            for cls, conf in zip(
                r2.boxes.cls.cpu().numpy().astype(int),
                r2.boxes.conf.cpu().numpy(),
            ):
                name = names.get(cls, str(cls))
                desk_classes[name] += 1
                if cls in OBJ_CLASS_MAP:
                    suspect.append((OBJ_CLASS_MAP[cls], float(conf)))
    cap.release()

    print(f"frames probed: {n}")
    print(f"probe confidence floor: {_PROBE_CONF}  "
          f"(production floor: {perf_config.OBJ_CONF_THRESHOLD})")
    print()
    print("whole frame:")
    for name, cnt in all_classes.most_common(12):
        print(f"   {name:<18} {cnt}")
    print()
    print("desk region only:")
    for name, cnt in desk_classes.most_common(12):
        print(f"   {name:<18} {cnt}")
    print()
    if suspect:
        arr = np.array([s[1] for s in suspect])
        print(f"suspicious-class proposals in the desk region: {len(suspect)}")
        print(f"   max confidence {arr.max():.3f}  "
              f"median {np.median(arr):.3f}")
        for label, conf in sorted(suspect, key=lambda s: -s[1])[:10]:
            print(f"   {label:<12} {conf:.3f}")
        print()
        print(f"production floor is {perf_config.OBJ_CONF_THRESHOLD}, so "
              f"{100*np.mean(arr >= perf_config.OBJ_CONF_THRESHOLD):.0f}% "
              "of these would be reported.")
    else:
        print("no laptop/phone/book proposals anywhere in the desk region,")
        print("even at conf 0.01. The absence of objects is a property of")
        print("the footage, not the threshold: this clip genuinely shows no")
        print("detectable desk object.")

    print()
    print("classes COCO-80 cannot express, which are the ones that matter")
    print("for a written exam:")
    for missing in ("paper", "pen", "pencil", "notebook", "desk", "exam sheet"):
        print(f"   {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
