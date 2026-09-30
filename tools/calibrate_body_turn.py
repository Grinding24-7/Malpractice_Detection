"""Measure the body-turn ratio distribution on a known-normal clip.

Body turn (P10/P11) fires at 40% on a normally-seated student.  The rule is
`nose_offset_from_shoulder_midpoint / shoulder_width`, so this reports the
normal band and the cut that clears it.

Usage:
    python tools/calibrate_body_turn.py [clip] [--frames N]
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

logging.disable(logging.WARNING)

from desktop_app.core import perf_config  # noqa: E402
from desktop_app.core.pose_classifier import (  # noqa: E402
    MP_LEFT_SHOULDER, MP_NOSE, MP_RIGHT_SHOULDER, PoseClassifier,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "clip", nargs="?", type=pathlib.Path,
        default=_ROOT / "backend" / "sample_exam.mp4",
    )
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    c = PoseClassifier()
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        print(f"cannot open {args.clip}")
        return 1

    ratios, widths = [], []
    read = 0
    while read < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        if (read - 1) % args.stride:
            continue
        boxes, _ = c._yolo_detect(frame)
        if not boxes:
            continue
        lms = c._mediapipe_landmarks(frame, boxes)
        if not lms:
            continue
        L = lms[0]
        if min(L[MP_NOSE][2], L[MP_LEFT_SHOULDER][2], L[MP_RIGHT_SHOULDER][2]) <= 0.5:
            continue
        w = abs(L[MP_LEFT_SHOULDER][0] - L[MP_RIGHT_SHOULDER][0])
        if w < perf_config.MIN_SCALE_PX:
            continue
        smx = (L[MP_LEFT_SHOULDER][0] + L[MP_RIGHT_SHOULDER][0]) / 2.0
        ratios.append((L[MP_NOSE][0] - smx) / w)
        widths.append(w)
    cap.release()

    if not ratios:
        print("no usable samples")
        return 1

    a = np.array(ratios)
    w = np.array(widths)
    print(f"=== {args.clip.name}: {len(a)} samples ===")
    print(f"  shoulder width px : median {np.median(w):.0f} "
          f"(p05 {np.percentile(w,5):.0f} p95 {np.percentile(w,95):.0f})")
    print(f"  turn ratio       : median {np.median(a):+.3f}")
    print(f"                     p05 {np.percentile(a,5):+.3f} "
          f"p95 {np.percentile(a,95):+.3f}")
    print(f"                     min {a.min():+.3f}  max {a.max():+.3f}")
    print(f"                     |ratio| p95 {np.percentile(np.abs(a),95):.3f}")
    print()
    print("  NOTE: an absolute cut on this ratio is not usable.  Turning the")
    print("  torso foreshortens the shoulders, so the ratio scales itself up")
    print("  at the same moment a real turn happens.  The numbers above show")
    print("  how little the false-positive rate improves as the cut rises.")
    print()

    # Robust z-score, matching _TemporalBuffer.turn_deviation_z.
    baseline = a[0]
    mad = 0.0
    zs = []
    for v in a:
        if len(zs) < 20:
            zs.append(0.0)
        else:
            zs.append(abs(v - baseline) / max(mad, 0.05))
        baseline += (v - baseline) / 60.0
        mad += (abs(v - baseline) - mad) / 60.0
    z = np.array(zs[20:])

    print(f"  robust z-score of the normal band (after 20-frame warmup):")
    print(f"     median {np.median(z):.2f}  p95 {np.percentile(z,95):.2f}  "
          f"max {z.max():.2f}")
    print()
    print(f"  current BODY_TURN_Z = {perf_config.BODY_TURN_Z} "
          f"-> fires on {100*np.mean(z > perf_config.BODY_TURN_Z):.1f}%")
    print("  candidate cuts and resulting false-positive rate:")
    for x in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        print(f"     {x:5.1f} sigma  -> {100*np.mean(z > x):5.1f}%")
    print()
    print("  A genuine torso turn must sit above this band.  Validate on footage")
    print("  that actually contains a turn before trusting a cut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
