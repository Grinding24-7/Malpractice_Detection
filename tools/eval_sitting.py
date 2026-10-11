"""Measure false positives on a known-normal clip.

Replays a video of a student sitting normally and reports the pose
distribution.  The headline number is the share of frames reported as
"P01 Normal Sitting": everything else is a false positive.

Usage:
    python tools/eval_sitting.py [clip] [--frames N] [--json out.json]

With no argument it evaluates every clip it can find, so a regression can be
compared against a committed baseline.
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

from desktop_app.core.pose_classifier import POSE_MAP, PoseClassifier  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Poses that are correct answers even on a "normal" clip: an empty seat or an
# extra person are genuine findings, not noise.
_ALWAYS_VALID = {"P16", "P35"}

# Poses that assert nothing about a student.  They must be excluded from both
# the false-positive rate and the P01 share, because counting them as misses
# would make the warmup and the dead-source handling look like regressions
# when they are the correct behaviour.
_NOT_A_VERDICT = {"INITIALIZING", "NO_SIGNAL"}


def default_clips() -> list[pathlib.Path]:
    """Find candidate clips, preferring the real exam footage."""
    candidates = [
        _ROOT / "backend" / "sample_exam.mp4",
        _ROOT / "backend" / "1_hour_exam_test.mp4",
        _ROOT / "assets" / "test_feed.mp4",
    ]
    return [p for p in candidates if p.is_file()]


def evaluate(
    clip: pathlib.Path,
    frames: int = 240,
    stride: int = 2,
    classifier: PoseClassifier | None = None,
) -> dict:
    """Replay *clip* and summarise the pose distribution."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")

    # Capture geometry before the handle is released.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    classifier = classifier or PoseClassifier()
    dist: collections.Counter[str] = collections.Counter()
    yaws: list[float] = []
    pitches: list[float] = []
    luma: list[float] = []
    durations: list[float] = []

    read = 0
    while read < frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        luma.append(float(frame.mean()))

        if (read - 1) % stride:
            continue

        t0 = time.perf_counter()
        result = classifier.classify(frame)
        durations.append((time.perf_counter() - t0) * 1000.0)

        dist[result["pose_id"]] += 1
        head = result.get("head_pose") or {}
        if head:
            yaws.append(float(head.get("yaw", 0.0)))
            pitches.append(float(head.get("pitch", 0.0)))

    cap.release()
    total = sum(dist.values()) or 1
    # Frames that carried no verdict about a student are excluded from the
    # headline rates, so the numbers describe real decisions only.
    scored = sum(n for pose, n in dist.items() if pose not in _NOT_A_VERDICT)
    scored = scored or 1

    false_positive = sum(
        n for pose, n in dist.items()
        if pose != "P01" and pose not in _ALWAYS_VALID
        and pose not in _NOT_A_VERDICT
    )
    held = sum(n for pose, n in dist.items() if pose in _NOT_A_VERDICT)

    return {
        "clip": clip.name,
        "resolution": f"{width}x{height}",
        "frames_read": read,
        "frames_classified": total,
        "frames_scored": scored,
        "frames_no_verdict": held,
        "mean_luma": round(float(np.mean(luma)), 1) if luma else 0.0,
        "is_black": bool(luma and np.mean(luma) < 2.0),
        "p01_share": round(100.0 * dist["P01"] / scored, 1),
        "false_positive_rate": round(100.0 * false_positive / scored, 1),
        "median_latency_ms": round(float(np.median(durations)), 1) if durations else 0.0,
        "distribution": {k: v for k, v in dist.most_common()},
        "named": {f"{k} {POSE_MAP.get(k, '?')}": v for k, v in dist.most_common()},
        "yaw": _spread(yaws),
        "pitch": _spread(pitches),
    }


def _spread(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "median": round(float(np.median(arr)), 1),
        "p05": round(float(np.percentile(arr, 5)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", nargs="?", type=pathlib.Path)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--json", type=pathlib.Path, help="write results here")
    args = parser.parse_args()

    clips = [args.clip] if args.clip else default_clips()
    if not clips:
        print("no clips found")
        return 1

    classifier = PoseClassifier()  # load models once, reuse across clips
    results = [evaluate(c, args.frames, args.stride, classifier) for c in clips]

    for r in results:
        flag = "  [BLACK CLIP]" if r["is_black"] else ""
        print(f"\n=== {r['clip']} {r['resolution']}{flag} ===")
        print(f"  frames classified : {r['frames_classified']} "
              f"(mean luma {r['mean_luma']})")
        if r["frames_no_verdict"]:
            print(f"  no verdict        : {r['frames_no_verdict']} "
                  f"(warmup / no image, excluded below)")
        print(f"  frames scored     : {r['frames_scored']}")
        print(f"  P01 Normal Sitting: {r['p01_share']}%")
        print(f"  FALSE POSITIVES  : {r['false_positive_rate']}%")
        print(f"  median latency   : {r['median_latency_ms']} ms")
        if r["yaw"]:
            print(f"  yaw   median {r['yaw']['median']:>7}  "
                  f"p05 {r['yaw']['p05']}  p95 {r['yaw']['p95']}")
        if r["pitch"]:
            print(f"  pitch median {r['pitch']['median']:>7}  "
                  f"p05 {r['pitch']['p05']}  p95 {r['pitch']['p95']}")
        print("  distribution:")
        for name, n in r["named"].items():
            pct = 100.0 * n / max(1, r["frames_classified"])
            print(f"     {name:<28} {n:>4}  {pct:5.1f}%")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
