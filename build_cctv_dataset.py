"""
build_cctv_dataset.py — Week 3: build a scale-normalized pose feature dataset
from public CCTV/classroom images and videos.

Pipeline:
    1. Collects media from `datasets/` (downloaded by download_public_dataset.py)
       and the local `training_videos/` directory.
    2. Runs YOLO11-pose ('yolo11n-pose.pt') on every image / sampled video frame.
    3. Because CCTV feeds contain multiple students per frame, iterates over ALL
       detected candidate keypoint sets and extracts a scale-normalized feature
       vector for each.
    4. Maps ground-truth folder names / categories to target labels:
            0 = Normal / Writing / Facing forward
            1 = Looking Away / Head Turn (Left/Right)
            2 = Leaning / Bending / Slouching
    5. Appends every feature row to 'pose_dataset_cctv.csv' and prints per-class
       summary statistics.

Usage:
    uv run python build_cctv_dataset.py
    uv run python build_cctv_dataset.py --frame-step 5 --out backend/pose_dataset_cctv.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from feature_extractor import extract_normalized_pose_features  # noqa: E402

SOURCE_DIRS = [ROOT / "datasets", ROOT / "training_videos"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEDIA_EXTS = IMAGE_EXTS | {".mp4", ".avi", ".mov", ".mkv"}

DATASET_HEADERS = [
    "ear_ratio",
    "norm_vertical_drop",
    "shoulder_angle",
    "norm_nose_ear_drop",
    "nose_conf",
    "l_ear_conf",
    "r_ear_conf",
    "label",
]

CLASS0_KEYWORDS = (
    "normal", "writing", "facing", "forward", "front", "good", "reading",
    "listen", "sitting", "sit", "look_forward", "facing_forward", "straight",
)
CLASS1_KEYWORDS = (
    "looking", "look", "away", "turn", "turning", "around", "left", "right",
    "side", "gaze", "glance", "share", "sharing", "mobile", "phone", "wander",
)
CLASS2_KEYWORDS = (
    "lean", "leaning", "bend", "bending", "slouch", "slouching", "hunch",
    "hunching", "lying", "sleep", "sleeping", "bowing", "bend_over", "stoop",
    "tilt",
)
# Most-specific posture signals win first: lean/slouching, then head-turn/away,
# otherwise normal.
KEYWORD_CLASSES = [(2, CLASS2_KEYWORDS), (1, CLASS1_KEYWORDS), (0, CLASS0_KEYWORDS)]


def label_from_path(path: Path, default: int = 0) -> int:
    """Infer a target label from folder names, e.g.
    datasets/.../leaning_to_copy/img.jpg -> 2, looking_around/img.jpg -> 1."""
    parts = [p.lower() for p in path.parts]
    for cls, keywords in KEYWORD_CLASSES:
        if any(kw in part for part in parts for kw in keywords):
            return cls
    return default


def list_media() -> list[Path]:
    files: list[Path] = []
    for base in SOURCE_DIRS:
        if base.is_dir():
            files.extend(p for p in base.rglob("*") if p.suffix.lower() in MEDIA_EXTS)
    return sorted(files)


def collect_video_frames(path: Path, frame_step: int):
    """Yield frames from a video, sampled every `frame_step` frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"    [warn] failed to open video: {path}")
        return
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_step == 0:
                yield frame
            idx += 1
    finally:
        cap.release()


def is_valid_candidate(kpts: np.ndarray) -> bool:
    """Skip placeholder candidates where every keypoint is the origin."""
    return bool(np.any(np.abs(kpts[:, :2]).sum(axis=1) > 0))


def initialize_dataset(path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(DATASET_HEADERS)


def process_media(model, media_files: list[Path], out_path: Path,
                  frame_step: int, verbose: bool) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    processed_rows = 0

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        with torch.no_grad():
            for media in media_files:
                label = label_from_path(media)
                frames = (
                    iter([cv2.imread(str(media))])
                    if media.suffix.lower() in IMAGE_EXTS
                    else collect_video_frames(media, frame_step)
                )
                frame_idx = 0
                for frame in frames:
                    if frame is None:
                        continue
                    frame_idx += 1
                    results = model(frame, verbose=False)
                    r = results[0]
                    kpts = (
                        r.keypoints.data.cpu().numpy()
                        if r.keypoints is not None
                        else np.empty((0, 17, 3))
                    )
                    for person in kpts:
                        if not is_valid_candidate(person):
                            continue
                        features = extract_normalized_pose_features(person)
                        writer.writerow([float(v) for v in features] + [label])
                        counts[label] += 1
                        processed_rows += 1

                if verbose:
                    print(f"  [{media.relative_to(ROOT)}] label={label} "
                          f"frames={frame_idx} cumulative_rows={processed_rows}")

    return counts


def print_summary(counts: dict[int, int], out_path: Path) -> None:
    print()
    print("=" * 60)
    print("Summary — pose_dataset_cctv.csv")
    print("=" * 60)
    print(f"{'Label':<10}{'Meaning':<32}{'Samples'}")
    for label, name in ((0, "Normal / Writing / Facing forward"),
                        (1, "Looking Away / Head Turn"),
                        (2, "Leaning / Bending / Slouching")):
        print(f"{label:<10}{name:<32}{counts[label]}")
    print("-" * 60)
    total = sum(counts.values())
    print(f"{'TOTAL':<10}{'':<32}{total}")
    print(f"\nRows appended to: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CCTV pose feature dataset")
    parser.add_argument("--out", default=str(BACKEND_DIR / "pose_dataset_cctv.csv"))
    parser.add_argument("--model", default=str(BACKEND_DIR / "yolo11n-pose.pt"))
    parser.add_argument("--frame-step", type=int, default=10,
                        help="sample every Nth frame from videos (default: 10)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    initialize_dataset(out_path)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # warm-up

    media_files = list_media()
    if not media_files:
        print("[error] no media found under datasets/ or training_videos/. Run "
              "download_public_dataset.py first (or drop videos into training_videos/).")
        return 1

    print(f"[build] {len(media_files)} media file(s) found. Processing...")
    counts = process_media(model, media_files, out_path, args.frame_step, args.verbose)
    print_summary(counts, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
