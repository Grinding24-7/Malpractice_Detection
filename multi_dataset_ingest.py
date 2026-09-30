"""
multi_dataset_ingest.py — expand the CCTV pose dataset by ingesting and
combining several public datasets into 'pose_dataset_cctv.csv'.

Public sources ingested:
    a) Kaggle ExamCheating_Dataset
       (ardutraagiginting/exam-cheating-dataset)
       Categories: normal_act, looking_friend, giving_object, giving_code,
       cheating  (+ a roboflow-exported test/ split).
    b) Roboflow "Exam Cheating" Universe dataset
       (workspace 'kattal', project 'exam-cheating').  Requires
       ROBOFLOW_API_KEY; skipped with a warning when absent.
    c) Mendeley "Students' Suspicious Behaviors" tabular dataset
       (https://data.mendeley.com/datasets/39xs8th543/1).  This is a
       5,500-row / 38-column tabular set (head pose, phone, gaze, ...) whose
       feature schema does NOT match the pose-keypoint features stored in
       'pose_dataset_cctv.csv', so it is staged separately under
       datasets/mendeley_tabular/ and reported as not pose-merged.

Pipeline (per ingested image):
    1. Runs YOLO11-pose ('yolo11n-pose.pt') on every image.
    2. Extracts scale-normalized pose features for *every* detected student
       via backend/feature_extractor.extract_normalized_pose_features().
    3. Maps the source folder / label tag to an integer class via LABEL_MAP.
    4. Appends each feature row + label to 'pose_dataset_cctv.csv'.

Quality control:
    - Wipes/rebuilds the target CSV so every row uses the unified 4-class
      mapping (0 Normal, 1 Head Turn / Peeking, 2 Leaning / Concealing,
      3 Passing / Reaching).
    - Prints a per-class row breakdown.
    - If any class has < --min-samples rows, applies small Gaussian feature
      jittering (data augmentation) to rebalance it up to the threshold.

Usage:
    uv run python multi_dataset_ingest.py
    uv run python multi_dataset_ingest.py --out backend/pose_dataset_cctv.csv \
        --min-samples 500 --sources kaggle,mendeley
    ROBOFLOW_API_KEY=... uv run python multi_dataset_ingest.py --sources kaggle,roboflow,mendeley
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
DATASETS_DIR = ROOT / "datasets"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Late import (needs torch/ultralytics installed) inside main().
# Model weights default to the copy already tracked at backend/yolo11n-pose.pt.
MODEL_DEFAULT = BACKEND_DIR / "yolo11n-pose.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

# --------------------------------------------------------------------------
# Unified label mapping (source category tags -> integer class).
# --------------------------------------------------------------------------
LABEL_MAP = {
    # 0 = Normal / attentive / working
    "normal": 0, "normal_act": 0, "writing": 0, "reading": 0, "attentive": 0,
    # 1 = Head Turn / Peeking
    "looking_friend": 1, "looking_around": 1, "turning_head": 1, "peeking": 1,
    # 2 = Leaning / Concealing (incl. phone use)
    "cheating": 2, "leaning_down": 2, "using_phone": 2, "phone": 2,
    # 3 = Passing / Reaching
    "giving_code": 3, "giving_object": 3, "passing_note": 3, "reaching": 3,
}

# Raw folder/tag aliases (folder strings found in the public datasets) mapped
# onto canonical LABEL_MAP keys.  Full-tag aliases win over word-level matches.
TAG_ALIASES = {
    "using_mobile": "using_phone",      # Kaggle: "using mobile"
    "leaning_to_copy": "cheating",      # Kaggle: "leaning to copy"
    "sharing_answers": "giving_object",  # Kaggle: "sharing answers"
    "students_cheating": "cheating",    # Roboflow kattal/exam-cheating
    "students_not_cheating": "normal",  # Roboflow kattal/exam-cheating
    "not_cheating": "normal",           # negation guard
    "mobile": "phone",
    "cheat": "cheating",                # roboflow-exported stem prefix
    "good": "normal",                   # roboflow-exported stem prefix
    "copy": "cheating",
    "share": "giving_object",
    "suspicious": "cheating",
}

CLASS_NAMES = {
    0: "Normal / Writing / Facing forward",
    1: "Head Turn / Peeking",
    2: "Leaning / Concealing",
    3: "Passing / Reaching",
}

# --------------------------------------------------------------------------
# Source specifications
# --------------------------------------------------------------------------
KAGGLE_EXAM_CHEATING_DATASET = "ardutraagiginting/exam-cheating-dataset"
MENDELEY_FILES_URL = "https://data.mendeley.com/api/datasets/39xs8th543/files"
MENDELEY_FILE_URL = (
    "https://data.mendeley.com/public-files/datasets/39xs8th543/files/"
    "e27dd7c5-d672-4849-a41e-83cc3be681c2/file_downloaded"
)
MENDELEY_FILENAME = "students_suspicious_behaviors.csv"
ROBOFLOW_DEFAULTS = {"workspace": "kattal", "project": "exam-cheating", "version": 1}

REQUIRED_PACKAGES = ["roboflow", "kagglehub", "pandas", "requests"]


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _ensure_dependencies() -> None:
    """Install missing packages via uv into the project environment."""
    missing = [pkg for pkg in REQUIRED_PACKAGES if not _importable(pkg)]
    if not missing:
        print(f"[deps] all required packages already installed: {', '.join(REQUIRED_PACKAGES)}")
        return
    cmd = ["uv", "pip", "install", *missing]
    print(f"[deps] installing via uv: {' '.join(missing)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    for pkg in missing:
        if not _importable(pkg):
            raise RuntimeError(f"failed to install required package: {pkg}")


# --------------------------------------------------------------------------
# Downloaders
# --------------------------------------------------------------------------
def _stage_kaggle_exam_cheating() -> Path:
    import kagglehub

    print(f"[kaggle] downloading {KAGGLE_EXAM_CHEATING_DATASET} ...")
    cached = Path(kagglehub.dataset_download(KAGGLE_EXAM_CHEATING_DATASET))
    location = DATASETS_DIR / "kaggle_exam_cheating_dataset"
    if location.exists():
        shutil.rmtree(location)
    shutil.copytree(cached, location)
    print(f"[kaggle] staged {KAGGLE_EXAM_CHEATING_DATASET} -> {location}")
    return location


def _download_roboflow(args: argparse.Namespace) -> Path | None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("[roboflow] skipped: ROBOFLOW_API_KEY is not set (https://roboflow.com/settings/account)")
        return None
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    location = DATASETS_DIR / f"roboflow_{args.project}-{args.version}"
    print(f"[roboflow] downloading {args.workspace}/{args.project} v{args.version} "
          f"as yolov8 -> {location}")
    project.version(args.version).download("yolov8", location=str(location))
    return location


def _download_mendeley() -> Path | None:
    """Download the tabular Mendeley dataset and stage it separately."""
    import requests

    try:
        meta = requests.get(MENDELEY_FILES_URL, timeout=30).json()
        download_url = meta[0]["content_details"].get("download_url") or meta[0].get("download_url")
        if not download_url:
            raise ValueError("no download_url in Mendeley metadata")
    except Exception as exc:  # noqa: BLE001 - non-fatal source
        print(f"[mendeley] metadata lookup failed: {exc}")
        return None

    stadir = DATASETS_DIR / "mendeley_tabular"
    stadir.mkdir(parents=True, exist_ok=True)
    dest = stadir / MENDELEY_FILENAME
    try:
        with requests.get(download_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as exc:  # noqa: BLE001 - non-fatal source
        print(f"[mendeley] download failed: {exc}")
        return None
    print(f"[mendeley] staged tabular dataset ({dest.stat().st_size} bytes) -> {dest}")
    return dest


# --------------------------------------------------------------------------
# Label inference
# --------------------------------------------------------------------------
def _tags(path: Path) -> list[str]:
    """Lowercased, underscore-normalised path components (dirs + file stem)."""
    parts = [p for p in path.parts if not p.startswith(".")]
    return [re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_") for p in parts]


def label_from_path(path: Path, default: int = 0) -> int:
    """
    Map a source image path to a unified class using LABEL_MAP + aliases.

    Resolution order (most-specific first):
      1. Exact full-tag alias / LABEL_MAP key on any path component,
         scanning nearest-to-file first.
      2. Word-level matches inside a component/filename stem.
      3. `default` (Normal) if nothing matched.
    """
    tags = _tags(path)
    for tag in reversed(tags):
        if not tag:
            continue
        tag = TAG_ALIASES.get(tag, tag)
        if tag in LABEL_MAP:
            return LABEL_MAP[tag]
    for tag in reversed(tags):
        if not tag:
            continue
        for word in re.findall(r"[a-z]+", tag):
            word = TAG_ALIASES.get(word, word)
            if word in LABEL_MAP:
                return LABEL_MAP[word]
    return default


def list_images() -> list[Path]:
    """All image files staged under datasets/, excluding the tabular dir."""
    if not DATASETS_DIR.is_dir():
        return []
    return sorted(
        p for p in DATASETS_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and "mendeley_tabular" not in p.parts
    )


def collect_tabular_sources() -> list[Path]:
    return sorted((DATASETS_DIR / "mendeley_tabular").glob("*.csv")) \
        if (DATASETS_DIR / "mendeley_tabular").is_dir() else []


# --------------------------------------------------------------------------
# Pose feature extraction via YOLO11-pose
# --------------------------------------------------------------------------
def is_valid_candidate(kpts: np.ndarray) -> bool:
    """Skip placeholder candidates where every keypoint is the origin."""
    return bool(np.any(np.abs(kpts[:, :2]).sum(axis=1) > 0))


def process_images(model, image_files: list[Path], out_path: Path,
                   verbose: bool) -> dict[int, int]:
    counts = {label: 0 for label in CLASS_NAMES}
    processed_rows = 0

    from feature_extractor import extract_normalized_pose_features

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_HEADERS)
        with np.errstate(all="ignore"):
            for image in image_files:
                import cv2

                frame = cv2.imread(str(image))
                if frame is None:
                    continue
                label = label_from_path(image)
                results = model(frame, verbose=False)
                r = results[0]
                kpts = (
                    r.keypoints.data.cpu().numpy()
                    if r.keypoints is not None
                    else np.empty((0, 17, 3))
                )
                n_rows = 0
                for person in kpts:
                    if not is_valid_candidate(person):
                        continue
                    features = extract_normalized_pose_features(person)
                    writer.writerow([float(v) for v in features] + [label])
                    counts[label] += 1
                    n_rows += 1
                if n_rows and verbose:
                    processed_rows += n_rows
                    print(f"  [{image.relative_to(ROOT)}] label={label} persons={n_rows}")
    if verbose:
        print(f"  [processed] {len(image_files)} images -> {processed_rows} pose rows")
    return counts


# --------------------------------------------------------------------------
# Verification & balancing
# --------------------------------------------------------------------------
def load_counts(out_path: Path) -> dict[int, int]:
    counts = {label: 0 for label in CLASS_NAMES}
    if not out_path.is_file():
        return counts
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            counts[int(row["label"])] += 1
    return counts


def print_breakdown(title: str, counts: dict[int, int], out_path: Path) -> None:
    print()
    print("=" * 64)
    print(f"{title} — {out_path.name}")
    print("=" * 64)
    print(f"{'Class':<6}{'Meaning':<34}{'Samples'}")
    for label, name in CLASS_NAMES.items():
        print(f"{label:<6}{name:<34}{counts[label]}")
    print("-" * 64)
    print(f"{'TOTAL':<6}{'':<34}{sum(counts.values())}")


def jitter_row(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add small multiplicative Gaussian jitter (data augmentation)."""
    noise = 1.0 + rng.normal(0.0, 0.02, size=features.shape)
    jittered = features * noise
    jittered += rng.normal(0.0, 1e-4, size=features.shape)
    return jittered


def balance_dataset(out_path: Path, min_samples: int, seed: int) -> dict[int, int]:
    """Jitter-augment any class with < min_samples rows up to the threshold."""
    if not out_path.is_file():
        return {}
    rows = list(csv.DictReader(open(out_path, newline="")))
    rng = np.random.default_rng(seed)
    append_cols = [c for c in DATASET_HEADERS if c != "label"]

    augmented = 0
    for label in CLASS_NAMES:
        target = int(label)
        pool = [r for r in rows if int(r["label"]) == target]
        if len(pool) >= min_samples:
            continue
        needed = min_samples - len(pool)
        print(f"  [balance] class {target} has {len(pool)} rows — augmenting +{needed}")
        for _ in range(needed):
            src = pool[int(rng.integers(0, len(pool)))] if pool else rows[int(rng.integers(0, len(rows)))]
            feats = np.asarray([float(src[c]) for c in append_cols], dtype=np.float64)
            jit = jitter_row(feats, rng)
            rows.append({**{c: float(v) for c, v in zip(append_cols, jit, strict=True)},
                         "label": str(target)})
            augmented += 1

    if augmented:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(DATASET_HEADERS)
            for r in rows:
                writer.writerow([r[c] for c in DATASET_HEADERS])
        print(f"  [balance] appended {augmented} jittered rows -> {out_path}")
    return load_counts(out_path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-dataset CCTV pose ingestion")
    parser.add_argument("--out", default=str(BACKEND_DIR / "pose_dataset_cctv.csv"))
    parser.add_argument("--model", default=str(MODEL_DEFAULT))
    parser.add_argument("--sources", default="kaggle,roboflow,mendeley")
    parser.add_argument("--workspace", default=ROBOFLOW_DEFAULTS["workspace"])
    parser.add_argument("--project", default=ROBOFLOW_DEFAULTS["project"])
    parser.add_argument("--version", type=int, default=ROBOFLOW_DEFAULTS["version"])
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _ensure_dependencies()
    sources = {s.strip() for s in args.sources.split(",")}

    # ---- 1. Download / stage the public sources ---------------------------
    if "kaggle" in sources:
        _stage_kaggle_exam_cheating()
    if "roboflow" in sources:
        _download_roboflow(args)
    if "mendeley" in sources:
        _download_mendeley()

    image_files = list_images()
    if not image_files:
        print("[error] no images staged under datasets/. Re-run with valid sources.")
        return 1

    print(f"[ingest] {len(image_files)} image(s) found under {DATASETS_DIR.relative_to(ROOT)}")

    # ---- 2. YOLO11-pose feature extraction loop ----------------------------
    from ultralytics import YOLO

    model = YOLO(args.model)
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)  # warm-up

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = process_images(model, image_files, out_path, args.verbose)
    print_breakdown("Baseline (before balancing)", counts, out_path)

    # ---- 3. Verify + balance (minor jitter augmentation) -------------------
    final = balance_dataset(out_path, args.min_samples, args.seed)
    print_breakdown("Final balanced dataset", final, out_path)
    print(f"\nRows written to: {out_path}")

    # ---- 4. Tabular Mendeley note ------------------------------------------
    tabular = collect_tabular_sources()
    if tabular:
        print(f"\n[mendeley] tabular dataset staged (NOT pose-merged): {tabular[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())