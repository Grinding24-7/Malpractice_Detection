"""
download_public_dataset.py — Week 3: automated public CCTV dataset ingestion.

Downloads a public classroom / exam-behaviour dataset from Kaggle (anonymous,
no auth needed) or Roboflow Universe, and stages it under `datasets/` for the
pose-feature pipeline (build_cctv_dataset.py).  As a fallback (or when
`--source local` is used) it ingests classroom / surveillance video files from
the local `training_videos/` directory.

Sources (defaults can be overridden on the CLI):
    Kaggle   : "aneelapervez/classroom-exam-cheating-detection"  (default)
    Roboflow : workspace "object-detection-ufdpb", project "classroom-detected"

Auth:
    Kaggle public datasets download anonymously via kagglehub (no credentials).
    Roboflow requires ROBOFLOW_API_KEY even for public projects.
    `--source auto` (default) tries Kaggle first, then Roboflow, and finally
    falls back to the local `training_videos/` directory so the downstream
    pipeline never runs dry.

Note on roboflow: roboflow==1.4.0 hard-pins opencv-python-headless==4.10.0.84,
which conflicts with this project's opencv pin, so it is installed on demand by
`uv pip install` (managed via _ensure_dependencies) rather than added to the
lockfile.  Running `uv run ...` will restore the locked opencv afterwards.

Usage:
    uv run python download_public_dataset.py                      # auto
    uv run python download_public_dataset.py --source kaggle --kaggle-dataset user/name
    uv run python download_public_dataset.py --source roboflow \
        --workspace W --project P --version 1 --format yolov8
    uv run python download_public_dataset.py --source local
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINING_VIDEOS_DIR = ROOT / "training_videos"
OUT_DIR = ROOT / "datasets"

REQUIRED_PACKAGES = ["roboflow", "kagglehub", "pandas"]

MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".mov", ".mkv"}


def _ensure_dependencies() -> None:
    """Install missing packages via uv into the project environment."""
    missing = [pkg for pkg in REQUIRED_PACKAGES if _importable(pkg)]
    if not missing:
        print(f"[deps] all required packages already installed: {', '.join(REQUIRED_PACKAGES)}")
        return
    cmd = ["uv", "pip", "install", *missing]
    print(f"[deps] installing via uv: {' '.join(missing)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    for pkg in missing:
        if not _importable(pkg):
            raise RuntimeError(f"failed to install required package: {pkg}")


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _download_roboflow(args: argparse.Namespace) -> Path:
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set.  Get one from "
            "https://roboflow.com/settings/account and export it, then re-run."
        )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    location = OUT_DIR / f"{args.project}-{args.version}"
    print(f"[roboflow] downloading {args.workspace}/{args.project} v{args.version} "
          f"as {args.format} -> {location}")
    project.version(args.version).download(args.format, location=str(location))
    return location


def _download_kaggle(args: argparse.Namespace) -> Path:
    import kagglehub

    print(f"[kaggle] downloading {args.kaggle_dataset} ...")
    cached = Path(kagglehub.dataset_download(args.kaggle_dataset))
    # Stage under datasets/ so build_cctv_dataset.py can pick it up.
    slug = args.kaggle_dataset.replace("/", "_").replace("-", "_")
    location = OUT_DIR / f"kaggle_{slug}"
    if location.exists():
        shutil.rmtree(location)
    shutil.copytree(cached, location)
    print(f"[kaggle] staged {args.kaggle_dataset} -> {location}")
    return location


def _ingest_local() -> Path:
    """Copy local training_videos into datasets/local for the pipeline."""
    if not TRAINING_VIDEOS_DIR.is_dir():
        print("[local] 'training_videos/' not found - nothing to ingest.")
        return OUT_DIR / "local"
    dest = OUT_DIR / "local"
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in TRAINING_VIDEOS_DIR.rglob("*") if p.suffix.lower() in MEDIA_EXTS
    )
    if not files:
        print(f"[local] no media files found under {TRAINING_VIDEOS_DIR}")
        return dest
    for src in files:
        shutil.copy2(src, dest / src.name)
    print(f"[local] ingested {len(files)} media file(s) -> {dest}")
    return dest


def list_media(root: Path) -> list[Path]:
    """Recursively list image/video files under a directory."""
    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in MEDIA_EXTS
    )


def _run_auto(args: argparse.Namespace) -> Path:
    """Try Kaggle -> Roboflow -> local, returning the first that yields media."""
    attempts = []
    try:
        source_dir = _download_kaggle(args)
        if list_media(source_dir):
            return source_dir
        attempts.append("kaggle: no media found")
    except Exception as exc:  # noqa: BLE001 - fall through to next source
        attempts.append(f"kaggle: {exc}")
    try:
        source_dir = _download_roboflow(args)
        if list_media(source_dir):
            return source_dir
        attempts.append("roboflow: no media found")
    except Exception as exc:  # noqa: BLE001 - fall through to local
        attempts.append(f"roboflow: {exc}")
    print("[auto] Kaggle/Roboflow unavailable, using local training_videos/")
    for note in attempts:
        print(f"    - {note}")
    return _ingest_local()


def main() -> int:
    parser = argparse.ArgumentParser(description="Public CCTV dataset ingestion")
    parser.add_argument(
        "--source",
        choices=["auto", "roboflow", "kaggle", "local"],
        default="auto",
        help="data source (default: auto = kaggle -> roboflow -> local)",
    )
    parser.add_argument("--workspace", default="object-detection-ufdpb")
    parser.add_argument("--project", default="classroom-detected")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--format", default="yolov8")
    parser.add_argument("--kaggle-dataset", default="aneelapervez/classroom-exam-cheating-detection")
    args = parser.parse_args()

    _ensure_dependencies()

    if args.source == "auto":
        source_dir = _run_auto(args)
    elif args.source == "roboflow":
        source_dir = _download_roboflow(args)
    elif args.source == "kaggle":
        source_dir = _download_kaggle(args)
    else:
        source_dir = _ingest_local()

    media = list_media(source_dir)
    print(f"[done] {len(media)} media file(s) staged under {source_dir}")
    if not media:
        print("[warn] no media staged - dropping classroom/surveillance videos into "
              f"{TRAINING_VIDEOS_DIR} (or re-run with valid --source) and then run "
              "build_cctv_dataset.py")
        return 1
    for p in media[:10]:
        print(f"    {p.relative_to(ROOT)}")
    if len(media) > 10:
        print(f"    ... and {len(media) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
