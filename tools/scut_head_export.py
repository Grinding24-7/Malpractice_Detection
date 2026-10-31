"""
scut_head_export.py — convert the validated SCUT-HEAD manifest into a YOLO
head-detection dataset plus replay lists for the PyQt6 desktop app.

SCUT-HEAD labels are head bounding boxes only.  This exporter therefore makes
a one-class detector ("head"), not a pose/behaviour classifier.  Images are
hard-linked by default so converting all 4,405 images costs almost no extra
disk space.

Outputs (default ``datasets/SCUT_HEAD/yolo``):
    images/{train,val,test}/*.jpg      hard links (or copies)
    labels/{train,val,test}/*.txt      YOLO ``0 cx cy w h`` head boxes
    data.yaml                          Ultralytics dataset description
    replay/{A,B}_{split}.txt           image lists for desktop replay
    export.summary.json                counts and warnings

Usage:
    python tools/scut_head_export.py
    python tools/scut_head_export.py --parts A --splits test --force
    python tools/scut_head_export.py --link-mode copy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "datasets" / "SCUT_HEAD" / "manifest.csv"
DEFAULT_DEST = ROOT / "datasets" / "SCUT_HEAD" / "yolo"
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_split(value: str) -> str:
    """Map manifest split names onto YOLO's three split folders."""
    value = (value or "").strip().lower()
    return value if value in SPLITS else "train"


def _rel_or_abs(path: Path) -> str:
    """Prefer a project-relative POSIX path; fall back to absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _place(src: Path, dst: Path, mode: str, force: bool) -> str:
    """Hard-link, symlink or copy *src* to *dst*.  Returns the method used."""
    if dst.exists() and not force:
        return "exists"
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode in ("hardlink", "symlink"):
        try:
            if mode == "hardlink":
                os.link(src, dst)
            else:
                os.symlink(src, dst)
            return mode
        except OSError:
            # Windows may forbid symlinks, and hard links need one volume.
            pass
    shutil.copy2(src, dst)
    return "copy"


def _yolo_lines(
    boxes: Iterable[Iterable[float]], width: int, height: int
) -> tuple[list[str], int]:
    """Convert absolute VOC boxes to normalised YOLO lines.

    Coordinates are clamped to the image, because a handful of SCUT-HEAD boxes
    touch or slightly exceed the frame edge.  Degenerate boxes are dropped and
    counted so the export report does not hide them.
    """
    lines: list[str] = []
    skipped = 0
    if width <= 0 or height <= 0:
        return lines, sum(1 for _ in boxes)
    for raw in boxes:
        values = list(raw)
        if len(values) != 4:
            skipped += 1
            continue
        x1 = max(0.0, min(float(values[0]), width - 1.0))
        y1 = max(0.0, min(float(values[1]), height - 1.0))
        x2 = max(0.0, min(float(values[2]), width - 1.0))
        y2 = max(0.0, min(float(values[3]), height - 1.0))
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, skipped


def _write_data_yaml(dest: Path) -> Path:
    yaml_path = dest / "data.yaml"
    text = (
        f"path: '{dest.resolve().as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['head']\n"
    )
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def build(
    manifest: Path,
    dest: Path,
    parts: set[str],
    splits: set[str],
    link_mode: str,
    force: bool,
) -> dict:
    """Build the YOLO tree and replay lists.  Returns a summary dictionary."""
    if not manifest.is_file():
        raise FileNotFoundError(
            f"manifest not found: {manifest}. Run tools/scut_head_ingest.py first."
        )
    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected = [
        row for row in rows
        if str(row.get("part", "")).strip().upper() in parts
        and _safe_split(row.get("split", "")) in splits
    ]
    if not selected:
        raise RuntimeError("no manifest rows matched the requested parts/splits")

    per_split: Counter[str] = Counter()
    boxes_per_split: Counter[str] = Counter()
    per_part_split: Counter[tuple[str, str]] = Counter()
    link_methods: Counter[str] = Counter()
    missing_images: list[str] = []
    unparsable_boxes = 0
    bad_dimensions = 0
    replay_paths: dict[str, list[str]] = {}

    for row in selected:
        part = str(row.get("part", "")).strip().upper()
        split = _safe_split(row.get("split", ""))
        image_value = str(row.get("image", "")).strip()
        if not image_value:
            missing_images.append("<empty image path>")
            continue
        src = Path(image_value)
        if not src.is_absolute():
            src = ROOT / src
        if not src.is_file():
            missing_images.append(image_value)
            continue
        try:
            width = int(float(row.get("width", "0") or 0))
            height = int(float(row.get("height", "0") or 0))
        except ValueError:
            width = height = 0
        if width <= 0 or height <= 0:
            # A few Part B XML files omit <size>.  The image itself still has
            # dimensions, so probe only those rows rather than dropping data.
            try:
                import cv2

                probe = cv2.imread(str(src))
                if probe is not None:
                    height, width = probe.shape[:2]
            except Exception:
                pass
        if width <= 0 or height <= 0:
            bad_dimensions += 1
            continue

        try:
            boxes = json.loads(row.get("boxes", "[]") or "[]")
        except json.JSONDecodeError:
            unparsable_boxes += 1
            boxes = []
        lines, dropped = _yolo_lines(boxes, width, height)
        unparsable_boxes += dropped

        # Part prefixes guarantee unique stems when A and B have same names.
        stem = f"{part}_{src.stem}"
        image_dst = dest / "images" / split / f"{stem}{src.suffix.lower()}"
        label_dst = dest / "labels" / split / f"{stem}.txt"
        method = _place(src, image_dst, link_mode, force)
        link_methods[method] += 1

        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        per_split[split] += 1
        boxes_per_split[split] += len(lines)
        per_part_split[(part, split)] += 1
        replay_paths.setdefault(f"{part}_{split}", []).append(_rel_or_abs(src))

    if per_split.get("train", 0) == 0 or per_split.get("val", 0) == 0:
        raise RuntimeError(
            "YOLO training needs at least one train and one val image; "
            f"got train={per_split.get('train', 0)}, val={per_split.get('val', 0)}"
        )

    yaml_path = _write_data_yaml(dest)
    replay_dir = dest / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_files: dict[str, str] = {}
    for key, paths in sorted(replay_paths.items()):
        list_path = replay_dir / f"{key}.txt"
        list_path.write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
        )
        replay_files[key] = _rel_or_abs(list_path)
    # Per-part lists are convenient for "replay everything in Part A".
    for part in sorted(parts):
        paths = [
            path for key, values in replay_paths.items()
            if key.startswith(f"{part}_") for path in values
        ]
        if paths:
            list_path = replay_dir / f"{part}_all.txt"
            list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
            replay_files[f"{part}_all"] = _rel_or_abs(list_path)

    summary = {
        "manifest": _rel_or_abs(manifest),
        "dataset_root": _rel_or_abs(dest),
        "data_yaml": _rel_or_abs(yaml_path),
        "parts": sorted(parts),
        "splits": sorted(splits),
        "images": sum(per_split.values()),
        "heads": sum(boxes_per_split.values()),
        "per_split": {
            split: {
                "images": per_split[split],
                "heads": boxes_per_split[split],
            }
            for split in SPLITS if per_split.get(split)
        },
        "per_part_split": {
            f"{part}/{split}": count
            for (part, split), count in sorted(per_part_split.items())
        },
        "link_methods": dict(link_methods),
        "replay_files": replay_files,
        "missing_images": len(missing_images),
        "bad_dimensions": bad_dimensions,
        "dropped_boxes": unparsable_boxes,
    }
    (dest / "export.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _parse_set(value: str, allowed: tuple[str, ...], label: str) -> set[str]:
    chosen = {item.strip().upper() for item in value.split(",") if item.strip()}
    unknown = chosen - set(allowed)
    if unknown:
        raise SystemExit(
            f"unknown {label}: {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(allowed)}"
        )
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export SCUT-HEAD to YOLO head-detection format."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--parts", default="A,B", help="A, B or A,B (default)")
    parser.add_argument(
        "--splits", default="train,val,test",
        help="comma-separated train,val,test (default: all)",
    )
    parser.add_argument(
        "--link-mode", choices=("hardlink", "symlink", "copy"),
        default="hardlink", help="how to place images (default: hardlink)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace existing links/copies instead of reusing them",
    )
    args = parser.parse_args()

    parts = _parse_set(args.parts, ("A", "B"), "part")
    splits = {
        item.strip().lower()
        for item in args.splits.split(",") if item.strip()
    }
    unknown_splits = splits - set(SPLITS)
    if unknown_splits:
        raise SystemExit(
            f"unknown split(s): {', '.join(sorted(unknown_splits))}"
        )

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    summary = build(
        args.manifest.resolve(), dest, parts, splits,
        args.link_mode, args.force,
    )

    print("[scut-export] summary")
    print(f"  dataset root : {summary['dataset_root']}")
    print(f"  data.yaml    : {summary['data_yaml']}")
    print(f"  images       : {summary['images']:,}")
    print(f"  head boxes   : {summary['heads']:,}")
    print(f"  link methods : {summary['link_methods']}")
    if summary["missing_images"]:
        print(f"  missing imgs : {summary['missing_images']:,}")
    if summary["dropped_boxes"]:
        print(f"  dropped boxes: {summary['dropped_boxes']:,}")
    for name, stats in summary["per_split"].items():
        print(
            f"  {name:<12} : {stats['images']:,} images, "
            f"{stats['heads']:,} heads"
        )
    print("  replay lists :")
    for name, path in summary["replay_files"].items():
        print(f"    {name:<10} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
