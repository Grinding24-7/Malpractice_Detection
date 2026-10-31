"""
scut_head_ingest.py — download, extract and normalise the SCUT-HEAD dataset.

SCUT-HEAD is a *head detection* dataset: 4,405 classroom images with 111,248
Pascal-VOC head bounding boxes.  It has no pose keypoints and no
normal/cheating behaviour labels, so it cannot train or validate the desktop
P01-P38 rule engine directly.  Its correct use in this project is:

    * real classroom occupancy / head-count evaluation;
    * a real-image stress test for the person detector that feeds the
      desktop classifier;
    * clean Pascal-VOC boxes for training a dedicated head detector later.

The dataset is free for academic research only.  It is downloaded to
``datasets/SCUT_HEAD`` (git-ignored) and is never committed.

Usage:
    python tools/scut_head_ingest.py --part both
    python tools/scut_head_ingest.py --part A --dest datasets/SCUT_HEAD
    python tools/scut_head_ingest.py --skip-download   # use an existing extract
    python tools/scut_head_ingest.py --part both --force

The Google Drive links published by HCIILAB are resolved through the normal
Drive "virus scan" confirmation page, so no third-party downloader is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from http.cookiejar import CookieJar
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "datasets" / "SCUT_HEAD"

# Published in https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
DRIVE_IDS = {
    "A": "1yaOF9os5wPVNNG4GVzNyULLVe74vdrBE",
    "B": "1LZ_KlTPStDEcqycfqUkDkqQ-aNMMC3cl",
}
ZIP_NAMES = {
    "A": "SCUT_HEAD_Part_A.zip",
    "B": "SCUT_HEAD_Part_B.zip",
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CHUNK = 1 << 20
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------
def _request(url: str, data: bytes | None = None) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )


def _download_drive(file_id: str, dest: Path, force: bool = False) -> Path:
    """Download one public Google Drive file, including the confirm page."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force and zipfile.is_zipfile(dest):
        print(f"[scut] already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return dest

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar())
    )
    initial = (
        "https://drive.usercontent.google.com/download"
        f"?id={urllib.parse.quote(file_id)}&export=download"
    )
    print(f"[scut] resolving Google Drive file {file_id} ...")
    with opener.open(_request(initial), timeout=60) as response:
        body = response.read()

    # A small HTML page is the virus-scan confirmation.  A direct response is
    # already the zip.  Handle both.
    if zipfile.is_zipfile(dest) and not force:
        return dest
    if body[:4] == b"PK\x03\x04":
        dest.write_bytes(body)
        print(f"[scut] downloaded {dest.name}: {len(body):,} bytes")
        return dest

    html = body.decode("utf-8", "replace")
    action_match = re.search(r'<form[^>]+action="([^"]+)"', html, re.I)
    if not action_match:
        raise RuntimeError(
            "Google Drive returned an unexpected page instead of the archive. "
            "Download it manually and pass --skip-download after extracting."
        )
    inputs = dict(
        re.findall(
            r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
            html,
            re.I,
        )
    )
    action = action_match.group(1).replace("&amp;", "&")
    download_url = action + "?" + urllib.parse.urlencode(inputs)

    with opener.open(_request(download_url), timeout=120) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(
                "Google Drive still returned HTML; the file may be quota-limited "
                "or require a browser session. Download it manually from the "
                "GitHub README and use --skip-download."
            )
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if total:
                    pct = min(100, received * 100 // total)
                    print(f"\r[scut] {dest.name}: {pct:3d}% "
                          f"({received:,}/{total:,} bytes)", end="", flush=True)
        print()
        tmp.replace(dest)
    if not zipfile.is_zipfile(dest):
        raise RuntimeError(f"downloaded file is not a valid zip: {dest}")
    print(f"[scut] downloaded {dest.name}: {dest.stat().st_size:,} bytes")
    return dest


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def _part_roots(destination: Path) -> dict[str, Path]:
    """Find the directory that actually contains JPEGImages + Annotations.

    Matching is strict: the folder name must identify the part.  Without that
    check, Part A would be returned for Part B too, and the manifest would
    silently contain Part A twice (which is exactly what happened before this
    guard was added).
    """
    candidates = [
        p.parent
        for p in destination.rglob("Annotations")
        if p.is_dir() and (p.parent / "JPEGImages").is_dir()
    ]
    roots: dict[str, Path] = {}
    for part in ("A", "B"):
        token_part = part.lower()
        tokens = (
            f"part_{token_part}",
            f"part{token_part}",
            f"part {token_part}",
        )
        matching = [
            path for path in candidates
            if any(token in path.name.lower() for token in tokens)
        ]
        if matching:
            matching.sort(key=lambda path: len(path.parts))
            roots[part] = matching[0]
    return roots


def _ensure_part(
    part: str,
    destination: Path,
    force: bool,
    skip_download: bool,
) -> Path:
    zip_path = destination / "_downloads" / ZIP_NAMES[part]
    if not skip_download:
        _download_drive(DRIVE_IDS[part], zip_path, force=force)

    roots = _part_roots(destination)
    if part in roots and not force:
        print(f"[scut] Part {part} already extracted: {roots[part]}")
        return roots[part]

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Part {part} is not extracted and {zip_path} is missing. "
            "Run without --skip-download or place the official zip there."
        )

    extract_dir = destination / f"Part{part}"
    print(f"[scut] extracting {zip_path.name} -> {extract_dir} ...")
    _safe_extract(zip_path, extract_dir)
    roots = _part_roots(destination)
    if part not in roots:
        raise RuntimeError(
            f"Part {part} extracted but no JPEGImages/Annotations pair was found "
            f"under {destination}."
        )
    print(f"[scut] Part {part} ready: {roots[part]}")
    return roots[part]


# ---------------------------------------------------------------------------
# Pascal VOC normalisation
# ---------------------------------------------------------------------------
def _read_split(root: Path) -> dict[str, str]:
    """Map image stem -> split from ImageSets/Main/{train,test}.txt."""
    splits: dict[str, str] = {}
    image_sets = root / "ImageSets" / "Main"
    for split in ("train", "test", "val"):
        path = image_sets / f"{split}.txt"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stem = line.strip().split()[0] if line.strip() else ""
            if stem:
                splits[stem] = split
    return splits


def _parse_annotation(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def text(tag: str, default: str = "") -> str:
        node = root.find(tag)
        return node.text.strip() if node is not None and node.text else default

    size = root.find("size")
    width = int(text("width", "0"))
    height = int(text("height", "0"))
    if size is not None:
        width = int(size.findtext("width", width) or width)
        height = int(size.findtext("height", height) or height)

    boxes: list[list[int]] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            x1 = int(float(bnd.findtext("xmin", "0")))
            y1 = int(float(bnd.findtext("ymin", "0")))
            x2 = int(float(bnd.findtext("xmax", "0")))
            y2 = int(float(bnd.findtext("ymax", "0")))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return {
        "filename": text("filename", xml_path.with_suffix(".jpg").name),
        "width": width,
        "height": height,
        "boxes": boxes,
    }


def _find_image(root: Path, filename: str, stem: str) -> Path | None:
    direct = root / "JPEGImages" / filename
    if direct.is_file():
        return direct
    for ext in IMAGE_EXTS:
        candidate = root / "JPEGImages" / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def build_manifest(
    roots: dict[str, Path],
    manifest_path: Path,
) -> dict:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    missing_images = 0
    unparsable = 0
    per_split: Counter[tuple[str, str]] = Counter()
    heads_per_split: Counter[tuple[str, str]] = Counter()

    for part, root in sorted(roots.items()):
        split_map = _read_split(root)
        xml_files = sorted((root / "Annotations").glob("*.xml"))
        print(f"[scut] Part {part}: {len(xml_files):,} annotation files")
        for xml_path in xml_files:
            try:
                ann = _parse_annotation(xml_path)
            except (ET.ParseError, OSError, ValueError):
                unparsable += 1
                continue
            stem = xml_path.stem
            image = _find_image(root, ann["filename"], stem)
            if image is None:
                missing_images += 1
                image_rel = ""
            else:
                image_rel = str(image.relative_to(ROOT))
            split = split_map.get(stem, "unspecified")
            rows.append(
                {
                    "part": part,
                    "split": split,
                    "image": image_rel,
                    "annotation": str(xml_path.relative_to(ROOT)),
                    "width": ann["width"],
                    "height": ann["height"],
                    "n_heads": len(ann["boxes"]),
                    "boxes": json.dumps(ann["boxes"], separators=(",", ":")),
                }
            )
            per_split[(part, split)] += 1
            heads_per_split[(part, split)] += len(ann["boxes"])

    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part", "split", "image", "annotation",
                "width", "height", "n_heads", "boxes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "images": len(rows),
        "heads": sum(row["n_heads"] for row in rows),
        "missing_images": missing_images,
        "unparsable_annotations": unparsable,
        "per_split": {
            f"{part}/{split}": {
                "images": count,
                "heads": heads_per_split[(part, split)],
            }
            for (part, split), count in sorted(per_split.items())
        },
    }
    print("\n[scut] summary")
    print(f"  images               : {summary['images']:,}")
    print(f"  heads                : {summary['heads']:,}")
    print(f"  missing images       : {summary['missing_images']:,}")
    print(f"  unparsable XML       : {summary['unparsable_annotations']:,}")
    for key, value in summary["per_split"].items():
        print(f"  {key:<18}: {value['images']:,} images, "
              f"{value['heads']:,} heads")
    print(f"  manifest             : {manifest_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download, extract and manifest SCUT-HEAD."
    )
    parser.add_argument(
        "--part", choices=("A", "B", "both"), default="both",
        help="dataset part to fetch (default: both)",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"dataset directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest CSV path (default: <dest>/manifest.csv)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="use existing zips/extracted folders; never touch the network",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-download and re-extract even if files already exist",
    )
    args = parser.parse_args()

    destination = args.dest.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    parts = ("A", "B") if args.part == "both" else (args.part,)

    roots: dict[str, Path] = {}
    for part in parts:
        try:
            roots[part] = _ensure_part(
                part,
                destination,
                force=args.force,
                skip_download=args.skip_download,
            )
        except Exception as exc:  # noqa: BLE001 - one part may fail
            print(f"[scut] Part {part} failed: {exc}", file=sys.stderr)
            if len(parts) == 1:
                return 1

    if not roots:
        print("[scut] no parts available", file=sys.stderr)
        return 1

    manifest = args.manifest or (destination / "manifest.csv")
    summary = build_manifest(roots, manifest.resolve())
    summary_path = manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  summary              : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
