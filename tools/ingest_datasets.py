"""Modular, provenance-preserving dataset ingestion for the proctoring model.

What this tool does
-------------------
* ``kaggle-exam``: reads the real, hand-labelled ExamCheatingDataset classes
  ``normal act`` and ``looking friend`` and maps them to the rule-engine
  verdicts **P01 Normal Sitting** and **P20 Looking at Neighbor**.  Only those
  two classes have an honest pose-only mapping; the other three source
  classes are documented and skipped rather than forced into a wrong label.
* ``scut-head``: reads the SCUT-HEAD YOLO boxes and writes a separate
  occupancy/head-density table.  SCUT-HEAD has **head boxes only** -- no pose
  keypoints and no behaviour labels -- so it must never become P01-P40
  training rows.
* ``coco``: reads COCO person-keypoint annotations and writes **unlabelled**
  54-D rows to a separate pretraining CSV.  COCO has no exam-behaviour labels;
  writing it into the supervised training file would invent labels.
* ``mpii``, ``dmd`` and ``desk``: accepted for CLI symmetry but skipped with a
  precise reason when the data is absent or cannot be aligned to the 54-D
  schema.  No source is silently treated as normal.

Every labelled row is written through ``DatasetLogger`` so the CSV contract,
NaN handling and bbox-normalised keypoints are identical to live capture.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from desktop_app.core.dataset_logger import DatasetLogger
from desktop_app.core.pose_classifier import PoseClassifier

logger = logging.getLogger("ingest_datasets")

DATASETS = _PROJECT_ROOT / "datasets"
DEFAULT_OUT = _PROJECT_ROOT / "data" / "proctoring_features.csv"
DEFAULT_OCCUPANCY = _PROJECT_ROOT / "data" / "scut_head_occupancy.csv"
DEFAULT_UNLABELED = _PROJECT_ROOT / "data" / "pose_pretraining_features.csv"
KAGGLE_ROOT = DATASETS / "kaggle_exam_cheating_dataset" / "ExamCheatingDataset"
SCUT_ROOT = DATASETS / "SCUT_HEAD"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FIXED_CANVAS = 640

# Only classes whose source meaning is a pose-only rule-engine verdict.
KAGGLE_LABEL_MAP: dict[str, str] = {
    "normal act": "P01",       # writing / reading / facing forward
    "looking friend": "P20",   # looking sideways at a neighbour
}

# Documented exclusions.  Keeping this explicit is the difference between a
# real model card and a fabricated 40-class claim.
KAGGLE_EXCLUDED: dict[str, str] = {
    "giving code": (
        "hand-to-face signalling / possible answer passing has no exact "
        "pose-only verdict; object and interaction context are required"
    ),
    "giving object": (
        "passing an object is object-dependent and only 15 images are present"
    ),
    "cheating": (
        "phone use is detected by the desk-object path, not by pose alone; "
        "only 12 images are present"
    ),
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _fixed_canvas(
    frame: np.ndarray, size: int = FIXED_CANVAS
) -> tuple[np.ndarray, float, int, int]:
    """Letterbox *frame* onto a fixed square canvas.

    A fixed tensor shape avoids per-image cuDNN autotuning, which made the
    first inference on every new screenshot shape take ~5 s instead of ~0.05 s.
    """
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("empty frame")
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=frame.dtype)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _map_box(
    box: Iterable[float],
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> list[int]:
    x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return [x1, y1, x2, y2]


def _map_keypoints(
    keypoints: np.ndarray | None,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray | None:
    if keypoints is None:
        return None
    mapped = np.array(keypoints, dtype=np.float64, copy=True)
    if mapped.ndim != 2 or mapped.shape[1] < 2:
        return None
    mapped[:, 0] = (mapped[:, 0] - pad_x) / scale
    mapped[:, 1] = (mapped[:, 1] - pad_y) / scale
    return mapped


def _box_area(box: Iterable[float]) -> float:
    values = list(box)
    return max(0.0, float(values[2]) - float(values[0])) * max(
        0.0, float(values[3]) - float(values[1])
    )


# COCO-17 -> MediaPipe-33 positions.  MediaPipe's mouth corners are not in
# COCO, so they deliberately stay missing and the head-pose solver returns
# "unavailable" rather than a fabricated angle.
_COCO_TO_MP: dict[int, int] = {
    0: 0,    # nose
    1: 3,    # left eye
    2: 6,    # right eye
    3: 7,    # left ear
    4: 8,    # right ear
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}


def _coco_to_mediapipe(
    keypoints: np.ndarray | None,
) -> list[tuple[float, float, float]]:
    landmarks: list[tuple[float, float, float]] = [
        (float("nan"), float("nan"), 0.0) for _ in range(33)
    ]
    if keypoints is None:
        return landmarks
    rows = np.asarray(keypoints, dtype=np.float64)
    for coco_index, mp_index in _COCO_TO_MP.items():
        if coco_index >= rows.shape[0]:
            continue
        row = rows[coco_index]
        if row.size < 2:
            continue
        x, y = float(row[0]), float(row[1])
        confidence = float(row[2]) if row.size >= 3 else 1.0
        if not (np.isfinite(x) and np.isfinite(y) and confidence >= 0.25):
            continue
        landmarks[mp_index] = (x, y, confidence)
    return landmarks


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class OfflineExtractor:
    """One :class:`PoseClassifier` shared by all offline sources."""

    def __init__(self, use_mediapipe: bool = True) -> None:
        self.use_mediapipe = use_mediapipe
        self.classifier = PoseClassifier()

    def _detect(self, canvas: np.ndarray):
        model = self.classifier._yolo
        if model is None:
            return [], []
        kwargs = dict(
            imgsz=640,
            max_det=8,
            device=self.classifier._device_arg,
            quantize=16 if self.classifier._half else None,
            verbose=False,
        )
        results = model(canvas, **kwargs)
        result = results[0]
        boxes = (
            result.boxes.xyxy.cpu().numpy().tolist()
            if result.boxes is not None
            else []
        )
        keypoints = (
            result.keypoints.data.cpu().numpy()
            if result.keypoints is not None
            else np.empty((0, 17, 3))
        )
        return boxes, keypoints

    def extract(
        self,
        frame: np.ndarray,
        pose_id: str,
        source: str,
    ) -> tuple[dict[str, Any] | None, str]:
        """Extract the primary (largest) candidate and return (result, mode)."""
        h, w = frame.shape[:2]
        canvas, scale, pad_x, pad_y = _fixed_canvas(frame)
        raw_boxes, raw_keypoints = self._detect(canvas)
        if not raw_boxes:
            return None, "no_person"

        mapped_boxes = [
            _map_box(box, scale, pad_x, pad_y, w, h) for box in raw_boxes
        ]
        primary = max(
            range(len(mapped_boxes)),
            key=lambda i: _box_area(mapped_boxes[i]),
        )
        box = mapped_boxes[primary]
        keypoints = (
            _map_keypoints(
                raw_keypoints[primary] if primary < len(raw_keypoints) else None,
                scale,
                pad_x,
                pad_y,
            )
            if len(raw_keypoints) > primary
            else None
        )

        landmarks: list[tuple[float, float, float]] = []
        head_pose: dict[str, float] = {
            "yaw": float("nan"),
            "pitch": float("nan"),
            "roll": float("nan"),
        }
        mode = "coco_only"

        if self.use_mediapipe and keypoints is not None:
            found = self.classifier._mediapipe_landmarks(frame, [box])
            if found:
                landmarks = found[0]
                hp = self.classifier._head_pose_estimator.estimate(
                    landmarks, image_shape=(h, w)
                )
                # rvec is None when the face landmarks needed for PnP were not
                # measured; the estimator's zeros are "unavailable", not a
                # measured neutral pose.
                if hp.get("rvec") is not None:
                    head_pose = {
                        "yaw": float(hp.get("yaw", 0.0)),
                        "pitch": float(hp.get("pitch", 0.0)),
                        "roll": float(hp.get("roll", 0.0)),
                    }
                mode = "mediapipe"

        if not landmarks:
            landmarks = _coco_to_mediapipe(keypoints)

        geometric = self.classifier._compute_geometric_features(landmarks)
        spatial = self.classifier._compute_spatial_features(
            [box], frame.shape
        )
        spatial.pop("person_count", None)

        result = {
            "pose_id": pose_id,
            "confidence": 1.0,
            "bounding_boxes": [box],
            "keypoints": [keypoints] if keypoints is not None else [],
            "head_pose": head_pose,
            "geometric_features": geometric,
            "spatial_features": spatial,
            "source": source,
        }
        return result, mode


# ---------------------------------------------------------------------------
# Source: Kaggle ExamCheatingDataset (real labelled rows)
# ---------------------------------------------------------------------------

def _image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def ingest_kaggle(
    extractor: OfflineExtractor,
    logger_: DatasetLogger,
    max_per_class: int,
    seed: int,
) -> dict[str, Any]:
    if not KAGGLE_ROOT.is_dir():
        return {"status": "missing", "root": str(KAGGLE_ROOT)}

    rng = random.Random(seed)
    counts: dict[str, int] = {}
    modes: dict[str, int] = {}
    skipped: dict[str, str] = {}
    attempted = 0

    for folder_name, pose_id in KAGGLE_LABEL_MAP.items():
        files = _image_files(KAGGLE_ROOT / "train" / folder_name)
        rng.shuffle(files)
        if max_per_class > 0:
            files = files[:max_per_class]
        logger.info(
            "[kaggle] %s -> %s: %d image(s)", folder_name, pose_id, len(files)
        )
        for image_path in files:
            attempted += 1
            frame = cv2.imread(str(image_path))
            if frame is None:
                skipped["unreadable"] = skipped.get("unreadable", 0) + 1
                continue
            result, mode = extractor.extract(
                frame,
                pose_id=pose_id,
                source=f"kaggle_exam/{folder_name}",
            )
            modes[mode] = modes.get(mode, 0) + 1
            if result is None:
                skipped[mode] = skipped.get(mode, 0) + 1
                continue
            logger_.log_frame(result, motion_delta=0.0)
            counts[pose_id] = counts.get(pose_id, 0) + 1
            if attempted % 50 == 0:
                logger.info("[kaggle] processed %d image(s)", attempted)

    for folder_name, reason in KAGGLE_EXCLUDED.items():
        skipped[f"excluded:{folder_name}"] = reason

    return {
        "status": "ok",
        "attempted": attempted,
        "rows_by_label": counts,
        "extraction_modes": modes,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Source: SCUT-HEAD (occupancy only, never P labels)
# ---------------------------------------------------------------------------

def _read_yolo_labels(path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _cid, cx, cy, nw, nh = (
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        except ValueError:
            continue
        boxes.append((cx, cy, max(0.0, nw), max(0.0, nh)))
    return boxes


def ingest_scut_occupancy(out_path: Path) -> dict[str, Any]:
    if not SCUT_ROOT.is_dir():
        return {"status": "missing", "root": str(SCUT_ROOT)}

    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        image_dir = SCUT_ROOT / "yolo" / "images" / split
        label_dir = SCUT_ROOT / "yolo" / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in _image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = _read_yolo_labels(label_path)
            if not boxes:
                continue
            areas = [nw * nh for _cx, _cy, nw, nh in boxes]
            count = len(boxes)
            rows.append(
                {
                    "image": str(image_path.relative_to(SCUT_ROOT)),
                    "split": split,
                    "head_count": count,
                    "mean_area_ratio": round(float(np.mean(areas)), 6),
                    "max_area_ratio": round(float(np.max(areas)), 6),
                    "density": round(float(np.sum(areas)), 6),
                    "mean_center_x": round(
                        float(np.mean([b[0] for b in boxes])), 6
                    ),
                    "mean_center_y": round(
                        float(np.mean([b[1] for b in boxes])), 6
                    ),
                    "source": "scut_head_yolo_labels",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    fieldnames = [
        "image", "split", "head_count", "mean_area_ratio",
        "max_area_ratio", "density", "mean_center_x", "mean_center_y",
        "source",
    ]
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(out_path)

    by_split: dict[str, int] = {}
    for row in rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    return {
        "status": "ok",
        "images": len(rows),
        "rows_by_split": by_split,
        "output": str(out_path),
    }


# ---------------------------------------------------------------------------
# Source: COCO keypoints (unlabelled pretraining only)
# ---------------------------------------------------------------------------

def ingest_coco_unlabelled(
    extractor: OfflineExtractor,
    out_path: Path,
    coco_root: Path,
    limit: int = 0,
) -> dict[str, Any]:
    ann_dir = coco_root / "annotations"
    if not ann_dir.is_dir():
        return {"status": "missing", "root": str(coco_root)}

    ann_files = sorted(ann_dir.glob("person_keypoints_*.json"))
    if not ann_files:
        return {"status": "missing_annotations", "root": str(coco_root)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    logger_ = DatasetLogger(tmp_path)
    logger_.start_recording()
    written = 0
    considered = 0

    for ann_file in ann_files:
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        images = {int(img["id"]): img for img in data.get("images", [])}
        for annotation in data.get("annotations", []):
            if limit and written >= limit:
                break
            if int(annotation.get("num_keypoints", 0)) < 5:
                continue
            image_meta = images.get(int(annotation["image_id"]))
            if not image_meta:
                continue
            candidate_dirs = [
                coco_root / "train2017",
                coco_root / "val2017",
                coco_root / "images",
            ]
            image_path = next(
                (
                    folder / image_meta["file_name"]
                    for folder in candidate_dirs
                    if (folder / image_meta["file_name"]).is_file()
                ),
                None,
            )
            if image_path is None:
                continue
            kpts = np.asarray(
                annotation.get("keypoints", []), dtype=np.float64
            ).reshape(-1, 3)
            if kpts.shape[0] < 17:
                continue
            x, y, bw, bh = (float(v) for v in annotation["bbox"])
            box = [
                int(round(x)),
                int(round(y)),
                int(round(x + bw)),
                int(round(y + bh)),
            ]
            landmarks = _coco_to_mediapipe(kpts)
            geometric = extractor.classifier._compute_geometric_features(
                landmarks
            )
            spatial = extractor.classifier._compute_spatial_features(
                [box], (int(image_meta["height"]), int(image_meta["width"]), 3)
            )
            spatial.pop("person_count", None)
            result = {
                "pose_id": "UNLABELED",
                "confidence": 0.0,
                "bounding_boxes": [box],
                "keypoints": [kpts],
                "head_pose": {
                    "yaw": float("nan"),
                    "pitch": float("nan"),
                    "roll": float("nan"),
                },
                "geometric_features": geometric,
                "spatial_features": spatial,
                "source": "coco_person_keypoints",
            }
            logger_.log_frame(result)
            written += 1
            considered += 1
        if limit and written >= limit:
            break

    logger_.close()
    if written == 0:
        tmp_path.unlink(missing_ok=True)
        return {
            "status": "no_rows",
            "annotations_found": len(ann_files),
            "root": str(coco_root),
        }
    tmp_path.replace(out_path)
    return {
        "status": "ok",
        "rows": written,
        "output": str(out_path),
        "note": "UNLABELED rows are for pretraining only, never supervised classes",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sources(value: str) -> set[str]:
    parsed = {part.strip() for part in value.split(",") if part.strip()}
    if "all" in parsed:
        return {"kaggle-exam", "scut-head", "coco", "mpii", "dmd", "desk"}
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest modular public datasets into the 54-D schema"
    )
    parser.add_argument(
        "--sources",
        default="all",
        help="comma list: kaggle-exam,scut-head,coco,mpii,dmd,desk,all",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--occupancy-out", type=Path, default=DEFAULT_OCCUPANCY
    )
    parser.add_argument(
        "--unlabeled-out", type=Path, default=DEFAULT_UNLABELED
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="cap images per labelled class (0 = all)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coco-root", type=Path, default=DATASETS / "COCO")
    parser.add_argument(
        "--no-mediapipe", action="store_true",
        help="use YOLO/COCO keypoints only (faster; head pose becomes NaN)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="replace the output CSV instead of appending",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    selected = _sources(args.sources)
    summary: dict[str, Any] = {"sources": sorted(selected)}
    needs_models = bool(selected & {"kaggle-exam", "coco"})
    extractor = (
        OfflineExtractor(use_mediapipe=not args.no_mediapipe)
        if needs_models
        else None
    )

    if "scut-head" in selected:
        summary["scut_head"] = ingest_scut_occupancy(args.occupancy_out)
        logger.info("SCUT-HEAD occupancy: %s", summary["scut_head"])

    if "coco" in selected:
        assert extractor is not None
        summary["coco"] = ingest_coco_unlabelled(
            extractor, args.unlabeled_out, args.coco_root, args.limit
        )
        logger.info("COCO unlabelled: %s", summary["coco"])

    for name, reason in {
        "mpii": (
            "MPII's 16-joint layout has no COCO face keypoints and cannot be "
            "aligned to the 54-D schema without a learned joint mapping"
        ),
        "dmd": (
            "no DMD frames/annotations are present; its head-pose labels also "
            "need P12 for Looking Away (P16 is Absent Seat in this codebase)"
        ),
        "desk": (
            "data/desk contains desk.yaml but no images/labels, so wrist-object "
            "features cannot be extracted"
        ),
    }.items():
        if name in selected:
            summary[name] = {"status": "skipped", "reason": reason}
            logger.warning("%s skipped: %s", name, reason)

    if "kaggle-exam" in selected:
        assert extractor is not None
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")
        if args.rebuild and tmp_path.exists():
            tmp_path.unlink()
        logger_ = DatasetLogger(tmp_path)
        logger_.start_recording()
        try:
            summary["kaggle_exam"] = ingest_kaggle(
                extractor, logger_, args.max_per_class, args.seed
            )
        finally:
            logger_.close()
        rows = logger_.sample_count
        if rows > 0:
            tmp_path.replace(args.out)
            summary["kaggle_exam"]["output"] = str(args.out)
            summary["kaggle_exam"]["rows_written"] = rows
            # Any cached feature matrix for the old CSV is now invalid.
            cache = args.out.with_suffix(args.out.suffix + ".feature_cache.npz")
            cache.unlink(missing_ok=True)
        else:
            tmp_path.unlink(missing_ok=True)
            summary["kaggle_exam"]["rows_written"] = 0
        logger.info("Kaggle labelled rows: %d", rows)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
