"""
benchmark_head_detector.py — evaluate a YOLO head detector on SCUT-HEAD.

SCUT-HEAD has head bounding boxes only.  The honest question it can answer is
"how well does the detector find and count heads?", not "does it detect
cheating?".  This script reports box-level precision/recall/F1 at a matching
IoU, head-count MAE/bias, and exact-count agreement.

It can also run the stock YOLO11n-pose model as a proxy baseline.  That model
was trained for full-body pose, not heads, so its numbers are labelled as a
proxy: fraction of GT head centres covered by a person box, and person-count
error.  It is there to show why SCUT-HEAD needs its own head detector.

Usage:
    python tools/benchmark_head_detector.py --model models/head/scut_head_yolo11n.pt
    python tools/benchmark_head_detector.py --limit 100 --split test
    python tools/benchmark_head_detector.py --baseline-pose backend/yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "datasets" / "SCUT_HEAD" / "manifest.csv"
DEFAULT_OUTPUT = ROOT / "datasets" / "SCUT_HEAD" / "benchmark"


def _box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _match(
    gt: list[list[float]], pred: list[list[float]], iou_thr: float
) -> tuple[int, int, int]:
    """Greedy confidence-ordered matching.  Returns (TP, FP, FN)."""
    used: set[int] = set()
    tp = 0
    for box in sorted(pred, key=lambda item: item[4], reverse=True):
        best_i, best_iou = -1, 0.0
        for i, target in enumerate(gt):
            if i in used:
                continue
            iou = _box_iou(box, target)
            if iou > best_iou:
                best_i, best_iou = i, iou
        if best_i >= 0 and best_iou >= iou_thr:
            used.add(best_i)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def _load_rows(
    manifest: Path, parts: set[str], split: str, limit: int, seed: int
) -> list[dict[str, str]]:
    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("part", "")).strip().upper() in parts
            and str(row.get("split", "")).strip().lower() == split
        ]
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return sorted(rows, key=lambda row: row.get("image", ""))


def _metrics(
    tp: int, fp: int, fn: int, images: int,
    count_abs: float, count_signed: float, exact: int,
    gt_heads: int, pred_heads: int,
) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": images,
        "gt_heads": gt_heads,
        "pred_heads": pred_heads,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "count_mae": round(count_abs / images, 3) if images else 0.0,
        "count_bias": round(count_signed / images, 3) if images else 0.0,
        "exact_count_match": round(exact / images, 4) if images else 0.0,
    }


def _head_class_ids(model: Any) -> set[int]:
    names = getattr(model, "names", None) or {}
    if isinstance(names, (list, tuple)):
        names = {i: name for i, name in enumerate(names)}
    ids = {
        int(cid) for cid, name in names.items()
        if "head" in str(name).lower()
    }
    return ids or {0}


def _run_head_model(
    model_path: Path,
    rows: list[dict[str, str]],
    conf: float,
    nms_iou: float,
    match_iou: float,
    imgsz: int,
    max_det: int,
    device: str,
    half: bool = False,
) -> dict[str, Any]:
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    head_ids = _head_class_ids(model)
    groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "images": 0, "gt_heads": 0, "pred_heads": 0,
            "tp": 0, "fp": 0, "fn": 0,
            "count_abs": 0.0, "count_signed": 0.0, "exact": 0,
        }
    )
    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        image_path = Path(row.get("image", ""))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        try:
            gt = json.loads(row.get("boxes", "[]") or "[]")
        except json.JSONDecodeError:
            gt = []

        result = model(
            frame, imgsz=imgsz, conf=conf, iou=nms_iou,
            max_det=max_det, classes=sorted(head_ids),
            device=device, quantize=16 if half else None, verbose=False,
        )[0]
        pred: list[list[float]] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(np.int32)
            keep = np.isin(classes, list(head_ids))
            for box, score in zip(boxes[keep], scores[keep]):
                pred.append([
                    float(box[0]), float(box[1]),
                    float(box[2]), float(box[3]), float(score),
                ])

        tp, fp, fn = _match(gt, pred, match_iou)
        group_keys = [
            "overall",
            f"{row.get('part', '?')}/{row.get('split', '?')}",
            f"part_{row.get('part', '?')}",
        ]
        for key in group_keys:
            stats = groups[key]
            stats["images"] += 1
            stats["gt_heads"] += len(gt)
            stats["pred_heads"] += len(pred)
            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn
            stats["count_abs"] += abs(len(pred) - len(gt))
            stats["count_signed"] += len(pred) - len(gt)
            stats["exact"] += int(len(pred) == len(gt))

        if index % 100 == 0:
            print(f"[head-bench] {index}/{len(rows)} images")

    elapsed = time.perf_counter() - started
    metrics = {
        key: _metrics(
            int(stats["tp"]), int(stats["fp"]), int(stats["fn"]),
            int(stats["images"]), stats["count_abs"], stats["count_signed"],
            int(stats["exact"]), int(stats["gt_heads"]),
            int(stats["pred_heads"]),
        )
        for key, stats in groups.items()
    }
    return {
        "model": str(model_path),
        "class_ids": sorted(head_ids),
        "conf": conf,
        "nms_iou": nms_iou,
        "match_iou": match_iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
    }


def _run_pose_proxy(
    model_path: Path,
    rows: list[dict[str, str]],
    conf: float,
    imgsz: int,
    device: str,
    half: bool = False,
) -> dict[str, Any]:
    """Measure how poorly a stock pose model covers SCUT-HEAD head boxes."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "images": 0, "gt_heads": 0, "covered": 0,
            "person_boxes": 0, "count_abs": 0.0,
        }
    )
    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        image_path = Path(row.get("image", ""))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        try:
            gt = json.loads(row.get("boxes", "[]") or "[]")
        except json.JSONDecodeError:
            gt = []
        result = model(
            frame, imgsz=imgsz, conf=conf, device=device,
            quantize=16 if half else None,
            verbose=False,
        )[0]
        persons: list[list[float]] = []
        if result.boxes is not None:
            persons = [
                [float(v) for v in box]
                for box in result.boxes.xyxy.cpu().numpy()
            ]
        covered = 0
        for x1, y1, x2, y2 in gt:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if any(
                p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in persons
            ):
                covered += 1

        for key in ("overall", f"{row.get('part', '?')}/{row.get('split', '?')}"):
            stats = groups[key]
            stats["images"] += 1
            stats["gt_heads"] += len(gt)
            stats["covered"] += covered
            stats["person_boxes"] += len(persons)
            stats["count_abs"] += abs(len(persons) - len(gt))

        if index % 100 == 0:
            print(f"[pose-proxy] {index}/{len(rows)} images")

    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {}
    for key, stats in groups.items():
        images = max(1, int(stats["images"]))
        gt_heads = max(1, int(stats["gt_heads"]))
        metrics[key] = {
            "images": int(stats["images"]),
            "gt_heads": int(stats["gt_heads"]),
            "person_boxes": int(stats["person_boxes"]),
            "head_centre_coverage": round(stats["covered"] / gt_heads, 4),
            "count_mae": round(stats["count_abs"] / images, 3),
        }
    return {
        "model": str(model_path),
        "note": (
            "Proxy only: YOLO11n-pose detects full bodies. "
            "head_centre_coverage is the fraction of GT head centres inside "
            "any person box; it is NOT a head-detector score."
        ),
        "conf": conf,
        "imgsz": imgsz,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
    }


def _order_keys(metrics: dict[str, Any]) -> list[str]:
    keys = sorted(metrics)
    if "overall" in keys:
        keys.remove("overall")
        keys.insert(0, "overall")
    return keys


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    head = report["head"]
    lines = [
        "# SCUT-HEAD head-detection benchmark",
        "",
        "SCUT-HEAD labels are head bounding boxes only. These numbers measure "
        "head detection and counting, not cheating/behaviour classification.",
        "",
        f"- Model: `{head['model']}`",
        f"- Split: `{report['split']}`",
        f"- Images evaluated: {report['metrics']['overall']['images']}",
        f"- Confidence: {head['conf']}",
        f"- NMS IoU: {head['nms_iou']}",
        f"- Matching IoU: {head['match_iou']}",
        f"- Input size: {head['imgsz']}",
        "",
        "| Group | Images | GT heads | Pred heads | TP | FP | FN | Precision | Recall | F1 | Count MAE | Exact count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in _order_keys(report["metrics"]):
        m = report["metrics"][key]
        lines.append(
            f"| {key} | {m['images']} | {m['gt_heads']} | {m['pred_heads']} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['count_mae']:.3f} | "
            f"{m['exact_count_match']:.4f} |"
        )

    pose = report.get("pose_proxy")
    if pose:
        lines += [
            "",
            "## Pose-model proxy baseline",
            "",
            pose["note"],
            "",
            "| Group | Images | GT heads | Person boxes | Head-centre coverage | Count MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key in _order_keys(pose["metrics"]):
            m = pose["metrics"][key]
            lines.append(
                f"| {key} | {m['images']} | {m['gt_heads']} | "
                f"{m['person_boxes']} | {m['head_centre_coverage']:.4f} | "
                f"{m['count_mae']:.3f} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_parts(value: str) -> set[str]:
    parts = {item.strip().upper() for item in value.split(",") if item.strip()}
    unknown = parts - {"A", "B"}
    if unknown:
        raise SystemExit(f"unknown part(s): {', '.join(sorted(unknown))}")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a YOLO head detector on SCUT-HEAD."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "models" / "head" / "scut_head_yolo11n.pt",
    )
    parser.add_argument("--parts", default="A,B")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--limit", type=int, default=0,
        help="randomly sample at most N images (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--half", action=argparse.BooleanOptionalAction, default=None,
        help="FP16 inference on CUDA (default: on)",
    )
    parser.add_argument(
        "--baseline-pose", type=Path, default=None,
        help="optional YOLO pose weights for the proxy baseline",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(
            f"head model not found: {model_path}\n"
            "Train one first: python tools/train_head_detector.py"
        )
    if not args.manifest.is_file():
        raise SystemExit(
            f"manifest not found: {args.manifest}. "
            "Run tools/scut_head_ingest.py first."
        )

    parts = _parse_parts(args.parts)
    rows = _load_rows(
        args.manifest.resolve(), parts, args.split, args.limit, args.seed,
    )
    if not rows:
        raise SystemExit("no manifest rows matched the requested split")

    try:
        from backend.runtime import resolve_device, yolo_device
    except ImportError:
        sys.path.insert(0, str(ROOT))
        from backend.runtime import resolve_device, yolo_device

    device_obj = resolve_device(args.device)
    device = yolo_device(device_obj)
    half = (
        bool(args.half)
        if args.half is not None
        else device_obj.type == "cuda"
    )

    print(
        f"[head-bench] model={model_path} split={args.split} "
        f"images={len(rows)} device={device} half={half}"
    )
    head = _run_head_model(
        model_path, rows, args.conf, args.nms_iou, args.match_iou,
        args.imgsz, args.max_det, device, half,
    )
    head_metrics = head["metrics"]

    pose = None
    if args.baseline_pose is not None:
        pose_path = args.baseline_pose.resolve()
        if not pose_path.is_file():
            raise SystemExit(f"pose baseline not found: {pose_path}")
        print(f"[pose-proxy] model={pose_path}")
        pose = _run_pose_proxy(pose_path, rows, 0.25, args.imgsz, device, half)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "SCUT-HEAD",
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "parts": sorted(parts),
        "requested_images": len(rows),
        "limit": args.limit,
        "seed": args.seed,
        "device": device,
        "half": half,
        "metrics": head_metrics,
        "head": head,
        "pose_proxy": pose,
    }
    json_path = output / "head_benchmark.json"
    md_path = output / "head_benchmark.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)

    overall = head_metrics["overall"]
    print("[head-bench] overall")
    print(f"  images    : {overall['images']}")
    print(f"  GT heads  : {overall['gt_heads']}")
    print(f"  predicted : {overall['pred_heads']}")
    print(f"  precision : {overall['precision']:.4f}")
    print(f"  recall    : {overall['recall']:.4f}")
    print(f"  F1        : {overall['f1']:.4f}")
    print(f"  count MAE : {overall['count_mae']:.3f}")
    print(f"  exact     : {overall['exact_count_match']:.4f}")
    print(f"  json      : {json_path}")
    print(f"  markdown  : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
