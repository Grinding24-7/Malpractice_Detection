"""
tracking_metrics.py — Week 11: Tracking Stability Benchmark.

Evaluates ByteTrack association performance over dense test video clips:

    1. Identity Switches (ID Swaps)
       Count instances where a candidate's assigned track_id changes during
       an active video stream.  An ID swap is recorded when the same spatial
       region (IoU > 0.5 with a previous detection) receives a new track_id
       while the old track_id is still active within a temporal gap.

    2. Keypoint Completeness
       Percentage of frames retaining valid 17-point COCO pose predictions
       under desk occlusions.  Measured per-track as the ratio of frames
       with >= N valid keypoints (confidence > 0.3) out of total frames.

Usage:
    python tracking_metrics.py
    python tracking_metrics.py --video path/to/test.mp4 --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("tracking")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "tracking_results"
COCO_KEYPOINTS = 17
MIN_KEYPOINTS_VALID = 5  # minimum keypoints with conf > 0.3 for a valid detection
IOU_SWAP_THRESHOLD = 0.5
TEMPORAL_GAP_FRAMES = 10  # frames within which an IoU match triggers an ID swap


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-Union for two (4,) xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Track history data structure
# ---------------------------------------------------------------------------

@dataclass
class TrackEvent:
    frame_idx: int
    track_id: int
    box: np.ndarray  # (4,) xyxy
    n_valid_keypoints: int
    n_total_keypoints: int
    confidence: float


@dataclass
class TrackHistory:
    """Complete tracking history for one video clip."""
    events: list[TrackEvent] = field(default_factory=list)
    frame_count: int = 0

    def add(self, event: TrackEvent) -> None:
        self.events.append(event)

    def tracks_at_frame(self, frame_idx: int) -> list[TrackEvent]:
        return [e for e in self.events if e.frame_idx == frame_idx]

    def unique_track_ids(self) -> set[int]:
        return {e.track_id for e in self.events}

    def frames_with_detections(self) -> int:
        return len({e.frame_idx for e in self.events})


# ---------------------------------------------------------------------------
# ID Swap detection
# ---------------------------------------------------------------------------

def count_id_swaps(history: TrackHistory, temporal_gap: int = TEMPORAL_GAP_FRAMES) -> dict:
    """
    Count identity switches across the tracking history.

    An ID swap occurs when:
        1. Two detections at different frames have IoU > threshold.
        2. They have different track_ids.
        3. The temporal distance between them is <= temporal_gap frames.
        4. The older track_id is no longer active at the newer frame.

    Returns:
        dict with keys: n_swaps, swap_details (list of dicts).
    """
    # Group events by frame
    frame_to_events: dict[int, list[TrackEvent]] = defaultdict(list)
    for e in history.events:
        frame_to_events[e.frame_idx].append(e)

    sorted_frames = sorted(frame_to_events.keys())
    active_ids: dict[int, int] = {}  # track_id -> last_frame_seen
    swap_details: list[dict] = []
    n_swaps = 0

    for frame_idx in sorted_frames:
        current_events = frame_to_events[frame_idx]

        # Check for IoU matches with recent past
        for curr in current_events:
            for past_frame in range(max(0, frame_idx - temporal_gap), frame_idx):
                for past in frame_to_events.get(past_frame, []):
                    if curr.track_id == past.track_id:
                        continue
                    iou = _compute_iou(curr.box, past.box)
                    if iou > IOU_SWAP_THRESHOLD:
                        # Check if the old track is no longer active
                        if active_ids.get(past.track_id, -1) < frame_idx - 1:
                            n_swaps += 1
                            swap_details.append({
                                "frame": frame_idx,
                                "old_track_id": int(past.track_id),
                                "new_track_id": int(curr.track_id),
                                "iou": round(float(iou), 4),
                                "gap_frames": frame_idx - past.frame_idx,
                            })
                            break

        # Update active IDs
        for curr in current_events:
            active_ids[curr.track_id] = frame_idx

    return {"n_swaps": n_swaps, "swap_details": swap_details}


# ---------------------------------------------------------------------------
# Keypoint completeness
# ---------------------------------------------------------------------------

def compute_keypoint_completeness(
    history: TrackHistory,
    min_valid_kpts: int = MIN_KEYPOINTS_VALID,
) -> dict:
    """
    Measure the percentage of frames with valid keypoint predictions.

    A frame is "complete" if the detection has >= min_valid_kpts keypoints
    with confidence > 0.3.

    Returns:
        dict with keys:
            overall_completeness: float (0-1)
            per_track: dict[track_id -> completeness]
            total_frames: int
            complete_frames: int
            tracks_analysed: int
    """
    total_frames = history.frame_count
    frames_with_valid_detection: set[int] = set()

    per_track: dict[int, dict] = {}

    track_ids = history.unique_track_ids()
    for tid in track_ids:
        track_events = [e for e in history.events if e.track_id == tid]
        track_frames = {e.frame_idx for e in track_events}
        complete = sum(
            1 for e in track_events if e.n_valid_keypoints >= min_valid_kpts
        )
        total = len(track_events)
        completeness = complete / total if total > 0 else 0.0
        per_track[int(tid)] = {
            "completeness": round(completeness, 4),
            "total_frames": total,
            "complete_frames": complete,
        }
        # Mark frames as having valid detections
        for e in track_events:
            if e.n_valid_keypoints >= min_valid_kpts:
                frames_with_valid_detection.add(e.frame_idx)

    complete_count = len(frames_with_valid_detection)
    overall = complete_count / total_frames if total_frames > 0 else 0.0

    return {
        "overall_completeness": round(overall, 4),
        "total_frames": total_frames,
        "complete_frames": complete_count,
        "tracks_analysed": len(track_ids),
        "per_track": per_track,
    }


# ---------------------------------------------------------------------------
# Synthetic tracking data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_tracking(
    n_frames: int = 150,
    n_candidates: int = 3,
    id_swap_prob: float = 0.05,
    occlusion_prob: float = 0.15,
    seed: int = 42,
) -> TrackHistory:
    """
    Generate synthetic ByteTrack-like tracking data for benchmarking.

    Simulates:
        - Multiple candidates tracked across frames
        - Occasional ID switches (track_id reassignment)
        - Occlusion events (keypoint dropout)
    """
    rng = np.random.default_rng(seed)
    history = TrackHistory(frame_count=n_frames)

    # Simulate candidate trajectories
    candidates = []
    for c in range(n_candidates):
        base_x = 100 + c * 300
        base_y = 200 + c * 50
        candidates.append({
            "cx": base_x, "cy": base_y,
            "vx": rng.uniform(-2, 2), "vy": rng.uniform(-0.5, 0.5),
        })

    current_ids = list(range(n_candidates))

    for frame in range(n_frames):
        for c_idx in range(n_candidates):
            cand = candidates[c_idx]
            # Random walk
            cand["cx"] += cand["vx"] + rng.normal(0, 3)
            cand["cy"] += cand["vy"] + rng.normal(0, 2)

            # Generate box
            box_w, box_h = 80, 120
            box = np.array([
                cand["cx"] - box_w / 2, cand["cy"] - box_h / 2,
                cand["cx"] + box_w / 2, cand["cy"] + box_h / 2,
            ])

            # Keypoints
            n_valid = COCO_KEYPOINTS
            if rng.random() < occlusion_prob:
                n_valid = int(rng.integers(MIN_KEYPOINTS_VALID, COCO_KEYPOINTS))

            # ID swap
            track_id = current_ids[c_idx]
            if rng.random() < id_swap_prob and frame > 5:
                track_id = int(rng.integers(100, 200))
                current_ids[c_idx] = track_id

            confidence = float(rng.uniform(0.4, 0.95))

            history.add(TrackEvent(
                frame_idx=frame,
                track_id=track_id,
                box=box,
                n_valid_keypoints=n_valid,
                n_total_keypoints=COCO_KEYPOINTS,
                confidence=confidence,
            ))

    return history


# ---------------------------------------------------------------------------
# Real video evaluation (requires YOLO + ByteTrack)
# ---------------------------------------------------------------------------

def evaluate_real_video(video_path: str, max_frames: int = 300) -> TrackHistory:
    """
    Run YOLO11n-pose + ByteTrack on a real video and build a TrackHistory.

    Args:
        video_path: path to an MP4/AVI video file.
        max_frames: maximum number of frames to process.

    Returns:
        TrackHistory with per-frame detection events.
    """
    import cv2

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics required for real video evaluation")

    model = YOLO(str(BACKEND_DIR / "yolo11n-pose.pt"))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    history = TrackHistory(frame_count=0)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False,
        )
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(np.int64)
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(ids))
            kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

            for i in range(len(ids)):
                n_valid = 0
                if kpts is not None and i < kpts.shape[0]:
                    confs_kpts = kpts[i, :, 2] if kpts.shape[-1] >= 3 else np.ones(COCO_KEYPOINTS)
                    n_valid = int((confs_kpts > 0.3).sum())

                history.add(TrackEvent(
                    frame_idx=frame_idx,
                    track_id=int(ids[i]),
                    box=boxes[i],
                    n_valid_keypoints=n_valid,
                    n_total_keypoints=COCO_KEYPOINTS,
                    confidence=float(confs[i]),
                ))

        history.frame_count = frame_idx + 1
        frame_idx += 1

    cap.release()
    logger.info("Processed %d frames from %s", frame_idx, video_path)
    return history


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_tracking_report(
    id_swap_result: dict,
    completeness_result: dict,
    source: str,
) -> str:
    """Render a formatted tracking stability report."""
    lines = [
        "=" * 75,
        "  WEEK 11 — TRACKING STABILITY BENCHMARK RESULTS",
        f"  Source: {source}",
        "=" * 75,
        "",
        "  Identity Switches (ID Swaps)",
        "  " + "-" * 40,
        f"    Total ID Swaps:     {id_swap_result['n_swaps']}",
        f"    Events Recorded:    {len(id_swap_result['swap_details'])}",
        "",
    ]

    if id_swap_result["swap_details"]:
        lines.append(f"    {'Frame':>8}  {'Old ID':>8}  {'New ID':>8}  {'IoU':>8}  {'Gap':>5}")
        lines.append("    " + "-" * 45)
        for s in id_swap_result["swap_details"][:10]:
            lines.append(
                f"    {s['frame']:>8d}  {s['old_track_id']:>8d}  "
                f"{s['new_track_id']:>8d}  {s['iou']:>8.4f}  {s['gap_frames']:>5d}"
            )
        if len(id_swap_result["swap_details"]) > 10:
            lines.append(f"    ... and {len(id_swap_result['swap_details']) - 10} more")

    lines.extend([
        "",
        "  Keypoint Completeness",
        "  " + "-" * 40,
        f"    Overall Completeness:  {completeness_result['overall_completeness']:.2%}",
        f"    Total Frames:          {completeness_result['total_frames']}",
        f"    Complete Frames:       {completeness_result['complete_frames']}",
        f"    Tracks Analysed:       {completeness_result['tracks_analysed']}",
        "",
    ])

    per_track = completeness_result.get("per_track", {})
    if per_track:
        lines.append(f"    {'Track ID':>10}  {'Completeness':>14}  {'Frames':>8}  {'Complete':>10}")
        lines.append("    " + "-" * 48)
        for tid, info in sorted(per_track.items())[:10]:
            lines.append(
                f"    {tid:>10d}  {info['completeness']:>13.2%}  "
                f"{info['total_frames']:>8d}  {info['complete_frames']:>10d}"
            )

    lines.extend(["", "=" * 75])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 11 Tracking Stability Benchmark")
    parser.add_argument("--video", type=str, default=None, help="Path to test video")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--synthetic-frames", type=int, default=150)
    parser.add_argument("--synthetic-candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build tracking history ---
    if args.video and Path(args.video).exists():
        logger.info("Evaluating real video: %s", args.video)
        history = evaluate_real_video(args.video, max_frames=args.max_frames)
        source = args.video
    else:
        logger.info("No video provided — generating synthetic tracking data")
        history = _generate_synthetic_tracking(
            n_frames=args.synthetic_frames,
            n_candidates=args.synthetic_candidates,
            seed=args.seed,
        )
        source = f"synthetic ({args.synthetic_frames} frames, {args.synthetic_candidates} candidates)"

    # --- Compute metrics ---
    id_swap_result = count_id_swaps(history)
    completeness_result = compute_keypoint_completeness(history)

    # --- Print report ---
    report = format_tracking_report(id_swap_result, completeness_result, source)
    print(report)

    # --- Save JSON ---
    json_report = {
        "source": source,
        "id_swaps": {
            "total": id_swap_result["n_swaps"],
            "details": id_swap_result["swap_details"],
        },
        "keypoint_completeness": {
            "overall": completeness_result["overall_completeness"],
            "total_frames": completeness_result["total_frames"],
            "complete_frames": completeness_result["complete_frames"],
            "tracks_analysed": completeness_result["tracks_analysed"],
            "per_track": {
                str(k): v for k, v in completeness_result["per_track"].items()
            },
        },
    }
    json_path = output_dir / "tracking_report.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    logger.info("JSON report saved -> %s", json_path)


if __name__ == "__main__":
    main()
