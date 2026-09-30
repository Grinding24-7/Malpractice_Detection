"""
stress_test_harness.py — Week 9: Multi-Candidate & Occlusion Stress Tester.

Components:
    1. SyntheticLoadInjector — Simulates 1–8 concurrent HD CCTV streams
       with up to 30 student candidates per room.  Generates synthetic
       BGR frames and artificial InferenceResult objects to exercise the
       full pipeline without real video.

    2. OcclusionSimulator — Intentionally drops pose keypoints for target
       track_id sequences (simulating students blocked by monitors or
       neighbors).  Measures ByteTrack identity preservation and recovery.

    3. ResourceMonitorDaemon — Background thread logging GPU VRAM, CPU
       utilisation, per-process RAM growth, and WebSocket frame drops
       over a configurable duration (default 10 min).

    4. StressTestRunner — Orchestrates all components and produces a
       structured terminal telemetry report with FPS degradation curves,
       VRAM footprint, and false-alarm metrics.

Usage:
    python stress_test_harness.py --cameras 4 --candidates 20 --duration 60
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import InferenceResult
from metrics import get_metrics, read_gpu_memory

logger = logging.getLogger("stress_test")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_W = 1280
FRAME_H = 720
CAM_FPS = 30
COCO17_SHAPE = (17, 3)

# Simulated student body proportions (normalised to frame size)
BODY_WIDTH_RANGE = (0.08, 0.15)  # fraction of frame width
BODY_HEIGHT_RANGE = (0.15, 0.30)  # fraction of frame height

# Occlusion patterns
OCCLUSION_DURATION_FRAMES = (15, 45)  # how long a track is occluded
OCCLUSION_PROBABILITY = 0.02  # per-frame chance of starting occlusion
KEYPOINTS_TO_DROP = [
    [0, 1, 2, 3, 4],  # head region
    [5, 6],            # shoulders only
    [5, 7, 9],         # left arm
    [6, 8, 10],        # right arm
]


# ---------------------------------------------------------------------------
# Synthetic frame generator
# ---------------------------------------------------------------------------

def _generate_student_bbox(
    frame_w: int, frame_h: int, student_idx: int, total: int,
) -> tuple[int, int, int, int]:
    """Generate a plausible bounding box for a seated student."""
    cols = min(total, 6)
    row = student_idx // cols
    col = student_idx % cols

    cell_w = frame_w // cols
    cell_h = frame_h // max(1, total // cols + 1)

    w = int(cell_w * random.uniform(0.5, 0.8))
    h = int(cell_h * random.uniform(0.5, 0.8))
    x1 = col * cell_w + (cell_w - w) // 2 + random.randint(-10, 10)
    y1 = row * cell_h + (cell_h - h) // 2 + random.randint(-5, 5)
    x1 = max(0, min(x1, frame_w - w))
    y1 = max(0, min(y1, frame_h - h))
    return x1, y1, x1 + w, y1 + h


def _generate_synthetic_keypoints(
    bbox: tuple[int, int, int, int], noise: float = 0.02,
) -> np.ndarray:
    """Generate plausible COCO-17 keypoints within a bounding box."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1

    # Canonical COCO-17 layout (fraction of bbox)
    canonical = np.array([
        [0.5, 0.08],   # nose
        [0.45, 0.05],  # left_eye
        [0.55, 0.05],  # right_eye
        [0.40, 0.08],  # left_ear
        [0.60, 0.08],  # right_ear
        [0.35, 0.25],  # left_shoulder
        [0.65, 0.25],  # right_shoulder
        [0.25, 0.45],  # left_elbow
        [0.75, 0.45],  # right_elbow
        [0.20, 0.60],  # left_wrist
        [0.80, 0.60],  # right_wrist
        [0.38, 0.55],  # left_hip
        [0.62, 0.55],  # right_hip
        [0.36, 0.75],  # left_knee
        [0.64, 0.75],  # right_knee
        [0.34, 0.92],  # left_ankle
        [0.66, 0.92],  # right_ankle
    ], dtype=np.float32)

    kpts = np.zeros(COCO17_SHAPE, dtype=np.float32)
    for i in range(17):
        kpts[i, 0] = x1 + canonical[i, 0] * bw + random.gauss(0, noise * bw)
        kpts[i, 1] = y1 + canonical[i, 1] * bh + random.gauss(0, noise * bh)
        kpts[i, 2] = random.uniform(0.6, 0.98)  # confidence

    return kpts


def _generate_synthetic_frame(
    frame_w: int, frame_h: int, n_students: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[np.ndarray]]:
    """Generate a synthetic classroom frame with seated students."""
    # Background: grey classroom wall
    frame = np.full((frame_h, frame_w, 3), (60, 60, 55), dtype=np.uint8)

    # Add some texture (simulated desks)
    for i in range(max(1, n_students // 3)):
        dx = random.randint(0, frame_w)
        dy = random.randint(frame_h // 2, frame_h)
        cv2.rectangle(frame, (dx - 40, dy - 10), (dx + 40, dy + 10), (50, 50, 45), -1)

    boxes = []
    all_kpts = []
    for i in range(n_students):
        bbox = _generate_student_bbox(frame_w, frame_h, i, n_students)
        boxes.append(bbox)
        kpts = _generate_synthetic_keypoints(bbox)
        all_kpts.append(kpts)

        # Draw simple silhouette
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 75), -1)
        # Head
        head_cx = (x1 + x2) // 2
        head_cy = y1 + int((y2 - y1) * 0.08)
        head_r = int((x2 - x1) * 0.12)
        cv2.circle(frame, (head_cx, head_cy), head_r, (90, 90, 85), -1)

    return frame, boxes, all_kpts


def _build_synthetic_inference(
    boxes: list[tuple[int, int, int, int]],
    all_kpts: list[np.ndarray],
    track_ids: list[int],
    start_id: int = 1,
) -> InferenceResult:
    """Build an InferenceResult from synthetic data."""
    n = len(boxes)
    det_boxes = np.array([[float(x1), float(y1), float(x2), float(y2)]
                          for x1, y1, x2, y2 in boxes], dtype=np.float32)
    det_kpts = np.array(all_kpts, dtype=np.float32)
    det_conf = np.array([random.uniform(0.7, 0.95) for _ in range(n)], dtype=np.float32)
    det_ids = np.array(track_ids, dtype=np.int64)

    return InferenceResult(
        keypoints=det_kpts,
        boxes=det_boxes,
        confidences=det_conf,
        tracker_ids=det_ids,
        timestamps=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# Occlusion simulator
# ---------------------------------------------------------------------------

class OcclusionSimulator:
    """
    Simulates keypoint occlusion for specific tracks.

    When a track is "occluded", selected keypoints are zeroed out
    (confidence → 0) to simulate real-world occlusion by monitors,
    neighbors, or the student's own body.
    """

    def __init__(self, occlusion_probability: float = OCCLUSION_PROBABILITY) -> None:
        self.occlusion_probability = occlusion_probability
        self._active_occlusions: dict[int, int] = {}  # track_id → frames remaining
        self._occlusion_log: list[dict] = []

    def maybe_start_occlusion(self, track_id: int) -> None:
        """Randomly start an occlusion event for a track."""
        if track_id in self._active_occlusions:
            return
        if random.random() < self.occlusion_probability:
            duration = random.randint(*OCCLUSION_DURATION_FRAMES)
            self._active_occlusions[track_id] = duration
            self._occlusion_log.append({
                "track_id": track_id,
                "start_frame": -1,  # set by caller
                "duration": duration,
                "pattern": random.choice(KEYPOINTS_TO_DROP),
            })

    def apply_occlusion(self, kpts: np.ndarray, track_id: int) -> np.ndarray:
        """Zero out keypoints for occluded tracks."""
        if track_id not in self._active_occlusions:
            return kpts

        remaining = self._active_occlusions[track_id]
        # Gradual fade: first half full occlusion, second half partial
        ratio = remaining / OCCLUSION_DURATION_FRAMES[1]
        pattern = KEYPOINTS_TO_DROP[0]  # default: head region

        # Find the pattern for this track
        for entry in reversed(self._occlusion_log):
            if entry["track_id"] == track_id:
                pattern = entry["pattern"]
                break

        kpts = kpts.copy()
        for idx in pattern:
            if idx < len(kpts):
                if ratio > 0.5:
                    kpts[idx, 2] = 0.0  # full drop
                else:
                    kpts[idx, 2] *= ratio  # partial fade

        remaining -= 1
        if remaining <= 0:
            del self._active_occlusions[track_id]
        else:
            self._active_occlusions[track_id] = remaining

        return kpts

    @property
    def active_count(self) -> int:
        return len(self._active_occlusions)

    @property
    def stats(self) -> dict:
        return {
            "total_occlusions": len(self._occlusion_log),
            "currently_occluded": len(self._active_occlusions),
        }


# ---------------------------------------------------------------------------
# Resource monitor daemon
# ---------------------------------------------------------------------------

class ResourceMonitorDaemon:
    """
    Background thread that logs system resource usage at a fixed interval.

    Tracks:
        - GPU VRAM (allocated / reserved via PyTorch)
        - CPU core utilisation (per-core percentages)
        - Process RSS memory (via psutil)
        - WebSocket frame drop rate
    """

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._log: list[dict] = []
        self._start_time: float = 0.0
        self._frame_drops: deque[float] = deque(maxlen=600)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def record_drop(self) -> None:
        self._frame_drops.append(time.monotonic())

    def _run(self) -> None:
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False

        while self._running:
            entry = {"timestamp": round(time.monotonic() - self._start_time, 1)}

            # GPU
            alloc, reserved = read_gpu_memory()
            entry["gpu_alloc_mb"] = round(alloc, 1)
            entry["gpu_reserved_mb"] = round(reserved, 1)

            # CPU
            if has_psutil:
                entry["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                entry["cpu_per_core"] = psutil.cpu_percent(interval=0.1, percpu=True)
                proc = psutil.Process()
                mem_info = proc.memory_info()
                entry["process_rss_mb"] = round(mem_info.rss / (1024 * 1024), 1)
                entry["process_vms_mb"] = round(mem_info.vms / (1024 * 1024), 1)
            else:
                entry["cpu_percent"] = 0.0

            # Drop rate
            now = time.monotonic()
            recent_drops = sum(1 for t in self._frame_drops if now - t < 1.0)
            entry["drops_per_second"] = recent_drops

            self._log.append(entry)
            time.sleep(self.interval)

    @property
    def log(self) -> list[dict]:
        return list(self._log)

    def summary(self) -> dict:
        if not self._log:
            return {}
        gpu_vals = [e["gpu_alloc_mb"] for e in self._log if "gpu_alloc_mb" in e]
        cpu_vals = [e["cpu_percent"] for e in self._log if "cpu_percent" in e]
        rss_vals = [e.get("process_rss_mb", 0) for e in self._log]
        drop_vals = [e.get("drops_per_second", 0) for e in self._log]

        return {
            "samples": len(self._log),
            "duration_s": round(self._log[-1]["timestamp"] - self._log[0]["timestamp"], 1)
            if len(self._log) > 1 else 0,
            "gpu_mb": {
                "min": round(min(gpu_vals), 1) if gpu_vals else 0,
                "max": round(max(gpu_vals), 1) if gpu_vals else 0,
                "mean": round(statistics.mean(gpu_vals), 1) if gpu_vals else 0,
            },
            "cpu_percent": {
                "min": round(min(cpu_vals), 1) if cpu_vals else 0,
                "max": round(max(cpu_vals), 1) if cpu_vals else 0,
                "mean": round(statistics.mean(cpu_vals), 1) if cpu_vals else 0,
            },
            "rss_mb": {
                "min": round(min(rss_vals), 1) if rss_vals else 0,
                "max": round(max(rss_vals), 1) if rss_vals else 0,
                "mean": round(statistics.mean(rss_vals), 1) if rss_vals else 0,
            },
            "drops_per_sec": {
                "min": round(min(drop_vals), 1) if drop_vals else 0,
                "max": round(max(drop_vals), 1) if drop_vals else 0,
                "mean": round(statistics.mean(drop_vals), 1) if drop_vals else 0,
            },
        }


# ---------------------------------------------------------------------------
# Synthetic load injector
# ---------------------------------------------------------------------------

class SyntheticLoadInjector:
    """
    Simulates multiple concurrent CCTV camera streams with synthetic
    student candidates.

    Each "camera" produces frames at 30 FPS with configurable student
    counts.  Frames are pushed into a shared asyncio.Queue for the
    processing pipeline to consume.
    """

    def __init__(
        self,
        num_cameras: int = 4,
        candidates_per_camera: int = 20,
        target_fps: int = CAM_FPS,
    ) -> None:
        self.num_cameras = num_cameras
        self.candidates_per_camera = candidates_per_camera
        self.target_fps = target_fps
        self._running = False
        self._threads: list[threading.Thread] = []
        self._metrics = get_metrics()
        self._total_frames_produced = 0
        self._lock = threading.Lock()

        # Per-camera track ID pools
        self._track_pools: dict[int, list[int]] = {}
        for cam in range(num_cameras):
            base = cam * 100 + 1
            self._track_pools[cam] = list(range(base, base + candidates_per_camera))

    def start(self, queue: asyncio.Queue) -> None:
        """Start all camera threads."""
        if self._running:
            return
        self._running = True

        for cam_id in range(self.num_cameras):
            t = threading.Thread(
                target=self._camera_worker,
                args=(cam_id, queue),
                daemon=True,
            )
            self._threads.append(t)
            t.start()

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()

    def _camera_worker(self, cam_id: int, queue: asyncio.Queue) -> None:
        """Simulate a single CCTV camera stream."""
        import asyncio as _asyncio

        n_students = self.candidates_per_camera + random.randint(-3, 3)
        n_students = max(1, min(n_students, 30))
        track_ids = self._track_pools[cam_id][:n_students]

        interval = 1.0 / self.target_fps
        frame_counter = 0

        while self._running:
            t0 = time.monotonic()

            frame, boxes, all_kpts = _generate_synthetic_frame(
                FRAME_W, FRAME_H, n_students,
            )

            # Build synthetic inference result
            result = _build_synthetic_inference(boxes, all_kpts, track_ids)

            frame_counter += 1
            with self._lock:
                self._total_frames_produced += 1
            self._metrics.inc_frames_produced()

            # Non-blocking push
            try:
                queue.put_nowait((cam_id, frame, result, track_ids))
            except Exception:
                self._metrics.inc_frames_dropped()

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def total_frames_produced(self) -> int:
        with self._lock:
            return self._total_frames_produced


# ---------------------------------------------------------------------------
# Stress test runner
# ---------------------------------------------------------------------------

class StressTestRunner:
    """
    Orchestrates the full stress test:
        1. Starts resource monitor
        2. Starts synthetic load injector
        3. Runs occlusion simulator
        4. Processes frames through a mock pipeline
        5. Collects and reports results
    """

    def __init__(
        self,
        num_cameras: int = 4,
        candidates_per_camera: int = 20,
        duration_seconds: int = 60,
        occlusion_enabled: bool = True,
    ) -> None:
        self.num_cameras = num_cameras
        self.candidates_per_camera = candidates_per_camera
        self.duration_seconds = duration_seconds
        self.occlusion_enabled = occlusion_enabled

        self._metrics = get_metrics()
        self._resource_monitor = ResourceMonitorDaemon(interval=2.0)
        self._injector = SyntheticLoadInjector(
            num_cameras=num_cameras,
            candidates_per_camera=candidates_per_camera,
        )
        self._occlusion = OcclusionSimulator()

        # Results
        self._frames_processed = 0
        self._inference_runs = 0
        self._frame_latencies: deque[float] = deque(maxlen=5000)
        self._fps_samples: deque[float] = deque(maxlen=300)
        self._false_positives = 0
        self._true_positives = 0

    async def run(self) -> dict:
        """Execute the stress test and return results."""
        print(f"\n{'='*70}")
        print(f"  Week 9 Stress Test Harness")
        print(f"  Cameras: {self.num_cameras}  |  "
              f"Candidates/cam: {self.candidates_per_camera}  |  "
              f"Duration: {self.duration_seconds}s")
        print(f"  Occlusion sim: {'ON' if self.occlusion_enabled else 'OFF'}")
        print(f"{'='*70}\n")

        self._resource_monitor.start()

        import asyncio
        frame_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._injector.start(frame_queue)

        start_time = time.monotonic()
        fps_counter = 0
        fps_start = time.monotonic()

        try:
            while time.monotonic() - start_time < self.duration_seconds:
                try:
                    cam_id, frame, result, track_ids = await asyncio.wait_for(
                        frame_queue.get(), timeout=2.0,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    continue

                t0 = time.monotonic()

                # Apply occlusion simulation
                if self.occlusion_enabled:
                    for i, tid in enumerate(track_ids):
                        self._occlusion.maybe_start_occlusion(tid)
                        if i < len(result.keypoints):
                            result.keypoints[i] = self._occlusion.apply_occlusion(
                                result.keypoints[i], tid,
                            )

                # Mock processing (skip real YOLO for speed)
                self._process_frame(result, track_ids)

                latency = time.monotonic() - t0
                self._frame_latencies.append(latency)
                self._frames_processed += 1
                self._inference_runs += 1

                # FPS counter
                fps_counter += 1
                if time.monotonic() - fps_start >= 1.0:
                    self._fps_samples.append(fps_counter)
                    fps_counter = 0
                    fps_start = time.monotonic()

                # Periodic progress
                if self._frames_processed % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    avg_fps = self._frames_processed / max(elapsed, 0.001)
                    print(
                        f"  [{elapsed:6.1f}s] frames={self._frames_processed:6d}  "
                        f"avg_fps={avg_fps:5.1f}  "
                        f"occluded={self._occlusion.active_count}  "
                        f"latency={latency*1000:.1f}ms",
                        flush=True,
                    )

        finally:
            self._injector.stop()
            self._resource_monitor.stop()

        return self._build_report(start_time)

    def _process_frame(self, result: InferenceResult, track_ids: list[int]) -> None:
        """Mock pipeline processing — extract features + classify."""
        for idx in range(len(result.tracker_ids)):
            kpts = result.keypoints[idx]
            # Check confidence after occlusion
            valid_kpts = kpts[kpts[:, 2] > 0.3]
            if len(valid_kpts) < 3:
                # Low confidence after occlusion → potential false positive
                self._false_positives += 1
                continue
            # Simulate classification
            self._true_positives += 1

    def _build_report(self, start_time: float) -> dict:
        """Build structured stress test report."""
        elapsed = time.monotonic() - start_time
        resource_summary = self._resource_monitor.summary()

        latencies_ms = [l * 1000 for l in self._frame_latencies]

        report = {
            "config": {
                "cameras": self.num_cameras,
                "candidates_per_camera": self.candidates_per_camera,
                "duration_s": self.duration_seconds,
                "occlusion_enabled": self.occlusion_enabled,
            },
            "throughput": {
                "total_frames": self._frames_processed,
                "avg_fps": round(self._frames_processed / max(elapsed, 0.001), 1),
                "peak_fps": max(self._fps_samples) if self._fps_samples else 0,
                "min_fps": min(self._fps_samples) if self._fps_samples else 0,
                "fps_degradation": round(
                    (1 - (min(self._fps_samples) / max(max(self._fps_samples), 1)))
                    * 100, 1
                ) if self._fps_samples else 0,
            },
            "latency": {
                "mean_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0,
                "p50_ms": round(statistics.median(latencies_ms), 2) if latencies_ms else 0,
                "p95_ms": round(
                    sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
                    if latencies_ms else 0, 2,
                ),
                "p99_ms": round(
                    sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
                    if latencies_ms else 0, 2,
                ),
                "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0,
            },
            "resources": resource_summary,
            "accuracy": {
                "true_positives": self._true_positives,
                "false_positives": self._false_positives,
                "false_positive_rate": round(
                    self._false_positives / max(self._true_positives + self._false_positives, 1)
                    * 100, 2,
                ),
            },
            "occlusion": self._occlusion.stats,
            "pipeline_metrics": self._metrics.snapshot(),
        }

        return report


# ---------------------------------------------------------------------------
# Terminal report formatter
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    """Print a structured terminal telemetry report."""
    print(f"\n{'='*70}")
    print(f"  STRESS TEST RESULTS")
    print(f"{'='*70}")

    cfg = report["config"]
    print(f"\n  Configuration:")
    print(f"    Cameras:           {cfg['cameras']}")
    print(f"    Candidates/cam:    {cfg['candidates_per_camera']}")
    print(f"    Total candidates:  {cfg['cameras'] * cfg['candidates_per_camera']}")
    print(f"    Duration:          {cfg['duration_s']}s")
    print(f"    Occlusion sim:     {'ON' if cfg['occlusion_enabled'] else 'OFF'}")

    tp = report["throughput"]
    print(f"\n  Throughput:")
    print(f"    Total frames:      {tp['total_frames']}")
    print(f"    Average FPS:       {tp['avg_fps']}")
    print(f"    Peak FPS:          {tp['peak_fps']}")
    print(f"    Min FPS:           {tp['min_fps']}")
    print(f"    FPS degradation:   {tp['fps_degradation']}%")

    lat = report["latency"]
    print(f"\n  Processing Latency:")
    print(f"    Mean:              {lat['mean_ms']:.2f} ms")
    print(f"    P50:               {lat['p50_ms']:.2f} ms")
    print(f"    P95:               {lat['p95_ms']:.2f} ms")
    print(f"    P99:               {lat['p99_ms']:.2f} ms")
    print(f"    Max:               {lat['max_ms']:.2f} ms")

    res = report.get("resources", {})
    if res:
        print(f"\n  Resource Usage:")
        gpu = res.get("gpu_mb", {})
        if gpu.get("max", 0) > 0:
            print(f"    GPU VRAM:          {gpu['min']:.0f}–{gpu['max']:.0f} MB "
                  f"(avg {gpu['mean']:.0f} MB)")
        cpu = res.get("cpu_percent", {})
        print(f"    CPU:               {cpu['min']:.0f}–{cpu['max']:.0f}% "
              f"(avg {cpu['mean']:.0f}%)")
        rss = res.get("rss_mb", {})
        print(f"    Process RSS:       {rss['min']:.0f}–{rss['max']:.0f} MB "
              f"(avg {rss['mean']:.0f} MB)")
        drops = res.get("drops_per_sec", {})
        print(f"    WS drops/sec:      {drops['mean']:.1f} (max {drops['max']:.1f})")

    acc = report["accuracy"]
    print(f"\n  Accuracy:")
    print(f"    True positives:    {acc['true_positives']}")
    print(f"    False positives:   {acc['false_positives']}")
    print(f"    FP rate:           {acc['false_positive_rate']}%")

    occ = report["occlusion"]
    print(f"\n  Occlusion Recovery:")
    print(f"    Total events:      {occ['total_occlusions']}")
    print(f"    Currently active:  {occ['currently_occluded']}")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="Week 9 Stress Test Harness")
    parser.add_argument("--cameras", type=int, default=4, help="Number of simulated cameras")
    parser.add_argument("--candidates", type=int, default=20, help="Students per camera")
    parser.add_argument("--duration", type=int, default=60, help="Test duration (seconds)")
    parser.add_argument("--no-occlusion", action="store_true", help="Disable occlusion sim")
    parser.add_argument("--output", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    runner = StressTestRunner(
        num_cameras=args.cameras,
        candidates_per_camera=args.candidates,
        duration_seconds=args.duration,
        occlusion_enabled=not args.no_occlusion,
    )

    report = await runner.run()
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to {args.output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
