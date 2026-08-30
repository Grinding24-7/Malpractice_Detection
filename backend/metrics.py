"""
metrics.py — Prometheus metrics collector for Week 8 streaming pipeline.

Exposes counters, gauges, and histograms for:
    - Total FPS (producer + consumer)
    - GPU memory usage (allocated / reserved)
    - Frame drop rate (backpressure discards)
    - Active WebSocket connections
    - Inference latency per frame
    - Queue depth (producer → consumer buffer)

Usage:
    from metrics import PipelineMetrics
    m = PipelineMetrics()
    m.inc_frames_produced()
    m.set_gpu_memory_mb(1024.0)
    ...
    # GET /metrics returns Prometheus text format
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


# ---------------------------------------------------------------------------
# Prometheus text-format exposition (zero external deps)
# ---------------------------------------------------------------------------

@dataclass
class _Counter:
    name: str
    help: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        label_str = ""
        if self.labels:
            pairs = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
            label_str = "{" + pairs + "}"
        lines.append(f"{self.name}{label_str} {self.value}")
        return "\n".join(lines)


@dataclass
class _Gauge:
    name: str
    help: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        label_str = ""
        if self.labels:
            pairs = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
            label_str = "{" + pairs + "}"
        lines.append(f"{self.name}{label_str} {self.value}")
        return "\n".join(lines)


@dataclass
class _Histogram:
    name: str
    help: str
    buckets: list[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ])
    _counts: list[float] = field(default_factory=lambda: [0.0] * 11)
    _sum: float = 0.0
    _count: float = 0.0

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        for i, le in enumerate(self.buckets):
            if value <= le:
                self._counts[i] += 1

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for i, le in enumerate(self.buckets):
            lines.append(f'{self.name}_bucket{{le="{le}"}} {self._counts[i]}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline metrics
# ---------------------------------------------------------------------------

class PipelineMetrics:
    """Thread-safe metrics collector for the streaming pipeline."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._started = time.monotonic()

        # Counters
        self._frames_produced = _Counter(
            "pipeline_frames_produced_total",
            "Total frames read from video source by the producer.",
        )
        self._frames_consumed = _Counter(
            "pipeline_frames_consumed_total",
            "Total frames delivered to WebSocket / MJPEG consumers.",
        )
        self._frames_dropped = _Counter(
            "pipeline_frames_dropped_total",
            "Total frames discarded due to backpressure.",
        )
        self._inference_runs = _Counter(
            "pipeline_inference_runs_total",
            "Total YOLO11-pose inference executions.",
        )

        # Gauges
        self._active_connections = _Gauge(
            "pipeline_ws_active_connections",
            "Number of active WebSocket connections.",
        )
        self._queue_depth = _Gauge(
            "pipeline_queue_depth",
            "Current depth of the producer→consumer frame queue.",
        )
        self._gpu_memory_mb = _Gauge(
            "pipeline_gpu_memory_mb",
            "GPU memory allocated (MiB) — 0 when CPU-only.",
        )
        self._gpu_memory_reserved_mb = _Gauge(
            "pipeline_gpu_memory_reserved_mb",
            "GPU memory reserved (MiB) — 0 when CPU-only.",
        )
        self._producer_fps = _Gauge(
            "pipeline_producer_fps",
            "Current producer frame rate (rolling 1s window).",
        )
        self._consumer_fps = _Gauge(
            "pipeline_consumer_fps",
            "Current consumer frame rate (rolling 1s window).",
        )
        self._jpeg_quality = _Gauge(
            "pipeline_jpeg_quality",
            "Current adaptive JPEG encoding quality (0-100).",
        )
        self._frame_skip_k = _Gauge(
            "pipeline_frame_skip_k",
            "Current adaptive frame-skip interval (full inference every k frames).",
        )

        # Histogram
        self._inference_latency = _Histogram(
            "pipeline_inference_latency_seconds",
            "YOLO11-pose inference latency per frame.",
        )

        # Rolling FPS trackers
        self._prod_frame_times: list[float] = []
        self._cons_frame_times: list[float] = []

    # -- Mutators (called from pipeline workers) ----------------------------

    def inc_frames_produced(self, n: int = 1) -> None:
        with self._lock:
            self._frames_produced.value += n
            self._prod_frame_times.append(time.monotonic())

    def inc_frames_consumed(self, n: int = 1) -> None:
        with self._lock:
            self._frames_consumed.value += n
            self._cons_frame_times.append(time.monotonic())

    def inc_frames_dropped(self, n: int = 1) -> None:
        with self._lock:
            self._frames_dropped.value += n

    def inc_inference_runs(self, n: int = 1) -> None:
        with self._lock:
            self._inference_runs.value += n

    def observe_inference_latency(self, seconds: float) -> None:
        with self._lock:
            self._inference_latency.observe(seconds)

    def set_active_connections(self, n: int) -> None:
        with self._lock:
            self._active_connections.value = float(n)

    def inc_active_connections(self, delta: int = 1) -> None:
        with self._lock:
            self._active_connections.value += delta

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth.value = float(depth)

    def set_gpu_memory(self, allocated_mb: float, reserved_mb: float) -> None:
        with self._lock:
            self._gpu_memory_mb.value = allocated_mb
            self._gpu_memory_reserved_mb.value = reserved_mb

    def set_jpeg_quality(self, quality: int) -> None:
        with self._lock:
            self._jpeg_quality.value = float(quality)

    def set_frame_skip_k(self, k: int) -> None:
        with self._lock:
            self._frame_skip_k.value = float(k)

    # -- Query --------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a dict snapshot for JSON APIs."""
        now = time.monotonic()
        with self._lock:
            # Rolling FPS: count frames in last 1 second
            cutoff = now - 1.0
            prod_fps = sum(1 for t in self._prod_frame_times if t > cutoff)
            cons_fps = sum(1 for t in self._cons_frame_times if t > cutoff)
            # Trim old entries
            self._prod_frame_times = [t for t in self._prod_frame_times if t > cutoff]
            self._cons_frame_times = [t for t in self._cons_frame_times if t > cutoff]

            return {
                "uptime_s": round(now - self._started, 1),
                "frames_produced": int(self._frames_produced.value),
                "frames_consumed": int(self._frames_consumed.value),
                "frames_dropped": int(self._frames_dropped.value),
                "inference_runs": int(self._inference_runs.value),
                "producer_fps": prod_fps,
                "consumer_fps": cons_fps,
                "active_connections": int(self._active_connections.value),
                "queue_depth": int(self._queue_depth.value),
                "gpu_memory_mb": round(self._gpu_memory_mb.value, 1),
                "gpu_memory_reserved_mb": round(self._gpu_memory_reserved_mb.value, 1),
                "jpeg_quality": int(self._jpeg_quality.value),
                "frame_skip_k": int(self._frame_skip_k.value),
            }

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - 1.0
            prod_fps = float(sum(1 for t in self._prod_frame_times if t > cutoff))
            cons_fps = float(sum(1 for t in self._cons_frame_times if t > cutoff))
            self._prod_frame_times = [t for t in self._prod_frame_times if t > cutoff]
            self._cons_frame_times = [t for t in self._cons_frame_times if t > cutoff]
            self._producer_fps.value = prod_fps
            self._consumer_fps.value = cons_fps

        parts = [
            self._frames_produced.render(),
            self._frames_consumed.render(),
            self._frames_dropped.render(),
            self._inference_runs.render(),
            self._active_connections.render(),
            self._queue_depth.render(),
            self._gpu_memory_mb.render(),
            self._gpu_memory_reserved_mb.render(),
            self._producer_fps.render(),
            self._consumer_fps.render(),
            self._jpeg_quality.render(),
            self._frame_skip_k.render(),
            self._inference_latency.render(),
        ]
        return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def read_gpu_memory() -> tuple[float, float]:
    """Return (allocated_mb, reserved_mb) from PyTorch, or (0, 0) if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return alloc, reserved
    except Exception:
        pass
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_metrics: Optional[PipelineMetrics] = None


def get_metrics() -> PipelineMetrics:
    global _metrics
    if _metrics is None:
        _metrics = PipelineMetrics()
    return _metrics
