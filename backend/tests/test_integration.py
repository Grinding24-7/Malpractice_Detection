"""
test_integration.py — Week 9: Comprehensive pytest Suite.

Tests cover:
    1. ErrorBoundary isolation and recovery
    2. PersistenceFilter correctness
    3. ThresholdCalibrator grid search
    4. SyntheticLoadInjector frame generation
    5. OcclusionSimulator keypoint dropping
    6. ExamSurveillanceEngine status and lifecycle
    7. ResourceMonitorDaemon logging
    8. PipelineMetrics counters

Run:
    cd backend && python -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system_integrator import (
    ErrorBoundary,
    EngineState,
    ExamSurveillanceEngine,
    TrackState,
    ANOMALY_PERSISTENCE_THRESHOLD,
)
from calibration import (
    ClassificationThresholds,
    PersistenceFilter,
    ThresholdCalibrator,
    CalibrationRunner,
    _generate_synthetic_features,
)
from stress_test_harness import (
    OcclusionSimulator,
    ResourceMonitorDaemon,
    SyntheticLoadInjector,
    _generate_synthetic_frame,
    _generate_synthetic_keypoints,
    _build_synthetic_inference,
)
from metrics import PipelineMetrics, get_metrics
from detector import InferenceResult


# ===================================================================
# 1. ErrorBoundary
# ===================================================================

class TestErrorBoundary:
    def test_success(self):
        b = ErrorBoundary("test")
        result = b(lambda: 42)
        assert result == 42
        assert b.error_count == 0
        assert b.healthy

    def test_exception_captured(self):
        b = ErrorBoundary("test")
        result = b(lambda: 1 / 0)
        assert result is None
        assert b.error_count == 1
        assert not b.healthy
        assert "ZeroDivisionError" in b.last_error

    def test_multiple_errors(self):
        b = ErrorBoundary("test")
        for _ in range(5):
            b(lambda: 1 / 0)
        assert b.error_count == 5

    def test_recovery_after_timeout(self):
        b = ErrorBoundary("test")
        b(lambda: 1 / 0)
        assert not b.healthy
        # Simulate time passage
        b.last_error_time = time.monotonic() - 31
        assert b.healthy

    def test_to_dict(self):
        b = ErrorBoundary("test")
        b(lambda: 1 / 0)
        d = b.to_dict()
        assert d["name"] == "test"
        assert d["error_count"] == 1
        assert "last_error" in d


# ===================================================================
# 2. PersistenceFilter
# ===================================================================

class TestPersistenceFilter:
    def test_no_alert_below_threshold(self):
        f = PersistenceFilter(min_frames=5, window_size=10)
        for _ in range(4):
            assert f.update(1, True) is False
        assert not f.should_alert(1)

    def test_alert_at_threshold(self):
        f = PersistenceFilter(min_frames=5, window_size=10)
        for _ in range(5):
            f.update(1, True)
        assert f.should_alert(1)

    def test_window_sliding(self):
        f = PersistenceFilter(min_frames=5, window_size=10)
        # Fill with anomalies then normals
        for _ in range(5):
            f.update(1, True)
        for _ in range(10):
            f.update(1, False)
        # Old anomalies should have slid out
        assert not f.should_alert(1)

    def test_mixed_signals(self):
        f = PersistenceFilter(min_frames=3, window_size=5)
        f.update(1, True)
        f.update(1, False)
        f.update(1, True)
        f.update(1, True)
        assert f.should_alert(1)

    def test_reset(self):
        f = PersistenceFilter(min_frames=3, window_size=10)
        for _ in range(5):
            f.update(1, True)
        f.reset(1)
        assert not f.should_alert(1)

    def test_prune(self):
        f = PersistenceFilter(min_frames=3, window_size=10)
        f.update(1, True)
        f.update(2, True)
        f.update(3, True)
        pruned = f.prune({1, 3})
        assert pruned == 1
        assert 2 not in f._buffers

    def test_stats(self):
        f = PersistenceFilter(min_frames=3, window_size=10)
        f.update(1, True)
        f.update(2, False)
        s = f.stats
        assert s["tracked_tracks"] == 2
        assert s["total_observations"] == 2


# ===================================================================
# 3. ThresholdCalibrator
# ===================================================================

class TestThresholdCalibrator:
    def test_generate_synthetic_features(self):
        features, labels = _generate_synthetic_features(n_samples=200)
        assert features.shape == (200, 7)
        assert labels.shape == (200,)
        assert set(np.unique(labels)) == {0, 1, 2, 3}

    def test_grid_search_finds_solution(self):
        features, labels = _generate_synthetic_features(n_samples=400)
        cal = ThresholdCalibrator(
            tau_head_low_range=np.array([0.65, 0.70, 0.75]),
            tau_head_high_range=np.array([1.30, 1.40]),
            tau_peek_range=np.array([0.85, 0.90]),
            persistence_range=[6, 10],
        )
        best = cal.grid_search(features, labels, verbose=False)
        assert "f1" in best
        assert best["f1"] > 0.3  # should be better than random
        assert cal.best is not None
        assert len(cal.top_n(5)) == 5

    def test_classify_batch(self):
        features, labels = _generate_synthetic_features(n_samples=100)
        thresholds = ClassificationThresholds()
        cal = ThresholdCalibrator()
        preds = cal._classify_batch(features, thresholds)
        assert preds.shape == (100,)
        assert all(p in [0, 1, 2, 3] for p in preds)

    def test_thresholds_to_dict(self):
        t = ClassificationThresholds()
        d = t.to_dict()
        assert "tau_head_low" in d
        assert "persistence_frames" in d
        assert isinstance(d["persistence_frames"], int)


# ===================================================================
# 4. SyntheticLoadInjector
# ===================================================================

class TestSyntheticFrame:
    def test_generate_frame(self):
        frame, boxes, kpts = _generate_synthetic_frame(1280, 720, 5)
        assert frame.shape == (720, 1280, 3)
        assert len(boxes) == 5
        assert len(kpts) == 5

    def test_generate_keypoints_shape(self):
        bbox = (100, 100, 300, 400)
        kpts = _generate_synthetic_keypoints(bbox)
        assert kpts.shape == (17, 3)
        assert all(kpts[i, 2] > 0 for i in range(17))

    def test_build_inference_result(self):
        frame, boxes, kpts_list = _generate_synthetic_frame(640, 480, 3)
        track_ids = [1, 2, 3]
        result = _build_synthetic_inference(boxes, kpts_list, track_ids)
        assert isinstance(result, InferenceResult)
        assert len(result.tracker_ids) == 3
        assert result.keypoints.shape == (3, 17, 3)


# ===================================================================
# 5. OcclusionSimulator
# ===================================================================

class TestOcclusionSimulator:
    def test_no_occlusion_by_default(self):
        sim = OcclusionSimulator(occlusion_probability=0.0)
        kpts = np.ones((17, 3), dtype=np.float32)
        result = sim.apply_occlusion(kpts, 1)
        np.testing.assert_array_equal(result, kpts)

    def test_active_occlusion(self):
        sim = OcclusionSimulator(occlusion_probability=1.0)
        sim.maybe_start_occlusion(1)
        assert sim.active_count == 1

    def test_occlusion_drops_keypoints(self):
        sim = OcclusionSimulator(occlusion_probability=0.0)
        # Manually start occlusion
        sim._active_occlusions[1] = 50  # long enough for full drop
        kpts = np.ones((17, 3), dtype=np.float32)
        result = sim.apply_occlusion(kpts, 1)
        # Head region should be zeroed
        assert result[0, 2] == 0.0  # nose

    def test_occlusion_fades(self):
        sim = OcclusionSimulator(occlusion_probability=0.0)
        sim._active_occlusions[1] = 10  # short → partial
        kpts = np.ones((17, 3), dtype=np.float32)
        result = sim.apply_occlusion(kpts, 1)
        # Should be partial (not zero)
        assert 0 < result[0, 2] < 1.0

    def test_stats(self):
        sim = OcclusionSimulator(occlusion_probability=1.0)
        sim.maybe_start_occlusion(1)
        sim.maybe_start_occlusion(2)
        s = sim.stats
        assert s["total_occlusions"] == 2
        assert s["currently_occluded"] == 2


# ===================================================================
# 6. ResourceMonitorDaemon
# ===================================================================

class TestResourceMonitorDaemon:
    def test_start_stop(self):
        monitor = ResourceMonitorDaemon(interval=0.5)
        monitor.start()
        time.sleep(1.5)
        monitor.stop()
        assert len(monitor.log) >= 1

    def test_summary(self):
        monitor = ResourceMonitorDaemon(interval=0.5)
        monitor.start()
        time.sleep(1.5)
        monitor.stop()
        s = monitor.summary()
        assert "gpu_mb" in s
        assert "cpu_percent" in s
        assert "samples" in s
        assert s["samples"] >= 1


# ===================================================================
# 7. PipelineMetrics
# ===================================================================

class TestPipelineMetrics:
    def test_counters(self):
        m = PipelineMetrics()
        m.inc_frames_produced(10)
        m.inc_frames_consumed(8)
        m.inc_frames_dropped(2)
        snap = m.snapshot()
        assert snap["frames_produced"] == 10
        assert snap["frames_consumed"] == 8
        assert snap["frames_dropped"] == 2

    def test_gauges(self):
        m = PipelineMetrics()
        m.set_active_connections(5)
        m.set_queue_depth(3)
        m.set_jpeg_quality(72)
        m.set_frame_skip_k(4)
        snap = m.snapshot()
        assert snap["active_connections"] == 5
        assert snap["queue_depth"] == 3
        assert snap["jpeg_quality"] == 72
        assert snap["frame_skip_k"] == 4

    def test_prometheus_render(self):
        m = PipelineMetrics()
        m.inc_frames_produced(100)
        text = m.render_prometheus()
        assert "pipeline_frames_produced_total" in text
        assert "pipeline_frames_consumed_total" in text
        assert "pipeline_inference_latency_seconds" in text

    def test_gpu_memory(self):
        m = PipelineMetrics()
        m.set_gpu_memory(1024.5, 2048.0)
        snap = m.snapshot()
        assert snap["gpu_memory_mb"] == 1024.5
        assert snap["gpu_memory_reserved_mb"] == 2048.0


# ===================================================================
# 8. ExamSurveillanceEngine
# ===================================================================

class TestExamSurveillanceEngine:
    def test_initial_status(self):
        engine = ExamSurveillanceEngine(
            enable_evidence=False,
            enable_purge=False,
        )
        status = engine.status()
        assert status["state"] == EngineState.IDLE.value
        assert status["active_tracks"] == 0
        assert status["frame_counter"] == 0

    def test_track_state(self):
        t = TrackState(track_id=1)
        assert t.anomaly_frames == 0
        assert t.total_frames == 0
        assert t.last_prediction == "NORMAL"
        assert t.persistence_ratio == 0.0
        assert t.age_seconds >= 0

    def test_track_state_anomaly(self):
        t = TrackState(track_id=1)
        t.anomaly_frames = 15
        t.total_frames = 30
        assert t.persistence_ratio == 0.5

    def test_boundaries_initialised(self):
        engine = ExamSurveillanceEngine(
            enable_evidence=False,
            enable_purge=False,
        )
        assert "ingestion" in engine._boundaries
        assert "tracking" in engine._boundaries
        assert "evidence" in engine._boundaries
        assert all(b.healthy for b in engine._boundaries.values())

    def test_classify_normal(self):
        engine = ExamSurveillanceEngine(
            enable_evidence=False,
            enable_purge=False,
        )
        # Create a mock frame and keypoints
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Set normal features: ear_ratio ~1.0, norm_vert ~0.5
        kpts[3] = [100, 200, 0.9]  # left_ear
        kpts[4] = [200, 200, 0.9]  # right_ear
        kpts[0] = [150, 180, 0.95]  # nose
        kpts[5] = [80, 300, 0.9]   # left_shoulder
        kpts[6] = [220, 300, 0.9]  # right_shoulder

        features = np.array([
            1.0,   # ear_ratio (normal)
            0.5,   # norm_vertical_drop (normal)
            0.1,   # shoulder_angle
            0.3,   # norm_nose_ear_drop
            0.95,  # nose_conf
            0.9,   # l_ear_conf
            0.9,   # r_ear_conf
        ], dtype=np.float32)

        pred = engine._classify_track(1, features, frame, kpts)
        assert pred == "NORMAL"

    def test_engine_singleton(self):
        from system_integrator import _engine, get_exam_engine
        # Reset singleton
        import system_integrator as si
        si._engine = None
        e1 = get_exam_engine()
        e2 = get_exam_engine()
        assert e1 is e2
        si._engine = None  # cleanup


# ===================================================================
# 9. Integration: full synthetic pipeline
# ===================================================================

class TestFullSyntheticPipeline:
    def test_injector_produces_frames(self):
        """Test that SyntheticLoadInjector produces frames at expected rate."""
        import asyncio
        from stress_test_harness import SyntheticLoadInjector

        injector = SyntheticLoadInjector(
            num_cameras=1,
            candidates_per_camera=5,
            target_fps=30,
        )
        queue = asyncio.Queue(maxsize=100)

        injector.start(queue)
        time.sleep(0.5)
        injector.stop()

        assert queue.qsize() > 0
        cam_id, frame, result, track_ids = queue.get_nowait()
        assert frame.shape == (720, 1280, 3)
        assert isinstance(result, InferenceResult)
        assert len(track_ids) == 5

    def test_calibration_end_to_end(self):
        """Test full calibration pipeline with synthetic data."""
        runner = CalibrationRunner()
        runner.load_data()
        best = runner.run_grid_search(verbose=False)
        assert "f1" in best
        persistence = runner.evaluate_with_persistence(verbose=False)
        assert "fp_reduction_pct" in persistence
        assert persistence["fp_reduction_pct"] >= 0
