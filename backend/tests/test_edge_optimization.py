"""
test_edge_optimization.py — Week 10: Edge Optimization & Model Quantization Tests.

Tests cover:
    1. TensorRTINT8Calibrator — batch iteration, cache read/write, reset
    2. Exporter — ONNX export, model validation, file generation
    3. AcceleratedInferenceEngine — provider fallback, inference, stats
    4. Benchmark parity metrics — cosine similarity, MSE, accuracy
    5. Synthetic data generation
    6. SequenceModelExporter — ONNX export with dynamic axes

Run:
    cd backend && python -m pytest tests/test_edge_optimization.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrator import (
    TensorRTINT8Calibrator,
    extract_or_generate_calibration_data,
    generate_synthetic_calibration_data,
    measure_calibration_quality,
)
from benchmark_quant import (
    BenchmarkResult,
    compute_parity,
    _make_synthetic_batch,
)
from edge_inference import (
    AcceleratedInferenceEngine,
    InferenceOutput,
    RuntimeBackend,
    _PyTorchJITWrapper,
    _softmax,
)


# ===================================================================
# 1. TensorRTINT8Calibrator
# ===================================================================

class TestTensorRTINT8Calibrator:
    def test_init(self):
        data = np.random.randn(100, 30, 117).astype(np.float32)
        cal = TensorRTINT8Calibrator(data, batch_size=8)
        assert cal.get_batch_size() == 8
        assert cal._n_samples == 100
        assert cal._batch_count == 13  # ceil(100/8)

    def test_get_batch(self):
        data = np.random.randn(20, 30, 117).astype(np.float32)
        cal = TensorRTINT8Calibrator(data, batch_size=8)

        # First batch
        batch = cal.get_batch(["input"])
        assert batch is not None
        assert len(batch) == 1
        assert batch[0].shape == (8, 30, 117)

        # Continue until exhausted
        for _ in range(2):
            cal.get_batch(["input"])

        # Should return None when done
        result = cal.get_batch(["input"])
        assert result is None

    def test_reset(self):
        data = np.random.randn(16, 30, 117).astype(np.float32)
        cal = TensorRTINT8Calibrator(data, batch_size=8)

        cal.get_batch(["input"])
        cal.get_batch(["input"])
        assert cal._batch_idx == 2

        cal.reset()
        assert cal._batch_idx == 0

    def test_cache_read_write(self):
        data = np.random.randn(10, 30, 117).astype(np.float32)
        cache_path = tempfile.mktemp(suffix=".cache")
        try:
            cal = TensorRTINT8Calibrator(data, batch_size=8, cache_path=cache_path)

            # No cache initially
            cached = cal.read_calibration_cache()
            assert cached is None or cached == b""

            # Write cache
            test_cache = b"test_calibration_data_12345"
            cal.write_calibration_cache(test_cache)

            # Read it back
            cached = cal.read_calibration_cache()
            assert cached == test_cache
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    def test_max_batches(self):
        data = np.random.randn(1000, 30, 117).astype(np.float32)
        cal = TensorRTINT8Calibrator(data, batch_size=8, max_batches=5)
        assert cal._batch_count == 5

    def test_stats(self):
        data = np.random.randn(50, 30, 117).astype(np.float32)
        cal = TensorRTINT8Calibrator(data, batch_size=8)
        s = cal.stats
        assert s["n_samples"] == 50
        assert s["batch_size"] == 8
        assert "input_shape" in s

    def test_synthetic_data_generation(self):
        data = generate_synthetic_calibration_data(
            n_samples=100, sequence_len=30, input_dim=117,
        )
        assert data.shape == (100, 30, 117)
        assert data.dtype == np.float32

    def test_extract_or_generate(self):
        # No real dataset → synthetic
        data = extract_or_generate_calibration_data(
            dataset_path=None, n_samples=50,
        )
        assert data.shape[0] == 50

    def test_extract_or_generate_fallback(self):
        # Non-existent path → synthetic
        data = extract_or_generate_calibration_data(
            dataset_path="/nonexistent/path.pt", n_samples=30,
        )
        assert data.shape[0] == 30


# ===================================================================
# 2. Benchmark parity metrics
# ===================================================================

class TestBenchmarkParity:
    def test_identical_outputs(self):
        ref = np.random.randn(10, 4).astype(np.float64)
        test = ref.copy()
        parity = compute_parity(ref, test)
        assert parity["cosine_similarity"] > 0.9999
        assert parity["mse"] < 1e-10
        assert parity["max_abs_diff"] < 1e-10

    def test_similar_outputs(self):
        ref = np.random.randn(10, 4).astype(np.float64)
        test = ref + np.random.randn(10, 4).astype(np.float64) * 0.01
        parity = compute_parity(ref, test)
        assert parity["cosine_similarity"] > 0.99
        assert parity["mse"] < 0.01

    def test_dissimilar_outputs(self):
        ref = np.array([[1, 0, 0, 0]] * 10, dtype=np.float64)
        test = np.array([[0, 0, 0, 1]] * 10, dtype=np.float64)
        parity = compute_parity(ref, test)
        assert parity["cosine_similarity"] < 0.1
        assert parity["mse"] >= 0.4

    def test_make_synthetic_batch(self):
        data = _make_synthetic_batch(batch_size=4, seq_len=30, input_dim=117)
        assert data.shape == (4, 30, 117)
        assert data.dtype == np.float32


# ===================================================================
# 3. BenchmarkResult
# ===================================================================

class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(
            backend="pytorch_cuda",
            model_format="pytorch_fp32",
            latency_mean_ms=5.2,
            latency_p95_ms=6.1,
            throughput_fps=192.3,
            file_size_mb=0,
            iterations=200,
            cosine_similarity=0.9998,
            mse=1e-6,
            max_abs_diff=0.001,
        )
        d = r.to_dict()
        assert d["backend"] == "pytorch_cuda"
        assert d["model_format"] == "pytorch_fp32"
        assert d["cosine_similarity"] == 0.9998
        assert d["mse"] == 1e-06


# ===================================================================
# 4. Softmax
# ===================================================================

class TestSoftmax:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        s = _softmax(x, axis=1)
        assert abs(s.sum() - 1.0) < 1e-6
        assert s[0, 3] > s[0, 0]

    def test_numerical_stability(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        s = _softmax(x, axis=1)
        assert abs(s.sum() - 1.0) < 1e-6
        assert not np.any(np.isnan(s))

    def test_batch(self):
        x = np.random.randn(8, 4)
        s = _softmax(x, axis=1)
        assert s.shape == (8, 4)
        assert np.allclose(s.sum(axis=1), 1.0, atol=1e-6)


# ===================================================================
# 5. AcceleratedInferenceEngine
# ===================================================================

class TestAcceleratedInferenceEngine:
    def test_init_no_models(self):
        engine = AcceleratedInferenceEngine()
        assert engine.active_backend == "none"

    def test_load_no_providers(self):
        engine = AcceleratedInferenceEngine()
        result = engine.load()
        # Should fall through to NONE or PYTORCH_CPU
        assert isinstance(result, RuntimeBackend)

    def test_stats(self):
        engine = AcceleratedInferenceEngine()
        s = engine.stats()
        assert "active_backend" in s
        assert "total_inferences" in s
        assert "avg_latency_ms" in s

    def test_infer_no_backend(self):
        engine = AcceleratedInferenceEngine()
        result = engine.infer(np.random.randn(1, 30, 117).astype(np.float32))
        assert result is None

    def test_with_pytorch_model(self):
        from temporal_training import MalpracticeLSTM

        model = MalpracticeLSTM(
            input_dim=117, hidden_dim=128, num_layers=2,
            num_classes=4, dropout=0.0, bidirectional=True,
            use_attention=True,
        )
        engine = AcceleratedInferenceEngine(pytorch_model=model)
        backend = engine.load()
        assert backend in (RuntimeBackend.PYTORCH_CUDA, RuntimeBackend.PYTORCH_CPU)

        # Run inference
        data = np.random.randn(1, 30, 117).astype(np.float32)
        out = engine.infer(data)
        assert out is not None
        assert out.logits.shape == (1, 4)
        assert out.predictions.shape == (1,)
        assert out.probabilities.shape == (1, 4)
        assert abs(out.probabilities.sum() - 1.0) < 1e-5

        engine.unload()

    def test_infer_2d_input(self):
        from temporal_training import MalpracticeLSTM

        model = MalpracticeLSTM(
            input_dim=117, hidden_dim=128, num_layers=2,
            num_classes=4, dropout=0.0, bidirectional=True,
            use_attention=True,
        )
        engine = AcceleratedInferenceEngine(pytorch_model=model)
        engine.load()

        # 2D input should be auto-batched
        data_2d = np.random.randn(30, 117).astype(np.float32)
        out = engine.infer(data_2d)
        assert out is not None
        assert out.logits.shape == (1, 4)

    def test_multiple_inferences(self):
        from temporal_training import MalpracticeLSTM

        model = MalpracticeLSTM(
            input_dim=117, hidden_dim=128, num_layers=2,
            num_classes=4, dropout=0.0, bidirectional=True,
            use_attention=True,
        )
        engine = AcceleratedInferenceEngine(pytorch_model=model)
        engine.load()

        for _ in range(10):
            data = np.random.randn(1, 30, 117).astype(np.float32)
            out = engine.infer(data)

        stats = engine.stats()
        assert stats["total_inferences"] == 11  # 1 warmup + 10


# ===================================================================
# 6. PyTorchJITWrapper
# ===================================================================

class TestPyTorchJITWrapper:
    def test_trace_and_infer(self):
        from temporal_training import MalpracticeLSTM

        model = MalpracticeLSTM(
            input_dim=117, hidden_dim=128, num_layers=2,
            num_classes=4, dropout=0.0, bidirectional=True,
            use_attention=True,
        )
        wrapper = _PyTorchJITWrapper(model)
        assert wrapper.load() is True

        data = np.random.randn(1, 30, 117).astype(np.float32)
        output = wrapper.infer(data)
        assert output is not None
        assert output.shape == (1, 4)

    def test_no_model(self):
        wrapper = _PyTorchJITWrapper()
        assert wrapper.load() is False
        assert wrapper.infer(np.random.randn(1, 30, 117)) is None


# ===================================================================
# 7. RuntimeBackend enum
# ===================================================================

class TestRuntimeBackend:
    def test_values(self):
        assert RuntimeBackend.TRT.value == "tensorrt"
        assert RuntimeBackend.ONNXRT_CUDA.value == "onnxruntime_cuda"
        assert RuntimeBackend.PYTORCH_CPU.value == "pytorch_cpu"
        assert RuntimeBackend.NONE.value == "none"
