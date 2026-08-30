"""
edge_inference.py — Week 10: Accelerated Unified Runtime Wrapper.

Provides ``AcceleratedInferenceEngine`` with runtime provider fallback:
    1. Primary:   NVIDIA TensorRT Engine (pycuda / cuda-python)
    2. Secondary: ONNX Runtime (TensorrtExecutionProvider / CUDAExecutionProvider)
    3. Fallback:  Standard PyTorch (torch.jit.trace / CUDA / CPU)

The engine transparently loads the best available model format and
dispatches inference through the fastest available provider.

Usage:
    engine = AcceleratedInferenceEngine(
        onnx_path="exported_models/malpractice_lstm.onnx",
        trt_path="exported_models/malpractice_lstm_fp16.engine",
        pytorch_model=my_lstm_model,
    )
    logits = engine.infer(sequence_tensor)
    engine.print_status()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger("edge_inference")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BACKEND_DIR / "exported_models"


# ---------------------------------------------------------------------------
# Runtime providers
# ---------------------------------------------------------------------------

class RuntimeBackend(str, Enum):
    TRT = "tensorrt"
    ONNXRT_CUDA = "onnxruntime_cuda"
    ONNXRT_CPU = "onnxruntime_cpu"
    PYTORCH_CUDA = "pytorch_cuda"
    PYTORCH_CPU = "pytorch_cpu"
    NONE = "none"


# ---------------------------------------------------------------------------
# Feature availability checks
# ---------------------------------------------------------------------------

def _check_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _check_onnxruntime() -> tuple[bool, list[str]]:
    """Returns (available, list_of_providers)."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return True, providers
    except ImportError:
        return False, []


def _check_pycuda() -> bool:
    try:
        import pycuda  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------

@dataclass
class InferenceOutput:
    """Result from a single inference call."""
    logits: np.ndarray           # (B, C) raw logits
    predictions: np.ndarray      # (B,) argmax class indices
    probabilities: np.ndarray    # (B, C) softmax probabilities
    backend_used: str
    latency_ms: float
    input_shape: list[int]

    def to_dict(self) -> dict:
        return {
            "backend": self.backend_used,
            "latency_ms": round(self.latency_ms, 3),
            "predictions": self.predictions.tolist(),
            "probabilities": self.probabilities.tolist(),
            "input_shape": self.input_shape,
        }


# ---------------------------------------------------------------------------
# TensorRT engine wrapper
# ---------------------------------------------------------------------------

class _TRTEngineWrapper:
    """TensorRT engine inference wrapper using pycuda."""

    def __init__(self, engine_path: str) -> None:
        self.engine_path = engine_path
        self._engine = None
        self._context = None
        self._stream = None
        self._loaded = False

    def load(self) -> bool:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(self.engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()
            self._stream = cuda.Stream()
            self._loaded = True
            logger.info(f"[trt] loaded engine: {self.engine_path}")
            return True
        except Exception as e:
            logger.warning(f"[trt] failed to load {self.engine_path}: {e}")
            return False

    def infer(self, input_array: np.ndarray) -> Optional[np.ndarray]:
        """Run inference on a numpy array, return numpy output."""
        if not self._loaded:
            return None

        try:
            import pycuda.driver as cuda

            import tensorrt as trt

            batch_size, seq_len, feat_dim = input_array.shape

            # Allocate device memory
            d_input = cuda.mem_alloc(input_array.nbytes)
            output_shape = (batch_size, 4)  # 4 classes
            output = np.empty(output_shape, dtype=np.float32)
            d_output = cuda.mem_alloc(output.nbytes)

            # Set input shape for dynamic networks
            self._context.set_input_shape("input", input_array.shape)

            # Transfer input → device
            cuda.memcpy_htod_async(d_input, input_array.ravel(), self._stream)

            # Run inference
            self._context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=self._stream.handle,
            )

            # Transfer output → host
            cuda.memcpy_dtoh_async(output, d_output, self._stream)
            self._stream.synchronize()

            return output

        except Exception as e:
            logger.error(f"[trt] inference error: {e}")
            return None

    def unload(self) -> None:
        self._loaded = False
        self._engine = None
        self._context = None
        self._stream = None


# ---------------------------------------------------------------------------
# ONNX Runtime wrapper
# ---------------------------------------------------------------------------

class _ONNXRTWrapper:
    """ONNX Runtime inference wrapper with provider fallback."""

    def __init__(self, onnx_path: str) -> None:
        self.onnx_path = onnx_path
        self._session = None
        self._provider = None
        self._loaded = False

    def load(self, preferred_providers: Optional[list[str]] = None) -> bool:
        try:
            import onnxruntime as ort

            if preferred_providers is None:
                available = ort.get_available_providers()
                preferred_providers = []
                if "TensorrtExecutionProvider" in available:
                    preferred_providers.append("TensorrtExecutionProvider")
                if "CUDAExecutionProvider" in available:
                    preferred_providers.append("CUDAExecutionProvider")
                preferred_providers.append("CPUExecutionProvider")

            # Filter to only available providers
            available = ort.get_available_providers()
            providers = [p for p in preferred_providers if p in available]

            self._session = ort.InferenceSession(
                self.onnx_path,
                providers=providers,
            )
            self._provider = self._session.get_providers()[0]
            self._loaded = True
            logger.info(f"[onnxrt] loaded: {self.onnx_path} (provider={self._provider})")
            return True

        except Exception as e:
            logger.warning(f"[onnxrt] failed to load {self.onnx_path}: {e}")
            return False

    def infer(self, input_array: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded:
            return None

        try:
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: input_array})
            return output[0]
        except Exception as e:
            logger.error(f"[onnxrt] inference error: {e}")
            return None

    def unload(self) -> None:
        self._loaded = False
        self._session = None


# ---------------------------------------------------------------------------
# PyTorch JIT wrapper
# ---------------------------------------------------------------------------

class _PyTorchJITWrapper:
    """PyTorch JIT-traced model wrapper with CUDA/CPU fallback."""

    def __init__(self, model: Optional[torch.nn.Module] = None) -> None:
        self._model = model
        self._traced = None
        self._device = None
        self._loaded = False

    def load(self, model: Optional[torch.nn.Module] = None) -> bool:
        if model is not None:
            self._model = model

        if self._model is None:
            return False

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(self._device).eval()

            # JIT trace with dummy input
            dummy = torch.randn(1, 30, 117).to(self._device)
            self._traced = torch.jit.trace(self._model, dummy)
            self._loaded = True
            logger.info(f"[pytorch] traced model on {self._device}")
            return True

        except Exception as e:
            logger.warning(f"[pytorch] trace failed: {e}")
            return False

    def infer(self, input_array: np.ndarray) -> Optional[np.ndarray]:
        if not self._loaded:
            return None

        try:
            tensor = torch.from_numpy(input_array).to(self._device)
            with torch.no_grad():
                output = self._traced(tensor)
            return output.cpu().numpy()
        except Exception as e:
            logger.error(f"[pytorch] inference error: {e}")
            return None

    def unload(self) -> None:
        self._loaded = False
        self._traced = None
        self._model = None


# ---------------------------------------------------------------------------
# AcceleratedInferenceEngine
# ---------------------------------------------------------------------------

class AcceleratedInferenceEngine:
    """
    Unified inference engine with automatic runtime provider fallback.

    Tries providers in order: TensorRT → ONNX Runtime → PyTorch.
    Once a provider loads successfully, it becomes the active backend.

    Args:
        onnx_path: path to ONNX model file.
        trt_path: path to TensorRT engine file (optional).
        pytorch_model: PyTorch nn.Module instance (optional).
        class_names: list of class names for the output.
        input_shape: expected input shape tuple ``(B, T, F)``.
        sequence_len: sequence length (default 30).
        input_dim: feature dimension (default 117).
    """

    CLASS_NAMES = ["Normal", "Head Turning", "Note Passing", "Peeking"]

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        trt_path: Optional[str] = None,
        pytorch_model: Optional[torch.nn.Module] = None,
        class_names: Optional[list[str]] = None,
        sequence_len: int = 30,
        input_dim: int = 117,
    ) -> None:
        self.onnx_path = onnx_path
        self.trt_path = trt_path
        self.class_names = class_names or self.CLASS_NAMES
        self.sequence_len = sequence_len
        self.input_dim = input_dim

        # Runtime wrappers
        self._trt = _TRTEngineWrapper(trt_path) if trt_path else None
        self._onnxrt = _ONNXRTWrapper(onnx_path) if onnx_path else None
        self._pytorch = _PyTorchJITWrapper(pytorch_model) if pytorch_model else None

        # Active backend
        self._active: RuntimeBackend = RuntimeBackend.NONE
        self._warm = False

        # Stats
        self._total_inferences = 0
        self._total_latency_ms = 0.0
        self._latencies: list[float] = []

    def load(self) -> RuntimeBackend:
        """
        Try to load the fastest available backend.

        Returns the RuntimeBackend that was successfully loaded.
        """
        logger.info("[engine] probing runtime providers...")

        # 1. Try TensorRT
        if self._trt and _check_tensorrt() and _check_pycuda():
            if self._trt.load():
                self._active = RuntimeBackend.TRT
                self._warmup()
                return self._active

        # 2. Try ONNX Runtime with CUDA/TensorRT providers
        if self._onnxrt:
            has_ort, _ = _check_onnxruntime()
            if has_ort:
                if self._onnxrt.load():
                    provider = self._onnxrt._provider or "CPUExecutionProvider"
                    if "cuda" in provider.lower() or "tensorrt" in provider.lower():
                        self._active = RuntimeBackend.ONNXRT_CUDA
                    else:
                        self._active = RuntimeBackend.ONNXRT_CPU
                    self._warmup()
                    return self._active

        # 3. Try PyTorch JIT
        if self._pytorch:
            if self._pytorch.load():
                device = self._pytorch._device
                if device and device.type == "cuda":
                    self._active = RuntimeBackend.PYTORCH_CUDA
                else:
                    self._active = RuntimeBackend.PYTORCH_CPU
                self._warmup()
                return self._active

        logger.warning("[engine] no runtime provider available")
        self._active = RuntimeBackend.NONE
        return self._active

    def _warmup(self) -> None:
        """Run a warm-up inference to initialise CUDA kernels."""
        if self._warm:
            return
        dummy = np.random.randn(1, self.sequence_len, self.input_dim).astype(np.float32)
        self.infer(dummy)
        self._warm = True
        logger.info(f"[engine] warmed up on {self._active.value}")

    def infer(self, input_array: np.ndarray) -> Optional[InferenceOutput]:
        """
        Run inference on a single or batched input.

        Args:
            input_array: numpy array of shape ``(B, T, F)`` or ``(T, F)``.

        Returns:
            InferenceOutput with logits, predictions, probabilities.
        """
        # Ensure correct shape
        if input_array.ndim == 2:
            input_array = input_array[np.newaxis, ...]
        input_array = input_array.astype(np.float32)

        batch_size = input_array.shape[0]

        # Dispatch to active backend
        t0 = time.monotonic()
        logits = None

        if self._active == RuntimeBackend.TRT:
            logits = self._trt.infer(input_array)
        elif self._active in (RuntimeBackend.ONNXRT_CUDA, RuntimeBackend.ONNXRT_CPU):
            logits = self._onnxrt.infer(input_array)
        elif self._active in (RuntimeBackend.PYTORCH_CUDA, RuntimeBackend.PYTORCH_CPU):
            logits = self._pytorch.infer(input_array)

        latency_ms = (time.monotonic() - t0) * 1000

        if logits is None:
            return None

        # Post-process
        probs = _softmax(logits, axis=1)
        preds = np.argmax(logits, axis=1)

        self._total_inferences += 1
        self._total_latency_ms += latency_ms
        self._latencies.append(latency_ms)

        return InferenceOutput(
            logits=logits,
            predictions=preds,
            probabilities=probs,
            backend_used=self._active.value,
            latency_ms=latency_ms,
            input_shape=list(input_array.shape),
        )

    def infer_batch(
        self, batch: np.ndarray, batch_size: int = 32,
    ) -> list[InferenceOutput]:
        """Run inference on a large array in mini-batches."""
        results = []
        n = len(batch)
        for start in range(0, n, batch_size):
            chunk = batch[start:start + batch_size]
            out = self.infer(chunk)
            if out is not None:
                results.append(out)
        return results

    def unload(self) -> None:
        """Release all runtime resources."""
        if self._trt:
            self._trt.unload()
        if self._onnxrt:
            self._onnxrt.unload()
        if self._pytorch:
            self._pytorch.unload()
        self._active = RuntimeBackend.NONE

    # ------------------------------------------------------------------
    # Stats / reporting
    # ------------------------------------------------------------------

    @property
    def active_backend(self) -> str:
        return self._active.value

    def stats(self) -> dict:
        """Return inference statistics."""
        latencies = self._latencies if self._latencies else [0.0]
        return {
            "active_backend": self._active.value,
            "total_inferences": self._total_inferences,
            "avg_latency_ms": round(
                self._total_latency_ms / max(self._total_inferences, 1), 3
            ),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
            "throughput_fps": round(
                1000.0 / max(self._total_latency_ms / max(self._total_inferences, 1), 0.001),
                1,
            ),
            "warm": self._warm,
        }

    def print_status(self) -> None:
        """Print a formatted status report."""
        s = self.stats()
        print(f"\n{'='*50}")
        print(f"  AcceleratedInferenceEngine Status")
        print(f"{'='*50}")
        print(f"  Backend:      {s['active_backend']}")
        print(f"  Inferences:   {s['total_inferences']}")
        print(f"  Avg latency:  {s['avg_latency_ms']:.3f} ms")
        print(f"  P50:          {s['p50_latency_ms']:.3f} ms")
        print(f"  P95:          {s['p95_latency_ms']:.3f} ms")
        print(f"  P99:          {s['p99_latency_ms']:.3f} ms")
        print(f"  Throughput:   {s['throughput_fps']:.1f} FPS")
        print(f"  Warm:         {s['warm']}")
        print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_edge_engine(
    model_type: str = "lstm",
    checkpoint_path: Optional[str] = None,
    export_dir: Path = EXPORT_DIR,
    device: str = "auto",
) -> AcceleratedInferenceEngine:
    """
    Create an AcceleratedInferenceEngine with the best available models.

    Args:
        model_type: "lstm" or "gru".
        checkpoint_path: path to PyTorch checkpoint.
        export_dir: directory containing exported ONNX/TRT models.
        device: "auto", "cuda", or "cpu".

    Returns:
        Configured AcceleratedInferenceEngine (call .load() to activate).
    """
    # Find available model files
    onnx_path = str(export_dir / f"malpractice_{model_type}.onnx")
    trt_path = str(export_dir / f"malpractice_{model_type}_fp16.engine")

    # Build PyTorch fallback if checkpoint exists
    pytorch_model = None
    if checkpoint_path and Path(checkpoint_path).exists():
        from temporal_training import MalpracticeLSTM, MalpracticeGRU, load_checkpoint

        if model_type == "lstm":
            pytorch_model = MalpracticeLSTM(
                input_dim=117, hidden_dim=128, num_layers=2,
                num_classes=4, dropout=0.0, bidirectional=True,
                use_attention=True,
            )
        else:
            pytorch_model = MalpracticeGRU(
                input_dim=117, hidden_dim=128, num_layers=2,
                num_classes=4, dropout=0.0, bidirectional=True,
                use_attention=True,
            )
        load_checkpoint(pytorch_model, checkpoint_path)

    engine = AcceleratedInferenceEngine(
        onnx_path=onnx_path if Path(onnx_path).exists() else None,
        trt_path=trt_path if Path(trt_path).exists() else None,
        pytorch_model=pytorch_model,
    )

    return engine


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick self-test with synthetic data."""
    print("\n  Edge Inference Engine — Self Test")
    print("  " + "-"*40)

    engine = AcceleratedInferenceEngine(
        onnx_path=None,
        trt_path=None,
        pytorch_model=None,
    )

    from temporal_training import MalpracticeLSTM
    model = MalpracticeLSTM(
        input_dim=117, hidden_dim=128, num_layers=2,
        num_classes=4, dropout=0.0, bidirectional=True,
        use_attention=True,
    )
    engine._pytorch = _PyTorchJITWrapper(model)
    engine.load()

    # Run 100 inferences
    for _ in range(100):
        dummy = np.random.randn(1, 30, 117).astype(np.float32)
        out = engine.infer(dummy)

    engine.print_status()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
