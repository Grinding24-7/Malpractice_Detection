"""
benchmark_quant.py — Week 10: Parity & Benchmark Validation.

Measures and logs execution metrics across model formats:
    - Mean & P95 latency (ms)
    - Throughput (FPS)
    - Model file footprint (MB)
    - Cosine similarity / MSE between PyTorch FP32 and quantised outputs

Compares:
    1. PyTorch FP32 (baseline)
    2. ONNX Runtime FP32
    3. TensorRT FP16
    4. TensorRT INT8

Usage:
    python benchmark_quant.py --model lstm
    python benchmark_quant.py --model gru --warmup 50 --iterations 500
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BACKEND_DIR / "exported_models"
DEFAULT_INPUT_DIM = 117
DEFAULT_SEQUENCE_LEN = 30
DEFAULT_NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results from benchmarking a single backend."""
    backend: str
    model_format: str  # "pytorch_fp32", "onnx_fp32", "trt_fp16", "trt_int8"
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_fps: float = 0.0
    file_size_mb: float = 0.0
    iterations: int = 0
    warmup_time_s: float = 0.0
    benchmark_time_s: float = 0.0
    # Parity metrics vs FP32 baseline
    cosine_similarity: Optional[float] = None
    mse: Optional[float] = None
    max_abs_diff: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "backend": self.backend,
            "model_format": self.model_format,
            "latency_mean_ms": round(self.latency_mean_ms, 3),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_p99_ms": round(self.latency_p99_ms, 3),
            "throughput_fps": round(self.throughput_fps, 1),
            "file_size_mb": round(self.file_size_mb, 2),
            "iterations": self.iterations,
        }
        if self.cosine_similarity is not None:
            d["cosine_similarity"] = round(self.cosine_similarity, 6)
        if self.mse is not None:
            d["mse"] = round(self.mse, 8)
        if self.max_abs_diff is not None:
            d["max_abs_diff"] = round(self.max_abs_diff, 6)
        return d


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _make_synthetic_batch(
    batch_size: int = 1,
    seq_len: int = DEFAULT_SEQUENCE_LEN,
    input_dim: int = DEFAULT_INPUT_DIM,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic input data for benchmarking."""
    rng = np.random.RandomState(seed)
    data = rng.randn(batch_size, seq_len, input_dim).astype(np.float32)
    return data


# ---------------------------------------------------------------------------
# Parity comparison
# ---------------------------------------------------------------------------

def compute_parity(
    reference: np.ndarray,
    test: np.ndarray,
) -> dict:
    """
    Compute parity metrics between reference (FP32) and test outputs.

    Args:
        reference: FP32 logits, shape ``(N, C)``
        test: quantised logits, shape ``(N, C)``

    Returns:
        Dict with cosine_similarity, mse, max_abs_diff.
    """
    ref = reference.astype(np.float64)
    tst = test.astype(np.float64)

    # Cosine similarity (per-sample, then mean)
    ref_norm = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
    tst_norm = np.linalg.norm(tst, axis=1, keepdims=True) + 1e-8
    cosine = np.sum(ref * tst, axis=1) / (ref_norm.squeeze() * tst_norm.squeeze())
    cosine_mean = float(np.mean(cosine))

    # MSE
    mse = float(np.mean((ref - tst) ** 2))

    # Max absolute difference
    max_diff = float(np.max(np.abs(ref - tst)))

    return {
        "cosine_similarity": cosine_mean,
        "mse": mse,
        "max_abs_diff": max_diff,
    }


# ---------------------------------------------------------------------------
# Individual backend benchmarks
# ---------------------------------------------------------------------------

def _benchmark_pytorch(
    model: torch.nn.Module,
    data: np.ndarray,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> BenchmarkResult:
    """Benchmark PyTorch FP32 model."""
    model = model.to(device).eval()
    dummy = torch.from_numpy(data).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Benchmark
    latencies = []
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.monotonic()
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.monotonic()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.monotonic() - t0) * 1000)
    bench_time = time.monotonic() - t_start

    # Reference output
    with torch.no_grad():
        ref_output = model(dummy).cpu().numpy()

    latencies = np.array(latencies)

    return BenchmarkResult(
        backend=f"pytorch_{device.type}",
        model_format="pytorch_fp32",
        latency_mean_ms=float(np.mean(latencies)),
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        latency_p99_ms=float(np.percentile(latencies, 99)),
        throughput_fps=iterations / bench_time,
        file_size_mb=0.0,  # no file for eager model
        iterations=iterations,
        benchmark_time_s=bench_time,
    ), ref_output


def _benchmark_onnxrt(
    onnx_path: str,
    data: np.ndarray,
    warmup: int,
    iterations: int,
    ref_output: Optional[np.ndarray] = None,
) -> BenchmarkResult:
    """Benchmark ONNX Runtime FP32 model."""
    try:
        import onnxruntime as ort
    except ImportError:
        return BenchmarkResult(
            backend="onnxruntime",
            model_format="onnx_fp32",
        )

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: data})

    # Benchmark
    latencies = []
    t_start = time.monotonic()
    for _ in range(iterations):
        t0 = time.monotonic()
        output = session.run(None, {input_name: data})
        latencies.append((time.monotonic() - t0) * 1000)
    bench_time = time.monotonic() - t_start

    onnx_output = output[0]
    latencies = np.array(latencies)

    result = BenchmarkResult(
        backend=providers[0],
        model_format="onnx_fp32",
        latency_mean_ms=float(np.mean(latencies)),
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        latency_p99_ms=float(np.percentile(latencies, 99)),
        throughput_fps=iterations / bench_time,
        file_size_mb=Path(onnx_path).stat().st_size / (1024 * 1024),
        iterations=iterations,
        benchmark_time_s=bench_time,
    )

    if ref_output is not None:
        parity = compute_parity(ref_output, onnx_output)
        result.cosine_similarity = parity["cosine_similarity"]
        result.mse = parity["mse"]
        result.max_abs_diff = parity["max_abs_diff"]

    return result


def _benchmark_trt(
    engine_path: str,
    data: np.ndarray,
    warmup: int,
    iterations: int,
    ref_output: Optional[np.ndarray] = None,
) -> BenchmarkResult:
    """Benchmark TensorRT engine (FP16 or INT8)."""
    if not Path(engine_path).exists():
        return BenchmarkResult(
            backend="tensorrt",
            model_format="trt",
        )

    try:
        from edge_inference import _TRTEngineWrapper
    except ImportError:
        return BenchmarkResult(
            backend="tensorrt",
            model_format="trt",
        )

    wrapper = _TRTEngineWrapper(engine_path)
    if not wrapper.load():
        return BenchmarkResult(
            backend="tensorrt",
            model_format="trt",
        )

    # Warmup
    for _ in range(warmup):
        wrapper.infer(data)

    # Benchmark
    latencies = []
    t_start = time.monotonic()
    for _ in range(iterations):
        t0 = time.monotonic()
        output = wrapper.infer(data)
        latencies.append((time.monotonic() - t0) * 1000)
    bench_time = time.monotonic() - t_start

    latencies = np.array(latencies)

    fmt = "trt_fp16" if "fp16" in engine_path else ("trt_int8" if "int8" in engine_path else "trt_fp32")

    result = BenchmarkResult(
        backend="tensorrt",
        model_format=fmt,
        latency_mean_ms=float(np.mean(latencies)),
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        latency_p99_ms=float(np.percentile(latencies, 99)),
        throughput_fps=iterations / bench_time,
        file_size_mb=Path(engine_path).stat().st_size / (1024 * 1024),
        iterations=iterations,
        benchmark_time_s=bench_time,
    )

    if ref_output is not None and output is not None:
        parity = compute_parity(ref_output, output)
        result.cosine_similarity = parity["cosine_similarity"]
        result.mse = parity["mse"]
        result.max_abs_diff = parity["max_abs_diff"]

    wrapper.unload()
    return result


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------

def run_benchmark(
    model_type: str = "lstm",
    checkpoint_path: Optional[str] = None,
    export_dir: Path = EXPORT_DIR,
    warmup: int = 20,
    iterations: int = 200,
    batch_sizes: Optional[list[int]] = None,
) -> list[BenchmarkResult]:
    """
    Run full benchmark across all available model formats.

    Returns list of BenchmarkResult for each backend tested.
    """
    if batch_sizes is None:
        batch_sizes = [1]

    results: list[BenchmarkResult] = []
    ref_output: Optional[np.ndarray] = None

    # 1. PyTorch FP32 (baseline)
    print("\n  [1/4] PyTorch FP32 baseline...")
    from temporal_training import MalpracticeLSTM, MalpracticeGRU

    if model_type == "lstm":
        model = MalpracticeLSTM(
            input_dim=DEFAULT_INPUT_DIM, hidden_dim=128, num_layers=2,
            num_classes=DEFAULT_NUM_CLASSES, dropout=0.0,
            bidirectional=True, use_attention=True,
        )
    else:
        model = MalpracticeGRU(
            input_dim=DEFAULT_INPUT_DIM, hidden_dim=128, num_layers=2,
            num_classes=DEFAULT_NUM_CLASSES, dropout=0.0,
            bidirectional=True, use_attention=True,
        )

    # Load checkpoint if available
    if checkpoint_path and Path(checkpoint_path).exists():
        from temporal_training import load_checkpoint
        load_checkpoint(model, checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _make_synthetic_batch(batch_size=1)

    pytorch_result, ref_output = _benchmark_pytorch(
        model, data, warmup, iterations, device,
    )
    results.append(pytorch_result)
    print(f"         {pytorch_result.latency_mean_ms:.2f} ms avg, "
          f"{pytorch_result.throughput_fps:.1f} FPS")

    # Free PyTorch model
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. ONNX Runtime
    onnx_path = str(export_dir / f"malpractice_{model_type}.onnx")
    if Path(onnx_path).exists():
        print("  [2/4] ONNX Runtime FP32...")
        onnx_result = _benchmark_onnxrt(
            onnx_path, data, warmup, iterations, ref_output,
        )
        results.append(onnx_result)
        print(f"         {onnx_result.latency_mean_ms:.2f} ms avg, "
              f"{onnx_result.throughput_fps:.1f} FPS")
    else:
        print(f"  [2/4] ONNX not found at {onnx_path} — skipping")

    # 3. TensorRT FP16
    trt_fp16_path = str(export_dir / f"malpractice_{model_type}_fp16.engine")
    if Path(trt_fp16_path).exists():
        print("  [3/4] TensorRT FP16...")
        trt_fp16_result = _benchmark_trt(
            trt_fp16_path, data, warmup, iterations, ref_output,
        )
        results.append(trt_fp16_result)
        print(f"         {trt_fp16_result.latency_mean_ms:.2f} ms avg, "
              f"{trt_fp16_result.throughput_fps:.1f} FPS")
    else:
        print(f"  [3/4] TRT FP16 not found — skipping")

    # 4. TensorRT INT8
    trt_int8_path = str(export_dir / f"malpractice_{model_type}_int8.engine")
    if Path(trt_int8_path).exists():
        print("  [4/4] TensorRT INT8...")
        trt_int8_result = _benchmark_trt(
            trt_int8_path, data, warmup, iterations, ref_output,
        )
        results.append(trt_int8_result)
        print(f"         {trt_int8_result.latency_mean_ms:.2f} ms avg, "
              f"{trt_int8_result.throughput_fps:.1f} FPS")
    else:
        print(f"  [4/4] TRT INT8 not found — skipping")

    return results


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def print_report(results: list[BenchmarkResult], model_type: str) -> None:
    """Print formatted benchmark report."""
    print(f"\n{'='*75}")
    print(f"  BENCHMARK REPORT — {model_type.upper()} Sequence Model")
    print(f"{'='*75}")

    if not results:
        print("  No results to display.")
        return

    # Header
    print(f"\n  {'Backend':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} "
          f"{'FPS':<10} {'Size (MB)':<12} {'Cosine':<10} {'MSE':<12}")
    print(f"  {'-'*70}")

    for r in results:
        cosine_str = f"{r.cosine_similarity:.6f}" if r.cosine_similarity is not None else "—"
        mse_str = f"{r.mse:.8f}" if r.mse is not None else "—"
        size_str = f"{r.file_size_mb:.2f}" if r.file_size_mb > 0 else "—"
        print(
            f"  {r.model_format:<20} "
            f"{r.latency_mean_ms:<12.3f} "
            f"{r.latency_p95_ms:<12.3f} "
            f"{r.throughput_fps:<10.1f} "
            f"{size_str:<12} "
            f"{cosine_str:<10} "
            f"{mse_str:<12}"
        )

    # Speedup analysis
    if len(results) > 1:
        baseline = results[0]
        print(f"\n  Speedup vs PyTorch FP32:")
        for r in results[1:]:
            if r.latency_mean_ms > 0 and baseline.latency_mean_ms > 0:
                speedup = baseline.latency_mean_ms / r.latency_mean_ms
                print(f"    {r.model_format}: {speedup:.2f}x")

    # Parity verdict
    for r in results[1:]:
        if r.cosine_similarity is not None:
            if r.cosine_similarity > 0.99:
                verdict = "PASS (cosine > 0.99)"
            elif r.cosine_similarity > 0.95:
                verdict = "MARGINAL (0.95 < cosine < 0.99)"
            else:
                verdict = "FAIL (cosine < 0.95)"
            print(f"\n  Parity [{r.model_format}]: {verdict}")

    print(f"\n{'='*75}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 10 Quantisation Benchmark")
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--checkpoint", default=str(BACKEND_DIR / "best_malpractice_model.pth"))
    parser.add_argument("--export-dir", default=str(EXPORT_DIR))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=str, default=None, help="Save JSON report")
    args = parser.parse_args()

    results = run_benchmark(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        export_dir=Path(args.export_dir),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print_report(results, args.model)

    if args.output:
        import json
        report = {
            "model_type": args.model,
            "results": [r.to_dict() for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
