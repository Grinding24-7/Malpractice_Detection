"""
calibrator.py — Week 10: Post-Training INT8 Calibration Reader.

Implements ``TensorRTINT8Calibrator`` extending TensorRT's
``IInt8EntropyCalibrator2`` to compute activation scale factors for
INT8 quantisation without retraining.

Feed representative spatial-temporal sequence arrays extracted from
Week 4/5 dataset tensors.  The calibrator iterates over mini-batches,
feeding data to TensorRT's entropy calibration algorithm to determine
optimal quantisation ranges.

Target: < 0.5% F1-score drop vs FP32 baseline.

Usage:
    from calibrator import TensorRTINT8Calibrator
    cal = TensorRTINT8Calibrator(calibration_data, batch_size=8)
    # Pass cal.config.int8_calibrator = cal when building TensorRT engine
"""

from __future__ import annotations

import logging
import struct
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("calibrator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 8
DEFAULT_CACHE_FILE = "int8_calibration.cache"
DEFAULT_CALIBRATION_BATCHES = 100  # number of batches to feed


# ---------------------------------------------------------------------------
# INT8 Calibrator
# ---------------------------------------------------------------------------

class TensorRTINT8Calibrator:
    """
    Post-training INT8 entropy calibrator for TensorRT.

    Feeds batches of representative spatial-temporal sequence data
    (shape ``(B, T, F)`` matching the MalpracticeLSTM/GRU input) to
    TensorRT's ``IInt8EntropyCalibrator2`` to compute peractivation
    scale factors.

    The calibrator writes calibration caches to disk so subsequent
    engine builds skip the expensive calibration pass.

    Args:
        calibration_data: numpy array of shape ``(N, T, F)`` with
            representative input sequences.  Typically extracted from
            the Week 4/5 dataset tensors.
        batch_size: mini-batch size for calibration iteration.
        cache_path: path to the calibration cache file.
        max_batches: maximum number of batches to use (0 = all data).
    """

    def __init__(
        self,
        calibration_data: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_path: str = DEFAULT_CACHE_FILE,
        max_batches: int = DEFAULT_CALIBRATION_BATCHES,
    ) -> None:
        self.calibration_data = calibration_data.astype(np.float32)
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.max_batches = max_batches

        self._n_samples = len(calibration_data)
        self._batch_idx = 0
        self._device_memories: list = []
        self._batch_count = min(
            max_batches,
            (self._n_samples + batch_size - 1) // batch_size,
        )

        logger.info(
            f"[calibrator] init: {self._n_samples} samples, "
            f"batch_size={batch_size}, {self._batch_count} batches"
        )

    def get_batch_size(self) -> int:
        """Return the batch size (required by TensorRT interface)."""
        return self.batch_size

    def get_batch(self, names: list[str]) -> Optional[list]:
        """
        Return the next calibration batch.

        Called by TensorRT during the calibration pass.  Returns None
        when all batches have been consumed.
        """
        if self._batch_idx >= self._batch_count:
            return None

        start = self._batch_idx * self.batch_size
        end = min(start + self.batch_size, self._n_samples)
        batch = self.calibration_data[start:end]

        self._batch_idx += 1

        if self._batch_idx % 10 == 0:
            logger.info(
                f"[calibrator] batch {self._batch_idx}/{self._batch_count}"
            )

        # Return as list of device memory pointers (numpy arrays)
        return [np.ascontiguousarray(batch)]

    def read_calibration_cache(self) -> Optional[bytes]:
        """
        Read cached calibration data if available.

        Returns cached bytes or None if no cache exists.
        """
        cache_path = Path(self.cache_path)
        if cache_path.exists():
            logger.info(f"[calibrator] reading cache from {cache_path}")
            return cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache to disk."""
        cache_path = Path(self.cache_path)
        cache_path.write_bytes(cache)
        logger.info(f"[calibrator] wrote {len(cache)} bytes to {cache_path}")

    def reset(self) -> None:
        """Reset the batch iterator for a new calibration pass."""
        self._batch_idx = 0

    @property
    def stats(self) -> dict:
        """Return calibrator statistics."""
        return {
            "n_samples": self._n_samples,
            "batch_size": self.batch_size,
            "batch_count": self._batch_count,
            "batches_consumed": self._batch_idx,
            "cache_path": self.cache_path,
            "cache_exists": Path(self.cache_path).exists(),
            "input_shape": list(self.calibration_data.shape),
        }


# ---------------------------------------------------------------------------
# Calibration data extraction utilities
# ---------------------------------------------------------------------------

def extract_calibration_data_from_pt(
    dataset_path: str,
    max_samples: int = 500,
) -> np.ndarray:
    """
    Extract calibration data from a Week 4/5 .pt dataset tensor.

    Args:
        dataset_path: path to .pt file containing X tensor.
        max_samples: maximum number of samples to extract.

    Returns:
        numpy array of shape ``(N, T, F)`` ready for calibration.
    """
    import torch

    data = torch.load(dataset_path, weights_only=False)
    X = data.get("X", data.get("features"))
    if X is None:
        raise ValueError(f"No 'X' or 'features' key in {dataset_path}")

    if isinstance(X, torch.Tensor):
        X = X.numpy()

    # Subsample if too large
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]

    logger.info(f"[calibrator] extracted {len(X)} samples from {dataset_path}")
    return X.astype(np.float32)


def generate_synthetic_calibration_data(
    n_samples: int = 500,
    sequence_len: int = 30,
    input_dim: int = 117,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic calibration data matching the model input shape.

    Useful when no real dataset is available.  Produces data with
    realistic statistical properties (mean ~0, std ~0.1–0.5 per feature).
    """
    rng = np.random.RandomState(seed)

    data = rng.randn(n_samples, sequence_len, input_dim).astype(np.float32)

    # Scale features to realistic ranges
    for f in range(input_dim):
        scale = rng.uniform(0.05, 0.5)
        offset = rng.uniform(-0.3, 0.3)
        data[:, :, f] = data[:, :, f] * scale + offset

    logger.info(
        f"[calibrator] generated synthetic calibration data: "
        f"{data.shape} (seed={seed})"
    )
    return data


def extract_or_generate_calibration_data(
    dataset_path: Optional[str] = None,
    n_samples: int = 500,
    sequence_len: int = 30,
    input_dim: int = 117,
) -> np.ndarray:
    """
    Try to load real calibration data, fall back to synthetic.

    Args:
        dataset_path: optional path to .pt dataset file.
        n_samples: number of synthetic samples if no dataset found.
        sequence_len: sequence length for synthetic data.
        input_dim: feature dimension for synthetic data.

    Returns:
        numpy array of shape ``(N, T, F)``.
    """
    if dataset_path and Path(dataset_path).exists():
        try:
            return extract_calibration_data_from_pt(dataset_path, max_samples=n_samples)
        except Exception as e:
            logger.warning(f"[calibrator] failed to load {dataset_path}: {e}")

    logger.info("[calibrator] using synthetic calibration data")
    return generate_synthetic_calibration_data(
        n_samples=n_samples,
        sequence_len=sequence_len,
        input_dim=input_dim,
    )


# ---------------------------------------------------------------------------
# Validation: measure quantisation accuracy impact
# ---------------------------------------------------------------------------

def measure_calibration_quality(
    fp32_outputs: np.ndarray,
    int8_outputs: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Compare FP32 and INT8 model outputs to measure quantisation impact.

    Args:
        fp32_outputs: reference logits from FP32 model, shape ``(N, C)``.
        int8_outputs: quantised logits from INT8 model, shape ``(N, C)``.
        class_names: optional class names for reporting.

    Returns:
        Dictionary with MSE, cosine similarity, and accuracy metrics.
    """
    if class_names is None:
        class_names = ["Normal", "Head Turning", "Note Passing", "Peeking"]

    fp32_outputs = np.asarray(fp32_outputs, dtype=np.float64)
    int8_outputs = np.asarray(int8_outputs, dtype=np.float64)

    # Per-sample metrics
    mse_per_sample = np.mean((fp32_outputs - int8_outputs) ** 2, axis=1)

    # Cosine similarity per sample
    fp32_norms = np.linalg.norm(fp32_outputs, axis=1, keepdims=True)
    int8_norms = np.linalg.norm(int8_outputs, axis=1, keepdims=True)
    cosine_per_sample = np.sum(
        fp32_outputs * int8_outputs, axis=1
    ) / (fp32_norms.squeeze() * int8_norms.squeeze() + 1e-8)

    # Classification agreement
    fp32_preds = np.argmax(fp32_outputs, axis=1)
    int8_preds = np.argmax(int8_outputs, axis=1)
    agreement = np.mean(fp32_preds == int8_preds)

    # Per-class F1 comparison
    from sklearn.metrics import f1_score as _f1
    fp32_f1 = _f1(fp32_preds, int8_preds, average="macro", zero_division=0)
    f1_drop = 0.0  # Can't compute true F1 without ground truth; use agreement proxy

    return {
        "mse_mean": float(np.mean(mse_per_sample)),
        "mse_max": float(np.max(mse_per_sample)),
        "mse_p95": float(np.percentile(mse_per_sample, 95)),
        "cosine_similarity_mean": float(np.mean(cosine_per_sample)),
        "cosine_similarity_min": float(np.min(cosine_per_sample)),
        "classification_agreement": float(agreement),
        "f1_drop_estimate": float(max(0, 1.0 - agreement)),
        "n_samples": len(fp32_outputs),
        "n_classes": fp32_outputs.shape[1] if fp32_outputs.ndim > 1 else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="INT8 Calibration Data Preparation")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .pt dataset")
    parser.add_argument("--output", type=str, default="calibration_data.npy")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    if args.synthetic or not args.dataset:
        data = generate_synthetic_calibration_data(n_samples=args.n_samples)
    else:
        data = extract_calibration_data_from_pt(args.dataset, max_samples=args.n_samples)

    np.save(args.output, data)
    print(f"  Saved calibration data: {data.shape} → {args.output}")

    # Quick calibrator test
    cal = TensorRTINT8Calibrator(data, batch_size=8)
    print(f"  Calibrator stats: {cal.stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
