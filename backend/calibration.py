"""
calibration.py — Week 9: Threshold Calibration & False-Positive Tuning.

Provides:
    1. ThresholdCalibrator — grid-search optimiser over threshold space
       to find optimal τ_head, τ_peek, τ_pass that minimise false
       positives while preserving recall.

    2. PersistenceFilter — minimum frame-persistence rules that require
       a behaviour to persist for N frames within a sliding window
       before it triggers an alert (eliminates momentary shifts from
       page-turning, stretching, etc.).

    3. CalibrationRunner — end-to-end pipeline that loads a labelled
       dataset, runs grid search, evaluates the best thresholds, and
       reports metrics.

Usage:
    # Grid search over synthetic data
    python calibration.py --mode grid-search --samples 500

    # Evaluate specific thresholds on labelled data
    python calibration.py --mode evaluate --data path/to/dataset.pt

    # Generate calibration report
    python calibration.py --mode report --data path/to/dataset.pt
"""

from __future__ import annotations

import argparse
import json
import itertools
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("calibration")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent

# Default threshold ranges for grid search
DEFAULT_TAU_HEAD_RANGE = np.arange(0.55, 0.80, 0.05)  # ear_ratio bounds
DEFAULT_TAU_PEEK_RANGE = np.arange(0.80, 0.98, 0.02)  # norm_vertical_drop bound
DEFAULT_TAU_PASS_RANGE = np.arange(0.60, 0.90, 0.05)  # combined confidence

# Persistence filter defaults
DEFAULT_PERSISTENCE_FRAMES = 12    # minimum anomaly frames to trigger
DEFAULT_WINDOW_SIZE = 30           # sliding window (matches SEQUENCE_LEN)


# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------

@dataclass
class ClassificationThresholds:
    """
    Tunable thresholds for the heuristic classifier.

    Attributes:
        tau_head_low:  ear_ratio below this → head turning (low ear_ratio)
        tau_head_high: ear_ratio above this → head turning (high ear_ratio)
        tau_peek:      norm_vertical_drop above this → peeking
        tau_conf_min:  minimum nose confidence to consider a detection
        tau_ear_conf:  minimum ear confidence for valid detection
        persistence_frames: anomaly must persist this many frames
        window_size:  sliding window for persistence check
    """
    tau_head_low: float = 0.70
    tau_head_high: float = 1.40
    tau_peek: float = 0.90
    tau_conf_min: float = 0.35
    tau_ear_conf: float = 0.20
    persistence_frames: int = DEFAULT_PERSISTENCE_FRAMES
    window_size: int = DEFAULT_WINDOW_SIZE

    def to_dict(self) -> dict:
        return {
            "tau_head_low": round(self.tau_head_low, 4),
            "tau_head_high": round(self.tau_head_high, 4),
            "tau_peek": round(self.tau_peek, 4),
            "tau_conf_min": round(self.tau_conf_min, 4),
            "tau_ear_conf": round(self.tau_ear_conf, 4),
            "persistence_frames": self.persistence_frames,
            "window_size": self.window_size,
        }


# ---------------------------------------------------------------------------
# Persistence filter
# ---------------------------------------------------------------------------

class PersistenceFilter:
    """
    Sliding-window persistence filter.

    Requires a behaviour to persist for at least ``min_frames`` within
    a ``window_size`` sliding window before it triggers an alert.
    This eliminates momentary false positives from:
        - Page turning
        - Stretching
        - Brief head movement
        - Glancing at a clock

    State is tracked per track_id.
    """

    def __init__(
        self,
        min_frames: int = DEFAULT_PERSISTENCE_FRAMES,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> None:
        self.min_frames = min_frames
        self.window_size = window_size
        self._buffers: dict[int, list[bool]] = {}

    def update(self, track_id: int, is_anomaly: bool) -> bool:
        """
        Update the filter with a new observation.

        Returns True only if the anomaly has persisted long enough
        within the window.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = []

        buf = self._buffers[track_id]
        buf.append(is_anomaly)

        # Keep only the window
        if len(buf) > self.window_size:
            buf.pop(0)

        # Count anomalies in window
        anomaly_count = sum(buf)
        return anomaly_count >= self.min_frames

    def should_alert(self, track_id: int) -> bool:
        """Check if the current buffer state warrants an alert."""
        buf = self._buffers.get(track_id, [])
        return sum(buf) >= self.min_frames

    def reset(self, track_id: int) -> None:
        """Clear buffer for a track (e.g., when track is lost)."""
        self._buffers.pop(track_id, None)

    def prune(self, active_ids: set[int]) -> int:
        """Remove buffers for tracks no longer active."""
        stale = [tid for tid in self._buffers if tid not in active_ids]
        for tid in stale:
            del self._buffers[tid]
        return len(stale)

    def clear(self) -> None:
        self._buffers.clear()

    @property
    def stats(self) -> dict:
        return {
            "tracked_tracks": len(self._buffers),
            "total_observations": sum(len(b) for b in self._buffers.values()),
        }


# ---------------------------------------------------------------------------
# Synthetic data generator (for calibration)
# ---------------------------------------------------------------------------

def _generate_synthetic_features(
    n_samples: int = 500, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic pose features with known labels.

    Returns:
        features: (N, 7) array — [ear_ratio, norm_vert_drop, shoulder_angle,
                  norm_nose_ear_drop, nose_conf, l_ear_conf, r_ear_conf]
        labels:   (N,) int array — 0=Normal, 1=HeadTurn, 2=Peeking, 3=Passing
    """
    rng = np.random.RandomState(seed)

    n_per_class = n_samples // 4
    remainder = n_samples - n_per_class * 4

    features_list = []
    labels_list = []

    # Class 0: Normal
    for _ in range(n_per_class + (1 if remainder > 0 else 0)):
        ear_ratio = rng.uniform(0.80, 1.20)
        norm_vert = rng.uniform(0.30, 0.75)
        shoulder = rng.uniform(-0.1, 0.3)
        norm_ear_drop = rng.uniform(0.1, 0.5)
        nose_conf = rng.uniform(0.6, 0.98)
        ear_conf = rng.uniform(0.5, 0.95)
        features_list.append([ear_ratio, norm_vert, shoulder, norm_ear_drop,
                              nose_conf, ear_conf, ear_conf])
        labels_list.append(0)
    remainder = max(0, remainder - 1)

    # Class 1: Head Turning (extreme ear_ratio)
    for _ in range(n_per_class + (1 if remainder > 0 else 0)):
        side = rng.choice([-1, 1])
        ear_ratio = side * rng.uniform(1.5, 2.5) if side < 0 else rng.uniform(1.45, 2.0)
        norm_vert = rng.uniform(0.30, 0.70)
        shoulder = rng.uniform(-0.3, 0.5)
        norm_ear_drop = rng.uniform(0.1, 0.4)
        nose_conf = rng.uniform(0.5, 0.95)
        ear_conf = rng.uniform(0.4, 0.90)
        features_list.append([abs(ear_ratio), norm_vert, shoulder, norm_ear_drop,
                              nose_conf, ear_conf, ear_conf])
        labels_list.append(1)
    remainder = max(0, remainder - 1)

    # Class 2: Peeking (high norm_vertical_drop)
    for _ in range(n_per_class + (1 if remainder > 0 else 0)):
        ear_ratio = rng.uniform(0.85, 1.15)
        norm_vert = rng.uniform(0.91, 1.5)
        shoulder = rng.uniform(-0.2, 0.4)
        norm_ear_drop = rng.uniform(0.5, 1.0)
        nose_conf = rng.uniform(0.55, 0.92)
        ear_conf = rng.uniform(0.45, 0.88)
        features_list.append([ear_ratio, norm_vert, shoulder, norm_ear_drop,
                              nose_conf, ear_conf, ear_conf])
        labels_list.append(2)
    remainder = max(0, remainder - 1)

    # Class 3: Note Passing (low confidence, unusual pose)
    for _ in range(n_per_class + remainder):
        ear_ratio = rng.uniform(0.75, 1.30)
        norm_vert = rng.uniform(0.50, 0.88)
        shoulder = rng.uniform(-0.4, 0.6)
        norm_ear_drop = rng.uniform(0.3, 0.8)
        nose_conf = rng.uniform(0.30, 0.65)
        ear_conf = rng.uniform(0.20, 0.60)
        features_list.append([ear_ratio, norm_vert, shoulder, norm_ear_drop,
                              nose_conf, ear_conf, ear_conf])
        labels_list.append(3)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Threshold calibrator (grid search)
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """
    Grid-search optimiser over threshold space.

    Searches for the combination of (tau_head_low, tau_head_high, tau_peek,
    persistence_frames) that maximises the F1 score on a labelled dataset.
    """

    def __init__(
        self,
        tau_head_low_range: Optional[np.ndarray] = None,
        tau_head_high_range: Optional[np.ndarray] = None,
        tau_peek_range: Optional[np.ndarray] = None,
        persistence_range: Optional[list[int]] = None,
    ) -> None:
        self.tau_head_low_range = tau_head_low_range if tau_head_low_range is not None else DEFAULT_TAU_HEAD_RANGE
        self.tau_head_high_range = tau_head_high_range if tau_head_high_range is not None else DEFAULT_TAU_HEAD_RANGE
        self.tau_peek_range = tau_peek_range if tau_peek_range is not None else DEFAULT_TAU_PEEK_RANGE
        self.persistence_range = persistence_range if persistence_range is not None else [6, 8, 10, 12, 15, 20]

        self._results: list[dict] = []
        self._best: Optional[dict] = None

    def grid_search(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Run exhaustive grid search over threshold combinations.

        Returns the best configuration with its F1, precision, recall.
        """
        combos = list(itertools.product(
            self.tau_head_low_range,
            self.tau_head_high_range,
            self.tau_peek_range,
            self.persistence_range,
        ))

        total = len(combos)
        if verbose:
            print(f"\n  Grid search: {total} threshold combinations")
            print(f"  tau_head_low:  {len(self.tau_head_low_range)} values "
                  f"({self.tau_head_low_range[0]:.2f}–{self.tau_head_low_range[-1]:.2f})")
            print(f"  tau_head_high: {len(self.tau_head_high_range)} values "
                  f"({self.tau_head_high_range[0]:.2f}–{self.tau_head_high_range[-1]:.2f})")
            print(f"  tau_peek:      {len(self.tau_peek_range)} values "
                  f"({self.tau_peek_range[0]:.2f}–{self.tau_peek_range[-1]:.2f})")
            print(f"  persistence:   {self.persistence_range}")
            print()

        best_f1 = -1.0
        best_config = None
        start = time.monotonic()

        for i, (th_low, th_high, th_peek, pers) in enumerate(combos):
            # Skip invalid combos
            if th_low >= th_high:
                continue

            thresholds = ClassificationThresholds(
                tau_head_low=th_low,
                tau_head_high=th_high,
                tau_peek=th_peek,
                persistence_frames=pers,
            )

            metrics = self._evaluate_thresholds(features, labels, thresholds)

            result = {
                "tau_head_low": th_low,
                "tau_head_high": th_high,
                "tau_peek": th_peek,
                "persistence_frames": pers,
                **metrics,
            }
            self._results.append(result)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_config = result.copy()

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.monotonic() - start
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(
                    f"  [{i+1:4d}/{total}] best_f1={best_f1:.4f}  "
                    f"rate={rate:.0f}/s  ETA={eta:.1f}s",
                    flush=True,
                )

        self._best = best_config

        if verbose:
            elapsed = time.monotonic() - start
            print(f"\n  Grid search completed in {elapsed:.1f}s")
            print(f"  Best F1: {best_f1:.4f}")
            if best_config:
                print(f"  Config:  tau_head_low={best_config['tau_head_low']:.2f}, "
                      f"tau_head_high={best_config['tau_head_high']:.2f}, "
                      f"tau_peek={best_config['tau_peek']:.2f}, "
                      f"persistence={best_config['persistence_frames']}")

        return best_config or {}

    def _evaluate_thresholds(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        thresholds: ClassificationThresholds,
    ) -> dict:
        """Evaluate a single threshold configuration."""
        predictions = self._classify_batch(features, thresholds)

        # Binary: anomaly vs normal (classes 1,2,3 vs 0)
        y_true = (labels > 0).astype(int)
        y_pred = (predictions > 0).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def _classify_batch(
        self,
        features: np.ndarray,
        thresholds: ClassificationThresholds,
    ) -> np.ndarray:
        """Classify a batch of features using given thresholds."""
        preds = np.zeros(len(features), dtype=np.int64)

        for i in range(len(features)):
            ear_ratio = features[i, 0]
            norm_vert = features[i, 1]
            nose_conf = features[i, 4]
            ear_conf_l = features[i, 5]
            ear_conf_r = features[i, 6]

            # Confidence gate
            if nose_conf < thresholds.tau_conf_min:
                preds[i] = 0
                continue
            if ear_conf_l < thresholds.tau_ear_conf or ear_conf_r < thresholds.tau_ear_conf:
                preds[i] = 0
                continue

            # Classification
            if ear_ratio < thresholds.tau_head_low or ear_ratio > thresholds.tau_head_high:
                preds[i] = 1  # Head turning
            elif norm_vert > thresholds.tau_peek:
                preds[i] = 2  # Peeking
            else:
                preds[i] = 0  # Normal

        return preds

    @property
    def best(self) -> Optional[dict]:
        return self._best

    @property
    def all_results(self) -> list[dict]:
        return sorted(self._results, key=lambda x: x.get("f1", 0), reverse=True)

    def top_n(self, n: int = 10) -> list[dict]:
        return self.all_results[:n]


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------

class CalibrationRunner:
    """
    End-to-end calibration pipeline:
        1. Generate or load labelled data
        2. Run grid search
        3. Apply persistence filter
        4. Evaluate and report
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = data_path
        self._features: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._thresholds: Optional[ClassificationThresholds] = None
        self._filter: Optional[PersistenceFilter] = None
        self._results: dict = {}

    def load_data(self) -> None:
        """Load labelled dataset or generate synthetic data."""
        if self.data_path and Path(self.data_path).exists():
            # Load from .pt or .npy
            path = Path(self.data_path)
            if path.suffix == ".pt":
                import torch
                data = torch.load(path, weights_only=False)
                self._features = data.get("X", data.get("features")).numpy()
                self._labels = data.get("labels").numpy()
            elif path.suffix == ".npy":
                self._features = np.load(str(path.with_suffix(".features.npy")))
                self._labels = np.load(str(path.with_suffix(".labels.npy")))
            else:
                raise ValueError(f"Unsupported format: {path.suffix}")
        else:
            print("  No dataset found — generating synthetic data (2000 samples)...")
            self._features, self._labels = _generate_synthetic_features(n_samples=2000)

        n_classes = len(np.unique(self._labels))
        print(f"  Loaded {len(self._features)} samples, {n_classes} classes")
        for c in range(n_classes):
            count = int(np.sum(self._labels == c))
            print(f"    Class {c}: {count} samples ({count/len(self._labels)*100:.1f}%)")

    def run_grid_search(self, verbose: bool = True) -> dict:
        """Run grid search and return best configuration."""
        if self._features is None:
            self.load_data()

        calibrator = ThresholdCalibrator()
        best = calibrator.grid_search(self._features, self._labels, verbose=verbose)

        self._thresholds = ClassificationThresholds(
            tau_head_low=best.get("tau_head_low", 0.70),
            tau_head_high=best.get("tau_head_high", 1.40),
            tau_peek=best.get("tau_peek", 0.90),
            persistence_frames=best.get("persistence_frames", 12),
        )
        self._filter = PersistenceFilter(
            min_frames=self._thresholds.persistence_frames,
            window_size=self._thresholds.window_size,
        )
        self._results = best
        return best

    def evaluate_with_persistence(
        self, verbose: bool = True,
    ) -> dict:
        """
        Evaluate classification WITH persistence filtering applied.

        Simulates a temporal sequence and measures how many momentary
        false positives are eliminated by the persistence filter.
        """
        if self._features is None or self._thresholds is None:
            self.run_grid_search(verbose=False)

        filter_obj = PersistenceFilter(
            min_frames=self._thresholds.persistence_frames,
            window_size=self._thresholds.window_size,
        )

        calibrator = ThresholdCalibrator()
        raw_preds = calibrator._classify_batch(self._features, self._thresholds)

        # Simulate temporal sequences (group by chunks of window_size)
        window = self._thresholds.window_size
        n = len(raw_preds)

        fp_before = 0
        fp_after = 0
        tp_before = 0
        tp_after = 0

        for start in range(0, n, window):
            chunk_true = self._labels[start:start + window]
            chunk_pred = raw_preds[start:start + window]

            for i, (true, pred) in enumerate(zip(chunk_true, chunk_pred)):
                is_anomaly = pred > 0
                filtered = filter_obj.update(i, is_anomaly)

                if true == 0 and is_anomaly:
                    fp_before += 1
                if true == 0 and filtered:
                    fp_after += 1
                if true > 0 and is_anomaly:
                    tp_before += 1
                if true > 0 and filtered:
                    tp_after += 1

        persistence_stats = {
            "raw_false_positives": fp_before,
            "filtered_false_positives": fp_after,
            "fp_reduction_pct": round(
                (1 - fp_after / max(fp_before, 1)) * 100, 1
            ),
            "raw_true_positives": tp_before,
            "filtered_true_positives": tp_after,
            "tp_preservation_pct": round(
                tp_after / max(tp_before, 1) * 100, 1
            ),
            "persistence_frames": self._thresholds.persistence_frames,
            "window_size": self._thresholds.window_size,
        }

        self._results["persistence_evaluation"] = persistence_stats

        if verbose:
            print(f"\n  Persistence Filter Evaluation:")
            print(f"    Raw FP:           {fp_before}")
            print(f"    Filtered FP:      {fp_after}")
            print(f"    FP reduction:     {persistence_stats['fp_reduction_pct']}%")
            print(f"    Raw TP:           {tp_before}")
            print(f"    Filtered TP:      {tp_after}")
            print(f"    TP preservation:  {persistence_stats['tp_preservation_pct']}%")

        return persistence_stats

    def print_report(self) -> None:
        """Print the full calibration report."""
        print(f"\n{'='*60}")
        print(f"  CALIBRATION REPORT")
        print(f"{'='*60}")

        if self._thresholds:
            print(f"\n  Optimal Thresholds:")
            t = self._thresholds.to_dict()
            for k, v in t.items():
                print(f"    {k:25s}: {v}")

        if "precision" in self._results:
            print(f"\n  Classification Metrics:")
            for key in ["precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"]:
                if key in self._results:
                    val = self._results[key]
                    if isinstance(val, float):
                        print(f"    {key:25s}: {val:.4f}")
                    else:
                        print(f"    {key:25s}: {val}")

        if "persistence_evaluation" in self._results:
            print(f"\n  Persistence Filter Impact:")
            for k, v in self._results["persistence_evaluation"].items():
                if isinstance(v, float):
                    print(f"    {k:25s}: {v:.1f}")
                else:
                    print(f"    {k:25s}: {v}")

        print(f"\n{'='*60}\n")

    def export_config(self, path: str) -> None:
        """Export calibrated thresholds to JSON."""
        config = {
            "thresholds": self._thresholds.to_dict() if self._thresholds else {},
            "results": {k: v for k, v in self._results.items()
                       if not isinstance(v, np.ndarray)},
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Config exported to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 9 Threshold Calibration")
    parser.add_argument(
        "--mode", choices=["grid-search", "evaluate", "report"],
        default="grid-search",
        help="Calibration mode",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to labelled dataset")
    parser.add_argument("--samples", type=int, default=2000, help="Synthetic samples")
    parser.add_argument("--output", type=str, default=None, help="Export config path")
    args = parser.parse_args()

    runner = CalibrationRunner(data_path=args.data)

    if args.mode == "grid-search":
        runner.load_data()
        runner.run_grid_search(verbose=True)
        runner.evaluate_with_persistence(verbose=True)
        runner.print_report()
    elif args.mode == "evaluate":
        runner.load_data()
        runner.run_grid_search(verbose=False)
        runner.evaluate_with_persistence(verbose=True)
        runner.print_report()
    elif args.mode == "report":
        runner.load_data()
        runner.run_grid_search(verbose=False)
        runner.evaluate_with_persistence(verbose=False)
        runner.print_report()

    if args.output:
        runner.export_config(args.output)


if __name__ == "__main__":
    main()
