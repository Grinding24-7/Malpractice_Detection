"""
ablation_study.py — Week 11: Comprehensive Ablation Matrix.

Executes automated parameter sweeps across:

    Sweep 1: Sequence Buffer Window Size (T)
        Train and evaluate models on sequence lengths T ∈ {10, 20, 30, 45, 60}.
        Measures the trade-off between F1-score accuracy and sliding-window
        memory footprint.

    Sweep 2: Kinematic Feature Engineering Layers
        Config A: Raw keypoint coordinates (x, y).
        Config B: Bounding-box normalised coordinates (x_hat, y_hat).
        Config C: Normalized coordinates + 1st-order velocity vectors (Δp)
                  + joint angular displacements (θ).

Usage:
    python ablation_study.py
    python ablation_study.py --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_training import (
    DEFAULT_CLASS_NAMES,
    MalpracticeGRU,
    _class_weight_tensor,
    evaluate_classifier,
    format_classification_report,
    get_device,
    make_dataloaders,
    pool_sequence_features,
    set_seed,
    train_baseline,
    train_sequence_model,
)

logger = logging.getLogger("ablation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "ablation_results"
DEFAULT_INPUT_DIM = 117
DEFAULT_NUM_CLASSES = 4

SEQUENCE_LENGTHS = [10, 20, 30, 45, 60]
FEATURE_CONFIGS = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Feature configuration definitions
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Defines a kinematic feature engineering configuration."""
    name: str
    description: str = ""
    raw_coords: bool = True
    bbox_normalized: bool = False
    velocities: bool = False
    angular_displacements: bool = False
    output_dim: int = 34  # base: 17 keypoints * 2

    def __post_init__(self) -> None:
        if self.name == "A":
            self.description = "Raw keypoint coordinates (x, y)"
            self.raw_coords = True
            self.output_dim = 34
        elif self.name == "B":
            self.description = "Bounding-box normalised coordinates"
            self.raw_coords = False
            self.bbox_normalized = True
            self.output_dim = 34
        elif self.name == "C":
            self.description = "Normalised coords + velocity + angular displacement"
            self.raw_coords = False
            self.bbox_normalized = True
            self.velocities = True
            self.angular_displacements = True
            self.output_dim = 117  # full TemporalFeatureExtractor output


FEATURE_CONFIG_MAP = {
    "A": FeatureConfig(name="A"),
    "B": FeatureConfig(name="B"),
    "C": FeatureConfig(name="C"),
}


# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------

def _generate_corpus(
    n_per_class: int = 120,
    window_size: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 4-class synthetic (N, T, F) corpus."""
    from temporal_features import FEATURE_DIM, TemporalFeatureExtractor, make_synthetic_sequence

    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    labels: list[int] = []

    for label, behavior in enumerate(["normal", "head_turn", "hand_reach", "peeking"]):
        for i in range(n_per_class):
            if behavior == "peeking":
                from temporal_training import _make_peeking_sequence
                seq = _make_peeking_sequence(window_size, seed=int(rng.integers(0, 10_000)))
            else:
                seq = make_synthetic_sequence(
                    behavior, window_size, seed=int(rng.integers(0, 10_000))
                )
            windows.append(seq)
            labels.append(label)

    extractor = TemporalFeatureExtractor(window_size=window_size)
    X = extractor.extract_batch(np.stack(windows))
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def _extract_features_for_config(
    windows: list[np.ndarray],
    config: FeatureConfig,
    window_size: int,
) -> np.ndarray:
    """
    Extract features from raw keypoint windows according to a FeatureConfig.

    Args:
        windows: list of (T, 17, 2) normalised keypoint windows.
        config: feature engineering configuration.
        window_size: temporal window length T.

    Returns:
        (B, T, F) float32 feature tensor.
    """
    from temporal_features import TemporalFeatureExtractor

    extractor = TemporalFeatureExtractor(window_size=window_size)

    if config.name == "A":
        # Raw keypoints only: (B, T, 34)
        arr = np.stack(windows).astype(np.float32)  # (B, T, 17, 2)
        return arr.reshape(arr.shape[0], arr.shape[1], -1)  # (B, T, 34)

    elif config.name == "B":
        # Bounding-box normalised coordinates
        arr = np.stack(windows).astype(np.float32)  # (B, T, 17, 2)
        # Normalise per-frame to [0, 1] using the bounding box of all keypoints
        for t in range(arr.shape[1]):
            frame = arr[:, t, :, :]  # (B, 17, 2)
            xy_min = frame.min(axis=1, keepdims=True)  # (B, 1, 2)
            xy_max = frame.max(axis=1, keepdims=True)  # (B, 1, 2)
            range_xy = xy_max - xy_min
            range_xy = np.where(range_xy < 1e-6, 1.0, range_xy)
            arr[:, t, :, :] = (frame - xy_min) / range_xy
        return arr.reshape(arr.shape[0], arr.shape[1], -1)  # (B, T, 34)

    elif config.name == "C":
        # Full TemporalFeatureExtractor output (keypoints + vel + acc + angles)
        return extractor.extract_batch(np.stack(windows))  # (B, T, 117)

    raise ValueError(f"Unknown config: {config.name}")


# ---------------------------------------------------------------------------
# Ablation result
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    config_name: str
    sequence_length: int
    feature_dim: int
    accuracy: float
    macro_f1: float
    per_class_f1: list[float]
    memory_bytes: int
    train_time_s: float
    model_type: str  # "gru" or "xgb"


# ---------------------------------------------------------------------------
# Run one ablation point
# ---------------------------------------------------------------------------

def _run_single_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_dim: int,
    sequence_length: int,
    config_name: str,
    model_type: str,
    device: torch.device,
    epochs: int = 20,
    seed: int = 0,
) -> AblationResult:
    """Train and evaluate one model at a single ablation configuration."""
    set_seed(seed)
    class_names = list(DEFAULT_CLASS_NAMES)

    t0 = time.monotonic()

    if model_type == "gru":
        model = MalpracticeGRU(
            input_dim=feature_dim, hidden_dim=128, num_layers=2,
            num_classes=DEFAULT_NUM_CLASSES, dropout=0.3,
            bidirectional=True, use_attention=True,
        )
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed,
        )
        train_loader, val_loader, class_weights = make_dataloaders(
            X_tr, y_tr, X_va, y_va, batch_size=32, class_weighted=True,
        )
        train_sequence_model(
            model, train_loader, val_loader,
            class_weights=class_weights, class_names=class_names,
            epochs=epochs, lr=1e-3, patience=6,
            checkpoint_path=Path("/tmp") / f"ablation_{config_name}_{sequence_length}.pth",
            device=device, seed=seed,
        )

        # Evaluate
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(y_test).long(),
            ),
            batch_size=64, shuffle=False,
        )
        model.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb.to(device)).argmax(dim=1).cpu().tolist()
                y_pred_all.extend(preds)
                y_true_all.extend(yb.tolist())
        y_true_np = np.asarray(y_true_all, dtype=np.int64)
        y_pred_np = np.asarray(y_pred_all, dtype=np.int64)
        metrics = evaluate_classifier(y_true_np, y_pred_np, class_names=class_names)

        # Memory footprint estimate
        n_params = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_bytes = n_params

    else:  # xgb
        X_tr_pooled = pool_sequence_features(X_train)
        X_te_pooled = pool_sequence_features(X_test)
        model, metrics = train_baseline(
            X_tr_pooled, y_train, X_te_pooled, y_test,
            kind="xgb", class_names=class_names, n_estimators=200,
        )
        y_pred_np = model.predict(X_te_pooled)
        metrics = evaluate_classifier(y_test, y_pred_np, class_names=class_names)
        memory_bytes = 0  # sklearn models not easily measured this way

    train_time = time.monotonic() - t0

    per_class_f1 = [row["f1"] for row in metrics["per_class"]]

    return AblationResult(
        config_name=config_name,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        per_class_f1=per_class_f1,
        memory_bytes=memory_bytes,
        train_time_s=train_time,
        model_type=model_type,
    )


# ---------------------------------------------------------------------------
# Sweep 1: Sequence length
# ---------------------------------------------------------------------------

def run_sequence_length_sweep(
    X_full: np.ndarray,
    y_full: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    seed: int = 0,
) -> list[AblationResult]:
    """
    Sweep sequence buffer window size T ∈ {10, 20, 30, 45, 60}.

    For each T, re-generates the corpus with the given window size,
    extracts full features (Config C), and trains a Bi-LSTM+Attention.
    """
    results: list[AblationResult] = []
    feature_dim = DEFAULT_INPUT_DIM  # Config C always uses full features

    for T in SEQUENCE_LENGTHS:
        logger.info("Sweep 1: T=%d (feature_dim=%d)", T, feature_dim)
        X, y = _generate_corpus(n_per_class=100, window_size=T, seed=seed)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed,
        )

        result = _run_single_ablation(
            X_train, y_train, X_test, y_test,
            feature_dim=feature_dim, sequence_length=T,
            config_name="C", model_type="gru",
            device=device, epochs=epochs, seed=seed,
        )
        results.append(result)
        logger.info("  T=%d -> F1=%.4f acc=%.4f train=%.1fs",
                     T, result.macro_f1, result.accuracy, result.train_time_s)

    return results


# ---------------------------------------------------------------------------
# Sweep 2: Feature engineering layers
# ---------------------------------------------------------------------------

def run_feature_sweep(
    n_per_class: int = 120,
    window_size: int = 30,
    epochs: int = 20,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> list[AblationResult]:
    """
    Sweep kinematic feature engineering configs A, B, C at fixed T=30.

    Config A: raw keypoints (F=34)
    Config B: bbox-normalised keypoints (F=34)
    Config C: full features (F=117)
    """
    if device is None:
        device = get_device()
    results: list[AblationResult] = []

    # Generate raw windows once
    from temporal_features import make_synthetic_sequence
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    labels: list[int] = []
    for label, behavior in enumerate(["normal", "head_turn", "hand_reach", "peeking"]):
        for i in range(n_per_class):
            if behavior == "peeking":
                from temporal_training import _make_peeking_sequence
                seq = _make_peeking_sequence(window_size, seed=int(rng.integers(0, 10_000)))
            else:
                seq = make_synthetic_sequence(
                    behavior, window_size, seed=int(rng.integers(0, 10_000))
                )
            windows.append(seq)
            labels.append(label)

    y = np.asarray(labels, dtype=np.int64)

    for config_name in FEATURE_CONFIGS:
        config = FEATURE_CONFIG_MAP[config_name]
        logger.info("Sweep 2: Config %s — %s", config_name, config.description)

        X = _extract_features_for_config(windows, config, window_size)
        logger.info("  Extracted X=%s for config %s", X.shape, config_name)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed,
        )

        result = _run_single_ablation(
            X_train, y_train, X_test, y_test,
            feature_dim=config.output_dim, sequence_length=window_size,
            config_name=config_name, model_type="gru",
            device=device, epochs=epochs, seed=seed,
        )
        results.append(result)
        logger.info("  Config %s -> F1=%.4f acc=%.4f F=%d train=%.1fs",
                     config_name, result.macro_f1, result.accuracy,
                     config.output_dim, result.train_time_s)

    return results


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_sweep1_report(results: list[AblationResult]) -> str:
    """Format the sequence length sweep results."""
    lines = [
        "=" * 80,
        "  SWEEP 1: SEQUENCE BUFFER WINDOW SIZE (T)",
        "  Bi-LSTM + Attention | Config C (full features)",
        "=" * 80,
        "",
        f"  {'T':>6}  {'F':>6}  {'Acc':>7}  {'Macro F1':>9}  "
        f"{'Memory (B)':>12}  {'Train (s)':>10}  {'F1 Δ vs T=30':>14}",
        "  " + "-" * 70,
    ]

    base_f1 = None
    for r in results:
        if r.sequence_length == 30:
            base_f1 = r.macro_f1
            break
    if base_f1 is None:
        base_f1 = results[len(results) // 2].macro_f1 if results else 0.0

    for r in results:
        delta = r.macro_f1 - base_f1 if base_f1 else 0.0
        delta_str = f"{delta:+.4f}" if r.sequence_length != 30 else "  (base)"
        lines.append(
            f"  {r.sequence_length:>6d}  {r.feature_dim:>6d}  "
            f"{r.accuracy:>7.4f}  {r.macro_f1:>9.4f}  "
            f"{r.memory_bytes:>12,d}  {r.train_time_s:>10.1f}  "
            f"{delta_str:>14}"
        )

    lines.extend([
        "",
        "  Memory footprint = raw parameter bytes (hidden=128, 2-layer Bi-GRU).",
        "",
        "=" * 80,
    ])
    return "\n".join(lines)


def format_sweep2_report(results: list[AblationResult]) -> str:
    """Format the feature engineering sweep results."""
    lines = [
        "=" * 80,
        "  SWEEP 2: KINEMATIC FEATURE ENGINEERING LAYERS",
        "  Bi-LSTM + Attention | T=30",
        "=" * 80,
        "",
        f"  {'Config':>8}  {'Description':<45}  {'F':>5}  {'Acc':>7}  {'Macro F1':>9}",
        "  " + "-" * 80,
    ]

    descs = {
        "A": "Raw keypoint coordinates (x, y)",
        "B": "Bounding-box normalised coordinates",
        "C": "Normalised + velocity + angular displacement",
    }

    for r in results:
        lines.append(
            f"  {r.config_name:>8}  {descs.get(r.config_name, ''):<45}  "
            f"{r.feature_dim:>5d}  {r.accuracy:>7.4f}  {r.macro_f1:>9.4f}"
        )

    # Delta analysis
    if len(results) >= 2:
        lines.extend(["", "  F1 improvement over Config A (raw coords):"])
        base = results[0].macro_f1
        for r in results[1:]:
            delta = r.macro_f1 - base
            lines.append(f"    {r.config_name} vs A: {delta:+.4f}")

    lines.extend(["", "=" * 80])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_ablation_results(
    sweep1_results: list[AblationResult],
    sweep2_results: list[AblationResult],
    output_dir: Path,
) -> None:
    """Generate ablation comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sweep 1: F1 vs T
    ax = axes[0]
    Ts = [r.sequence_length for r in sweep1_results]
    f1s = [r.macro_f1 for r in sweep1_results]
    accs = [r.accuracy for r in sweep1_results]
    ax.plot(Ts, f1s, "o-", label="Macro F1", linewidth=2, markersize=8)
    ax.plot(Ts, accs, "s--", label="Accuracy", linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length (T)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Sweep 1: F1 vs Sequence Length", fontsize=12, fontweight="bold")
    ax.set_xticks(Ts)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    # Sweep 2: Config comparison
    ax = axes[1]
    configs = [r.config_name for r in sweep2_results]
    f1s2 = [r.macro_f1 for r in sweep2_results]
    colors = sns.color_palette("Set2", len(configs))
    bars = ax.bar(configs, f1s2, color=colors, edgecolor="black", linewidth=0.8)
    for bar, f1 in zip(bars, f1s2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f1:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Feature Config", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Sweep 2: F1 vs Feature Layers", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Week 11 Ablation Study", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / "ablation_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation plots -> %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 11 Ablation Study")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Sweep 1: Sequence length ---
    logger.info("=" * 60)
    logger.info("SWEEP 1: Sequence Buffer Window Size")
    logger.info("=" * 60)
    sweep1 = run_sequence_length_sweep(
        X_full=np.empty(0), y_full=np.empty(0),  # unused; corpus regenerated per T
        device=device, epochs=args.epochs, seed=args.seed,
    )
    report1 = format_sweep1_report(sweep1)
    print(report1)

    # --- Sweep 2: Feature engineering ---
    logger.info("=" * 60)
    logger.info("SWEEP 2: Kinematic Feature Engineering Layers")
    logger.info("=" * 60)
    sweep2 = run_feature_sweep(
        n_per_class=args.n_per_class, window_size=30,
        device=device, epochs=args.epochs, seed=args.seed,
    )
    report2 = format_sweep2_report(sweep2)
    print(report2)

    # --- Plots ---
    _plot_ablation_results(sweep1, sweep2, output_dir)

    # --- Combined JSON ---
    json_report = {
        "sweep1_sequence_length": [
            {
                "T": r.sequence_length,
                "F": r.feature_dim,
                "accuracy": round(r.accuracy, 4),
                "macro_f1": round(r.macro_f1, 4),
                "per_class_f1": [round(f, 4) for f in r.per_class_f1],
                "memory_bytes": r.memory_bytes,
                "train_time_s": round(r.train_time_s, 3),
            }
            for r in sweep1
        ],
        "sweep2_feature_config": [
            {
                "config": r.config_name,
                "F": r.feature_dim,
                "accuracy": round(r.accuracy, 4),
                "macro_f1": round(r.macro_f1, 4),
                "per_class_f1": [round(f, 4) for f in r.per_class_f1],
                "train_time_s": round(r.train_time_s, 3),
            }
            for r in sweep2
        ],
    }
    json_path = output_dir / "ablation_report.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    logger.info("JSON report saved -> %s", json_path)


if __name__ == "__main__":
    main()
