"""
evaluate.py — Week 11: Automated Model Evaluation Engine.

Loads test-set tensors (N, T, F) and runs predictions through:
    - XGBoost Baseline (pooled features)
    - Random Forest Baseline (pooled features)
    - PyTorch LSTM (Unidirectional)
    - PyTorch Bi-LSTM with Attention

Computes and logs per-class Precision, Recall, F1-Score, Support, and
False Positive Rate (FPR).  Exports confusion_matrix.png and
precision_recall_curves.png via Seaborn/Matplotlib.

Usage:
    python evaluate.py
    python evaluate.py --dataset path/to/corpus.pt --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_training import (
    DEFAULT_CLASS_NAMES,
    MalpracticeLSTM,
    MalpracticeGRU,
    _MalpracticeRecurrent,
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

logger = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "eval_results"
DEFAULT_INPUT_DIM = 117
DEFAULT_NUM_CLASSES = 4
DEFAULT_SEQUENCE_LEN = 30


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback when no dataset file exists)
# ---------------------------------------------------------------------------

def _generate_synthetic_corpus(
    n_per_class: int = 100,
    window_size: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 4-class synthetic (N, T, F) corpus for evaluation.

    Returns:
        X: (N, T, F) float32 feature tensor.
        y: (N,) int64 class labels.
    """
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


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_xgboost() -> object:
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
        random_state=0, n_jobs=-1,
    )


def _build_random_forest() -> object:
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300, random_state=0, n_jobs=-1,
    )


def _build_lstm(input_dim: int, num_classes: int) -> MalpracticeLSTM:
    return MalpracticeLSTM(
        input_dim=input_dim, hidden_dim=128, num_layers=2,
        num_classes=num_classes, dropout=0.3, bidirectional=False,
        use_attention=False,
    )


def _build_bilstm_attention(input_dim: int, num_classes: int) -> MalpracticeGRU:
    """Bi-LSTM with temporal attention (uses GRU backbone for parity)."""
    return MalpracticeGRU(
        input_dim=input_dim, hidden_dim=128, num_layers=2,
        num_classes=num_classes, dropout=0.3, bidirectional=True,
        use_attention=True,
    )


# ---------------------------------------------------------------------------
# FPR computation
# ---------------------------------------------------------------------------

def _compute_fpr_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    False Positive Rate per class: FP / (FP + TN).

    Args:
        y_true: (N,) ground-truth labels.
        y_pred: (N,) predicted labels.
        n_classes: number of classes.

    Returns:
        (C,) FPR per class.
    """
    fpr = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        fp = int(((y_pred == c) & (y_true != c)).sum())
        tn = int(((y_pred != c) & (y_true != c)).sum())
        denom = fp + tn
        fpr[c] = fp / denom if denom > 0 else 0.0
    return fpr


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    metrics: dict
    fpr_per_class: np.ndarray
    train_time_s: float
    predict_time_s: float


def _evaluate_baseline(
    name: str,
    X_train_pooled: np.ndarray,
    y_train: np.ndarray,
    X_test_pooled: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    class_names: list[str],
) -> ModelResult:
    """Train and evaluate a sklearn/XGBoost baseline."""
    t0 = time.monotonic()
    model, metrics = train_baseline(
        X_train_pooled, y_train, X_test_pooled, y_test,
        kind=kind, class_names=class_names, n_estimators=300,
    )
    train_time = time.monotonic() - t0

    t1 = time.monotonic()
    y_pred = model.predict(X_test_pooled)
    predict_time = time.monotonic() - t1

    fpr = _compute_fpr_per_class(y_test, y_pred, len(class_names))
    return ModelResult(
        name=name, metrics=metrics, fpr_per_class=fpr,
        train_time_s=train_time, predict_time_s=predict_time,
    )


def _evaluate_deep(
    name: str,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    device: torch.device,
    epochs: int = 25,
) -> ModelResult:
    """Train and evaluate a PyTorch sequence classifier."""
    from sklearn.model_selection import train_test_split

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=0,
    )
    train_loader, val_loader, class_weights = make_dataloaders(
        X_tr, y_tr, X_va, y_va, batch_size=32, class_weighted=True,
    )

    t0 = time.monotonic()
    history, best_f1 = train_sequence_model(
        model, train_loader, val_loader,
        class_weights=class_weights, class_names=class_names,
        epochs=epochs, lr=1e-3, patience=8,
        checkpoint_path=Path("/tmp") / f"eval_{name}.pth",
        device=device, seed=0,
    )
    train_time = time.monotonic() - t0

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).long(),
        ),
        batch_size=64, shuffle=False,
    )

    model.eval()
    t1 = time.monotonic()
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().tolist()
            y_pred_all.extend(preds)
            y_true_all.extend(yb.tolist())
    predict_time = time.monotonic() - t1

    y_true_np = np.asarray(y_true_all, dtype=np.int64)
    y_pred_np = np.asarray(y_pred_all, dtype=np.int64)
    metrics = evaluate_classifier(y_true_np, y_pred_np, class_names=class_names)
    fpr = _compute_fpr_per_class(y_true_np, y_pred_np, len(class_names))

    return ModelResult(
        name=name, metrics=metrics, fpr_per_class=fpr,
        train_time_s=train_time, predict_time_s=predict_time,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_confusion_matrices(
    results: list[ModelResult],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Generate a grid of confusion matrix heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res.metrics["confusion_matrix"]
        cm_pct = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            cm_pct, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, vmin=0, vmax=1, cbar=False,
        )
        ax.set_title(res.name, fontsize=11, fontweight="bold")
        ax.set_ylabel("True" if ax == axes[0] else "")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices (Normalised)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrices -> %s", out)


def _plot_precision_recall_curves(
    results: list[ModelResult],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Generate per-class precision-recall bar chart comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_classes = len(class_names)
    n_models = len(results)
    metrics_data = []

    for res in results:
        for row in res.metrics["per_class"]:
            metrics_data.append({
                "Model": res.name,
                "Class": row["class"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
            })

    import pandas as pd
    df = pd.DataFrame(metrics_data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["Precision", "Recall", "F1"]):
        sns.barplot(data=df, x="Class", y=metric, hue="Model", ax=ax, palette="Set2")
        ax.set_ylim(0, 1.05)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        if ax != axes[0]:
            ax.legend().remove()

    fig.suptitle("Per-Class Metric Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / "precision_recall_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curves -> %s", out)


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_full_report(results: list[ModelResult], class_names: list[str]) -> str:
    """Render a comprehensive markdown-style evaluation report."""
    lines = [
        "=" * 90,
        "  WEEK 11 — MODEL EVALUATION ENGINE RESULTS",
        "=" * 90,
        "",
    ]

    # Summary table
    lines.append(f"{'Model':<28} {'Acc':>7} {'Macro F1':>9} {'Train (s)':>10} {'Infer (s)':>10}")
    lines.append("-" * 70)
    for res in results:
        lines.append(
            f"{res.name:<28} "
            f"{res.metrics['accuracy']:>7.4f} "
            f"{res.metrics['macro_f1']:>9.4f} "
            f"{res.train_time_s:>10.2f} "
            f"{res.predict_time_s:>10.4f}"
        )

    lines.append("")

    # Per-class FPR table
    lines.append("False Positive Rate (FPR) per class:")
    header = f"{'Model':<28}" + "".join(f"{n[:12]:>14}" for n in class_names)
    lines.append(header)
    lines.append("-" * (28 + 14 * len(class_names)))
    for res in results:
        row = f"{res.name:<28}" + "".join(f"{v:>14.4f}" for v in res.fpr_per_class)
        lines.append(row)

    lines.append("")

    # Detailed per-model reports
    for res in results:
        lines.append(f"\n--- {res.name} ---")
        lines.append(format_classification_report(res.metrics))

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 11 Model Evaluation Engine")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .pt dataset")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--sequence-len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(DEFAULT_CLASS_NAMES)

    # --- Load or generate data ---
    if args.dataset and Path(args.dataset).exists():
        from temporal_features import load_dataset
        data = load_dataset(args.dataset)
        X = data["X"].numpy() if isinstance(data["X"], torch.Tensor) else np.asarray(data["X"])
        y = data["labels"].numpy() if isinstance(data["labels"], torch.Tensor) else np.asarray(data["labels"])
        y = y.astype(np.int64)
    else:
        logger.info("No dataset found — generating synthetic corpus (%d per class)", args.n_per_class)
        X, y = _generate_synthetic_corpus(
            n_per_class=args.n_per_class, window_size=args.sequence_len, seed=args.seed,
        )

    logger.info("Dataset: X=%s y=%s classes=%s", X.shape, y.shape, class_names)

    # --- Train/test split ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed,
    )
    logger.info("Split: train=%d test=%d", len(X_train), len(X_test))

    # --- Pool features for baselines ---
    X_train_pooled = pool_sequence_features(X_train)
    X_test_pooled = pool_sequence_features(X_test)
    logger.info("Pooled features: train=%s test=%s", X_train_pooled.shape, X_test_pooled.shape)

    # --- Evaluate all models ---
    results: list[ModelResult] = []

    # 1. XGBoost
    logger.info("[1/4] Training XGBoost baseline...")
    results.append(_evaluate_baseline(
        "XGBoost", X_train_pooled, y_train, X_test_pooled, y_test, "xgb", class_names,
    ))

    # 2. Random Forest
    logger.info("[2/4] Training Random Forest baseline...")
    results.append(_evaluate_baseline(
        "Random Forest", X_train_pooled, y_train, X_test_pooled, y_test, "rf", class_names,
    ))

    # 3. Unidirectional LSTM
    logger.info("[3/4] Training LSTM (unidirectional)...")
    lstm = _build_lstm(X.shape[-1], len(class_names))
    results.append(_evaluate_deep(
        "LSTM (Uni)", lstm, X_train, y_train, X_test, y_test, class_names, device, args.epochs,
    ))

    # 4. Bi-LSTM with Attention
    logger.info("[4/4] Training Bi-LSTM + Attention...")
    bilstm = _build_bilstm_attention(X.shape[-1], len(class_names))
    results.append(_evaluate_deep(
        "Bi-LSTM+Attn", bilstm, X_train, y_train, X_test, y_test, class_names, device, args.epochs,
    ))

    # --- Generate plots ---
    logger.info("Generating visual plots...")
    _plot_confusion_matrices(results, class_names, output_dir)
    _plot_precision_recall_curves(results, class_names, output_dir)

    # --- Print report ---
    report = format_full_report(results, class_names)
    print(report)

    # --- Save JSON ---
    json_report = {
        "models": [
            {
                "name": r.name,
                "accuracy": r.metrics["accuracy"],
                "macro_f1": r.metrics["macro_f1"],
                "per_class": r.metrics["per_class"],
                "fpr_per_class": r.fpr_per_class.tolist(),
                "train_time_s": round(r.train_time_s, 3),
                "predict_time_s": round(r.predict_time_s, 4),
            }
            for r in results
        ],
        "class_names": class_names,
    }
    json_path = output_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    logger.info("JSON report saved -> %s", json_path)


if __name__ == "__main__":
    main()
