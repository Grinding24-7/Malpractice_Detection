"""
temporal_training.py — Week 5: sequence model training & malpractice classification.

Consumes the (B, T, F) spatial-temporal feature tensors produced in Week 4
(backend/temporal_features.py) and trains / compares a suite of classifiers:

    Baseline models (fast benchmarking)
        XGBoost + RandomForest trained on temporally POOLED features
        (mean / max / std across the T axis), with per-class precision,
        recall, F1 and confusion-matrix reporting.

    Deep sequence classifiers
        ``MalpracticeLSTM`` / ``MalpracticeGRU`` — 2-layer bidirectional
        recurrent backbones (dropout=0.3, hidden_dim=128) with an optional
        temporal-attention block and a BatchNorm+ReLU+Linear classification
        head, trained with CrossEntropyLoss (class-weighted), AdamW,
        ReduceLROnPlateau and early stopping on validation macro-F1.

Tensor conventions
------------------
    X (features):  (N, T, F) float32 — N samples, T = 30 frames, F = Week 4
                   per-frame feature dimension (keypoints + velocities + angles).
    Y (labels):    (N,) int64 class indices in {0, ..., C-1}.
    Pooled X:      (N, F * n_modes) float32 — for the sklearn baselines.

Module layout
-------------
    ExamDataset / make_dataloaders        — ingestion + class weighting
    pool_sequence_features                — temporal aggregation for baselines
    train_baseline / evaluate_classifier  — sklearn/XGBoost benchmarking
    TemporalAttention / ClassificationHead / MalpracticeLSTM / MalpracticeGRU
    train_sequence_model / evaluate_sequence_model — deep training engine
    __main__                              — synthetic end-to-end smoke test
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "ExamDataset",
    "make_dataloaders",
    "pool_sequence_features",
    "train_baseline",
    "evaluate_classifier",
    "format_classification_report",
    "TemporalAttention",
    "ClassificationHead",
    "MalpracticeLSTM",
    "MalpracticeGRU",
    "train_sequence_model",
    "evaluate_sequence_model",
    "save_checkpoint",
    "load_checkpoint",
    "get_device",
    "set_seed",
]

DEFAULT_CLASS_NAMES: list[str] = [
    "Normal",
    "Head Turning",
    "Note Passing",
    "Peeking",
]

CHECKPOINT_PATH: str = "best_malpractice_model.pth"


# ---------------------------------------------------------------------------
# Device / seeding helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Auto-detect the fastest available compute device.

    Returns:
        torch.device("cuda") if a GPU is present, else torch.device("mps")
        on Apple Silicon, else torch.device("cpu").
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. Dataset loading & preprocessing
# ---------------------------------------------------------------------------

class ExamDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset over saved (N, T, F) temporal feature tensors.

    Ingests the artefacts written by Week 4's ``save_dataset`` (.pt or .npz),
    which contain::

        X:       (N, T, F) float32
        labels:  (N,) int64     (optional)
        summary: (N, S) float32 (optional)

    Samples returned by ``__getitem__`` are ``(X_i, y_i)`` when labels are
    present, otherwise just ``(X_i,)``.

    Args:
        X: (N, T, F) tensor or array of sequence features.
        y: optional (N,) integer class labels.
        transform: optional per-sample callable (applied to X_i).
    """

    def __init__(
        self,
        X,
        y: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
    ) -> None:
        if isinstance(X, torch.Tensor):
            self.X = X.float().detach().cpu()
        else:
            self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y: Optional[torch.Tensor] = None
        if y is not None:
            self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.transform = transform
        self.class_names: list[str] = list(DEFAULT_CLASS_NAMES)
        if self.X.ndim != 3:
            raise ValueError(
                f"X must have shape (N, T, F), got {tuple(self.X.shape)}"
            )

    @classmethod
    def from_saved(cls, path: str | Path) -> "ExamDataset":
        """
        Build a dataset from a Week 4 .pt / .npz artefact.

        Args:
            path: dataset file produced by temporal_features.save_dataset.

        Returns:
            ExamDataset populated with the saved features and labels.
        """
        from temporal_features import load_dataset

        data = load_dataset(str(path))
        labels = (
            data["labels"].numpy()
            if "labels" in data and data["labels"] is not None
            else None
        )
        return cls(X=data["X"].numpy(), y=labels)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.y is None:
            return (x,)
        return x, self.y[idx]


def _class_weight_tensor(y: np.ndarray) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss.

    Args:
        y: (N,) integer labels.

    Returns:
        (C,) float32 tensor where w_c = N / (C * count_c); classes absent
        from the sample receive weight 0.
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=0)
    n_classes = max(1, int(counts.shape[0]))
    total = float(counts.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(
            counts > 0, total / (n_classes * counts), 0.0
        )
    return torch.from_numpy(weights.astype(np.float32))


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    class_weighted: bool = True,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.Tensor]]:
    """
    Build train/validation DataLoaders with class-imbalance handling.

    The train loader uses a ``WeightedRandomSampler`` weighted by inverse
    class frequency so rare malpractice behaviours are re-sampled every epoch.
    Tensors are ``pin_memory``-pinned when running on CUDA.

    Args:
        X_train: (N_tr, T, F) feature tensor.
        y_train: (N_tr,) integer labels.
        X_val: (N_va, T, F) feature tensor.
        y_val: (N_va,) integer labels.
        batch_size: samples per batch.
        class_weighted: enable WeightedRandomSampler + class weights.
        num_workers: DataLoader worker processes.
        seed: RNG seed for the sampler.

    Returns:
        (train_loader, val_loader, class_weights) where class_weights is a
        (C,) float32 tensor (or None when class_weighted=False).
    """
    train_ds = ExamDataset(X=X_train, y=y_train)
    val_ds = ExamDataset(X=X_val, y=y_val)

    pin = torch.cuda.is_available()
    weights_tensor: Optional[torch.Tensor] = None
    sampler = None
    if class_weighted:
        weights_tensor = _class_weight_tensor(y_train)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights_tensor[y_train],
            num_samples=len(y_train),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, weights_tensor


# ---------------------------------------------------------------------------
# 2. Temporal feature pooling + baseline model suite
# ---------------------------------------------------------------------------

def pool_sequence_features(
    X, modes: tuple[str, ...] = ("mean", "max", "std")
) -> np.ndarray:
    """
    Aggregate a sequence tensor along the time axis for sklearn baselines.

    Args:
        X: (N, T, F) features — torch.Tensor or NumPy array.
        modes: statistics computed across T, e.g. ("mean", "max", "std").

    Returns:
        (N, F * len(modes)) float32 pooled summary vectors, one per sequence.
    """
    if isinstance(X, torch.Tensor):
        arr = X.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (N, T, F), got {arr.shape}")

    agg: dict[str, np.ndarray] = {
        "mean": np.mean(arr, axis=-2),
        "max": np.max(arr, axis=-2),
        "std": np.std(arr, axis=-2),
        "min": np.min(arr, axis=-2),
        "median": np.median(arr, axis=-2),
    }
    return np.concatenate(
        [agg[m] for m in modes], axis=-1
    ).astype(np.float32)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute classification metrics including a confusion matrix.

    Args:
        y_true: (N,) ground-truth integer labels.
        y_pred: (N,) predicted integer labels.
        class_names: optional display names per class index.

    Returns:
        dict with keys:
            accuracy, macro_f1, per_class ([{class, precision, recall, f1,
            support}] * C), confusion_matrix ((C, C) int array), class_names.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    names = class_names or DEFAULT_CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec, rec, f1s, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class = [
        {
            "class": names[i] if i < len(names) else f"class_{i}",
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for i, (p, r, f, s) in enumerate(zip(prec, rec, f1s, sup))
    ]
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": names,
    }


def format_classification_report(metrics: dict) -> str:
    """
    Render an evaluation dict into a readable console report.

    Args:
        metrics: output of :func:`evaluate_classifier`.

    Returns:
        Multi-line string with a per-class precision/recall/F1 table and the
        confusion matrix.
    """
    names = metrics["class_names"]
    lines = [
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Macro F1 : {metrics['macro_f1']:.4f}",
        "",
        f"{'Class':<18}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Supp':>8}",
    ]
    for row in metrics["per_class"]:
        lines.append(
            f"{row['class']:<18}{row['precision']:>8.3f}{row['recall']:>8.3f}"
            f"{row['f1']:>8.3f}{row['support']:>8d}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows=truth, cols=pred):")
    header = " " * 20 + "".join(f"{n[:8]:>9}" for n in names)
    lines.append(header)
    cm = metrics["confusion_matrix"]
    for i, row in enumerate(cm):
        label = names[i][:18] if i < len(names) else f"class_{i}"
        lines.append(f"{label:<20}" + "".join(f"{v:>9d}" for v in row))
    return "\n".join(lines)


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kind: str = "xgb",
    class_names: Optional[list[str]] = None,
    random_state: int = 0,
    **kwargs,
) -> tuple[object, dict]:
    """
    Train a fast sklearn / XGBoost baseline on pooled sequence features.

    Args:
        X_train: (N_tr, F_pooled) pooled training features.
        y_train: (N_tr,) training labels.
        X_val: (N_va, F_pooled) pooled validation features.
        y_val: (N_va,) validation labels.
        kind: "xgb" (XGBClassifier) or "rf" (RandomForestClassifier).
        class_names: display names for metrics.
        random_state: reproducibility seed.
        **kwargs: extra hyper-parameters forwarded to the classifier.

    Returns:
        (fitted_model, metrics) where metrics comes from
        :func:`evaluate_classifier` computed on the validation split.
    """
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=kwargs.pop("n_estimators", 300),
            max_depth=kwargs.pop("max_depth", None),
            min_samples_leaf=kwargs.pop("min_samples_leaf", 1),
            random_state=random_state,
            n_jobs=kwargs.pop("n_jobs", -1),
            **kwargs,
        )
    elif kind == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=kwargs.pop("n_estimators", 300),
            max_depth=kwargs.pop("max_depth", 6),
            learning_rate=kwargs.pop("learning_rate", 0.1),
            subsample=kwargs.pop("subsample", 0.9),
            colsample_bytree=kwargs.pop("colsample_bytree", 0.9),
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=kwargs.pop("n_jobs", -1),
            **kwargs,
        )
    else:
        raise ValueError(f"unknown baseline kind: {kind!r} (use 'xgb' or 'rf')")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = evaluate_classifier(y_val, y_pred, class_names=class_names)
    return model, metrics


# ---------------------------------------------------------------------------
# 3. PyTorch deep sequence classifiers
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Scalar-additive temporal attention over recurrent sequence outputs.

    Learns a per-frame importance score and produces a weighted context
    vector, highlighting frames where anomalous behaviour peaks.

    Args:
        in_dim: feature dimension per frame, D = hidden * num_directions.

    Shapes:
        input:  (B, T, D)
        output: context (B, D), weights (B, T)
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.tanh(self.score(h)).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)          # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        return context, weights


class ClassificationHead(nn.Module):
    """
    BatchNorm + ReLU + Linear classification head.

    Args:
        in_dim: feature dimension coming out of the backbone (e.g. 256).
        hidden: width of the intermediate linear layer.
        num_classes: number of output classes C.
        dropout: dropout probability before the final linear layer.

    Shapes:
        input:  (B, in_dim)
        output: (B, num_classes) logits
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MalpracticeRecurrent(nn.Module):
    """
    Shared 2-layer bidirectional recurrent backbone + attention + head.

    Subclassed by :class:`MalpracticeLSTM` and :class:`MalpracticeGRU`.

    Args:
        cell: "lstm" or "gru" backbone type.
        input_dim: per-frame feature dimension F.
        hidden_dim: hidden units per direction (default 128).
        num_layers: stacked recurrent layers (default 2).
        num_classes: number of output classes C.
        dropout: dropout between stacked layers and in the head (default 0.3).
        bidirectional: process the window in both directions.
        use_attention: temporal-attention context instead of last-frame pool.

    Shapes:
        input:  (B, T, F)
        output: (B, C) class logits
    """

    def __init__(
        self,
        cell: str,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 to enable recurrent dropout")
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(self.feat_dim)
        self.head = ClassificationHead(
            in_dim=self.feat_dim,
            hidden=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (B, T, D)
        if self.use_attention:
            context, _ = self.attention(out)  # (B, D)
        else:
            context = out[:, -1, :]           # (B, D)
        return self.head(context)


class MalpracticeLSTM(_MalpracticeRecurrent):
    """2-layer bidirectional LSTM sequence classifier (see base class)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(cell="lstm", *args, **kwargs)


class MalpracticeGRU(_MalpracticeRecurrent):
    """2-layer bidirectional GRU sequence classifier (see base class)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(cell="gru", *args, **kwargs)


# ---------------------------------------------------------------------------
# 4. Training engine & evaluation pipeline
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    One full training pass over the train loader.

    Returns:
        (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)                     # (B, C)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum())
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def _validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    One full evaluation pass over the validation loader.

    Returns:
        (mean_loss, accuracy, y_true, y_pred).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            n += x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return (
        total_loss / max(n, 1),
        correct / max(n, 1),
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
    )


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    epoch: int,
    val_f1: float,
    class_names: Optional[list[str]] = None,
) -> Path:
    """
    Persist the best model state to disk.

    Args:
        model: the (best-state) model to snapshot.
        path: checkpoint destination (.pth).
        epoch: epoch at which the best state was reached.
        val_f1: macro-F1 used as the early-stopping criterion.
        class_names: optional class labels stored as metadata.

    Returns:
        Path of the written checkpoint.
    """
    path = Path(path)
    payload = {
        "arch": type(model).__name__,
        "epoch": int(epoch),
        "val_f1": float(val_f1),
        "input_dim": int(model.rnn.input_size),
        "num_classes": int(model.head.net[-1].out_features),
        "class_names": class_names or list(DEFAULT_CLASS_NAMES),
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, str(path))
    return path


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Restore a model's weights from a saved checkpoint.

    Args:
        model: model instance whose architecture matches the checkpoint.
        path: checkpoint file from :func:`save_checkpoint`.
        device: device to map weights onto (defaults to the best available).

    Returns:
        The raw checkpoint dict (metadata + state dict).
    """
    device = device or get_device()
    ckpt = torch.load(str(path), map_location=str(device), weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return ckpt


def train_sequence_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    class_names: Optional[list[str]] = None,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> tuple[dict, float]:
    """
    Train a deep sequence classifier with the full Week 5 pipeline.

    Optimisation setup:
        - loss:    nn.CrossEntropyLoss(weight=class_weights)
        - optim:   torch.optim.AdamW(lr, weight_decay)
        - sched:   torch.optim.lr_scheduler.ReduceLROnPlateau(mode="max")
        - stop:    early stopping on validation macro-F1 (patience epochs)
        - save:    best model state -> checkpoint_path

    Args:
        model: MalpracticeLSTM / MalpracticeGRU (or compatible nn.Module).
        train_loader: training DataLoader yielding (B, T, F) + (B,).
        val_loader: validation DataLoader.
        class_weights: (C,) float32 tensor for loss weighting, or None.
        class_names: class labels stored in the checkpoint metadata.
        epochs: maximum number of training epochs.
        lr: initial AdamW learning rate.
        weight_decay: AdamW L2 regularisation.
        patience: epochs without validation-F1 improvement before stopping.
        lr_patience: epochs without improvement before LR decay.
        lr_factor: LR multiplication factor on plateau.
        checkpoint_path: where the best model checkpoint is written.
        device: compute device (auto-detected when None).
        seed: RNG seed for reproducible training.

    Returns:
        (history, best_val_f1) where history holds per-epoch train/val loss,
        accuracy and macro-F1.
    """
    device = device or get_device()
    set_seed(seed)
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_f1 = -np.inf
    best_epoch = -1
    best_state: Optional[dict] = None
    no_improve = 0
    stop_epoch: Optional[int] = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, y_true, y_pred = _validate_epoch(
            model, val_loader, criterion, device
        )
        from sklearn.metrics import f1_score

        val_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(val_f1)

        scheduler.step(val_f1)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            "epoch %2d | train loss %.4f acc %.4f | val loss %.4f acc %.4f "
            "macroF1 %.4f | lr %.1e",
            epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr_now,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("early stopping at epoch %d (best F1 %.4f @ %d)",
                            epoch, best_f1, best_epoch)
                stop_epoch = epoch
                break

    # Restore the best validation state and persist it.
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(
        model, checkpoint_path, epoch=best_epoch, val_f1=best_f1,
        class_names=class_names,
    )
    logger.info("best model checkpoint -> %s (macroF1 %.4f @ epoch %d)",
                checkpoint_path, best_f1, best_epoch)

    history["best_val_f1"] = best_f1
    history["best_epoch"] = best_epoch
    history["stop_epoch"] = stop_epoch
    return history, float(best_f1)


def evaluate_sequence_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: Optional[list[str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate a trained sequence classifier on a DataLoader.

    Args:
        model: trained MalpracticeLSTM / MalpracticeGRU.
        loader: DataLoader yielding (B, T, F) + (B,).
        class_names: display names for the metrics report.
        device: compute device (auto-detected when None).

    Returns:
        metrics dict (accuracy, macro_f1, per_class, confusion_matrix, ...).
    """
    device = device or get_device()
    model.to(device)
    _, _, y_true, y_pred = _validate_epoch(
        model, loader, nn.CrossEntropyLoss(), device
    )
    return evaluate_classifier(y_true, y_pred, class_names=class_names)


# ---------------------------------------------------------------------------
# Synthetic data (smoke-test block)
# ---------------------------------------------------------------------------

def _make_peeking_sequence(window_size: int = 30, seed: int = 0) -> np.ndarray:
    """
    Synthetic "peeking" behaviour: rapid lateral head saccades toward a
    neighbour (periodic left/right swings) from frame 8 onward.

    Returns:
        (window_size, 17, 2) float32 normalised keypoints.
    """
    from temporal_features import HEAD_KEYPOINTS, _canonical_pose

    rng = np.random.default_rng(seed)
    base = _canonical_pose()
    seq = np.repeat(base[None, :, :], window_size, axis=0).copy()
    seq += rng.normal(0.0, 0.004, size=seq.shape).astype(np.float32)

    frames = np.arange(window_size)
    active = frames >= 8
    t = np.maximum(frames - 8, 0)
    shift = 0.16 * np.sin(0.9 * t) * active  # oscillating lateral head swing
    seq[:, HEAD_KEYPOINTS, 0] += shift[:, None]
    return np.asarray(seq, dtype=np.float32)


def _demo() -> None:
    import tempfile

    from sklearn.model_selection import train_test_split

    from temporal_features import (
        FEATURE_DIM,
        TemporalFeatureExtractor,
        make_synthetic_sequence,
        save_dataset,
    )

    T = 30
    PER_CLASS = 80
    class_names = list(DEFAULT_CLASS_NAMES)

    # --- build a 4-class synthetic corpus (N = 320, T = 30) ---------------
    windows: list[np.ndarray] = []
    labels: list[int] = []
    for label, behavior in enumerate(
        ("normal", "head_turn", "hand_reach", "peeking")
    ):
        for i in range(PER_CLASS):
            if behavior == "peeking":
                seq = _make_peeking_sequence(T, seed=i)
            else:
                seq = make_synthetic_sequence(behavior, T, seed=i)
            windows.append(seq)
            labels.append(label)

    # --- Week 4 feature extraction (N, T, F) ------------------------------
    extractor = TemporalFeatureExtractor(window_size=T)
    X = extractor.extract_batch(np.stack(windows))
    y = np.asarray(labels, dtype=np.int64)
    assert X.shape == (len(windows), T, FEATURE_DIM), X.shape
    print(f"[data] synthetic corpus X={tuple(X.shape)} y={tuple(y.shape)} "
          f"classes={class_names}")

    # --- Week 4 save / Week 5 reload round-trip ----------------------------
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = save_dataset(
            Path(tmp) / "corpus.pt", windows, labels=list(y),
            extractor=extractor, fmt="pt",
        )
        ds = ExamDataset.from_saved(pt_path)
        assert len(ds) == len(windows)
        assert tuple(ds.X.shape) == tuple(X.shape)
        x0, y0 = ds[0]
        assert tuple(x0.shape) == (T, X.shape[-1]) and int(y0) == 0
        print(f"[data] ExamDataset.from_saved round-trip OK ({pt_path.name})")

        # --- stratified split + weighted loaders ---------------------------
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=0
        )
        train_loader, val_loader, class_weights = make_dataloaders(
            X_tr, y_tr, X_va, y_va, batch_size=32, class_weighted=True
        )
        assert class_weights is not None and class_weights.shape[0] == 4
        for xb, yb in train_loader:
            assert tuple(xb.shape) == (32, T, X.shape[-1])
            break
        print(f"[data] loaders ready; class_weights={class_weights.tolist()}")

        # --- 2. baseline models on pooled features -------------------------
        pooled_tr = pool_sequence_features(X_tr)
        pooled_va = pool_sequence_features(X_va)
        assert pooled_tr.shape[1] == X.shape[-1] * 3

        for kind in ("rf", "xgb"):
            model, metrics = train_baseline(
                pooled_tr, y_tr, pooled_va, y_va, kind=kind,
                class_names=class_names, n_estimators=200,
            )
            print(f"\n=== Baseline: {kind.upper()} (pooled mean/max/std) ===")
            print(format_classification_report(metrics))

        # --- 3. deep sequence classifier -----------------------------------
        F = X.shape[-1]
        set_seed(0)  # seed before construction so init + training are reproducible
        model = MalpracticeGRU(
            input_dim=F, hidden_dim=64, num_layers=2, num_classes=4,
            dropout=0.3, use_attention=True,
        )
        ckpt_path = Path(tmp) / "best_malpractice_model.pth"
        history, best_f1 = train_sequence_model(
            model, train_loader, val_loader, class_weights=class_weights,
            class_names=class_names, epochs=25, lr=1e-3, patience=8,
            checkpoint_path=ckpt_path, device=torch.device("cpu"),
        )
        assert ckpt_path.is_file(), "best-model checkpoint missing"
        assert history["val_f1"], "no validation history recorded"

        # --- reload checkpoint + re-evaluate -------------------------------
        fresh = MalpracticeGRU(
            input_dim=F, hidden_dim=64, num_layers=2, num_classes=4,
            dropout=0.3, use_attention=True,
        )
        meta = load_checkpoint(fresh, ckpt_path, device=torch.device("cpu"))
        assert meta["arch"] == "MalpracticeGRU"
        metrics = evaluate_sequence_model(
            fresh, val_loader, class_names=class_names,
            device=torch.device("cpu"),
        )
        print("\n=== Deep: MalpracticeGRU (reloaded from checkpoint) ===")
        print(format_classification_report(metrics))
        assert ckpt_path.stat().st_size > 0

        print(f"\nALL WEEK 5 SMOKE TESTS PASSED "
              f"(baselines + GRU, best val macro-F1 = {best_f1:.4f})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _demo()
