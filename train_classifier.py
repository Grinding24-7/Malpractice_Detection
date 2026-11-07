"""Train the proctoring posture classifier from logged 54-D features.

Reads the CSV written by :class:`desktop_app.core.dataset_logger.DatasetLogger`
and produces ``models/pose_rf_classifier.pkl`` for the optional hybrid
Rule + ML path in :class:`desktop_app.core.pose_classifier.PoseClassifier`.

Design notes
------------
* The split is stratified, so a small infraction class is represented in both
  train and test partitions.
* Every fold owns its imputer and scaler.  Scaling the whole matrix before
  cross-validation leaks test-fold statistics into training and inflates the
  score; the pipeline here is cloned and fitted inside each fold.
* Missing measurements are imputed with the training median, never with zero.
* The saved artifact contains the model, the exact feature-name order, the
  class list, dataset provenance and the measured metrics.  The runtime
  refuses an artifact whose feature contract does not match.
* Accuracy is only reported from data that was actually supplied.  With no
  labelled rows the script exits instead of inventing a result.

Usage:
    python train_classifier.py \
        --csv data/proctoring_features.csv \
        --model models/pose_rf_classifier.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from desktop_app.core.feature_schema import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_classifier")

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "proctoring_features.csv"
_DEFAULT_MODEL = _PROJECT_ROOT / "models" / "pose_rf_classifier.pkl"

FEATURE_COLS: list[str] = list(FEATURE_NAMES)
LABEL_COL = "pose_id"
SOURCE_COL = "source"
_SCHEMA_CONTRACT = "54d-v2"


def _schema_digest() -> str:
    """Stable fingerprint of the columns the cached matrix depends on."""
    payload = "\x1f".join([_SCHEMA_CONTRACT, LABEL_COL, *FEATURE_COLS]).encode(
        "utf-8"
    )
    return hashlib.sha1(payload).hexdigest()[:16]


def _cache_meta(csv_path: Path, min_class_samples: int) -> dict[str, Any]:
    stat = csv_path.stat()
    return {
        "source": str(csv_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "schema": _schema_digest(),
        "features": len(FEATURE_COLS),
        "min_class_samples": int(min_class_samples),
        "contract": _SCHEMA_CONTRACT,
    }


def _read_frame(csv_path: Path):
    import pandas as pd

    columns = {LABEL_COL, SOURCE_COL, *FEATURE_COLS}
    frame = pd.read_csv(
        csv_path,
        usecols=lambda column: column in columns,
        engine="c",
    )
    missing = [column for column in FEATURE_COLS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} feature column(s): {missing[:5]}"
        )
    if LABEL_COL not in frame.columns:
        raise ValueError(f"CSV is missing label column {LABEL_COL!r}")
    return frame


def load_dataset(
    csv_path: Path,
    cache_path: Path | None = None,
    use_cache: bool = True,
    min_class_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV and return ``(X, y)`` with finite-or-NaN features.

    Rows whose label is empty or whose 54 features are all missing are
    dropped.  Individual missing measurements are kept as NaN for the
    training pipeline's imputer.
    """
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    cache_path = cache_path or csv_path.with_suffix(
        csv_path.suffix + ".feature_cache.npz"
    )
    meta = _cache_meta(csv_path, min_class_samples)

    if use_cache and cache_path.is_file():
        try:
            started = time.perf_counter()
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_meta = json.loads(str(cached["meta"]))
                if cached_meta == meta:
                    X = np.asarray(cached["X"], dtype=np.float64)
                    y = np.asarray(cached["y"], dtype=str)
                    logger.info(
                        "Loaded %d samples from cache %s in %.1f ms",
                        X.shape[0],
                        cache_path.name,
                        (time.perf_counter() - started) * 1000,
                    )
                    return X, y
                logger.info("Feature cache stale; rebuilding from %s", csv_path)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Ignoring unreadable feature cache %s: %s", cache_path, exc
            )

    import pandas as pd

    started = time.perf_counter()
    frame = _read_frame(csv_path)

    labels = frame[LABEL_COL].astype("string").str.strip()
    feature_values = (
        frame[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64, copy=False)
    )

    valid = labels.notna() & labels.ne("") & ~np.isnan(feature_values).all(axis=1)
    dropped = int(valid.size - valid.sum())
    X = np.ascontiguousarray(feature_values[valid])
    y = np.asarray(labels.fillna("").to_numpy(dtype=str)[valid], dtype=str)

    if X.shape[0] == 0:
        logger.error("No valid labelled rows found in %s", csv_path)
        sys.exit(1)
    if dropped:
        logger.warning(
            "Dropped %d row(s) with an empty label or no measured features",
            dropped,
        )

    counts = pd.Series(y).value_counts().sort_index()
    logger.info("Loaded %d rows, %d features", *X.shape)
    logger.info(
        "Class distribution: %s",
        ", ".join(f"{label}={count}" for label, count in counts.items()),
    )

    missing_rate = np.isnan(X).mean(axis=0)
    top_missing = np.argsort(missing_rate)[-10:][::-1]
    logger.info(
        "Highest missing-feature rates: %s",
        ", ".join(
            f"{FEATURE_COLS[i]}={missing_rate[i]:.1%}" for i in top_missing
        ),
    )

    if min_class_samples > 1:
        keep = counts[counts >= min_class_samples].index
        removed = counts[counts < min_class_samples]
        if len(removed):
            logger.warning(
                "Dropping %d class(es) with <%d samples: %s",
                len(removed),
                min_class_samples,
                ", ".join(f"{label}={count}" for label, count in removed.items()),
            )
        mask = np.isin(y, keep.to_numpy(dtype=str))
        X = np.ascontiguousarray(X[mask])
        y = y[mask]
        if X.shape[0] == 0:
            logger.error(
                "No classes remain after --min-class-samples=%d",
                min_class_samples,
            )
            sys.exit(1)

    if len(np.unique(y)) < 2:
        logger.error(
            "At least two classes are required to train a classifier; found %s",
            np.unique(y).tolist(),
        )
        sys.exit(1)

    if SOURCE_COL in frame.columns:
        sources = frame.loc[valid, SOURCE_COL].fillna("unknown").astype(str)
        if min_class_samples > 1:
            sources = sources[mask]
        breakdown = (
            pd.DataFrame({"label": y, "source": sources.to_numpy(dtype=str)})
            .value_counts()
            .sort_index()
        )
        logger.info(
            "Source breakdown: %s",
            ", ".join(
                f"{label}/{source}={count}"
                for (label, source), count in breakdown.items()
            ),
        )

    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                X=X,
                y=y,
                meta=np.asarray(json.dumps(meta)),
            )
            logger.info("Feature matrix cached -> %s", cache_path)
        except OSError as exc:
            logger.warning("Could not write feature cache %s: %s", cache_path, exc)
    return X, y


def _make_pipeline(model: Any):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("classifier", model),
        ]
    )


def _freeze_inference_threads(model: Any) -> None:
    """Serialise one-sample inference before saving.

    ``n_jobs=-1`` is right for fitting, but a single-row ``predict_proba``
    with a thread pool costs far more than it saves: measured ~78 ms versus
    ~28 ms with one thread on this 200-tree forest.  The desktop app scores
    one candidate at a time, so persist the single-thread setting.
    """
    candidates = [model]
    inner = getattr(model, "pipeline", None)
    if inner is not None:
        candidates.append(inner)
    for candidate in candidates:
        steps = getattr(candidate, "named_steps", None)
        if not steps:
            continue
        for step in steps.values():
            try:
                params = step.get_params(deep=False)
            except Exception:
                continue
            if "n_jobs" in params:
                try:
                    step.set_params(n_jobs=1)
                except Exception:
                    pass


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "labels": labels,
    }


def _cv_metrics(
    pipeline: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold, cross_validate

    counts = np.unique(y, return_counts=True)[1]
    folds = max(2, min(int(cv_folds), int(counts.min())))
    if folds != cv_folds:
        logger.warning(
            "Reducing CV folds from %d to %d (smallest class has %d samples)",
            cv_folds,
            folds,
            int(counts.min()),
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
        },
        n_jobs=-1,
        error_score="raise",
    )
    return {
        "folds": folds,
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "balanced_accuracy_mean": float(
            np.mean(scores["test_balanced_accuracy"])
        ),
        "balanced_accuracy_std": float(
            np.std(scores["test_balanced_accuracy"])
        ),
        "macro_f1_mean": float(np.mean(scores["test_macro_f1"])),
        "macro_f1_std": float(np.std(scores["test_macro_f1"])),
    }


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    cv_folds: int = 5,
) -> tuple[Any, dict[str, Any]]:
    """Train and evaluate a leakage-free RandomForest pipeline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pipeline = _make_pipeline(
        RandomForestClassifier(
            # 100 trees reproduces the 200-tree hold-out metrics on the
            # current 1,142-row set (94.76% accuracy) at half the one-sample
            # inference cost, which matters for the 4 FPS live floor.
            n_estimators=100,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    )

    logger.info(
        "Training RandomForest on %d samples / %d classes (test=%d, stratified)",
        len(y_train),
        len(np.unique(y_train)),
        len(y_test),
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_metrics = _classification_metrics(y_test, y_pred)
    logger.info(
        "Hold-out accuracy %.2f%% | balanced %.2f%% | macro-F1 %.2f%%",
        test_metrics["accuracy"] * 100,
        test_metrics["balanced_accuracy"] * 100,
        test_metrics["macro_f1"] * 100,
    )
    print("\nHold-out classification report:")
    print(test_metrics["classification_report"])

    cv = _cv_metrics(pipeline, X, y, cv_folds=cv_folds, seed=seed)
    logger.info(
        "%d-fold CV (pipeline refit inside every fold): accuracy %.2f%% +/- %.2f%% | macro-F1 %.2f%%",
        cv["folds"],
        cv["accuracy_mean"] * 100,
        cv["accuracy_std"] * 100,
        cv["macro_f1_mean"] * 100,
    )

    classifier = pipeline.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", None)
    top_features: list[dict[str, Any]] = []
    if importances is not None and len(importances) == len(FEATURE_COLS):
        order = np.argsort(importances)[-15:][::-1]
        top_features = [
            {"feature": FEATURE_COLS[i], "importance": float(importances[i])}
            for i in order
        ]

    metrics = {
        "backend": "random_forest",
        "test_size": float(test_size),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "classification_report": test_metrics["classification_report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": test_metrics["labels"],
        "cv": cv,
        "top_features": top_features,
    }
    return pipeline, metrics


class LabelDecodingClassifier:
    """Decode an XGBoost numeric-label pipeline back to P-IDs.

    The wrapper lives in the project package so joblib can import it when the
    artifact is loaded by the desktop app.
    """

    def __init__(self, pipeline: Any, classes: np.ndarray) -> None:
        self.pipeline = pipeline
        self.classes_ = np.asarray(classes, dtype=str)

    @property
    def n_features_in_(self) -> int:
        return int(getattr(self.pipeline, "n_features_in_", 0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded = np.asarray(self.pipeline.predict(X)).astype(int)
        return self.classes_[encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(X))


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    cv_folds: int = 5,
) -> tuple[Any, dict[str, Any]]:
    """Train XGBoost when installed, otherwise fall back to RandomForest."""
    try:
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed - falling back to RandomForest")
        return train_random_forest(X, y, test_size, seed, cv_folds)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded,
    )

    pipeline = _make_pipeline(
        XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    )

    counts = np.unique(y_encoded, return_counts=True)[1]
    folds = max(2, min(int(cv_folds), int(counts.min())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for train_idx, valid_idx in splitter.split(X, y_encoded):
        fold_model = clone(pipeline)
        fold_model.fit(X[train_idx], y_encoded[train_idx])
        fold_pred = fold_model.predict(X[valid_idx])
        fold_scores.append(float(np.mean(fold_pred == y_encoded[valid_idx])))

    pipeline.fit(X_train, y_train)
    y_pred = encoder.inverse_transform(pipeline.predict(X_test).astype(int))
    test_metrics = _classification_metrics(encoder.inverse_transform(y_test), y_pred)
    logger.info(
        "XGBoost hold-out accuracy %.2f%% | macro-F1 %.2f%%",
        test_metrics["accuracy"] * 100,
        test_metrics["macro_f1"] * 100,
    )
    print("\nHold-out classification report:")
    print(test_metrics["classification_report"])

    wrapper = LabelDecodingClassifier(pipeline, encoder.classes_)
    metrics = {
        "backend": "xgboost",
        "test_size": float(test_size),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "classification_report": test_metrics["classification_report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": test_metrics["labels"],
        "cv": {
            "folds": folds,
            "accuracy_mean": float(np.mean(fold_scores)),
            "accuracy_std": float(np.std(fold_scores)),
        },
        "top_features": [],
    }
    return wrapper, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the P01-P40 hybrid pose classifier"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV,
        help="Path to training CSV (default: data/proctoring_features.csv)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL,
        help="Output model path (default: models/pose_rf_classifier.pkl)",
    )
    parser.add_argument(
        "--backend",
        choices=["rf", "xgb"],
        default="rf",
        help="Training backend: rf=RandomForest, xgb=XGBoost (default: rf)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Stratified hold-out fraction (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds (default: 5; reduced for tiny classes)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=2,
        help="Drop classes with fewer than this many rows (default: 2)",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Metrics sidecar path (default: <model>.metrics.json)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Feature-matrix cache path (default: <csv>.feature_cache.npz)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Always re-parse the CSV instead of using/writing the cache",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore an existing cache and parse the CSV again",
    )
    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    logger.info("Loading dataset from %s", args.csv)
    cache = args.cache or args.csv.with_suffix(
        args.csv.suffix + ".feature_cache.npz"
    )
    if args.rebuild_cache and cache.is_file():
        cache.unlink()
    X, y = load_dataset(
        args.csv,
        cache_path=cache,
        use_cache=not args.no_cache,
        min_class_samples=args.min_class_samples,
    )

    if args.backend == "xgb":
        model, metrics = train_xgboost(
            X,
            y,
            test_size=args.test_size,
            seed=args.seed,
            cv_folds=args.cv_folds,
        )
    else:
        model, metrics = train_random_forest(
            X,
            y,
            test_size=args.test_size,
            seed=args.seed,
            cv_folds=args.cv_folds,
        )

    counts = {str(label): int((y == label).sum()) for label in np.unique(y)}
    _freeze_inference_threads(model)
    artifact = {
        "model": model,
        "feature_names": list(FEATURE_COLS),
        "label_column": LABEL_COL,
        "classes": sorted(counts),
        "class_counts": counts,
        "rows": int(len(y)),
        "dataset": str(args.csv.resolve()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        **metrics,
    }

    args.model.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(artifact, str(args.model))
    logger.info(
        "Model saved to %s (%d rows, %d classes, test accuracy %.2f%%)",
        args.model,
        artifact["rows"],
        len(artifact["classes"]),
        artifact["test_accuracy"] * 100,
    )

    metrics_path = args.metrics_json or args.model.with_suffix(
        args.model.suffix + ".metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(artifact, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Metrics written to %s", metrics_path)
    logger.info("Trained classes: %s", ", ".join(artifact["classes"]))


if __name__ == "__main__":
    main()
