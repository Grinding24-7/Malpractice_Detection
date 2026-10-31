"""
train_classifier.py — Standalone training script for P01-P40 pose classifiers.

Reads the feature CSV logged by DatasetLogger and trains a multi-class
classifier (RandomForest or XGBoost) to predict pose categories.

Usage:
    python train_classifier.py [--csv data/proctoring_features.csv] [--model models/pose_rf_classifier.pkl]

Output:
    - Trained model saved to models/pose_rf_classifier.pkl
    - Classification report printed to stdout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_classifier")

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "proctoring_features.csv"
_DEFAULT_MODEL = _PROJECT_ROOT / "models" / "pose_rf_classifier.pkl"

# Feature column names (must match dataset_logger.py)
FEATURE_COLS: list[str] = [
    *[f"kpt_x{i}" for i in range(17)],
    *[f"kpt_y{i}" for i in range(17)],
    "yaw", "pitch", "roll",
    "elbow_flexion_left", "elbow_flexion_right", "neck_tilt_angle",
    "wrist_to_eye_ratio", "wrist_to_ear_ratio", "eye_to_shoulder_ratio",
    "hand_mouth_distance", "shoulder_hip_offset_ratio", "torso_leg_ratio",
    "shoulder_width_ratio", "hip_width_ratio", "torso_asymmetry",
    "motion_delta",
]

LABEL_COL = "pose_id"


def _schema_digest() -> str:
    """Stable fingerprint of the columns the cached matrix depends on."""
    payload = "\x1f".join([LABEL_COL, *FEATURE_COLS]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _cache_meta(csv_path: Path) -> dict[str, Any]:
    stat = csv_path.stat()
    return {
        "source": str(csv_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "schema": _schema_digest(),
        "features": len(FEATURE_COLS),
    }


def load_dataset(
    csv_path: Path,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV and return (X, y) as float32 features and string labels.

    Parsing is vectorised with pandas and the resulting matrix is cached in a
    sibling ``.feature_cache.npz`` keyed by file size, mtime and column
    schema.  Repeated training runs therefore skip CSV/token parsing entirely;
    editing the CSV or the feature columns invalidates the cache.
    """
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    cache_path = cache_path or csv_path.with_suffix(
        csv_path.suffix + ".feature_cache.npz"
    )
    meta = _cache_meta(csv_path)

    if use_cache and cache_path.is_file():
        try:
            started = time.perf_counter()
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_meta = json.loads(str(cached["meta"]))
                if cached_meta == meta:
                    X = np.asarray(cached["X"], dtype=np.float32)
                    y = np.asarray(cached["y"], dtype=str)
                    logger.info(
                        "Loaded %d samples from cache %s in %.1f ms",
                        X.shape[0], cache_path.name,
                        (time.perf_counter() - started) * 1000,
                    )
                    return X, y
                logger.info("Feature cache stale; rebuilding from %s", csv_path)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable feature cache %s: %s", cache_path, exc)

    import pandas as pd

    started = time.perf_counter()
    read_kwargs = dict(
        usecols=[LABEL_COL, *FEATURE_COLS],
        engine="c",
        memory_map=True,
    )
    try:
        # Fast path: a clean numeric CSV is parsed directly into float32.
        frame = pd.read_csv(
            csv_path,
            dtype={
                LABEL_COL: "string",
                **{col: "float32" for col in FEATURE_COLS},
            },
            **read_kwargs,
        )
        feature_values = frame[FEATURE_COLS].to_numpy(
            dtype=np.float32, copy=False
        )
    except ValueError as exc:
        if "could not convert" not in str(exc):
            logger.error("CSV schema mismatch in %s: %s", csv_path, exc)
            sys.exit(1)
        # A single malformed cell must not abort a large dataset.  Re-read
        # without the float32 dtype and coerce bad cells to NaN, which the
        # validity mask below drops.
        frame = pd.read_csv(
            csv_path, dtype={LABEL_COL: "string"}, **read_kwargs
        )
        feature_values = (
            frame[FEATURE_COLS]
            .apply(pd.to_numeric, errors="coerce", downcast="float")
            .to_numpy(dtype=np.float32, copy=False)
        )

    y_series = frame[LABEL_COL]
    valid = np.isfinite(feature_values).all(axis=1) & y_series.notna().to_numpy()
    dropped = int(valid.size - valid.sum())
    X = np.ascontiguousarray(feature_values[valid])
    y = np.asarray(y_series.fillna("").to_numpy(dtype=str)[valid], dtype=str)

    if X.shape[0] == 0:
        logger.error("No valid rows found in %s", csv_path)
        sys.exit(1)
    if dropped:
        logger.warning("Dropped %d rows with missing/non-numeric features", dropped)

    logger.info(
        "Parsed %d samples, %d features, %d classes in %.1f ms",
        *X.shape, len(np.unique(y)), (time.perf_counter() - started) * 1000,
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


def train_random_forest(
    X: np.ndarray, y: np.ndarray
) -> tuple[Any, float]:
    """Train a RandomForestClassifier and return (model, accuracy)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    accuracy = model.score(X_test_s, y_test)
    logger.info("RandomForest test accuracy: %.2f%%", accuracy * 100)

    # Cross-validation
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, n_jobs=-1)
    logger.info("5-fold CV accuracy: %.2f%% (+/- %.2f%%)", cv_scores.mean() * 100, cv_scores.std() * 100)

    # Classification report
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test_s)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Wrap scaler + model as a simple pipeline-like object
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("scaler", scaler), ("classifier", model)])
    return pipeline, accuracy


def train_xgboost(
    X: np.ndarray, y: np.ndarray
) -> tuple[Any, float]:
    """Train an XGBoost classifier and return (model, accuracy)."""
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        logger.warning("xgboost not installed — falling back to RandomForest")
        return train_random_forest(X, y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    logger.info("XGBoost test accuracy: %.2f%%", accuracy * 100)

    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])
    return pipeline, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P01-P40 pose classifier")
    parser.add_argument(
        "--csv", type=Path, default=_DEFAULT_CSV,
        help="Path to training CSV (default: data/proctoring_features.csv)",
    )
    parser.add_argument(
        "--model", type=Path, default=_DEFAULT_MODEL,
        help="Output model path (default: models/pose_rf_classifier.pkl)",
    )
    parser.add_argument(
        "--backend", choices=["rf", "xgb"], default="rf",
        help="Training backend: rf=RandomForest, xgb=XGBoost (default: rf)",
    )
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="Feature-matrix cache path (default: <csv>.feature_cache.npz)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="always re-parse the CSV instead of using/writing the cache",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="ignore an existing cache and parse the CSV again",
    )
    args = parser.parse_args()

    logger.info("Loading dataset from %s", args.csv)
    cache = args.cache or args.csv.with_suffix(
        args.csv.suffix + ".feature_cache.npz"
    )
    if args.rebuild_cache and cache.is_file():
        cache.unlink()
    X, y = load_dataset(args.csv, cache_path=cache, use_cache=not args.no_cache)

    if args.backend == "xgb":
        model, accuracy = train_xgboost(X, y)
    else:
        model, accuracy = train_random_forest(X, y)

    # Save model
    args.model.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, str(args.model))
    logger.info("Model saved to %s (%.2f%% accuracy)", args.model, accuracy * 100)


if __name__ == "__main__":
    main()
