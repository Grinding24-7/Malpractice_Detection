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
import logging
import sys
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


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV and return (X, y) feature matrix and label vector."""
    import csv

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    X_rows: list[list[float]] = []
    y_labels: list[str] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row.get(col, 0.0)) for col in FEATURE_COLS]
                label = row.get(LABEL_COL, "P01")
                X_rows.append(features)
                y_labels.append(label)
            except (ValueError, KeyError):
                continue

    if not X_rows:
        logger.error("No valid rows found in %s", csv_path)
        sys.exit(1)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_labels)
    logger.info("Loaded %d samples, %d features, %d classes", *X.shape, len(set(y)))
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
    args = parser.parse_args()

    logger.info("Loading dataset from %s", args.csv)
    X, y = load_dataset(args.csv)

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
