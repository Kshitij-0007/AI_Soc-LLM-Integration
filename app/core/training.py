
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.core.contracts import FeatureFrame


def train_and_save(
    ff: FeatureFrame,
    lane_id: str,
    version: str = "1.0.0",
    models_dir: Path = Path(__file__).resolve().parents[2] / "models",
) -> dict:
    """Train, evaluate and persist an XGBoost classification model."""

    if ff.y is None:
        raise ValueError("Training requires a label column.")

    X = ff.X.copy()
    raw_y = pd.Series(ff.y).reset_index(drop=True)

    if X.empty or X.shape[1] == 0:
        raise ValueError("Feature matrix is empty.")

    if len(X) != len(raw_y):
        raise ValueError("Feature and label row counts do not match.")

    if raw_y.isna().any():
        raise ValueError("Training labels contain missing values.")

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("All model features must be numeric.")

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing or infinite values.")

    # Convert class names to contiguous integer IDs for XGBoost.
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y.astype(str))

    num_classes = len(encoder.classes_)

    if num_classes < 2:
        raise ValueError("At least two classes are required.")

    class_counts = pd.Series(y).value_counts().sort_index()

    # Inverse-square-root weighting reduces class imbalance.
    class_weights = {
        int(cls): 1.0 / np.sqrt(count)
        for cls, count in class_counts.items()
    }
    sample_weights = np.array(
        [class_weights[int(cls)] for cls in y],
        dtype=float,
    )

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "nthread": 4,
        "seed": 42,
    }

    folds = []
    min_class_count = int(class_counts.min())

    # Cross-validation requires at least two examples in every class.
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y),
            start=1,
        ):
            dtrain = xgb.DMatrix(
                X.iloc[train_idx],
                label=y[train_idx],
                weight=sample_weights[train_idx],
            )
            dtest = xgb.DMatrix(
                X.iloc[test_idx],
                label=y[test_idx],
            )

            fold_model = xgb.train(
                params,
                dtrain,
                num_boost_round=50,
                evals=[(dtest, "validation")],
                verbose_eval=False,
            )

            predictions = fold_model.predict(dtest).argmax(axis=1)

            score = f1_score(
                y[test_idx],
                predictions,
                average="macro",
                zero_division=0,
            )

            folds.append({
                "fold": fold,
                "macro_f1": float(score),
            })

    # Fit the final model using all supplied training rows.
    dtrain_full = xgb.DMatrix(
        X,
        label=y,
        weight=sample_weights,
        feature_names=list(X.columns),
    )

    model = xgb.train(
        params,
        dtrain_full,
        num_boost_round=50,
        verbose_eval=False,
    )

    train_predictions = model.predict(dtrain_full).argmax(axis=1)

    training_macro_f1 = float(
        f1_score(
            y,
            train_predictions,
            average="macro",
            zero_division=0,
        )
    )

    # Persist artifacts.
    artifact_dir = models_dir / lane_id / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.json"
    model.save_model(str(model_path))

    feature_path = artifact_dir / "feature_list.json"
    feature_path.write_text(
        json.dumps(list(X.columns), indent=2),
        encoding="utf-8",
    )

    class_labels = encoder.classes_.tolist()

    class_path = artifact_dir / "class_labels.json"
    class_path.write_text(
        json.dumps(class_labels, indent=2),
        encoding="utf-8",
    )

    class_count_map = {
        str(label): int(count)
        for label, count in zip(
            encoder.classes_,
            np.bincount(y, minlength=num_classes),
        )
    }

    stats = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "class_counts": class_count_map,
        "cross_validation_folds": len(folds),
        "cross_validation_macro_f1": (
            float(np.mean([f["macro_f1"] for f in folds])
                  ) if folds else None
        ),
        "folds": folds,
        "training_macro_f1": training_macro_f1,
    }

    stats_path = artifact_dir / "train_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    metrics = {
        **stats,
        "data_source": "synthetic",
        "warning": (
            "Training metrics are not an independent estimate of "
            "real-world performance. Use held-out real traffic for validation."
        ),
    }

    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return {
        "model_id": str(model_path),
        "version": version,
        "lane_id": lane_id,
        "class_labels": class_labels,
        "feature_list": list(X.columns),
        "metrics_path": str(metrics_path),
        "train_stats_path": str(stats_path),
        "cross_validation_macro_f1": stats["cross_validation_macro_f1"],
        "training_macro_f1": training_macro_f1,
    }
