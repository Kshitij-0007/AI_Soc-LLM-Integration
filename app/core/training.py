"""Shared XGBoost training for every lane.

``train_and_save`` trains one ``multi:softprob`` model, measures it, and
writes the artifacts plus ``manifest.json`` to ``models/<lane>/<version>/``.

Measurement, from most to least trustworthy:

1. ``eval_frame``: rows the lane held out (another dataset, another capture,
   a later time period). This becomes the headline ``macro_f1``.
2. Out-of-fold predictions from (group) k-fold on the training rows. Pass
   ``groups`` so related rows (same capture window, same host, augmented
   copies) never sit on both sides of a split; random row splits on flow data
   leak neighbouring flows and overstate accuracy.

Missing values are passed to XGBoost as NaN (it learns a default branch);
medians are recorded only for reference and for legacy models.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from app.core.contracts import FeatureFrame
from app.core.model_manifest import ModelManifest, model_version_dir, write_manifest
from app.core.resources import xgb_threads

PROJECT_DIR = Path(__file__).resolve().parents[2]
BENIGN = "BENIGN"

# Shallow, regularised trees generalise better across networks than deep ones.
BASE_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 2.0,
    "seed": 42,
}
MAX_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 30
DEFAULT_ROUNDS = 150


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse square-root class frequency."""
    counts = np.bincount(y)
    return 1.0 / np.sqrt(counts[y])


def _cv_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
):
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return splitter.split(X, y)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return splitter.split(X, y, groups=np.asarray(groups))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> dict:
    """Accuracy, macro-F1, per-class scores, confusion matrix, attack-vs-benign."""
    labels = np.arange(len(class_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    # Macro-F1 over classes that occur in y_true; classes absent from the
    # evaluation file would otherwise count as F1 = 0.
    present = np.unique(y_true)
    metrics = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "macro_f1_classes": [class_labels[i] for i in present],
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(class_labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_class_order": class_labels,
    }
    if BENIGN in class_labels:
        benign = class_labels.index(BENIGN)
        actual_attack = y_true != benign
        predicted_attack = y_pred != benign
        attacks = int(actual_attack.sum())
        benigns = int((~actual_attack).sum())
        metrics["attack_vs_benign"] = {
            "detection_rate": (
                float((predicted_attack & actual_attack).sum() / attacks)
                if attacks
                else None
            ),
            "false_positive_rate": (
                float((predicted_attack & ~actual_attack).sum() / benigns)
                if benigns
                else None
            ),
        }
    return metrics


def feature_statistics(X: pd.DataFrame) -> dict[str, dict]:
    """Per-feature ranges used at scoring time to flag unfamiliar uploads."""
    stats = {}
    for name in X.columns:
        values = X[name]
        present = values.dropna()
        entry: dict = {"missing_rate": float(values.isna().mean())}
        if len(present):
            entry.update(
                {
                    "min": float(present.min()),
                    "p01": float(present.quantile(0.01)),
                    "median": float(present.median()),
                    "p99": float(present.quantile(0.99)),
                    "max": float(present.max()),
                }
            )
        stats[str(name)] = entry
    return stats


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _holdout_metrics(
    model: xgb.Booster,
    eval_frame: FeatureFrame,
    feature_list: list[str],
    class_labels: list[str],
) -> dict:
    if eval_frame.y is None:
        raise ValueError("eval_frame needs labels.")
    known = eval_frame.y.isin(class_labels).to_numpy()
    X_eval = eval_frame.X.loc[known, feature_list].astype(float)
    X_eval = X_eval.replace([np.inf, -np.inf], np.nan)
    y_eval = (
        pd.Categorical(eval_frame.y[known], categories=class_labels).codes.astype(int)
    )
    probabilities = model.predict(xgb.DMatrix(X_eval, feature_names=feature_list))
    metrics = classification_metrics(
        y_eval, probabilities.argmax(axis=1), class_labels
    )
    metrics["rows_with_unseen_labels_skipped"] = int((~known).sum())
    return metrics


def train_and_save(
    ff: FeatureFrame,
    lane_id: str,
    version: str = "1.0.0",
    models_dir: Path = PROJECT_DIR / "models",
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    eval_frame: FeatureFrame | None = None,
    eval_description: str = "held-out rows supplied by the lane",
    data_source: str = "unspecified",
    feature_set_version: str | None = None,
    reference_rows: np.ndarray | None = None,
    n_jobs: int | None = None,
    cv_folds: int = 5,
    notes: list[str] | None = None,
) -> dict:
    """Train, evaluate and persist an XGBoost classification model.

    ``sample_weight`` scales the class-balance weights per row.
    ``groups`` keeps related rows in the same cross-validation fold.
    ``reference_rows`` (boolean mask) selects the rows used for the feature
    statistics, e.g. to exclude augmented copies with blanked columns.
    ``n_jobs`` defaults to the runtime budget; offline training scripts can
    pass all cores. ``cv_folds=0`` skips cross-validation.
    """
    if ff.y is None:
        raise ValueError("Training requires a label column.")

    X = ff.X.copy()
    y_labels = ff.y.astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if X.empty:
        raise ValueError("Feature matrix is empty.")
    if len(X) != len(y_labels):
        raise ValueError("Feature and label row counts differ.")
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("All model features must be numeric.")

    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    feature_list = [str(c) for c in X.columns]

    class_labels = sorted(y_labels.unique().tolist())
    if BENIGN not in class_labels:
        raise ValueError(f"Training data needs a class named exactly '{BENIGN}'.")
    if len(class_labels) < 2:
        raise ValueError("At least two classes are required.")
    y = pd.Categorical(y_labels, categories=class_labels).codes.astype(int)
    num_classes = len(class_labels)
    class_counts = np.bincount(y, minlength=num_classes)

    sample_weights = _sample_weights(y)
    if sample_weight is not None:
        sample_weights = sample_weights * np.asarray(sample_weight, dtype=float)
    # Mean 1, so min_child_weight keeps its meaning (roughly "rows per leaf")
    # whatever the dataset size; raw 1/sqrt(count) weights are tiny and
    # would block almost every split.
    sample_weights = sample_weights / sample_weights.mean()

    params = {
        **BASE_PARAMS,
        "num_class": num_classes,
        "nthread": n_jobs or xgb_threads(),
    }

    folds = []
    best_rounds = []
    oof_probabilities = np.full((len(X), num_classes), np.nan)
    min_class_count = int(class_counts.min())

    if cv_folds and min_class_count >= 2:
        n_splits = min(cv_folds, min_class_count)
        for fold, (train_idx, test_idx) in enumerate(
            _cv_splits(X, y, groups, n_splits), start=1
        ):
            dtrain = xgb.DMatrix(
                X.iloc[train_idx], label=y[train_idx], weight=sample_weights[train_idx]
            )
            dtest = xgb.DMatrix(
                X.iloc[test_idx], label=y[test_idx], weight=sample_weights[test_idx]
            )
            fold_model = xgb.train(
                params,
                dtrain,
                num_boost_round=MAX_ROUNDS,
                evals=[(dtest, "validation")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            rounds = fold_model.best_iteration + 1
            best_rounds.append(rounds)
            probabilities = fold_model.predict(dtest, iteration_range=(0, rounds))
            oof_probabilities[test_idx] = probabilities
            folds.append(
                {
                    "fold": fold,
                    "macro_f1": float(
                        f1_score(
                            y[test_idx],
                            probabilities.argmax(axis=1),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "best_rounds": int(rounds),
                }
            )
            print(
                f"      fold {fold}/{n_splits}: macro-F1 "
                f"{folds[-1]['macro_f1']:.4f} ({rounds} rounds)",
                flush=True,
            )

    num_rounds = int(np.median(best_rounds)) if best_rounds else DEFAULT_ROUNDS

    model = xgb.train(
        params,
        xgb.DMatrix(X, label=y, weight=sample_weights),
        num_boost_round=num_rounds,
        verbose_eval=False,
    )

    training_predictions = model.predict(
        xgb.DMatrix(X, feature_names=feature_list)
    ).argmax(axis=1)
    training_macro_f1 = float(
        f1_score(y, training_predictions, average="macro", zero_division=0)
    )

    scored = ~np.isnan(oof_probabilities).any(axis=1)
    oof = (
        classification_metrics(
            y[scored], oof_probabilities[scored].argmax(axis=1), class_labels
        )
        if folds and scored.any()
        else None
    )
    holdout = (
        _holdout_metrics(model, eval_frame, feature_list, class_labels)
        if eval_frame is not None
        else None
    )

    reference = X if reference_rows is None else X.loc[np.asarray(reference_rows)]
    medians = reference.median().fillna(0.0)
    class_count_map = {
        label: int(count)
        for label, count in zip(class_labels, class_counts, strict=True)
    }

    artifact_dir = model_version_dir(models_dir, lane_id, version)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.json"
    model.save_model(str(model_path))
    _write_json(artifact_dir / "feature_list.json", feature_list)
    _write_json(artifact_dir / "class_labels.json", class_labels)

    stats = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "medians": {str(k): float(v) for k, v in medians.items()},
        "feature_stats": feature_statistics(reference),
        "class_counts": class_count_map,
        "num_boost_round": num_rounds,
        "params": {k: v for k, v in params.items() if k != "nthread"},
        "cross_validation_folds": len(folds),
        "cross_validation_grouped": groups is not None,
        "folds": folds,
        "training_macro_f1": training_macro_f1,
    }
    _write_json(artifact_dir / "train_stats.json", stats)

    if holdout is not None:
        headline, evaluation = holdout, eval_description
    elif oof is not None:
        headline = oof
        evaluation = (
            "group k-fold out-of-fold predictions"
            if groups is not None
            else "stratified k-fold out-of-fold predictions (random rows; optimistic)"
        )
    else:
        headline, evaluation = {}, "none (too few rows per class)"

    metrics = {
        "macro_f1": headline.get("macro_f1"),
        "accuracy": headline.get("accuracy"),
        "per_class": headline.get("per_class", {}),
        "attack_vs_benign": headline.get("attack_vs_benign"),
        "confusion_matrix": headline.get("confusion_matrix"),
        "confusion_matrix_class_order": class_labels,
        "evaluation": evaluation,
        "holdout": holdout,
        "cross_validation": oof,
        "class_counts": class_count_map,
        "training_macro_f1": training_macro_f1,
        "data_source": data_source,
        "warning": (
            "Scores only describe data like the evaluation set. Accuracy on a "
            "network or exporter the model never saw can be much lower; check "
            "the compatibility report on each upload and evaluate on a second "
            "dataset with scripts/evaluate_netflow.py --mode cross-dataset."
        ),
    }
    _write_json(artifact_dir / "metrics.json", metrics)

    write_manifest(
        artifact_dir,
        ModelManifest(
            lane_id=lane_id,
            version=version,
            model_format="xgboost_json",
            model_file="model.json",
            missing_values="native",
            benign_label=BENIGN,
            objective=params["objective"],
            feature_set_version=feature_set_version,
            data_source=data_source,
            trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
            notes=notes or [],
        ),
    )

    return {
        "lane_id": lane_id,
        "version": version,
        "artifact_dir": str(artifact_dir),
        "model_path": str(model_path),
        "macro_f1": metrics["macro_f1"],
        "evaluation": evaluation,
        "cross_validation_macro_f1": oof["macro_f1"] if oof else None,
        "holdout_macro_f1": holdout["macro_f1"] if holdout else None,
        "training_macro_f1": training_macro_f1,
        "metrics": metrics,
    }
