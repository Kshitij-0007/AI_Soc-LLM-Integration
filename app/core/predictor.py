"""Load a model version folder and score a lane's FeatureFrame.

Shared by every lane. The lane only supplies ``build_features``; this module
handles artifacts, feature order, missing values, probabilities, TreeSHAP
explanations and a compatibility check that warns when an upload does not
look like the training data (the main cause of silent accuracy drops on new
CSVs).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from app.core.explain import top_contributions
from app.core.model_manifest import ModelManifest, load_manifest

PROJECT_DIR = Path(__file__).resolve().parents[2]

# A feature counts as out of range when more than this share of the upload
# falls outside the training 1st-99th percentile band (plus a margin).
OUT_OF_RANGE_WARN_SHARE = 0.25
RANGE_MARGIN = 0.05
LOW_CONFIDENCE = 0.5
LOW_CONFIDENCE_WARN_SHARE = 0.3


@dataclass
class LoadedModel:
    model_dir: Path
    manifest: ModelManifest
    booster: xgb.Booster
    feature_list: list[str]
    class_labels: list[str]
    train_stats: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def medians(self) -> dict[str, float]:
        return self.train_stats.get("medians", {})

    @property
    def feature_stats(self) -> dict[str, dict]:
        return self.train_stats.get("feature_stats", {})


_cache: dict[tuple[str, float], LoadedModel] = {}
_cache_lock = threading.Lock()


def resolve_model_dir(model_path: str | Path) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.parent if path.is_file() else path


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load(model_dir: Path) -> LoadedModel:
    manifest = load_manifest(model_dir)
    model_file = model_dir / manifest.model_file
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    booster = xgb.Booster()
    booster.load_model(str(model_file))

    features_path = model_dir / manifest.feature_list_file
    if features_path.is_file():
        feature_list = list(_read_json(features_path))
    elif booster.feature_names:
        feature_list = list(booster.feature_names)
    else:
        raise FileNotFoundError(
            f"{features_path.name} missing and the model stores no feature names"
        )

    labels_path = model_dir / manifest.class_labels_file
    if not labels_path.is_file():
        raise FileNotFoundError(f"Class labels not found: {labels_path}")
    class_labels = [str(c) for c in _read_json(labels_path)]

    def optional(name: str | None) -> dict:
        if name and (model_dir / name).is_file():
            return dict(_read_json(model_dir / name))
        return {}

    return LoadedModel(
        model_dir=model_dir,
        manifest=manifest,
        booster=booster,
        feature_list=feature_list,
        class_labels=class_labels,
        train_stats=optional(manifest.train_stats_file),
        metrics=optional(manifest.metrics_file),
    )


def load_model(model_path: str | Path, threads: int | None = None) -> LoadedModel:
    """Load (once) and cache a model folder. Reloads if the model file changes."""
    model_dir = resolve_model_dir(model_path)
    manifest = load_manifest(model_dir)
    model_file = model_dir / manifest.model_file
    key = (str(model_dir), model_file.stat().st_mtime if model_file.exists() else 0)
    with _cache_lock:
        if key not in _cache:
            loaded = _load(model_dir)
            if threads:
                loaded.booster.set_param({"nthread": threads})
            _cache[key] = loaded
        return _cache[key]


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def check_feature_set(model: LoadedModel, lane_feature_set: str | None) -> None:
    expected = model.manifest.feature_set_version
    if expected and lane_feature_set and expected != lane_feature_set:
        raise ValueError(
            f"Model {model.manifest.lane_id}/{model.manifest.version} was trained "
            f"with feature set {expected}, but the lane now builds feature set "
            f"{lane_feature_set}. Retrain the model or activate a matching version."
        )


def prepare_matrix(model: LoadedModel, X: pd.DataFrame) -> pd.DataFrame:
    """Reorder to the training feature list and apply the training NaN policy."""
    missing = [name for name in model.feature_list if name not in X.columns]
    if missing:
        raise ValueError(
            "Model expects features the lane did not produce "
            f"(feature set mismatch): {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    matrix = X[model.feature_list].astype(float)
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    policy = model.manifest.missing_values
    if policy == "median":
        medians = pd.Series(model.medians, dtype=float).reindex(model.feature_list)
        matrix = matrix.fillna(medians).fillna(0.0)
    elif policy == "zero":
        matrix = matrix.fillna(0.0)
    return matrix


def _probabilities(model: LoadedModel, matrix: pd.DataFrame) -> np.ndarray:
    dmatrix = xgb.DMatrix(matrix, feature_names=model.feature_list)
    raw = model.booster.predict(dmatrix)
    if raw.ndim == 1:
        # Binary objective: probability of class 1.
        if len(model.class_labels) != 2:
            raise ValueError(
                "Binary model needs exactly 2 class labels, "
                f"got {len(model.class_labels)}"
            )
        raw = np.column_stack([1.0 - raw, raw])
    if raw.shape[1] != len(model.class_labels):
        raise ValueError(
            f"Model outputs {raw.shape[1]} classes but class_labels has "
            f"{len(model.class_labels)}"
        )
    return raw


def predict(
    model: LoadedModel,
    X: pd.DataFrame,
    shap_top_k: int = 5,
    shap_max_rows: int = 2000,
    chunk_rows: int = 200_000,
) -> pd.DataFrame:
    """Score every row.

    Returns columns: ``predicted_class``, ``confidence``,
    ``class_probabilities`` (dict per row) and ``shap_top_features`` (list per
    row, filled only for the most confident non-benign rows, at most
    ``shap_max_rows``). Every row is always predicted.
    """
    matrix = prepare_matrix(model, X)
    parts = [
        _probabilities(model, matrix.iloc[start : start + chunk_rows])
        for start in range(0, len(matrix), chunk_rows)
    ]
    probabilities = (
        np.vstack(parts) if parts else np.empty((0, len(model.class_labels)))
    )
    labels = np.array(model.class_labels, dtype=object)
    class_index = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    result = pd.DataFrame(
        {
            "predicted_class": labels[class_index],
            "confidence": confidence.astype(float),
            "class_probabilities": [
                {
                    label: round(float(p), 6)
                    for label, p in zip(labels, row, strict=True)
                }
                for row in probabilities
            ],
        },
        index=X.index,
    )

    shap_values: list[list[dict]] = [[] for _ in range(len(result))]
    if shap_top_k > 0 and shap_max_rows > 0:
        alerts = np.flatnonzero(labels[class_index] != model.manifest.benign_label)
        if len(alerts):
            ranked = alerts[np.argsort(-confidence[alerts])][:shap_max_rows]
            explained = top_contributions(
                model.booster,
                matrix.iloc[ranked],
                class_index[ranked],
                top_k=shap_top_k,
            )
            for position, values in zip(ranked, explained, strict=True):
                shap_values[position] = values
    result["shap_top_features"] = shap_values
    return result


def compatibility_report(
    model: LoadedModel,
    X: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
) -> dict:
    """How closely an upload resembles the model's training data.

    ``status`` is ``warning`` when features the model relied on are absent,
    when many values fall outside the training range, or when the model is
    unsure about many rows. Accuracy on such files is not covered by the
    model's reported metrics.
    """
    stats = model.feature_stats
    missing_features: list[str] = []
    out_of_range: dict[str, float] = {}

    for name in model.feature_list:
        if name not in X.columns:
            missing_features.append(name)
            continue
        values = pd.to_numeric(X[name], errors="coerce")
        present = values.notna()
        info = stats.get(name)
        if not info:
            continue
        if not present.any():
            if info.get("missing_rate", 0.0) < 0.5:
                missing_features.append(name)
            continue
        low, high = info.get("p01"), info.get("p99")
        if low is None or high is None:
            continue
        margin = RANGE_MARGIN * max(high - low, 1e-9)
        observed = values[present]
        share = float(((observed < low - margin) | (observed > high + margin)).mean())
        if share > OUT_OF_RANGE_WARN_SHARE:
            out_of_range[name] = round(share, 3)

    report: dict = {
        "model": f"{model.manifest.lane_id}/{model.manifest.version}",
        "data_source": model.manifest.data_source,
        "features_missing": missing_features,
        "features_out_of_range": out_of_range,
    }
    reasons = []
    if missing_features:
        reasons.append(f"{len(missing_features)} feature(s) missing from upload")
    if out_of_range:
        reasons.append(
            f"{len(out_of_range)} feature(s) mostly outside the training range"
        )
    if predictions is not None and len(predictions):
        low_share = float((predictions["confidence"] < LOW_CONFIDENCE).mean())
        report["low_confidence_share"] = round(low_share, 3)
        if low_share > LOW_CONFIDENCE_WARN_SHARE:
            reasons.append(f"{low_share:.0%} of rows scored below {LOW_CONFIDENCE}")
    if not stats:
        reasons.append("model has no training feature statistics to compare against")
    report["status"] = "warning" if reasons else "ok"
    report["reasons"] = reasons
    return report
