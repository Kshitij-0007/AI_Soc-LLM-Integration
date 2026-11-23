"""Model manifest: how to load and feed one model version folder.

Every ``models/<lane_id>/<version>/`` folder may contain ``manifest.json``.
The predictor reads it to know the file format, the feature order, the class
names and how missing values were handled during training. This is what
lets a model trained outside this repository (for example a scikit-learn
``XGBClassifier`` saved with ``save_model("model.json")``) be dropped in
without code changes.

Pickle / joblib files are deliberately not loadable: unpickling runs
arbitrary code and breaks across library versions. Convert them once with
``scripts/convert_xgb_model.py``.

Folders without a manifest (models trained before manifests existed) get one
inferred: model.json or model.ubj, median imputation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MANIFEST_FILE = "manifest.json"
MODEL_FORMATS = {".json": "xgboost_json", ".ubj": "xgboost_ubj"}
NETFLOW_FOLDER = "netflow_model"


def model_version_dir(models_dir: Path, lane_id: str, version: str) -> Path:
    folder = NETFLOW_FOLDER if lane_id == "netflow" else lane_id
    return models_dir / folder / version


class ModelManifest(BaseModel):
    manifest_version: int = 1
    lane_id: str
    version: str
    model_format: Literal["xgboost_json", "xgboost_ubj"] = "xgboost_json"
    model_file: str = "model.json"
    feature_list_file: str = "feature_list.json"
    class_labels_file: str = "class_labels.json"
    train_stats_file: str | None = "train_stats.json"
    metrics_file: str | None = "metrics.json"
    # native: pass NaN to XGBoost. median: fill with train_stats medians.
    # zero: fill with 0.
    missing_values: Literal["native", "median", "zero"] = "native"
    benign_label: str = "BENIGN"
    objective: str | None = None
    # Version of the lane's features.py the model was trained with. The
    # predictor refuses to score when the lane produces different features.
    feature_set_version: str | None = None
    data_source: str | None = None
    trained_at: str | None = None
    notes: list[str] = Field(default_factory=list)


def _infer_manifest(model_dir: Path) -> ModelManifest:
    model_file = next(
        (name for name in ("model.json", "model.ubj") if (model_dir / name).is_file()),
        None,
    )
    if model_file is None:
        raise FileNotFoundError(f"No model.json or model.ubj in {model_dir}")
    folder = model_dir.parent.name
    return ModelManifest(
        lane_id="netflow" if folder == NETFLOW_FOLDER else folder,
        version=model_dir.name,
        model_format=MODEL_FORMATS[Path(model_file).suffix],
        model_file=model_file,
        missing_values="median",
        notes=["manifest inferred: folder has no manifest.json"],
    )


def load_manifest(model_dir: Path) -> ModelManifest:
    path = model_dir / MANIFEST_FILE
    if not path.is_file():
        return _infer_manifest(model_dir)
    return ModelManifest(**json.loads(path.read_text(encoding="utf-8")))


def write_manifest(model_dir: Path, manifest: ModelManifest) -> Path:
    path = model_dir / MANIFEST_FILE
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path
