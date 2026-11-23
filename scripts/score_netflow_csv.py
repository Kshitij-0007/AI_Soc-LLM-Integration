#!/usr/bin/env python3
"""Score a canonical NetFlow CSV using the project's saved XGBoost lane model.

Run from the project root:
  python scripts/score_netflow_csv.py --data data/raw/netflow/cicids2017_benign_smoke_sample.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.lane_spec import LaneSpec
from app.core.validator import validate
from app.lanes.netflow.features import build_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a NetFlow CSV with the saved lane model.")
    parser.add_argument("--data", type=Path, required=True, help="Canonical NetFlow CSV to score.")
    parser.add_argument("--version", default="1.0.0", help="Model version directory (default: 1.0.0).")
    parser.add_argument("--output", type=Path, default=None, help="Output prediction CSV path.")
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / "models" / "netflow_model" / args.version
    required_artifacts = ["model.json", "feature_list.json", "class_labels.json", "train_stats.json"]
    missing = [name for name in required_artifacts if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}: {missing}")

    data_path = args.data if args.data.is_absolute() else (Path.cwd() / args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    raw = pd.read_csv(data_path)
    spec_data = yaml.safe_load((PROJECT_ROOT / "app/lanes/netflow/spec.yaml").read_text(encoding="utf-8"))
    spec = LaneSpec(**spec_data)
    validated = validate(raw, spec)
    if validated.status == "rejected":
        raise ValueError(f"CSV rejected; missing required columns: {validated.missing_required}")

    feature_frame = build_features(validated.clean_df, spec)
    feature_names = json.loads((model_dir / "feature_list.json").read_text(encoding="utf-8"))
    class_labels = json.loads((model_dir / "class_labels.json").read_text(encoding="utf-8"))
    stats = json.loads((model_dir / "train_stats.json").read_text(encoding="utf-8"))

    X = feature_frame.X.reindex(columns=feature_names)
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = pd.Series(stats.get("medians", {}), dtype=float).reindex(feature_names)
    X = X.fillna(medians).fillna(0.0)

    model = xgb.Booster()
    model.load_model(str(model_dir / "model.json"))
    probabilities = model.predict(xgb.DMatrix(X, feature_names=feature_names))
    pred_indices = probabilities.argmax(axis=1)

    results = raw.copy().reset_index(drop=True)
    results["predicted_class"] = [class_labels[i] for i in pred_indices]
    results["confidence"] = probabilities.max(axis=1)
    for index, label in enumerate(class_labels):
        safe_label = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        results[f"probability_{safe_label}"] = probabilities[:, index]

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / f"{data_path.stem}_predictions.csv"
    elif not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Rows scored: {len(results)}")
    print("Predicted class counts:")
    print(results["predicted_class"].value_counts().to_string())
    print(f"Predictions saved to: {output_path}")

    if "label" in raw.columns:
        known_actual = raw["label"].astype(str).str.strip()
        if known_actual.nunique() < 2:
            print("NOTE: This sample has only one ground-truth class; it is a smoke test, not a meaningful multiclass evaluation.")
        else:
            from sklearn.metrics import classification_report, confusion_matrix
            print("\nClassification report (only classes represented in the model can be scored correctly):")
            print(classification_report(known_actual, results["predicted_class"], zero_division=0))
            print("Confusion matrix labels:", class_labels)
            print(confusion_matrix(known_actual, results["predicted_class"], labels=class_labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
