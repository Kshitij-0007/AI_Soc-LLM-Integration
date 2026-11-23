#!/usr/bin/env python3
"""Turn an externally trained XGBoost model into a standard model folder.

    # booster saved with save_model (.json / .ubj), feature names inside:
    python scripts/convert_xgb_model.py --input ext/model.json --lane siem_edr \
        --version 1.1.0 --classes BENIGN "Credential Access" Execution

    # scikit-learn XGBClassifier pickled with joblib/pickle (trusted files only):
    python scripts/convert_xgb_model.py --input ext/clf.joblib --trust-pickle \
        --lane netflow --version 3.0.0 --classes-file ext/classes.json \
        --feature-list ext/features.json --missing-values native

Writes models/<lane>/<version>/ with model.json, feature_list.json,
class_labels.json and manifest.json, then run:

    python scripts/register_models.py <lane> <version> --check-only

Class order must match the model's output columns (label-encoded order used
in training). Pickle files execute code when loaded: only convert files you
produced or fully trust; the app itself never loads pickles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import xgboost as xgb

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.core.model_manifest import ModelManifest, model_version_dir, write_manifest  # noqa: E402


def _load_booster(path: Path, trust_pickle: bool) -> xgb.Booster:
    suffix = path.suffix.lower()
    if suffix in {".json", ".ubj"}:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    if suffix in {".pkl", ".pickle", ".joblib"}:
        if not trust_pickle:
            raise SystemExit(
                "Refusing to unpickle without --trust-pickle (pickles run code)."
            )
        import joblib

        obj = joblib.load(path)
        if isinstance(obj, xgb.Booster):
            return obj
        if hasattr(obj, "get_booster"):
            return obj.get_booster()
        raise SystemExit(f"Unsupported object in pickle: {type(obj).__name__}")
    raise SystemExit(f"Unsupported model file type: {suffix}")


def _read_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        return [str(x) for x in json.loads(path.read_text(encoding="utf-8"))]
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--classes-file", type=Path, default=None)
    parser.add_argument("--feature-list", type=Path, default=None)
    parser.add_argument(
        "--missing-values", choices=["native", "median", "zero"], default="native",
        help="How the model saw missing values during training.",
    )
    parser.add_argument("--benign-label", default="BENIGN")
    parser.add_argument("--feature-set-version", default=None)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--trust-pickle", action="store_true")
    args = parser.parse_args()

    booster = _load_booster(args.input, args.trust_pickle)
    features = _read_list(args.feature_list) or booster.feature_names
    if not features:
        raise SystemExit("No feature names in the model; pass --feature-list.")
    classes = args.classes or _read_list(args.classes_file)
    if not classes:
        raise SystemExit("Pass --classes or --classes-file (model output order).")
    if args.benign_label not in classes:
        raise SystemExit(f"Class list needs '{args.benign_label}'.")

    out_dir = model_version_dir(PROJECT_DIR / "models", args.lane, args.version)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and is not empty.")
    out_dir.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(out_dir / "model.json"))
    (out_dir / "feature_list.json").write_text(json.dumps(features, indent=2))
    (out_dir / "class_labels.json").write_text(json.dumps(classes, indent=2))
    config = json.loads(booster.save_config())
    objective = config["learner"]["objective"]["name"]
    write_manifest(
        out_dir,
        ModelManifest(
            lane_id=args.lane,
            version=args.version,
            model_format="xgboost_json",
            model_file="model.json",
            train_stats_file=None,
            metrics_file=None,
            missing_values=args.missing_values,
            benign_label=args.benign_label,
            objective=objective,
            feature_set_version=args.feature_set_version,
            data_source=args.data_source,
            notes=[f"converted from {args.input.name}"],
        ),
    )
    print(f"Wrote {out_dir} ({len(classes)} classes, {len(features)} features, "
          f"objective {objective})")
    print(f"Next: python scripts/register_models.py {args.lane} {args.version} "
          "--check-only")


if __name__ == "__main__":
    main()
