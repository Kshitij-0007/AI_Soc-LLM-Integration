#!/usr/bin/env python3
"""Check a model folder and register it in model_versions.

    python scripts/register_models.py netflow 2.0.0 --activate
    python scripts/register_models.py netflow 2.0.0 --check-only   # no DB needed

Works for models trained here and for XGBoost models trained elsewhere, as
long as ``models/<lane>/<version>/`` holds the model file, feature list and
class labels (see app/core/model_manifest.py). Checks before writing:

* the model loads and its output width matches class_labels
* every class has an entry in the lane's taxonomy.yaml
* the lane's features.py produces every feature the model needs
* the manifest feature_set_version matches the lane's FEATURE_SET_VERSION
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.core import predictor  # noqa: E402
from app.core.model_manifest import model_version_dir  # noqa: E402
from app.core.taxonomy import load_taxonomy_yaml  # noqa: E402


def check_model(lane_id: str, version: str) -> tuple[predictor.LoadedModel, list[str]]:
    """Return the loaded model and a list of problems (empty = OK)."""
    model_dir = model_version_dir(PROJECT_DIR / "models", lane_id, version)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    model = predictor.load_model(model_dir)
    problems: list[str] = []

    if model.manifest.lane_id != lane_id:
        problems.append(
            f"manifest lane_id is '{model.manifest.lane_id}', expected '{lane_id}'"
        )

    taxonomy = load_taxonomy_yaml(lane_id).get("classes", {})
    lower = {k.lower() for k in taxonomy}
    missing_taxonomy = [c for c in model.class_labels if c.lower() not in lower]
    if missing_taxonomy:
        problems.append(f"classes without taxonomy.yaml entry: {missing_taxonomy}")
    if model.manifest.benign_label not in model.class_labels:
        problems.append(f"no class named '{model.manifest.benign_label}'")

    try:
        features = importlib.import_module(f"app.lanes.{lane_id}.features")
    except ImportError:
        features = None
    lane_columns = getattr(features, "FEATURE_COLUMNS", None)
    if lane_columns is not None:
        absent = [f for f in model.feature_list if f not in lane_columns]
        if absent:
            problems.append(f"lane features.py does not produce: {absent}")
    lane_set = getattr(features, "FEATURE_SET_VERSION", None)
    if model.manifest.feature_set_version and lane_set:
        if model.manifest.feature_set_version != lane_set:
            problems.append(
                f"model feature set {model.manifest.feature_set_version} != "
                f"lane feature set {lane_set}"
            )
    elif lane_set and not model.manifest.feature_set_version:
        problems.append(
            "model has no feature_set_version (legacy or external model); "
            "confirm its features match the lane before activating"
        )

    # A smoke prediction proves the booster, feature list and labels agree.
    probe = pd.DataFrame(
        np.zeros((2, len(model.feature_list))), columns=model.feature_list
    )
    try:
        predictor.predict(model, probe, shap_top_k=0)
    except Exception as error:  # noqa: BLE001
        problems.append(f"test prediction failed: {error}")

    return model, problems


def register(lane_id: str, version: str, activate: bool, force: bool) -> None:
    from app.db.repositories.model_versions import (
        activate_model,
        create_model_version,
        get_model_by_version,
    )
    from app.db.session import SessionLocal

    model, problems = check_model(lane_id, version)
    for problem in problems:
        print(f"  ! {problem}")
    if problems and not force:
        raise SystemExit("Refusing to register; fix the problems or pass --force.")

    relative_path = str(model.model_dir.relative_to(PROJECT_DIR).as_posix())
    session = SessionLocal()
    try:
        row = get_model_by_version(session, lane_id, version)
        if row is None:
            row = create_model_version(
                session,
                lane_id=lane_id,
                version=version,
                model_path=relative_path,
                class_labels=model.class_labels,
                feature_list=model.feature_list,
                train_stats={
                    k: v for k, v in model.train_stats.items() if k != "feature_stats"
                },
                metrics=model.metrics,
                data_source=model.manifest.data_source,
            )
            print(f"Registered {lane_id}/{version} as {row.model_id}")
        else:
            row.model_path = relative_path
            row.class_labels = model.class_labels
            row.feature_list = model.feature_list
            row.metrics = model.metrics
            row.data_source = model.manifest.data_source
            print(f"Updated existing {lane_id}/{version} ({row.model_id})")
        if activate:
            activate_model(session, str(row.model_id))
            print(f"Activated {lane_id}/{version}")
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lane_id")
    parser.add_argument("version")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true", help="Run checks without the DB."
    )
    parser.add_argument(
        "--force", action="store_true", help="Register despite check problems."
    )
    args = parser.parse_args()

    if args.check_only:
        model, problems = check_model(args.lane_id, args.version)
        print(
            f"{args.lane_id}/{args.version}: {len(model.class_labels)} classes, "
            f"{len(model.feature_list)} features, format "
            f"{model.manifest.model_format}, missing values "
            f"{model.manifest.missing_values}"
        )
        for problem in problems:
            print(f"  ! {problem}")
        print("OK" if not problems else "PROBLEMS FOUND")
        raise SystemExit(1 if problems else 0)

    register(args.lane_id, args.version, args.activate, args.force)


if __name__ == "__main__":
    main()
