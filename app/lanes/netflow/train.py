from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from app.core.lane_spec import LaneSpec
from app.core.training import train_and_save
from app.core.validator import validate


def load_spec(lane_id: str) -> dict:
    """Load the configuration for the requested lane."""
    spec_path = Path("app") / "lanes" / lane_id / "spec.yaml"

    if not spec_path.is_file():
        raise FileNotFoundError(
            f"Lane specification not found: {spec_path}"
        )

    with spec_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid lane specification: {spec_path}")

    return config


def main(data_path: Path, lane_id: str = "netflow") -> None:
    """Validate data, build features, train, and save the lane model."""

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}"
        )

    print(f"[1/5] Loading dataset: {data_path}", flush=True)

    from app.lanes.netflow.parser import parse

    df = parse(data_path, "csv")
    print(f"      Loaded {len(df)} rows and {len(df.columns)} columns.", flush=True)

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("[2/5] Loading lane specification...", flush=True)

    spec = LaneSpec(**load_spec(lane_id))

    print("[3/5] Validating dataset...", flush=True)

    validated = validate(df, spec)

    print("[4/5] Building features...", flush=True)

    from app.lanes.netflow.features import build_features

    feature_frame = build_features(validated.clean_df, spec)

    print(
        f"      Feature matrix shape: {feature_frame.X.shape}",
        flush=True,
    )

    if feature_frame.X.empty:
        raise ValueError("Feature extraction produced an empty matrix.")

    if feature_frame.y is None:
        raise ValueError(
            "No training labels were generated. "
            "Check label_column in spec.yaml and the CSV label column."
        )

    if len(feature_frame.X) != len(feature_frame.y):
        raise ValueError(
            "Feature/label row-count mismatch: "
            f"{len(feature_frame.X)} features versus "
            f"{len(feature_frame.y)} labels."
        )

    print("[5/5] Training and saving model...", flush=True)

    train_and_save(
        feature_frame,
        lane_id=lane_id,
        version="1.0.0",
    )

    print(
        f"Training completed for lane '{lane_id}'.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for the NetFlow SOC lane."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the input CSV dataset.",
    )

    parser.add_argument(
        "--lane",
        default="netflow",
        help="Lane identifier (default: netflow).",
    )

    args = parser.parse_args()

    main(
        data_path=args.data,
        lane_id=args.lane,
    )