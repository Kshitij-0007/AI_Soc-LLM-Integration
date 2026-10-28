from __future__ import annotations

from pathlib import Path

from app.core.lane_spec import LaneSpec
from app.core.training import train_and_save
from app.core.validator import validate


def main(data_path: str, lane_id: str = "cloud_entity") -> None:
    """Train lane model from data file.

    Usage: python -m app.lanes.cloud_entity.train --data data/raw/cloud_entity/valid_sample.csv
    """
    data_path = Path(data_path)

    # Parse the file
    from app.lanes.cloud_entity.parser import parse
    df = parse(data_path, "csv")

    # Load spec
    spec = LaneSpec(**load_spec(lane_id))

    # Validate
    validated = validate(df, spec)

    # Build features
    from app.lanes.cloud_entity.features import build_features
    ff = build_features(validated.clean_df, spec)

    # Train and save model
    train_and_save(ff, lane_id=lane_id, version="1.0.0")

    print(f"Model trained and saved for lane {lane_id}")


def load_spec(lane_id: str) -> dict:
    import yaml
    return yaml.safe_load(open(f"app/lanes/{lane_id}/spec.yaml"))
