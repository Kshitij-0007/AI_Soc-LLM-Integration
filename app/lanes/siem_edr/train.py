from __future__ import annotations

from pathlib import Path

from app.core.lane_spec import LaneSpec
from app.core.training import train_and_save
from app.core.validator import validate


def main(data_path: str, lane_id: str = "siem_edr") -> None:
    """Train lane model from data file.

    Usage: python -m app.lanes.siem_edr.train --data data/raw/siem_edr/valid_sample.csv
    """
    data_path = Path(data_path)

    # Parse the file
    from app.lanes.siem_edr.parser import parse
    df = parse(data_path, "csv")

    # Load spec
    spec = LaneSpec(**load_spec(lane_id))

    # Validate
    validated = validate(df, spec)

    # Build features
    from app.lanes.siem_edr.features import build_features
    ff = build_features(validated.clean_df, spec)

    # Train and save model
    train_and_save(ff, lane_id=lane_id, version="1.0.0")

    print(f"Model trained and saved for lane {lane_id}")


def load_spec(lane_id: str) -> dict:
    import yaml
    return yaml.safe_load(open(f"app/lanes/{lane_id}/spec.yaml"))
