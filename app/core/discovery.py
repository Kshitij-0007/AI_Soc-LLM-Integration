from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from app.core.lane_spec import LaneSpec

PROJECT_DIR = Path(__file__).resolve().parents[2]  # up from app/core to ai_soc
LANES_DIR = PROJECT_DIR / "app" / "lanes"


def discover_lanes(lanes_dir: Path = LANES_DIR) -> dict[str, LaneSpec]:
    """Scan lanes_dir for spec.yaml files, load them as LaneSpec models.

    Skips folders starting with '_'.
    Returns dict of lane_id -> LaneSpec.
    """
    lanes: dict[str, LaneSpec] = {}
    if not lanes_dir.exists():
        return lanes

    for entry in sorted(lanes_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_"):
            continue  # skip template/folders

        spec_path = entry / "spec.yaml"
        if not spec_path.exists():
            continue

        with open(spec_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        try:
            spec = LaneSpec(**data)
            lanes[spec.lane_id] = spec
        except Exception as e:
            print(f"Warning: Failed to load spec from {spec_path}: {e}")

    return lanes


def load_lane_module(lane_id: str, module_path: Path) -> object:
    """Dynamically import a lane module from a given path."""
    spec = importlib.util.spec_from_file_location(lane_id, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parser(lane_id: str, lane_dir: Path) -> callable:
    """Load the parser.parse function for a lane.

    Returns a function: parse(path: Path, fmt: str) -> pd.DataFrame
    """
    parser_path = lane_dir / "parser.py"
    if not parser_path.exists():
        raise FileNotFoundError(f"Parser not found at {parser_path}")
    module = load_lane_module(lane_id, parser_path)
    return module.parse


def load_features(lane_id: str, lane_dir: Path) -> callable:
    """Load the build_features function for a lane.

    Returns a function: build_features(df: pd.DataFrame, spec: LaneSpec) -> FeatureFrame
    """
    features_path = lane_dir / "features.py"
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found at {features_path}")
    module = load_lane_module(lane_id, features_path)
    return module.build_features


def load_build_features(lane_id: str, lane_dir: Path | None = None) -> callable:
    """Load the build_features function for a lane.

    Returns a function: build_features(df: pd.DataFrame, spec: LaneSpec) -> FeatureFrame
    """
    if lane_dir is None:
        lane_dir = LANES_DIR / lane_id
    return load_features(lane_id, lane_dir)
