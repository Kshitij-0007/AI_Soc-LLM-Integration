from __future__ import annotations

from pathlib import Path

import yaml

from app.db.repositories.taxonomy import upsert_taxonomy_map


def load_taxonomy_yaml(lane_id: str, lanes_dir: Path = Path(__file__).resolve().parents[2] / "app" / "lanes") -> dict:
    """Load taxonomy.yaml for a given lane.

    Returns the taxonomy dict from the yaml file.
    """
    taxonomy_path = lanes_dir / lane_id / "taxonomy.yaml"
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy not found at {taxonomy_path}")
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def upsert_lane_taxonomy(session, lane_id: str, taxonomy_data: dict) -> None:
    """Upsert taxonomy entries for a lane into taxonomy_map.

    Each entry has: native_label -> {unified_class, tactic, weight}
    """
    entries = taxonomy_data.get("classes", {})
    for native_label, info in entries.items():
        unified_class = info.get("unified_class")
        tactic = info.get("tactic")
        weight = info.get("weight", 0.0)
        upsert_taxonomy_map(session, lane_id, [
            {
                "native_label": native_label,
                "unified_class": unified_class,
                "tactic": tactic,
                "weight": weight,
            }
        ])


def map_prediction_to_taxonomy(
    predicted_class: str,
    lane_id: str,
    taxonomy_data: dict,
) -> dict | None:
    """Map a native predicted class to unified_class and tactic.

    Returns dict with unified_class and tactic, or None if not found.
    """
    classes = taxonomy_data.get("classes", {})
    if predicted_class in classes:
        info = classes[predicted_class]
        return {
            "unified_class": info.get("unified_class"),
            "tactic": info.get("tactic"),
        }
    # Try case-insensitive match
    for native_label, info in classes.items():
        if native_label.lower() == predicted_class.lower():
            return {
                "unified_class": info.get("unified_class"),
                "tactic": info.get("tactic"),
            }
    return None
