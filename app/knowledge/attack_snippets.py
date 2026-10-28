from __future__ import annotations

import yaml
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(PROJECT_DIR, "knowledge", "attack_snippets.yaml")


def load_yaml() -> dict:
    """Load attack snippets from YAML file."""
    with open(YAML_PATH, "r") as f:
        return yaml.safe_load(f)


def get_remedies() -> dict[str, dict]:
    """Return all remedy templates grouped by tactic."""
    data = load_yaml()
    return data


def get_remedy(tactic: str) -> dict | None:
    """Get remedy template for a specific tactic."""
    data = load_yaml()
    return data.get(tactic)