from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

LANES_DIR = PROJECT_DIR / "app" / "lanes"


@pytest.fixture
def netflow_spec():
    from app.lanes.netflow.train import load_spec

    return load_spec()


@pytest.fixture
def netflow_samples() -> Path:
    return LANES_DIR / "netflow" / "samples"
