"""Shared fixtures."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG = REPO_ROOT / "configs" / "roman_gbtds_demo.yaml"


@pytest.fixture(scope="session")
def demo_config_path() -> Path:
    return DEMO_CONFIG


@pytest.fixture
def demo_dict() -> dict[str, Any]:
    """A fresh, mutable copy of the demo configuration document."""
    return copy.deepcopy(yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8")))
