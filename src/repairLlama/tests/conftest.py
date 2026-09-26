"""Shared pytest fixtures.

``src`` is added to ``sys.path`` here as well as in pyproject so the suite
runs from a plain checkout without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repairllama.config import RepairConfig  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_path(project_root: Path) -> Path:
    return project_root / "configs" / "java_repair.yaml"


@pytest.fixture()
def default_config() -> RepairConfig:
    return RepairConfig.default()


@pytest.fixture()
def shipped_config(config_path: Path) -> RepairConfig:
    return RepairConfig.from_yaml(config_path)
