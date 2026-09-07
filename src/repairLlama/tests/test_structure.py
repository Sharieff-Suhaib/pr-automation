"""Structural checks: the layout and every package import."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SUBPACKAGES = [
    "repairllama.data",
    "repairllama.representation",
    "repairllama.localization",
    "repairllama.model",
    "repairllama.training",
    "repairllama.inference",
    "repairllama.patching",
    "repairllama.evaluation",
    "repairllama.utils",
]

EXPECTED_DIRS = [
    "models",
    "adapters/java-repair",
    "data/raw",
    "data/processed",
    "data/splits",
    "src/repairllama",
    "tests",
    "configs",
    "scripts",
]

EXPECTED_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    ".gitignore",
    "configs/java_repair.yaml",
]


@pytest.mark.parametrize("relative", EXPECTED_DIRS)
def test_expected_directories_exist(project_root: Path, relative: str) -> None:
    assert (project_root / relative).is_dir(), f"missing directory: {relative}"


@pytest.mark.parametrize("relative", EXPECTED_FILES)
def test_expected_files_exist(project_root: Path, relative: str) -> None:
    assert (project_root / relative).is_file(), f"missing file: {relative}"


def test_root_package_imports() -> None:
    module = importlib.import_module("repairllama")
    assert module.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} should document its responsibility"


def test_importing_package_does_not_pull_in_torch() -> None:
    """The config/CLI layer must stay usable without the training stack."""
    code = (
        "import sys; import repairllama, repairllama.cli; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1] / "src"),
    )
    assert result.returncode == 0, result.stderr
