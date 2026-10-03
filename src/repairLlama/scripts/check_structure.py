#!/usr/bin/env python3
"""Smoke check: verify the project layout and that every package imports.

Runs without pytest and without the model stack installed:

    python scripts/check_structure.py

Exits non-zero (and lists what is wrong) if anything is missing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_DIRS = [
    "models",
    "adapters/java-repair",
    "data/raw",
    "data/processed",
    "data/splits",
    "src/repairllama",
    "tests/data",
    "tests/model",
    "tests/representation",
    "tests/patching",
    "tests/inference",
    "configs",
    "scripts",
]

REQUIRED_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    ".gitignore",
    "configs/java_repair.yaml",
    "scripts/train_java_repair.py",
]

MODULES = [
    "repairllama",
    "repairllama.cli",
    "repairllama.config",
    "repairllama.data",
    "repairllama.data.models",
    "repairllama.data.loader",
    "repairllama.data.cleaner",
    "repairllama.data.deduplicator",
    "repairllama.data.tokenizer_filter",
    "repairllama.data.dataset_builder",
    "repairllama.representation",
    "repairllama.representation.ir4",
    "repairllama.representation.or2",
    "repairllama.representation.builder",
    "repairllama.localization",
    "repairllama.model",
    "repairllama.model.device",
    "repairllama.model.tokenizer",
    "repairllama.model.loader",
    "repairllama.model.adapter",
    "repairllama.model.lora",
    "repairllama.training",
    "repairllama.training.collator",
    "repairllama.training.metrics",
    "repairllama.training.checkpoint",
    "repairllama.training.trainer",
    "repairllama.inference",
    "repairllama.patching",
    "repairllama.evaluation",
    "repairllama.utils",
    "repairllama.utils.io",
    "repairllama.utils.logging",
    "repairllama.utils.paths",
    "repairllama.utils.seed",
]


def main() -> int:
    problems: list[str] = []

    for relative in REQUIRED_DIRS:
        if not (ROOT / relative).is_dir():
            problems.append(f"missing directory: {relative}")
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            problems.append(f"missing file: {relative}")

    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            problems.append(f"import failed: {name} ({exc.__class__.__name__}: {exc})")

    try:
        from repairllama.config import RepairConfig

        RepairConfig.from_yaml(ROOT / "configs" / "java_repair.yaml")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"configs/java_repair.yaml did not validate: {exc}")

    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"OK — {len(REQUIRED_DIRS)} directories, {len(REQUIRED_FILES)} files, "
          f"{len(MODULES)} modules, config validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
