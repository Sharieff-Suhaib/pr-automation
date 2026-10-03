"""Path helpers for locating the project root and its data directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

__all__ = ["project_root", "ensure_dir", "find_upwards", "relative_to_root"]

# Marker files that identify the repairllama-java project root.
_ROOT_MARKERS = ("pyproject.toml", "configs/java_repair.yaml")


def find_upwards(start: Path, names: Iterable[str]) -> Path | None:
    """Walk up from ``start`` looking for the first directory holding a marker."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if any((candidate / name).exists() for name in names):
            return candidate
    return None


def project_root() -> Path:
    """Best-effort project root.

    Honours ``REPAIRLLAMA_ROOT`` when set, otherwise walks up from this file
    looking for ``pyproject.toml``.  Falls back to two levels above the package.
    """
    override = os.environ.get("REPAIRLLAMA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    found = find_upwards(Path(__file__).parent, _ROOT_MARKERS)
    if found is not None:
        return found
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def relative_to_root(path: str | os.PathLike[str]) -> Path:
    """Render ``path`` relative to the project root when it lives inside it."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root())
    except ValueError:
        return resolved


def default_config_path() -> Path:
    """Location of the shipped ``configs/java_repair.yaml``."""
    return project_root() / "configs" / "java_repair.yaml"
