"""Placeholder suite for :mod:`repairllama.patching`.

The stage is not implemented yet.  These checks keep the test tree honest —
the package must exist and stay importable — and record what this directory
will cover once the stage lands:

      - applying a hunk to a Java file at the right offset
      - re-indenting generated code to its insertion site
      - unified-diff rendering and original-file backup
"""

from __future__ import annotations

import importlib

import pytest


def test_package_is_importable() -> None:
    module = importlib.import_module("repairllama.patching")
    assert module.__doc__


@pytest.mark.skip(reason="stage not implemented yet")
def test_stage_behaviour() -> None:  # pragma: no cover - placeholder
    raise AssertionError("replace with real cases when repairllama.patching lands")
