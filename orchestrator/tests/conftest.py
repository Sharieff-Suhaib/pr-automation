"""Shared fixtures for the orchestrator tests."""

from __future__ import annotations

import pytest

from src.agents.coding_agent import fault_localization


@pytest.fixture(autouse=True)
def heuristic_fault_localization(monkeypatch):
    """No test reaches a live Ollama for fault localization: the remote repair
    sites come from the heuristics, as when Ollama is down. Tests of the
    Ollama localizer switch it back on and fake its reply."""
    monkeypatch.setattr(fault_localization, "LOCALIZER", "heuristic")
