"""Fixtures for the model-layer tests.

The tiny checkpoint is built once per session (it takes a moment to
serialise) and reused; tests that mutate a model load their own copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tiny_checkpoint as tc


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local directory holding a tiny Llama model and its tokenizer."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return tc.build_checkpoint(tmp_path_factory.mktemp("checkpoints") / "tiny")


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding only a tokenizer."""
    pytest.importorskip("transformers")
    return tc.build_tokenizer(tmp_path_factory.mktemp("tokenizers") / "tiny")


@pytest.fixture()
def tiny_model(tiny_checkpoint: Path):
    """A freshly loaded tiny model (mutable — each test gets its own)."""
    from repairllama.model.loader import LoadOptions, load_base_model

    return load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
    )
