"""Fixtures for the training tests.

Reuses the tiny local checkpoint built for the model-layer tests, so training
is exercised against a real (very small) Llama rather than a mock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))

import tiny_checkpoint as tc  # noqa: E402


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return tc.build_checkpoint(tmp_path_factory.mktemp("train-checkpoints") / "tiny")


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_checkpoint: Path):
    """A tokenizer with a pad token and the fill marker, as training uses it."""
    from repairllama.model.tokenizer import load_tokenizer

    tokenizer, _ = load_tokenizer(str(tiny_checkpoint), extra_tokens=("<FILL_ME>",))
    return tokenizer


def _rows(count: int) -> list[dict]:
    return [
        {
            "input": (
                f"public class Variant{index} {{\n"
                f"    public int value(int a) {{\n"
                "        // buggy lines start here\n"
                f"        // return a - {index};\n"
                "        // buggy lines end here\n"
                "        <FILL_ME>\n    }\n}"
            ),
            "output": f"        return a + {index};",
            "bug_id": f"bug-{index:03d}",
            "suspicious_start": 3,
            "suspicious_end": 3,
            "project": f"project-{index % 4}",
        }
        for index in range(count)
    ]


@pytest.fixture()
def splits_dir(tmp_path: Path) -> Path:
    """A splits directory shaped like the one `prepare-data` writes."""
    target = tmp_path / "splits"
    target.mkdir(parents=True, exist_ok=True)
    for name, count in (("train", 8), ("validation", 2), ("test", 2)):
        with (target / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in _rows(count):
                handle.write(json.dumps(row) + "\n")
    return target


@pytest.fixture()
def training_config(tiny_checkpoint: Path, tmp_path: Path):
    """A config wired to the tiny model and a short, CPU-only run."""
    from repairllama.config import RepairConfig

    return RepairConfig.from_dict(
        {
            "experiment_name": "test-run",
            "paths": {"project_root": str(tmp_path)},
            "model": {
                "base_model": str(tiny_checkpoint),
                "device": "cpu",
                "dtype": "float32",
            },
            "training": {
                "max_steps": 2,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_length": 128,
                "mixed_precision": "no",
                "gradient_checkpointing": False,
                "eval_strategy": "no",
                "save_strategy": "no",
                "logging_steps": 1,
            },
        }
    )
