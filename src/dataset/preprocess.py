"""Turn RepairExamples into the text records the SFT trainer consumes.

Kept deliberately thin: loading lives in loader.py, prompt formatting lives in
representation/repair_prompt.py, and this module only wires them together and
(optionally) hands the result to `datasets.Dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.dataset.loader import RepairExample, load_examples, train_eval_split
from src.representation.repair_prompt import (
    build_messages,
    build_target,
    format_for_inference,
    format_for_training,
)
from src.training.config import DataConfig


def example_to_record(example: RepairExample, tokenizer=None) -> dict[str, Any]:
    """One training record.

    `text` is what SFTTrainer trains on. `prompt`/`completion` are kept alongside
    it for evaluation and debugging (they let us measure how much of `text` is
    prompt versus target without re-rendering).
    """
    return {
        "id": example.id,
        "text": format_for_training(example, tokenizer),
        "prompt": format_for_inference(
            issue=example.issue,
            buggy_code=example.buggy_code,
            fault_location=example.fault_location,
            extras=example.extras,
            tokenizer=tokenizer,
        ),
        "completion": build_target(example.fixed_code),
        "messages": build_messages(example, include_answer=True),
    }


def build_records(examples: list[RepairExample], tokenizer=None) -> list[dict[str, Any]]:
    return [example_to_record(example, tokenizer) for example in examples]


def prepare_datasets(config: DataConfig, tokenizer=None, verbose: bool = True):
    """Load, split, and format the dataset.

    Returns `(train_records, eval_records, train_examples, eval_examples)`.
    The raw examples are returned too because evaluation needs the original
    `fixed_code` for comparison, not just the rendered text.
    """
    train_examples = load_examples(config.train_file, verbose=verbose)

    if config.eval_file is not None:
        eval_examples = load_examples(config.eval_file, verbose=verbose)
    else:
        train_examples, eval_examples = train_eval_split(
            train_examples, config.eval_split, config.seed
        )

    if verbose:
        print(f"[preprocess] train={len(train_examples)}  eval={len(eval_examples)}")

    return (
        build_records(train_examples, tokenizer),
        build_records(eval_examples, tokenizer),
        train_examples,
        eval_examples,
    )


def to_hf_dataset(records: list[dict[str, Any]]):
    """Wrap records in a `datasets.Dataset` (only `text` is needed for SFT)."""
    from datasets import Dataset

    return Dataset.from_list([{"text": r["text"]} for r in records])


def save_processed(records: list[dict[str, Any]], path: Path) -> Path:
    """Persist formatted records to JSONL for inspection or reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
