"""Lightweight smoke test for the program-repair pipeline.

Verifies, in order:

    dataset -> preprocessing -> prompt construction
            -> tokenization -> model loading -> one forward/backward pass

Stages that need heavy dependencies are skipped (not failed) when those
dependencies are missing, so the data half can be checked on any machine.

Run:
    python scripts/test_pipeline.py            # data stages only, no model download
    python scripts/test_pipeline.py --full     # also downloads the model + one training step
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/test_pipeline.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import FaultLocation, load_examples  # noqa: E402
from src.dataset.preprocess import prepare_datasets  # noqa: E402
from src.representation.repair_prompt import (  # noqa: E402
    annotate_fault_lines,
    format_for_inference,
)
from src.training.config import load_config  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def record(stage: str, status: str, detail: str = "") -> None:
    _results.append((stage, status, detail))
    symbol = {PASS: "[ok]", FAIL: "[!!]", SKIP: "[--]"}[status]
    print(f"{symbol} {stage}" + (f" — {detail}" if detail else ""))


def stage_dataset(config):
    examples = load_examples(config.data.train_file)
    assert examples, "no examples loaded"
    assert all(e.buggy_code and e.fixed_code for e in examples), "empty code field"
    record("dataset", PASS, f"{len(examples)} examples")
    return examples


def stage_preprocess(config):
    train_records, eval_records, _, eval_examples = prepare_datasets(config.data, verbose=False)
    assert train_records, "empty training set"
    assert eval_records, "empty holdout — lower AGENT_SWE_EVAL_SPLIT or add examples"
    record("preprocess", PASS, f"train={len(train_records)} eval={len(eval_records)}")
    return train_records, eval_examples


def stage_prompt(train_records):
    text = train_records[0]["text"]
    for marker in ("### TASK", "### BUGGY CODE", "### FAULT LOCATION", "### FIX"):
        assert marker in text, f"missing {marker} in rendered prompt"

    # Fault annotation must actually mark the reported line.
    annotated = annotate_fault_lines("a\nb\nc", FaultLocation(start_line=2, end_line=2))
    assert ">>> 2 | b" in annotated, f"fault marker not applied:\n{annotated}"

    # Optional fields must degrade gracefully.
    no_context = format_for_inference(issue=None, buggy_code="x = 1", fault_location=None)
    assert "### ISSUE" in no_context and "### FIX" in no_context
    record("prompt", PASS, f"{len(text)} chars, fault markers + missing-field fallback ok")


def stage_tokenize(config, train_records):
    try:
        from src.training.model_loader import load_tokenizer
    except ImportError as exc:
        record("tokenize", SKIP, f"transformers/torch not installed ({exc.name})")
        return None
    try:
        tokenizer = load_tokenizer(config.model)
    except Exception as exc:
        record("tokenize", FAIL, f"{type(exc).__name__}: {exc}")
        return None

    # Re-render with the real chat template and check it fits the context window.
    from src.dataset.preprocess import example_to_record  # local import: needs transformers
    from src.dataset.loader import load_examples as _load

    example = _load(config.data.train_file, verbose=False)[0]
    text = example_to_record(example, tokenizer)["text"]
    token_ids = tokenizer(text)["input_ids"]
    assert len(token_ids) > 0, "tokenizer produced no tokens"
    fits = len(token_ids) <= config.model.max_seq_length
    detail = f"{len(token_ids)} tokens (max_seq_len={config.model.max_seq_length})"
    record("tokenize", PASS if fits else FAIL, detail if fits else detail + " — TRUNCATION RISK")
    return tokenizer


def stage_model_and_step(config):
    """Load the model, attach LoRA, and run one forward+backward pass."""
    try:
        import torch
        from src.training.model_loader import apply_lora, detect_device, load_base_model
    except ImportError as exc:
        record("model+step", SKIP, f"missing dependency ({exc.name})")
        return
    try:
        tokenizer_module = __import__("src.training.model_loader", fromlist=["load_tokenizer"])
        tokenizer = tokenizer_module.load_tokenizer(config.model)
        model = load_base_model(config.model, for_training=True)
        model, _ = apply_lora(model, config.lora, gradient_checkpointing=False)

        from src.dataset.preprocess import example_to_record
        from src.dataset.loader import load_examples as _load

        example = _load(config.data.train_file, verbose=False)[0]
        text = example_to_record(example, tokenizer)["text"]

        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.model.max_seq_length,
        )
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"].clone()

        outputs = model(**batch)
        loss = outputs.loss
        assert torch.isfinite(loss), f"non-finite loss: {loss}"
        loss.backward()  # proves gradients reach the LoRA parameters

        trainable = [p for p in model.parameters() if p.requires_grad]
        with_grad = sum(1 for p in trainable if p.grad is not None)
        assert with_grad > 0, "no LoRA parameter received a gradient"
        record(
            "model+step",
            PASS,
            f"device={detect_device()} loss={loss.item():.4f} "
            f"grads on {with_grad}/{len(trainable)} LoRA tensors",
        )
    except Exception as exc:
        record("model+step", FAIL, f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also load the model and run one training step (downloads ~3GB on first run).",
    )
    args = parser.parse_args()

    config = load_config()
    print(f"base model : {config.model.base_model}")
    print(f"dataset    : {config.data.train_file}\n")

    stage_dataset(config)
    train_records, _ = stage_preprocess(config)
    stage_prompt(train_records)

    if args.full:
        stage_tokenize(config, train_records)
        stage_model_and_step(config)
    else:
        record("tokenize", SKIP, "run with --full")
        record("model+step", SKIP, "run with --full")

    failures = [stage for stage, status, _ in _results if status == FAIL]
    print("\n" + "=" * 60)
    print(f"{len(_results) - len(failures)}/{len(_results)} stages ok" + (
        f" — FAILED: {', '.join(failures)}" if failures else ""
    ))
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
