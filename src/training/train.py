"""QLoRA fine-tuning entry point for the program-repair model.

Run:
    python -m src.training.train
    AGENT_SWE_MAX_STEPS=2 python -m src.training.train      # tiny smoke run

Only the LoRA adapter is trained; the base model stays frozen (and quantized to
4-bit when a CUDA GPU with bitsandbytes is available).
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from src.dataset.preprocess import prepare_datasets, save_processed, to_hf_dataset
from src.training.config import Config, load_config
from src.training.model_loader import (
    apply_lora,
    detect_device,
    load_base_model,
    load_tokenizer,
)


def _supported_kwargs(cls, candidate: dict) -> dict:
    """Keep only kwargs the installed TRL/transformers version accepts.

    TRL renames SFT arguments fairly often (`max_seq_length` -> `max_length`,
    `evaluation_strategy` -> `eval_strategy`, ...). Filtering by the actual
    signature keeps this script working across versions instead of crashing on
    an unexpected keyword.
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return candidate
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidate

    accepted: dict = {}
    dropped: list[str] = []
    for key, value in candidate.items():
        if key in params:
            accepted[key] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"[train] ignoring unsupported trainer args for this version: {', '.join(dropped)}")
    return accepted


def build_sft_config(config: Config, has_eval: bool):
    """Assemble the TRL SFTConfig from our Config, tolerating API drift."""
    from trl import SFTConfig

    train_cfg, model_cfg = config.train, config.model
    candidate = {
        "output_dir": str(train_cfg.output_dir),
        "num_train_epochs": train_cfg.num_train_epochs,
        "max_steps": train_cfg.max_steps,
        "learning_rate": train_cfg.learning_rate,
        "per_device_train_batch_size": train_cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "warmup_ratio": train_cfg.warmup_ratio,
        "weight_decay": train_cfg.weight_decay,
        "lr_scheduler_type": train_cfg.lr_scheduler_type,
        "logging_steps": train_cfg.logging_steps,
        "save_strategy": train_cfg.save_strategy,
        "gradient_checkpointing": train_cfg.gradient_checkpointing,
        "seed": train_cfg.seed,
        "report_to": "none",  # no wandb/tensorboard by default
        "dataset_text_field": "text",
        "packing": False,  # one bug-fix example per sequence
        # Version-dependent aliases; the filter keeps whichever exists.
        "max_length": model_cfg.max_seq_length,
        "max_seq_length": model_cfg.max_seq_length,
        "eval_strategy": "epoch" if has_eval else "no",
        "evaluation_strategy": "epoch" if has_eval else "no",
    }

    device = detect_device()
    if device == "cuda":
        import torch

        # Mixed precision only pays off on CUDA; MPS/CPU keep full precision.
        candidate["bf16"] = torch.cuda.is_bf16_supported()
        candidate["fp16"] = not torch.cuda.is_bf16_supported()
        # Paged optimizer keeps optimizer state off the GPU during spikes.
        candidate["optim"] = "paged_adamw_8bit"

    return SFTConfig(**_supported_kwargs(SFTConfig, candidate))


def save_artifacts(config: Config, trainer, tokenizer, peft_config) -> Path:
    """Persist the LoRA adapter, tokenizer, and the exact config used."""
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(output_dir))  # adapter weights only
    tokenizer.save_pretrained(str(output_dir))

    payload = config.to_dict()
    payload["resolved"] = {
        "device": detect_device(),
        "lora_target_modules": list(peft_config.target_modules),
    }
    with open(output_dir / "agent_swe_config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[train] saved adapter + tokenizer + config to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for program repair")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data and build the trainer, but do not run training.",
    )
    args = parser.parse_args()

    config = load_config()

    # 1. Tokenizer first: the chat template shapes the training text.
    tokenizer = load_tokenizer(config.model)

    # 2. Data -> repair-specific prompts -> HF datasets.
    train_records, eval_records, _, _ = prepare_datasets(config.data, tokenizer)
    save_processed(train_records, config.data.processed_dir / "train.jsonl")
    if eval_records:
        save_processed(eval_records, config.data.processed_dir / "eval.jsonl")

    train_dataset = to_hf_dataset(train_records)
    eval_dataset = to_hf_dataset(eval_records) if eval_records else None

    # 3. Base model (4-bit when supported) + LoRA adapters.
    model = load_base_model(config.model, for_training=True)
    model, peft_config = apply_lora(
        model, config.lora, gradient_checkpointing=config.train.gradient_checkpointing
    )

    # 4. Supervised fine-tuning on the rendered `text` field.
    from trl import SFTTrainer

    sft_config = build_sft_config(config, has_eval=eval_dataset is not None)
    trainer_kwargs = _supported_kwargs(
        SFTTrainer,
        {
            "model": model,
            "args": sft_config,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,  # older TRL name; filtered out if absent
        },
    )
    # Never pass both tokenizer spellings.
    if "processing_class" in trainer_kwargs:
        trainer_kwargs.pop("tokenizer", None)
    trainer = SFTTrainer(**trainer_kwargs)

    if args.dry_run:
        print("[train] --dry-run: trainer built successfully, skipping training.")
        return

    trainer.train()
    save_artifacts(config, trainer, tokenizer, peft_config)


if __name__ == "__main__":
    main()
