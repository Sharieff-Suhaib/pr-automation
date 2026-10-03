"""The supervised fine-tuning loop: frozen CodeLlama + a LoRA adapter.

    repairllama train
    python scripts/train_java_repair.py --config configs/java_repair.yaml

The run is assembled from pieces that already exist and are already tested:
splits from ``prepare-data``, the frozen base from
:mod:`repairllama.model.loader`, the paper's adapter from
:mod:`repairllama.model.lora`, and masked batches from
:mod:`repairllama.training.collator`.  What this module adds is the loop, the
guarantees around it, and the record it leaves behind.

Guarantees
----------
*Only the adapter learns.*  The base is frozen at load time and again before
PEFT wraps it, and :func:`~repairllama.model.lora.assert_java_repair_lora` runs
before the first step: a run whose ``target_modules`` matched nothing fails in
the first second rather than after an hour of updating nothing.  A
``BaseWeightGuard`` fingerprints the base weights before training and verifies
them afterwards, so "the base did not change" is checked, not asserted.

*Only the adapter is saved.*  The final artifact in ``adapters/java-repair/``
holds adapter weights, the tokenizer and metadata — no optimizer state, no base
checkpoint, nothing merged.

*Reproducibility, as far as it goes.*  The seed covers Python, numpy and torch,
and every setting is recorded in the training summary.  Note that a GPU run is
not bit-for-bit reproducible even so: cuDNN kernel selection and reduction
order vary.

On reproducing the paper
------------------------
The defaults match the hyper-parameters the RepairLLaMA paper reports (2
epochs, AdamW at 5e-4, cosine schedule, 1024 tokens, LoRA r=8/alpha=16 on
q_proj and v_proj).  Matching those is not the same as reproducing the paper's
results, which also depend on the corpus, the batch size, the hardware and
details the paper does not report.  The training summary records what actually
ran so the difference is visible rather than assumed.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from repairllama.model.adapter import BaseWeightGuard, LoRASettings
from repairllama.model.device import format_bytes, format_count, memory_snapshot
from repairllama.model.loader import (
    LoadOptions,
    ModelError,
    ParameterCounts,
    count_parameters,
    load_base_model,
)
from repairllama.model.lora import (
    JAVA_REPAIR,
    assert_java_repair_lora,
    attach_java_repair_adapter,
    check_java_repair_lora,
)
from repairllama.training.checkpoint import (
    RunPaths,
    resolve_resume,
    run_directory,
    save_training_artifact,
)
from repairllama.training.collator import CompletionCollator
from repairllama.training.metrics import (
    compute_metrics,
    preprocess_logits_for_metrics,
    summarise_history,
)
from repairllama.utils.io import read_jsonl
from repairllama.utils.logging import get_logger, log_section
from repairllama.utils.seed import set_seed

__all__ = [
    "TrainingError",
    "JsonlDataset",
    "TrainingResult",
    "load_split",
    "build_training_arguments",
    "print_training_banner",
    "run_training",
]

log = get_logger("training.trainer")

SPLIT_FILES = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}


class TrainingError(RuntimeError):
    """Raised when a training run cannot be set up or completed."""


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class JsonlDataset:
    """A split held in memory as raw ``{"input", "output"}`` rows.

    Tokenization happens in the collator, so batches are padded to their own
    longest sequence rather than to ``max_length``.  Implemented as a plain
    sequence rather than a ``datasets.Dataset`` to keep the dependency out.
    """

    def __init__(self, rows: Sequence[Dict[str, Any]], name: str = "") -> None:
        self.rows = list(rows)
        self.name = name

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def token_estimate(self, tokenizer: Any) -> int:  # pragma: no cover - reporting
        return sum(
            len(tokenizer.encode(row["input"], add_special_tokens=False))
            + len(tokenizer.encode(row["output"], add_special_tokens=False))
            for row in self.rows
        )


def load_split(
    splits_dir: Union[str, Path], split: str, *, required: bool = True
) -> Optional[JsonlDataset]:
    """Load ``<splits_dir>/<split>.jsonl`` produced by ``prepare-data``."""
    if split not in SPLIT_FILES:
        raise TrainingError(f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    path = Path(splits_dir).expanduser() / SPLIT_FILES[split]
    if not path.is_file():
        if required:
            raise TrainingError(
                f"no {split} split at {path}.\n"
                "Build the dataset first:  repairllama prepare-data --input <corpus>"
            )
        return None

    rows = []
    for index, row in enumerate(read_jsonl(path)):
        if "input" not in row or "output" not in row:
            raise TrainingError(
                f"{path}:{index + 1}: a training row needs 'input' and 'output'; "
                f"got {sorted(row)}"
            )
        rows.append(row)
    if required and not rows:
        raise TrainingError(f"{path} is empty; there is nothing to train on")
    log.info("loaded %d %s example(s) from %s", len(rows), split, path)
    return JsonlDataset(rows, name=split)


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def resolve_mixed_precision(setting: str, device: str) -> Tuple[bool, bool]:
    """Return ``(fp16, bf16)`` flags for ``TrainingArguments``.

    ``auto`` means bf16 on CUDA that supports it, fp16 on CUDA that does not,
    and full precision anywhere else — mixed precision on CPU is slower, not
    faster, and on Apple Metal it is not supported by the trainer.
    """
    import torch

    is_cuda = device.startswith("cuda")
    if setting == "no":
        return (False, False)
    if setting == "fp16":
        if not is_cuda:
            log.warning(
                "mixed_precision=fp16 was requested on %s; the trainer only "
                "supports it on CUDA, continuing in full precision",
                device,
            )
            return (False, False)
        return (True, False)
    if setting == "bf16":
        if not is_cuda:
            log.warning(
                "mixed_precision=bf16 was requested on %s; continuing in full "
                "precision",
                device,
            )
            return (False, False)
        return (False, True)

    # auto
    if not is_cuda:
        return (False, False)
    supported = getattr(torch.cuda, "is_bf16_supported", None)
    if supported is not None and supported():
        return (False, True)
    return (True, False)


def build_training_arguments(
    cfg: Any,
    run_dir: Path,
    *,
    device: str,
    dataset_size: int,
    has_eval_dataset: bool = True,
) -> Any:
    """Map the config onto ``transformers.TrainingArguments``.

    ``has_eval_dataset=False`` forces evaluation off: the Trainer refuses to
    start with an eval strategy set but no dataset to evaluate on, and a
    missing validation split should degrade to "train without validation
    loss", not to a crash.
    """
    from transformers import TrainingArguments

    training = cfg.training
    fp16, bf16 = resolve_mixed_precision(training.mixed_precision, device)

    kwargs: Dict[str, Any] = {
        "output_dir": str(run_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": training.epochs,
        "per_device_train_batch_size": training.batch_size,
        "per_device_eval_batch_size": training.eval_batch,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "warmup_ratio": training.warmup_ratio,
        "lr_scheduler_type": training.lr_scheduler,
        "max_grad_norm": training.max_grad_norm,
        "optim": training.optimizer,
        "logging_steps": training.logging_steps,
        "save_strategy": training.save_strategy,
        "save_total_limit": training.save_total_limit,
        "seed": training.seed,
        "data_seed": training.seed,
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": training.gradient_checkpointing,
        "group_by_length": training.group_by_length,
        "dataloader_num_workers": training.dataloader_num_workers,
        "report_to": [] if training.report_to == "none" else [training.report_to],
        "load_best_model_at_end": training.load_best_model_at_end,
        "remove_unused_columns": False,  # rows are dicts the collator reads
        "label_names": ["labels"],
    }
    if training.max_steps:
        kwargs["max_steps"] = training.max_steps
    if training.save_strategy == "steps":
        kwargs["save_steps"] = training.save_steps
    if training.eval_strategy != "no" and has_eval_dataset:
        kwargs["eval_strategy"] = training.eval_strategy
        if training.eval_strategy == "steps":
            kwargs["eval_steps"] = training.eval_steps
        if training.load_best_model_at_end:
            kwargs["metric_for_best_model"] = "eval_loss"
            kwargs["greater_is_better"] = False
    else:
        kwargs["eval_strategy"] = "no"

    if training.gradient_checkpointing:
        # Required with PEFT: without it the checkpointed graph has no input
        # that requires grad and the backward pass fails.
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if device == "cpu":
        # The Trainer otherwise picks an accelerator on its own (MPS on a Mac),
        # silently ignoring model.device: cpu.
        kwargs["use_cpu"] = True

    return _construct_training_arguments(TrainingArguments, kwargs)


def _construct_training_arguments(cls: Any, kwargs: Dict[str, Any]) -> Any:
    """Build TrainingArguments, dropping keys this version does not accept."""
    import inspect

    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        accepted = set(kwargs)
    if "eval_strategy" in kwargs and "eval_strategy" not in accepted:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    unknown = [key for key in kwargs if key not in accepted and key != "kwargs"]
    for key in unknown:
        log.debug("this transformers version ignores TrainingArguments.%s", key)
        kwargs.pop(key)
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# the banner
# --------------------------------------------------------------------------- #
def print_training_banner(
    *,
    counts: ParameterCounts,
    settings: LoRASettings,
    dataset_sizes: Dict[str, int],
    max_length: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    epochs: int,
    device: str,
    dtype: str,
    base_model: str,
    printer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Print what is about to run, and return it for the summary."""
    emit = printer or (lambda line: log.info("%s", line))
    percentage = counts.trainable_fraction * 100

    emit("=" * 72)
    emit("RepairLLaMA Java LoRA fine-tuning")
    emit("=" * 72)
    emit(f"Base model:            {base_model}")
    emit(f"Device / dtype:        {device} / {dtype}")
    emit(f"Base parameters:       {counts.frozen:,} ({format_count(counts.frozen)}, frozen)")
    emit(f"Trainable parameters:  {counts.trainable:,} ({format_count(counts.trainable)})")
    emit(f"Trainable percentage:  {percentage:.4f}%")
    emit(
        "Dataset size:          "
        + ", ".join(f"{name} {size:,}" for name, size in dataset_sizes.items())
    )
    emit(f"Max sequence length:   {max_length} tokens")
    emit(
        f"Batch size:            {batch_size} x {gradient_accumulation} accumulation "
        f"= {batch_size * gradient_accumulation} effective"
    )
    emit(f"Learning rate:         {learning_rate:g}")
    emit(f"Epochs:                {epochs}")
    emit(
        "LoRA configuration:    "
        f"r={settings.r}, alpha={settings.alpha}, dropout={settings.dropout}, "
        f"targets={list(settings.target_modules)}, bias={settings.bias}"
    )
    emit("=" * 72)

    return {
        "base_model": base_model,
        "device": device,
        "dtype": dtype,
        "base_parameters": counts.frozen,
        "trainable_parameters": counts.trainable,
        "total_parameters": counts.total,
        "trainable_percentage": round(percentage, 6),
        "dataset_size": dict(dataset_sizes),
        "max_sequence_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "lora": settings.to_dict(),
    }


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
@dataclass
class TrainingResult:
    """Everything a finished run produced."""

    artifact_dir: Optional[Path] = None
    run_dir: Optional[Path] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    tokenizer: Any = None

    @property
    def final_train_loss(self) -> Optional[float]:
        return self.summary.get("history", {}).get("final_train_loss")

    @property
    def final_validation_loss(self) -> Optional[float]:
        return self.summary.get("history", {}).get("final_validation_loss")


def run_training(
    cfg: Any,
    *,
    splits_dir: Optional[Union[str, Path]] = None,
    adapters_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    resume: Optional[str] = None,
    dry_run: bool = False,
    max_examples: int = 0,
) -> TrainingResult:
    """Fine-tune the Java repair adapter and write the artifact.

    Args:
        cfg: a :class:`repairllama.config.RepairConfig`.
        splits_dir: where ``train.jsonl``/``validation.jsonl`` live.
        adapters_dir: parent of ``java-repair/``; the artifact goes there.
        resume: ``auto`` for the newest checkpoint, or a checkpoint path.
        dry_run: set everything up, print the banner, then stop before the
            first optimizer step — the cheap way to check a configuration.
        max_examples: cap on training rows, for smoke runs.

    Returns:
        A :class:`TrainingResult` with the artifact directory and the summary.
    """
    started = time.time()
    try:
        import torch
        import transformers
        from transformers import Trainer
    except ImportError as exc:
        raise TrainingError(
            "torch, transformers and peft are required to train. Install the "
            'model stack with `pip install -e ".[train]"`.'
        ) from exc

    training = cfg.training
    set_seed(training.seed)
    log_section(log, f"training '{cfg.experiment_name}' (seed {training.seed})")

    splits_path = Path(splits_dir) if splits_dir else cfg.paths.resolve("splits_dir")
    adapters_path = Path(adapters_dir) if adapters_dir else cfg.paths.resolve("adapters_dir")
    outputs_path = Path(output_dir) if output_dir else cfg.paths.resolve("output_dir")
    run_dir = run_directory(outputs_path, cfg.experiment_name)
    paths = RunPaths(run_dir=run_dir, adapters_dir=adapters_path).prepare()

    # -- data ---------------------------------------------------------------- #
    train_dataset = load_split(splits_path, "train")
    if max_examples:
        train_dataset = JsonlDataset(train_dataset.rows[:max_examples], "train")
    eval_dataset = (
        load_split(splits_path, "validation", required=False)
        if training.eval_strategy != "no"
        else None
    )
    if eval_dataset is not None and not len(eval_dataset):
        log.warning("the validation split is empty; evaluation is disabled")
        eval_dataset = None
    if training.eval_strategy != "no" and eval_dataset is None:
        log.warning(
            "no validation split found in %s; training without validation loss",
            splits_path,
        )

    # -- model --------------------------------------------------------------- #
    loaded = load_base_model(LoadOptions.from_config(cfg))
    base_counts = count_parameters(loaded.model)
    guard = BaseWeightGuard.capture(loaded.model)

    settings = LoRASettings.from_config(cfg.model.lora)
    model = attach_java_repair_adapter(loaded.model, settings, guard=guard)
    assert_java_repair_lora(model, settings)
    counts = count_parameters(model)

    if training.gradient_checkpointing:
        model.enable_input_require_grads()

    collator = CompletionCollator(
        tokenizer=loaded.tokenizer,
        max_length=training.max_length,
        truncation="left",
    )

    dataset_sizes = {"train": len(train_dataset)}
    if eval_dataset is not None:
        dataset_sizes["validation"] = len(eval_dataset)

    banner = print_training_banner(
        counts=counts,
        settings=settings,
        dataset_sizes=dataset_sizes,
        max_length=training.max_length,
        batch_size=training.batch_size,
        gradient_accumulation=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        epochs=training.epochs,
        device=loaded.actual_device(),
        dtype=loaded.actual_dtype(),
        base_model=loaded.source.reference,
    )

    if dry_run:
        log.info("dry run: stopping before the first optimizer step")
        summary = _build_summary(
            cfg=cfg,
            banner=banner,
            paths=paths,
            base_counts=base_counts,
            counts=counts,
            collator=collator,
            splits_dir=splits_path,
            history={},
            metrics={},
            duration=time.time() - started,
            resumed_from=None,
            dry_run=True,
        )
        return TrainingResult(
            artifact_dir=None,
            run_dir=run_dir,
            summary=summary,
            model=model,
            tokenizer=loaded.tokenizer,
        )

    # -- the trainer ---------------------------------------------------------- #
    arguments = build_training_arguments(
        cfg,
        run_dir,
        device=loaded.spec.device,
        dataset_size=len(train_dataset),
        has_eval_dataset=eval_dataset is not None,
    )
    callbacks = []
    if training.early_stopping_patience and eval_dataset is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=training.early_stopping_patience)
        )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": callbacks,
    }
    if eval_dataset is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics
        trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics
    trainer = _construct_trainer(Trainer, trainer_kwargs, loaded.tokenizer)

    resume_from = resolve_resume(
        resume if resume is not None else training.resume_from_checkpoint, run_dir
    )

    log.info("starting training (%s)", transformers.__version__)
    train_output = trainer.train(resume_from_checkpoint=resume_from)
    metrics = dict(train_output.metrics or {})

    # -- validation ----------------------------------------------------------- #
    if eval_dataset is not None:
        from repairllama.training.metrics import EvalReport

        evaluation = trainer.evaluate()
        metrics.update(evaluation)
        log.info("final validation: %s", EvalReport.from_metrics(evaluation).render())

    # -- the base must not have moved ----------------------------------------- #
    changed = guard.changed(model)
    if changed:
        raise TrainingError(
            "base weights changed during training, which must never happen with "
            "LoRA: " + ", ".join(changed[:5])
        )
    log.info("verified: the base model is unchanged after training")

    # -- the artifact ---------------------------------------------------------- #
    history = summarise_history(trainer.state.log_history)
    if history.get("final_train_loss") is None and "train_loss" in metrics:
        # A run shorter than logging_steps logs no loss line; the trainer's
        # own averaged train_loss is still the honest number to report.
        history["final_train_loss"] = metrics["train_loss"]
    summary = _build_summary(
        cfg=cfg,
        banner=banner,
        paths=paths,
        base_counts=base_counts,
        counts=counts,
        collator=collator,
        splits_dir=splits_path,
        history=history,
        metrics=metrics,
        duration=time.time() - started,
        resumed_from=resume_from,
        dry_run=False,
    )
    artifact = save_training_artifact(
        model,
        loaded.tokenizer,
        adapters_path,
        language="java",
        base_model=loaded.source.reference,
        metadata={
            "experiment": cfg.experiment_name,
            "epochs": training.epochs,
            "learning_rate": training.learning_rate,
            "train_examples": len(train_dataset),
        },
        summary=summary,
        guard=guard,
    )

    log_section(log, "training complete")
    log.info("adapter artifact: %s", artifact)
    log.info(
        "final train loss: %s | final validation loss: %s",
        history.get("final_train_loss"),
        history.get("final_validation_loss"),
    )
    return TrainingResult(
        artifact_dir=artifact,
        run_dir=run_dir,
        summary=summary,
        metrics=metrics,
        model=model,
        tokenizer=loaded.tokenizer,
    )


def _construct_trainer(cls: Any, kwargs: Dict[str, Any], tokenizer: Any) -> Any:
    """Pass the tokenizer under the name this transformers version expects."""
    import inspect

    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        accepted = set()
    if "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer
    return cls(**kwargs)


def _build_summary(
    *,
    cfg: Any,
    banner: Dict[str, Any],
    paths: RunPaths,
    base_counts: ParameterCounts,
    counts: ParameterCounts,
    collator: CompletionCollator,
    splits_dir: Path,
    history: Dict[str, Any],
    metrics: Dict[str, Any],
    duration: float,
    resumed_from: Optional[str],
    dry_run: bool,
) -> Dict[str, Any]:
    """Assemble ``training_summary.json``."""
    import torch
    import transformers

    training = cfg.training
    return {
        "experiment": cfg.experiment_name,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 2),
        "dry_run": dry_run,
        "resumed_from": resumed_from,
        "run": banner,
        "parameters": {
            "base_total": base_counts.total,
            "base_frozen": base_counts.frozen,
            "trainable": counts.trainable,
            "trainable_percentage": round(counts.trainable_fraction * 100, 6),
            "adapter_only": True,
            "base_modified": False,
        },
        "settings": {
            "training": _dataclass_to_dict(training),
            "lora": _dataclass_to_dict(cfg.model.lora),
            "representation": {
                "input_format": cfg.representation.input_format,
                "output_format": cfg.representation.output_format,
                "fill_token": cfg.representation.fill_token,
            },
        },
        "data": {
            "splits_dir": str(splits_dir),
            **collator.stats(),
        },
        "history": history,
        "metrics": {key: value for key, value in metrics.items() if _is_scalar(value)},
        "paths": paths.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": _package_version("peft"),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "reproduction": {
            "matches_paper_reported_hyperparameters": _matches_paper(cfg),
            "verified_reproduction_of_paper_results": False,
            "note": (
                "The defaults match the hyper-parameters the RepairLLaMA paper "
                "reports. Matching them does not by itself reproduce the paper's "
                "results, which also depend on the training corpus, the effective "
                "batch size, the hardware and details the paper does not report. "
                "This field records the settings that actually ran; it is not a "
                "claim of equivalence."
            ),
        },
    }


PAPER_SETTINGS = {
    "learning_rate": 5e-4,
    "lr_scheduler": "cosine",
    "epochs": 2,
    "optimizer": "adamw_torch",
    "max_length": 1024,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
}


def _matches_paper(cfg: Any) -> Dict[str, Any]:
    """Which reported hyper-parameters this run actually used."""
    training = cfg.training
    lora = cfg.model.lora
    actual = {
        "learning_rate": training.learning_rate,
        "lr_scheduler": training.lr_scheduler,
        "epochs": training.epochs,
        "optimizer": training.optimizer,
        "max_length": training.max_length,
        "lora_r": lora.r,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "target_modules": list(lora.target_modules),
    }
    differences = {
        key: {"paper": value, "used": actual[key]}
        for key, value in PAPER_SETTINGS.items()
        if actual[key] != value
    }
    return {
        "all_match": not differences,
        "used": actual,
        "differences": differences,
    }


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    return asdict(obj) if is_dataclass(obj) else dict(obj)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - informational only
        return None
