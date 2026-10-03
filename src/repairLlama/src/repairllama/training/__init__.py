"""Supervised fine-tuning of the Java repair adapter.

    collator.py   IR4/OR2 rows -> masked batches (the prompt is not a target)
    metrics.py    validation loss, perplexity, token and sequence accuracy
    checkpoint.py resuming, and the final adapter artifact
    trainer.py    the loop that ties them together

Typical use::

    from repairllama.training import run_training

    result = run_training(cfg)
    result.artifact_dir        # adapters/java-repair/
    result.final_validation_loss

or from the command line::

    repairllama train
    python scripts/train_java_repair.py --epochs 2 --dry-run

The base model stays frozen and only the LoRA adapter is updated and saved;
see :mod:`repairllama.training.trainer` for the guarantees and for what
"matching the paper's hyper-parameters" does and does not mean.
"""

from repairllama.training.checkpoint import (
    CheckpointError,
    find_last_checkpoint,
    list_checkpoints,
    read_training_summary,
    resolve_resume,
    run_directory,
    save_training_artifact,
    write_training_summary,
)
from repairllama.training.collator import (
    IGNORE_INDEX,
    CollatorError,
    CompletionCollator,
    encode_example,
)
from repairllama.training.metrics import (
    EvalReport,
    compute_metrics,
    perplexity,
    preprocess_logits_for_metrics,
    summarise_history,
    token_accuracy,
)
from repairllama.training.trainer import (
    PAPER_SETTINGS,
    JsonlDataset,
    TrainingError,
    TrainingResult,
    build_training_arguments,
    load_split,
    print_training_banner,
    resolve_mixed_precision,
    run_training,
)

__all__ = [
    # trainer
    "run_training",
    "TrainingResult",
    "TrainingError",
    "JsonlDataset",
    "load_split",
    "build_training_arguments",
    "print_training_banner",
    "resolve_mixed_precision",
    "PAPER_SETTINGS",
    # collator
    "CompletionCollator",
    "CollatorError",
    "encode_example",
    "IGNORE_INDEX",
    # metrics
    "compute_metrics",
    "preprocess_logits_for_metrics",
    "perplexity",
    "token_accuracy",
    "summarise_history",
    "EvalReport",
    # checkpoint
    "CheckpointError",
    "list_checkpoints",
    "find_last_checkpoint",
    "resolve_resume",
    "run_directory",
    "save_training_artifact",
    "write_training_summary",
    "read_training_summary",
]
