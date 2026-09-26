"""Validation metrics for supervised fine-tuning.

Validation loss is what a training run is steered by, and perplexity is that
loss made readable (``exp(loss)``: roughly, how many equally-likely tokens the
model is choosing between at each step).  Both are reported per evaluation.

On top of those, :func:`compute_metrics` reports two things loss alone hides:

``token_accuracy``
    The fraction of *completion* tokens predicted correctly under teacher
    forcing.  Prompt positions are masked out (label ``-100``), so this
    measures the patch, not the ability to echo the buggy code back.

``sequence_accuracy``
    The fraction of examples where every completion token is right — an
    optimistic proxy for exact-match patches, since a whole hunk had to be
    predicted correctly.  It is not the paper's plausible/correct rate: that
    requires compiling and running the tests, which the evaluation stage does.

Logits for a 32k-token vocabulary over a 1024-token batch are hundreds of MB,
so :func:`preprocess_logits_for_metrics` reduces them to argmax token ids on
the GPU before they are gathered — without it, evaluation runs out of memory
on any real model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from repairllama.training.collator import IGNORE_INDEX
from repairllama.utils.logging import get_logger

__all__ = [
    "perplexity",
    "preprocess_logits_for_metrics",
    "compute_metrics",
    "token_accuracy",
    "EvalReport",
    "summarise_history",
]

log = get_logger("training.metrics")

#: Losses above this are reported as infinite perplexity rather than overflowing.
_MAX_LOSS_FOR_PPL = 60.0


def perplexity(loss: Optional[float]) -> Optional[float]:
    """``exp(loss)``, guarding against overflow and missing values."""
    if loss is None:
        return None
    if loss > _MAX_LOSS_FOR_PPL:
        return float("inf")
    try:
        return float(math.exp(loss))
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return float("inf")


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
    """Reduce logits to predicted token ids before they are accumulated.

    Passed to ``Trainer(preprocess_logits_for_metrics=...)``.  Without it the
    trainer keeps the full ``(batch, sequence, vocab)`` tensor for every
    evaluation batch, which is gigabytes on a 7B model.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def token_accuracy(predictions: Any, labels: Any) -> Tuple[int, int, int, int]:
    """Count correct/total completion tokens and fully-correct sequences.

    Returns ``(correct_tokens, total_tokens, correct_sequences, sequences)``.
    Predictions are shifted against labels the way a causal LM scores them:
    position ``i`` predicts token ``i + 1``.
    """
    import numpy as np

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.ndim == 1:  # a single example
        predictions = predictions[None, :]
        labels = labels[None, :]

    shifted_predictions = predictions[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != IGNORE_INDEX

    correct = ((shifted_predictions == shifted_labels) & mask).sum()
    total = mask.sum()

    per_row_total = mask.sum(axis=1)
    per_row_correct = ((shifted_predictions == shifted_labels) & mask).sum(axis=1)
    scored = per_row_total > 0
    exact = ((per_row_correct == per_row_total) & scored).sum()
    return (int(correct), int(total), int(exact), int(scored.sum()))


def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
    """``compute_metrics`` for ``transformers.Trainer``.

    Expects predictions already reduced to token ids by
    :func:`preprocess_logits_for_metrics`.
    """
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    correct, total, exact, sequences = token_accuracy(predictions, labels)
    return {
        "token_accuracy": (correct / total) if total else 0.0,
        "sequence_accuracy": (exact / sequences) if sequences else 0.0,
        "completion_tokens": float(total),
    }


@dataclass
class EvalReport:
    """One evaluation's numbers, ready for the training summary."""

    loss: Optional[float] = None
    perplexity: Optional[float] = None
    token_accuracy: Optional[float] = None
    sequence_accuracy: Optional[float] = None
    step: Optional[int] = None
    epoch: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metrics(cls, metrics: Dict[str, Any]) -> "EvalReport":
        """Build from a ``trainer.evaluate()`` dictionary."""
        loss = metrics.get("eval_loss")
        known = {
            "eval_loss",
            "eval_token_accuracy",
            "eval_sequence_accuracy",
            "eval_completion_tokens",
            "epoch",
            "step",
        }
        return cls(
            loss=loss,
            perplexity=perplexity(loss),
            token_accuracy=metrics.get("eval_token_accuracy"),
            sequence_accuracy=metrics.get("eval_sequence_accuracy"),
            step=metrics.get("step"),
            epoch=metrics.get("epoch"),
            extras={k: v for k, v in metrics.items() if k not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "token_accuracy": self.token_accuracy,
            "sequence_accuracy": self.sequence_accuracy,
            "step": self.step,
            "epoch": self.epoch,
        }
        return {key: value for key, value in payload.items() if value is not None}

    def render(self) -> str:
        parts = []
        if self.loss is not None:
            parts.append(f"loss {self.loss:.4f}")
        if self.perplexity is not None:
            parts.append(f"ppl {self.perplexity:.2f}")
        if self.token_accuracy is not None:
            parts.append(f"token acc {self.token_accuracy:.2%}")
        if self.sequence_accuracy is not None:
            parts.append(f"exact hunks {self.sequence_accuracy:.2%}")
        return " | ".join(parts) or "(no metrics)"


def summarise_history(history: Any) -> Dict[str, Any]:
    """Condense ``trainer.state.log_history`` into what a summary needs.

    Keeps the training-loss curve, every evaluation, and the best validation
    loss with the step it was reached at.
    """
    train_losses = []
    evaluations = []
    for entry in history or []:
        if "loss" in entry and "eval_loss" not in entry:
            train_losses.append(
                {
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "learning_rate": entry.get("learning_rate"),
                }
            )
        if "eval_loss" in entry:
            report = EvalReport.from_metrics(entry)
            report.step = entry.get("step")
            evaluations.append(report.to_dict())

    best = min(evaluations, key=lambda item: item["loss"], default=None) if evaluations else None
    return {
        "train_loss_curve": train_losses,
        "evaluations": evaluations,
        "best_validation": best,
        "final_train_loss": train_losses[-1]["loss"] if train_losses else None,
        "final_validation_loss": evaluations[-1]["loss"] if evaluations else None,
    }
