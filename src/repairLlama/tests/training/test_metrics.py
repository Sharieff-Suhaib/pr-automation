"""Validation metrics: perplexity, token/sequence accuracy, history."""

from __future__ import annotations

import math

import pytest

from repairllama.training.collator import IGNORE_INDEX
from repairllama.training.metrics import (
    EvalReport,
    compute_metrics,
    perplexity,
    preprocess_logits_for_metrics,
    summarise_history,
    token_accuracy,
)

pytest.importorskip("numpy")


class _Prediction:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


# --------------------------------------------------------------------------- #
# perplexity
# --------------------------------------------------------------------------- #
def test_perplexity_of_a_known_loss() -> None:
    assert perplexity(0.0) == 1.0
    assert perplexity(1.0) == pytest.approx(math.e)


def test_perplexity_of_nothing() -> None:
    assert perplexity(None) is None


def test_perplexity_does_not_overflow() -> None:
    assert perplexity(1e6) == float("inf")


# --------------------------------------------------------------------------- #
# token accuracy
# --------------------------------------------------------------------------- #
def test_a_perfect_prediction_scores_one() -> None:
    labels = [[IGNORE_INDEX, IGNORE_INDEX, 5, 6, 7]]
    # position i predicts token i+1, so shift the predictions left
    predictions = [[0, 5, 6, 7, 0]]
    correct, total, exact, sequences = token_accuracy(predictions, labels)
    assert (correct, total) == (3, 3)
    assert (exact, sequences) == (1, 1)


def test_prompt_positions_are_not_scored() -> None:
    """Accuracy measures the patch, not the ability to echo the prompt back."""
    labels = [[IGNORE_INDEX, IGNORE_INDEX, 5, 6]]
    # position i predicts token i+1: wrong on the prompt, right on the completion
    predictions = [[99, 5, 6, 0]]
    correct, total, _, _ = token_accuracy(predictions, labels)
    assert (correct, total) == (2, 2)


def test_a_partly_correct_sequence_is_not_exact() -> None:
    labels = [[IGNORE_INDEX, 5, 6, 7]]
    predictions = [[5, 6, 99, 0]]
    correct, total, exact, sequences = token_accuracy(predictions, labels)
    assert (correct, total) == (2, 3)
    assert (exact, sequences) == (0, 1)


def test_accuracy_across_a_batch() -> None:
    labels = [[IGNORE_INDEX, 5, 6], [IGNORE_INDEX, 7, 8]]
    predictions = [[5, 6, 0], [7, 99, 0]]
    correct, total, exact, sequences = token_accuracy(predictions, labels)
    assert (correct, total) == (3, 4)
    assert (exact, sequences) == (1, 2)


def test_a_fully_masked_row_is_not_counted() -> None:
    labels = [[IGNORE_INDEX, IGNORE_INDEX]]
    correct, total, exact, sequences = token_accuracy([[1, 2]], labels)
    assert (correct, total, exact, sequences) == (0, 0, 0, 0)


def test_a_single_example_is_accepted() -> None:
    correct, total, _, _ = token_accuracy([5, 6, 0], [IGNORE_INDEX, 5, 6])
    assert (correct, total) == (2, 2)


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_reports_both_accuracies() -> None:
    labels = [[IGNORE_INDEX, 5, 6], [IGNORE_INDEX, 7, 8]]
    predictions = [[5, 6, 0], [7, 99, 0]]
    metrics = compute_metrics(_Prediction(predictions, labels))
    assert metrics["token_accuracy"] == pytest.approx(0.75)
    assert metrics["sequence_accuracy"] == pytest.approx(0.5)
    assert metrics["completion_tokens"] == 4


def test_compute_metrics_survives_an_empty_evaluation() -> None:
    metrics = compute_metrics(_Prediction([[1, 2]], [[IGNORE_INDEX, IGNORE_INDEX]]))
    assert metrics["token_accuracy"] == 0.0
    assert metrics["sequence_accuracy"] == 0.0


def test_compute_metrics_unwraps_tuples() -> None:
    labels = [[IGNORE_INDEX, 5]]
    metrics = compute_metrics(_Prediction(([[5, 0]],), labels))
    assert metrics["token_accuracy"] == 1.0


def test_logits_are_reduced_before_accumulation() -> None:
    """Keeping full logits would run evaluation out of memory on a real model."""
    torch = pytest.importorskip("torch")

    logits = torch.zeros(2, 3, 10)
    logits[..., 7] = 1.0
    reduced = preprocess_logits_for_metrics(logits, None)
    assert reduced.shape == (2, 3)
    assert (reduced == 7).all()


def test_logits_tuples_are_unwrapped() -> None:
    torch = pytest.importorskip("torch")

    logits = torch.zeros(1, 2, 5)
    assert preprocess_logits_for_metrics((logits, None), None).shape == (1, 2)


# --------------------------------------------------------------------------- #
# reports and history
# --------------------------------------------------------------------------- #
def test_eval_report_from_trainer_metrics() -> None:
    report = EvalReport.from_metrics(
        {
            "eval_loss": 1.0,
            "eval_token_accuracy": 0.5,
            "eval_sequence_accuracy": 0.25,
            "epoch": 1.0,
        }
    )
    assert report.loss == 1.0
    assert report.perplexity == pytest.approx(math.e)
    assert report.token_accuracy == 0.5
    assert "loss" in report.render() and "ppl" in report.render()


def test_eval_report_serialises_only_what_it_has() -> None:
    payload = EvalReport(loss=0.5).to_dict()
    assert payload["loss"] == 0.5
    assert "token_accuracy" not in payload


def test_summarise_history_splits_training_from_evaluation() -> None:
    history = [
        {"loss": 3.0, "step": 10, "epoch": 0.5, "learning_rate": 5e-4},
        {"eval_loss": 2.5, "step": 10, "epoch": 0.5},
        {"loss": 2.0, "step": 20, "epoch": 1.0, "learning_rate": 1e-4},
        {"eval_loss": 1.5, "eval_token_accuracy": 0.9, "step": 20, "epoch": 1.0},
    ]
    summary = summarise_history(history)
    assert [entry["step"] for entry in summary["train_loss_curve"]] == [10, 20]
    assert [entry["step"] for entry in summary["evaluations"]] == [10, 20]
    assert summary["final_train_loss"] == 2.0
    assert summary["final_validation_loss"] == 1.5
    assert summary["best_validation"]["loss"] == 1.5


def test_summarise_history_finds_the_best_not_the_last() -> None:
    history = [
        {"eval_loss": 1.0, "step": 10},
        {"eval_loss": 2.0, "step": 20},  # got worse
    ]
    summary = summarise_history(history)
    assert summary["best_validation"]["step"] == 10
    assert summary["final_validation_loss"] == 2.0


def test_summarise_empty_history() -> None:
    summary = summarise_history([])
    assert summary["train_loss_curve"] == []
    assert summary["best_validation"] is None
