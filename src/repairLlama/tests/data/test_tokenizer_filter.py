"""Token counting and length filtering."""

from __future__ import annotations

import java_pairs as jp
import pytest

from repairllama.data.models import RejectionReason, TrainingRecord
from repairllama.data.tokenizer_filter import (
    HeuristicTokenCounter,
    HuggingFaceTokenCounter,
    TokenizerFilter,
    TokenizerFilterOptions,
    TokenLengths,
    filter_by_length,
    get_token_counter,
)


def _record(bug_id: str = "b1", input_text: str = "prompt", output_text: str = "patch"):
    return TrainingRecord(
        input=input_text,
        output=output_text,
        bug_id=bug_id,
        suspicious_start=1,
        suspicious_end=1,
    )


# --------------------------------------------------------------------------- #
# the heuristic counter
# --------------------------------------------------------------------------- #
def test_empty_text_costs_nothing() -> None:
    assert HeuristicTokenCounter().count("") == 0


def test_counting_is_deterministic() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count(jp.CALCULATOR_BUGGY) == counter.count(jp.CALCULATOR_BUGGY)


def test_longer_code_costs_more() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count(jp.CALCULATOR_BUGGY) > counter.count("int a = 1;")


def test_token_count_beats_a_line_count() -> None:
    """Tokens, not lines: one dense line can outweigh several sparse ones."""
    counter = HeuristicTokenCounter()
    dense = "map.put(computeKeyFor(entry), transform(entry.getValue(), options));"
    sparse = "int a;\nint b;\nint c;\n"
    assert counter.count(dense) > counter.count(sparse)


def test_long_identifiers_cost_several_tokens() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("aVeryLongDescriptiveIdentifierName") > 1


def test_newlines_are_counted() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("a;\nb;") > counter.count("a;b;")


def test_the_counter_names_itself() -> None:
    assert HeuristicTokenCounter().name == "heuristic"


# --------------------------------------------------------------------------- #
# choosing a counter
# --------------------------------------------------------------------------- #
def test_default_counter_needs_no_model() -> None:
    counter = get_token_counter("codellama/CodeLlama-7b-hf")
    assert isinstance(counter, HeuristicTokenCounter)


def test_missing_tokenizer_falls_back_with_a_warning() -> None:
    counter = get_token_counter(
        "definitely/not-a-real-model-xyz", use_model_tokenizer=True
    )
    assert isinstance(counter, HeuristicTokenCounter)


def test_fallback_can_be_refused() -> None:
    with pytest.raises(Exception):
        get_token_counter(
            "definitely/not-a-real-model-xyz",
            use_model_tokenizer=True,
            fallback=False,
        )


def test_hugging_face_counter_is_never_constructed_by_default() -> None:
    """Counting tokens must not pull in transformers.

    Checked in a fresh interpreter: this process may already have imported
    transformers via the tests that exercise the HuggingFace counter.
    """
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys;"
        "from repairllama.data.tokenizer_filter import get_token_counter;"
        "c = get_token_counter('codellama/CodeLlama-7b-hf');"
        "assert c.count('int a = 1;') > 0;"
        "assert 'transformers' not in sys.modules and 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2] / "src"),
    )
    assert result.returncode == 0, result.stderr


def test_hugging_face_counter_reports_a_missing_checkpoint() -> None:
    pytest.importorskip("transformers")
    with pytest.raises(Exception):
        HuggingFaceTokenCounter("definitely/not-a-real-model-xyz")


def test_a_custom_counter_can_be_plugged_in() -> None:
    class CharCounter:
        name = "chars"

        def count(self, text: str) -> int:
            return len(text)

    result = TokenizerFilter(CharCounter(), TokenizerFilterOptions(10, 10)).run(
        [_record(input_text="x" * 5), _record("b2", input_text="x" * 50)]
    )
    assert len(result.kept) == 1
    assert result.kept[0].input_tokens == 5


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def test_records_within_budget_survive() -> None:
    result = filter_by_length([_record()], options=TokenizerFilterOptions(100, 100))
    assert len(result.kept) == 1
    assert not result.rejections


def test_overlong_input_is_dropped() -> None:
    record = _record(input_text=jp.CALCULATOR_BUGGY * 10)
    result = filter_by_length([record], options=TokenizerFilterOptions(50, 500))
    assert not result.kept
    assert result.rejections[0].reason is RejectionReason.INPUT_TOO_LONG


def test_overlong_output_is_dropped() -> None:
    record = _record(output_text=jp.CALCULATOR_BUGGY * 10)
    result = filter_by_length([record], options=TokenizerFilterOptions(5000, 20))
    assert result.rejections[0].reason is RejectionReason.OUTPUT_TOO_LONG


def test_total_budget_is_enforced() -> None:
    options = TokenizerFilterOptions(
        max_input_tokens=10_000, max_output_tokens=10_000, max_total_tokens=5
    )
    result = filter_by_length([_record(input_text=jp.CALCULATOR_BUGGY)], options=options)
    assert not result.kept
    assert "total tokens" in result.rejections[0].detail


def test_minimum_output_length_is_enforced() -> None:
    options = TokenizerFilterOptions(1000, 1000, min_output_tokens=5)
    result = filter_by_length([_record(output_text="x")], options=options)
    assert not result.kept


def test_rejections_explain_the_limit() -> None:
    record = _record(input_text=jp.CALCULATOR_BUGGY * 10)
    result = filter_by_length([record], options=TokenizerFilterOptions(50, 500))
    assert "> 50 input tokens" in result.rejections[0].detail
    assert result.rejections[0].stage == "tokenizer_filter"


def test_kept_records_carry_their_measurements() -> None:
    result = filter_by_length([_record(input_text=jp.CALCULATOR_BUGGY)])
    kept = result.kept[0]
    assert kept.input_tokens and kept.output_tokens is not None
    assert kept.total_tokens == kept.input_tokens + kept.output_tokens


def test_measure_returns_both_lengths() -> None:
    lengths = TokenizerFilter().measure(_record(input_text="a b c", output_text="d"))
    assert isinstance(lengths, TokenLengths)
    assert lengths.total == lengths.input_tokens + lengths.output_tokens


# --------------------------------------------------------------------------- #
# the distribution
# --------------------------------------------------------------------------- #
def test_stats_cover_every_measured_record_not_just_survivors() -> None:
    records = [
        _record("small", input_text="int a;"),
        _record("huge", input_text=jp.CALCULATOR_BUGGY * 10),
    ]
    result = filter_by_length(records, options=TokenizerFilterOptions(50, 500))
    assert len(result.kept) == 1
    assert result.stats["input"].count == 2
    assert result.stats["input_kept"].count == 1


def test_stats_are_reported_per_side() -> None:
    result = filter_by_length([_record(input_text=jp.CALCULATOR_BUGGY)])
    assert set(result.stats) == {"input", "output", "total", "input_kept"}
    assert result.stats["total"].minimum >= result.stats["input"].minimum


def test_empty_input_yields_empty_stats() -> None:
    result = filter_by_length([])
    assert result.received == 0
    assert result.stats["input"].count == 0


def test_limits_must_be_positive() -> None:
    with pytest.raises(ValueError, match="token limits"):
        TokenizerFilterOptions(max_input_tokens=0)
