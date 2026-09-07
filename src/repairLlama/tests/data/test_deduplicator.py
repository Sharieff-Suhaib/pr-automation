"""Duplicate detection: normalisation, the three strategies, and bookkeeping."""

from __future__ import annotations

import java_pairs as jp
import pytest

from repairllama.data.deduplicator import (
    Deduplicator,
    deduplicate,
    fingerprint,
    normalize_code,
)
from repairllama.data.models import BugFixPair, RejectionReason


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_normalize_collapses_whitespace() -> None:
    assert normalize_code("int   a;\n\n  int b;") == "int a; int b;"


def test_normalize_removes_comments() -> None:
    assert normalize_code("int a; // note\n") == normalize_code("int a;")


def test_normalize_keeps_string_contents() -> None:
    assert normalize_code('log("a");') != normalize_code('log("b");')


def test_normalize_is_stable() -> None:
    assert normalize_code(jp.CALCULATOR_BUGGY) == normalize_code(jp.CALCULATOR_BUGGY)


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def test_fingerprint_ignores_the_bug_id() -> None:
    assert fingerprint(jp.calculator_pair("a")) == fingerprint(jp.calculator_pair("b"))


def test_exact_strategy_is_sensitive_to_formatting() -> None:
    original = jp.calculator_pair("a")
    reindented = BugFixPair(
        "b",
        original.buggy_code.replace("    ", "\t"),
        original.fixed_code.replace("    ", "\t"),
    )
    assert fingerprint(original, "exact") != fingerprint(reindented, "exact")


def test_normalized_strategy_sees_through_formatting_and_comments() -> None:
    original = jp.calculator_pair("a")
    restyled = BugFixPair(
        "b",
        original.buggy_code.replace("    ", "\t").replace(
            "// returns the larger of two values", "// max"
        ),
        original.fixed_code.replace("    ", "\t"),
    )
    assert fingerprint(original, "normalized") == fingerprint(restyled, "normalized")


def test_input_only_strategy_ignores_the_fix() -> None:
    a = jp.calculator_pair("a").with_region()
    b = BugFixPair(
        "b",
        a.buggy_code,
        a.buggy_code.replace("return b;", "return Math.max(a, b);", 1),
        suspicious_start=a.suspicious_start,
        suspicious_end=a.suspicious_end,
    )
    assert fingerprint(a, "input_only") == fingerprint(b, "input_only")
    assert fingerprint(a, "normalized") != fingerprint(b, "normalized")


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dedup strategy"):
        fingerprint(jp.calculator_pair(), "fuzzy")


def test_deduplicator_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown dedup strategy"):
        Deduplicator("fuzzy")


# --------------------------------------------------------------------------- #
# running over a corpus
# --------------------------------------------------------------------------- #
def test_first_occurrence_is_kept() -> None:
    pairs = [jp.calculator_pair("first"), jp.calculator_pair("second")]
    result = deduplicate(pairs)
    assert [pair.bug_id for pair in result.kept] == ["first"]
    assert result.duplicates == 1
    assert result.duplicate_of == {"second": "first"}


def test_distinct_pairs_all_survive() -> None:
    result = deduplicate(jp.corpus(8))
    assert len(result.kept) == 8
    assert result.duplicates == 0


def test_every_input_is_accounted_for() -> None:
    pairs = jp.corpus(5) + [jp.corpus(5)[0]]
    result = deduplicate(pairs)
    assert result.received == len(pairs)
    assert len(result.kept) + result.duplicates == len(pairs)


def test_duplicates_are_reported_with_their_original() -> None:
    result = deduplicate([jp.calculator_pair("a"), jp.calculator_pair("b")])
    rejection = result.rejections[0]
    assert rejection.reason is RejectionReason.DUPLICATE
    assert rejection.stage == "deduplicate"
    assert "a" in rejection.detail


def test_counts_by_reason() -> None:
    result = deduplicate([jp.calculator_pair("a"), jp.calculator_pair("b")])
    assert result.counts_by_reason() == {"duplicate": 1}


def test_exact_strategy_keeps_reformatted_copies() -> None:
    original = jp.calculator_pair("a")
    restyled = BugFixPair(
        "b",
        original.buggy_code.replace("    ", "\t"),
        original.fixed_code.replace("    ", "\t"),
    )
    assert len(deduplicate([original, restyled], "exact").kept) == 2
    assert len(deduplicate([original, restyled], "normalized").kept) == 1


def test_deduplication_is_order_stable() -> None:
    pairs = [jp.calculator_pair("z"), jp.calculator_pair("a")]
    assert deduplicate(pairs).kept[0].bug_id == "z"
    assert deduplicate(list(reversed(pairs))).kept[0].bug_id == "a"


def test_check_records_what_it_sees() -> None:
    dedup = Deduplicator()
    assert dedup.check(jp.calculator_pair("a")) is None
    assert dedup.check(jp.calculator_pair("b")) == "a"


def test_reset_forgets_history() -> None:
    dedup = Deduplicator()
    dedup.check(jp.calculator_pair("a"))
    dedup.reset()
    assert dedup.check(jp.calculator_pair("b")) is None
