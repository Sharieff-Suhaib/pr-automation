"""The intermediate format, the training record, and the report types."""

from __future__ import annotations

import java_pairs as jp
import pytest

from repairllama.data.models import (
    BugFixPair,
    DataError,
    DatasetReport,
    Rejection,
    RejectionReason,
    StageReport,
    TokenStats,
    TrainingRecord,
)


# --------------------------------------------------------------------------- #
# BugFixPair
# --------------------------------------------------------------------------- #
def test_pair_requires_an_id() -> None:
    with pytest.raises(DataError, match="bug_id"):
        BugFixPair("", "a", "b")


def test_pair_requires_string_sources() -> None:
    with pytest.raises(DataError, match="must be strings"):
        BugFixPair("x", None, "b")  # type: ignore[arg-type]


def test_region_bounds_come_as_a_pair() -> None:
    with pytest.raises(DataError, match="together"):
        BugFixPair("x", "a", "b", suspicious_start=1)


def test_group_key_prefers_project() -> None:
    pair = jp.calculator_pair()
    assert pair.group_key == "calculator"
    assert BugFixPair("x", "a", "b", file_path="F.java").group_key == "F.java"
    assert BugFixPair("x", "a", "b").group_key == "x"


def test_changed_span_of_a_replacement() -> None:
    assert jp.calculator_pair().changed_span() == (5, 5)


def test_changed_span_of_a_multi_line_replacement() -> None:
    pair = BugFixPair("m", "a\nb\nc\nd\n", "a\nB\nC\nd\n")
    assert pair.changed_span() == (2, 3)


def test_changed_span_of_a_deletion() -> None:
    pair = BugFixPair("d", "a\nb\nc\n", "a\nc\n")
    assert pair.changed_span() == (2, 2)


def test_changed_span_of_an_insertion_anchors_on_a_real_line() -> None:
    pair = BugFixPair("i", "a\nb\n", "a\nnew\nb\n")
    start, end = pair.changed_span()
    assert start <= end
    assert (start, end) == (1, 1)


def test_changed_span_of_an_insertion_at_the_top() -> None:
    pair = BugFixPair("i", "a\nb\n", "new\na\nb\n")
    assert pair.changed_span() == (1, 1)


def test_changed_span_is_none_for_identical_sources() -> None:
    assert BugFixPair("s", "a\n", "a\n").changed_span() is None


def test_with_region_derives_and_is_idempotent() -> None:
    pair = jp.calculator_pair()
    derived = pair.with_region()
    assert (derived.suspicious_start, derived.suspicious_end) == (5, 5)
    assert derived.with_region() is derived


def test_with_region_rejects_identical_sources() -> None:
    with pytest.raises(DataError, match="identical"):
        BugFixPair("s", "a\n", "a\n").with_region()


def test_with_region_accepts_explicit_bounds() -> None:
    pair = jp.calculator_pair().with_region(4, 6)
    assert (pair.suspicious_start, pair.suspicious_end) == (4, 6)


def test_pair_round_trips_through_a_dict() -> None:
    pair = jp.calculator_pair().with_region()
    restored = BugFixPair.from_dict(pair.to_dict())
    assert restored == pair


def test_to_dict_omits_empty_optional_fields() -> None:
    payload = BugFixPair("x", "a", "b").to_dict()
    assert set(payload) == {"bug_id", "buggy_code", "fixed_code"}


def test_from_dict_reports_missing_fields() -> None:
    with pytest.raises(DataError, match="missing required field"):
        BugFixPair.from_dict({"bug_id": "x", "buggy_code": "a"})


# --------------------------------------------------------------------------- #
# TrainingRecord
# --------------------------------------------------------------------------- #
def _record(**kwargs) -> TrainingRecord:
    base = dict(
        input="prompt", output="patch", bug_id="b1", suspicious_start=5, suspicious_end=5
    )
    base.update(kwargs)
    return TrainingRecord(**base)


def test_training_row_leads_with_input_and_output() -> None:
    payload = _record().to_json_dict()
    assert list(payload)[:2] == ["input", "output"]
    assert payload["input"] == "prompt"
    assert payload["output"] == "patch"


def test_minimal_row_has_only_input_and_output() -> None:
    assert _record(project="p").to_json_dict(minimal=True) == {
        "input": "prompt",
        "output": "patch",
    }


def test_row_includes_tokens_and_split_when_known() -> None:
    record = _record().with_tokens(10, 3).with_split("train")
    payload = record.to_json_dict()
    assert payload["input_tokens"] == 10
    assert payload["output_tokens"] == 3
    assert payload["split"] == "train"
    assert record.total_tokens == 13


def test_total_tokens_is_none_until_measured() -> None:
    assert _record().total_tokens is None


def test_record_group_key_prefers_project() -> None:
    assert _record(project="p", file_path="F.java").group_key == "p"
    assert _record(file_path="F.java").group_key == "F.java"


# --------------------------------------------------------------------------- #
# TokenStats
# --------------------------------------------------------------------------- #
def test_token_stats_of_an_empty_dataset() -> None:
    stats = TokenStats.from_values([])
    assert stats.count == 0 and stats.maximum == 0 and stats.histogram == {}


def test_token_stats_summary() -> None:
    stats = TokenStats.from_values(list(range(1, 101)))
    assert stats.count == 100
    assert stats.minimum == 1 and stats.maximum == 100
    assert stats.mean == 50.5
    assert stats.median == 50.5
    assert (stats.p90, stats.p95, stats.p99) == (90, 95, 99)


def test_token_stats_histogram_covers_every_value() -> None:
    values = [10, 100, 300, 5000]
    stats = TokenStats.from_values(values, buckets=(64, 128, 256, 512))
    assert sum(stats.histogram.values()) == len(values)
    assert stats.histogram["0-64"] == 1
    assert stats.histogram[">512"] == 1


# --------------------------------------------------------------------------- #
# DatasetReport
# --------------------------------------------------------------------------- #
def _report() -> DatasetReport:
    return DatasetReport(
        dataset_name="demo",
        raw_samples=100,
        remaining=70,
        duplicates=12,
        stages=[StageReport("clean", 100, 82, {"test_only": 18})],
        removed_by_reason={"test_only": 18, "duplicate": 12},
        token_stats={"input": TokenStats.from_values([10, 20, 30])},
        split_counts={"train": 63, "validation": 4, "test": 3},
        settings={"seed": 42},
    )


def test_report_arithmetic() -> None:
    report = _report()
    assert report.removed == 30
    assert report.raw_samples == report.remaining + report.removed


def test_report_render_covers_every_section() -> None:
    text = _report().render()
    for fragment in [
        "raw samples",
        "removed",
        "duplicates" if "duplicates" in text else "dupes",
        "removed by reason",
        "test_only",
        "stages",
        "token length: input",
        "splits",
        "train",
        "settings",
    ]:
        assert fragment in text


def test_report_serialises_every_count() -> None:
    payload = _report().to_dict()
    assert payload["raw_samples"] == 100
    assert payload["removed"] == 30
    assert payload["duplicates"] == 12
    assert payload["remaining"] == 70
    assert payload["splits"] == {"train": 63, "validation": 4, "test": 3}
    assert payload["removed_by_reason"]["test_only"] == 18
    assert payload["token_stats"]["input"]["count"] == 3
    assert payload["stages"][0]["removed"] == 18


def test_report_saves_json_and_text(tmp_path) -> None:
    import json

    path = _report().save(tmp_path / "nested" / "report.json")
    assert json.loads(path.read_text())["dataset_name"] == "demo"
    assert "raw samples" in path.with_suffix(".txt").read_text()


# --------------------------------------------------------------------------- #
# Rejection
# --------------------------------------------------------------------------- #
def test_rejection_serialises() -> None:
    payload = Rejection("b1", RejectionReason.DUPLICATE, "of b0", "deduplicate").to_dict()
    assert payload == {
        "bug_id": "b1",
        "reason": "duplicate",
        "detail": "of b0",
        "stage": "deduplicate",
    }


def test_rejection_reasons_are_plain_strings() -> None:
    assert RejectionReason.TEST_ONLY.value == "test_only"
    assert str(RejectionReason.TEST_ONLY) == "test_only"
