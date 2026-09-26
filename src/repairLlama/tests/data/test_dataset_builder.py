"""The end-to-end build: representation, splitting, JSONL output, the report."""

from __future__ import annotations

import json
from pathlib import Path

import java_pairs as jp
import pytest

from repairllama.config import RepairConfig
from repairllama.data.cleaner import CleanerOptions
from repairllama.data.dataset_builder import (
    BuildOptions,
    DatasetBuilder,
    SplitRatios,
    build_dataset,
    split_records,
)
from repairllama.data.models import BugFixPair, DataError, TrainingRecord
from repairllama.data.tokenizer_filter import TokenizerFilterOptions


def _builder(**options) -> DatasetBuilder:
    return DatasetBuilder(BuildOptions(seed=7, **options))


def _grouped_records(count: int, projects: int) -> list[TrainingRecord]:
    return [
        TrainingRecord(
            input="p",
            output="o",
            bug_id=f"bug-{index:04d}",
            suspicious_start=1,
            suspicious_end=1,
            project=f"project-{index % projects}",
        )
        for index in range(count)
    ]


def _records(count: int = 30) -> list[TrainingRecord]:
    return [
        TrainingRecord(
            input=f"prompt {index}",
            output=f"patch {index}",
            bug_id=f"bug-{index:03d}",
            suspicious_start=1,
            suspicious_end=1,
            project=f"project-{index % 10}",
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# representation
# --------------------------------------------------------------------------- #
def test_representation_produces_ir4_input_and_or2_output() -> None:
    records, rejections = _builder().represent([jp.calculator_pair().with_region()])
    assert not rejections
    record = records[0]
    assert "<FILL_ME>" in record.input
    assert "// buggy lines start here" in record.input
    assert record.output == "            return a;"
    assert record.suspicious_start == 5


def test_representation_derives_a_missing_region() -> None:
    records, _ = _builder().represent([jp.calculator_pair()])
    assert records[0].suspicious_start == 5


def test_representation_carries_provenance() -> None:
    pair = jp.calculator_pair(metadata={"commit": "abc"})
    record = _builder().represent([pair.with_region()])[0][0]
    assert record.bug_id == "calc-1"
    assert record.project == "calculator"
    assert record.file_path.endswith("Calculator.java")
    assert record.metadata["commit"] == "abc"


def test_representation_errors_become_rejections() -> None:
    broken = BugFixPair(
        "broken",
        jp.CALCULATOR_BUGGY,
        jp.CALCULATOR_FIXED,
        suspicious_start=1,
        suspicious_end=999,
    )
    records, rejections = _builder().represent([broken])
    assert not records
    assert rejections[0].reason.value == "representation_error"
    assert rejections[0].stage == "represent"


def test_representation_errors_can_be_fatal() -> None:
    broken = BugFixPair(
        "broken", jp.CALCULATOR_BUGGY, jp.CALCULATOR_FIXED, 1, 999
    )
    builder = DatasetBuilder(BuildOptions(skip_representation_errors=False))
    with pytest.raises(Exception):
        builder.represent([broken])


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_splitting_is_exhaustive_and_disjoint() -> None:
    records = _records(30)
    splits = split_records(records, seed=1)
    ids = [record.bug_id for group in splits.values() for record in group]
    assert sorted(ids) == sorted(record.bug_id for record in records)
    assert len(ids) == len(set(ids))


def test_splitting_is_deterministic_for_a_seed() -> None:
    records = _records(40)
    assert split_records(records, seed=3) == split_records(records, seed=3)


def test_record_splitting_depends_on_the_seed() -> None:
    records = _records(40)
    first = {r.bug_id for r in split_records(records, seed=1, group_by_project=False)["train"]}
    second = {r.bug_id for r in split_records(records, seed=2, group_by_project=False)["train"]}
    assert first != second


def test_group_splitting_depends_on_the_seed() -> None:
    """With enough groups to fill every split, the seed decides which go where."""
    records = _grouped_records(600, projects=60)
    first = {r.project for r in split_records(records, seed=1)["test"]}
    second = {r.project for r in split_records(records, seed=2)["test"]}
    assert first and second and first != second


def test_group_splitting_is_stable_when_everything_fits_in_train() -> None:
    """Too few groups to fill val/test: membership is seed-independent."""
    records = _records(40)  # 10 projects, all of which fit inside the 90% target

    def membership(seed: int) -> dict[str, set[str]]:
        return {
            name: {record.bug_id for record in group}
            for name, group in split_records(records, seed=seed).items()
        }

    assert membership(1) == membership(2)


def test_splitting_ignores_input_order() -> None:
    records = _records(40)
    shuffled = list(reversed(records))
    assert split_records(records, seed=5) == split_records(shuffled, seed=5)


def test_records_are_tagged_with_their_split() -> None:
    splits = split_records(_records(30), seed=1)
    for name, group in splits.items():
        assert all(record.split == name for record in group)


def test_projects_do_not_straddle_splits() -> None:
    splits = split_records(_records(100), seed=4, group_by_project=True)
    owners: dict[str, str] = {}
    for name, group in splits.items():
        for record in group:
            assert owners.setdefault(record.project, name) == name


def test_record_level_splitting_hits_the_ratios() -> None:
    splits = split_records(_records(100), seed=4, group_by_project=False)
    assert len(splits["train"]) == 90
    assert len(splits["validation"]) == 5
    assert len(splits["test"]) == 5


def test_group_splitting_approaches_the_ratios_at_scale() -> None:
    splits = split_records(_grouped_records(600, projects=60), seed=11)
    assert 520 <= len(splits["train"]) <= 560
    assert splits["validation"] and splits["test"]


def test_empty_input_yields_empty_splits() -> None:
    assert split_records([], seed=1) == {"train": [], "validation": [], "test": []}


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(DataError, match="sum to 1"):
        SplitRatios(train=0.5, validation=0.1, test=0.1)


def test_custom_ratios_are_respected() -> None:
    ratios = SplitRatios(train=0.5, validation=0.25, test=0.25)
    splits = split_records(_records(100), ratios, seed=2, group_by_project=False)
    assert len(splits["train"]) == 50 and len(splits["test"]) == 25


# --------------------------------------------------------------------------- #
# the whole pipeline
# --------------------------------------------------------------------------- #
def test_build_over_a_clean_corpus() -> None:
    result = _builder().build(jp.corpus(30, projects=10))
    assert result.total == 30
    assert result.report.raw_samples == 30
    assert result.report.remaining == 30


def test_report_accounts_for_every_sample() -> None:
    pairs = [
        *jp.corpus(10, projects=5),
        BugFixPair("two", jp.TWO_METHODS_BUGGY, jp.TWO_METHODS_FIXED),
        BugFixPair("same", jp.CALCULATOR_BUGGY, jp.CALCULATOR_BUGGY),
        jp.calculator_pair("dup-a"),
        jp.calculator_pair("dup-b"),
        jp.big_diff_pair("big"),
    ]
    report = _builder().build(pairs).report

    assert report.raw_samples == len(pairs)
    assert report.remaining + report.removed == report.raw_samples
    assert report.removed == sum(report.removed_by_reason.values())
    assert report.duplicates == 1
    assert report.removed_by_reason["not_single_function"] == 1
    assert report.removed_by_reason["identical"] == 1
    assert report.removed_by_reason["diff_too_large"] == 1
    assert report.removed_by_reason["duplicate"] == 1
    assert sum(report.split_counts.values()) == report.remaining


def test_every_stage_appears_in_the_report() -> None:
    report = _builder().build(jp.corpus(6, projects=3)).report
    assert [stage.name for stage in report.stages] == [
        "clean",
        "deduplicate",
        "represent",
        "tokenizer_filter",
    ]
    assert report.stages[0].received == 6


def test_deduplication_can_be_switched_off() -> None:
    pairs = [jp.calculator_pair("a"), jp.calculator_pair("b")]
    result = DatasetBuilder(BuildOptions(deduplicate=False)).build(pairs)
    assert result.total == 2
    assert result.report.duplicates == 0
    assert "deduplicate" not in [stage.name for stage in result.report.stages]


def test_token_stats_are_reported() -> None:
    report = _builder().build(jp.corpus(8, projects=4)).report
    assert report.token_stats["input"].count == 8
    assert report.token_stats["input"].maximum > 0
    assert sum(report.token_stats["input"].histogram.values()) == 8


def test_length_filter_drops_oversized_examples() -> None:
    builder = DatasetBuilder(
        BuildOptions(seed=1),
        token_options=TokenizerFilterOptions(max_input_tokens=10, max_output_tokens=10),
    )
    result = builder.build(jp.corpus(6, projects=3))
    assert result.total == 0
    assert result.report.removed_by_reason["input_too_long"] == 6


def test_max_examples_caps_the_input() -> None:
    result = DatasetBuilder(BuildOptions(max_examples=4)).build(jp.corpus(20, projects=5))
    assert result.report.raw_samples == 4


def test_cleaner_options_are_honoured() -> None:
    builder = DatasetBuilder(
        BuildOptions(seed=1),
        cleaner_options=CleanerOptions(require_single_function=False),
    )
    pairs = [BugFixPair("f", jp.FIELD_ONLY_BUGGY, jp.FIELD_ONLY_FIXED)]
    assert builder.build(pairs).total == 1


def test_report_settings_record_the_run() -> None:
    settings = _builder().build(jp.corpus(4, projects=2)).report.settings
    assert settings["seed"] == 7
    assert settings["input_representation"] == "IR4"
    assert settings["output_representation"] == "OR2"
    assert settings["token_counter"] == "heuristic"


def test_building_an_empty_corpus_is_not_an_error() -> None:
    result = _builder().build([])
    assert result.total == 0
    assert result.report.raw_samples == 0
    assert "raw samples" in result.report.render()


# --------------------------------------------------------------------------- #
# sources and output
# --------------------------------------------------------------------------- #
def test_build_from_a_jsonl_corpus(tmp_path: Path) -> None:
    path = jp.write_jsonl_corpus(tmp_path / "corpus.jsonl", jp.corpus(10, projects=5))
    result = _builder().build_from_source(path)
    assert result.total == 10


def test_build_from_a_directory_corpus(tmp_path: Path) -> None:
    from repairllama.data.loader import LoaderOptions

    root = jp.write_directory_corpus(tmp_path / "corpus", jp.corpus(6, projects=3))
    result = _builder().build_from_source(root, LoaderOptions(format="directory"))
    assert result.total == 6


def test_build_dataset_accepts_ready_made_pairs() -> None:
    assert build_dataset(jp.corpus(5, projects=2)).total == 5


def test_build_dataset_accepts_raw_records() -> None:
    assert build_dataset(jp.as_records(jp.corpus(5, projects=2))).total == 5


def test_written_rows_have_input_and_output(tmp_path: Path) -> None:
    result = _builder().build(jp.corpus(12, projects=6))
    paths = result.write(tmp_path / "splits")

    rows = [
        json.loads(line)
        for line in paths["train"].read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    for row in rows:
        assert list(row)[:2] == ["input", "output"]
        assert "<FILL_ME>" in row["input"]
        assert row["output"]
        assert row["split"] == "train"


def test_minimal_rows_carry_only_input_and_output(tmp_path: Path) -> None:
    result = _builder().build(jp.corpus(6, projects=3))
    paths = result.write(tmp_path / "splits", minimal=True)
    row = json.loads(paths["train"].read_text(encoding="utf-8").splitlines()[0])
    assert set(row) == {"input", "output"}


def test_write_creates_a_file_per_split(tmp_path: Path) -> None:
    result = _builder().build(jp.corpus(12, projects=6))
    paths = result.write(tmp_path / "splits")
    for name in ("train", "validation", "test"):
        assert paths[name].is_file()
        assert paths[name].name == f"{name}.jsonl"


def test_write_emits_the_report(tmp_path: Path) -> None:
    result = _builder().build(jp.corpus(8, projects=4))
    paths = result.write(tmp_path / "splits")
    payload = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert payload["raw_samples"] == 8
    assert payload["splits"]["train"] >= 1
    assert paths["report"].with_suffix(".txt").is_file()


def test_rejected_samples_are_written_for_audit(tmp_path: Path) -> None:
    pairs = [*jp.corpus(4, projects=2), jp.big_diff_pair("big")]
    result = _builder().build(pairs)
    paths = result.write(tmp_path / "splits")
    rows = [
        json.loads(line)
        for line in paths["rejected"].read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["bug_id"] == "big"
    assert rows[0]["reason"] == "diff_too_large"


def test_line_counts_match_the_report(tmp_path: Path) -> None:
    result = _builder().build(jp.corpus(20, projects=10))
    paths = result.write(tmp_path / "splits")
    for name, count in result.report.split_counts.items():
        lines = [
            line for line in paths[name].read_text(encoding="utf-8").splitlines() if line
        ]
        assert len(lines) == count


# --------------------------------------------------------------------------- #
# config bridge
# --------------------------------------------------------------------------- #
def test_builder_from_config() -> None:
    cfg = RepairConfig.default()
    builder = DatasetBuilder.from_config(cfg)
    assert builder.options.seed == cfg.data.shuffle_seed
    assert builder.options.dedup_strategy == cfg.data.dedup_strategy
    assert builder.token_filter.options.max_input_tokens == (
        cfg.representation.max_input_tokens
    )
    assert builder.cleaner.options.max_diff_lines == cfg.data.max_diff_lines


def test_builder_from_config_does_not_load_a_tokenizer() -> None:
    """Building a dataset must not pull in the model stack.

    Checked in a fresh interpreter: other tests in this session legitimately
    import transformers, so sys.modules here proves nothing.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "from repairllama.config import RepairConfig;"
        "from repairllama.data import DatasetBuilder;"
        "DatasetBuilder.from_config(RepairConfig.default());"
        "assert 'transformers' not in sys.modules and 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2] / "src"),
    )
    assert result.returncode == 0, result.stderr


def test_shipped_config_builds_a_dataset(shipped_config: RepairConfig) -> None:
    builder = DatasetBuilder.from_config(shipped_config)
    result = builder.build(jp.corpus(10, projects=5))
    assert result.total == 10
    assert result.report.settings["max_input_tokens"] == 1024
