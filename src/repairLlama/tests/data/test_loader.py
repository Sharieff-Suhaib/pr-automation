"""The loader: formats, field mapping, and the errors a bad corpus produces."""

from __future__ import annotations

import json
from pathlib import Path

import java_pairs as jp
import pytest

from repairllama.data.loader import (
    DirectoryLayout,
    FieldMap,
    LoaderOptions,
    available_formats,
    load_pairs,
    load_records,
    record_to_pair,
    register_loader,
)
from repairllama.data.models import DataError


# --------------------------------------------------------------------------- #
# JSONL / JSON
# --------------------------------------------------------------------------- #
def test_load_jsonl(tmp_path: Path) -> None:
    path = jp.write_jsonl_corpus(tmp_path / "corpus.jsonl", jp.corpus(5))
    pairs = list(load_pairs(path))
    assert len(pairs) == 5
    assert pairs[0].bug_id == "bug-000"
    assert pairs[0].buggy_code.startswith("public class Variant0")


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    record = json.dumps(jp.calculator_pair().to_dict())
    path.write_text(f"\n{record}\n\n", encoding="utf-8")
    assert len(list(load_pairs(path))) == 1


def test_load_json_array(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(jp.as_records(jp.corpus(3))), encoding="utf-8")
    pairs = list(load_pairs(path, LoaderOptions(format="json")))
    assert len(pairs) == 3


def test_load_json_object_of_records(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    records = {pair.bug_id: pair.to_dict() for pair in jp.corpus(3)}
    path.write_text(json.dumps(records), encoding="utf-8")
    assert len(list(load_pairs(path, LoaderOptions(format="json")))) == 3


def test_load_from_an_in_memory_iterable() -> None:
    pairs = list(load_pairs(jp.as_records(jp.corpus(4))))
    assert len(pairs) == 4


# --------------------------------------------------------------------------- #
# field mapping
# --------------------------------------------------------------------------- #
def test_default_aliases_are_accepted() -> None:
    record = {"id": "x1", "before": "class A {}", "after": "class B {}"}
    pair = record_to_pair(record, LoaderOptions(infer_region=False))
    assert pair.bug_id == "x1"
    assert pair.buggy_code == "class A {}"
    assert pair.fixed_code == "class B {}"


def test_custom_field_map() -> None:
    record = {"sha": "abc", "old": "class A {}", "new": "class B {}"}
    options = LoaderOptions(
        field_map=FieldMap(bug_id="sha", buggy_code="old", fixed_code="new"),
        infer_region=False,
    )
    assert record_to_pair(record, options).bug_id == "abc"


def test_dotted_paths_reach_into_nested_objects() -> None:
    record = {
        "commit": {"sha": "deadbeef", "repo": "apache/commons"},
        "code": {"buggy": "class A {}", "fixed": "class B {}"},
    }
    options = LoaderOptions(
        field_map=FieldMap(
            bug_id="commit.sha",
            buggy_code="code.buggy",
            fixed_code="code.fixed",
            project="commit.repo",
        ),
        infer_region=False,
    )
    pair = record_to_pair(record, options)
    assert pair.bug_id == "deadbeef"
    assert pair.project == "apache/commons"


def test_unmapped_keys_become_metadata() -> None:
    record = {
        "bug_id": "x",
        "buggy_code": "a",
        "fixed_code": "b",
        "commit": "abc123",
        "stars": 42,
    }
    pair = record_to_pair(record, LoaderOptions(infer_region=False))
    assert pair.metadata["commit"] == "abc123"
    assert pair.metadata["stars"] == 42


def test_metadata_capture_can_be_disabled() -> None:
    record = {"bug_id": "x", "buggy_code": "a", "fixed_code": "b", "extra": 1}
    options = LoaderOptions(
        field_map=FieldMap(keep_unmapped_as_metadata=False), infer_region=False
    )
    assert record_to_pair(record, options).metadata == {}


def test_explicit_region_is_kept() -> None:
    record = jp.calculator_pair().to_dict() | {
        "suspicious_start": 3,
        "suspicious_end": 7,
    }
    pair = record_to_pair(record)
    assert (pair.suspicious_start, pair.suspicious_end) == (3, 7)


def test_region_is_inferred_when_absent() -> None:
    pair = record_to_pair(jp.calculator_pair().to_dict())
    assert (pair.suspicious_start, pair.suspicious_end) == (5, 5)


def test_region_inference_can_be_disabled() -> None:
    pair = record_to_pair(
        jp.calculator_pair().to_dict(), LoaderOptions(infer_region=False)
    )
    assert not pair.has_region


def test_numeric_strings_are_accepted_as_line_numbers() -> None:
    record = jp.calculator_pair().to_dict() | {
        "suspicious_start": "5",
        "suspicious_end": "5",
    }
    assert record_to_pair(record).suspicious_start == 5


def test_nonsense_line_numbers_are_rejected() -> None:
    record = jp.calculator_pair().to_dict() | {
        "suspicious_start": "middle",
        "suspicious_end": 5,
    }
    with pytest.raises(DataError, match="line number"):
        record_to_pair(record)


# --------------------------------------------------------------------------- #
# directory corpora
# --------------------------------------------------------------------------- #
def test_load_directory(tmp_path: Path) -> None:
    root = jp.write_directory_corpus(tmp_path / "corpus", jp.corpus(4))
    pairs = list(load_pairs(root, LoaderOptions(format="directory")))
    assert len(pairs) == 4
    assert {pair.bug_id for pair in pairs} == {f"bug-{i:03d}" for i in range(4)}
    assert pairs[0].file_path.endswith("buggy.java")


def test_directory_custom_file_names(tmp_path: Path) -> None:
    root = jp.write_directory_corpus(
        tmp_path / "c", jp.corpus(2), buggy_name="before.java", fixed_name="after.java"
    )
    options = LoaderOptions(
        format="directory",
        layout=DirectoryLayout(buggy_name="before.java", fixed_name="after.java"),
    )
    assert len(list(load_pairs(root, options))) == 2


def test_directory_glob_style_names(tmp_path: Path) -> None:
    root = tmp_path / "c"
    (root / "bugA").mkdir(parents=True)
    (root / "bugA" / "Calculator.buggy.java").write_text(jp.CALCULATOR_BUGGY)
    (root / "bugA" / "Calculator.fixed.java").write_text(jp.CALCULATOR_FIXED)
    options = LoaderOptions(
        format="directory",
        layout=DirectoryLayout(buggy_name="*.buggy.java", fixed_name="*.fixed.java"),
    )
    pairs = list(load_pairs(root, options))
    assert len(pairs) == 1
    assert pairs[0].bug_id == "bugA"


def test_directory_reads_optional_metadata(tmp_path: Path) -> None:
    root = jp.write_directory_corpus(
        tmp_path / "c", jp.corpus(2), with_metadata=True
    )
    pairs = list(load_pairs(root, LoaderOptions(format="directory")))
    assert pairs[0].project == "project-0"
    assert pairs[0].metadata["origin"] == "test"


def test_directory_skips_bugs_without_a_fixed_version(tmp_path: Path) -> None:
    root = jp.write_directory_corpus(tmp_path / "c", jp.corpus(3))
    (root / "bug-001" / "fixed.java").unlink()
    assert len(list(load_pairs(root, LoaderOptions(format="directory")))) == 2


def test_directory_id_from_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "c"
    (root / "apache" / "bug1").mkdir(parents=True)
    (root / "apache" / "bug1" / "buggy.java").write_text(jp.CALCULATOR_BUGGY)
    (root / "apache" / "bug1" / "fixed.java").write_text(jp.CALCULATOR_FIXED)
    options = LoaderOptions(
        format="directory", layout=DirectoryLayout(id_from="relative_path")
    )
    assert list(load_pairs(root, options))[0].bug_id == "apache/bug1/buggy.java"


def test_directory_project_pattern(tmp_path: Path) -> None:
    root = tmp_path / "c"
    (root / "apache" / "bug1").mkdir(parents=True)
    (root / "apache" / "bug1" / "buggy.java").write_text(jp.CALCULATOR_BUGGY)
    (root / "apache" / "bug1" / "fixed.java").write_text(jp.CALCULATOR_FIXED)
    options = LoaderOptions(
        format="directory",
        layout=DirectoryLayout(project_pattern=r"^([^/]+)/"),
    )
    assert list(load_pairs(root, options))[0].project == "apache"


def test_unknown_id_from_is_rejected() -> None:
    with pytest.raises(DataError, match="id_from"):
        DirectoryLayout(id_from="magic")


# --------------------------------------------------------------------------- #
# limits, tolerance and errors
# --------------------------------------------------------------------------- #
def test_limit_caps_the_number_of_pairs(tmp_path: Path) -> None:
    path = jp.write_jsonl_corpus(tmp_path / "c.jsonl", jp.corpus(10))
    assert len(list(load_pairs(path, LoaderOptions(limit=3)))) == 3


def test_invalid_records_raise_by_default(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"bug_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing required field"):
        list(load_pairs(path))


def test_invalid_records_can_be_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    good = json.dumps(jp.calculator_pair().to_dict())
    path.write_text(f'{json.dumps({"bug_id": "x"})}\n{good}\n', encoding="utf-8")
    assert len(list(load_pairs(path, LoaderOptions(skip_invalid=True)))) == 1


def test_missing_source_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        list(load_pairs(tmp_path / "absent.jsonl"))


def test_malformed_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"c\.jsonl:1"):
        list(load_pairs(path))


def test_unknown_format_is_reported(tmp_path: Path) -> None:
    path = jp.write_jsonl_corpus(tmp_path / "c.jsonl", jp.corpus(1))
    with pytest.raises(DataError, match="unknown loader format"):
        list(load_pairs(path, LoaderOptions(format="parquet")))


def test_directory_format_needs_a_directory(tmp_path: Path) -> None:
    path = jp.write_jsonl_corpus(tmp_path / "c.jsonl", jp.corpus(1))
    with pytest.raises(DataError, match="needs a directory"):
        list(load_pairs(path, LoaderOptions(format="directory")))


# --------------------------------------------------------------------------- #
# extensibility
# --------------------------------------------------------------------------- #
def test_custom_loader_can_be_registered(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("A|B\nC|D\n", encoding="utf-8")

    def read_pipes(target, options):
        for index, line in enumerate(target.read_text().splitlines()):
            buggy, fixed = line.split("|")
            yield {"bug_id": f"p{index}", "buggy_code": buggy, "fixed_code": fixed}

    register_loader("pipes", read_pipes)
    assert "pipes" in available_formats()
    pairs = list(load_pairs(path, LoaderOptions(format="pipes", infer_region=False)))
    assert [pair.bug_id for pair in pairs] == ["p0", "p1"]


def test_load_records_yields_raw_dicts(tmp_path: Path) -> None:
    path = jp.write_jsonl_corpus(tmp_path / "c.jsonl", jp.corpus(2))
    records = list(load_records(path))
    assert isinstance(records[0], dict)
    assert records[0]["bug_id"] == "bug-000"
