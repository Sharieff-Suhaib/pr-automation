"""OR2 output representation: alignment, target building, completion parsing."""

from __future__ import annotations

import pytest
import samples

from repairllama.representation.ir4 import RepresentationError, describe_region
from repairllama.representation.or2 import (
    OR2Options,
    align_fixed_region,
    build_or2_target,
    dedent_lines,
    fixed_region_from_bounds,
    parse_or2_output,
    restore_indentation,
    splice_region,
)


# --------------------------------------------------------------------------- #
# aligning the buggy region onto the fixed source
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sample", samples.ALL_SAMPLES, ids=lambda s: s.name)
def test_alignment_reproduces_the_fixed_source(sample: samples.Sample) -> None:
    fixed = align_fixed_region(
        sample.buggy_lines, sample.fixed_lines, sample.start, sample.end
    )
    spliced = splice_region(sample.buggy_lines, sample.start, sample.end, fixed.lines)
    assert spliced == sample.fixed_lines


@pytest.mark.parametrize("sample", samples.ALL_SAMPLES, ids=lambda s: s.name)
def test_alignment_matches_the_expected_target(sample: samples.Sample) -> None:
    fixed = align_fixed_region(
        sample.buggy_lines, sample.fixed_lines, sample.start, sample.end
    )
    assert "\n".join(fixed.lines) == sample.target
    assert fixed.inferred is True


def test_alignment_reports_a_multi_line_range() -> None:
    sample = samples.NESTED_IF
    fixed = align_fixed_region(
        sample.buggy_lines, sample.fixed_lines, sample.start, sample.end
    )
    assert (fixed.start_line, fixed.end_line) == (5, 7)
    assert fixed.num_lines == 3
    assert fixed.changed_outside_region is False


def test_alignment_detects_a_deletion() -> None:
    sample = samples.DELETION
    fixed = align_fixed_region(
        sample.buggy_lines, sample.fixed_lines, sample.start, sample.end
    )
    assert fixed.is_deletion is True
    assert fixed.lines == ()
    assert fixed.end_line == fixed.start_line - 1


def test_alignment_handles_an_insertion() -> None:
    buggy = "int f() {\n    return 1;\n}\n"
    fixed = "int f() {\n    log();\n    return 1;\n}\n"
    aligned = align_fixed_region(buggy.splitlines(), fixed.splitlines(), 2, 2)
    assert list(aligned.lines) == ["    log();", "    return 1;"]
    assert aligned.changed_outside_region is False


def test_alignment_flags_edits_outside_the_region() -> None:
    sample = samples.ONE_LINE
    aligned = align_fixed_region(
        sample.buggy_lines,
        samples.ONE_LINE_FIXED_WITH_OUTSIDE_EDIT.splitlines(),
        sample.start,
        sample.end,
    )
    assert aligned.changed_outside_region is True


def test_identical_sources_align_to_an_unchanged_region() -> None:
    sample = samples.ONE_LINE
    aligned = align_fixed_region(
        sample.buggy_lines, sample.buggy_lines, sample.start, sample.end
    )
    assert list(aligned.lines) == sample.buggy_lines[sample.start - 1 : sample.end]
    assert aligned.changed_outside_region is False


# --------------------------------------------------------------------------- #
# explicit fixed bounds
# --------------------------------------------------------------------------- #
def test_explicit_bounds_are_not_inferred() -> None:
    sample = samples.MULTI_LINE
    fixed = fixed_region_from_bounds(sample.fixed_lines, 4, 5)
    assert "\n".join(fixed.lines) == sample.target
    assert fixed.inferred is False


def test_explicit_bounds_allow_an_empty_region() -> None:
    fixed = fixed_region_from_bounds(samples.DELETION.fixed_lines, 4, 3)
    assert fixed.lines == ()
    assert fixed.is_deletion is True


@pytest.mark.parametrize(("start", "end"), [(0, 2), (4, 2), (2, 999)])
def test_explicit_bounds_validate(start: int, end: int) -> None:
    with pytest.raises(RepresentationError):
        fixed_region_from_bounds(samples.ONE_LINE.fixed_lines, start, end)


# --------------------------------------------------------------------------- #
# building the target
# --------------------------------------------------------------------------- #
def test_target_keeps_original_indentation() -> None:
    target = build_or2_target(["        } else {", "            return 1;"])
    assert target == "        } else {\n            return 1;"


def test_target_is_verbatim_including_trailing_whitespace() -> None:
    lines = ["    int x = 1;   ", "\tint y = 2;"]
    assert build_or2_target(lines) == "\n".join(lines)


def test_target_can_be_dedented_and_restored() -> None:
    lines = ["        } else {", "            return 1;"]
    dedented = build_or2_target(lines, OR2Options(dedent=True))
    assert dedented == "} else {\n    return 1;"
    assert restore_indentation(dedented.split("\n"), "        ") == lines


def test_dedent_lines_reports_the_removed_prefix() -> None:
    stripped, prefix = dedent_lines(["\t\ta;", "\t\tb;"])
    assert stripped == ["a;", "b;"]
    assert prefix == "\t\t"


def test_dedent_leaves_blank_lines_blank() -> None:
    assert restore_indentation(["a;", "", "b;"], "    ") == ["    a;", "", "    b;"]


def test_empty_target_is_allowed_by_default() -> None:
    assert build_or2_target([]) == ""


def test_empty_target_can_be_rejected() -> None:
    with pytest.raises(RepresentationError, match="allow_empty"):
        build_or2_target([], OR2Options(allow_empty=False))


def test_oversized_target_is_rejected() -> None:
    with pytest.raises(RepresentationError, match="max_target_lines"):
        build_or2_target(["a;", "b;", "c;"], OR2Options(max_target_lines=2))


def test_invalid_max_target_lines() -> None:
    with pytest.raises(RepresentationError):
        OR2Options(max_target_lines=0)


# --------------------------------------------------------------------------- #
# parsing what a model actually returns
# --------------------------------------------------------------------------- #
def test_parse_plain_completion() -> None:
    prediction = parse_or2_output("            return a;")
    assert prediction.lines == ("            return a;",)
    assert prediction.is_deletion is False


def test_parse_strips_code_fences() -> None:
    prediction = parse_or2_output("```java\n    return a;\n```")
    assert prediction.lines == ("    return a;",)
    assert prediction.had_code_fence is True


def test_parse_cuts_at_a_stop_sequence() -> None:
    completion = "    return a;\n// buggy lines start here\n    junk;"
    prediction = parse_or2_output(
        completion, OR2Options(stop_sequences=("// buggy lines start here",))
    )
    assert prediction.lines == ("    return a;",)
    assert prediction.truncated_by == "// buggy lines start here"


def test_parse_drops_an_echoed_fill_token() -> None:
    prediction = parse_or2_output("<FILL_ME>\n    return a;")
    assert prediction.lines == ("    return a;",)


def test_parse_trims_blank_padding_but_not_inner_blanks() -> None:
    prediction = parse_or2_output("\n\n    a();\n\n    b();\n\n")
    assert prediction.lines == ("    a();", "", "    b();")


def test_parse_preserves_inner_indentation() -> None:
    prediction = parse_or2_output("    if (x) {\n        y();\n    }")
    assert prediction.lines == ("    if (x) {", "        y();", "    }")


def test_parse_empty_completion_is_a_deletion() -> None:
    prediction = parse_or2_output("   \n\n")
    assert prediction.lines == ()
    assert prediction.is_deletion is True


def test_parse_restores_indentation_when_dedenting() -> None:
    region = describe_region(samples.NESTED_IF.buggy_lines, 5, 7)
    prediction = parse_or2_output(
        "} else if (score >= 80) {",
        OR2Options(dedent=True),
        region=region,
    )
    assert prediction.lines == ("        } else if (score >= 80) {",)


def test_prediction_serialises() -> None:
    payload = parse_or2_output("    return a;").to_dict()
    assert payload["num_lines"] == 1
    assert payload["is_deletion"] is False


# --------------------------------------------------------------------------- #
# splicing
# --------------------------------------------------------------------------- #
def test_splice_replaces_a_multi_line_region() -> None:
    lines = ["a", "b", "c", "d"]
    assert splice_region(lines, 2, 3, ["X"]) == ["a", "X", "d"]


def test_splice_can_delete() -> None:
    assert splice_region(["a", "b", "c"], 2, 2, []) == ["a", "c"]


def test_splice_can_insert_without_removing() -> None:
    assert splice_region(["a", "b"], 2, 1, ["X"]) == ["a", "X", "b"]


@pytest.mark.parametrize(("start", "end"), [(0, 1), (2, 0), (1, 99)])
def test_splice_validates_bounds(start: int, end: int) -> None:
    with pytest.raises(RepresentationError, match="cannot splice"):
        splice_region(["a", "b"], start, end, ["X"])
