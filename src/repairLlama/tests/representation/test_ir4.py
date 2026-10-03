"""IR4 input representation: line handling, commenting, rendering, fidelity."""

from __future__ import annotations

import pytest
import samples

from repairllama.representation.ir4 import (
    IR4Options,
    RepresentationError,
    comment_line,
    common_indent,
    describe_region,
    detect_newline,
    extract_region,
    indent_width,
    recover_region_from_prompt,
    render_ir4,
    split_lines,
    uncomment_line,
    validate_region,
)


# --------------------------------------------------------------------------- #
# line handling
# --------------------------------------------------------------------------- #
def test_trailing_newline_does_not_add_a_line() -> None:
    assert split_lines("a\nb\n") == ["a", "b"]
    assert split_lines("a\nb") == ["a", "b"]


def test_blank_lines_are_kept() -> None:
    assert split_lines("a\n\n\nb\n") == ["a", "", "", "b"]


def test_split_lines_rejects_non_strings() -> None:
    with pytest.raises(RepresentationError):
        split_lines(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("a\nb", "\n"), ("a\r\nb", "\r\n"), ("a\rb", "\r"), ("single", "\n")],
)
def test_detect_newline(text: str, expected: str) -> None:
    assert detect_newline(text) == expected


def test_crlf_source_keeps_line_numbering() -> None:
    lines = split_lines(samples.ONE_LINE.buggy.replace("\n", "\r\n"))
    assert lines == samples.ONE_LINE.buggy_lines
    region = describe_region(lines, 4, 4, newline="\r\n")
    assert region.newline == "\r\n"


# --------------------------------------------------------------------------- #
# region validation — 1-based and inclusive, never clamped
# --------------------------------------------------------------------------- #
def test_region_is_one_based_and_inclusive() -> None:
    lines = samples.ONE_LINE.buggy_lines
    assert extract_region(lines, 4, 4) == ["            return b;"]
    assert extract_region(lines, 3, 5) == [
        "        if (a > b) {",
        "            return b;",
        "        }",
    ]


def test_whole_source_region_is_valid() -> None:
    lines = samples.NEAR_END.buggy_lines
    assert extract_region(lines, 1, len(lines)) == lines


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (0, 3, "1-based"),
        (-1, 2, "1-based"),
        (5, 4, "must be >= suspicious_start"),
        (1, 99, "past the end"),
        (99, 99, "past the end"),
    ],
)
def test_malformed_boundaries_raise(start: int, end: int, message: str) -> None:
    with pytest.raises(RepresentationError, match=message):
        validate_region(samples.ONE_LINE.buggy_lines, start, end)


@pytest.mark.parametrize("value", [1.5, "3", None, True])
def test_non_integer_boundaries_raise(value: object) -> None:
    with pytest.raises(RepresentationError, match="1-based"):
        validate_region(samples.ONE_LINE.buggy_lines, value, 4)  # type: ignore[arg-type]


def test_empty_source_raises() -> None:
    with pytest.raises(RepresentationError, match="no lines"):
        validate_region([], 1, 1)


# --------------------------------------------------------------------------- #
# indentation
# --------------------------------------------------------------------------- #
def test_common_indent_of_a_nested_block() -> None:
    assert common_indent(["        } else {", "            return 1;"]) == "        "


def test_common_indent_ignores_blank_lines() -> None:
    assert common_indent(["    a;", "", "    b;"]) == "    "


def test_common_indent_of_tab_indented_code() -> None:
    assert common_indent(["\t\treturn x;", "\t\treturn y;"]) == "\t\t"


def test_indent_width_expands_tabs() -> None:
    assert indent_width("    ") == 4
    assert indent_width("\t") == 4
    assert indent_width("\t\t") == 8


def test_region_metadata_records_indent() -> None:
    region = describe_region(samples.LOOPS.buggy_lines, 5, 5)
    assert region.indent == " " * 12
    assert region.indent_width == 12
    assert region.num_lines == 1
    assert region.line_range == (5, 5)
    assert (region.start_index, region.end_index) == (4, 5)


# --------------------------------------------------------------------------- #
# commenting is exactly invertible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "line",
    [
        "        return b;",
        "\t\tint x = 1;",
        "",
        "    ",
        "        // already a comment",
        "        int x = 1; // trailing comment",
        '        String s = "// not a comment";',
        "  spaced;   ",
    ],
)
def test_comment_round_trip(line: str) -> None:
    assert uncomment_line(comment_line(line)) == line


def test_comment_preserves_indentation() -> None:
    assert comment_line("        return b;") == "        // return b;"
    assert comment_line("\t\treturn b;") == "\t\t// return b;"


def test_comment_blank_line_has_no_trailing_space() -> None:
    assert comment_line("") == "//"
    assert comment_line("   ") == "   //"


def test_uncomment_leaves_plain_code_alone() -> None:
    assert uncomment_line("        return b;") == "        return b;"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _render(sample: samples.Sample, options: IR4Options | None = None, **kwargs) -> str:
    options = options or IR4Options()
    lines = sample.buggy_lines
    region = describe_region(lines, sample.start, sample.end, options=options, **kwargs)
    return render_ir4(lines, region, options)


def test_prompt_marks_the_region_and_offers_a_fill_slot() -> None:
    prompt = _render(samples.ONE_LINE)
    assert "// buggy lines start here" in prompt
    assert "// buggy lines end here" in prompt
    assert prompt.count("<FILL_ME>") == 1


def test_prompt_preserves_the_buggy_code_commented_out() -> None:
    prompt = _render(samples.ONE_LINE)
    assert "            // return b;" in prompt
    # the buggy line survives, but never as executable code
    assert "\n            return b;\n" not in prompt


def test_fill_token_is_indented_to_the_region() -> None:
    """The slot sits at the region's own indentation, not column 0."""
    prompt = _render(samples.LOOPS)
    assert " " * 12 + "<FILL_ME>" in prompt.split("\n")


def test_fill_token_indentation_follows_tabs() -> None:
    prompt = _render(samples.TABS)
    assert "\t\t<FILL_ME>" in prompt.split("\n")


def test_context_lines_are_emitted_verbatim() -> None:
    prompt = _render(samples.NESTED_IF)
    for line in ["public class Grade {", "            } else {", '                return "F";']:
        assert line in prompt.split("\n")


def test_markers_and_fill_are_in_order() -> None:
    lines = _render(samples.MULTI_LINE).split("\n")
    start = lines.index("        // buggy lines start here")
    end = lines.index("        // buggy lines end here")
    fill = lines.index("        <FILL_ME>")
    assert start < end < fill
    assert end - start - 1 == samples.MULTI_LINE.end - samples.MULTI_LINE.start + 1


def test_header_states_the_line_range_explicitly() -> None:
    prompt = _render(samples.NESTED_IF)
    assert "// suspicious region: lines 5-7 of 15" in prompt


def test_file_path_header_is_optional() -> None:
    assert "// file: Grade.java" in _render(samples.NESTED_IF, file_path="Grade.java")
    without = _render(
        samples.NESTED_IF, IR4Options(include_file_path=False), file_path="Grade.java"
    )
    assert "// file:" not in without


def test_render_rejects_mismatched_metadata() -> None:
    lines = samples.ONE_LINE.buggy_lines
    region = describe_region(lines, 4, 4)
    with pytest.raises(RepresentationError, match="diverged"):
        render_ir4(lines[:5], region)


# --------------------------------------------------------------------------- #
# context window
# --------------------------------------------------------------------------- #
def test_context_window_truncates_and_says_so() -> None:
    source = samples.long_source(total_lines=60, bug_line=30)
    lines = split_lines(source)
    options = IR4Options(context_lines=3)
    region = describe_region(lines, 30, 30, options=options)
    prompt = render_ir4(lines, region, options)

    assert region.context_start_line == 27
    assert region.context_end_line == 33
    assert region.truncated_above == 26
    assert region.truncated_below == 27
    assert "// ... 26 line(s) omitted ..." in prompt
    assert "// ... 27 line(s) omitted ..." in prompt
    assert "total += 26;" not in prompt
    assert "total += 27; // ok" in prompt


def test_context_lines_none_keeps_everything() -> None:
    lines = split_lines(samples.long_source(total_lines=60, bug_line=30))
    options = IR4Options(context_lines=None)
    region = describe_region(lines, 30, 30, options=options)
    assert (region.context_start_line, region.context_end_line) == (1, 60)
    assert region.truncated_above == region.truncated_below == 0
    assert "omitted" not in render_ir4(lines, region, options)


def test_zero_context_keeps_only_the_region() -> None:
    options = IR4Options(context_lines=0)
    prompt = _render(samples.ONE_LINE, options)
    assert "public class Calculator {" not in prompt
    assert "            // return b;" in prompt


def test_region_at_the_start_has_no_context_above() -> None:
    region = describe_region(samples.NEAR_START.buggy_lines, 1, 1)
    assert region.context_start_line == 1
    assert region.truncated_above == 0


def test_region_at_the_end_has_no_context_below() -> None:
    lines = samples.NEAR_END.buggy_lines
    region = describe_region(lines, 5, len(lines))
    assert region.context_end_line == len(lines)
    assert region.truncated_below == 0


# --------------------------------------------------------------------------- #
# line numbering
# --------------------------------------------------------------------------- #
def test_line_numbers_are_optional_and_correct() -> None:
    options = IR4Options(show_line_numbers=True, context_lines=1)
    prompt = _render(samples.ONE_LINE, options)
    numbered = [line for line in prompt.split("\n") if "|" in line]
    assert numbered[0].startswith("   3 | ")
    assert "   4 |             // return b;" in prompt
    # markers and the fill slot get a blank gutter of the same width
    gutter = len("   4 | ")
    marker = next(line for line in prompt.split("\n") if "<FILL_ME>" in line)
    assert marker[:gutter] == " " * gutter


def test_line_numbers_off_by_default() -> None:
    assert "|" not in _render(samples.ONE_LINE)


# --------------------------------------------------------------------------- #
# fidelity: the Java code is never silently modified
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sample", samples.ALL_SAMPLES, ids=lambda s: s.name)
def test_region_survives_the_round_trip(sample: samples.Sample) -> None:
    prompt = _render(sample)
    assert recover_region_from_prompt(prompt) == sample.buggy_lines[
        sample.start - 1 : sample.end
    ]


@pytest.mark.parametrize("sample", samples.ALL_SAMPLES, ids=lambda s: s.name)
def test_context_lines_are_unchanged(sample: samples.Sample) -> None:
    rendered = set(_render(sample, IR4Options(context_lines=None)).split("\n"))
    outside = [
        line
        for number, line in enumerate(sample.buggy_lines, start=1)
        if not sample.start <= number <= sample.end
    ]
    for line in outside:
        assert line in rendered


def test_recovery_requires_markers() -> None:
    prompt = _render(samples.ONE_LINE, IR4Options(include_markers=False))
    with pytest.raises(RepresentationError, match="does not contain"):
        recover_region_from_prompt(prompt)


def test_recovery_refuses_numbered_prompts() -> None:
    options = IR4Options(show_line_numbers=True)
    with pytest.raises(RepresentationError, match="line numbers"):
        recover_region_from_prompt(_render(samples.ONE_LINE, options), options)


# --------------------------------------------------------------------------- #
# options validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [{"fill_token": ""}, {"comment_prefix": ""}, {"context_lines": -1}],
)
def test_invalid_options_rejected(kwargs: dict) -> None:
    with pytest.raises(RepresentationError):
        IR4Options(**kwargs)


def test_metadata_serialises() -> None:
    region = describe_region(samples.ONE_LINE.buggy_lines, 4, 4, file_path="C.java")
    payload = region.to_dict()
    assert payload["start_line"] == 4
    assert payload["end_line"] == 4
    assert payload["file_path"] == "C.java"
    assert payload["total_lines"] == 8
