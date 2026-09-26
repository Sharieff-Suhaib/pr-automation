"""The cleaner: Java scanning, single-function detection, and each drop rule."""

from __future__ import annotations

import java_pairs as jp
import pytest

from repairllama.data.cleaner import (
    Cleaner,
    CleanerOptions,
    changed_methods,
    clean_pairs,
    diff_size,
    find_methods,
    looks_like_test,
    method_containing,
    strip_comments,
)
from repairllama.data.models import BugFixPair, RejectionReason


# --------------------------------------------------------------------------- #
# strip_comments
# --------------------------------------------------------------------------- #
def test_strip_comments_preserves_length_and_lines() -> None:
    source = "int a; // note\n/* block\n   comment */ int b;\n"
    stripped = strip_comments(source)
    assert len(stripped) == len(source)
    assert stripped.count("\n") == source.count("\n")


def test_line_comments_are_blanked() -> None:
    assert strip_comments("int a; // set a\n").rstrip() == "int a;"


def test_block_comments_are_blanked_across_lines() -> None:
    stripped = strip_comments("a;\n/* x\n y */\nb;\n")
    assert "x" not in stripped and "y" not in stripped
    assert "a;" in stripped and "b;" in stripped


def test_string_literals_are_kept_by_default() -> None:
    assert '"// not a comment"' in strip_comments('String s = "// not a comment";')


def test_string_literals_can_be_blanked() -> None:
    stripped = strip_comments('String s = "text";', blank_strings=True)
    assert "text" not in stripped
    assert stripped.startswith("String s = ")


def test_escaped_quotes_do_not_end_a_literal() -> None:
    source = 'String s = "a \\" // still a string"; int x;'
    assert "int x;" in strip_comments(source)


def test_braces_in_comments_and_literals_are_ignored() -> None:
    stripped = strip_comments(jp.TRICKY_BRACES_BUGGY, blank_strings=True)
    assert stripped.count("{") == 2  # class and method only
    assert stripped.count("}") == 2


# --------------------------------------------------------------------------- #
# find_methods
# --------------------------------------------------------------------------- #
def test_find_methods_locates_each_method() -> None:
    spans = find_methods(jp.TWO_METHODS_BUGGY)
    assert [span.name for span in spans] == ["first", "last"]
    assert spans[0].start_line == 2 and spans[0].end_line == 4


def test_find_methods_handles_comments_and_literal_braces() -> None:
    spans = find_methods(jp.TRICKY_BRACES_BUGGY)
    assert [span.name for span in spans] == ["render"]
    assert (spans[0].start_line, spans[0].end_line) == (2, 6)


def test_find_methods_ignores_control_flow_blocks() -> None:
    source = """\
public class C {
    public void run(int n) {
        if (n > 0) {
            while (n > 0) {
                n--;
            }
        }
    }
}
"""
    assert [span.name for span in find_methods(source)] == ["run"]


def test_find_methods_handles_annotations_generics_and_throws() -> None:
    source = """\
public class C {
    @Override
    public <T extends Number> List<T> convert(List<T> in) throws IOException {
        return in;
    }

    public C(int size) {
        this.size = size;
    }
}
"""
    assert [span.name for span in find_methods(source)] == ["convert", "C"]


def test_find_methods_sees_inner_class_methods() -> None:
    source = """\
public class Outer {
    public int outer() {
        return 1;
    }

    static class Inner {
        public int inner() {
            return 2;
        }
    }
}
"""
    assert [span.name for span in find_methods(source)] == ["outer", "inner"]


def test_find_methods_finds_nothing_in_a_field_only_class() -> None:
    assert find_methods(jp.FIELD_ONLY_BUGGY) == []


def test_unterminated_method_is_not_guessed() -> None:
    assert find_methods("public class C {\n    public int f() {\n        return 1;\n") == []


def test_method_containing_picks_the_innermost() -> None:
    spans = find_methods(jp.TWO_METHODS_BUGGY)
    assert method_containing(spans, 3).name == "first"
    assert method_containing(spans, 7).name == "last"
    assert method_containing(spans, 1) is None


# --------------------------------------------------------------------------- #
# change analysis
# --------------------------------------------------------------------------- #
def test_changed_methods_of_a_one_line_fix() -> None:
    assert [span.name for span in changed_methods(jp.calculator_pair())] == ["max"]


def test_changed_methods_of_a_two_method_fix() -> None:
    pair = BugFixPair("t", jp.TWO_METHODS_BUGGY, jp.TWO_METHODS_FIXED)
    assert [span.name for span in changed_methods(pair)] == ["first", "last"]


def test_changed_methods_is_empty_without_a_method() -> None:
    pair = BugFixPair("f", jp.FIELD_ONLY_BUGGY, jp.FIELD_ONLY_FIXED)
    assert changed_methods(pair) == []


def test_diff_size_counts_both_sides() -> None:
    pair = BugFixPair("d", "a\nb\nc\n", "a\nB\n")
    assert diff_size(pair) == 3  # two removed, one added


def test_diff_size_of_an_unchanged_pair() -> None:
    assert diff_size(BugFixPair("s", "a\n", "a\n")) == 0


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/test/java/FooTest.java", True),
        ("src/main/java/Foo.java", False),
        ("tests/BarIT.java", True),
        ("src/main/java/testing/Helper.java", False),
    ],
)
def test_test_paths_are_recognised(path: str, expected: bool) -> None:
    pair = BugFixPair("x", "class A {}", "class B {}", file_path=path)
    assert looks_like_test(pair) is expected


def test_junit_annotations_mark_test_code() -> None:
    pair = BugFixPair("x", jp.TEST_CLASS_BUGGY, jp.TEST_CLASS_FIXED)
    assert looks_like_test(pair) is True


# --------------------------------------------------------------------------- #
# the cleaning rules
# --------------------------------------------------------------------------- #
def test_a_clean_pair_survives_and_gains_a_region() -> None:
    kept, rejection = Cleaner().clean(jp.calculator_pair())
    assert rejection is None
    assert kept is not None and kept.has_region
    assert (kept.suspicious_start, kept.suspicious_end) == (5, 5)


def test_existing_region_is_left_alone() -> None:
    pair = jp.calculator_pair().with_region(4, 6)
    kept, _ = Cleaner().clean(pair)
    assert (kept.suspicious_start, kept.suspicious_end) == (4, 6)


def _reason(pair: BugFixPair, options: CleanerOptions | None = None) -> str | None:
    rejection = Cleaner(options).inspect(pair)
    return rejection.reason.value if rejection else None


def test_identical_pairs_are_dropped() -> None:
    pair = BugFixPair("s", jp.CALCULATOR_BUGGY, jp.CALCULATOR_BUGGY)
    assert _reason(pair) == RejectionReason.IDENTICAL.value


def test_blank_sources_are_dropped() -> None:
    assert _reason(BugFixPair("b", "   \n", "class A {}")) == (
        RejectionReason.EMPTY_SOURCE.value
    )


def test_multi_method_changes_are_dropped() -> None:
    pair = BugFixPair("t", jp.TWO_METHODS_BUGGY, jp.TWO_METHODS_FIXED)
    assert _reason(pair) == RejectionReason.NOT_SINGLE_FUNCTION.value


def test_changes_outside_any_method_are_dropped() -> None:
    pair = BugFixPair("f", jp.FIELD_ONLY_BUGGY, jp.FIELD_ONLY_FIXED)
    assert _reason(pair) == RejectionReason.NOT_SINGLE_FUNCTION.value


def test_single_function_rule_can_be_disabled() -> None:
    pair = BugFixPair("f", jp.FIELD_ONLY_BUGGY, jp.FIELD_ONLY_FIXED)
    assert _reason(pair, CleanerOptions(require_single_function=False)) is None


def test_multi_method_rejection_names_the_methods() -> None:
    pair = BugFixPair("t", jp.TWO_METHODS_BUGGY, jp.TWO_METHODS_FIXED)
    assert "first" in Cleaner().inspect(pair).detail


def test_test_only_changes_are_dropped() -> None:
    pair = BugFixPair(
        "t",
        jp.TEST_CLASS_BUGGY,
        jp.TEST_CLASS_FIXED,
        file_path="src/test/java/CalculatorTest.java",
    )
    assert _reason(pair) == RejectionReason.TEST_ONLY.value


def test_test_only_rule_can_be_disabled() -> None:
    pair = BugFixPair(
        "t", jp.TEST_CLASS_BUGGY, jp.TEST_CLASS_FIXED, file_path="src/test/java/T.java"
    )
    assert _reason(pair, CleanerOptions(drop_test_only_changes=False)) is None


def test_oversized_diffs_are_dropped() -> None:
    assert _reason(jp.big_diff_pair()) == RejectionReason.DIFF_TOO_LARGE.value


def test_diff_bounds_are_configurable() -> None:
    pair = jp.big_diff_pair(changed_lines=40)
    assert _reason(pair, CleanerOptions(max_diff_lines=200)) is None


def test_undersized_diffs_are_dropped() -> None:
    options = CleanerOptions(min_diff_lines=5, max_diff_lines=50)
    assert _reason(jp.calculator_pair(), options) == RejectionReason.DIFF_TOO_SMALL.value


def test_out_of_range_regions_are_dropped() -> None:
    pair = jp.calculator_pair().with_region(1, 999)
    assert _reason(pair) == RejectionReason.INVALID_REGION.value


def test_inverted_regions_are_dropped() -> None:
    pair = BugFixPair(
        "x",
        jp.CALCULATOR_BUGGY,
        jp.CALCULATOR_FIXED,
        suspicious_start=6,
        suspicious_end=4,
    )
    assert _reason(pair) == RejectionReason.INVALID_REGION.value


def test_max_region_lines_is_enforced() -> None:
    pair = jp.calculator_pair().with_region(2, 8)
    assert _reason(pair, CleanerOptions(max_region_lines=3)) == (
        RejectionReason.DIFF_TOO_LARGE.value
    )


def test_nonsense_options_are_rejected() -> None:
    with pytest.raises(ValueError):
        CleanerOptions(min_diff_lines=10, max_diff_lines=2)


# --------------------------------------------------------------------------- #
# running over a corpus
# --------------------------------------------------------------------------- #
def test_clean_pairs_accounts_for_every_input() -> None:
    pairs = [
        jp.calculator_pair("good-1"),
        BugFixPair("two", jp.TWO_METHODS_BUGGY, jp.TWO_METHODS_FIXED),
        BugFixPair("same", jp.CALCULATOR_BUGGY, jp.CALCULATOR_BUGGY),
        jp.big_diff_pair("big"),
    ]
    result = clean_pairs(pairs)
    assert result.received == len(pairs)
    assert [pair.bug_id for pair in result.kept] == ["good-1"]
    assert result.counts_by_reason() == {
        "not_single_function": 1,
        "identical": 1,
        "diff_too_large": 1,
    }


def test_rejections_carry_the_stage_and_bug_id() -> None:
    result = clean_pairs([BugFixPair("two", jp.TWO_METHODS_BUGGY, jp.TWO_METHODS_FIXED)])
    rejection = result.rejections[0]
    assert rejection.bug_id == "two"
    assert rejection.stage == "clean"
    assert rejection.detail
