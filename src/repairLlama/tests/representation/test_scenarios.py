"""End-to-end IR4 x OR2 scenarios.

One test class per required bug shape.  Each checks the same three
properties, which together are the contract of the representation:

1. the prompt marks the region and offers exactly one fill slot;
2. the target is the replacement for that region and nothing more;
3. splicing the target back over lines start..end reproduces the fixed source.
"""

from __future__ import annotations

import pytest
import samples

from repairllama.representation import (
    IR4Options,
    RepresentationError,
    RepresentationOptions,
    build_inference_input,
    build_training_example,
    recover_region_from_prompt,
)


def assert_representation_contract(sample: samples.Sample) -> None:
    """The three invariants above, for any sample."""
    example = build_training_example(
        sample.buggy, sample.fixed, sample.start, sample.end, bug_id=sample.name
    )
    prompt_lines = example.prompt.split("\n")

    # 1. the region is marked exactly once and a single slot is offered
    assert prompt_lines.count(f"{example.region.indent}<FILL_ME>") == 1
    assert example.prompt.count("// buggy lines start here") == 1
    assert example.prompt.count("// buggy lines end here") == 1
    assert recover_region_from_prompt(example.prompt) == sample.buggy_lines[
        sample.start - 1 : sample.end
    ]

    # 2. the target is the replacement only
    assert example.target == sample.target
    assert example.original_region == sample.original_region

    # 3. it reconstructs the fix
    assert example.apply_to_source() == sample.fixed_lines


class TestOneLineBug:
    """A single wrong line inside an if-block."""

    sample = samples.ONE_LINE

    def test_contract(self) -> None:
        assert_representation_contract(self.sample)

    def test_region_is_one_line(self) -> None:
        example = build_training_example(
            self.sample.buggy, self.sample.fixed, 4, 4
        )
        assert example.region.num_lines == 1
        assert example.region.line_range == (4, 4)
        assert len(example.target.splitlines()) == 1

    def test_prompt_keeps_the_buggy_line_commented(self) -> None:
        prompt = build_inference_input(self.sample.buggy, 4, 4).prompt
        assert "            // return b;" in prompt.split("\n")

    def test_target_is_not_the_whole_function(self) -> None:
        example = build_training_example(self.sample.buggy, self.sample.fixed, 4, 4)
        assert "public int max" not in example.target


class TestMultiLineBug:
    """Two consecutive buggy lines replaced by two new lines."""

    sample = samples.MULTI_LINE

    def test_contract(self) -> None:
        assert_representation_contract(self.sample)

    def test_all_region_lines_are_commented_out(self) -> None:
        example = build_training_example(
            self.sample.buggy, self.sample.fixed, self.sample.start, self.sample.end
        )
        lines = example.prompt.split("\n")
        start = lines.index("        // buggy lines start here")
        end = lines.index("        // buggy lines end here")
        assert lines[start + 1 : end] == [
            "        // for (int i = 0; i <= values.length; i++) {",
            "            // total += values[i];",
        ]

    def test_multi_line_target(self) -> None:
        example = build_training_example(
            self.sample.buggy, self.sample.fixed, self.sample.start, self.sample.end
        )
        assert example.region.num_lines == 2
        assert len(example.target.splitlines()) == 2


class TestNestedIfElse:
    """A region spanning an else-if branch inside a nested conditional."""

    sample = samples.NESTED_IF

    def test_contract(self) -> None:
        assert_representation_contract(self.sample)

    def test_region_indent_is_the_branch_indent(self) -> None:
        example = build_training_example(
            self.sample.buggy, self.sample.fixed, self.sample.start, self.sample.end
        )
        assert example.region.indent == " " * 8
        assert example.region.indent_width == 8

    def test_inner_indentation_is_preserved_in_the_target(self) -> None:
        example = build_training_example(
            self.sample.buggy, self.sample.fixed, self.sample.start, self.sample.end
        )
        assert example.target.splitlines()[1] == '            return "B";'

    def test_untouched_nested_branch_stays_in_the_context(self) -> None:
        prompt = build_inference_input(
            self.sample.buggy, self.sample.start, self.sample.end
        ).prompt
        assert '                return "F";' in prompt.split("\n")


class TestMethodWithLoops:
    """A one-line bug inside a nested for-loop."""

    sample = samples.LOOPS

    def test_contract(self) -> None:
        assert_representation_contract(self.sample)

    def test_deep_indentation_is_preserved(self) -> None:
        example = build_training_example(
            self.sample.buggy, self.sample.fixed, self.sample.start, self.sample.end
        )
        assert example.region.indent == " " * 12
        assert example.target.startswith(" " * 12)

    def test_both_loop_headers_are_visible(self) -> None:
        prompt = build_inference_input(self.sample.buggy, 5, 5).prompt
        assert "        for (int row = 0; row < grid.length; row++) {" in prompt
        assert "            // for (int col = 0; col < grid.length; col++) {" in prompt


class TestRegionNearBeginning:
    """The suspicious region is line 1 — there is nothing above it."""

    sample = samples.NEAR_START

    def test_contract(self) -> None:
        assert_representation_contract(self.sample)

    def test_no_context_above(self) -> None:
        example = build_training_example(self.sample.buggy, self.sample.fixed, 1, 1)
        assert example.region.context_start_line == 1
        assert example.region.truncated_above == 0

    def test_prompt_opens_with_the_marker(self) -> None:
        options = RepresentationOptions(
            ir4=IR4Options(include_region_header=False, include_file_path=False)
        )
        prompt = build_inference_input(self.sample.buggy, 1, 1, options=options).prompt
        assert prompt.split("\n")[0] == "// buggy lines start here"

    def test_region_indent_is_empty_at_column_zero(self) -> None:
        example = build_training_example(self.sample.buggy, self.sample.fixed, 1, 1)
        assert example.region.indent == ""
        assert "<FILL_ME>" in example.prompt.split("\n")


class TestRegionNearEnd:
    """The suspicious region touches the last line of the unit."""

    sample = samples.NEAR_END

    def test_contract(self) -> None:
        assert_representation_contract(self.sample)

    def test_contract_through_the_final_line(self) -> None:
        assert_representation_contract(samples.NEAR_END_THROUGH_LAST)

    def test_no_context_below(self) -> None:
        total = len(self.sample.buggy_lines)
        example = build_training_example(
            self.sample.buggy, self.sample.fixed, 5, total
        )
        assert example.region.context_end_line == total
        assert example.region.truncated_below == 0

    def test_fill_slot_is_the_last_line_of_the_prompt(self) -> None:
        total = len(self.sample.buggy_lines)
        prompt = build_inference_input(self.sample.buggy, 5, total).prompt
        assert prompt.split("\n")[-1].strip() == "<FILL_ME>"

    def test_trailing_context_kept_when_the_region_stops_short(self) -> None:
        prompt = build_inference_input(self.sample.buggy, 5, 5).prompt
        assert prompt.split("\n")[-1] == "}"


class TestMalformedRegionBoundaries:
    """Bad boundaries fail loudly; they are never clamped or guessed."""

    sample = samples.ONE_LINE

    @pytest.mark.parametrize(
        ("start", "end", "message"),
        [
            (0, 4, "1-based"),
            (-3, -1, "1-based"),
            (5, 4, "must be >= suspicious_start"),
            (4, 3, "must be >= suspicious_start"),
            (9, 9, "past the end"),
            (4, 100, "past the end"),
        ],
    )
    def test_inference_rejects_bad_bounds(
        self, start: int, end: int, message: str
    ) -> None:
        with pytest.raises(RepresentationError, match=message):
            build_inference_input(self.sample.buggy, start, end)

    @pytest.mark.parametrize(("start", "end"), [(0, 4), (5, 4), (4, 99)])
    def test_training_rejects_bad_bounds(self, start: int, end: int) -> None:
        with pytest.raises(RepresentationError):
            build_training_example(self.sample.buggy, self.sample.fixed, start, end)

    @pytest.mark.parametrize("value", ["4", 4.0, None, True, False])
    def test_non_integer_bounds_rejected(self, value: object) -> None:
        with pytest.raises(RepresentationError, match="1-based"):
            build_inference_input(self.sample.buggy, value, 4)  # type: ignore[arg-type]

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(RepresentationError, match="no lines"):
            build_inference_input("", 1, 1)

    def test_whitespace_only_source_is_a_valid_region(self) -> None:
        """Blank lines are still lines; only an empty unit is malformed."""
        result = build_inference_input("\n\n\n", 2, 2)
        assert result.region.is_blank_region is True
        assert result.original_region == ""

    def test_error_message_names_the_bounds(self) -> None:
        with pytest.raises(RepresentationError) as info:
            build_inference_input(self.sample.buggy, 4, 99)
        assert "99" in str(info.value)
        assert "8 lines" in str(info.value)

    def test_bad_bounds_do_not_produce_a_partial_result(self) -> None:
        with pytest.raises(RepresentationError):
            build_training_example(self.sample.buggy, self.sample.fixed, 4, 99)


# --------------------------------------------------------------------------- #
# cross-cutting: every sample, both entry points
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sample", samples.ALL_SAMPLES, ids=lambda s: s.name)
def test_contract_holds_for_every_sample(sample: samples.Sample) -> None:
    assert_representation_contract(sample)


@pytest.mark.parametrize("sample", samples.ALL_SAMPLES, ids=lambda s: s.name)
def test_inference_and_training_prompts_agree(sample: samples.Sample) -> None:
    """The prompt must not depend on knowing the fix."""
    inference = build_inference_input(sample.buggy, sample.start, sample.end)
    training = build_training_example(
        sample.buggy, sample.fixed, sample.start, sample.end
    )
    assert inference.prompt == training.prompt
    assert inference.original_region == training.original_region
