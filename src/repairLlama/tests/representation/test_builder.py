"""The public builder API: structured results, options, batch and decoding."""

from __future__ import annotations

import pytest
import samples

from repairllama.config import RepairConfig
from repairllama.representation import (
    IR4Options,
    OR2Options,
    RepresentationError,
    RepresentationOptions,
    build_inference_input,
    build_training_example,
    build_training_examples,
    decode_prediction,
    with_context,
)


# --------------------------------------------------------------------------- #
# build_inference_input
# --------------------------------------------------------------------------- #
def test_inference_input_is_structured() -> None:
    sample = samples.ONE_LINE
    result = build_inference_input(
        sample.buggy, sample.start, sample.end, file_path="Calculator.java", bug_id="C-1"
    )
    assert "<FILL_ME>" in result.prompt
    assert result.original_region == "            return b;"
    assert result.region.start_line == 4
    assert result.region.end_line == 4
    assert result.region.file_path == "Calculator.java"
    assert result.bug_id == "C-1"
    assert result.input_representation == "IR4"
    assert result.output_representation == "OR2"


def test_inference_input_exposes_an_input_alias() -> None:
    result = build_inference_input(samples.LOOPS.buggy, 5, 5)
    assert result.input == result.prompt
    assert result.original_region_lines == [samples.LOOPS.buggy_lines[4]]


def test_inference_input_never_contains_the_fix() -> None:
    """Inference has no target — only the prompt and the region."""
    result = build_inference_input(samples.ONE_LINE.buggy, 4, 4)
    assert not hasattr(result, "target")


def test_inference_input_serialises() -> None:
    payload = build_inference_input(samples.ONE_LINE.buggy, 4, 4, bug_id="C-1").to_dict()
    assert payload["bug_id"] == "C-1"
    assert payload["region"]["start_line"] == 4
    assert "prompt" in payload and "target" not in payload


# --------------------------------------------------------------------------- #
# build_training_example
# --------------------------------------------------------------------------- #
def test_training_example_is_structured() -> None:
    sample = samples.MULTI_LINE
    example = build_training_example(
        sample.buggy, sample.fixed, sample.start, sample.end, bug_id="S-1"
    )
    assert example.prompt.count("<FILL_ME>") == 1
    assert example.target == sample.target
    assert example.original_region == sample.original_region
    assert example.fixed_region == sample.target
    assert example.region.line_range == (4, 5)
    assert example.fixed.start_line == 4 and example.fixed.end_line == 5
    assert example.bug_id == "S-1"


def test_training_example_aliases() -> None:
    example = build_training_example(samples.ONE_LINE.buggy, samples.ONE_LINE.fixed, 4, 4)
    assert example.input == example.prompt
    assert example.output == example.target


def test_target_is_only_the_region_not_the_function() -> None:
    sample = samples.LOOPS
    example = build_training_example(sample.buggy, sample.fixed, sample.start, sample.end)
    assert example.target == sample.target
    assert "public class Matrix" not in example.target
    assert "public int sum" not in example.target
    assert len(example.target.splitlines()) == 1


@pytest.mark.parametrize("sample", samples.ALL_SAMPLES, ids=lambda s: s.name)
def test_target_reconstructs_the_fixed_source(sample: samples.Sample) -> None:
    example = build_training_example(sample.buggy, sample.fixed, sample.start, sample.end)
    assert example.apply_to_source() == sample.fixed_lines


def test_deletion_example() -> None:
    sample = samples.DELETION
    example = build_training_example(sample.buggy, sample.fixed, sample.start, sample.end)
    assert example.target == ""
    assert example.is_deletion is True
    assert example.apply_to_source() == sample.fixed_lines


def test_identical_region_is_flagged() -> None:
    sample = samples.ONE_LINE
    example = build_training_example(sample.buggy, sample.buggy, sample.start, sample.end)
    assert example.is_identical is True


def test_edits_outside_the_region_are_reported() -> None:
    sample = samples.ONE_LINE
    example = build_training_example(
        sample.buggy, samples.ONE_LINE_FIXED_WITH_OUTSIDE_EDIT, sample.start, sample.end
    )
    assert example.changed_outside_region is True


def test_strict_mode_rejects_edits_outside_the_region() -> None:
    sample = samples.ONE_LINE
    with pytest.raises(RepresentationError, match="outside the suspicious region"):
        build_training_example(
            sample.buggy,
            samples.ONE_LINE_FIXED_WITH_OUTSIDE_EDIT,
            sample.start,
            sample.end,
            strict=True,
        )


def test_explicit_fixed_bounds_override_alignment() -> None:
    sample = samples.MULTI_LINE
    example = build_training_example(
        sample.buggy, sample.fixed, sample.start, sample.end, fixed_start=4, fixed_end=5
    )
    assert example.target == sample.target
    assert example.fixed.inferred is False


def test_half_specified_fixed_bounds_rejected() -> None:
    sample = samples.ONE_LINE
    with pytest.raises(RepresentationError, match="together"):
        build_training_example(
            sample.buggy, sample.fixed, sample.start, sample.end, fixed_start=4
        )


def test_training_example_serialises() -> None:
    sample = samples.NESTED_IF
    payload = build_training_example(
        sample.buggy, sample.fixed, sample.start, sample.end
    ).to_dict()
    assert payload["target"] == sample.target
    assert payload["region"]["num_lines"] == 3
    assert payload["fixed"]["num_lines"] == 3
    assert payload["input_representation"] == "IR4"


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def test_options_from_config() -> None:
    cfg = RepairConfig.default().representation
    options = RepresentationOptions.from_config(cfg)
    assert options.ir4.fill_token == cfg.fill_token
    assert options.ir4.context_lines == cfg.context_lines
    assert options.or2.dedent is cfg.dedent_target


def test_options_from_config_rejects_unimplemented_pairs() -> None:
    cfg = RepairConfig.default().override({"representation.output_format": "or1"})
    with pytest.raises(RepresentationError, match="ir4 x or2"):
        RepresentationOptions.from_config(cfg.representation)


def test_shipped_config_yields_ir4_or2(shipped_config: RepairConfig) -> None:
    options = RepresentationOptions.from_config(shipped_config.representation)
    example = build_training_example(
        samples.ONE_LINE.buggy, samples.ONE_LINE.fixed, 4, 4, options=options
    )
    assert example.target == samples.ONE_LINE.target


def test_custom_fill_token_and_markers() -> None:
    options = RepresentationOptions(
        ir4=IR4Options(
            fill_token="[[FIX]]",
            region_start_marker="// >>> bug",
            region_end_marker="// <<< bug",
        )
    )
    prompt = build_inference_input(samples.ONE_LINE.buggy, 4, 4, options=options).prompt
    assert "[[FIX]]" in prompt and "<FILL_ME>" not in prompt
    assert "// >>> bug" in prompt and "// <<< bug" in prompt


def test_with_context_returns_a_copy() -> None:
    base = RepresentationOptions()
    widened = with_context(base, None)
    assert widened.ir4.context_lines is None
    assert base.ir4.context_lines == 10


def test_dedented_target_option() -> None:
    sample = samples.NESTED_IF
    options = RepresentationOptions(or2=OR2Options(dedent=True))
    example = build_training_example(
        sample.buggy, sample.fixed, sample.start, sample.end, options=options
    )
    assert example.target.startswith("} else if")
    assert example.region.indent == "        "


# --------------------------------------------------------------------------- #
# batch building and decoding
# --------------------------------------------------------------------------- #
def _record(sample: samples.Sample) -> dict:
    return {
        "bug_id": sample.name,
        "buggy_code": sample.buggy,
        "fixed_code": sample.fixed,
        "suspicious_start": sample.start,
        "suspicious_end": sample.end,
    }


def test_batch_building() -> None:
    records = [_record(sample) for sample in samples.ALL_SAMPLES]
    built = list(build_training_examples(records))
    assert len(built) == len(records)
    assert [example.bug_id for example in built] == [s.name for s in samples.ALL_SAMPLES]


def test_batch_propagates_malformed_records() -> None:
    bad = _record(samples.ONE_LINE) | {"suspicious_end": 99}
    with pytest.raises(RepresentationError):
        list(build_training_examples([bad]))


def test_batch_can_skip_invalid_records() -> None:
    good = _record(samples.ONE_LINE)
    bad = _record(samples.ONE_LINE) | {"suspicious_start": 0}
    missing_field = {"buggy_code": "a"}
    built = list(build_training_examples([good, bad, missing_field], skip_invalid=True))
    assert len(built) == 1


def test_decode_prediction_against_an_inference_input() -> None:
    result = build_inference_input(samples.ONE_LINE.buggy, 4, 4)
    prediction = decode_prediction("```java\n            return a;\n```", result)
    assert prediction.lines == ("            return a;",)


def test_decoded_prediction_can_be_spliced_back() -> None:
    from repairllama.representation import splice_region

    sample = samples.ONE_LINE
    result = build_inference_input(sample.buggy, sample.start, sample.end)
    prediction = decode_prediction("            return a;", result)
    patched = splice_region(
        list(result.source_lines), sample.start, sample.end, prediction.lines
    )
    assert patched == sample.fixed_lines
