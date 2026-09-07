"""The representation builder: the entry point the rest of the pipeline uses.

Two functions cover both directions:

    build_training_example(buggy_code, fixed_code, suspicious_start, suspicious_end)
    build_inference_input(buggy_code, suspicious_start, suspicious_end)

Both return structured objects carrying the prompt, the region metadata and
the original region text; the training example additionally carries the OR2
target.  Line numbers are 1-based and inclusive throughout.

    >>> example = build_training_example(buggy, fixed, 3, 3)
    >>> example.prompt.splitlines()[3]
    '    // buggy lines start here'
    >>> example.target
    '        return a;'
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from repairllama.representation.ir4 import (
    IR4Options,
    RegionMetadata,
    RepresentationError,
    describe_region,
    detect_newline,
    render_ir4,
    split_lines,
)
from repairllama.representation.or2 import (
    FixedRegion,
    OR2Options,
    OR2Prediction,
    align_fixed_region,
    build_or2_target,
    fixed_region_from_bounds,
    parse_or2_output,
    splice_region,
)

__all__ = [
    "RepresentationOptions",
    "InferenceInput",
    "TrainingExample",
    "build_inference_input",
    "build_training_example",
    "build_training_examples",
    "decode_prediction",
]

INPUT_REPRESENTATION = "IR4"
OUTPUT_REPRESENTATION = "OR2"


@dataclass(frozen=True)
class RepresentationOptions:
    """IR4 and OR2 options bundled so callers pass one object."""

    ir4: IR4Options = field(default_factory=IR4Options)
    or2: OR2Options = field(default_factory=OR2Options)

    @classmethod
    def from_config(cls, cfg: Any) -> "RepresentationOptions":
        """Build options from a :class:`repairllama.config.RepresentationConfig`.

        Only the implemented pair (IR4 x OR2) is accepted; asking for another
        pair is a configuration error rather than a silent fallback.
        """
        if cfg.input_format != "ir4" or cfg.output_format != "or2":
            raise RepresentationError(
                f"only ir4 x or2 is implemented, got "
                f"{cfg.input_format} x {cfg.output_format}"
            )
        return cls(
            ir4=IR4Options(
                fill_token=cfg.fill_token,
                region_start_marker=cfg.region_start_marker,
                region_end_marker=cfg.region_end_marker,
                comment_prefix=cfg.comment_prefix,
                context_lines=cfg.context_lines,
                show_line_numbers=cfg.show_line_numbers,
                include_file_path=cfg.include_file_path,
            ),
            or2=OR2Options(dedent=cfg.dedent_target, fill_token=cfg.fill_token),
        )


@dataclass(frozen=True)
class InferenceInput:
    """An IR4 prompt ready to be sent to the model."""

    prompt: str
    original_region: str
    region: RegionMetadata
    input_representation: str = INPUT_REPRESENTATION
    output_representation: str = OUTPUT_REPRESENTATION
    bug_id: Optional[str] = None
    source_lines: Tuple[str, ...] = ()

    @property
    def input(self) -> str:
        """Alias for :attr:`prompt`."""
        return self.prompt

    @property
    def original_region_lines(self) -> List[str]:
        return list(self.source_lines[self.region.start_index : self.region.end_index])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "prompt": self.prompt,
            "original_region": self.original_region,
            "region": self.region.to_dict(),
            "input_representation": self.input_representation,
            "output_representation": self.output_representation,
        }


@dataclass(frozen=True)
class TrainingExample:
    """An IR4 prompt paired with its OR2 target."""

    prompt: str
    target: str
    original_region: str
    fixed_region: str
    region: RegionMetadata
    fixed: FixedRegion
    input_representation: str = INPUT_REPRESENTATION
    output_representation: str = OUTPUT_REPRESENTATION
    bug_id: Optional[str] = None
    source_lines: Tuple[str, ...] = ()

    @property
    def input(self) -> str:
        """Alias for :attr:`prompt`."""
        return self.prompt

    @property
    def output(self) -> str:
        """Alias for :attr:`target`."""
        return self.target

    @property
    def is_deletion(self) -> bool:
        """True when the fix removes the region outright."""
        return self.fixed.is_deletion

    @property
    def is_identical(self) -> bool:
        """True when the 'fix' does not change the region — not trainable."""
        return self.original_region == self.fixed_region

    @property
    def changed_outside_region(self) -> bool:
        """True when the fix also edits lines outside the suspicious region."""
        return self.fixed.changed_outside_region

    def apply_to_source(self, replacement: Optional[Sequence[str]] = None) -> List[str]:
        """Splice ``replacement`` (default: the target) back into the source."""
        lines = list(replacement) if replacement is not None else list(self.fixed.lines)
        return splice_region(
            self.source_lines, self.region.start_line, self.region.end_line, lines
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "prompt": self.prompt,
            "target": self.target,
            "original_region": self.original_region,
            "fixed_region": self.fixed_region,
            "region": self.region.to_dict(),
            "fixed": self.fixed.to_dict(),
            "input_representation": self.input_representation,
            "output_representation": self.output_representation,
        }


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _prepare(
    code: str,
    suspicious_start: int,
    suspicious_end: int,
    options: RepresentationOptions,
    file_path: Optional[str],
) -> Tuple[List[str], RegionMetadata]:
    lines = split_lines(code)
    region = describe_region(
        lines,
        suspicious_start,
        suspicious_end,
        options=options.ir4,
        file_path=file_path,
        newline=detect_newline(code),
    )
    return lines, region


def build_inference_input(
    buggy_code: str,
    suspicious_start: int,
    suspicious_end: int,
    *,
    options: Optional[RepresentationOptions] = None,
    file_path: Optional[str] = None,
    bug_id: Optional[str] = None,
) -> InferenceInput:
    """Build the IR4 prompt for a bug whose fix is unknown.

    Args:
        buggy_code: the buggy Java unit (a method or a whole file).
        suspicious_start: first suspicious line, 1-based and inclusive.
        suspicious_end: last suspicious line, 1-based and inclusive.  Equal to
            ``suspicious_start`` for a one-line region.
        options: IR4/OR2 knobs; defaults to :class:`RepresentationOptions`.
        file_path: recorded in the metadata and, when
            ``IR4Options.include_file_path``, named in a header comment.
        bug_id: opaque identifier carried through to the result.

    Raises:
        RepresentationError: if the boundaries are malformed — inverted,
            below 1, past the end of the source, or not integers.
    """
    options = options or RepresentationOptions()
    lines, region = _prepare(
        buggy_code, suspicious_start, suspicious_end, options, file_path
    )
    return InferenceInput(
        prompt=render_ir4(lines, region, options.ir4),
        original_region="\n".join(lines[region.start_index : region.end_index]),
        region=region,
        bug_id=bug_id,
        source_lines=tuple(lines),
    )


def build_training_example(
    buggy_code: str,
    fixed_code: str,
    suspicious_start: int,
    suspicious_end: int,
    *,
    fixed_start: Optional[int] = None,
    fixed_end: Optional[int] = None,
    options: Optional[RepresentationOptions] = None,
    file_path: Optional[str] = None,
    bug_id: Optional[str] = None,
    strict: bool = False,
) -> TrainingExample:
    """Build an IR4 prompt and its OR2 target from a bug/fix pair.

    The fixed region is derived from ``fixed_code`` by line diff unless
    ``fixed_start``/``fixed_end`` are given (1-based, inclusive; pass
    ``fixed_end = fixed_start - 1`` for a deletion).

    Args:
        buggy_code: the buggy Java unit.
        fixed_code: the same unit after the fix.
        suspicious_start: first suspicious line in ``buggy_code``, 1-based.
        suspicious_end: last suspicious line in ``buggy_code``, 1-based.
        strict: when True, reject examples whose fix touches lines outside the
            suspicious region (those are usually mislocalized).

    Raises:
        RepresentationError: on malformed boundaries, or under ``strict`` when
            the fix reaches outside the region.
    """
    options = options or RepresentationOptions()
    buggy_lines, region = _prepare(
        buggy_code, suspicious_start, suspicious_end, options, file_path
    )
    fixed_lines = split_lines(fixed_code)

    if (fixed_start is None) != (fixed_end is None):
        raise RepresentationError(
            "fixed_start and fixed_end must be given together or not at all"
        )
    if fixed_start is not None and fixed_end is not None:
        fixed = fixed_region_from_bounds(fixed_lines, fixed_start, fixed_end)
    else:
        fixed = align_fixed_region(
            buggy_lines, fixed_lines, region.start_line, region.end_line
        )

    if strict and fixed.changed_outside_region:
        raise RepresentationError(
            f"the fix changes lines outside the suspicious region "
            f"({region.start_line}-{region.end_line}); "
            "widen the region or drop this example"
        )

    return TrainingExample(
        prompt=render_ir4(buggy_lines, region, options.ir4),
        target=build_or2_target(fixed.lines, options.or2),
        original_region="\n".join(buggy_lines[region.start_index : region.end_index]),
        fixed_region="\n".join(fixed.lines),
        region=region,
        fixed=fixed,
        bug_id=bug_id,
        source_lines=tuple(buggy_lines),
    )


def build_training_examples(
    records: Iterable[Dict[str, Any]],
    *,
    options: Optional[RepresentationOptions] = None,
    skip_invalid: bool = False,
    strict: bool = False,
) -> Iterator[TrainingExample]:
    """Map records to training examples.

    Each record needs ``buggy_code``, ``fixed_code``, ``suspicious_start`` and
    ``suspicious_end``; ``fixed_start``, ``fixed_end``, ``file_path`` and
    ``bug_id`` are optional.  With ``skip_invalid`` a malformed record is
    skipped instead of aborting the batch.
    """
    options = options or RepresentationOptions()
    for record in records:
        try:
            yield build_training_example(
                record["buggy_code"],
                record["fixed_code"],
                record["suspicious_start"],
                record["suspicious_end"],
                fixed_start=record.get("fixed_start"),
                fixed_end=record.get("fixed_end"),
                options=options,
                file_path=record.get("file_path"),
                bug_id=record.get("bug_id"),
                strict=strict,
            )
        except (RepresentationError, KeyError):
            if not skip_invalid:
                raise


def decode_prediction(
    completion: str,
    source: InferenceInput | TrainingExample,
    *,
    options: Optional[RepresentationOptions] = None,
) -> OR2Prediction:
    """Turn a raw model completion into replacement lines for ``source``."""
    options = options or RepresentationOptions()
    return parse_or2_output(completion, options.or2, region=source.region)


def with_context(
    options: RepresentationOptions, context_lines: Optional[int]
) -> RepresentationOptions:
    """Return a copy of ``options`` with a different context window."""
    return replace(options, ir4=replace(options.ir4, context_lines=context_lines))
