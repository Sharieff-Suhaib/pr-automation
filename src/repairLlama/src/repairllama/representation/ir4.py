"""IR4: the input representation for RepairLLaMA-style Java repair.

IR4 keeps the buggy code visible instead of deleting it.  The suspicious
region is bracketed by marker comments, each buggy line is commented out
verbatim, and a fill token marks where the replacement goes::

    public int max(int a, int b) {
        // suspicious region: lines 2-4 of 6
        // buggy lines start here
        //     if (a > b) {
        //         return b;
        //     }
        // buggy lines end here
        <FILL_ME>
        return a;
    }

The model therefore sees *what the bug was* as well as *where it is*, which
is what separates IR4 from IR3 (buggy lines removed) and IR2 (buggy lines
marked but no fill token).

Line-number contract
--------------------
Every line number in this module is **1-based** and every region is
**inclusive of both endpoints**: ``suspicious_start=3, suspicious_end=3``
selects exactly line 3.  Out-of-range or inverted bounds raise
:class:`RepresentationError` — they are never clamped.

Fidelity contract
-----------------
Java source is never rewritten.  Context lines are emitted byte-for-byte,
buggy lines are only prefixed with a comment marker after their own
indentation, and :func:`uncomment_line` inverts that exactly.  Anything the
representation drops (context outside the window) is announced by an explicit
truncation comment rather than removed silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "RepresentationError",
    "IR4Options",
    "RegionMetadata",
    "DEFAULT_FILL_TOKEN",
    "split_lines",
    "detect_newline",
    "validate_region",
    "describe_region",
    "extract_region",
    "common_indent",
    "comment_line",
    "uncomment_line",
    "render_ir4",
    "recover_region_from_prompt",
]

DEFAULT_FILL_TOKEN = "<FILL_ME>"
DEFAULT_REGION_START_MARKER = "// buggy lines start here"
DEFAULT_REGION_END_MARKER = "// buggy lines end here"


class RepresentationError(ValueError):
    """Raised for malformed region boundaries or unusable source input."""


@dataclass(frozen=True)
class IR4Options:
    """Knobs for rendering an IR4 prompt.

    ``context_lines`` is the number of lines kept on each side of the
    suspicious region; ``None`` keeps the whole input.
    """

    fill_token: str = DEFAULT_FILL_TOKEN
    region_start_marker: str = DEFAULT_REGION_START_MARKER
    region_end_marker: str = DEFAULT_REGION_END_MARKER
    comment_prefix: str = "//"
    context_lines: Optional[int] = 10
    show_line_numbers: bool = False
    include_file_path: bool = True
    include_region_header: bool = True
    include_markers: bool = True
    line_number_format: str = "{number:>4} | "
    truncation_format: str = "// ... {count} line(s) omitted ..."

    def __post_init__(self) -> None:
        if not self.fill_token:
            raise RepresentationError("fill_token must not be empty")
        if not self.comment_prefix:
            raise RepresentationError("comment_prefix must not be empty")
        if self.context_lines is not None and self.context_lines < 0:
            raise RepresentationError(
                f"context_lines must be >= 0 or None, got {self.context_lines!r}"
            )


@dataclass(frozen=True)
class RegionMetadata:
    """Everything known about a suspicious region, in explicit line numbers."""

    start_line: int              # 1-based, inclusive
    end_line: int                # 1-based, inclusive
    num_lines: int
    total_lines: int             # lines in the whole source unit
    indent: str                  # common leading whitespace of the region
    indent_width: int            # width in columns, tabs expanded to 4
    context_start_line: int      # 1-based, inclusive
    context_end_line: int        # 1-based, inclusive
    truncated_above: int         # lines dropped before the context window
    truncated_below: int         # lines dropped after the context window
    is_blank_region: bool
    file_path: Optional[str] = None
    newline: str = "\n"
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def start_index(self) -> int:
        """0-based index of the first region line."""
        return self.start_line - 1

    @property
    def end_index(self) -> int:
        """0-based index *past* the last region line (slice-friendly)."""
        return self.end_line

    @property
    def line_range(self) -> Tuple[int, int]:
        return (self.start_line, self.end_line)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "num_lines": self.num_lines,
            "total_lines": self.total_lines,
            "indent": self.indent,
            "indent_width": self.indent_width,
            "context_start_line": self.context_start_line,
            "context_end_line": self.context_end_line,
            "truncated_above": self.truncated_above,
            "truncated_below": self.truncated_below,
            "is_blank_region": self.is_blank_region,
            "file_path": self.file_path,
            "newline": self.newline,
            **({"extras": dict(self.extras)} if self.extras else {}),
        }


# --------------------------------------------------------------------------- #
# source handling
# --------------------------------------------------------------------------- #
def detect_newline(text: str) -> str:
    """Return the dominant line terminator so output can round-trip."""
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text and "\n" not in text:
        return "\r"
    return "\n"


def split_lines(text: str) -> List[str]:
    """Split source into lines without their terminators.

    A trailing newline does not produce a phantom final line, so
    ``"a\\nb\\n"`` and ``"a\\nb"`` both have 2 lines and identical numbering.
    """
    if not isinstance(text, str):
        raise RepresentationError(f"source must be a string, got {type(text).__name__}")
    return text.splitlines()


def _as_line_number(value: Any, name: str) -> int:
    # bool is an int subclass; a True/False line number is always a mistake.
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepresentationError(
            f"{name} must be an int line number (1-based), got {value!r}"
        )
    return value


def validate_region(lines: Sequence[str], start: int, end: int) -> None:
    """Validate a 1-based inclusive region against ``lines``.

    Raises :class:`RepresentationError` with a message naming the offending
    bound.  Boundaries are never clamped: a caller asking for line 999 of a
    20-line method has a bug worth surfacing.
    """
    start = _as_line_number(start, "suspicious_start")
    end = _as_line_number(end, "suspicious_end")
    total = len(lines)
    if total == 0:
        raise RepresentationError("source contains no lines")
    if start < 1:
        raise RepresentationError(
            f"suspicious_start must be >= 1 (line numbers are 1-based), got {start}"
        )
    if end < start:
        raise RepresentationError(
            f"suspicious_end ({end}) must be >= suspicious_start ({start}); "
            "the region is inclusive of both endpoints"
        )
    if start > total:
        raise RepresentationError(
            f"suspicious_start ({start}) is past the end of the source ({total} lines)"
        )
    if end > total:
        raise RepresentationError(
            f"suspicious_end ({end}) is past the end of the source ({total} lines)"
        )


def extract_region(lines: Sequence[str], start: int, end: int) -> List[str]:
    """Return the region's lines verbatim (1-based, inclusive)."""
    validate_region(lines, start, end)
    return list(lines[start - 1 : end])


def common_indent(lines: Sequence[str]) -> str:
    """Longest leading-whitespace prefix shared by all non-blank lines."""
    indents = [
        line[: len(line) - len(line.lstrip(" \t"))]
        for line in lines
        if line.strip()
    ]
    if not indents:
        # An all-blank region: fall back to the whitespace of the first line.
        return lines[0] if lines and not lines[0].strip() else ""
    prefix = os.path.commonprefix(indents)
    return prefix


def indent_width(indent: str, tab_size: int = 4) -> int:
    """Visual width of an indent string, expanding tabs."""
    width = 0
    for char in indent:
        width = width + (tab_size - width % tab_size) if char == "\t" else width + 1
    return width


def describe_region(
    lines: Sequence[str],
    start: int,
    end: int,
    *,
    options: Optional[IR4Options] = None,
    file_path: Optional[str] = None,
    newline: str = "\n",
) -> RegionMetadata:
    """Validate a region and compute its metadata and context window."""
    options = options or IR4Options()
    validate_region(lines, start, end)
    total = len(lines)
    region = list(lines[start - 1 : end])

    if options.context_lines is None:
        ctx_start, ctx_end = 1, total
    else:
        ctx_start = max(1, start - options.context_lines)
        ctx_end = min(total, end + options.context_lines)

    indent = common_indent(region)
    return RegionMetadata(
        start_line=start,
        end_line=end,
        num_lines=len(region),
        total_lines=total,
        indent=indent,
        indent_width=indent_width(indent),
        context_start_line=ctx_start,
        context_end_line=ctx_end,
        truncated_above=ctx_start - 1,
        truncated_below=total - ctx_end,
        is_blank_region=not any(line.strip() for line in region),
        file_path=file_path,
        newline=newline,
    )


# --------------------------------------------------------------------------- #
# commenting (exactly invertible)
# --------------------------------------------------------------------------- #
def comment_line(line: str, prefix: str = "//") -> str:
    """Comment out one line, preserving its indentation and its text.

    The marker is inserted *after* the leading whitespace so the block keeps
    its shape; :func:`uncomment_line` is the exact inverse.
    """
    stripped = line.lstrip(" \t")
    indent = line[: len(line) - len(stripped)]
    if not stripped:
        return f"{indent}{prefix}"
    return f"{indent}{prefix} {stripped}"


def uncomment_line(line: str, prefix: str = "//") -> str:
    """Inverse of :func:`comment_line`; returns ``line`` unchanged if unmarked."""
    stripped = line.lstrip(" \t")
    indent = line[: len(line) - len(stripped)]
    if not stripped.startswith(prefix):
        return line
    rest = stripped[len(prefix) :]
    if rest.startswith(" "):
        rest = rest[1:]
    return f"{indent}{rest}"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _gutter(number: Optional[int], total: int, options: IR4Options) -> str:
    if not options.show_line_numbers:
        return ""
    if number is None:
        return " " * len(options.line_number_format.format(number=total))
    return options.line_number_format.format(number=number)


def render_ir4(
    lines: Sequence[str],
    region: RegionMetadata,
    options: Optional[IR4Options] = None,
) -> str:
    """Render the IR4 prompt for ``region`` over ``lines``.

    ``lines`` must be the same sequence ``region`` was derived from.
    """
    options = options or IR4Options()
    if region.total_lines != len(lines):
        raise RepresentationError(
            f"region describes {region.total_lines} lines but {len(lines)} were given; "
            "the metadata and the source have diverged"
        )
    total = region.total_lines
    indent = region.indent
    out: List[str] = []

    def emit(text: str, number: Optional[int] = None) -> None:
        out.append(f"{_gutter(number, total, options)}{text}")

    if options.include_file_path and region.file_path:
        emit(f"{options.comment_prefix} file: {region.file_path}")
    if options.include_region_header:
        emit(
            f"{options.comment_prefix} suspicious region: lines "
            f"{region.start_line}-{region.end_line} of {total}"
        )
    if region.truncated_above:
        emit(options.truncation_format.format(count=region.truncated_above))

    for number in range(region.context_start_line, region.start_line):
        emit(lines[number - 1], number)

    if options.include_markers:
        emit(f"{indent}{options.region_start_marker}")
    for number in range(region.start_line, region.end_line + 1):
        emit(comment_line(lines[number - 1], options.comment_prefix), number)
    if options.include_markers:
        emit(f"{indent}{options.region_end_marker}")

    emit(f"{indent}{options.fill_token}")

    for number in range(region.end_line + 1, region.context_end_line + 1):
        emit(lines[number - 1], number)

    if region.truncated_below:
        emit(options.truncation_format.format(count=region.truncated_below))

    return "\n".join(out)


def recover_region_from_prompt(
    prompt: str, options: Optional[IR4Options] = None
) -> List[str]:
    """Recover the original buggy lines from a rendered IR4 prompt.

    Round-tripping a prompt back to the source it came from is how callers
    (and the test suite) prove the representation did not alter the Java code.
    Requires ``include_markers`` and no line numbers.
    """
    options = options or IR4Options()
    if not options.include_markers:
        raise RepresentationError(
            "cannot recover a region from a prompt rendered without markers"
        )
    if options.show_line_numbers:
        raise RepresentationError(
            "cannot recover a region from a prompt rendered with line numbers"
        )
    lines = prompt.split("\n")
    start = end = None
    for index, line in enumerate(lines):
        text = line.strip()
        if start is None and text == options.region_start_marker.strip():
            start = index
        elif start is not None and text == options.region_end_marker.strip():
            end = index
            break
    if start is None or end is None:
        raise RepresentationError("prompt does not contain a marked suspicious region")
    return [uncomment_line(line, options.comment_prefix) for line in lines[start + 1 : end]]
