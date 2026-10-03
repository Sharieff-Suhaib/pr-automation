"""OR2: the output representation — replacement code for the region only.

Under OR2 the model never reproduces the enclosing method.  Given an IR4
prompt, the expected completion is exactly the lines that replace the
suspicious region::

        if (a > b) {
            return a;
        }

That keeps targets short (cheaper training, fewer tokens to get wrong) and
makes patching mechanical: splice the completion in place of lines
``start..end`` of the original file.

Deriving the target
-------------------
A training example gives the buggy and fixed versions of a unit plus the
buggy region.  :func:`align_fixed_region` finds the corresponding lines in
the fixed version with :mod:`difflib`, so callers only supply the buggy
boundaries.  When the fix also touches lines outside the region, the result
says so (``changed_outside_region``) instead of pretending otherwise —
such examples are usually mislocalized and worth filtering out.

Indentation
-----------
Targets keep their original absolute indentation by default, so a completion
can be spliced in unmodified.  ``OR2Options.dedent`` strips the region's
common indent instead, and :func:`restore_indentation` puts it back.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from repairllama.representation.ir4 import (
    DEFAULT_FILL_TOKEN,
    RegionMetadata,
    RepresentationError,
)

__all__ = [
    "OR2Options",
    "FixedRegion",
    "OR2Prediction",
    "align_fixed_region",
    "build_or2_target",
    "parse_or2_output",
    "restore_indentation",
    "splice_region",
]

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*$")


@dataclass(frozen=True)
class OR2Options:
    """Knobs for building and parsing OR2 targets."""

    dedent: bool = False
    allow_empty: bool = True
    strip_code_fences: bool = True
    stop_sequences: Tuple[str, ...] = ()
    max_target_lines: Optional[int] = None
    fill_token: str = DEFAULT_FILL_TOKEN  # dropped if the model echoes it back

    def __post_init__(self) -> None:
        if self.max_target_lines is not None and self.max_target_lines < 1:
            raise RepresentationError(
                f"max_target_lines must be >= 1 or None, got {self.max_target_lines!r}"
            )


@dataclass(frozen=True)
class FixedRegion:
    """The fixed lines corresponding to a buggy region.

    ``start_line``/``end_line`` are 1-based inclusive positions in the *fixed*
    source.  An empty replacement (the fix deletes the region) is represented
    by ``end_line == start_line - 1`` and ``lines == []``.
    """

    lines: Tuple[str, ...]
    start_line: int
    end_line: int
    changed_outside_region: bool
    is_deletion: bool
    inferred: bool  # False when the caller supplied the boundaries

    @property
    def num_lines(self) -> int:
        return len(self.lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "num_lines": self.num_lines,
            "changed_outside_region": self.changed_outside_region,
            "is_deletion": self.is_deletion,
            "inferred": self.inferred,
        }


@dataclass(frozen=True)
class OR2Prediction:
    """A model completion normalised into replacement lines."""

    lines: Tuple[str, ...]
    text: str
    is_deletion: bool
    truncated_by: Optional[str] = None
    had_code_fence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "num_lines": len(self.lines),
            "is_deletion": self.is_deletion,
            "truncated_by": self.truncated_by,
            "had_code_fence": self.had_code_fence,
        }


# --------------------------------------------------------------------------- #
# aligning the buggy region onto the fixed source
# --------------------------------------------------------------------------- #
def _map_lower(opcodes: Sequence[tuple], index: int, fixed_len: int) -> int:
    """Map a 0-based buggy start index onto the fixed side.

    A pure insertion sitting exactly on the boundary is taken *into* the
    region: the target must reproduce the whole fix when spliced back, and an
    insertion at the region's first line belongs to that region's rewrite.
    """
    for tag, i1, i2, j1, _j2 in opcodes:
        if tag == "insert" and i1 == i2 == index:
            return j1
    for tag, i1, i2, j1, _j2 in opcodes:
        if i1 <= index < i2:
            return j1 + (index - i1) if tag == "equal" else j1
    return fixed_len


def _map_upper(opcodes: Sequence[tuple], index: int, fixed_len: int) -> int:
    """Map a 0-based exclusive buggy end index onto the fixed side.

    Mirror of :func:`_map_lower`: an insertion on the trailing boundary is
    also taken into the region.
    """
    for tag, i1, i2, _j1, j2 in opcodes:
        if tag == "insert" and i1 == i2 == index:
            return j2
    for tag, i1, i2, j1, j2 in opcodes:
        if i1 < index <= i2:
            return j1 + (index - i1) if tag == "equal" else j2
    return 0 if index == 0 else fixed_len


def align_fixed_region(
    buggy_lines: Sequence[str],
    fixed_lines: Sequence[str],
    start_line: int,
    end_line: int,
) -> FixedRegion:
    """Find the lines in ``fixed_lines`` that replace the buggy region.

    ``start_line``/``end_line`` are 1-based inclusive positions in
    ``buggy_lines``.  Alignment is a line-level diff: the region's boundaries
    are carried across the matching blocks, and any edit that straddles them
    is reported via ``changed_outside_region``.
    """
    start_index = start_line - 1
    end_index = end_line  # exclusive
    matcher = difflib.SequenceMatcher(
        a=list(buggy_lines), b=list(fixed_lines), autojunk=False
    )
    opcodes = matcher.get_opcodes()

    fixed_start = _map_lower(opcodes, start_index, len(fixed_lines))
    fixed_end = _map_upper(opcodes, end_index, len(fixed_lines))
    if fixed_end < fixed_start:
        fixed_end = fixed_start

    changed_outside = False
    for tag, i1, i2, _j1, _j2 in opcodes:
        if tag == "equal":
            continue
        if i1 == i2:  # pure insertion: judged by its position
            if not (start_index <= i1 <= end_index):
                changed_outside = True
        elif not (i1 >= start_index and i2 <= end_index):
            changed_outside = True

    region = tuple(fixed_lines[fixed_start:fixed_end])
    return FixedRegion(
        lines=region,
        start_line=fixed_start + 1,
        end_line=fixed_end,
        changed_outside_region=changed_outside,
        is_deletion=len(region) == 0,
        inferred=True,
    )


def fixed_region_from_bounds(
    fixed_lines: Sequence[str], start_line: int, end_line: int
) -> FixedRegion:
    """Build a :class:`FixedRegion` from caller-supplied 1-based bounds.

    ``end_line == start_line - 1`` denotes an empty (deletion) replacement.
    """
    total = len(fixed_lines)
    if start_line < 1:
        raise RepresentationError(f"fixed_start must be >= 1, got {start_line}")
    if end_line < start_line - 1:
        raise RepresentationError(
            f"fixed_end ({end_line}) must be >= fixed_start - 1 ({start_line - 1})"
        )
    if end_line > total:
        raise RepresentationError(
            f"fixed_end ({end_line}) is past the end of the fixed source ({total} lines)"
        )
    lines = tuple(fixed_lines[start_line - 1 : end_line])
    return FixedRegion(
        lines=lines,
        start_line=start_line,
        end_line=end_line,
        changed_outside_region=False,
        is_deletion=len(lines) == 0,
        inferred=False,
    )


# --------------------------------------------------------------------------- #
# building the target
# --------------------------------------------------------------------------- #
def dedent_lines(lines: Sequence[str]) -> Tuple[List[str], str]:
    """Strip the common leading whitespace; return the lines and what was cut."""
    indents = [
        line[: len(line) - len(line.lstrip(" \t"))] for line in lines if line.strip()
    ]
    if not indents:
        return list(lines), ""
    prefix = os.path.commonprefix(indents)
    if not prefix:
        return list(lines), ""
    return [line[len(prefix) :] if line.startswith(prefix) else line for line in lines], prefix


def restore_indentation(lines: Sequence[str], indent: str) -> List[str]:
    """Re-apply ``indent`` to every non-blank line (inverse of dedenting)."""
    return [f"{indent}{line}" if line.strip() else line for line in lines]


def build_or2_target(
    fixed_region_lines: Sequence[str],
    options: Optional[OR2Options] = None,
) -> str:
    """Render the OR2 target: the replacement code and nothing else.

    Lines are emitted verbatim — no reformatting, no whitespace tidying — so
    the target is exactly what a patcher would write back into the file.
    """
    options = options or OR2Options()
    lines = list(fixed_region_lines)
    if not lines and not options.allow_empty:
        raise RepresentationError(
            "empty OR2 target (the fix deletes the region) and allow_empty is False"
        )
    if options.max_target_lines is not None and len(lines) > options.max_target_lines:
        raise RepresentationError(
            f"OR2 target has {len(lines)} lines, over max_target_lines "
            f"({options.max_target_lines})"
        )
    if options.dedent:
        lines, _ = dedent_lines(lines)
    return "\n".join(lines)


def parse_or2_output(
    completion: str,
    options: Optional[OR2Options] = None,
    *,
    region: Optional[RegionMetadata] = None,
) -> OR2Prediction:
    """Normalise a raw model completion into replacement lines.

    Handles what models actually emit around a hunk: markdown fences, text
    after a stop sequence, an echoed fill token, and trailing blank lines.
    The code inside is not otherwise touched.  When ``region`` is given and
    the options dedent targets, the region indent is restored.
    """
    options = options or OR2Options()
    text = completion
    truncated_by: Optional[str] = None

    for stop in options.stop_sequences:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
            truncated_by = stop
            break

    lines = text.split("\n")
    had_fence = False
    if options.strip_code_fences:
        fenced = [index for index, line in enumerate(lines) if _FENCE_RE.match(line)]
        if fenced:
            had_fence = True
            start = fenced[0] + 1
            stop_index = fenced[1] if len(fenced) > 1 else len(lines)
            lines = lines[start:stop_index]

    # Drop an echoed fill token and blank padding at either end.
    lines = [line for line in lines if line.strip() != options.fill_token]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if options.dedent and region is not None and region.indent:
        lines = restore_indentation(lines, region.indent)

    return OR2Prediction(
        lines=tuple(lines),
        text="\n".join(lines),
        is_deletion=len(lines) == 0,
        truncated_by=truncated_by,
        had_code_fence=had_fence,
    )


# --------------------------------------------------------------------------- #
# splicing (used to validate targets; the patching stage builds on this)
# --------------------------------------------------------------------------- #
def splice_region(
    lines: Sequence[str],
    start_line: int,
    end_line: int,
    replacement: Sequence[str],
) -> List[str]:
    """Replace lines ``start_line..end_line`` (1-based, inclusive) verbatim."""
    if start_line < 1 or end_line < start_line - 1 or end_line > len(lines):
        raise RepresentationError(
            f"cannot splice lines {start_line}-{end_line} of a {len(lines)}-line source"
        )
    return [*lines[: start_line - 1], *replacement, *lines[end_line:]]
