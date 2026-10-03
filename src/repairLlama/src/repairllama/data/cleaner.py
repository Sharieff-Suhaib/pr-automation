"""Filtering bug/fix pairs down to usable single-function repairs.

RepairLLaMA trains on fixes that change exactly one function, so this stage
drops everything else: pairs with no change at all, changes spanning several
methods, diffs that are too large to be a plausible single hunk, and test-only
edits.  Each drop is recorded as a :class:`~repairllama.data.models.Rejection`
so the report can explain the funnel.

Java scanning
-------------
Method boundaries are found with a brace-matching scanner rather than a full
parser: :func:`strip_comments` blanks out comments and string/char literals so
braces inside them cannot confuse the matcher, and :func:`find_methods` pairs
a signature line with its closing brace.  This module owns those primitives
because it needs them; :mod:`repairllama.data.deduplicator` reuses
``strip_comments`` for normalisation.

The scanner is deliberately conservative — it understands annotations,
generics, constructors, throws clauses and nested classes, but it is a
heuristic.  A unit where no method is found is *not* silently accepted: it is
rejected as ``not_single_function``, so bad parses show up in the report
instead of poisoning the training set.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from repairllama.data.models import BugFixPair, Rejection, RejectionReason
from repairllama.utils.logging import get_logger

__all__ = [
    "MethodSpan",
    "CleanerOptions",
    "CleanResult",
    "Cleaner",
    "strip_comments",
    "find_methods",
    "method_containing",
    "changed_methods",
    "looks_like_test",
]

log = get_logger("data.cleaner")

# A method or constructor signature: annotations and modifiers, an optional
# generic parameter list, a name, a parameter list, and an opening brace.
_SIGNATURE_RE = re.compile(
    r"""
    ^\s*
    (?:(?:public|protected|private|static|final|abstract|synchronized|native|
         strictfp|default)\s+)*
    (?:<[^>]{0,200}>\s*)?                 # generic type parameters
    (?:[\w$.<>\[\],\s?]+\s+)?             # return type (absent for constructors)
    ([\w$]+)                              # method name
    \s*\([^;{]*\)                         # parameter list
    (?:\s*throws\s+[\w$.,\s]+)?           # throws clause
    \s*\{                                 # opening brace
    """,
    re.VERBOSE,
)

_CONTROL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "synchronized", "try", "do",
    "else", "return", "new", "case",
}

_TEST_PATH_RE = re.compile(r"(^|/)(test|tests|src/test)(/|$)", re.IGNORECASE)
_TEST_NAME_RE = re.compile(r"(Test|Tests|TestCase|IT)\.java$")


# --------------------------------------------------------------------------- #
# Java text scanning
# --------------------------------------------------------------------------- #
def strip_comments(source: str, blank_strings: bool = False) -> str:
    """Replace comments (and optionally string bodies) with spaces.

    Character positions and line breaks are preserved, so offsets computed on
    the stripped text apply to the original.  Nothing is deleted — the result
    has exactly the same length as ``source``.
    """
    out: List[str] = []
    index = 0
    length = len(source)
    state = "code"  # code | line_comment | block_comment | string | char

    while index < length:
        char = source[index]
        nxt = source[index + 1] if index + 1 < length else ""

        if state == "code":
            if char == "/" and nxt == "/":
                state, out, index = "line_comment", out + ["  "], index + 2
                continue
            if char == "/" and nxt == "*":
                state, out, index = "block_comment", out + ["  "], index + 2
                continue
            if char == '"':
                state = "string"
                out.append('"' if not blank_strings else '"')
                index += 1
                continue
            if char == "'":
                state = "char"
                out.append("'")
                index += 1
                continue
            out.append(char)
            index += 1
            continue

        if state == "line_comment":
            if char == "\n":
                state = "code"
                out.append("\n")
            else:
                out.append(" ")
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and nxt == "/":
                state, out, index = "code", out + ["  "], index + 2
                continue
            out.append("\n" if char == "\n" else " ")
            index += 1
            continue

        # string / char literal
        if char == "\\" and nxt:
            out.append("  " if blank_strings else source[index : index + 2])
            index += 2
            continue
        if (state == "string" and char == '"') or (state == "char" and char == "'"):
            state = "code"
            out.append(char)
            index += 1
            continue
        out.append("\n" if char == "\n" else (" " if blank_strings else char))
        index += 1

    return "".join(out)


@dataclass(frozen=True)
class MethodSpan:
    """A method's extent, in 1-based inclusive line numbers."""

    name: str
    start_line: int
    end_line: int

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    def overlaps(self, start: int, end: int) -> bool:
        return not (end < self.start_line or start > self.end_line)


def find_methods(source: str) -> List[MethodSpan]:
    """Find method and constructor spans in ``source``.

    Braces inside comments and literals are ignored.  Nested types are handled
    by brace matching, so an inner class's methods are found too; a method
    whose body never closes (truncated source) is dropped rather than guessed.
    """
    scrubbed = strip_comments(source, blank_strings=True)
    lines = scrubbed.split("\n")
    spans: List[MethodSpan] = []

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        match = _SIGNATURE_RE.match(line)
        if not match or match.group(1) in _CONTROL_KEYWORDS:
            line_index += 1
            continue

        end_line = _matching_brace_line(lines, line_index)
        if end_line is None:
            line_index += 1
            continue
        spans.append(
            MethodSpan(name=match.group(1), start_line=line_index + 1, end_line=end_line + 1)
        )
        line_index = end_line + 1

    return spans


def _matching_brace_line(lines: Sequence[str], start_index: int) -> Optional[int]:
    """Index of the line closing the first ``{`` opened at ``start_index``."""
    depth = 0
    opened = False
    for index in range(start_index, len(lines)):
        for char in lines[index]:
            if char == "{":
                depth += 1
                opened = True
            elif char == "}":
                depth -= 1
                if opened and depth == 0:
                    return index
    return None


def method_containing(spans: Sequence[MethodSpan], line: int) -> Optional[MethodSpan]:
    """Innermost method containing ``line`` (1-based), or None."""
    candidates = [span for span in spans if span.contains(line)]
    if not candidates:
        return None
    return min(candidates, key=lambda span: span.end_line - span.start_line)


def changed_line_span(buggy_lines: Sequence[str], fixed_lines: Sequence[str]) -> Optional[Tuple[int, int]]:
    """1-based inclusive span of buggy lines the diff touches, or None."""
    matcher = difflib.SequenceMatcher(a=list(buggy_lines), b=list(fixed_lines), autojunk=False)
    changed = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not changed:
        return None
    lower = min(op[1] for op in changed)
    upper = max(op[2] for op in changed)
    if upper <= lower:
        anchor = max(1, lower)
        return (anchor, anchor)
    return (lower + 1, upper)


def changed_methods(pair: BugFixPair) -> List[MethodSpan]:
    """Methods of the buggy source that the fix touches."""
    span = changed_line_span(pair.buggy_lines, pair.fixed_lines)
    if span is None:
        return []
    start, end = span
    return [method for method in find_methods(pair.buggy_code) if method.overlaps(start, end)]


def diff_size(pair: BugFixPair) -> int:
    """Number of changed lines (removed + added) between the two versions."""
    matcher = difflib.SequenceMatcher(
        a=pair.buggy_lines, b=pair.fixed_lines, autojunk=False
    )
    return sum(
        (i2 - i1) + (j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def looks_like_test(pair: BugFixPair) -> bool:
    """True when the pair appears to come from test code."""
    path = pair.file_path or ""
    if _TEST_PATH_RE.search(path) or _TEST_NAME_RE.search(path):
        return True
    head = "\n".join(pair.buggy_lines[:40])
    if re.search(r"^\s*@(Test|ParameterizedTest|BeforeEach|AfterEach)\b", head, re.M):
        return True
    return bool(re.search(r"\bclass\s+\w*(Test|TestCase|IT)\b", head))


# --------------------------------------------------------------------------- #
# the cleaning stage
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CleanerOptions:
    """Which samples survive cleaning."""

    min_diff_lines: int = 1
    max_diff_lines: int = 50
    require_single_function: bool = True
    drop_test_only_changes: bool = True
    drop_identical: bool = True
    require_region: bool = True  # derive one from the diff when missing
    max_region_lines: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_diff_lines < 0:
            raise ValueError("min_diff_lines must be >= 0")
        if self.max_diff_lines < self.min_diff_lines:
            raise ValueError("max_diff_lines must be >= min_diff_lines")


@dataclass
class CleanResult:
    """What survived cleaning, and why the rest did not."""

    kept: List[BugFixPair] = field(default_factory=list)
    rejections: List[Rejection] = field(default_factory=list)

    @property
    def received(self) -> int:
        return len(self.kept) + len(self.rejections)

    def counts_by_reason(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.reason.value] = counts.get(rejection.reason.value, 0) + 1
        return counts


class Cleaner:
    """Applies the cleaning rules to bug/fix pairs."""

    stage_name = "clean"

    def __init__(self, options: Optional[CleanerOptions] = None) -> None:
        self.options = options or CleanerOptions()

    def inspect(self, pair: BugFixPair) -> Optional[Rejection]:
        """Return why ``pair`` should be dropped, or None to keep it."""
        options = self.options

        if not pair.buggy_code.strip() or not pair.fixed_code.strip():
            return self._reject(pair, RejectionReason.EMPTY_SOURCE, "blank source")
        if options.drop_identical and pair.is_identical:
            return self._reject(
                pair, RejectionReason.IDENTICAL, "buggy and fixed code are identical"
            )

        span = changed_line_span(pair.buggy_lines, pair.fixed_lines)
        if span is None:
            return self._reject(pair, RejectionReason.NO_CHANGE, "no changed lines")

        if pair.has_region:
            start, end = pair.suspicious_start, pair.suspicious_end
            total = len(pair.buggy_lines)
            if start is None or end is None or start < 1 or end < start or end > total:
                return self._reject(
                    pair,
                    RejectionReason.INVALID_REGION,
                    f"region {start}-{end} is not within 1-{total}",
                )
        elif options.require_region:
            start, end = span
        else:
            start, end = span

        if options.max_region_lines is not None and (end - start + 1) > options.max_region_lines:
            return self._reject(
                pair,
                RejectionReason.DIFF_TOO_LARGE,
                f"region spans {end - start + 1} lines",
            )

        size = diff_size(pair)
        if size < options.min_diff_lines:
            return self._reject(
                pair, RejectionReason.DIFF_TOO_SMALL, f"{size} changed lines"
            )
        if size > options.max_diff_lines:
            return self._reject(
                pair, RejectionReason.DIFF_TOO_LARGE, f"{size} changed lines"
            )

        if options.drop_test_only_changes and looks_like_test(pair):
            return self._reject(pair, RejectionReason.TEST_ONLY, "test code")

        if options.require_single_function:
            touched = changed_methods(pair)
            if len(touched) != 1:
                detail = (
                    "no method found around the change"
                    if not touched
                    else f"change spans {len(touched)} methods: "
                    + ", ".join(method.name for method in touched)
                )
                return self._reject(pair, RejectionReason.NOT_SINGLE_FUNCTION, detail)

        return None

    def clean(self, pair: BugFixPair) -> Tuple[Optional[BugFixPair], Optional[Rejection]]:
        """Return ``(kept_pair, None)`` or ``(None, rejection)``."""
        rejection = self.inspect(pair)
        if rejection is not None:
            return (None, rejection)
        if self.options.require_region and not pair.has_region:
            pair = pair.with_region()
        return (pair, None)

    def run(self, pairs: Iterable[BugFixPair]) -> CleanResult:
        """Clean a stream of pairs, collecting rejections."""
        result = CleanResult()
        for pair in pairs:
            kept, rejection = self.clean(pair)
            if kept is not None:
                result.kept.append(kept)
            else:
                assert rejection is not None
                result.rejections.append(rejection)
        log.debug(
            "clean: kept %d of %d", len(result.kept), result.received
        )
        return result

    def _reject(
        self, pair: BugFixPair, reason: RejectionReason, detail: str
    ) -> Rejection:
        return Rejection(
            bug_id=pair.bug_id, reason=reason, detail=detail, stage=self.stage_name
        )


def clean_pairs(
    pairs: Iterable[BugFixPair], options: Optional[CleanerOptions] = None
) -> CleanResult:
    """Convenience wrapper around :class:`Cleaner`."""
    return Cleaner(options).run(pairs)
