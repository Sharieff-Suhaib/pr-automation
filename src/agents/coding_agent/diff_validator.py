"""Check that a model's patch really is a unified diff, before Git sees it.

A small model asked for a diff will often produce something that *looks* like
one — file headers, ``@@`` markers — but is really a human-readable summary::

    --- Auth.java (lines 3-6, method login)
    +++ Auth.java (lines 3-6, method login)
    @@ -3,7 +3,8 @@ public static boolean login(...) {
            return username.equals("admin") &&
                   password.equals("password123");
        }

Three things are wrong there: the header is a description rather than a path,
the body lines carry no ``+``/``-``/space prefix, and the hunk promises 7 old
and 8 new lines but supplies four.  Handing that to ``git apply`` produces
``error: corrupt patch at line 10`` — an accurate message about the wrong
line, because Git counts from where it lost its footing, not from the mistake.

This module finds the real mistake instead, and says which line it is on, so
the generator can put that sentence back in front of the model and retry.

The parser is deliberately strict: everything it accepts, ``git apply`` will
parse.  It says nothing about whether the patch *applies* to a particular working
tree — that is a separate question, answered by
:func:`patch_manager.check_patch`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


__all__ = [
    "DiffError",
    "Hunk",
    "FileDiff",
    "parse_unified_diff",
    "validate_unified_diff",
    "normalize_diff",
    "extract_diff",
    "describe_diff_error",
]

# @@ -old_start[,old_count] +new_start[,new_count] @@ [section heading]
_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<heading>.*)$"
)

# Lines Git allows inside a hunk body.
_BODY_PREFIXES = (" ", "+", "-", "\\")

# Metadata Git writes between the "diff --git" line and the file headers.
_HEADER_NOISE = (
    "diff --git ",
    "index ",
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "similarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "GIT binary patch",
)


class DiffError(ValueError):
    """A unified diff is malformed, with the offending line number."""

    def __init__(self, message: str, line_number: int | None = None, line: str = ""):
        self.line_number = line_number
        self.line = line
        location = f" (line {line_number})" if line_number else ""
        super().__init__(f"{message}{location}")
        self.reason = message


@dataclass
class Hunk:
    """One ``@@`` block: where it applies and what it changes."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)
    header_line_number: int = 0
    counts_corrected: bool = False

    @property
    def header(self) -> str:
        """The ``@@`` line for this hunk, with counts matching its body."""
        return (
            f"@@ -{self.old_start},{self.old_count} "
            f"+{self.new_start},{self.new_count} @@"
        )

    @property
    def added(self) -> int:
        return sum(1 for line in self.lines if line.startswith("+"))

    @property
    def removed(self) -> int:
        return sum(1 for line in self.lines if line.startswith("-"))

    @property
    def is_empty(self) -> bool:
        """A hunk that changes nothing — context only."""
        return self.added == 0 and self.removed == 0


@dataclass
class FileDiff:
    """The hunks that apply to one file."""

    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)
    header_line_number: int = 0

    @property
    def path(self) -> str:
        """The path to report: the new one, unless the file is being deleted."""
        return self.old_path if self.new_path == "/dev/null" else self.new_path

    @property
    def changes(self) -> int:
        return sum(hunk.added + hunk.removed for hunk in self.hunks)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_unified_diff(patch: str) -> list[FileDiff]:
    """Parse ``patch``, raising :class:`DiffError` on the first real problem.

    Tolerant where ``git apply`` is tolerant, strict where it is strict.  In
    particular, ``@@`` counts are treated as *hints*: Git recovers from a
    miscounted header by reading the body, and so does this parser, recording
    the true counts on the hunk (``counts_corrected``) so
    :func:`normalize_diff` can rewrite the header.  Small models get those
    numbers wrong constantly while producing a perfectly good body; rejecting
    them would fail patches Git would have applied.

    What is *not* recoverable, and is raised: a header that is a description
    rather than a path, a body line with no ``+``/``-``/space prefix, prose in
    the middle of a diff, and hunks or headers that are missing entirely.
    """
    if not patch.strip():
        raise DiffError("The patch is empty")

    lines = patch.splitlines()
    files: list[FileDiff] = []
    current: FileDiff | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        number = index + 1

        if line.startswith(_HEADER_NOISE):
            index += 1
            continue

        if line.startswith("--- "):
            old_path = _parse_header_path(line, number)
            plus = lines[index + 1] if index + 1 < len(lines) else ""
            if not plus.startswith("+++ "):
                raise DiffError(
                    "A '--- <old path>' header must be followed by '+++ <new path>'",
                    number + 1,
                    plus,
                )
            new_path = _parse_header_path(plus, number + 1)
            current = FileDiff(old_path, new_path, header_line_number=number)
            files.append(current)
            index += 2
            continue

        if line.startswith("@@"):
            if current is None:
                raise DiffError(
                    "A hunk appears before any '--- '/'+++ ' file header", number, line
                )
            hunk, promised_old, promised_new = _parse_hunk_header(line, number)
            index = _read_hunk_body(lines, index + 1, hunk, promised_old, promised_new)
            current.hunks.append(hunk)
            continue

        if not line.strip():
            index += 1  # blank lines between files are harmless
            continue

        raise DiffError(
            "Expected a file header ('--- '), a hunk header ('@@') or hunk "
            "content, but found prose",
            number,
            line,
        )

    if not files:
        raise DiffError("The patch contains no '--- '/'+++ ' file headers")
    for file_diff in files:
        if not file_diff.hunks:
            raise DiffError(
                f"No @@ hunk for {file_diff.path!r}", file_diff.header_line_number
            )
    return files


def _read_hunk_body(
    lines: list[str], index: int, hunk: Hunk, promised_old: int, promised_new: int
) -> int:
    """Consume one hunk's body, returning the index of the line after it.

    The body ends at the next hunk or file header, or at the first line that
    cannot belong to a hunk.  The promised counts decide only how far a
    ``---``-prefixed line is read as a deletion rather than a new file header,
    and whether a blank line counts as context.
    """
    old_seen = new_seen = 0

    while index < len(lines):
        line = lines[index]
        number = index + 1
        satisfied = old_seen >= promised_old and new_seen >= promised_new

        if line.startswith("@@"):
            break
        # "--- " is ambiguous: a deletion of a line starting with "--", or the
        # next file's header. While the hunk still owes lines it is a deletion.
        if line.startswith("--- ") and satisfied:
            break
        if line.startswith("+++ ") and satisfied:
            break

        if not line:
            # Some tools strip the trailing space from a blank context line.
            if satisfied:
                break
            hunk.lines.append(" ")
            old_seen += 1
            new_seen += 1
            index += 1
            continue

        if not line.startswith(_BODY_PREFIXES):
            if satisfied:
                break  # prose after a complete hunk; the caller reports it
            raise DiffError(
                "Every line inside a hunk must start with ' ' (context), "
                "'+' (added) or '-' (removed)",
                number,
                line,
            )

        if line.startswith("\\"):  # "\ No newline at end of file"
            hunk.lines.append(line)
            index += 1
            continue

        marker = line[0]
        if marker in " -":
            old_seen += 1
        if marker in " +":
            new_seen += 1
        hunk.lines.append(line)
        index += 1

    if not hunk.lines:
        raise DiffError("A hunk header is followed by no content", hunk.header_line_number)

    # Git recovers from miscounted headers, so record the truth instead of
    # rejecting: normalize_diff() rewrites the header from these numbers.
    hunk.counts_corrected = (old_seen, new_seen) != (promised_old, promised_new)
    hunk.old_count = old_seen
    hunk.new_count = new_seen
    return index


def _parse_header_path(line: str, number: int) -> str:
    """Take the path out of a ``---``/``+++`` header, rejecting descriptions."""
    raw = line[4:].split("\t", maxsplit=1)[0].strip()
    if not raw:
        raise DiffError("A file header has no path", number, line)
    if raw == "/dev/null":
        return raw
    if " " in raw:
        # "--- Auth.java (lines 3-6, method login)" is a description, not a
        # path. normalize_diff() strips a trailing "(...)" note before this
        # runs, so reaching here means the path itself is unusable.
        raise DiffError(
            "A file header must be a path such as 'a/src/Auth.java' with no "
            "description after it",
            number,
            line,
        )
    return raw


# --------------------------------------------------------------------------
# mechanical repair
# --------------------------------------------------------------------------
_HEADER_NOTE_RE = re.compile(r"\s*\((?:[^()]*)\)\s*$")


def normalize_diff(
    patch: str,
    *,
    add_prefixes: bool = True,
    repo_path: str | Path | None = None,
) -> str:
    """Fix the two things models get wrong that are safe to fix automatically.

    * **Header notes.** ``--- stats.py (original)`` becomes ``--- a/stats.py``.
      The note is a description the model added; the path in front of it is
      the real one.
    * **Hunk counts.** ``@@ -1,2 +1,3 @@`` over a body with four new lines is
      rewritten to ``+1,4``. The body is the ground truth and the counts are
      derivable from it, so a model getting them wrong is recoverable without
      another request. Git tolerates this too, but downstream tools are less
      forgiving.
    * **Invented directories**, when ``repo_path`` is given. Models assume a
      conventional layout and write ``src/Auth.java`` for a file that sits at
      the repository root. If exactly one file in the repository has that
      name, the path is corrected to it; if several do, or none, the path is
      left alone for Git to reject.

    **No ``+``, ``-`` or context line is ever touched** — this function
    rewrites headers only, so it cannot change what the patch does.  Anything
    it cannot make sense of is passed through untouched, for
    :func:`validate_unified_diff` to report afterwards.
    """
    if not patch.strip():
        return patch

    lines = patch.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith(("--- ", "+++ ")):
            marker, _, rest = line.partition(" ")
            path = _HEADER_NOTE_RE.sub("", rest.split("\t", maxsplit=1)[0]).strip()
            if path and path != "/dev/null":
                bare = path[2:] if path.startswith(("a/", "b/")) else path
                if repo_path is not None:
                    bare = _resolve_against_repo(bare, Path(repo_path))
                path = ("a/" if marker == "---" else "b/") + bare if add_prefixes else bare
            out.append(f"{marker} {path}" if path else line)
            index += 1
            continue

        if line.startswith("@@"):
            try:
                hunk, promised_old, promised_new = _parse_hunk_header(line, index + 1)
                body_end = _read_hunk_body(
                    lines, index + 1, hunk, promised_old, promised_new
                )
            except DiffError:
                # An unparseable hunk is left exactly as it is; validation will
                # explain it far better than a guess at what was meant.
                out.append(line)
                index += 1
                continue
            heading = _HUNK_RE.match(line)
            suffix = heading.group("heading") if heading else ""
            out.append(hunk.header + suffix)
            out.extend(lines[index + 1 : body_end])
            index = body_end
            continue

        out.append(line)
        index += 1

    return "\n".join(out) + "\n"


def _parse_hunk_header(line: str, number: int) -> tuple[Hunk, int, int]:
    match = _HUNK_RE.match(line)
    if match is None:
        raise DiffError(
            "A hunk header must look like '@@ -<old line>,<count> "
            "+<new line>,<count> @@'",
            number,
            line,
        )
    old_count = int(match.group("old_count") or 1)
    new_count = int(match.group("new_count") or 1)
    hunk = Hunk(
        old_start=int(match.group("old_start")),
        old_count=old_count,
        new_start=int(match.group("new_start")),
        new_count=new_count,
        header_line_number=number,
    )
    return hunk, old_count, new_count


def _body_prefix_message(line: str, old_remaining: int, new_remaining: int) -> str:
    """Explain a bad body line in terms of what the hunk still expects."""
    if line.startswith(("--- ", "+++ ", "@@")):
        return (
            "A new file or hunk header appears while the previous hunk is "
            f"still missing {old_remaining} original and {new_remaining} new "
            "line(s); its @@ counts do not match its body"
        )
    return (
        "Every line inside a hunk must start with ' ' (context), '+' (added) "
        "or '-' (removed)"
    )


# --------------------------------------------------------------------------
# validation and repair of the surrounding text
# --------------------------------------------------------------------------
def validate_unified_diff(patch: str, *, allow_empty_hunks: bool = False) -> list[FileDiff]:
    """Parse ``patch`` and additionally reject diffs that change nothing.

    A patch whose hunks are pure context applies cleanly and repairs nothing,
    which is worse than a rejected patch: the run reports success.
    """
    files = parse_unified_diff(patch)
    if allow_empty_hunks:
        return files
    if not any(file_diff.changes for file_diff in files):
        raise DiffError(
            "The patch contains no '+' or '-' lines, so it would change nothing"
        )
    return files


def extract_diff(text: str) -> str:
    """Trim prose from around a diff, keeping the diff itself.

    Models like to introduce a patch ("Here is the fix:") and to explain it
    afterwards.  Both are removed; anything that fails to parse *inside* the
    diff is left for :func:`validate_unified_diff` to report.
    """
    lines = text.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("--- ") or line.startswith("diff --git ")
        ),
        None,
    )
    if start is None:
        return text.strip()

    end = len(lines)
    for index in range(len(lines) - 1, start, -1):
        line = lines[index]
        if line.startswith(_BODY_PREFIXES) or line.startswith(("@@", "--- ", "+++ ")):
            end = index + 1
            break
    return "\n".join(lines[start:end]).strip()


def _resolve_against_repo(path: str, repo: Path) -> str:
    """Point ``path`` at the real file when the model invented a directory."""
    if (repo / path).is_file():
        return path

    name = PurePosixPath(path).name
    if not name:
        return path

    matches = [
        candidate
        for candidate in repo.rglob(name)
        if candidate.is_file() and ".git" not in candidate.parts
    ]
    if len(matches) != 1:
        # No match, or an ambiguous one: guessing would be worse than the
        # honest "No such file or directory" from git apply.
        return path
    return matches[0].relative_to(repo).as_posix()


def describe_diff_error(error: DiffError, patch: str, context: int = 2) -> str:
    """Render an error with the offending line, for a retry prompt or a log."""
    message = str(error)
    if error.line_number is None:
        return message

    lines = patch.splitlines()
    first = max(0, error.line_number - 1 - context)
    last = min(len(lines), error.line_number + context)
    excerpt = []
    for index in range(first, last):
        marker = ">>" if index + 1 == error.line_number else "  "
        excerpt.append(f"{marker} {index + 1:>3} | {lines[index]}")
    return message + "\n" + "\n".join(excerpt)
