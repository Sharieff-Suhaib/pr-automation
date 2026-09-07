"""Repair by rewriting a function, then computing the diff ourselves.

Three live runs against CodeLlama-7B showed the same split: the model writes
*correct repair logic* and *incorrect diff bookkeeping*.  It decorated file
headers, miscounted ``@@`` lines, invented a ``src/`` directory, and finally —
with all of those repaired automatically — produced context lines that did not
match the file, which no amount of post-processing can fix.  Every one of those
failures is about the diff format, not about the repair.

So this module stops asking for a diff.  The model is given one function and
asked for the corrected version of it; the unified diff is then produced by
:mod:`difflib` from the file on disk.  The context lines are *read*, never
recalled, so they match by construction, the ``@@`` counts are computed, and
the path is the one we already know.  The only thing left for the model to get
wrong is the code itself, which is the part we actually want it thinking about.

The cost is a narrower edit: one contiguous region of one file per patch.  For
a localized bug fix — which is what the repository agent retrieves — that is
what a minimal patch should look like anyway.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


__all__ = [
    "RewriteError",
    "TargetChunk",
    "build_rewrite_prompt",
    "clean_code_block",
    "align_indentation",
    "diff_from_replacement",
    "select_target",
]

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)(?:```|\Z)", re.DOTALL)
_LEADING_LABEL_RE = re.compile(r"^\s*(?:###\s*)?(?:FIXED|CORRECTED|PATCHED)\s+CODE\s*:?\s*\n", re.IGNORECASE)


class RewriteError(RuntimeError):
    """Raised when a rewrite cannot be turned into a patch."""


@dataclass(frozen=True)
class TargetChunk:
    """The single region a rewrite will replace.

    ``start_line``/``end_line`` are 1-based and inclusive, matching what the
    repository agent produces.
    """

    file: str
    start_line: int
    end_line: int
    code: str
    name: str = ""
    kind: str = ""
    language: str = ""

    @classmethod
    def from_mapping(cls, chunk: Mapping[str, Any]) -> "TargetChunk":
        missing = [key for key in ("file", "start_line", "end_line") if key not in chunk]
        if missing:
            raise RewriteError(f"Retrieved chunk is missing {missing}.")
        return cls(
            file=str(chunk["file"]),
            start_line=int(chunk["start_line"]),
            end_line=int(chunk["end_line"]),
            code=str(chunk.get("code", "")),
            name=str(chunk.get("name", "")),
            kind=str(chunk.get("type", "")),
            language=str(chunk.get("language", "")),
        )

    def describe(self) -> str:
        label = f"{self.kind} {self.name}".strip() or "code"
        return f"{self.file} lines {self.start_line}-{self.end_line} ({label})"


def select_target(
    chunks: Sequence[Mapping[str, Any]], language: str = ""
) -> TargetChunk:
    """Choose the one region to repair: the best match in the right language."""
    if not chunks:
        raise RewriteError("No retrieved code to repair.")
    wanted = (language or "").strip().lower()
    if wanted:
        preferred = [
            chunk
            for chunk in chunks
            if str(chunk.get("language", "")).lower() == wanted
        ]
        if preferred:
            return TargetChunk.from_mapping(preferred[0])
    return TargetChunk.from_mapping(chunks[0])


# --------------------------------------------------------------------------
# prompting
# --------------------------------------------------------------------------
def build_rewrite_prompt(
    issue: Any,
    target: TargetChunk,
    similar_bugs: Any = None,
    strategy: Any = None,
    tools: Any = None,
    tests: Any = None,
) -> str:
    """Ask for the corrected version of one function, and nothing else."""
    sections = [f"ISSUE:\n{_as_text(issue)}"]
    if strategy:
        sections.append(f"RECOMMENDED STRATEGY:\n{_as_text(strategy)}")
    if similar_bugs:
        sections.append(f"SIMILAR BUGS AND THEIR FIXES:\n{_as_text(similar_bugs)}")
    if tests:
        sections.append(f"THE FIX SHOULD SATISFY THESE TESTS:\n{_as_text(tests)}")

    language = target.language or ""
    sections.append(
        f"BUGGY CODE from {target.describe()}:\n{target.code.rstrip()}"
    )
    sections.append(
        "Rewrite this code so the issue is fixed.\n"
        "Rules:\n"
        f"- Output only {language} code, nothing else. No explanation, no diff, "
        "no markdown fence.\n"
        "- Output the complete replacement for the code shown above, from its "
        "first line to its last.\n"
        "- Keep the same indentation as the code above.\n"
        "- Change as little as possible; do not rename anything or reformat "
        "unrelated lines."
    )
    return "\n\n".join(sections)


def _as_text(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value) or "None"
    return str(value)


# --------------------------------------------------------------------------
# cleaning the model's answer
# --------------------------------------------------------------------------
def clean_code_block(response: str, language: str = "") -> str:
    """Pull plain code out of whatever the model wrapped it in.

    Leading whitespace on the first code line is *indentation* and must
    survive: stripping it would make a nested method look flush-left, and
    :func:`align_indentation` would then re-indent every line, doubling the
    indentation of all the lines that were already correct.
    """
    if not response.strip():
        raise RewriteError("The model returned nothing.")
    text = response.replace("\r\n", "\n")

    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1)
    text = _LEADING_LABEL_RE.sub("", text)

    # A sentence before the code ("Here is the corrected function:") is common
    # even when fences are not used. Drop leading prose lines: real code for
    # these languages does not start with an unindented English sentence.
    lines = text.split("\n")
    while lines and _looks_like_prose(lines[0]):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        raise RewriteError("The model returned no code, only prose.")
    return "\n".join(lines)


def _looks_like_prose(line: str) -> bool:
    """True for an English sentence, false for a line of code."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "//", "/*", "*")):
        return False
    if stripped.endswith((":", ".")) and " " in stripped:
        # "Here is the corrected code:" -- but not "int x = 1;" or "def f():"
        return not re.search(r"[(){};=]", stripped)
    return False


def align_indentation(original: str, replacement: str) -> str:
    """Restore the original leading indentation if the model dropped it.

    Models frequently return a function flush against the left margin even
    when the original was nested in a class.  Splicing that back would produce
    code that does not compile, so the original indent is re-applied to every
    line — but only when the replacement has none of its own, so correctly
    indented output is never touched.
    """
    original_indent = _first_indent(original)
    if not original_indent:
        return replacement
    replacement_indent = _first_indent(replacement)
    if replacement_indent:
        return replacement
    return "\n".join(
        f"{original_indent}{line}" if line.strip() else line
        for line in replacement.split("\n")
    )


def _first_indent(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line[: len(line) - len(line.lstrip(" \t"))]
    return ""


# --------------------------------------------------------------------------
# building the diff from the file on disk
# --------------------------------------------------------------------------
def diff_from_replacement(
    repo_path: str | Path,
    target: TargetChunk,
    replacement: str,
    context_lines: int = 3,
) -> str:
    """Return a unified diff replacing ``target``'s lines with ``replacement``.

    The context comes from the file itself, so it matches by construction and
    ``git apply`` has nothing to disagree with.
    """
    repo = Path(repo_path)
    path = repo / target.file
    if not path.is_file():
        raise RewriteError(f"{target.file} does not exist in {repo}.")

    text = path.read_text(encoding="utf-8")
    original = text.splitlines(keepends=True)
    total = len(original)
    if not 1 <= target.start_line <= target.end_line <= total:
        raise RewriteError(
            f"{target.file} has {total} lines, so lines "
            f"{target.start_line}-{target.end_line} cannot be replaced."
        )

    # Align against the file, not against the retrieved chunk: parsers hand
    # back a node's text without the leading whitespace of its first line, so
    # the chunk cannot say how deeply the code is nested. The file can.
    replaced_block = "".join(original[target.start_line - 1 : target.end_line])
    replacement = align_indentation(replaced_block, replacement)

    replacement_lines = [f"{line}\n" for line in replacement.split("\n")]
    if not text.endswith("\n") and target.end_line == total:
        # Preserve a file that ends without a newline.
        replacement_lines[-1] = replacement_lines[-1].rstrip("\n")

    patched = (
        original[: target.start_line - 1]
        + replacement_lines
        + original[target.end_line :]
    )
    if patched == original:
        raise RewriteError("The rewrite is identical to the original code.")

    relative = Path(target.file).as_posix()
    diff = difflib.unified_diff(
        original,
        patched,
        fromfile=f"a/{relative}",
        tofile=f"b/{relative}",
        n=context_lines,
    )
    patch = "".join(_mark_missing_newlines(diff))
    if not patch.strip():
        raise RewriteError("The rewrite produced an empty diff.")
    return patch if patch.endswith("\n") else patch + "\n"


def _mark_missing_newlines(diff: Any) -> list[str]:
    """Add Git's "\\ No newline at end of file" marker where one is needed.

    ``difflib`` emits a final line without a terminator when the file has no
    trailing newline, and the next diff line then runs straight into it --
    which Git reports as ``corrupt patch``. Git's own format marks the case
    explicitly instead, so do the same.
    """
    out: list[str] = []
    for line in diff:
        if line.endswith("\n"):
            out.append(line)
            continue
        out.append(line + "\n")
        out.append("\\ No newline at end of file\n")
    return out
