"""Generate a repair by asking for corrected functions, then build the diff locally.

Small local models fix a function reliably but write unified diffs badly: wrong
hunk line numbers, invented headers, duplicated context. `git apply` rejects
those. So this mode never asks for a diff:

    1. the model gets the relevant functions of one file and returns them fixed
    2. each returned function replaces the original's line range in the file
    3. the diff is computed with `difflib`, so its format is always valid

`generate_patch` in `code_generator.py` (the model writes the diff itself)
stays available as the fallback.
"""

from __future__ import annotations

import ast
import difflib
from pathlib import Path, PurePosixPath
import re
import textwrap
from typing import Any

from src.agents.coding_agent.code_generator import (
    DEFAULT_MODEL,
    PatchGenerationError,
    _call_ollama,
    _format_value,
)

SYSTEM_PROMPT = """You are a software repair agent.
Return only the corrected Python functions. Do not include explanations or Markdown."""

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_MAX_FUNCTIONS = 3


def generate_function_patch(
    issue: Any,
    repo_path: str | Path,
    chunks: list[dict[str, Any]],
    similar_bugs: Any = None,
    strategy: Any = None,
    tests: Any = None,
    model: str = DEFAULT_MODEL,
    feedback: str = "",
    temperature: float = 0,
) -> str:
    """Return a unified diff repairing the functions in ``chunks``.

    ``feedback`` explains why a previous attempt was rejected (retry loop);
    a retry also passes a non-zero ``temperature`` so it is not forced to
    reproduce the rejected answer word for word.

    ``chunks`` are the repository agent's dicts (``file``, ``name``,
    ``start_line``, ``end_line`` 1-indexed inclusive, ``code``). Only Python
    functions from the file of the first chunk are repaired.
    Raises ``PatchGenerationError`` when no usable repair comes back.
    """
    targets = _targets(chunks)
    if not targets:
        raise PatchGenerationError("No Python function was retrieved to repair.")

    file_path = targets[0]["file"]
    source_file = Path(repo_path) / file_path
    try:
        original = source_file.read_text(encoding="utf-8")
    except OSError as error:
        raise PatchGenerationError(f"Cannot read {file_path}: {error}") from error

    prompt = build_function_prompt(issue, file_path, targets, similar_bugs, strategy, tests, feedback)
    reply = _call_ollama(prompt, model=model, system=SYSTEM_PROMPT, temperature=temperature)
    repaired = splice_functions(original, targets, parse_functions(reply))

    if repaired == original:
        raise PatchGenerationError("The LLM returned the functions unchanged.")
    try:
        compile(repaired, file_path, "exec")
    except SyntaxError as error:
        raise PatchGenerationError(
            f"The repaired {file_path} is not valid Python: {error.msg} (line {error.lineno})."
        ) from error

    return make_diff(file_path, original, repaired)


def build_function_prompt(
    issue: Any,
    file_path: str,
    targets: list[dict[str, Any]],
    similar_bugs: Any = None,
    strategy: Any = None,
    tests: Any = None,
    feedback: str = "",
) -> str:
    """The repair brief: the issue, the functions to fix, the recommendations, and
    -- on a retry -- why the previous attempt was rejected."""
    functions = "\n\n".join(_dedent(chunk["code"]) for chunk in targets)
    names = ", ".join(chunk["name"] for chunk in targets)
    previous = f"\nPREVIOUS ATTEMPT:\n{feedback.strip()}\n" if feedback.strip() else ""
    return f"""You are a software repair agent.

ISSUE:
{_format_value(issue)}

FUNCTIONS FROM {file_path}:
{functions}

SIMILAR BUGS:
{_format_value(similar_bugs)}

RECOMMENDED STRATEGY:
{_format_value(strategy)}

RECOMMENDED TESTS:
{_format_value(tests)}
{previous}
Fix the issue by correcting the functions above ({names}).
Return the corrected functions as complete Python definitions with the same names and signatures.
Change only what the issue requires. Leave out any function that needs no change.
Return only Python code.
"""


def parse_functions(reply: str) -> dict[str, str]:
    """Map each top-level function in the model's reply to its source (decorators excluded)."""
    match = _FENCE_RE.search(reply)
    code = (match.group(1) if match else reply).strip("\n")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise PatchGenerationError(
            f"The LLM's functions are not valid Python: {error.msg} (line {error.lineno})."
        ) from error

    lines = code.splitlines()
    functions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    if not functions:
        raise PatchGenerationError("The LLM did not return any function definition.")
    return functions


def splice_functions(original: str, targets: list[dict[str, Any]], functions: dict[str, str]) -> str:
    """Replace each target's line range with the returned function of the same name.

    Targets the model left out are kept as they were. Replacements run bottom-up
    so earlier line numbers stay valid; each one is re-indented to the original
    function's indentation, so methods keep their place inside their class.
    """
    lines = original.splitlines(keepends=True)
    for chunk in sorted(targets, key=lambda chunk: chunk["start_line"], reverse=True):
        replacement = functions.get(chunk["name"])
        if replacement is None:
            continue
        start, end = chunk["start_line"] - 1, chunk["end_line"]
        indent = re.match(r"[ \t]*", lines[start]).group(0)
        new_lines = [
            (indent + line if line.strip() else "") + "\n"
            for line in _dedent(replacement).splitlines()
        ]
        lines[start:end] = new_lines
    return "".join(lines)


def make_diff(file_path: str, original: str, repaired: str) -> str:
    """A `git apply`-ready unified diff of one file."""
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            repaired.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )


def _targets(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Python function chunks from the first relevant source file, in retrieval order."""
    candidates = [
        chunk
        for chunk in chunks
        if str(chunk.get("file", "")).endswith(".py")
        and chunk.get("type") in ("function", "method")
        and not _is_test_file(chunk["file"])
    ]
    if not candidates:
        return []
    first_file = candidates[0]["file"]
    same_file = [chunk for chunk in candidates if chunk["file"] == first_file]
    return _without_overlaps(same_file)[:_MAX_FUNCTIONS]


def _without_overlaps(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop chunks nested inside an earlier one; splicing both would corrupt the file."""
    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        if all(
            chunk["end_line"] < other["start_line"] or chunk["start_line"] > other["end_line"]
            for other in kept
        ):
            kept.append(chunk)
    return kept


def _dedent(code: str) -> str:
    """Bring a function to column 0.

    A retrieved method starts at ``def`` with no indentation of its own while
    its body keeps the file's indentation (e.g. 8 spaces), which
    ``textwrap.dedent`` cannot fix; the body is shifted to 4 spaces instead.
    """
    lines = code.splitlines()
    if len(lines) > 1 and lines[0] == lines[0].lstrip():
        body = [line for line in lines[1:] if line.strip()]
        if body:
            excess = min(len(line) - len(line.lstrip()) for line in body) - 4
            if excess > 0:
                lines = [lines[0], *(line[excess:] if line.strip() else "" for line in lines[1:])]
            return "\n".join(lines)
    return textwrap.dedent(code)


def _is_test_file(file_path: str) -> bool:
    name = PurePosixPath(file_path).name
    return name.startswith("test_") or name.endswith("_test.py")
