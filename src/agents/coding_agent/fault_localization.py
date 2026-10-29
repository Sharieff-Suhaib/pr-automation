
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple


# ============================================================================
# Project setup
# ============================================================================

# Expected layout:
#
# pr-automation/
# └── src/
#     └── agents/
#         └── coding_agent/
#             └── fault_localization.py
#
# parents[3] points to pr-automation/

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from orchestrator.adapters import analyze_repository


# ============================================================================
# Configuration
# ============================================================================

FILL_TOKEN = "<FILL_ME>"

DEFAULT_TOP_K = 4

# "ollama" asks the local model which lines are buggy. "heuristic" skips it
# and scores lines with the rules below, as when Ollama cannot answer.
LOCALIZER = os.environ.get(
    "AGENT_SWE_FAULT_LOCALIZER",
    "ollama",
).lower()

# Empty: the coding agent's Ollama model.
LOCALIZER_MODEL = os.environ.get(
    "AGENT_SWE_LOCALIZER_MODEL",
    "",
)

# Tested with codellama:7b: asked only for "start" and "end", it stops at the
# first bug (lines 3-5 of calculateTotalRevenue, not the division on line 6).
# Made to check every line and list the buggy ones first, it covers them all.
LOCALIZER_SYSTEM_PROMPT = """You are a fault-localization tool. You read a bug report and source code
with line numbers. Check EVERY line of the code for a bug: loop bounds
(<= instead of <, - 1), wrong arithmetic operators, division that is wrong
or can divide by zero, wrong comparisons, wrong return values. The code
may hold several bugs. You never fix the code.
Answer with JSON only, in this shape:
{"buggy_lines": [<every buggy line number>], "start": <first buggy line>, "end": <last buggy line>}"""


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SuspiciousRegion:
    file: str
    start_line: int
    end_line: int
    score: float
    original_code: str
    ir4: str


class LocalizedBug(NamedTuple):
    """The region Ollama names, 1-based and inclusive."""

    start: int
    end: int
    # The lines in start..end that Ollama marked as buggy.
    buggy_lines: tuple[int, ...]


class LocalizationError(RuntimeError):
    """Ollama could not name a buggy region."""


# ============================================================================
# Text and path utilities
# ============================================================================

def normalize_path(
    value: str | Path,
) -> str:
    """
    Normalize Windows and Unix path separators.
    """

    return str(value).replace(
        "\\",
        "/",
    ).lstrip("./")


def tokenize(
    text: str,
) -> set[str]:
    """
    Extract meaningful tokens from the issue description.
    """

    ignored_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "where",
        "when",
        "into",
        "should",
        "could",
        "would",
        "have",
        "has",
        "does",
        "are",
        "was",
        "were",
        "bug",
        "buggy",
        "code",
        "issue",
        "function",
        "method",
        "file",
        "repository",
    }

    words = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        text.lower(),
    )

    return {
        word
        for word in words
        if len(word) > 2
        and word not in ignored_words
    }


def source_line_tokens(
    line: str,
) -> set[str]:
    """
    Extract source-code tokens from a line.
    """

    return set(
        re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            line.lower(),
        )
    )


def get_comment_prefix(
    file_path: str,
) -> str:
    """
    Return the comment prefix for a file type.
    """

    suffix = Path(
        file_path
    ).suffix.lower()

    if suffix in {
        ".py",
        ".rb",
        ".sh",
        ".yaml",
        ".yml",
    }:
        return "#"

    if suffix in {
        ".html",
        ".xml",
    }:
        return "<!--"

    return "//"


# ============================================================================
# Ollama fault localization
# ============================================================================

def build_localization_prompt(
    lines: list[str],
    ranges: list[tuple[int, int]],
    issue: str,
    file_path: str,
) -> str:
    """
    Show the code in `ranges` with its line numbers, under the bug report.
    """

    width = len(
        str(
            max(
                end
                for _, end in ranges
            )
        )
    )

    numbered: list[str] = []

    for start, end in ranges:
        if numbered:
            numbered.append("...")

        for number in range(
            start,
            end + 1,
        ):
            numbered.append(
                f"{number:>{width}} | {lines[number - 1]}"
            )

    code = "\n".join(numbered)

    report = (
        issue.strip()
        or "Not given; find the faulty code."
    )

    return f"""BUG REPORT:
{report}

FILE: {file_path}
{code}

Give the line where the buggy code starts and the line where it ends,
using the line numbers on the left. A one-line bug has "start" equal to
"end". When several lines are faulty, "start" is the first of them and
"end" the last, so the one region covers every bug. Mark only code that
must change, never a whole function."""


def reply_json(
    reply: str,
) -> Any:
    """
    The JSON in the model's reply, even with words around it.
    """

    text = reply.strip()

    match = re.search(
        r"[\[{].*[\]}]",
        text,
        re.DOTALL,
    )

    try:
        return json.loads(
            match.group(0)
            if match
            else text
        )

    except ValueError as error:
        raise LocalizationError(
            f"Ollama did not answer with JSON: {text[:200]!r}"
        ) from error


def parse_buggy_lines(
    reply: str,
) -> list[int]:
    """
    Read the "buggy_lines" list from the model's JSON reply.
    """

    data = reply_json(reply)

    numbers = (
        data.get("buggy_lines")
        if isinstance(data, dict)
        else None
    )

    result: list[int] = []

    for number in numbers if isinstance(numbers, list) else []:
        try:
            result.append(
                int(number)
            )

        except (TypeError, ValueError):
            continue

    return result


def parse_regions(
    reply: str,
) -> list[tuple[int, int]]:
    """
    Read (start, end) pairs from the model's JSON reply.
    """

    data = reply_json(reply)

    if isinstance(data, dict):
        # {"start", "end"} itself, a "regions" list, and every line of
        # "buggy_lines" all name buggy code; the span covers them all.
        regions = data.get(
            "regions",
            [data],
        )

        buggy_lines = data.get(
            "buggy_lines",
            [],
        )

        data = [
            *(regions if isinstance(regions, list) else []),
            *(buggy_lines if isinstance(buggy_lines, list) else []),
        ]

    if not isinstance(data, list):
        return []

    pairs: list[tuple[int, int]] = []

    for item in data:
        if isinstance(item, dict):
            start = item.get(
                "start",
                item.get("start_line"),
            )

            end = item.get(
                "end",
                item.get(
                    "end_line",
                    start,
                ),
            )

        elif isinstance(item, (list, tuple)) and item:
            start, end = item[0], item[-1]

        else:
            start = end = item

        try:
            pairs.append(
                (
                    int(start),
                    int(end),
                )
            )

        except (TypeError, ValueError):
            continue

    return pairs


def clean_regions(
    pairs: list[tuple[int, int]],
    ranges: list[tuple[int, int]],
    lines: list[str],
) -> list[tuple[int, int]]:
    """
    Keep the regions that start inside `ranges` and hold code, in source
    order, with overlapping and adjacent regions merged.
    """

    regions: list[tuple[int, int]] = []

    for start, end in pairs:
        start, end = min(start, end), max(start, end)

        container = next(
            (
                (low, high)
                for low, high in ranges
                if low <= start <= high
            ),
            None,
        )

        if container is None:
            continue

        end = min(
            end,
            container[1],
        )

        has_code = any(
            line.strip() not in {"", "{", "}"}
            for line in lines[start - 1:end]
        )

        if has_code:
            regions.append(
                (
                    start,
                    end,
                )
            )

    merged: list[tuple[int, int]] = []

    for start, end in sorted(regions):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (
                merged[-1][0],
                max(
                    end,
                    merged[-1][1],
                ),
            )
        else:
            merged.append(
                (
                    start,
                    end,
                )
            )

    return merged


def ollama_localize(
    lines: list[str],
    ranges: list[tuple[int, int]],
    issue: str,
    file_path: str,
) -> LocalizedBug:
    """
    Ask Ollama where the bug starts and ends inside `ranges`.

    `ranges` and the result are 1-based, inclusive line numbers. The
    result is one region from the first buggy line to the last, even when
    Ollama names several, and the lines in it Ollama marked as buggy.
    Raises LocalizationError when Ollama is switched off, unreachable, or
    names no usable line.
    """

    if LOCALIZER != "ollama":
        raise LocalizationError(
            "Ollama fault localization is off "
            f"(AGENT_SWE_FAULT_LOCALIZER={LOCALIZER})."
        )

    from src.agents.codegen_agent import (  # noqa: PLC0415
        DEFAULT_MODEL,
        OllamaError,
        call_ollama,
    )

    prompt = build_localization_prompt(
        lines=lines,
        ranges=ranges,
        issue=issue,
        file_path=file_path,
    )

    try:
        reply = call_ollama(
            prompt,
            model=LOCALIZER_MODEL or DEFAULT_MODEL,
            system=LOCALIZER_SYSTEM_PROMPT,
            num_predict=200,
            json_mode=True,
        )

    except (OllamaError, OSError, ValueError) as error:
        raise LocalizationError(
            f"Ollama fault localization failed: {error}"
        ) from error

    regions = clean_regions(
        pairs=parse_regions(reply),
        ranges=ranges,
        lines=lines,
    )

    if not regions:
        raise LocalizationError(
            "Ollama named no buggy line inside the code: "
            f"{reply[:200]!r}"
        )

    # One region from the bug's start to its end; regions in different
    # functions cannot be one IR4 hole, so the first function's are kept.
    container = next(
        (low, high)
        for low, high in ranges
        if low <= regions[0][0] <= high
    )

    start = regions[0][0]

    end = max(
        last
        for first, last in regions
        if container[0] <= first <= container[1]
    )

    def is_code(number: int) -> bool:
        return lines[number - 1].strip() not in {"", "{", "}"}

    # Without a "buggy_lines" list every line of code in the region counts.
    buggy_lines = sorted(
        {
            number
            for number in parse_buggy_lines(reply)
            if start <= number <= end
            and is_code(number)
        }
    ) or [
        number
        for number in range(
            start,
            end + 1,
        )
        if is_code(number)
    ]

    print(
        f"Ollama localized {file_path}: "
        f"lines {start}-{end}, buggy lines "
        f"{', '.join(str(number) for number in buggy_lines)}",
        flush=True,
    )

    return LocalizedBug(
        start=start,
        end=end,
        buggy_lines=tuple(buggy_lines),
    )


# ============================================================================
# Fault-localization scoring
# ============================================================================

def heuristic_score(
    line: str,
) -> float:
    """
    Rank suspicious source lines.
    """

    stripped = line.strip()
    lowered = stripped.lower()

    if not stripped:
        return 0.0

    score = 0.0

    if "todo" in lowered:
        score += 3.0

    if "fixme" in lowered:
        score += 3.0

    if "bug" in lowered:
        score += 3.0

    if lowered == "pass":
        score += 2.0

    if stripped.startswith(
        (
            "#",
            "//",
        )
    ):
        if re.search(
            r"\b(if|else|for|while|return|try|except|catch|def|class)\b",
            lowered,
        ):
            score += 3.0

    if re.search(
        r"\breturn\s+(none|null|nil|false|0)\b",
        lowered,
    ):
        score += 1.0

    if "not in" in lowered:
        score += 0.5

    if "==" in lowered:
        score += 0.5

    if "!=" in lowered:
        score += 0.5

    return score


def score_lines(
    lines: list[str],
    issue: str,
    allowed_ranges: list[tuple[int, int]],
) -> list[float]:
    """
    Score lines within repository-agent-selected code ranges.
    """

    issue_tokens = tokenize(issue)
    scores: list[float] = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        is_allowed = any(
            start <= line_number <= end
            for start, end in allowed_ranges
        )

        if not is_allowed:
            scores.append(0.0)
            continue

        code_tokens = source_line_tokens(
            line
        )

        matching_tokens = (
            code_tokens.intersection(
                issue_tokens
            )
        )

        score = float(
            len(matching_tokens) * 2
        )

        score += heuristic_score(line)

        scores.append(score)

    return scores


# ============================================================================
# Repository-agent result handling
# ============================================================================

def get_relevant_files(
    repository_result: dict[str, Any],
) -> list[str]:
    """
    Get relevant files returned by the repository agent.
    """

    result: list[str] = []

    for file_path in repository_result.get(
        "relevant_files",
        [],
    ):
        file_path = str(
            file_path
        ).strip()

        if (
            file_path
            and file_path not in result
        ):
            result.append(file_path)

    if result:
        return result

    for node in repository_result.get(
        "relevant_code",
        [],
    ):
        if not isinstance(node, dict):
            continue

        file_path = str(
            node.get(
                "file",
                "",
            )
        ).strip()

        if (
            file_path
            and file_path not in result
        ):
            result.append(file_path)

    return result


def get_node_ranges_for_file(
    relevant_code: list[dict[str, Any]],
    file_path: str,
    line_count: int,
) -> list[tuple[int, int]]:
    """
    Get function and method ranges for a file.
    """

    target = normalize_path(
        file_path
    )

    function_ranges: list[
        tuple[int, int]
    ] = []

    fallback_ranges: list[
        tuple[int, int]
    ] = []

    for node in relevant_code:
        if not isinstance(node, dict):
            continue

        node_file = normalize_path(
            str(
                node.get(
                    "file",
                    "",
                )
            )
        )

        same_file = (
            node_file == target
            or Path(node_file).name
            == Path(target).name
        )

        if not same_file:
            continue

        start_line = node.get(
            "start_line"
        )

        end_line = node.get(
            "end_line"
        )

        if not isinstance(
            start_line,
            int,
        ):
            continue

        if not isinstance(
            end_line,
            int,
        ):
            continue

        start_line = max(
            1,
            start_line,
        )

        end_line = min(
            line_count,
            end_line,
        )

        if start_line > end_line:
            continue

        node_type = str(
            node.get(
                "type",
                "",
            )
        ).lower()

        if node_type in {
            "class",
            "interface",
            "struct",
            "module",
        }:
            fallback_ranges.append(
                (
                    start_line,
                    end_line,
                )
            )
        else:
            function_ranges.append(
                (
                    start_line,
                    end_line,
                )
            )

    if function_ranges:
        return function_ranges

    if fallback_ranges:
        return fallback_ranges

    return [
        (
            1,
            line_count,
        )
    ]


def resolve_file_inside_repository(
    repo_path: str,
    file_path: str,
) -> Path:
    """
    Resolve a repository file safely.
    """

    repository_root = Path(
        repo_path
    ).resolve()

    candidate = Path(
        file_path
    )

    possible_paths = [
        candidate,
        repository_root / file_path,
        repository_root / normalize_path(file_path),
    ]

    for possible_path in possible_paths:
        try:
            resolved = possible_path.resolve()

            if not resolved.is_file():
                continue

            resolved.relative_to(
                repository_root
            )

            return resolved

        except (
            FileNotFoundError,
            ValueError,
        ):
            continue

    raise FileNotFoundError(
        f"Could not find '{file_path}' inside "
        f"repository '{repository_root}'"
    )


# ============================================================================
# Suspicious-region selection and IR4
# ============================================================================

def select_suspicious_region(
    file_path: str,
    lines: list[str],
    scores: list[float],
    allowed_ranges: list[tuple[int, int]],
) -> tuple[int, int, float]:
    """
    Select the complete function or method containing the highest-scoring line.
    """

    if not lines:
        raise ValueError(
            f"File is empty: {file_path}"
        )

    candidates: list[int] = []

    for index in range(
        len(lines)
    ):
        line_number = index + 1

        if any(
            start <= line_number <= end
            for start, end in allowed_ranges
        ):
            candidates.append(index)

    if not candidates:
        candidates = list(
            range(
                len(lines)
            )
        )

    best_index = max(
        candidates,
        key=lambda index: scores[index],
    )

    best_score = scores[
        best_index
    ]

    for start, end in allowed_ranges:
        if start <= best_index + 1 <= end:
            return (
                start,
                end,
                best_score,
            )

    return (
        best_index + 1,
        best_index + 1,
        best_score,
    )


def build_ir4(
    lines: list[str],
    start_line: int,
    end_line: int,
    file_path: str,
) -> str:
    """
    Comment out the suspicious region and insert <FILL_ME>.
    """

    if start_line < 1:
        raise ValueError(
            "start_line must be at least 1"
        )

    if end_line > len(lines):
        raise ValueError(
            f"end_line {end_line} exceeds "
            f"{len(lines)} source lines"
        )

    if start_line > end_line:
        raise ValueError(
            "start_line must be <= end_line"
        )

    comment_prefix = get_comment_prefix(
        file_path
    )

    prefix = lines[
        :start_line - 1
    ]

    buggy_lines = lines[
        start_line - 1:end_line
    ]

    suffix = lines[
        end_line:
    ]

    commented_buggy_lines: list[str] = []

    for line in buggy_lines:
        if comment_prefix == "<!--":
            commented_buggy_lines.append(
                f"<!-- {line} -->"
            )
        else:
            commented_buggy_lines.append(
                f"{comment_prefix} {line}"
            )

    return "\n".join(
        prefix
        + commented_buggy_lines
        + [FILL_TOKEN]
        + suffix
    )


# ============================================================================
# Per-file localization
# ============================================================================

def localize_relevant_file(
    repo_path: str,
    file_path: str,
    issue: str,
    relevant_code: list[dict[str, Any]],
) -> SuspiciousRegion:
    """
    Localize one suspicious region and build its IR4 representation:
    Ollama names where the bug starts and ends, and the heuristic scores
    pick the region when Ollama cannot.
    """

    actual_path = resolve_file_inside_repository(
        repo_path=repo_path,
        file_path=file_path,
    )

    source = actual_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = source.splitlines()

    if not lines:
        raise ValueError(
            f"Relevant file is empty: {file_path}"
        )

    allowed_ranges = get_node_ranges_for_file(
        relevant_code=relevant_code,
        file_path=file_path,
        line_count=len(lines),
    )

    try:
        start_line, end_line, _ = ollama_localize(
            lines=lines,
            ranges=allowed_ranges,
            issue=issue,
            file_path=file_path,
        )

        # Ollama names a region without ranking it.
        score = 1.0

    except LocalizationError as error:
        print(
            f"{error} Using heuristic fault "
            f"localization for '{file_path}'.",
            flush=True,
        )

        scores = score_lines(
            lines=lines,
            issue=issue,
            allowed_ranges=allowed_ranges,
        )

        start_line, end_line, score = (
            select_suspicious_region(
                file_path=file_path,
                lines=lines,
                scores=scores,
                allowed_ranges=allowed_ranges,
            )
        )

    original_code = "\n".join(
        lines[
            start_line - 1:end_line
        ]
    )

    ir4 = build_ir4(
        lines=lines,
        start_line=start_line,
        end_line=end_line,
        file_path=file_path,
    )

    return SuspiciousRegion(
        file=file_path,
        start_line=start_line,
        end_line=end_line,
        score=score,
        original_code=original_code,
        ir4=ir4,
    )


# ============================================================================
# Main fault-localization pipeline
# ============================================================================

def run_fault_localization(
    repo_url: str,
    issue: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> list[SuspiciousRegion]:
    """
    Run repository-agent fault localization and build IR4 inputs.
    """

    print(
        f"Repository: {repo_url}",
        flush=True,
    )

    repository_result = analyze_repository(
        repo_url=repo_url,
        issue=issue,
        top_k=top_k,
    )

    repo_path = str(
        repository_result.get(
            "repo_path",
            "",
        )
    )

    relevant_code = repository_result.get(
        "relevant_code",
        [],
    )

    relevant_files = get_relevant_files(
        repository_result
    )

    effective_issue = str(
        repository_result.get(
            "issue",
            issue,
        )
    ).strip()

    if not effective_issue:
        effective_issue = (
            "Find suspicious, incomplete, or faulty code "
            "in the repository."
        )

    print()
    print(
        "REPOSITORY AGENT RESULT",
        flush=True,
    )

    print(
        "=======================",
        flush=True,
    )

    print(
        f"Repository path: {repo_path}",
        flush=True,
    )

    print(
        "Indexed chunks: "
        f"{repository_result.get('indexed_chunks', 0)}",
        flush=True,
    )

    print(
        f"Relevant files: {relevant_files}",
        flush=True,
    )

    if not repo_path:
        print(
            "The repository agent did not return a "
            "repository path.",
            flush=True,
        )

        return []

    if not relevant_files:
        print(
            "The repository agent returned no relevant files.",
            flush=True,
        )

        return []

    results: list[SuspiciousRegion] = []

    for file_path in relevant_files:
        try:
            result = localize_relevant_file(
                repo_path=repo_path,
                file_path=file_path,
                issue=effective_issue,
                relevant_code=relevant_code,
            )

            results.append(result)

        except Exception as error:
            print(
                f"Could not localize "
                f"'{file_path}': {error}",
                flush=True,
            )

    return results


# ============================================================================
# Output
# ============================================================================

def print_results(
    results: list[SuspiciousRegion],
) -> None:
    """
    Print fault-localization results.
    """

    if not results:
        print()
        print(
            "No relevant files or suspicious code was found."
        )

        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            "=" * 80
        )

        print(
            f"FAULT-LOCALIZATION RESULT {index}"
        )

        print(
            "=" * 80
        )

        print(
            f"File: {result.file}"
        )

        print(
            f"Suspicious lines: "
            f"{result.start_line}-{result.end_line}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print()
        print(
            "ORIGINAL SUSPICIOUS CODE"
        )

        print(
            "-----------------------"
        )

        print(
            result.original_code
        )

        print()
        print(
            "IR4 INPUT"
        )

        print(
            "---------"
        )

        print(
            result.ir4
        )


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    repo_url = (
        "https://github.com/Sharieff-Suhaib/dummy_repo.git"
    )

    issue = """
The login function incorrectly accepts hard-coded credentials.
It should validate credentials using the repository's user store
instead of comparing the username and password directly with
literal strings.
"""

    results = run_fault_localization(
        repo_url=repo_url,
        issue=issue,
        top_k=4,
    )

    print_results(results)


if __name__ == "__main__":
    main()