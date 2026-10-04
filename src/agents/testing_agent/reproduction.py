"""Write a test that reproduces the issue, and keep it only if it really does.

Without such a test, a repository whose suite never exercised the bug can only
ever reach "unverified". So before any patch exists:

    1. the model writes `test_issue_reproduction.py` from the issue and the code
    2. the file is checked statically: valid Python, has tests, imports the code
       under test rather than pasting its own copy
    3. the suite runs on a copy of the unpatched repository with the file added
    4. the file is accepted only if at least one of its tests FAILS there for a
       real reason -- an assertion or an exception from the code under test, not
       a NameError or ImportError in the test itself
    5. tests that did not fail are removed from the file and the baseline is run
       again: they reproduce nothing, and one that passes on the buggy code may
       be asserting the bug itself, so a correct fix would show up as a regression
    6. otherwise the model gets the reason and one more attempt

The accepted run doubles as the baseline, so the reproduction tests take part in
the ordinary before/after comparison. The original repository is never written to.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable

from src.agents.testing_agent.agent import run_baseline
from src.agents.testing_agent.junit import FAILED

REPRO_FILE = "test_issue_reproduction.py"
REPRO_MODULE = "test_issue_reproduction"
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You are a QA engineer who writes pytest regression tests. Reply with one "
    "complete Python test file only -- no explanations, no markdown fences."
)

# A failure whose message starts with one of these means the test itself is
# broken, not that it found the bug.
BROKEN_TEST_ERRORS = (
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "IndentationError",
)

_MAX_CHUNKS = 4
_EXAMPLE_LINES = 60
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "site-packages", "__pycache__"}

# prompt -> the model's raw reply. Raises ReproductionError when it cannot answer.
Writer = Callable[[str], str]


class ReproductionError(RuntimeError):
    """The test writer could not produce a candidate (e.g. Ollama is unreachable)."""


def prepare_reproduction(
    repo_path: str | Path,
    issue: str,
    relevant_code: list[dict[str, Any]],
    workspace_root: str | Path,
    write: Writer,
    language: str = "",
    max_attempts: int = MAX_ATTEMPTS,
    python: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate and validate a reproduction test; return ``(reproduction, baseline)``.

    ``reproduction["status"]`` is ``accepted``, ``rejected`` or ``skipped``. When
    accepted, ``baseline`` is the suite run *with* the test, and the patched run
    must get the same file (``{reproduction["path"]: reproduction["source"]}``).
    Otherwise ``baseline`` is the plain suite run.
    """
    repository = Path(repo_path)
    reproduction: dict[str, Any] = {
        "status": "skipped",
        "reason": "",
        "path": REPRO_FILE,
        "source": "",
        "tests": [],
        "baseline_outcomes": {},
        "messages": {},
        "attempts": 0,
    }

    def plain_baseline() -> dict[str, Any]:
        return run_baseline(repository, workspace_root, language=language, python=python)

    if language and language.lower() != "python":
        reproduction["reason"] = f"Reproduction tests are only written for Python, not {language}."
        return reproduction, plain_baseline()

    chunks = [chunk for chunk in relevant_code if str(chunk.get("file", "")).endswith(".py")]
    chunks = [chunk for chunk in chunks if not _is_test_file(chunk["file"])][:_MAX_CHUNKS]
    if not chunks:
        reproduction["reason"] = "No Python source code was found to write a test against."
        return reproduction, plain_baseline()

    feedback = ""
    for attempt in range(1, max_attempts + 1):
        reproduction["attempts"] = attempt
        prompt = build_prompt(issue, chunks, example_test(repository, chunks), feedback)

        try:
            raw = write(prompt)
        except ReproductionError as error:
            reproduction.update(status="skipped", reason=str(error))
            return reproduction, plain_baseline()

        try:
            source = clean_test_source(raw, chunks, repository)
        except ValueError as error:
            reproduction.update(status="rejected", reason=str(error), source=raw.strip())
            feedback = _feedback(str(error), raw)
            continue

        baseline = run_baseline(
            repository, workspace_root, language=language, extra_files={REPRO_FILE: source}, python=python
        )
        accepted, reason, tests = classify(baseline)

        pruned = prune_tests(source, keep=tests) if accepted else source
        if pruned != source:
            # Rerun so the baseline holds exactly the tests the patched run will see.
            source = pruned
            baseline = run_baseline(
                repository, workspace_root, language=language, extra_files={REPRO_FILE: source}, python=python
            )
            accepted, reason, tests = classify(baseline)
        reproduction.update(
            status="accepted" if accepted else "rejected",
            reason=reason,
            source=source,
            tests=tests,
            baseline_outcomes=_repro_only(baseline.get("cases", {})),
            messages=_repro_only(baseline.get("messages", {})),
        )
        if accepted:
            return reproduction, baseline
        feedback = _feedback(reason, source)

    reproduction["tests"] = []
    return reproduction, plain_baseline()


def classify(baseline: dict[str, Any]) -> tuple[bool, str, list[str]]:
    """Decide from the unpatched run whether the reproduction test reproduces the issue.

    Returns ``(accepted, reason, reproducing_test_ids)``.
    """
    outcomes = _repro_only(baseline.get("cases", {}))
    messages = _repro_only(baseline.get("messages", {}))
    if not outcomes:
        return False, "pytest collected no test from the file.", []

    reproducing = [
        test_id
        for test_id, outcome in outcomes.items()
        if outcome == FAILED and not messages.get(test_id, "").startswith(BROKEN_TEST_ERRORS)
    ]
    if reproducing:
        return (
            True,
            f"{len(reproducing)} of {len(outcomes)} test(s) fail on the unpatched code, "
            "reproducing the issue.",
            reproducing,
        )

    # Any message left belongs to a test that errored or failed for a broken reason.
    if messages:
        test_id, message = next(iter(messages.items()))
        return False, f"The test is broken rather than failing on the bug ({test_id}: {message}).", []
    return False, "Every test passes on the unpatched code, so none of them reproduces the issue.", []


def prune_tests(source: str, keep: list[str]) -> str:
    """Remove the top-level test functions none of whose ids are in ``keep``.

    Ids look like ``test_issue_reproduction::test_x`` or, when parametrized,
    ``...::test_x[case]``. Tests inside classes are left alone.
    """
    kept_names = {test_id.split("::")[-1].split("[")[0] for test_id in keep}
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    drop: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            if node.name not in kept_names:
                start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                drop.append((start - 1, node.end_lineno or node.lineno))

    for start, end in reversed(drop):
        del lines[start:end]
    # Removed functions leave their surrounding blank lines behind.
    return re.sub(r"\n{4,}", "\n\n\n", "".join(lines)).rstrip() + "\n"


def build_prompt(
    issue: str,
    chunks: list[dict[str, Any]],
    example: str = "",
    feedback: str = "",
) -> str:
    """The brief handed to the model."""
    parts = ["### ISSUE", issue.strip(), "", "### CODE UNDER TEST"]
    for chunk in chunks:
        module = module_name(chunk["file"])
        parts.append(f"# File: {chunk['file']}  (import it with: from {module} import ...)")
        parts.append(chunk["code"].rstrip())
        parts.append("")

    if example:
        parts += ["### AN EXISTING TEST IN THIS REPOSITORY (follow its import style)", example, ""]

    parts += [
        "### TASK",
        f"Write a pytest file named {REPRO_FILE} that reproduces the issue.",
        "Rules:",
        "- Write 1 to 3 test functions, each named test_issue_<what_it_checks>.",
        "- Each test must FAIL on the code above, because of the bug, and PASS once the bug is fixed.",
        "- Assert the correct behaviour the issue asks for, never the current buggy behaviour.",
        "- Import the code under test as shown above. Do not copy or redefine it.",
        "- Use only pytest and the Python standard library. Do not use fixtures from conftest.py.",
        "Reply with the contents of the file only.",
    ]
    if feedback:
        parts += ["", feedback]
    return "\n".join(parts)


def clean_test_source(raw: str, chunks: list[dict[str, Any]], repository: Path) -> str:
    """Strip fences from the model's reply and reject a file that cannot be a valid test.

    Raises ``ValueError`` with a reason the model can act on.
    """
    from src.agents.codegen_agent import clean_output  # noqa: PLC0415 -- only needed here

    source = clean_output(raw)
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"The file is not valid Python: {error.msg} (line {error.lineno}).") from error

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if not any(name.startswith("test") for name in defined):
        raise ValueError("The file contains no test function (a function named test_...).")

    under_test = {chunk["name"] for chunk in chunks}
    redefined = sorted(name for name in defined & under_test if not name.startswith("test"))
    if redefined:
        raise ValueError(
            f"The file redefines {', '.join(redefined)} instead of importing it, "
            "so it would test its own copy rather than the repository's code."
        )

    return _src_layout_header(repository, chunks) + source.rstrip() + "\n"


def module_name(file_path: str) -> str:
    """``pkg/users.py`` -> ``pkg.users``; a leading ``src/`` is dropped (src layout)."""
    parts = list(PurePosixPath(file_path).with_suffix("").parts)
    if parts and parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def example_test(repository: Path, chunks: list[dict[str, Any]], limit: int = 300) -> str:
    """The start of an existing test file that imports the code under test, if any."""
    modules = {module_name(chunk["file"]) for chunk in chunks}
    seen = 0
    for path in sorted(repository.rglob("test_*.py")):
        if _SKIP_DIRS.intersection(path.relative_to(repository).parts) or path.name == REPRO_FILE:
            continue
        seen += 1
        if seen > limit:
            break
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(f"import {m}" in text or f"from {m} import" in text for m in modules):
            return "\n".join(text.splitlines()[:_EXAMPLE_LINES])
    return ""


def ollama_writer(model: str | None = None) -> Writer:
    """A `Writer` backed by the local Ollama server the coding agent uses."""
    from src.agents.codegen_agent import DEFAULT_MODEL, OllamaError, call_ollama  # noqa: PLC0415

    def write(prompt: str) -> str:
        try:
            return call_ollama(prompt, model=model or DEFAULT_MODEL, system=SYSTEM_PROMPT, num_predict=768)
        except OllamaError as error:
            raise ReproductionError(str(error)) from error

    return write


def stub_writer(path: str | Path) -> Writer:
    """A `Writer` that replays a checked-in test file, for runs without a model."""

    def write(prompt: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as error:
            raise ReproductionError(f"Stub reproduction test not found: {path}") from error

    return write


def _src_layout_header(repository: Path, chunks: list[dict[str, Any]]) -> str:
    """Let a root-level test import code under ``src/`` without installing the package."""
    if not any(str(chunk["file"]).startswith("src/") for chunk in chunks):
        return ""
    if not (repository / "src").is_dir() or (repository / "src" / "__init__.py").exists():
        return ""
    return (
        "import sys\n"
        "from pathlib import Path\n\n"
        'sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))\n\n'
    )


def _feedback(reason: str, previous: str) -> str:
    return "\n".join(
        [
            "### YOUR PREVIOUS ATTEMPT WAS REJECTED",
            f"Reason: {reason}",
            "Previous file:",
            previous.strip(),
            "Write a corrected file that follows every rule above.",
        ]
    )


def _repro_only(values: dict[str, str]) -> dict[str, str]:
    """Entries that belong to the reproduction file (including its collection error)."""
    return {test_id: value for test_id, value in values.items() if test_id.startswith(REPRO_MODULE)}


def _is_test_file(file_path: str) -> bool:
    name = PurePosixPath(file_path).name
    return name.startswith("test_") or name.endswith("_test.py")
