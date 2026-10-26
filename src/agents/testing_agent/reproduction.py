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
# Java: a JUnit 5 class next to the code under test (same package), run by java.py.
JAVA_REPRO_CLASS = "AgentReproTest"
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You are a QA engineer who writes regression tests. Reply with one "
    "complete test file only -- no explanations, no markdown fences."
)
_JAVA_FILE_LINES = 200
_JAVA_TYPE_RE = re.compile(r"\b(?:class|interface|enum|record)\s+(\w+)")
_JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+[\w.]+\s*;\s*$", re.M)
_JAVAC_ERROR_RE = re.compile(r"([\w$]+\.java):(\d+): error")

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

    lang = (language or "python").lower()
    if lang not in ("python", "java"):
        reproduction["reason"] = f"Reproduction tests are only written for Python and Java, not {language}."
        return reproduction, plain_baseline()

    suffix = ".py" if lang == "python" else ".java"
    chunks = [chunk for chunk in relevant_code if str(chunk.get("file", "")).endswith(suffix)]
    chunks = [chunk for chunk in chunks if not _is_test_file(chunk["file"])]
    if lang == "java":  # one class under test: the file the coding agent repairs
        chunks = [chunk for chunk in chunks if chunk["file"] == chunks[0]["file"]] if chunks else []
    chunks = chunks[:_MAX_CHUNKS]
    if not chunks:
        reproduction["reason"] = f"No {lang.title()} source code was found to write a test against."
        return reproduction, plain_baseline()

    if lang == "java":
        target = chunks[0]["file"]
        target_source = (repository / target).read_text(encoding="utf-8", errors="replace")
        # The method the coding agent repairs (its top-ranked function): every
        # test has to exercise it. Other retrieved methods are context only -- a
        # test of one of them says nothing about the fix, and once it asserted
        # a wrong expectation that failed before and after a correct fix.
        methods = [chunks[0]["name"]]
        reproduction["path"] = str(PurePosixPath(target).parent / f"{JAVA_REPRO_CLASS}.java")
    path = reproduction["path"]

    feedback = ""
    for attempt in range(1, max_attempts + 1):
        reproduction["attempts"] = attempt
        if lang == "java":
            prompt = build_java_prompt(issue, target, target_source, feedback, methods)
        else:
            prompt = build_prompt(issue, chunks, example_test(repository, chunks), feedback)

        try:
            raw = write(prompt)
        except ReproductionError as error:
            reproduction.update(status="skipped", reason=str(error))
            return reproduction, plain_baseline()

        try:
            if lang == "java":
                source = drop_guessed_exceptions(keep_tests_calling(clean_java_test(raw, target_source), methods), issue)
            else:
                source = clean_test_source(raw, chunks, repository)
        except ValueError as error:
            reproduction.update(status="rejected", reason=str(error), source=raw.strip())
            feedback = _feedback(str(error), raw)
            continue

        baseline = run_baseline(
            repository, workspace_root, language=language, extra_files={path: source}, python=python
        )
        accepted, reason, tests = classify(baseline)
        if not accepted and lang == "java" and (salvaged := drop_broken_java_tests(source, baseline)):
            # Some test methods call things that do not exist; the rest may still reproduce.
            source = salvaged
            baseline = run_baseline(
                repository, workspace_root, language=language, extra_files={path: source}, python=python
            )
            accepted, reason, tests = classify(baseline)

        # ponytail: Java tests are not pruned (no AST edit); a passing one stays and passes after too.
        pruned = prune_tests(source, keep=tests) if accepted and lang == "python" else source
        if pruned != source:
            # Rerun so the baseline holds exactly the tests the patched run will see.
            source = pruned
            baseline = run_baseline(
                repository, workspace_root, language=language, extra_files={path: source}, python=python
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

    if lang == "java" and (source := java_property_test(issue, target_source, methods)):
        # The model's tests reproduced nothing; a property any fix must satisfy might.
        baseline = run_baseline(
            repository, workspace_root, language=language, extra_files={path: source}, python=python
        )
        accepted, reason, tests = classify(baseline)
        if accepted:
            reproduction.update(
                status="accepted",
                reason=f"{reason} (built-in check: empty input must give a finite result, not an exception)",
                source=source,
                tests=tests,
                baseline_outcomes=_repro_only(baseline.get("cases", {})),
                messages=_repro_only(baseline.get("messages", {})),
            )
            return reproduction, baseline

    reproduction["tests"] = []
    return reproduction, plain_baseline()


def classify(baseline: dict[str, Any]) -> tuple[bool, str, list[str]]:
    """Decide from the unpatched run whether the reproduction test reproduces the issue.

    Returns ``(accepted, reason, reproducing_test_ids)``.
    """
    outcomes = _repro_only(baseline.get("cases", {}))
    messages = _repro_only(baseline.get("messages", {}))
    if not outcomes:
        errors = baseline.get("errors") or []
        detail = f" {errors[0]}" if errors and "collected no tests" not in errors[0] else ""
        return False, f"No test from the file ran.{detail}".strip(), []

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


def build_java_prompt(
    issue: str, target: str, target_source: str, feedback: str = "", methods: list[str] | None = None
) -> str:
    """The brief for a JUnit 5 reproduction test of ``methods`` in ``target``."""
    lines = target_source.splitlines()
    shown = "\n".join(lines[:_JAVA_FILE_LINES]) + ("\n// ... (rest of the file omitted)" if len(lines) > _JAVA_FILE_LINES else "")
    parts = [
        "### ISSUE", issue.strip(), "",
        f"### CODE UNDER TEST: {target}", shown, "",
        "### TASK",
        f"Write a JUnit 5 test class named {JAVA_REPRO_CLASS} that reproduces the issue.",
        "Rules:",
        f"- Declare `public class {JAVA_REPRO_CLASS}` with 1 or 2 methods annotated @Test, each named testIssue<WhatItChecks>.",
        *([f"- Every test must call {' or '.join(methods)}: that is the code the issue is about."] if methods else []),
        "- Each test checks ONE concrete symptom the issue describes, with the expected result the issue states.",
        "  Do not assert anything the issue does not say (no guessed exceptions, orders or messages).",
        "- Expect an exception only if the issue says the method SHOULD throw one; valid input must never throw.",
        "  If the issue says the code raises an exception, that exception IS the bug: set up the failing case",
        "  (for example an empty list) and assert the method returns a sensible value without throwing.",
        "- Each test must FAIL on the code above, because of the bug, and PASS once the bug is fixed.",
        "- Assert the correct behaviour the issue asks for, never the current buggy behaviour.",
        "- Use the classes above as they are, through their constructors and methods. Do not copy or redefine them.",
        "  They are in the same package as the test, so they need no import.",
        "- Use only JUnit 5 (org.junit.jupiter.api) and the Java standard library; import every java.util class you use.",
        "Reply with the contents of the file only.",
    ]
    if feedback:
        parts += ["", feedback]
    return "\n".join(parts)


def clean_java_test(raw: str, target_source: str) -> str:
    """Strip fences, reject a test that cannot work, add the JUnit imports it forgot.

    The class goes in the package of the code under test, so any package line
    the model wrote is replaced by the target's. Raises ``ValueError`` with a
    reason the model can act on.
    """
    from src.agents.codegen_agent import clean_output  # noqa: PLC0415 -- only needed here

    source = _JAVA_PACKAGE_RE.sub("", clean_output(raw)).strip()
    if not re.search(rf"\bclass\s+{JAVA_REPRO_CLASS}\b", source):
        raise ValueError(f"The file does not declare `public class {JAVA_REPRO_CLASS}`.")
    if "@Test" not in source:
        raise ValueError("The file has no method annotated @Test.")
    redefined = sorted((set(_JAVA_TYPE_RE.findall(source)) - {JAVA_REPRO_CLASS}) & set(_JAVA_TYPE_RE.findall(target_source)))
    if redefined:
        raise ValueError(
            f"The file redefines {', '.join(redefined)} instead of using the repository's class, "
            "so it would test its own copy."
        )

    header = []
    package = _JAVA_PACKAGE_RE.search(target_source)
    if package:
        header.append(package.group(0).strip())
    if not re.search(r"import\s+org\.junit\.jupiter\.api\.(Test|\*)\s*;", source):
        header.append("import org.junit.jupiter.api.Test;")
    if "Assertions" not in source:
        header.append("import static org.junit.jupiter.api.Assertions.*;")
    # The model often uses List/Map without importing them; this cannot clash with its own imports.
    if "import java.util.*;" not in source:
        header.append("import java.util.*;")
    return "\n".join([*header, "", source]).strip() + "\n"


def drop_broken_java_tests(source: str, baseline: dict[str, Any]) -> str | None:
    """The test class without the @Test methods javac rejected, or None when that cannot help.

    Only when every compile error lies inside a test method of the reproduction
    class -- an error in a field, an import or the repository itself is not
    fixed by dropping tests -- and at least one test is left.
    """
    text = baseline.get("output", "") + "\n" + "\n".join(baseline.get("errors") or [])
    errors = _JAVAC_ERROR_RE.findall(text)
    if not errors or any(name != f"{JAVA_REPRO_CLASS}.java" for name, _ in errors):
        return None
    bad_lines = {int(line) for _, line in errors}

    tests = _java_test_spans(source)
    broken = [span for span in tests if any(span[0] <= line <= span[1] for line in bad_lines)]
    outside = [line for line in bad_lines if not any(start <= line <= end for start, end in tests)]
    if not broken or outside or len(broken) == len(tests):
        return None
    return _without_spans(source, broken)


def keep_tests_calling(source: str, methods: list[str]) -> str:
    """The test class with only the @Test methods that call one of ``methods``.

    A test that never calls the method being repaired cannot show that the
    repair works -- it only makes the verdict depend on some other code.
    Raises ``ValueError`` (a reason the model can act on) when none is left.
    """
    lines = source.splitlines(keepends=True)
    calls = re.compile(r"\b(" + "|".join(map(re.escape, methods)) + r")\s*\(")
    tests = _java_test_spans(source)
    unrelated = [(start, end) for start, end in tests if not calls.search("".join(lines[start - 1 : end]))]
    if len(unrelated) == len(tests):
        raise ValueError(f"No test calls {', '.join(methods)}, the code the issue is about.")
    return _without_spans(source, unrelated)


_EMPTY_INPUT_SYMPTOM = re.compile(
    r"divi(de|sion|ded)|by zero|\bzero\b|\bempty\b|\bnan\b|average|\bmean\b|exception|crash|validat"
)


def java_property_test(issue: str, target_source: str, methods: list[str]) -> str | None:
    """A reproduction test needing no model: on an empty object, ``method()``
    must neither throw nor return NaN/Infinity.

    Divide-by-zero on an empty collection is the commonest form of the bugs
    these issues describe; the buggy code fails this (0.0 / 0 is NaN, an int
    division throws) and any correct fix passes it. Only for a method without
    arguments returning a number, on a class that `new X()` can build.
    """
    if not _EMPTY_INPUT_SYMPTOM.search(issue.lower()):
        return None
    for method in methods:
        found = re.search(rf"public\s+(double|float|int|long)\s+{re.escape(method)}\s*\(\s*\)", target_source)
        if not found:
            continue
        declared = list(re.finditer(r"\bclass\s+(\w+)", target_source[: found.start()]))
        if not declared:
            continue
        cls = declared[-1].group(1)
        constructors = re.findall(rf"\b{cls}\s*\(([^)]*)\)\s*\{{", target_source)
        if constructors and all(params.strip() for params in constructors):
            continue  # every constructor needs arguments
        floating = found.group(1) in ("double", "float")
        package = _JAVA_PACKAGE_RE.search(target_source)
        check = (
            f'        double result = assertDoesNotThrow(() -> subject.{method}());\n'
            f'        assertFalse(Double.isNaN(result) || Double.isInfinite(result),\n'
            f'            "{method}() returned " + result + " for an empty {cls}");\n'
            if floating
            else f"        assertDoesNotThrow(() -> subject.{method}());\n"
        )
        return (
            (package.group(0).strip() + "\n" if package else "")
            + "import org.junit.jupiter.api.Test;\n"
            "import static org.junit.jupiter.api.Assertions.*;\n\n"
            f"public class {JAVA_REPRO_CLASS} {{\n"
            "    @Test\n"
            f"    public void testIssue{method[0].upper() + method[1:]}OnEmptyInput() {{\n"
            f"        {cls} subject = new {cls}();\n"
            f"{check}"
            "    }\n"
            "}\n"
        )
    return None


def drop_guessed_exceptions(source: str, issue: str) -> str:
    """The test class without tests expecting an exception the issue never asks for.

    An issue that reports an exception ("it raises an exception") reports the
    bug; a test expecting that exception asserts the bug, and a correct fix
    could never pass it. Only "should/must throw" in the issue allows one.
    Raises ``ValueError`` when no test is left.
    """
    if re.search(r"\b(should|must|expected to|needs? to)\s+(throw|raise)", issue, re.I):
        return source
    lines = source.splitlines(keepends=True)
    tests = _java_test_spans(source)
    guessed = [(start, end) for start, end in tests if "assertThrows" in "".join(lines[start - 1 : end])]
    if guessed and len(guessed) == len(tests):
        raise ValueError(
            "Every test expects an exception, but the issue does not say the method should throw one. "
            "If the issue reports an exception, that exception is the bug: assert it is not thrown."
        )
    return _without_spans(source, guessed)


def _java_test_spans(source: str) -> list[tuple[int, int]]:
    """1-based inclusive line spans of the @Test methods (the annotation is part of them)."""
    from tree_sitter_languages import get_parser  # noqa: PLC0415 -- only Java needs it

    data = source.encode("utf-8")
    spans: list[tuple[int, int]] = []

    def walk(node) -> None:
        if node.type == "method_declaration":
            modifiers = next((c for c in node.children if c.type == "modifiers"), None)
            if modifiers is not None and b"@Test" in data[modifiers.start_byte : modifiers.end_byte]:
                spans.append((node.start_point[0] + 1, node.end_point[0] + 1))
            return
        for child in node.children:
            walk(child)

    walk(get_parser("java").parse(data).root_node)
    return spans


def _without_spans(source: str, spans: list[tuple[int, int]]) -> str:
    lines = source.splitlines(keepends=True)
    for start, end in sorted(spans, reverse=True):
        del lines[start - 1 : end]
    return "".join(lines)


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
    return {test_id: value for test_id, value in values.items() if _is_repro_id(test_id)}


def _is_repro_id(test_id: str) -> bool:
    """``test_issue_reproduction::test_x`` (pytest) or ``[pkg.]AgentReproTest::testX()`` (JUnit)."""
    module = test_id.split("::")[0]
    return module.startswith(REPRO_MODULE) or module.split(".")[-1] == JAVA_REPRO_CLASS


def _is_test_file(file_path: str) -> bool:
    name = PurePosixPath(file_path).name
    return name.startswith("test_") or name.endswith(("_test.py", "Test.java"))
