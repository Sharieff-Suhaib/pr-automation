"""Testing Agent: run the suite before and after the patch and compare the two.

    run_baseline    copy the unpatched repo -> run the suite -> discard the copy
    evaluate_patch  copy the repo -> git apply -> run the suite -> compare with baseline

The original repository is never written to. The baseline is returned as its own
value so a caller that tries several patches against one repository (the retry
loop) can run it once and pass it back in.

Both runs accept ``extra_files`` -- files written into the copy before the suite
runs. That is how the generated reproduction test (see `reproduction.py`) takes
part in both runs without ever touching the original repository.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

from src.agents.coding_agent.patch_manager import apply_patch
from src.agents.coding_agent.tester import run_tests, select_test_command
from src.agents.testing_agent.environment import working_copy_env
from src.agents.testing_agent.junit import PASSED, parse_junit_report
from src.agents.testing_agent.selection import id_prefix, related_tests
from src.agents.testing_agent.verdict import TestVerdict, compare_runs, not_tested

MAX_OUTPUT_CHARS = 20_000


def run_suite(
    working_repo: str | Path,
    language: str = "",
    paths: list[str] | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Run the repository's tests; adds per-test results when the runner is pytest.

    Returns `TestResult.to_dict()` plus ``cases: {test_id: outcome}`` and
    ``messages: {test_id: failure message}``. Other runners get both empty and
    are compared on their overall PASS/FAIL. ``paths`` limits a pytest run to
    those test files; ``python`` is the interpreter that runs pytest (the
    repository's test environment), and the working copy goes on PYTHONPATH.
    The stored output is capped at its last MAX_OUTPUT_CHARS characters.
    """
    repository = Path(working_repo)
    try:
        command = select_test_command(repository, language=language or None)
    except ValueError:
        # Let run_tests produce its usual "no framework detected" result.
        result = run_tests(repository, language=language or None).to_dict()
        return {**result, "cases": {}, "messages": {}}

    if command[:3] != [sys.executable, "-m", "pytest"]:
        return _capped({**run_tests(repository, command=command).to_dict(), "cases": {}, "messages": {}})
    if python:
        command = [python, *command[1:]]

    # The report is written outside the repository so it cannot show up as a change.
    # Without --continue-on-collection-errors, one test file that fails to import
    # stops pytest from running any test at all.
    with tempfile.TemporaryDirectory(prefix="junit-") as report_dir:
        report = Path(report_dir) / "report.xml"
        result = run_tests(
            repository,
            command=[*command, *(paths or []), "--continue-on-collection-errors", f"--junitxml={report}"],
            env=working_copy_env(repository),
        )
        cases, messages = parse_junit_report(report)

    return _capped({**result.to_dict(), "cases": cases, "messages": messages})


def _capped(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the end of a long runner output: the summary and failures are printed last."""
    output = result.get("output", "")
    if len(output) > MAX_OUTPUT_CHARS:
        result["output"] = f"... ({len(output) - MAX_OUTPUT_CHARS} characters cut)\n" + output[-MAX_OUTPUT_CHARS:]
    return result


def run_baseline(
    repo_path: str | Path,
    workspace_root: str | Path,
    language: str = "",
    extra_files: Mapping[str, str] | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Run the suite on a throwaway copy of the unpatched repository.

    A copy rather than the repository itself, because running tests writes
    caches (``__pycache__``, ``.pytest_cache``) that would dirty the clone.
    """
    source = Path(repo_path).resolve()
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{source.name}-baseline-", dir=root) as scratch:
        working = Path(scratch) / source.name
        try:
            shutil.copytree(source, working)
        except OSError as error:
            return _skipped(f"Could not copy the repository for the baseline run: {error}")
        write_files(working, extra_files)
        return run_suite(working, language=language, python=python)


def evaluate_patch(
    repo_path: str | Path,
    patch: str,
    workspace_root: str | Path,
    language: str = "",
    baseline: dict[str, Any] | None = None,
    extra_files: Mapping[str, str] | None = None,
    reproduction_tests: list[str] | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Apply ``patch`` to a fresh copy, run the suite, and compare it with the baseline.

    ``extra_files`` must be the same files the baseline ran with, and
    ``reproduction_tests`` the ids of the tests that reproduce the issue.

    Returns ``working_repo``, ``changed_files``, ``patch_status``,
    ``baseline_result``, ``test_result`` and ``test_verdict``. The patched copy
    is kept on disk so the repaired code can be inspected after the run.
    """
    if baseline is None:
        baseline = run_baseline(
            repo_path, workspace_root, language=language, extra_files=extra_files, python=python
        )

    patch_result = apply_patch(repo_path, new_workspace(repo_path, workspace_root), patch)

    if patch_result.status != "APPLIED":
        reason = patch_result.error or "The patch could not be applied."
        return {
            "working_repo": patch_result.working_repo,
            "changed_files": [],
            "patch_status": patch_result.status,
            "baseline_result": baseline,
            "test_result": _skipped(reason),
            "test_verdict": not_tested(f"The patch was not tested: {reason}", stage="apply").to_dict(),
        }

    # Written after `git apply`, so the patch is checked against the real repository.
    write_files(Path(patch_result.working_repo), extra_files)
    test_result, verdict = run_stages(
        Path(patch_result.working_repo),
        baseline,
        language=language,
        changed_files=patch_result.changed_files,
        reproduction_files=list(extra_files or {}) if reproduction_tests else [],
        reproduction_tests=reproduction_tests,
        python=python,
    )
    return {
        "working_repo": patch_result.working_repo,
        "changed_files": patch_result.changed_files,
        "patch_status": patch_result.status,
        "baseline_result": baseline,
        "test_result": test_result,
        "test_verdict": verdict.to_dict(),
    }


def run_stages(
    repository: Path,
    baseline: dict[str, Any],
    language: str = "",
    changed_files: list[str] | None = None,
    reproduction_files: list[str] | None = None,
    reproduction_tests: list[str] | None = None,
    python: str | None = None,
) -> tuple[dict[str, Any], TestVerdict]:
    """Test the patched copy cheapest stage first, stopping at the first that fails.

        syntax         compile the changed Python files          (no test runs)
        reproduction   only the reproduction test                stop if it still fails
        related        test files named after / importing the    stop if one breaks
                       changed files
        full           the whole suite                           the final verdict

    A partial stage is compared only with the baseline tests of the files it
    ran, so the tests it skipped do not count as broken. Returns the last
    stage's run (with ``stage`` and ``stages`` added) and its verdict.
    """
    changed_files = changed_files or []
    stages: list[dict[str, Any]] = []

    def finish(result: dict[str, Any], verdict: TestVerdict, stage: str) -> tuple[dict[str, Any], TestVerdict]:
        verdict.stage = stage
        return {**result, "stage": stage, "stages": stages}, verdict

    syntax_errors = compile_errors(repository, changed_files)
    stages.append({"name": "syntax", "status": "FAIL" if syntax_errors else "PASS", "detail": "; ".join(syntax_errors)})
    if syntax_errors:
        result = {**_skipped("The patched code does not compile."), "status": "FAIL", "errors": syntax_errors}
        return finish(result, not_tested(f"The patched code does not compile: {syntax_errors[0]}", "syntax"), "syntax")

    if not _uses_pytest(repository, language):
        result = run_suite(repository, python=python, language=language)
        stages.append(_stage("full", result))
        return finish(result, compare_runs(baseline, result, reproduction_tests), "full")

    if reproduction_files and reproduction_tests:
        result = run_suite(repository, python=python, language=language, paths=reproduction_files)
        stages.append(_stage("reproduction", result))
        verdict = compare_runs(_scoped(baseline, reproduction_files), result, reproduction_tests)
        if not all(outcome == PASSED for outcome in verdict.reproduction.values()):
            verdict.reason = (
                "The issue's reproduction test still fails, so the rest of the suite was not run. "
                + verdict.reason
            )
            return finish(result, verdict, "reproduction")

    related = related_tests(repository, changed_files, exclude=set(reproduction_files or []))
    if related:
        files = [*related, *(reproduction_files or [])]
        result = run_suite(repository, python=python, language=language, paths=files)
        stages.append({**_stage("related", result), "files": related})
        verdict = compare_runs(_scoped(baseline, files), result, reproduction_tests)
        if verdict.status == "failed":
            verdict.reason = f"{verdict.reason} Found by the related tests; the full suite was not run."
            return finish(result, verdict, "related")

    result = run_suite(repository, python=python, language=language)
    stages.append(_stage("full", result))
    return finish(result, compare_runs(baseline, result, reproduction_tests), "full")


def compile_errors(repository: Path, changed_files: list[str]) -> list[str]:
    """``file: message (line n)`` for each changed Python file that does not compile."""
    errors = []
    for relative in changed_files:
        path = repository / relative
        if not relative.endswith(".py") or not path.is_file():
            continue
        try:
            compile(path.read_text(encoding="utf-8"), relative, "exec")
        except SyntaxError as error:
            errors.append(f"{relative}: {error.msg} (line {error.lineno})")
        except (UnicodeDecodeError, ValueError) as error:
            errors.append(f"{relative}: {error}")
    return errors


def _uses_pytest(repository: Path, language: str) -> bool:
    try:
        command = select_test_command(repository, language=language or None)
    except ValueError:
        return False
    return command[:3] == [sys.executable, "-m", "pytest"]


def _scoped(baseline: dict[str, Any], files: list[str]) -> dict[str, Any]:
    """The baseline restricted to the tests of ``files`` -- what a partial run is compared with."""
    prefixes = {id_prefix(f) for f in files}

    def in_scope(test_id: str) -> bool:
        module = test_id.split("::")[0]
        return any(module == p or module.startswith(p + ".") for p in prefixes)

    return {
        **baseline,
        "cases": {t: o for t, o in (baseline.get("cases") or {}).items() if in_scope(t)},
        "messages": {t: m for t, m in (baseline.get("messages") or {}).items() if in_scope(t)},
    }


def _stage(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": result["status"], "passed": result["passed"], "failed": result["failed"]}


def write_files(repository: Path, files: Mapping[str, str] | None) -> None:
    """Write ``{relative path: content}`` into a working copy."""
    for relative, content in (files or {}).items():
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def new_workspace(repo_path: str | Path, workspace_root: str | Path) -> Path:
    """A fresh, non-existent directory for one patched copy of the repository."""
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return root / f"{Path(repo_path).name}-{stamp}"


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "status": "SKIPPED",
        "passed": 0,
        "failed": 0,
        "errors": [reason],
        "output": "",
        "cases": {},
        "messages": {},
    }
