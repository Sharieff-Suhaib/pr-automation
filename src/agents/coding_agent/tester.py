"""Run a repository's native test command and summarize the result."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import re
import subprocess
import sys
from typing import Mapping, Sequence


TEST_TIMEOUT_SECONDS = 300
_SUMMARY_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors)\b")

# Marker files identify the test ecosystem. Python remains the fallback because
# the first version of this repair workflow targets pytest repositories.
FRAMEWORK_MARKERS: tuple[tuple[str, str], ...] = (
    ("Cargo.toml", "rust"),
    ("go.mod", "go"),
    ("package.json", "javascript"),
    ("pom.xml", "java_maven"),
    ("build.gradle", "java_gradle"),
    ("build.gradle.kts", "java_gradle"),
    ("gradlew", "java_gradle_wrapper"),
    (".sln", "c_sharp"),
    (".csproj", "c_sharp"),
    ("Gemfile", "ruby"),
    ("composer.json", "php"),
    ("Package.swift", "swift"),
    ("pubspec.yaml", "dart"),
    ("CMakeLists.txt", "cpp_cmake"),
    ("pytest.ini", "python"),
    ("pyproject.toml", "python"),
    ("setup.cfg", "python"),
    ("tox.ini", "python"),
)


@dataclass
class TestResult:
    """A JSON-friendly summary of one test-command run."""

    status: str
    passed: int
    failed: int
    errors: list[str]
    output: str

    def to_dict(self) -> dict[str, object]:
        """Return the result in the format consumed by the repair workflow."""
        return asdict(self)


def run_tests(
    working_repo: str | Path,
    tests: Sequence[str] | None = None,
    language: str | None = None,
    command: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> TestResult:
    """Run a native test command in ``working_repo``.

    The framework is detected from project files unless ``language`` is given.
    Use ``command`` for an unsupported language or custom project command.
    Focused ``tests`` are supported for Python and JavaScript/TypeScript.
    ``env`` adds environment variables for the test command.
    """
    repository = Path(working_repo).resolve()
    if not repository.is_dir():
        return TestResult(
            status="FAIL",
            passed=0,
            failed=0,
            errors=[f"Working repository does not exist: {repository}"],
            output="",
        )

    try:
        test_command = list(command) if command else select_test_command(
            repository, tests=tests, language=language
        )
    except ValueError as error:
        return TestResult("FAIL", 0, 0, [str(error)], "")

    try:
        completed = subprocess.run(
            test_command,
            cwd=repository,
            text=True,
            capture_output=True,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
            env={**os.environ, **env} if env else None,
        )
    except subprocess.TimeoutExpired as error:
        output = _combine_output(error.stdout, error.stderr)
        return TestResult(
            status="FAIL",
            passed=0,
            failed=0,
            errors=[f"Test command timed out after {TEST_TIMEOUT_SECONDS} seconds."],
            output=output,
        )
    except FileNotFoundError:
        return TestResult(
            status="FAIL",
            passed=0,
            failed=0,
            errors=[f"Test command is not installed: {test_command[0]}"],
            output="",
        )

    output = _combine_output(completed.stdout, completed.stderr)
    passed, failed, error_count = _parse_summary(output)
    errors = _error_messages(output)

    if error_count and not errors:
        errors.append(f"The test runner reported {error_count} error(s).")
    if completed.returncode != 0 and not errors:
        errors.append("The test command exited with a non-zero status.")
    if passed == 0 and completed.returncode == 5:
        errors.append("pytest collected no tests.")

    is_python = test_command[:3] == [sys.executable, "-m", "pytest"]
    status = "PASS" if completed.returncode == 0 and (passed > 0 or not is_python) else "FAIL"
    return TestResult(status, passed, failed + error_count, errors, output)


def detect_framework(repository: str | Path) -> str | None:
    """Identify the likely test ecosystem from repository marker files."""
    root = Path(repository)
    for marker, framework in FRAMEWORK_MARKERS:
        if marker.startswith("."):
            if any(root.glob(f"*{marker}")):
                return framework
        elif (root / marker).exists():
            return framework

    if any(root.rglob("*.py")):
        return "python"
    return None


def select_test_command(
    repository: str | Path,
    tests: Sequence[str] | None = None,
    language: str | None = None,
) -> list[str]:
    """Choose the normal test command for a supported repository language."""
    framework = _normalise_framework(language, repository)
    selected_tests = list(tests or [])

    if framework == "python":
        return [sys.executable, "-m", "pytest", *selected_tests]
    if framework in {"javascript", "typescript"}:
        return ["npm", "test", *(["--", *selected_tests] if selected_tests else [])]
    if framework == "go":
        return ["go", "test", "./..."]
    if framework == "rust":
        return ["cargo", "test"]
    if framework == "java_maven":
        return ["mvn", "test"]
    if framework == "java_gradle":
        return ["gradle", "test"]
    if framework == "java_gradle_wrapper":
        return ["./gradlew", "test"]
    if framework == "c_sharp":
        return ["dotnet", "test"]
    if framework == "ruby":
        return ["bundle", "exec", "rspec"]
    if framework == "php":
        return ["composer", "test"]
    if framework == "swift":
        return ["swift", "test"]
    if framework == "dart":
        return ["dart", "test"]
    if framework == "cpp_cmake":
        return ["ctest", "--test-dir", "build", "--output-on-failure"]

    raise ValueError(
        "No supported test framework was detected. Supply language= or command=."
    )


def _normalise_framework(language: str | None, repository: str | Path) -> str | None:
    """Map friendly language names to the internal runner names."""
    if language is None:
        return detect_framework(repository)

    aliases = {
        "js": "javascript",
        "ts": "typescript",
        "c#": "c_sharp",
        "csharp": "c_sharp",
        "c++": "cpp_cmake",
        "cpp": "cpp_cmake",
    }
    if language == "java":
        detected = detect_framework(repository)
        return detected if detected and detected.startswith("java_") else "java_maven"
    return aliases.get(language.lower(), language.lower())


def _parse_summary(output: str) -> tuple[int, int, int]:
    """Read common passed, failed, and error counts from runner output."""
    counts = {"passed": 0, "failed": 0, "error": 0, "errors": 0}
    for count, label in _SUMMARY_RE.findall(output):
        counts[label] = int(count)
    return counts["passed"], counts["failed"], counts["error"] + counts["errors"]


def _error_messages(output: str) -> list[str]:
    """Extract concise failure lines from pytest's short summary, when present."""
    marker = "short test summary info"
    lower_output = output.lower()
    position = lower_output.rfind(marker)
    if position == -1:
        return []

    messages: list[str] = []
    for line in output[position + len(marker) :].splitlines():
        stripped = line.strip()
        if stripped.startswith(("FAILED ", "ERROR ")):
            messages.append(stripped)
    return messages


def _combine_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    """Keep both streams so test failures remain visible to the caller."""
    def as_text(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""

    return "\n".join(part for part in (as_text(stdout), as_text(stderr)) if part).strip()
