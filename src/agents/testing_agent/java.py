"""Run a Java repository's tests without Maven or Gradle: our own JUnit 5 sandbox.

Repositories without a build file (or a machine without the build tool) still
get per-test results:

    1. every .java file is copied into a staging directory, named after its
       public class -- javac insists on that, and loose repositories often break it
    2. javac compiles them with the JUnit 5 standalone jar on the class path
    3. the JUnit console launcher runs the test classes and writes JUnit XML,
       which `junit.py` reads exactly as it reads pytest's report

The jar is downloaded once from Maven Central, checked against its published
SHA-1, and cached next to the Python test environments. The repository and the
working copy are never written to.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import urllib.request
from typing import Any

from src.agents.testing_agent.junit import ERROR, FAILED, PASSED, parse_junit_report

JUNIT_VERSION = "1.10.2"
JUNIT_URL = (
    "https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/"
    f"{JUNIT_VERSION}/junit-platform-console-standalone-{JUNIT_VERSION}.jar"
)
JUNIT_JAR = (
    Path(__file__).resolve().parents[3] / ".test_envs" / "junit" / f"junit-platform-console-standalone-{JUNIT_VERSION}.jar"
)
TIMEOUT_SECONDS = 300
BUILD_FILES = ("pom.xml", "build.gradle", "build.gradle.kts", "gradlew")

_PUBLIC_TYPE_RE = re.compile(r"^\s*public\s+(?:(?:final|abstract|sealed|strictfp)\s+)*(?:class|interface|enum|record)\s+(\w+)", re.M)
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
_TYPE_DECLARATION_RE = re.compile(
    r"^\s*(?:(?:public|protected|private|final|abstract|static|sealed|non-sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record|@interface)\s+([A-Za-z_]\w*)",
    re.M,
)
# Lines that may come before the first type: package, imports, comments.
_HEADER_LINE_RE = re.compile(r"^\s*(package\s|import\s|//|/\*|\*)")
_SKIP_DIRS = {".git", "build", "target", "out", ".gradle", "node_modules"}


def uses_java_sandbox(repository: str | Path, language: str) -> bool:
    """Java code with no Maven/Gradle build file to run its tests."""
    return language.lower() == "java" and not any((Path(repository) / name).exists() for name in BUILD_FILES)


def run_suite(repository: str | Path, paths: list[str] | None = None) -> dict[str, Any]:
    """Compile and test ``repository``; ``paths`` limits the run to those test files.

    Returns the shape `agent.run_suite` returns: status, passed, failed, errors,
    output, cases, messages, plus ``no_tests`` when it compiled but had no test.
    """
    try:
        jar = ensure_junit_jar()
    except OSError as error:
        return _result("FAIL", errors=[f"Could not get the JUnit runner: {error}"])

    sources = compile_set(java_sources(Path(repository)))
    if not sources:
        return _result("FAIL", errors=["The repository has no Java source files."])

    with tempfile.TemporaryDirectory(prefix="java-sandbox-") as scratch:
        staging, classes, reports = (Path(scratch) / name for name in ("src", "classes", "reports"))
        staged = stage_sources(sources, staging)

        compiled = _run(["javac", "-nowarn", "-encoding", "UTF-8", "-d", str(classes), "-cp", str(jar), *staged])
        if compiled.returncode != 0:
            message = _compile_errors(compiled.stdout + compiled.stderr, staging)
            return _result("FAIL", errors=[f"javac failed: {message}"], output=compiled.stdout + compiled.stderr)

        selectors = ["--select-class=" + _class_name(Path(repository) / p) for p in paths or []]
        launched = _run(
            ["java", "-jar", str(jar), "execute", "--disable-banner", "--details=none",
             f"--class-path={classes}", f"--reports-dir={reports}",
             *(selectors or ["--scan-class-path"])]
        )
        cases: dict[str, str] = {}
        messages: dict[str, str] = {}
        for report in reports.glob("TEST-*.xml"):
            more_cases, more_messages = parse_junit_report(report)
            cases.update(more_cases)
            messages.update(more_messages)

    output = (launched.stdout + launched.stderr)[-20_000:]
    passed = sum(outcome == PASSED for outcome in cases.values())
    failed = sum(outcome in (FAILED, ERROR) for outcome in cases.values())
    if not cases:
        return {**_result("FAIL", errors=["No JUnit test was found."], output=output), "no_tests": True}
    errors = [f"FAILED {test_id}: {message}" for test_id, message in messages.items()]
    return {
        "status": "PASS" if failed == 0 else "FAIL",
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "output": output,
        "cases": cases,
        "messages": messages,
    }


def ensure_junit_jar() -> Path:
    """Download the JUnit console launcher once, verified against Maven Central's SHA-1."""
    if JUNIT_JAR.is_file():
        return JUNIT_JAR
    JUNIT_JAR.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(JUNIT_URL, timeout=120) as response:
        data = response.read()
    with urllib.request.urlopen(JUNIT_URL + ".sha1", timeout=30) as response:
        expected = response.read().decode().split()[0].strip()
    if hashlib.sha1(data).hexdigest() != expected:
        raise OSError("the downloaded JUnit jar does not match its published SHA-1")
    partial = JUNIT_JAR.with_suffix(".part")
    partial.write_bytes(data)
    os.replace(partial, JUNIT_JAR)
    return JUNIT_JAR


def java_sources(repository: Path) -> list[Path]:
    return sorted(
        path for path in repository.rglob("*.java")
        if not _SKIP_DIRS.intersection(path.relative_to(repository).parts)
    )


def compile_set(sources: list[Path]) -> list[Path]:
    """The files javac compiles together: all of them, unless two declare the
    same class.

    Loose repositories hold unrelated programs side by side in the default
    package (OrderService.java and E-Commerce.java each with its own `class
    Product`), and javac rejects such a pair outright. Then the set is every
    file outside a clash, every test, and each clashing file that one of those
    uses by name -- a file declaring a class itself never pulls in another
    file's copy of it.
    """
    texts = {path: as_compiled(path.read_text(encoding="utf-8", errors="replace"), path.name) for path in sources}
    declared = {path: set(_TYPE_DECLARATION_RE.findall(text)) for path, text in texts.items()}
    owners: dict[str, list[Path]] = {}
    for path, names in declared.items():
        for name in names:
            owners.setdefault(name, []).append(path)
    clashing = {path for paths in owners.values() if len(paths) > 1 for path in paths}
    if not clashing:
        return sources

    chosen = [path for path in sources if path not in clashing or "@Test" in texts[path]]
    queue = list(chosen)
    while queue:
        path = queue.pop()
        for name in set(re.findall(r"\b[A-Za-z_]\w*\b", texts[path])) - declared[path]:
            # A name two files declare is ambiguous; leave both out.
            found = owners.get(name, [])
            if len(found) == 1 and found[0] not in chosen:
                chosen.append(found[0])
                queue.append(found[0])
    return sorted(chosen)


def as_compiled(text: str, file_name: str) -> str:
    """The file as the sandbox compiles it: members with no class around them
    (a loose ``public int total(...) { ... }``, which javac rejects with "class,
    interface, enum, or record expected") go inside ``public class <file stem>``.

    The class opens on the first member's line, so javac's line numbers still
    match the repository's file. Any other file is returned as it is.
    """
    lines = text.splitlines()
    first = next((i for i, line in enumerate(lines) if line.strip() and not _HEADER_LINE_RE.match(line)), None)
    if first is None or _TYPE_DECLARATION_RE.search(text):
        return text
    name = re.sub(r"\W", "_", PurePosixPath(file_name).stem)
    name = f"_{name}" if name[0].isdigit() else name
    lines[first] = f"public class {name} {{ {lines[first]}"
    return "\n".join(lines) + "\n}\n"


def stage_sources(sources: list[Path], staging: Path) -> list[str]:
    """Copy each file to <package dirs>/<its public class>.java; return the staged paths."""
    staged = []
    for source in sources:
        text = as_compiled(source.read_text(encoding="utf-8", errors="replace"), source.name)
        package = _PACKAGE_RE.search(text)
        public = _PUBLIC_TYPE_RE.search(text)
        directory = staging.joinpath(*package.group(1).split(".")) if package else staging
        target = directory / (f"{public.group(1)}.java" if public else source.name)
        if target.exists():  # two files claiming one name: keep both compilable
            target = directory / f"_{len(staged)}_{target.name}"
        directory.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        staged.append(str(target))
    return staged


def _class_name(path: Path) -> str:
    """The fully qualified name of the public class in a test file."""
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    package = _PACKAGE_RE.search(text)
    public = _PUBLIC_TYPE_RE.search(text)
    name = public.group(1) if public else PurePosixPath(path.name).stem
    return f"{package.group(1)}.{name}" if package else name


def _compile_errors(output: str, staging: Path) -> str:
    """The first few javac error lines, with the staging prefix removed."""
    lines = [line.replace(str(staging) + os.sep, "") for line in output.splitlines() if "error:" in line]
    return "; ".join(lines[:3]) or output.strip()[:300]


def _run(command: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(command, 127, "", f"{command[0]} is not installed (install a JDK 11+).")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 124, "", f"{command[0]} timed out after {TIMEOUT_SECONDS}s.")


def _result(status: str, errors: list[str] | None = None, output: str = "") -> dict[str, Any]:
    return {"status": status, "passed": 0, "failed": 0, "errors": errors or [], "output": output,
            "cases": {}, "messages": {}}
