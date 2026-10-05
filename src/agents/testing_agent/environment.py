"""Give each repository's tests their own Python environment, and keep disk use bounded.

Running a cloned repository's tests with this project's interpreter fails as
soon as the repository needs a package this project does not have. So when the
repository declares dependencies, a virtual environment is built with them plus
pytest, and cached under a key derived from the dependency files -- it is built
once and reused until those files change.

The repository itself is deliberately *not* installed (no ``pip install -e .``):
tests must import the patched working copy, not the original clone. Instead the
working copy (and its ``src/`` directory) is put on ``PYTHONPATH``.

When the environment cannot be built (no network, a broken requirement), the
tests fall back to this project's interpreter -- the behaviour before
environments existed -- and the reason is reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import configparser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable

ENV_VERSION = "1"  # bump to invalidate every cached environment
INSTALL_TIMEOUT_SECONDS = 900

REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "requirements_dev.txt",
    "requirements_test.txt",
    "dev-requirements.txt",
    "test-requirements.txt",
)
PROJECT_FILES = ("pyproject.toml", "setup.cfg")
TEST_EXTRAS = ("test", "tests", "testing", "dev")

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass
class TestEnvironment:
    """Which interpreter runs a repository's tests, and why."""

    __test__ = False  # not a pytest test class, despite the name

    status: str  # "ready" | "none" (no dependencies declared) | "failed" | "disabled"
    python: str  # the interpreter to run pytest with
    detail: str
    reused: bool = False
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_environment(
    repo_path: str | Path,
    cache_root: str | Path,
    run: Runner = subprocess.run,
    enabled: bool = True,
) -> TestEnvironment:
    """Build (or reuse) the virtual environment for ``repo_path``'s tests."""
    if not enabled:
        return TestEnvironment("disabled", sys.executable, "Isolated test environments are turned off.")

    repository = Path(repo_path)
    files = dependency_files(repository)
    if not files:
        return TestEnvironment("none", sys.executable, "The repository declares no dependencies.")

    requirements = collect_requirements(repository)
    env_dir = Path(cache_root) / f"{repository.name}-{environment_key(repository, files)}"
    python = _venv_python(env_dir)
    if (env_dir / ".ready").is_file() and python.is_file():
        _touch(env_dir)
        return TestEnvironment("ready", str(python), f"Reused the environment for {', '.join(files)}.", True, str(env_dir))

    shutil.rmtree(env_dir, ignore_errors=True)  # a half-built environment from an interrupted run
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        _check(run([sys.executable, "-m", "venv", str(env_dir)], capture_output=True, text=True,
                   timeout=INSTALL_TIMEOUT_SECONDS, check=False), "create the virtual environment")
        _check(run([str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "pytest", *requirements], capture_output=True, text=True,
                   timeout=INSTALL_TIMEOUT_SECONDS, check=False, cwd=str(repository)),
               "install the dependencies")
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        shutil.rmtree(env_dir, ignore_errors=True)
        return TestEnvironment(
            "failed", sys.executable, f"Could not build an environment ({error}); using {sys.executable}."
        )

    (env_dir / ".ready").write_text(json.dumps({"files": files, "requirements": requirements}), encoding="utf-8")
    return TestEnvironment("ready", str(python), f"Built an environment for {', '.join(files)}.", False, str(env_dir))


def dependency_files(repository: Path) -> list[str]:
    """The dependency declarations at the repository root that the environment is built from."""
    return [name for name in (*REQUIREMENT_FILES, *PROJECT_FILES) if (repository / name).is_file()]


def environment_key(repository: Path, files: list[str]) -> str:
    """Changes whenever a dependency file, the Python version or ENV_VERSION changes."""
    digest = hashlib.sha256(f"{ENV_VERSION}|{sys.version}".encode())
    for name in files:
        digest.update(name.encode())
        digest.update((repository / name).read_bytes())
    return digest.hexdigest()[:12]


def collect_requirements(repository: Path) -> list[str]:
    """``pip install`` arguments for the repository's dependencies, never the repository itself."""
    groups: list[list[str]] = []
    for name in REQUIREMENT_FILES:
        path = repository / name
        if path.is_file():
            groups += _requirement_lines(path)
    groups += [[spec] for spec in _pyproject_requirements(repository / "pyproject.toml")]
    groups += [[spec] for spec in _setup_cfg_requirements(repository / "setup.cfg")]

    # Drop duplicates but keep `-r <file>` pairs together, in order.
    seen: set[tuple[str, ...]] = set()
    arguments: list[str] = []
    for group in groups:
        if tuple(group) not in seen:
            seen.add(tuple(group))
            arguments += group
    return arguments


def working_copy_env(working_repo: str | Path) -> dict[str, str]:
    """PYTHONPATH that makes the working copy (and a ``src/`` layout) importable."""
    root = Path(working_repo).resolve()
    paths = [str(root)]
    if (root / "src").is_dir():
        paths.insert(0, str(root / "src"))
    existing = os.environ.get("PYTHONPATH")
    return {"PYTHONPATH": os.pathsep.join([*paths, *([existing] if existing else [])])}


def cleanup_directories(root: str | Path, keep: int) -> list[str]:
    """Delete all but the ``keep`` most recently modified directories directly inside ``root``.

    Only direct children of ``root`` are touched. Returns the removed paths.
    """
    base = Path(root)
    if keep < 0 or not base.is_dir():
        return []
    children = [child for child in base.iterdir() if child.is_dir() and not child.is_symlink()]
    children.sort(key=lambda child: child.stat().st_mtime, reverse=True)
    removed = []
    for child in children[keep:]:
        if child.resolve().parent == base.resolve():
            shutil.rmtree(child, ignore_errors=True)
            removed.append(str(child))
    return removed


def _requirement_lines(path: Path) -> list[list[str]]:
    """Requirement specs from one file, minus anything that would install the repository itself."""
    arguments: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line in (".", "-e .", "-e.", "--editable .") or line.startswith(("-e file:.", "-e ./", "./", "file:.")):
            continue
        if line.startswith(("-r ", "-c ", "--requirement ", "--constraint ")):
            option, _, target = line.partition(" ")
            arguments.append([option, str((path.parent / target.strip()).resolve())])
            continue
        if line.startswith("-"):
            continue  # index URLs and other pip options are not passed through
        arguments.append([line])
    return arguments


def _pyproject_requirements(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        import tomllib  # noqa: PLC0415 -- Python 3.11+
    except ImportError:  # pragma: no cover - Python 3.10
        return []
    try:
        project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return []
    specs = list(project.get("dependencies", []))
    extras = project.get("optional-dependencies", {})
    for extra in TEST_EXTRAS:
        specs += extras.get(extra, [])
    return [spec for spec in specs if isinstance(spec, str)]


def _setup_cfg_requirements(path: Path) -> list[str]:
    if not path.is_file():
        return []
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error:
        return []
    specs = _cfg_list(parser.get("options", "install_requires", fallback=""))
    for extra in TEST_EXTRAS:
        specs += _cfg_list(parser.get("options.extras_require", extra, fallback=""))
    return specs


def _cfg_list(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip() and not line.strip().startswith("#")]


def _venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _check(result: subprocess.CompletedProcess, action: str) -> None:
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip().splitlines()
        raise RuntimeError(f"failed to {action}: {output[-1] if output else 'exit code ' + str(result.returncode)}")


def _touch(path: Path) -> None:
    """Mark a reused environment as recently used, so cleanup keeps it."""
    try:
        os.utime(path)
    except OSError:
        pass
