"""Pick the tests most likely to exercise a patch, so a bad patch fails fast.

A changed source file ``pkg/users.py`` is related to a test file when the test
file is named after it (``test_users.py`` / ``users_test.py``, anywhere in the
repository) or imports it (``import pkg.users``, ``from pkg.users import ...``,
``from pkg import users``). A changed test file is related to itself.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "site-packages", "__pycache__", ".tox"}
_MAX_RELATED = 20


def related_tests(repository: str | Path, changed_files: list[str], exclude: set[str] | None = None) -> list[str]:
    """Relative paths of the test files related to ``changed_files``, sorted."""
    root = Path(repository)
    exclude = exclude or set()
    changed = [f for f in changed_files if f.endswith(".py")]
    sources = [f for f in changed if not is_test_file(f)]

    related = {f for f in changed if is_test_file(f) and (root / f).is_file()}
    if sources:
        patterns = [_import_pattern(f) for f in sources]
        stems = {PurePosixPath(f).stem for f in sources}
        names = {f"test_{stem}.py" for stem in stems} | {f"{stem}_test.py" for stem in stems}
        for path in _test_files(root):
            relative = path.relative_to(root).as_posix()
            if path.name in names or _imports_any(path, patterns):
                related.add(relative)

    return sorted(related - exclude)[:_MAX_RELATED]


def id_prefix(file_path: str) -> str:
    """``tests/test_users.py`` -> ``tests.test_users``: the prefix pytest gives its test ids."""
    return PurePosixPath(file_path).with_suffix("").as_posix().replace("/", ".")


def is_test_file(file_path: str) -> bool:
    name = PurePosixPath(file_path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _test_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if is_test_file(path.name) and not _SKIP_DIRS.intersection(path.relative_to(root).parts):
            yield path


def _import_pattern(source_file: str) -> re.Pattern[str]:
    """Matches the ways a test can import ``source_file`` as a module."""
    parts = list(PurePosixPath(source_file).with_suffix("").parts)
    if parts and parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]  # src layout: the package is imported without "src."
    dotted = re.escape(".".join(parts))
    alternatives = [rf"^\s*import\s+{dotted}\b", rf"^\s*from\s+{dotted}\s+import\b"]
    if len(parts) > 1:
        parent, name = re.escape(".".join(parts[:-1])), re.escape(parts[-1])
        alternatives.append(rf"^\s*from\s+{parent}\s+import\s+[^\n]*\b{name}\b")
    return re.compile("|".join(alternatives), re.MULTILINE)


def _imports_any(path: Path, patterns: list[re.Pattern[str]]) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return any(pattern.search(text) for pattern in patterns)
