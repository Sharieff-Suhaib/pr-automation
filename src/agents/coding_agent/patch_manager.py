"""Create a safe working copy and apply a generated repair patch to it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import shutil
import subprocess

from src.agents.coding_agent.diff_validator import (
    DiffError,
    describe_diff_error,
    validate_unified_diff,
)


@dataclass
class PatchResult:
    """The outcome of copying a repository and applying one patch."""

    status: str
    working_repo: str
    changed_files: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation for the next workflow phase."""
        return asdict(self)


def apply_patch(
    original_repo: str | Path,
    working_repo: str | Path,
    patch: str,
) -> PatchResult:
    """Copy ``original_repo``, then apply ``patch`` only in ``working_repo``.

    The destination must not exist already. This prevents accidentally running
    a repair against an old working copy or overwriting a user's files.
    """
    source = Path(original_repo).resolve()
    destination = Path(working_repo).resolve()

    try:
        _copy_repository(source, destination)
    except ValueError as error:
        return PatchResult("WORKING_COPY_FAILED", str(destination), [], str(error))
    except OSError as error:
        return PatchResult("WORKING_COPY_FAILED", str(destination), [], str(error))

    try:
        changed_files = _changed_files(patch)
    except ValueError as error:
        return PatchResult("PATCH_APPLY_FAILED", str(destination), [], str(error))

    check = _git_apply(destination, patch, check_only=True)
    if check.returncode != 0:
        return PatchResult(
            "PATCH_APPLY_FAILED",
            str(destination),
            [],
            _command_error(check),
        )

    applied = _git_apply(destination, patch, check_only=False)
    if applied.returncode != 0:
        return PatchResult(
            "PATCH_APPLY_FAILED",
            str(destination),
            [],
            _command_error(applied),
        )

    return PatchResult("APPLIED", str(destination), changed_files)


def _copy_repository(source: Path, destination: Path) -> None:
    """Copy a repository after checking the source and destination are safe."""
    if not source.is_dir():
        raise ValueError(f"Original repository does not exist: {source}")
    if destination.exists():
        raise ValueError(f"Working repository already exists: {destination}")
    if _is_inside(destination, source):
        raise ValueError("Working repository cannot be inside the original repository.")

    shutil.copytree(source, destination)


def _changed_files(patch: str) -> list[str]:
    """Return patched files and reject paths that could escape the repository.

    The diff is fully parsed first, so a malformed patch is reported in terms
    of the mistake ("a file header must be a path...") rather than as Git's
    ``corrupt patch at line N``, which counts from where Git lost track.
    """
    try:
        files = validate_unified_diff(patch)
    except DiffError as error:
        raise ValueError(describe_diff_error(error, patch)) from error

    paths: list[str] = []
    for file_diff in files:
        for raw in (file_diff.old_path, file_diff.new_path):
            if raw == "/dev/null":
                continue
            path = raw[2:] if raw.startswith(("a/", "b/")) else raw
            _validate_relative_path(path)
            if path not in paths:
                paths.append(path)
    return paths


def check_patch(repo: str | Path, patch: str) -> tuple[bool, str]:
    """Ask Git whether ``patch`` would apply to ``repo``, changing nothing.

    Used by the code generator to turn a rejected patch into a retry with the
    real reason attached, instead of failing the whole run.  Returns
    ``(ok, error)``; ``error`` is empty when the patch would apply.
    """
    try:
        _changed_files(patch)
    except ValueError as error:
        return (False, str(error))

    result = _git_apply(Path(repo), patch, check_only=True)
    if result.returncode != 0:
        return (False, _command_error(result))
    return (True, "")


def _validate_relative_path(path: str) -> None:
    """Allow only normal paths relative to the working repository."""
    pure_path = PurePosixPath(path)
    if not path or pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Unsafe patch path: {path}")


def _git_apply(repo: Path, patch: str, check_only: bool) -> subprocess.CompletedProcess[str]:
    """Run Git's patch parser, optionally without modifying the working copy."""
    command = ["git", "apply", "--whitespace=nowarn"]
    if check_only:
        command.append("--check")
    try:
        # The patch is written to Git as bytes, not text. In text mode Python
        # rewrites every "\n" to os.linesep, so on Windows Git would receive a
        # CRLF patch and reject it against LF source files.
        result = subprocess.run(
            command,
            cwd=repo,
            input=patch.encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(command, 1, "", "git is not installed.")

    return subprocess.CompletedProcess(
        command,
        result.returncode,
        result.stdout.decode("utf-8", errors="replace"),
        result.stderr.decode("utf-8", errors="replace"),
    )


def _command_error(result: subprocess.CompletedProcess[str]) -> str:
    """Prefer Git's useful error stream, falling back to standard output."""
    return (result.stderr or result.stdout or "git apply failed.").strip()


def _is_inside(child: Path, parent: Path) -> bool:
    """Return whether ``child`` is nested inside ``parent``."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
