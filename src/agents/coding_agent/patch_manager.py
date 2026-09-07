"""Create a safe working copy and apply a generated repair patch to it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import shutil
import subprocess


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
    """Return patched files and reject paths that could escape the repository."""
    if not patch.strip():
        raise ValueError("Patch is empty.")

    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue

        path = line[4:].split("\t", maxsplit=1)[0].strip()
        if path == "/dev/null":
            continue
        if path.startswith(("a/", "b/")):
            path = path[2:]
        _validate_relative_path(path)
        if path not in paths:
            paths.append(path)

    if not paths or "\n+++ " not in patch:
        raise ValueError("Patch must contain --- and +++ file headers.")
    return paths


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
        return subprocess.run(
            command,
            cwd=repo,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(command, 1, "", "git is not installed.")


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
