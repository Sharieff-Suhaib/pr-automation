"""Per-repository test environments and cleanup.

`pip` and `venv` are replaced by a fake runner, so these run offline; the fake
"creates" the venv by writing its python file.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.agents.testing_agent.environment import (
    cleanup_directories,
    collect_requirements,
    prepare_environment,
    working_copy_env,
)


class FakeRunner:
    """Records commands; `fail_on` makes the command containing that word fail."""

    def __init__(self, fail_on: str = ""):
        self.commands: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            return subprocess.CompletedProcess(command, 1, "", "ERROR: No matching distribution for nope")
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[3]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.write_text("")
        return subprocess.CompletedProcess(command, 0, "", "")


def write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "proj"
    write(root, "requirements.txt", "requests>=2\n")
    return root


def test_a_repository_without_dependencies_uses_this_interpreter(tmp_path):
    runner = FakeRunner()

    env = prepare_environment(tmp_path, tmp_path / "envs", run=runner)

    assert (env.status, env.python) == ("none", sys.executable)
    assert runner.commands == []


def test_disabled_environments_use_this_interpreter(repo, tmp_path):
    env = prepare_environment(repo, tmp_path / "envs", run=FakeRunner(), enabled=False)

    assert (env.status, env.python) == ("disabled", sys.executable)


def test_an_environment_is_built_once_and_then_reused(repo, tmp_path):
    runner = FakeRunner()

    first = prepare_environment(repo, tmp_path / "envs", run=runner)
    second = prepare_environment(repo, tmp_path / "envs", run=runner)

    assert first.status == second.status == "ready"
    assert (first.reused, second.reused) == (False, True)
    assert first.python == second.python != sys.executable
    assert len(runner.commands) == 2  # venv + pip, only for the first call
    assert runner.commands[1][-2:] == ["pytest", "requests>=2"]


def test_changing_the_dependencies_builds_a_new_environment(repo, tmp_path):
    first = prepare_environment(repo, tmp_path / "envs", run=FakeRunner())
    write(repo, "requirements.txt", "requests>=3\n")

    second = prepare_environment(repo, tmp_path / "envs", run=FakeRunner())

    assert first.path != second.path
    assert not second.reused


def test_a_failed_install_falls_back_and_leaves_nothing_behind(repo, tmp_path):
    env = prepare_environment(repo, tmp_path / "envs", run=FakeRunner(fail_on="install"))

    assert (env.status, env.python) == ("failed", sys.executable)
    assert "No matching distribution" in env.detail
    assert list((tmp_path / "envs").iterdir()) == []


def test_requirements_never_install_the_repository_itself(tmp_path):
    write(tmp_path, "requirements.txt", "\n".join([
        "# comment", "requests==2.31  # pinned", ".", "-e .", "--index-url https://example.org",
        "-r requirements-dev.txt", "numpy",
    ]))
    write(tmp_path, "requirements-dev.txt", "pytest-mock\nnumpy\n")
    write(tmp_path, "pyproject.toml", (
        '[project]\nname = "proj"\ndependencies = ["click>=8"]\n'
        '[project.optional-dependencies]\ntest = ["hypothesis"]\ndocs = ["sphinx"]\n'
    ))
    write(tmp_path, "setup.cfg", "[options]\ninstall_requires =\n    attrs\n[options.extras_require]\ndev = tox\n")

    arguments = collect_requirements(tmp_path)

    assert arguments == [
        "requests==2.31",
        "-r", str((tmp_path / "requirements-dev.txt").resolve()),
        "numpy",
        "pytest-mock",
        "click>=8",
        "hypothesis",
        "attrs",
        "tox",
    ]


def test_working_copy_env_puts_the_copy_and_its_src_first(tmp_path):
    (tmp_path / "src").mkdir()

    paths = working_copy_env(tmp_path)["PYTHONPATH"].split(os.pathsep)

    assert paths[:2] == [str((tmp_path / "src").resolve()), str(tmp_path.resolve())]


def test_cleanup_keeps_the_newest_directories(tmp_path):
    for index, name in enumerate(["old", "middle", "new"]):
        (tmp_path / name).mkdir()
        stamp = time.time() - 100 + index * 10
        os.utime(tmp_path / name, (stamp, stamp))
    (tmp_path / "a-file.txt").write_text("kept")

    removed = cleanup_directories(tmp_path, keep=2)

    assert removed == [str(tmp_path / "old")]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a-file.txt", "middle", "new"]
    assert cleanup_directories(tmp_path / "missing", keep=1) == []
