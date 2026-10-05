"""The reproduction test: generated, validated on the unpatched code, then used as evidence.

The model is replaced by a fake `Writer` that returns canned files, so these run
without Ollama. Each test works on a copy of `orchestrator/sample_repo`.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from src.agents.testing_agent import evaluate_patch, run_suite
from src.agents.testing_agent.reproduction import (
    REPRO_FILE,
    ReproductionError,
    module_name,
    prepare_reproduction,
    prune_tests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_REPO = PROJECT_ROOT / "orchestrator" / "sample_repo"
FIX_PATCH = PROJECT_ROOT / "orchestrator" / "fixtures" / "sample_repo_fix.patch"

ISSUE = "get_user crashes with a KeyError for an unknown user; it should return None."
CHUNKS = [
    {"file": "users.py", "name": "get_user", "code": "def get_user(users, name):\n    return users[name]\n"},
    {"file": "users.py", "name": "login", "code": "def login(users, name, password): ..."},
]

HEADER = 'from users import get_user, login\n\nUSERS = {"ada": {"password": "lovelace"}}\n\n'
REPRODUCES = HEADER + 'def test_issue_unknown_user_is_none():\n    assert get_user(USERS, "ghost") is None\n'
PASSES_ON_BUG = HEADER + 'def test_issue_known_user():\n    assert get_user(USERS, "ada")\n'
ASSERTS_THE_BUG = (
    "import pytest\n" + REPRODUCES + "\n\ndef test_issue_still_raises():\n"
    '    with pytest.raises(KeyError):\n        get_user(USERS, "ghost")\n'
)
BROKEN = HEADER + "def test_issue_typo():\n    assert get_usr(USERS, 'ghost') is None\n"
BAD_IMPORT = "from no_such_module import thing\n\ndef test_issue_x():\n    assert thing\n"


class FakeWriter:
    """Returns the given replies in order and records every prompt it was sent."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


@pytest.fixture
def repo(tmp_path):
    destination = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, destination)
    return destination


def prepare(repo: Path, tmp_path: Path, writer, **kwargs):
    return prepare_reproduction(repo, ISSUE, CHUNKS, tmp_path / "workspaces", writer, **kwargs)


def test_a_test_that_fails_on_the_bug_is_accepted(repo, tmp_path):
    reproduction, baseline = prepare(repo, tmp_path, FakeWriter(f"```python\n{REPRODUCES}```"))

    assert reproduction["status"] == "accepted"
    assert reproduction["tests"] == ["test_issue_reproduction::test_issue_unknown_user_is_none"]
    assert reproduction["messages"][reproduction["tests"][0]] == "KeyError: 'ghost'"
    assert baseline["cases"]["test_issue_reproduction::test_issue_unknown_user_is_none"] == "failed"
    assert not (repo / REPRO_FILE).exists()  # never written into the original


def test_tests_that_pass_on_the_bug_are_removed(repo, tmp_path):
    reproduction, baseline = prepare(repo, tmp_path, FakeWriter(ASSERTS_THE_BUG))

    assert reproduction["status"] == "accepted"
    assert "test_issue_still_raises" not in reproduction["source"]
    assert "test_issue_reproduction::test_issue_still_raises" not in baseline["cases"]


def test_a_test_that_passes_on_the_bug_is_retried_with_feedback(repo, tmp_path):
    writer = FakeWriter(PASSES_ON_BUG, REPRODUCES)

    reproduction, _ = prepare(repo, tmp_path, writer)

    assert reproduction["status"] == "accepted"
    assert reproduction["attempts"] == 2
    assert "YOUR PREVIOUS ATTEMPT WAS REJECTED" in writer.prompts[1]
    assert "Every test passes on the unpatched code" in writer.prompts[1]


def test_a_broken_test_is_rejected_and_the_plain_baseline_is_used(repo, tmp_path):
    reproduction, baseline = prepare(repo, tmp_path, FakeWriter(BROKEN, BROKEN))

    assert reproduction["status"] == "rejected"
    assert "NameError" in reproduction["reason"]
    assert reproduction["tests"] == []
    assert not any(test_id.startswith("test_issue_reproduction") for test_id in baseline["cases"])
    assert len(baseline["cases"]) == 4


def test_a_test_that_cannot_import_is_retried(repo, tmp_path):
    writer = FakeWriter(BAD_IMPORT, REPRODUCES)

    reproduction, _ = prepare(repo, tmp_path, writer)

    assert reproduction["attempts"] == 2
    assert "collection failure" in writer.prompts[1]
    assert reproduction["status"] == "accepted"


def test_a_test_file_that_cannot_import_does_not_hide_the_other_tests(repo):
    (repo / REPRO_FILE).write_text(BAD_IMPORT, encoding="utf-8")

    result = run_suite(repo)

    assert result["cases"]["test_issue_reproduction"] == "error"
    assert sum(test_id.startswith("test_users::") for test_id in result["cases"]) == 4


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("def test_issue(:\n    pass\n", "not valid Python"),
        ("from users import get_user\n\nX = 1\n", "no test function"),
        (
            "def get_user(users, name):\n    return users.get(name)\n\n"
            "def test_issue():\n    assert get_user({}, 'x') is None\n",
            "redefines get_user",
        ),
    ],
)
def test_unusable_files_are_rejected_before_running(repo, tmp_path, reply, reason):
    reproduction, _ = prepare(repo, tmp_path, FakeWriter(reply), max_attempts=1)

    assert reproduction["status"] == "rejected"
    assert reason in reproduction["reason"]


def test_an_unreachable_model_skips_the_reproduction(repo, tmp_path):
    def unreachable(prompt: str) -> str:
        raise ReproductionError("Cannot reach Ollama")

    reproduction, baseline = prepare(repo, tmp_path, unreachable)

    assert reproduction["status"] == "skipped"
    assert reproduction["reason"] == "Cannot reach Ollama"
    assert len(baseline["cases"]) == 4


def test_other_languages_are_skipped(repo, tmp_path):
    writer = FakeWriter()

    reproduction, _ = prepare(repo, tmp_path, writer, language="javascript")

    assert reproduction["status"] == "skipped"
    assert writer.prompts == []


def test_the_reproduction_test_decides_the_verdict(repo, tmp_path):
    reproduction, baseline = prepare(repo, tmp_path, FakeWriter(REPRODUCES))
    extra = {reproduction["path"]: reproduction["source"]}

    result = evaluate_patch(
        repo,
        FIX_PATCH.read_text(encoding="utf-8"),
        tmp_path / "workspaces",
        baseline=baseline,
        extra_files=extra,
        reproduction_tests=reproduction["tests"],
    )

    verdict = result["test_verdict"]
    assert verdict["status"] == "solved"
    assert verdict["reproduction"] == {"test_issue_reproduction::test_issue_unknown_user_is_none": "passed"}
    assert "reproduction test now passes" in verdict["reason"]
    assert (Path(result["working_repo"]) / REPRO_FILE).exists()


def test_module_name():
    assert module_name("users.py") == "users"
    assert module_name("pkg/users.py") == "pkg.users"
    assert module_name("src/pkg/users.py") == "pkg.users"
    assert module_name("pkg/__init__.py") == "pkg"


def test_prune_keeps_only_the_named_tests_and_their_decorators():
    source = (
        "import pytest\n\n"
        "@pytest.mark.parametrize('x', [1, 2])\n"
        "def test_keep(x):\n    assert x\n\n"
        "@pytest.mark.slow\n"
        "def test_drop():\n    assert True\n"
    )

    pruned = prune_tests(source, keep=["test_issue_reproduction::test_keep[1]"])

    assert "def test_keep" in pruned
    assert "test_drop" not in pruned
    assert "pytest.mark.slow" not in pruned
