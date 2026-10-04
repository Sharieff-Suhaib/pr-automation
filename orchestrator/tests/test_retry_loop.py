"""The retry loop: testing_agent -> coding_agent until solved or out of attempts.

The repository and recommendation agents and the model are replaced by fakes,
so the real graph runs end to end in seconds on a copy of `sample_repo`.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import shutil

import pytest

from orchestrator import adapters
from orchestrator.adapters import AdapterError
from orchestrator.manager_agent import solve_issue

ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1]
SAMPLE_REPO = ORCHESTRATOR_DIR / "sample_repo"
FIX_PATCH = (ORCHESTRATOR_DIR / "fixtures" / "sample_repo_fix.patch").read_text(encoding="utf-8")
USERS = (SAMPLE_REPO / "users.py").read_text(encoding="utf-8")


def edit(*replacements: tuple[str, str]) -> str:
    modified = USERS
    for old, new in replacements:
        modified = modified.replace(old, new)
    return "".join(
        difflib.unified_diff(
            USERS.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="a/users.py",
            tofile="b/users.py",
        )
    )


# Fixes get_user but makes login reject everyone: breaks a passing test.
BREAKS_LOGIN = edit(
    ("    return users[name]\n", "    return users.get(name)\n"),
    ('    user = get_user(users, name)\n    return user["password"] == password\n', "    return False\n"),
)
NO_OP = edit(("with a reporting bug", "with a known reporting bug"))
DOES_NOT_APPLY = FIX_PATCH.replace("def get_user", "def fetch_user")


class FakeCoder:
    """Stands in for `adapters.generate_patch`: replays patches (or errors) in order."""

    def __init__(self, *results):
        self.results = list(results)
        self.feedback: list[str] = []

    def __call__(self, **kwargs):
        self.feedback.append(kwargs.get("feedback", ""))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result, "ollama-function"


@pytest.fixture
def run(tmp_path, monkeypatch):
    repo = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, repo)
    monkeypatch.setattr(adapters, "WORKSPACES_DIR", tmp_path / "workspaces")
    monkeypatch.setattr(adapters, "TEST_ENVS_DIR", tmp_path / "test_envs")
    monkeypatch.setattr(
        adapters,
        "analyze_repository",
        lambda repo_url, issue, top_k=5: {
            "repo_path": str(repo),
            "relevant_code": [{"file": "users.py", "name": "get_user", "type": "function",
                               "start_line": 4, "end_line": 6, "code": "def get_user(users, name): ..."}],
            "relevant_files": ["users.py"],
            "language": "python",
            "indexed_chunks": 1,
        },
    )
    monkeypatch.setattr(
        adapters,
        "recommend",
        lambda issue, code="", language="python", top_k=3: {
            "similar_bugs": [], "bug_type": "KeyError", "strategy": "check membership",
            "strategy_source": "test", "tools": [], "tests": [],
        },
    )

    def solve(coder, **kwargs):
        monkeypatch.setattr(adapters, "generate_patch", coder)
        kwargs.setdefault("reproduce", False)
        return solve_issue(repo_url=str(repo), issue="get_user crashes on unknown users", **kwargs)

    return solve


def test_a_broken_first_patch_is_retried_with_feedback_and_solved(run):
    coder = FakeCoder(BREAKS_LOGIN, FIX_PATCH)

    report = run(coder)

    assert report["status"] == "solved"
    assert [row["status"] for row in report["attempts"]] == ["failed", "solved"]
    assert coder.feedback[0] == ""
    assert "test_users::test_login_succeeds_for_valid_credentials" in coder.feedback[1]
    assert "do not break these" in coder.feedback[1]
    assert report["patch"] == FIX_PATCH


def test_a_repeated_patch_is_called_out_in_the_feedback(run):
    coder = FakeCoder(BREAKS_LOGIN, BREAKS_LOGIN, FIX_PATCH)

    report = run(coder)

    assert "identical to an earlier rejected attempt" not in coder.feedback[1]
    assert "identical to an earlier rejected attempt" in coder.feedback[2]
    assert report["status"] == "solved"
    assert [row["patch"] for row in report["attempts"]] == [BREAKS_LOGIN, BREAKS_LOGIN, FIX_PATCH]


def test_when_nothing_solves_the_best_attempt_is_reported(run):
    report = run(FakeCoder(BREAKS_LOGIN, NO_OP, DOES_NOT_APPLY))

    assert [row["status"] for row in report["attempts"]] == ["failed", "unverified", "failed"]
    assert report["status"] == "unverified"
    assert report["patch"] == NO_OP  # attempt 2, not the last one
    assert report["patch_status"] == "APPLIED"
    assert report["test_verdict"]["status"] == "unverified"


def test_a_coding_failure_counts_as_an_attempt(run):
    coder = FakeCoder(AdapterError("Ollama returned garbage"), FIX_PATCH)

    report = run(coder)

    assert report["status"] == "solved"
    assert len(report["attempts"]) == 2
    assert "No usable patch was produced. Ollama returned garbage" in coder.feedback[1]


def test_max_attempts_is_respected(run):
    report = run(FakeCoder(BREAKS_LOGIN, BREAKS_LOGIN), max_attempts=2)

    assert len(report["attempts"]) == 2
    assert report["status"] == "failed"


def test_a_solved_first_attempt_does_not_retry(run):
    coder = FakeCoder(FIX_PATCH)

    report = run(coder)

    assert report["status"] == "solved"
    assert len(report["attempts"]) == 1


def test_the_stub_backend_never_retries(run):
    # Replaying the same fixture cannot produce a different patch.
    report = run(FakeCoder(BREAKS_LOGIN), codegen_backend="stub")

    assert len(report["attempts"]) == 1
    assert report["status"] == "failed"
