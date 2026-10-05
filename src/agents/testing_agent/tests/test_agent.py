"""End-to-end checks on a copy of `orchestrator/sample_repo`.

The sample has 4 tests; 2 fail before the fixture patch and all 4 pass after it.
Each test copies the sample into `tmp_path` so neither the checked-in sample nor
the project's `workspaces/` directory is written to.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import shutil

import pytest

from src.agents.coding_agent import tester
from src.agents.testing_agent import evaluate_patch, run_baseline, run_suite

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_REPO = PROJECT_ROOT / "orchestrator" / "sample_repo"
FIX_PATCH = PROJECT_ROOT / "orchestrator" / "fixtures" / "sample_repo_fix.patch"


@pytest.fixture
def repo(tmp_path):
    destination = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, destination)
    return destination


@pytest.fixture
def workspaces(tmp_path):
    return tmp_path / "workspaces"


def edit_patch(repo: Path, *replacements: tuple[str, str], file: str = "users.py") -> str:
    """A unified diff that applies each `(old, new)` replacement to one file of `repo`."""
    original = (repo / file).read_text(encoding="utf-8")
    modified = original
    for old, new in replacements:
        assert old in modified
        modified = modified.replace(old, new)
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{file}",
            tofile=f"b/{file}",
        )
    )


def test_baseline_records_each_test_and_leaves_no_copy_behind(repo, workspaces):
    baseline = run_baseline(repo, workspaces)

    assert baseline["status"] == "FAIL"
    assert (baseline["passed"], baseline["failed"]) == (2, 2)
    assert baseline["cases"] == {
        "test_users::test_known_user_is_returned": "passed",
        "test_users::test_missing_user_returns_none": "failed",
        "test_users::test_login_succeeds_for_valid_credentials": "passed",
        "test_users::test_login_rejects_unknown_user": "failed",
    }
    assert list(workspaces.iterdir()) == []
    assert not (repo / "__pycache__").exists()


def test_the_real_fix_is_solved(repo, workspaces):
    result = evaluate_patch(repo, FIX_PATCH.read_text(encoding="utf-8"), workspaces)

    verdict = result["test_verdict"]
    assert result["patch_status"] == "APPLIED"
    assert result["test_result"]["status"] == "PASS"
    assert verdict["status"] == "solved"
    assert sorted(verdict["fail_to_pass"]) == [
        "test_users::test_login_rejects_unknown_user",
        "test_users::test_missing_user_returns_none",
    ]
    assert verdict["pass_to_fail"] == []
    # The patched copy is kept for inspection; the original is untouched.
    assert "if name not in users" in (Path(result["working_repo"]) / "users.py").read_text()
    assert "if name not in users" not in (repo / "users.py").read_text()


def test_a_fix_that_breaks_a_passing_test_is_not_solved(repo, workspaces):
    # Fixes get_user, but login now rejects everyone.
    patch = edit_patch(
        repo,
        ("    return users[name]\n", "    return users.get(name)\n"),
        ('    user = get_user(users, name)\n    return user["password"] == password\n', "    return False\n"),
    )

    verdict = evaluate_patch(repo, patch, workspaces)["test_verdict"]

    assert verdict["status"] == "failed"
    assert verdict["pass_to_fail"] == ["test_users::test_login_succeeds_for_valid_credentials"]
    assert "test_users::test_missing_user_returns_none" in verdict["fail_to_pass"]


def test_a_patch_that_changes_no_behaviour_is_unverified(repo, workspaces):
    patch = edit_patch(repo, ("with a reporting bug", "with a known reporting bug"))

    verdict = evaluate_patch(repo, patch, workspaces)["test_verdict"]

    assert verdict["status"] == "unverified"
    assert len(verdict["fail_to_fail"]) == 2
    assert len(verdict["pass_to_pass"]) == 2


def test_a_syntax_error_is_caught_before_any_test_runs(repo, workspaces):
    patch = edit_patch(repo, ("def login(users, name, password):", "def login(users, name, password)"))

    result = evaluate_patch(repo, patch, workspaces)

    assert result["patch_status"] == "APPLIED"
    assert result["test_verdict"]["status"] == "failed"
    assert result["test_verdict"]["stage"] == "syntax"
    assert "users.py: expected ':'" in result["test_verdict"]["reason"]
    assert [stage["name"] for stage in result["test_result"]["stages"]] == ["syntax"]
    assert result["test_result"]["cases"] == {}


def test_a_patch_that_does_not_apply_is_not_tested(repo, workspaces):
    patch = FIX_PATCH.read_text(encoding="utf-8").replace("def get_user", "def fetch_user")

    result = evaluate_patch(repo, patch, workspaces)

    assert result["patch_status"] == "PATCH_APPLY_FAILED"
    assert result["test_result"]["status"] == "SKIPPED"
    assert result["test_verdict"]["status"] == "failed"
    assert result["test_verdict"]["reason"].startswith("The patch was not tested")
    assert result["baseline_result"]["cases"]  # the baseline still ran


def test_a_supplied_baseline_is_reused(repo, workspaces):
    # Pretend every test passed before: the real fix then shows no improvement.
    baseline = run_baseline(repo, workspaces)
    baseline["cases"] = {test_id: "passed" for test_id in baseline["cases"]}

    result = evaluate_patch(repo, FIX_PATCH.read_text(encoding="utf-8"), workspaces, baseline=baseline)

    assert result["baseline_result"] is baseline
    assert result["test_verdict"]["status"] == "unverified"


def test_a_hanging_suite_times_out(tmp_path, monkeypatch):
    repo = tmp_path / "slow_repo"
    repo.mkdir()
    (repo / "test_slow.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8"
    )
    monkeypatch.setattr(tester, "TEST_TIMEOUT_SECONDS", 2)

    result = run_suite(repo)

    assert result["status"] == "FAIL"
    assert "timed out" in result["errors"][0]
    assert result["cases"] == {}
