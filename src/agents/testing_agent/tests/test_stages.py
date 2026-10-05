"""Staged runs: syntax -> reproduction -> related -> full, stopping at the first failure.

Runs on a copy of `orchestrator/sample_repo` with an extra, unrelated test file
(`test_other.py`), which only the full stage runs.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import shutil

import pytest

from src.agents.testing_agent import evaluate_patch, run_baseline
from src.agents.testing_agent.selection import id_prefix, related_tests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_REPO = PROJECT_ROOT / "orchestrator" / "sample_repo"
FIX_PATCH = (PROJECT_ROOT / "orchestrator" / "fixtures" / "sample_repo_fix.patch").read_text(encoding="utf-8")

REPRO_FILE = "test_issue_reproduction.py"
REPRO_SOURCE = 'from users import get_user\n\n\ndef test_issue_none():\n    assert get_user({}, "ghost") is None\n'
REPRO_TESTS = ["test_issue_reproduction::test_issue_none"]
OTHER_TEST = "test_other::test_unrelated"


@pytest.fixture
def repo(tmp_path):
    destination = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, destination)
    (destination / "test_other.py").write_text("def test_unrelated():\n    assert True\n", encoding="utf-8")
    return destination


def edit_patch(repo: Path, *replacements: tuple[str, str]) -> str:
    original = (repo / "users.py").read_text(encoding="utf-8")
    modified = original
    for old, new in replacements:
        modified = modified.replace(old, new)
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="a/users.py",
            tofile="b/users.py",
        )
    )


def evaluate(repo: Path, tmp_path: Path, patch: str, with_reproduction: bool = True):
    extra = {REPRO_FILE: REPRO_SOURCE} if with_reproduction else None
    baseline = run_baseline(repo, tmp_path / "workspaces", extra_files=extra)
    return evaluate_patch(
        repo,
        patch,
        tmp_path / "workspaces",
        baseline=baseline,
        extra_files=extra,
        reproduction_tests=REPRO_TESTS if with_reproduction else None,
    )


def stage_names(result) -> list[str]:
    return [stage["name"] for stage in result["test_result"]["stages"]]


def test_a_good_patch_goes_through_every_stage(repo, tmp_path):
    result = evaluate(repo, tmp_path, FIX_PATCH)

    assert stage_names(result) == ["syntax", "reproduction", "related", "full"]
    assert result["test_result"]["stages"][2]["files"] == ["test_users.py"]
    assert result["test_verdict"]["status"] == "solved"
    assert result["test_verdict"]["stage"] == "full"
    assert OTHER_TEST in result["test_result"]["cases"]


def test_a_still_failing_reproduction_test_stops_before_the_suite(repo, tmp_path):
    patch = edit_patch(repo, ("with a reporting bug", "with a known reporting bug"))

    result = evaluate(repo, tmp_path, patch)

    assert stage_names(result) == ["syntax", "reproduction"]
    assert result["test_verdict"]["stage"] == "reproduction"
    assert result["test_verdict"]["status"] == "unverified"
    assert result["test_verdict"]["reason"].startswith("The issue's reproduction test still fails")
    assert set(result["test_result"]["cases"]) == set(REPRO_TESTS)
    # Tests that were not run are not counted as broken.
    assert result["test_verdict"]["pass_to_fail"] == []


def test_a_regression_is_caught_by_the_related_tests(repo, tmp_path):
    # Fixes get_user (so the reproduction test passes) but breaks login.
    patch = edit_patch(
        repo,
        ("    return users[name]\n", "    return users.get(name)\n"),
        ('    user = get_user(users, name)\n    return user["password"] == password\n', "    return False\n"),
    )

    result = evaluate(repo, tmp_path, patch)

    assert stage_names(result) == ["syntax", "reproduction", "related"]
    assert result["test_verdict"]["status"] == "failed"
    assert result["test_verdict"]["stage"] == "related"
    assert result["test_verdict"]["pass_to_fail"] == ["test_users::test_login_succeeds_for_valid_credentials"]
    assert OTHER_TEST not in result["test_result"]["cases"]  # the full suite never ran


def test_without_a_reproduction_test_the_stage_is_skipped(repo, tmp_path):
    result = evaluate(repo, tmp_path, FIX_PATCH, with_reproduction=False)

    assert stage_names(result) == ["syntax", "related", "full"]
    assert result["test_verdict"]["status"] == "solved"


def write(root: Path, relative: str, text: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_related_tests_are_found_by_name_and_by_import(tmp_path):
    write(tmp_path, "src/shop/cart.py")
    write(tmp_path, "tests/test_cart.py")  # named after it
    write(tmp_path, "tests/test_checkout.py", "from shop.cart import Cart\n")  # src layout import
    write(tmp_path, "tests/test_api.py", "from shop import cart, prices\n")  # parent import
    write(tmp_path, "tests/test_dotted.py", "import shop.cart as c\n")
    write(tmp_path, "tests/test_unrelated.py", "from shop.prices import x  # mentions cart\n")
    write(tmp_path, ".venv/lib/test_cart.py")  # ignored directory

    found = related_tests(tmp_path, ["src/shop/cart.py"])

    assert found == ["tests/test_api.py", "tests/test_cart.py", "tests/test_checkout.py", "tests/test_dotted.py"]


def test_a_changed_test_file_is_related_to_itself_and_excludes_apply(tmp_path):
    write(tmp_path, "tests/test_cart.py")
    write(tmp_path, "test_issue_reproduction.py", "import cart\n")
    write(tmp_path, "cart.py")

    assert related_tests(tmp_path, ["tests/test_cart.py"]) == ["tests/test_cart.py"]
    # The reproduction file imports cart too, but is excluded.
    assert related_tests(tmp_path, ["cart.py"], exclude={"test_issue_reproduction.py"}) == ["tests/test_cart.py"]


def test_id_prefix_matches_pytest_test_ids():
    assert id_prefix("test_users.py") == "test_users"
    assert id_prefix("tests/unit/test_users.py") == "tests.unit.test_users"
