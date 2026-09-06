"""Run a local Phase 3 pytest check without changing this project."""

from __future__ import annotations

from pathlib import Path
import tempfile

from src.agents.coding_agent.tester import (
    detect_framework,
    run_tests,
    select_test_command,
)


def main() -> int:
    """Create a tiny working repository and verify that pytest passes in it."""
    with tempfile.TemporaryDirectory() as directory:
        working_repo = Path(directory) / "working_repo"
        working_repo.mkdir()

        (working_repo / "users.py").write_text(
            "def get_user(users, name):\n"
            "    if name not in users:\n"
            "        return None\n"
            "    return users[name]\n",
            encoding="utf-8",
        )
        (working_repo / "test_users.py").write_text(
            "from users import get_user\n\n"
            "def test_missing_user_returns_none():\n"
            "    assert get_user({}, 'missing') is None\n",
            encoding="utf-8",
        )

        result = run_tests(working_repo)
        print(result.to_dict())
        if result.status != "PASS":
            return 1

        javascript_repo = Path(directory) / "javascript_repo"
        javascript_repo.mkdir()
        (javascript_repo / "package.json").write_text("{}", encoding="utf-8")
        assert detect_framework(javascript_repo) == "javascript"
        assert select_test_command(javascript_repo) == ["npm", "test"]

    print("Python execution and JavaScript command selection passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
