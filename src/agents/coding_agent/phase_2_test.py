"""Run a local Phase 2 safety check without changing this project."""

from __future__ import annotations

from pathlib import Path
import tempfile

from src.agents.coding_agent.patch_manager import apply_patch


PATCH = """--- a/users.py
+++ b/users.py
@@ -1,2 +1,4 @@
 def get_user(users, name):
+    if name not in users:
+        return None
     return users[name]
"""


def main() -> int:
    """Copy a tiny repository, apply a patch, and confirm the source is intact."""
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        original_repo = temporary_root / "original_repo"
        working_repo = temporary_root / "working_repo"
        original_repo.mkdir()

        original_code = "def get_user(users, name):\n    return users[name]\n"
        (original_repo / "users.py").write_text(original_code, encoding="utf-8")

        result = apply_patch(original_repo, working_repo, PATCH)
        print(result.to_dict())

        if result.status != "APPLIED":
            return 1
        if (original_repo / "users.py").read_text(encoding="utf-8") != original_code:
            print("Original repository was changed.")
            return 1
        if "if name not in users" not in (working_repo / "users.py").read_text(encoding="utf-8"):
            print("Patch was not applied to the working repository.")
            return 1

    print("Phase 2 safety check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
