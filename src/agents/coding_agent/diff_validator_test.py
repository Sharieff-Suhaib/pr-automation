"""Check the diff validator against the malformed patches models actually emit.

Run with:  python -m src.agents.coding_agent.diff_validator_test
"""

from __future__ import annotations

from src.agents.coding_agent.diff_validator import (
    DiffError,
    describe_diff_error,
    extract_diff,
    normalize_diff,
    parse_unified_diff,
    validate_unified_diff,
)


VALID = """--- a/users.py
+++ b/users.py
@@ -1,2 +1,4 @@
 def get_user(users, name):
+    if name not in users:
+        return None
     return users[name]
"""

# The patch from the failing run: descriptions instead of paths, unprefixed
# body lines, and @@ counts that do not match the body.
MODEL_PSEUDO_DIFF = """--- Auth.java (lines 3-6, method login)
+++ Auth.java (lines 3-6, method login)
@@ -3,7 +3,8 @@ public static boolean login(String username, String password) {
        return username.equals("admin") &&
               password.equals("password123");
    }

--- auth.py (lines 1-2, function login)
+++ auth.py (lines 1-2, function login)
@@ -1,4 +1,5 @@ def login(username, password):
    return username == "admin" and password == "password123"
"""


def _expect_error(patch: str, fragment: str, label: str) -> None:
    try:
        validate_unified_diff(patch)
    except DiffError as error:
        assert fragment in str(error), f"{label}: got {error!r}, wanted {fragment!r}"
        print(f"  rejected {label}: {error}")
        return
    raise AssertionError(f"{label}: expected a DiffError, but the patch was accepted")


def check_valid_patch() -> None:
    files = validate_unified_diff(VALID)
    assert len(files) == 1
    assert files[0].path == "b/users.py"
    assert files[0].changes == 2
    assert files[0].hunks[0].added == 2
    assert files[0].hunks[0].removed == 0
    print("  accepted a well-formed patch")


def check_the_failing_patch() -> None:
    _expect_error(MODEL_PSEUDO_DIFF, "must be a path", "the run's pseudo-diff")


def check_header_problems() -> None:
    _expect_error(
        "--- a/x.py\n@@ -1 +1 @@\n-a\n+b\n",
        "must be followed by '+++",
        "a '---' with no '+++'",
    )
    _expect_error("@@ -1 +1 @@\n-a\n+b\n", "before any", "a hunk with no file header")
    _expect_error("--- a/x.py\n+++ b/x.py\n", "No @@ hunk", "headers with no hunk")
    _expect_error("", "empty", "an empty patch")
    _expect_error("Here is the fix.\n", "found prose", "prose only")


def check_miscounted_hunks_are_recovered() -> None:
    """Git treats @@ counts as hints and recovers from the body; so do we.

    Rejecting a miscounted header would fail patches that Git applies -- and
    small models miscount constantly while producing a good body.
    """
    understated = "--- a/x.py\n+++ b/x.py\n@@ -1,5 +1,6 @@\n a\n+b\n"
    hunk = validate_unified_diff(understated)[0].hunks[0]
    assert (hunk.old_count, hunk.new_count) == (1, 2), hunk.header
    assert hunk.counts_corrected is True
    assert hunk.header == "@@ -1,1 +1,2 @@"

    overstated = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n a\n b\n"
    hunk = validate_unified_diff(overstated, allow_empty_hunks=True)[0].hunks[0]
    assert (hunk.old_count, hunk.new_count) == (2, 2), hunk.header
    print("  recovered the true counts from two miscounted headers")


def check_hunk_problems() -> None:
    _expect_error(
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n a\nb\n",
        "must start with",
        "an unprefixed body line",
    )
    _expect_error(
        "--- a/x.py\n+++ b/x.py\n@@ x @@\n a\n",
        "hunk header must look like",
        "a malformed @@ header",
    )
    _expect_error(
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n",
        "followed by no content",
        "a hunk header with an empty body",
    )


def check_no_op_patch() -> None:
    """A patch that applies cleanly and changes nothing is worse than a reject."""
    _expect_error(
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n a\n b\n",
        "change nothing",
        "a context-only patch",
    )
    # ...unless the caller explicitly wants to allow it.
    validate_unified_diff(
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n a\n b\n", allow_empty_hunks=True
    )
    print("  allow_empty_hunks accepts a context-only patch when asked")


def check_accepted_forms() -> None:
    """Everything Git writes must parse."""
    git_style = (
        "diff --git a/x.py b/x.py\n"
        "index 83db48f..bf269f4 100644\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    assert validate_unified_diff(git_style)[0].path == "b/x.py"

    new_file = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+print('hi')\n"
    assert validate_unified_diff(new_file)[0].path == "b/new.py"

    no_newline = (
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n"
    )
    assert validate_unified_diff(no_newline)[0].changes == 2

    single_line_counts = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert validate_unified_diff(single_line_counts)[0].changes == 2

    multi_file = VALID + "--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-c\n+d\n"
    assert len(validate_unified_diff(multi_file)) == 2
    print("  accepted git metadata, /dev/null, no-newline and multi-file diffs")


def check_normalization() -> None:
    """The two mistakes that are safe to fix without another model request."""
    decorated = (
        "--- stats.py (original)\n"
        "+++ stats.py (patched)\n"
        "@@ -1,2 +1,3 @@\n"
        " def average(values):\n"
        "-    return sum(values) / len(values)\n"
        "+    if not values:\n"
        "+        return None\n"
        "+    return sum(values) / len(values)\n"
    )
    # As written, this is not a valid diff.
    _expect_error(decorated, "must be a path", "a decorated header")

    fixed = normalize_diff(decorated)
    files = validate_unified_diff(fixed)
    assert files[0].old_path == "a/stats.py"
    assert files[0].new_path == "b/stats.py"
    assert "@@ -1,2 +1,4 @@" in fixed, fixed

    # The changed lines must survive untouched: normalisation rewrites headers
    # only, so it can never alter what the patch does.
    original_body = [
        line for line in decorated.splitlines() if line[:1] in {" ", "+", "-"}
        and not line.startswith(("--- ", "+++ "))
    ]
    fixed_body = [
        line for line in fixed.splitlines() if line[:1] in {" ", "+", "-"}
        and not line.startswith(("--- ", "+++ "))
    ]
    assert original_body == fixed_body
    print("  normalised decorated headers and miscounted hunks")

    # A patch that is already correct must come back unchanged.
    assert normalize_diff(VALID).strip() == VALID.strip()

    # Something unparseable is passed through for validation to explain.
    assert normalize_diff("not a diff").strip() == "not a diff"
    print("  left a correct patch and an unparseable one alone")


def check_prose_is_trimmed() -> None:
    wrapped = "Here is the fix:\n\n" + VALID + "\nThat resolves the issue.\n"
    assert validate_unified_diff(extract_diff(wrapped))[0].changes == 2
    assert extract_diff("no diff here") == "no diff here"
    print("  trimmed prose from around a diff")


def check_error_reporting() -> None:
    try:
        validate_unified_diff(MODEL_PSEUDO_DIFF)
    except DiffError as error:
        report = describe_diff_error(error, MODEL_PSEUDO_DIFF)
        assert "line 1" in report
        assert ">>" in report and "Auth.java" in report
        print("  reported the offending line with context")


def main() -> int:
    print("Diff validator")
    for check in (
        check_valid_patch,
        check_the_failing_patch,
        check_header_problems,
        check_miscounted_hunks_are_recovered,
        check_hunk_problems,
        check_normalization,
        check_no_op_patch,
        check_accepted_forms,
        check_prose_is_trimmed,
        check_error_reporting,
    ):
        check()
    print("All diff validator checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
