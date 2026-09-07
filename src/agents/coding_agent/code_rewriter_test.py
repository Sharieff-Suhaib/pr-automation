"""Check the rewrite strategy without calling a model.

The point of rewriting is that the diff is computed, not written, so these
checks build diffs from fixed "model answers" and confirm Git accepts them.

Run with:  python -m src.agents.coding_agent.code_rewriter_test
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from src.agents.coding_agent.code_rewriter import (
    RewriteError,
    TargetChunk,
    align_indentation,
    build_rewrite_prompt,
    clean_code_block,
    diff_from_replacement,
    select_target,
    select_targets,
)
from src.agents.coding_agent.diff_validator import validate_unified_diff


AUTH_JAVA = """public class Auth {

    public static boolean login(String username, String password) {
        return username.equals("admin") &&
               password.equals("password123");
    }

    public static void logout() {
        System.out.println("User logged out");
    }
}
"""

LOGIN_CHUNK = TargetChunk(
    file="Auth.java",
    start_line=3,
    end_line=6,
    code="""    public static boolean login(String username, String password) {
        return username.equals("admin") &&
               password.equals("password123");
    }""",
    name="login",
    kind="method",
    language="java",
)

FIXED_LOGIN = """    public static boolean login(String username, String password) {
        if (username == null || username.isEmpty()) {
            return false;
        }
        return username.equals("admin") &&
               password.equals("password123");
    }"""


def _repo(files: dict[str, str]) -> Path:
    directory = Path(tempfile.mkdtemp())
    subprocess.run(["git", "init", "-q", str(directory)], check=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


def _git_accepts(repo: Path, patch: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repo,
        input=patch.encode("utf-8"),
        capture_output=True,
    )
    return (result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))


# --------------------------------------------------------------------------
def check_diff_is_accepted_by_git() -> None:
    """The whole point: a computed diff applies, because context is read."""
    repo = _repo({"Auth.java": AUTH_JAVA})
    patch = diff_from_replacement(repo, LOGIN_CHUNK, FIXED_LOGIN)

    validate_unified_diff(patch)
    accepted, error = _git_accepts(repo, patch)
    assert accepted, f"git rejected the computed diff:\n{error}\n{patch}"
    print("  git accepted a computed diff")


def check_the_patch_actually_changes_the_file() -> None:
    repo = _repo({"Auth.java": AUTH_JAVA})
    patch = diff_from_replacement(repo, LOGIN_CHUNK, FIXED_LOGIN)
    result = subprocess.run(
        ["git", "apply", "-"], cwd=repo, input=patch.encode("utf-8"), capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()

    patched = (repo / "Auth.java").read_text(encoding="utf-8")
    assert "username.isEmpty()" in patched
    assert "public static void logout()" in patched  # untouched code survives
    assert patched.count("public static boolean login") == 1
    print("  applying it produces the intended file")


def check_context_comes_from_the_file() -> None:
    """Context lines must be the file's, not anything the model supplied."""
    repo = _repo({"Auth.java": AUTH_JAVA})
    patch = diff_from_replacement(repo, LOGIN_CHUNK, FIXED_LOGIN)

    file_lines = set(AUTH_JAVA.splitlines())
    for line in patch.splitlines():
        if line.startswith(" ") and line.strip():
            assert line[1:] in file_lines, f"context line not from the file: {line!r}"
    print("  every context line came from the file on disk")


def check_hunk_counts_are_correct() -> None:
    repo = _repo({"Auth.java": AUTH_JAVA})
    patch = diff_from_replacement(repo, LOGIN_CHUNK, FIXED_LOGIN)
    hunk = validate_unified_diff(patch)[0].hunks[0]
    assert hunk.counts_corrected is False, "difflib should not need correcting"
    print("  the @@ counts were right the first time")


def check_indentation_is_restored() -> None:
    """A model that answers flush-left must not corrupt a nested method."""
    dedented = "\n".join(line[4:] if line.startswith("    ") else line
                         for line in FIXED_LOGIN.split("\n"))
    assert not dedented.startswith("    ")

    aligned = align_indentation(LOGIN_CHUNK.code, dedented)
    assert aligned.startswith("    public static boolean login")

    repo = _repo({"Auth.java": AUTH_JAVA})
    patch = diff_from_replacement(repo, LOGIN_CHUNK, aligned)
    accepted, error = _git_accepts(repo, patch)
    assert accepted, error
    print("  re-indented a flush-left answer")


def check_indentation_comes_from_the_file_not_the_chunk() -> None:
    """The case the live run exposed.

    Tree-sitter returns a node's text without the leading whitespace of its
    first line, so the retrieved chunk starts flush-left even though the file
    has it indented four spaces. Aligning against the chunk would therefore do
    nothing and splice a flush-left method into a class body.
    """
    chunk_without_indent = TargetChunk(
        file="Auth.java",
        start_line=3,
        end_line=6,
        code=LOGIN_CHUNK.code.replace("\n    ", "\n").lstrip(),  # as the parser gives it
        name="login",
        kind="method",
        language="java",
    )
    assert not chunk_without_indent.code.startswith(" ")

    model_answer = FIXED_LOGIN.replace("\n    ", "\n").lstrip()  # also flush-left
    repo = _repo({"Auth.java": AUTH_JAVA})
    patch = diff_from_replacement(repo, chunk_without_indent, model_answer)

    added = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]
    assert added, patch
    assert added[0].startswith("    "), f"indentation was lost: {added[0]!r}"

    accepted, error = _git_accepts(repo, patch)
    assert accepted, error
    print("  took the indentation from the file, not the chunk")


def check_correct_indentation_is_left_alone() -> None:
    assert align_indentation(LOGIN_CHUNK.code, FIXED_LOGIN) == FIXED_LOGIN
    print("  left correctly indented output untouched")


def check_no_change_is_rejected() -> None:
    """An unchanged rewrite must not become an empty patch that 'succeeds'."""
    repo = _repo({"Auth.java": AUTH_JAVA})
    try:
        diff_from_replacement(repo, LOGIN_CHUNK, LOGIN_CHUNK.code)
    except RewriteError as error:
        assert "identical" in str(error)
        print("  rejected a rewrite that changed nothing")
        return
    raise AssertionError("an unchanged rewrite should raise")


def check_bad_line_range_is_rejected() -> None:
    repo = _repo({"Auth.java": AUTH_JAVA})
    beyond = TargetChunk(file="Auth.java", start_line=50, end_line=60, code="x")
    try:
        diff_from_replacement(repo, beyond, "y")
    except RewriteError as error:
        assert "cannot be replaced" in str(error)
        print("  rejected a line range past the end of the file")
        return
    raise AssertionError("an impossible range should raise")


def check_missing_file_is_rejected() -> None:
    repo = _repo({"Auth.java": AUTH_JAVA})
    absent = TargetChunk(file="Nope.java", start_line=1, end_line=1, code="x")
    try:
        diff_from_replacement(repo, absent, "y")
    except RewriteError as error:
        assert "does not exist" in str(error)
        print("  rejected a file that is not in the repository")
        return
    raise AssertionError("a missing file should raise")


def check_answer_cleaning() -> None:
    fenced = "Here is the corrected code:\n```java\n    int x = 1;\n```\n"
    assert clean_code_block(fenced) == "    int x = 1;"

    labelled = "### FIXED CODE\n    int x = 1;\n"
    assert clean_code_block(labelled) == "    int x = 1;"

    plain = "    int x = 1;\n"
    assert clean_code_block(plain) == "    int x = 1;"

    # A comment is code, not prose, and must survive.
    commented = "// guard the empty case\nif (x == null) return;"
    assert clean_code_block(commented).startswith("// guard")

    try:
        clean_code_block("   ")
    except RewriteError:
        pass
    else:
        raise AssertionError("empty output should raise")
    print("  cleaned fences, labels and leading prose")


def check_target_selection() -> None:
    chunks = [
        {"file": "auth.py", "language": "python", "start_line": 1, "end_line": 2, "code": "a"},
        {"file": "Auth.java", "language": "java", "start_line": 3, "end_line": 6, "code": "b"},
    ]
    assert select_target(chunks, "java").file == "Auth.java"
    assert select_target(chunks, "").file == "auth.py"  # best match wins
    assert select_target(chunks, "rust").file == "auth.py"  # no match: don't drop everything
    print("  selected the target in the requested language")


def check_multi_file_patch_applies() -> None:
    """A polyglot repo gets one diff per implementation, in one patch."""
    auth_py = 'def login(username, password):\n    return username == "admin"\n'
    repo = _repo({"Auth.java": AUTH_JAVA, "auth.py": auth_py})

    java = diff_from_replacement(repo, LOGIN_CHUNK, FIXED_LOGIN)
    python_chunk = TargetChunk(
        file="auth.py", start_line=1, end_line=2, code=auth_py.rstrip(), language="python"
    )
    python = diff_from_replacement(
        repo,
        python_chunk,
        'def login(username, password):\n    if not username:\n        return False\n'
        '    return username == "admin"',
    )

    combined = java + python
    files = validate_unified_diff(combined)
    assert [f.path for f in files] == ["b/Auth.java", "b/auth.py"], [f.path for f in files]

    accepted, error = _git_accepts(repo, combined)
    assert accepted, f"git rejected the combined patch:\n{error}\n{combined}"

    result = subprocess.run(
        ["git", "apply", "-"], cwd=repo, input=combined.encode("utf-8"), capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()
    assert "isEmpty()" in (repo / "Auth.java").read_text(encoding="utf-8")
    assert "if not username" in (repo / "auth.py").read_text(encoding="utf-8")
    print("  a combined multi-file patch applies to every file")


def check_one_target_per_file() -> None:
    """Two hunks in one file would be computed against stale line numbers."""
    chunks = [
        {"file": "Auth.java", "language": "java", "start_line": 3, "end_line": 6, "code": "a"},
        {"file": "Auth.java", "language": "java", "start_line": 8, "end_line": 10, "code": "b"},
        {"file": "auth.py", "language": "python", "start_line": 1, "end_line": 2, "code": "c"},
    ]
    targets = select_targets(chunks, "java", max_targets=9)
    assert [t.file for t in targets] == ["Auth.java", "auth.py"]
    assert targets[0].start_line == 3  # the better-ranked hunk wins
    print("  selected at most one region per file")


def check_target_ordering_and_cap() -> None:
    chunks = [
        {"file": "auth.py", "language": "python", "start_line": 1, "end_line": 2, "code": "a"},
        {"file": "auth.cpp", "language": "cpp", "start_line": 1, "end_line": 4, "code": "b"},
        {"file": "Auth.java", "language": "java", "start_line": 3, "end_line": 6, "code": "c"},
    ]
    # The chosen language goes first, but the others are still repaired.
    assert [t.file for t in select_targets(chunks, "java", max_targets=9)] == [
        "Auth.java", "auth.py", "auth.cpp",
    ]
    assert len(select_targets(chunks, "java", max_targets=2)) == 2
    assert [t.file for t in select_targets(chunks, "", max_targets=1)] == ["auth.py"]
    print("  ordered by language and capped by max_targets")


def check_prompt_shape() -> None:
    prompt = build_rewrite_prompt(
        "login accepts an empty username", LOGIN_CHUNK, strategy="Validate the input"
    )
    assert "Auth.java lines 3-6" in prompt
    assert "no diff" in prompt
    assert "Validate the input" in prompt
    assert LOGIN_CHUNK.code in prompt
    print("  the prompt shows the code and forbids diff syntax")


def check_multiline_and_trailing_newline() -> None:
    """A file with no trailing newline must stay that way."""
    repo = _repo({"x.py": "def f():\n    return 1"})
    chunk = TargetChunk(file="x.py", start_line=2, end_line=2, code="    return 1")
    patch = diff_from_replacement(repo, chunk, "    return 2")
    accepted, error = _git_accepts(repo, patch)
    assert accepted, error
    print("  handled a file with no trailing newline")


def main() -> int:
    print("Rewrite strategy")
    for check in (
        check_diff_is_accepted_by_git,
        check_the_patch_actually_changes_the_file,
        check_context_comes_from_the_file,
        check_hunk_counts_are_correct,
        check_indentation_is_restored,
        check_indentation_comes_from_the_file_not_the_chunk,
        check_correct_indentation_is_left_alone,
        check_no_change_is_rejected,
        check_bad_line_range_is_rejected,
        check_missing_file_is_rejected,
        check_answer_cleaning,
        check_target_selection,
        check_multi_file_patch_applies,
        check_one_target_per_file,
        check_target_ordering_and_cap,
        check_prompt_shape,
        check_multiline_and_trailing_newline,
    ):
        check()
    print("All rewrite checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
