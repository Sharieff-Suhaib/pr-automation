"""The remote fine-tuned model (IR4 in, OR2 out) and the push after an approved review.

No network: the endpoint is faked, and the push goes to a local bare repository.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from orchestrator import adapters

SAMPLE_REPO = Path(__file__).resolve().parents[1] / "sample_repo"
GET_USER = {"file": "users.py", "name": "get_user", "type": "function", "language": "python",
            "start_line": 4, "end_line": 6, "code": ""}


def test_the_remote_reply_replaces_the_function_body(tmp_path, monkeypatch):
    shutil.copytree(SAMPLE_REPO, tmp_path / "repo")
    sent = []

    def fake_urlopen(request, timeout):
        sent.append({**json.loads(request.data), "url": request.full_url})
        # OR2 for the one commented-out line; the first line lost its indentation.
        reply = 'if name not in users:\n        return None\n    return users[name]\n</s>'
        return io.BytesIO(json.dumps({"response": reply}).encode())

    monkeypatch.setattr(adapters.urllib.request, "urlopen", fake_urlopen)

    patch = adapters.remote_patch(str(tmp_path / "repo"), [GET_USER])

    assert sent[0]["url"].endswith("/repair")
    # The IR4 input exactly as in the BugsInPy training data; OR2 comes back.
    assert sent[0]["prompt"] == (
        'def get_user(users, name):\n    """Return the profile stored for `name`."""\n'
        "# BUGGY CODE\n#     return users[name]\n<FILL_ME>"
    )
    assert "+    if name not in users:\n+        return None\n" in patch
    subprocess.run(["git", "apply", "--check", "-"], cwd=tmp_path / "repo", input=patch, text=True, check=True)


FIND_ORDER = """public class OrderService {
    public Order findOrder(int orderId) {
        for (int i = 0; i < orders.size() - 1; i++) {
            if (orders.get(i).getOrderId() == orderId) {
                return orders.get(i);
            }
        }

        return null;
    }
}
"""


def test_the_remote_model_gets_one_commented_line_and_its_reply_replaces_it(tmp_path, monkeypatch):
    (tmp_path / "OrderService.java").write_text(FIND_ORDER, encoding="utf-8")
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data)["prompt"])
        # The reply the Colab endpoint gave for this prompt, </s> included.
        return io.BytesIO(json.dumps({"repair": "for (int i = 0; i < orders.size(); i++) {\n</s>"}).encode())

    monkeypatch.setattr(adapters.urllib.request, "urlopen", fake_urlopen)
    chunk = {"file": "OrderService.java", "name": "findOrder", "type": "method", "language": "java",
             "start_line": 2, "end_line": 10, "code": ""}

    patch = adapters.remote_patch(str(tmp_path), [chunk], issue="the last order is never found")

    assert sent == [
        "Repair the bug in this Java code: the last order is never found.\nReturn only the missing code.\n\n"
        "    public Order findOrder(int orderId) {\n"
        "// buggy code\n"
        "//         for (int i = 0; i < orders.size() - 1; i++) {\n"
        "        <FILL_ME>\n"
        "            if (orders.get(i).getOrderId() == orderId) {\n"
        "                return orders.get(i);\n"
        "            }\n"
        "        }\n"
        "\n"
        "        return null;\n"
        "    }"
    ]
    assert "-        for (int i = 0; i < orders.size() - 1; i++) {\n" \
           "+        for (int i = 0; i < orders.size(); i++) {\n" in patch
    subprocess.run(["git", "apply", "--check", "-"], cwd=tmp_path, input=patch, text=True, check=True)


STUDENT_SERVICE = """public class StudentService {
    public Student getStudentById(int id) {
        for (int i = 0; i < students.size() - 1; i++) {
            if (students.get(i).getId() == id) {
                return students.get(i);
            }
        }
        return null;
    }

    public double calculateAverage() {
        double total = 0;

        for (Student student : students) {
            total += student.getMarks();
        }

        return total / students.size();
    }

    public List<Student> getTopStudents(double cutoff) {
        List<Student> result = new ArrayList<>();

        for (Student student : students) {
            if (student.getMarks() > cutoff) {
                result.add(student);
            }
        }

        Collections.sort(result, (a, b) ->
            (int) (b.getMarks() - a.getMarks())
        );

        return result;
    }
}""".splitlines()

# The dummy_repo issue: four symptoms, and "student"/"marks" on nearly every line.
STUDENT_ISSUE = """Bug: Student Service Returns Incorrect Results
Several methods in StudentService produce incorrect results due to boundary conditions, missing validation,
and incorrect sorting logic. Add three students with IDs 101, 102, and 103. Attempt to retrieve or delete the
student with ID 103. Calculate the average when the student list is empty. Add students with marks 85.5 and
85.9 and retrieve the top students. The last student is skipped during lookup. Average calculation divides by
zero when no students exist. The sorting comparator can treat students with different marks as equal."""


@pytest.mark.parametrize(
    ("method", "buggy_line"),
    [
        ("getStudentById", "for (int i = 0; i < students.size() - 1; i++) {"),
        ("calculateAverage", "return total / students.size();"),
        ("getTopStudents", "(int) (b.getMarks() - a.getMarks())"),
    ],
)
def test_fault_localization_puts_the_buggy_line_first(method, buggy_line):
    start = next(i for i, line in enumerate(STUDENT_SERVICE) if f" {method}(" in line) + 1
    end = next(i for i in range(start, len(STUDENT_SERVICE)) if STUDENT_SERVICE[i] == "    }")

    ranked = adapters._suspicious_lines(STUDENT_SERVICE, start, end, STUDENT_ISSUE)

    assert STUDENT_SERVICE[ranked[0]].strip() == buggy_line


def test_a_reported_division_by_zero_tries_a_guard_at_the_top_of_the_body_first(tmp_path, monkeypatch):
    (tmp_path / "Student.java").write_text("\n".join(STUDENT_SERVICE) + "\n", encoding="utf-8")
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data)["prompt"])
        # The reply the Colab endpoint gave for this prompt.
        return io.BytesIO(json.dumps({"repair": "if (students.size() == 0) {\n            return 0;\n        }\n</s>"}).encode())

    monkeypatch.setattr(adapters.urllib.request, "urlopen", fake_urlopen)
    start = next(i for i, line in enumerate(STUDENT_SERVICE, 1) if " calculateAverage(" in line)
    chunk = {"file": "Student.java", "name": "calculateAverage", "type": "method", "language": "java",
             "start_line": start, "end_line": start + 8, "code": ""}

    patch = adapters.remote_patch(str(tmp_path), [chunk], issue=STUDENT_ISSUE)

    assert sent[0] == (
        "Repair the bug in this Java code.\nReturn only the missing code.\n\n"
        "    public double calculateAverage() {\n"
        "        <FILL_ME>\n"
        "        double total = 0;\n"
        "\n"
        "        for (Student student : students) {\n"
        "            total += student.getMarks();\n"
        "        }\n"
        "\n"
        "        return total / students.size();\n"
        "    }"
    )
    assert "     public double calculateAverage() {\n" \
           "+        if (students.size() == 0) {\n+            return 0;\n+        }\n" \
           "         double total = 0;\n" in patch
    subprocess.run(["git", "apply", "--check", "-"], cwd=tmp_path, input=patch, text=True, check=True)


def test_a_vague_validation_issue_on_a_dividing_method_tries_the_guard_first():
    # The issue from the UI run that never tried a guard.
    issue = ("Bug: The Average calculation has bug\n\n"
             "Add the validation for calculating the average for student service. It raises exception.")
    start = next(i for i, line in enumerate(STUDENT_SERVICE) if " calculateAverage(" in line)

    sites = adapters._repair_sites(STUDENT_SERVICE, start, start + 1, start + 8, issue)

    assert sites[0] == (start + 1, start + 1)  # before "double total = 0;"
    assert adapters._symptom_sentence(issue, "return total / students.size();") == (
        "Add the validation for calculating the average for student service."
    )


def test_a_wrong_line_still_comes_before_a_guard_for_invalid_input():
    start = next(i for i, line in enumerate(STUDENT_SERVICE) if " getStudentById(" in line)
    end = start + 7  # the body; the method's "}" is not part of it

    sites = adapters._repair_sites(STUDENT_SERVICE, start, start + 1, end, STUDENT_ISSUE)

    assert STUDENT_SERVICE[sites[0][0]].strip().startswith("for (int i = 0; i < students.size() - 1")
    assert (start + 1, start + 1) in sites[:3]  # the guard is still among the first tries


def test_code_quoted_in_the_issue_is_tried_first():
    # Issue #5 of dummy_repo: the loop and its condition outscore "return true"
    # on bug shapes, but the issue names that line.
    trie = [
        "    public boolean search(String word) {",
        "        TrieNode node = root;",
        "        for (char ch : word.toCharArray()) {",
        "            if (!node.children.containsKey(ch)) {",
        "                return false;",
        "            }",
        "            node = node.children.get(ch);",
        "        }",
        "        return true;",
        "    }",
    ]
    issue = ("Trie search bug\n\nIssue: search() returns True for a prefix that is not a complete word.\n"
             "- Bug location: return true at the end of search().")

    sites = adapters._repair_sites(trie, 0, 1, 9, issue)

    assert sites[0] == (8, 9)


TRIE = """public class BuggyTrie {
    private static class TrieNode {
        Map<Character, TrieNode> children = new HashMap<>();
        boolean isEnd = false;
    }

    public boolean search(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            if (!node.children.containsKey(ch)) {
                return false;
            }
            node = node.children.get(ch);
        }
        return true;
    }
}
"""
TRIE_ISSUE = "search() returns true for a prefix. Bug location: return true at the end of search()."
JAVAC_ISWORD = ("javac failed: BuggyTrie.java:15: error: cannot find symbol\n        return node.isWord;\n"
                "                   ^\n  symbol:   variable isWord\n  location: variable node of type TrieNode\n1 error")


def test_a_guessed_field_name_is_swapped_for_the_closest_declared_member():
    patch = "--- a/BuggyTrie.java\n+++ b/BuggyTrie.java\n@@ -15 +15 @@\n-        return true;\n+        return node.isWord;\n"

    # As the Java sandbox reports it: a one-line summary in the errors, the rest in the output.
    summary, details = JAVAC_ISWORD.split("\n", 1)
    variants = adapters._symbol_variants(patch, TRIE, {"errors": [summary], "output": details})

    # "word" (a parameter) is closer to "isWord", but node.word is no member: isEnd comes first.
    assert [(wrong, right) for wrong, right, _ in variants][0] == ("isWord", "isEnd")
    assert ("isWord", "word") not in [(wrong, right) for wrong, right, _ in variants]
    assert "+        return node.isEnd;\n" in variants[0][2] and "-        return true;\n" in variants[0][2]


def test_the_search_repairs_a_name_that_does_not_compile_and_goes_on(tmp_path, monkeypatch):
    (tmp_path / "BuggyTrie.java").write_text(TRIE, encoding="utf-8")
    monkeypatch.setattr(
        adapters.urllib.request, "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps({"repair": "return node.isWord;\n</s>"}).encode()),
    )
    tested = []

    def test(patch):
        tested.append(patch)
        if "isWord" in patch:  # what javac says about the model's reply
            return {"patch_status": "APPLIED", "test_result": {"passed": 0, "failed": 0, "errors": [JAVAC_ISWORD]},
                    "test_verdict": {"status": "failed", "stage": "compile", "reason": "does not compile",
                                     "pass_to_fail": [], "fail_to_pass": []}}
        solved = "+        return node.isEnd;" in patch
        return {"patch_status": "APPLIED", "test_result": {"passed": 1 + solved, "failed": 1 - solved, "errors": []},
                "test_verdict": {"status": "solved" if solved else "unverified", "stage": "full", "reason": "",
                                 "pass_to_fail": [], "fail_to_pass": ["t"] if solved else []}}

    chunk = {"file": "BuggyTrie.java", "name": "search", "type": "method", "language": "java",
             "start_line": 7, "end_line": 16, "code": ""}
    searched = adapters.search_remote_patch(str(tmp_path), [chunk], TRIE_ISSUE, test)

    assert [(row["where"], row["status"]) for row in searched["candidates"]] == [
        ("line 15", "failed"), ("line 15, isWord renamed to isEnd", "solved"),
    ]
    assert searched["best"]["status"] == "solved" and searched["next_site"] == 1


def test_the_prompt_names_the_symptom_of_the_chosen_line():
    line = "        return total / students.size();"

    # The sentence is wrapped across two lines in the issue, and an earlier one
    # ("Calculate the average when the student list is empty.") matches less.
    assert adapters._symptom_sentence(STUDENT_ISSUE, line) == "Average calculation divides by zero when no students exist."
    assert adapters._symptom_sentence("Something is wrong.", line) == ""
    prompt = adapters._ir4([line], 0, 0, 1, 1, "Student.java", "java", issue="Average divides by zero.")
    assert prompt.startswith("Repair the bug in this Java code: average divides by zero.\nReturn only the missing code.\n")


def test_the_function_mode_repairs_a_java_method(tmp_path, monkeypatch):
    from src.agents.coding_agent import function_patch

    (tmp_path / "Grades.java").write_text(
        "public class Grades {\n"
        "    public double average(List<Double> marks) {\n"
        "        double total = 0;\n"
        "        for (double m : marks) total += m;\n"
        "        return total / marks.size();\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    # Returned inside its class, as codellama often does: the method's "}" is its least-indented line.
    reply = (
        "```java\npublic class Grades {\n    public double average(List<Double> marks) {\n"
        "        if (marks.isEmpty()) {\n            return 0;\n        }\n"
        "        double total = 0;\n        for (double m : marks) total += m;\n"
        "        return total / marks.size();\n    }\n}\n```"
    )
    monkeypatch.setattr(function_patch, "_call_ollama", lambda prompt, **kwargs: reply)
    chunk = {"file": "Grades.java", "name": "average", "type": "method", "start_line": 2, "end_line": 6,
             "code": "public double average(List<Double> marks) {\n        double total = 0;\n    }"}

    patch = function_patch.generate_function_patch("average divides by zero", tmp_path, [chunk])

    assert "+        if (marks.isEmpty()) {\n+            return 0;\n+        }\n" in patch
    subprocess.run(["git", "apply", "--check", "-"], cwd=tmp_path, input=patch, text=True, check=True)

    # The same reply in the skewed shape the retrieved chunks have (signature at
    # column 0, body and "}" at the file's indentation), which the model copies.
    skewed = "public double average(List<Double> marks) {\n        if (marks.isEmpty()) {\n            return 0;\n" \
             "        }\n        double total = 0;\n        for (double m : marks) total += m;\n" \
             "        return total / marks.size();\n    }"
    monkeypatch.setattr(function_patch, "_call_ollama", lambda prompt, **kwargs: skewed)
    assert function_patch.generate_function_patch("average divides by zero", tmp_path, [chunk]) == patch


def test_an_approved_patch_is_pushed_to_a_new_branch(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "")
    origin, repo = tmp_path / "origin.git", tmp_path / "repo"
    git = lambda *args, cwd=repo: subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    shutil.copytree(SAMPLE_REPO, repo)
    git("init", "--bare", str(origin), cwd=tmp_path)
    git("init")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    git("remote", "add", "origin", str(origin))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    patch = (SAMPLE_REPO.parent / "fixtures" / "sample_repo_fix.patch").read_text(encoding="utf-8")

    pushed = adapters.commit_and_push(str(repo), patch, "get_user crashes on unknown users")

    assert "if name not in users" in git("show", f"{pushed['branch']}:users.py", cwd=origin).stdout
    assert git("status", "--porcelain").stdout == ""  # the clone itself is untouched
