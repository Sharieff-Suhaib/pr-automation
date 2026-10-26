"""The Java sandbox (javac + JUnit 5, no build file) and Java reproduction tests."""

from __future__ import annotations

import shutil

import pytest

from src.agents.testing_agent import evaluate_patch
from src.agents.testing_agent.java import run_suite
from src.agents.testing_agent.reproduction import clean_java_test, prepare_reproduction, stub_writer

pytestmark = pytest.mark.skipif(not shutil.which("javac"), reason="needs a JDK")

# A public class in a file of another name, as in loose repositories: javac
# would reject it in place, the sandbox renames it while staging.
SERVICE = """public class StudentService {
    private final java.util.List<Integer> ids = new java.util.ArrayList<>();

    public void add(int id) { ids.add(id); }

    public boolean contains(int id) {
        for (int i = 0; i < ids.size() - 1; i++) {
            if (ids.get(i) == id) return true;
        }
        return false;
    }
}
"""
REPRO = """public class AgentReproTest {
    @Test
    public void testIssueLastStudentIsFound() {
        StudentService service = new StudentService();
        service.add(101);
        service.add(103);
        assertTrue(service.contains(103));
    }
}
"""
FIX = """--- a/Student.java
+++ b/Student.java
@@ -4,7 +4,7 @@
     public void add(int id) { ids.add(id); }

     public boolean contains(int id) {
-        for (int i = 0; i < ids.size() - 1; i++) {
+        for (int i = 0; i < ids.size(); i++) {
             if (ids.get(i) == id) return true;
         }
         return false;
"""
CHUNK = {"file": "Student.java", "name": "contains", "type": "method", "language": "java",
         "start_line": 6, "end_line": 11, "code": ""}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "Student.java").write_text(SERVICE, encoding="utf-8")
    return root


def test_a_repository_without_tests_compiles_and_reports_no_tests(repo):
    result = run_suite(repo)

    assert result["no_tests"] and result["cases"] == {}


def test_a_compile_error_is_reported_not_hidden(repo):
    (repo / "Student.java").write_text(SERVICE.replace("return false;", "return false"), encoding="utf-8")

    result = run_suite(repo)

    assert result["status"] == "FAIL" and "javac failed" in result["errors"][0]
    assert "no_tests" not in result


def test_the_java_reproduction_is_accepted_and_the_fix_is_solved(repo, tmp_path):
    workspaces = tmp_path / "workspaces"
    writer = tmp_path / "repro.java"
    writer.write_text(REPRO, encoding="utf-8")

    reproduction, baseline = prepare_reproduction(
        repo, "contains() skips the last student", [CHUNK], workspaces, stub_writer(writer), language="java"
    )

    assert reproduction["status"] == "accepted", reproduction["reason"]
    assert reproduction["tests"] == ["AgentReproTest::testIssueLastStudentIsFound()"]
    result = evaluate_patch(
        repo, FIX, workspaces, language="java", baseline=baseline,
        extra_files={reproduction["path"]: reproduction["source"]}, reproduction_tests=reproduction["tests"],
    )
    assert result["test_verdict"]["status"] == "solved", result["test_verdict"]["reason"]
    assert result["test_verdict"]["fail_to_pass"] == reproduction["tests"]


def test_test_methods_that_do_not_compile_are_dropped_and_the_rest_kept(repo, tmp_path):
    writer = tmp_path / "repro.java"
    writer.write_text(
        REPRO.replace(
            "\n}\n",
            "\n    @Test\n    public void testIssueRemoves() {\n"
            "        assertTrue(new StudentService().remove(103));\n    }\n}\n",
        ),
        encoding="utf-8",
    )

    reproduction, _ = prepare_reproduction(
        repo, "contains() skips the last student", [CHUNK], tmp_path / "ws", stub_writer(writer), language="java"
    )

    assert reproduction["status"] == "accepted", reproduction["reason"]
    assert reproduction["tests"] == ["AgentReproTest::testIssueLastStudentIsFound()"]
    assert "remove(103)" not in reproduction["source"]


def test_tests_that_never_call_the_repaired_method_are_dropped():
    from src.agents.testing_agent.reproduction import keep_tests_calling

    unrelated = REPRO.replace(
        "\n}\n",
        "\n    @Test\n    public void testIssueTopStudents() {\n"
        "        assertEquals(2, new StudentService().getTopStudents(80.0).size());\n    }\n}\n",
    )

    kept = keep_tests_calling(unrelated, ["contains"])

    assert "testIssueLastStudentIsFound" in kept and "testIssueTopStudents" not in kept
    with pytest.raises(ValueError, match="No test calls average"):
        keep_tests_calling(unrelated, ["average"])


def test_the_empty_input_check_reproduces_a_divide_by_zero_and_the_guard_solves_it(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "Grades.java").write_text(
        "import java.util.*;\n\npublic class Grades {\n"
        "    private List<Double> marks = new ArrayList<>();\n\n"
        "    public double average() {\n"
        "        double total = 0;\n"
        "        for (double m : marks) total += m;\n"
        "        return total / marks.size();\n"
        "    }\n}\n",
        encoding="utf-8",
    )
    guard = (
        "--- a/Grades.java\n+++ b/Grades.java\n@@ -6,2 +6,5 @@\n"
        "     public double average() {\n+        if (marks.isEmpty()) {\n+            return 0;\n+        }\n"
        "         double total = 0;\n"
    )
    chunk = {"file": "Grades.java", "name": "average", "type": "method", "language": "java",
             "start_line": 6, "end_line": 10, "code": ""}
    workspaces = tmp_path / "ws"

    # The model's test asserts nothing about empty input: rejected, the built-in check takes over.
    reproduction, baseline = prepare_reproduction(
        root, "The average raises an exception.", [chunk], workspaces,
        lambda prompt: "public class AgentReproTest { @Test public void testIssueX() { assertEquals(0.0, 0.0); } }",
        language="java",
    )

    assert reproduction["status"] == "accepted", reproduction["reason"]
    assert "built-in check" in reproduction["reason"]
    result = evaluate_patch(
        root, guard, workspaces, language="java", baseline=baseline,
        extra_files={reproduction["path"]: reproduction["source"]}, reproduction_tests=reproduction["tests"],
    )
    assert result["test_verdict"]["status"] == "solved", result["test_verdict"]["reason"]


def test_a_test_expecting_the_reported_exception_is_dropped():
    from src.agents.testing_agent.reproduction import drop_guessed_exceptions

    guessed = REPRO.replace(
        "\n}\n",
        "\n    @Test\n    public void testIssueThrows() {\n"
        "        assertThrows(IllegalStateException.class, () -> new StudentService().contains(1));\n    }\n}\n",
    )

    assert "testIssueThrows" not in drop_guessed_exceptions(guessed, "contains() raises an exception.")
    assert "testIssueThrows" in drop_guessed_exceptions(guessed, "contains() should throw for unknown ids.")


def test_clean_java_test_adds_imports_and_the_package_and_rejects_copies():
    source = clean_java_test("```java\npackage wrong;\n" + REPRO + "```", "package school;\n" + SERVICE)

    assert source.startswith("package school;\nimport org.junit.jupiter.api.Test;\n"
                             "import static org.junit.jupiter.api.Assertions.*;\nimport java.util.*;\n")
    assert "package wrong" not in source
    with pytest.raises(ValueError, match="redefines StudentService"):
        clean_java_test(REPRO + SERVICE, SERVICE)
