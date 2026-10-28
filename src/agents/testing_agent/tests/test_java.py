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


def test_a_patch_that_does_not_compile_is_reported_as_such_not_as_broken_tests(repo, tmp_path):
    workspaces = tmp_path / "workspaces"
    writer = tmp_path / "repro.java"
    writer.write_text(REPRO, encoding="utf-8")
    reproduction, baseline = prepare_reproduction(
        repo, "contains() skips the last student", [CHUNK], workspaces, stub_writer(writer), language="java"
    )

    result = evaluate_patch(
        repo, FIX.replace("+        for (int i = 0; i < ids.size(); i++) {", "+        for (int i = 0; i < ids.count(); i++) {"),
        workspaces, language="java", baseline=baseline,
        extra_files={reproduction["path"]: reproduction["source"]}, reproduction_tests=reproduction["tests"],
    )

    verdict = result["test_verdict"]
    assert verdict["stage"] == "compile" and verdict["status"] != "solved"
    assert verdict["reason"].startswith("The patched code does not compile: javac failed")
    assert verdict["pass_to_fail"] == []


# The dummy_repo file: a method with no class around it.
LOOSE_METHOD = """public int calculateTotalRevenue(int[] prices, int[] quantities) {
    int total = 0;
    for (int i = 0; i <= prices.length; i++) {
        total += prices[i] - quantities[i];
    }
    return total / prices.length;
}"""
REVENUE_REPRO = """import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class AgentReproTest {
    @Test
    public void testIssueCorrectTotal() {
        assertEquals(80, new calculateRevenue().calculateTotalRevenue(new int[]{10, 20}, new int[]{2, 3}));
    }
}
"""


def test_a_loose_method_compiles_inside_a_class_named_after_its_file(tmp_path):
    from src.agents.testing_agent.java import as_compiled

    wrapped = as_compiled(LOOSE_METHOD, "calculateRevenue.java")
    # Same line numbers as the file, so javac errors still point at its lines.
    assert wrapped.splitlines()[0] == "public class calculateRevenue { " + LOOSE_METHOD.splitlines()[0]
    assert len(wrapped.splitlines()) == len(LOOSE_METHOD.splitlines()) + 1
    assert as_compiled(SERVICE, "Student.java") == SERVICE

    (tmp_path / "calculateRevenue.java").write_text(LOOSE_METHOD, encoding="utf-8")
    (tmp_path / "AgentReproTest.java").write_text(REVENUE_REPRO, encoding="utf-8")
    before = run_suite(tmp_path)
    assert (before["passed"], before["failed"]) == (0, 1)  # the index is out of bounds, not a javac error

    fixed = LOOSE_METHOD.replace("<=", "<").replace(" - ", " * ").replace(" / prices.length", "")
    (tmp_path / "calculateRevenue.java").write_text(fixed, encoding="utf-8")
    assert run_suite(tmp_path)["status"] == "PASS"


def test_two_unrelated_files_declaring_one_class_do_not_stop_the_build(tmp_path):
    # OrderService.java and E-Commerce.java in dummy_repo each declare `class Product`.
    (tmp_path / "OrderService.java").write_text(
        "class Product { int price() { return 1; } }\npublic class OrderService { int total() { return new Product().price(); } }\n"
    )
    (tmp_path / "E-Commerce.java").write_text("class Product { int stock() { return 2; } }\n")
    (tmp_path / "Student.java").write_text(SERVICE)
    (tmp_path / "AgentReproTest.java").write_text(
        "import org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.*;\n"
        "public class AgentReproTest {\n    @Test\n    public void testIssueTotal() { assertEquals(1, new OrderService().total()); }\n}\n"
    )

    result = run_suite(tmp_path)

    # The test uses OrderService, so its Product is compiled and E-Commerce's is not.
    assert (result["status"], result["passed"]) == ("PASS", 1)


def test_a_loose_array_method_is_reproduced_and_the_correct_repair_solved(tmp_path):
    # dummy_repo issue #7, with the test codellama:7b wrote for it: a bare call
    # (no class to call it on) and a guessed total of 20 (the right one is 140).
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calculateRevenue.java").write_text(LOOSE_METHOD, encoding="utf-8")
    model_test = (
        "class AgentReproTest {\n    @Test\n    void testIssue1() {\n"
        "        int[] prices = {10, 20, 30};\n        int[] quantities = {1, 2, 3};\n"
        "        assertEquals(20, calculateTotalRevenue(prices, quantities));\n    }\n}\n"
    )
    chunk = {"file": "calculateRevenue.java", "name": "calculateTotalRevenue", "type": "method", "language": "java",
             "start_line": 1, "end_line": 7, "code": ""}
    workspaces = tmp_path / "ws"
    issue = "The function of calculate revenue is wrong. Please fix the buggy code which gives the correct total."

    reproduction, baseline = prepare_reproduction(root, issue, [chunk], workspaces, lambda prompt: model_test,
                                                  language="java")

    # The bare call compiled (it now goes through new calculateRevenue()), but the index
    # error is no assertion: the built-in check on arrays of one length reproduces it.
    assert reproduction["status"] == "accepted", reproduction["reason"]
    assert "new calculateRevenue().calculateTotalRevenue(new int[]{3, 5, 7}, new int[]{3, 5, 7})" in reproduction["source"]
    patch = (
        "--- a/calculateRevenue.java\n+++ b/calculateRevenue.java\n@@ -1,7 +1,7 @@\n"
        " public int calculateTotalRevenue(int[] prices, int[] quantities) {\n     int total = 0;\n"
        "-    for (int i = 0; i <= prices.length; i++) {\n-        total += prices[i] - quantities[i];\n"
        "+    for (int i = 0; i < prices.length; i++) {\n+        total += prices[i] * quantities[i];\n"
        "     }\n-    return total / prices.length;\n+    return total;\n }\n\\ No newline at end of file\n"
    )
    result = evaluate_patch(
        root, patch, workspaces, language="java", baseline=baseline,
        extra_files={reproduction["path"]: reproduction["source"]}, reproduction_tests=reproduction["tests"],
    )
    assert result["test_verdict"]["status"] == "solved", result["test_verdict"]["reason"]


# dummy_repo issues #4 and #5, with the tests codellama:7b wrote for them.
ORDER_SERVICE = '\nimport java.util.*;\n\nclass Product {\n    private int id;\n    private String name;\n    private double price;\n    private int stock;\n\n    public Product(int id, String name, double price, int stock) {\n        this.id = id;\n        this.name = name;\n        this.price = price;\n        this.stock = stock;\n    }\n\n    public int getId() {\n        return id;\n    }\n\n    public String getName() {\n        return name;\n    }\n\n    public double getPrice() {\n        return price;\n    }\n\n    public int getStock() {\n        return stock;\n    }\n\n    public void reduceStock(int quantity) {\n        stock -= quantity;\n    }\n}\n\nclass OrderItem {\n    private Product product;\n    private int quantity;\n\n    public OrderItem(Product product, int quantity) {\n        this.product = product;\n        this.quantity = quantity;\n    }\n\n    public Product getProduct() {\n        return product;\n    }\n\n    public int getQuantity() {\n        return quantity;\n    }\n\n    public double getSubtotal() {\n        return product.getPrice() * quantity;\n    }\n}\n\nclass Order {\n    private int orderId;\n    private List<OrderItem> items;\n    private String status;\n    private double total;\n\n    public Order(int orderId, List<OrderItem> items) {\n        this.orderId = orderId;\n        this.items = items;\n        this.status = "CREATED";\n        this.total = 0;\n    }\n\n    public int getOrderId() {\n        return orderId;\n    }\n\n    public List<OrderItem> getItems() {\n        return items;\n    }\n\n    public String getStatus() {\n        return status;\n    }\n\n    public void setStatus(String status) {\n        this.status = status;\n    }\n\n    public double getTotal() {\n        return total;\n    }\n\n    public void setTotal(double total) {\n        this.total = total;\n    }\n}\n\npublic class OrderService {\n\n    private List<Order> orders = new ArrayList<>();\n\n    public Order createOrder(\n            int orderId, List<OrderItem> items) {\n\n        Order order = new Order(orderId, items);\n\n        for (OrderItem item : items) {\n            Product product = item.getProduct();\n\n            if (product.getStock() >= item.getQuantity()) {\n                product.reduceStock(item.getQuantity());\n            }\n        }\n\n        orders.add(order);\n        return order;\n    }\n\n    public double calculateTotal(Order order) {\n        double total = 0;\n\n        for (OrderItem item : order.getItems()) {\n            total += item.getSubtotal();\n        }\n\n        order.setTotal(total);\n        return total;\n    }\n\n    public boolean cancelOrder(int orderId) {\n        Order order = findOrder(orderId);\n\n        if (order != null) {\n            order.setStatus("CANCELLED");\n            return true;\n        }\n\n        return false;\n    }\n\n    public Order findOrder(int orderId) {\n        for (int i = 0; i < orders.size() - 1; i++) {\n            if (orders.get(i).getOrderId() == orderId) {\n                return orders.get(i);\n            }\n        }\n\n        return null;\n    }\n\n    public List<Order> getOrdersByStatus(String status) {\n        List<Order> result = new ArrayList<>();\n\n        for (Order order : orders) {\n            if (order.getStatus() == status) {\n                result.add(order);\n            }\n        }\n\n        return result;\n    }\n\n    public double applyDiscount(\n            Order order, double discountPercentage) {\n\n        double total = calculateTotal(order);\n        return total - (total * discountPercentage);\n    }\n}\n'
BUGGY_TRIE = 'import java.util.HashMap;\nimport java.util.Map;\n\npublic class BuggyTrie {\n    private static class TrieNode {\n        Map<Character, TrieNode> children = new HashMap<>();\n        boolean isEnd = false;\n    }\n\n    private final TrieNode root = new TrieNode();\n\n    public void insert(String word) {\n        TrieNode node = root;\n        for (char ch : word.toCharArray()) {\n            node.children.putIfAbsent(ch, new TrieNode());\n            node = node.children.get(ch);\n        }\n        node.isEnd = true;\n    }\n\n    public boolean search(String word) {\n        TrieNode node = root;\n        for (char ch : word.toCharArray()) {\n            if (!node.children.containsKey(ch)) {\n                return false;\n            }\n            node = node.children.get(ch);\n        }\n        return true; // BUG: should return node.isEnd\n    }\n\n    public static void main(String[] args) {\n        BuggyTrie trie = new BuggyTrie();\n        trie.insert("apple");\n\n        System.out.println(trie.search("apple"));\n        System.out.println(trie.search("app")); \n    }\n}'


def _solved(tmp_path, file, source, chunk, issue, model_test, old, new):
    root = tmp_path / "repo"
    root.mkdir()
    (root / file).write_text(source, encoding="utf-8")
    workspaces = tmp_path / "ws"
    reproduction, baseline = prepare_reproduction(root, issue, [chunk], workspaces, lambda prompt: model_test,
                                                  language="java")
    assert reproduction["status"] == "accepted", reproduction["reason"]
    from src.agents.coding_agent.function_patch import make_diff

    patch = make_diff(file, source, source.replace(old, new))
    result = evaluate_patch(
        root, patch, workspaces, language="java", baseline=baseline,
        extra_files={reproduction["path"]: reproduction["source"]}, reproduction_tests=reproduction["tests"],
    )
    assert result["test_verdict"]["status"] == "solved", result["test_verdict"]["reason"]
    return reproduction


def test_a_lookup_on_an_empty_service_is_replaced_by_the_built_in_last_item_check(tmp_path):
    # Nothing is ever added, so findOrder returns null before and after any fix.
    model_test = (
        "public class AgentReproTest {\n    @Test\n    void testIssue1() {\n"
        "        Order order = new OrderService().findOrder(1);\n        assertNotNull(order);\n    }\n}\n"
    )
    chunk = {"file": "OrderService.java", "name": "findOrder", "type": "method", "language": "java",
             "start_line": 141, "end_line": 149, "code": ""}
    issue = ("In OrderService to find a order, it is omitting the last order from the list. Fix the buggy code. "
             "Changing the loop condition alone would be suffice.")

    reproduction = _solved(tmp_path, "OrderService.java", ORDER_SERVICE, chunk, issue, model_test,
                           "        for (int i = 0; i < orders.size() - 1; i++) {",
                           "        for (int i = 0; i < orders.size(); i++) {")

    assert "subject.createOrder(103, new ArrayList<>());" in reproduction["source"]


def test_the_test_is_told_the_class_that_holds_the_method_not_a_nested_one(tmp_path):
    from src.agents.testing_agent.reproduction import build_java_prompt, java_call

    assert java_call(BUGGY_TRIE, "search") == "new BuggyTrie().search"
    prompt = build_java_prompt("issue", "Trie.java", BUGGY_TRIE, methods=["search"])
    assert "Call search exactly as `new BuggyTrie().search(...)`." in prompt
    assert "    public void insert(String word)" in prompt
    # The test codellama:7b wrote once told the right class.
    model_test = (
        "class AgentReproTest {\n    @Test\n    void testIssueSearch() {\n"
        "        BuggyTrie subject = new BuggyTrie();\n        subject.insert(\"apple\");\n"
        "        assertFalse(subject.search(\"app\"));\n    }\n}\n"
    )
    chunk = {"file": "Trie.java", "name": "search", "type": "method", "language": "java",
             "start_line": 21, "end_line": 30, "code": ""}

    _solved(tmp_path, "Trie.java", BUGGY_TRIE, chunk, "search() returns True for a prefix.", model_test,
            "        return true; // BUG: should return node.isEnd", "        return node.isEnd;")
