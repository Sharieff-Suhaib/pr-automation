"""The remote fine-tuned model (IR4 in, OR2 out) and the push after an approved review.

No network: the endpoint is faked, and the push goes to a local bare repository.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import subprocess

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
        "Repair the bug in this Java code.\nReturn only the missing code.\n\n"
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
