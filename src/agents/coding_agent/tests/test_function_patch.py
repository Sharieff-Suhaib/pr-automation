"""Function mode: the model returns fixed functions; the diff is built locally.

The model is replaced by a canned reply, so these run without Ollama.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from src.agents.coding_agent import function_patch
from src.agents.coding_agent.code_generator import PatchGenerationError
from src.agents.coding_agent.function_patch import generate_function_patch
from src.agents.testing_agent import evaluate_patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_REPO = PROJECT_ROOT / "orchestrator" / "sample_repo"

# The repository agent's chunks for sample_repo/users.py (1-indexed, inclusive).
CHUNKS = [
    {"file": "users.py", "type": "function", "name": "login", "start_line": 9, "end_line": 12,
     "code": 'def login(users, name, password):\n    """Return True when `name` exists and the password matches."""\n'
             '    user = get_user(users, name)\n    return user["password"] == password'},
    {"file": "users.py", "type": "function", "name": "get_user", "start_line": 4, "end_line": 6,
     "code": 'def get_user(users, name):\n    """Return the profile stored for `name`."""\n    return users[name]'},
    {"file": "test_users.py", "type": "function", "name": "test_known_user_is_returned", "start_line": 6,
     "end_line": 7, "code": "def test_known_user_is_returned(): ..."},
]

FIXED = '''Here is the fix:
```python
def get_user(users, name):
    """Return the profile stored for `name`."""
    return users.get(name)


def login(users, name, password):
    """Return True when `name` exists and the password matches."""
    user = get_user(users, name)
    if user is None:
        return False
    return user["password"] == password
```'''


@pytest.fixture
def repo(tmp_path):
    destination = tmp_path / "sample_repo"
    shutil.copytree(SAMPLE_REPO, destination)
    return destination


def reply_with(monkeypatch, reply: str) -> list[str]:
    """Make the model return `reply`; returns the list the prompts are recorded in."""
    prompts: list[str] = []

    def fake_call(prompt, model=None, system=None, temperature=0):
        prompts.append(prompt)
        return reply

    monkeypatch.setattr(function_patch, "_call_ollama", fake_call)
    return prompts


def test_the_generated_diff_applies_and_fixes_the_bug(repo, tmp_path, monkeypatch):
    prompts = reply_with(monkeypatch, FIXED)

    patch = generate_function_patch("get_user crashes on unknown users", repo, CHUNKS)

    assert patch.startswith("--- a/users.py\n+++ b/users.py\n")
    assert "test_known_user_is_returned" not in prompts[0]  # test files are not sent
    result = evaluate_patch(repo, patch, tmp_path / "workspaces")
    assert result["patch_status"] == "APPLIED"
    assert result["test_verdict"]["status"] == "solved"


def test_a_function_the_model_left_out_is_kept(repo, monkeypatch):
    reply_with(monkeypatch, "def get_user(users, name):\n    return users.get(name)\n")

    patch = generate_function_patch("issue", repo, CHUNKS)

    assert "+    return users.get(name)" in patch
    assert "login" not in "".join(line for line in patch.splitlines() if line.startswith(("+", "-")))


def test_a_method_is_put_back_inside_its_class(tmp_path, monkeypatch):
    (tmp_path / "store.py").write_text(
        "class Store:\n"
        "    @property\n"
        "    def size(self):\n"
        "        return len(self.items) + 1\n",
        encoding="utf-8",
    )
    chunk = {"file": "store.py", "type": "method", "name": "size", "start_line": 3, "end_line": 4,
             "code": "def size(self):\n        return len(self.items) + 1"}
    prompts = reply_with(monkeypatch, "@property\ndef size(self):\n    return len(self.items)\n")

    patch = generate_function_patch("size is off by one", tmp_path, [chunk])

    assert "def size(self):\n    return len(self.items) + 1" in prompts[0]  # dedented for the model
    assert "-        return len(self.items) + 1\n+        return len(self.items)\n" in patch
    assert patch.count("@property") == 1  # the original decorator is not duplicated


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (CHUNKS[1]["code"], "unchanged"),
        ("def get_user(users, name)\n    return None\n", "not valid Python"),
        ("x = 1\n", "did not return any function"),
    ],
)
def test_unusable_replies_raise(repo, monkeypatch, reply, reason):
    reply_with(monkeypatch, reply)

    with pytest.raises(PatchGenerationError, match=reason):
        generate_function_patch("issue", repo, CHUNKS)


def test_no_python_function_to_repair_raises(repo):
    with pytest.raises(PatchGenerationError, match="No Python function"):
        generate_function_patch("issue", repo, [CHUNKS[2]])
