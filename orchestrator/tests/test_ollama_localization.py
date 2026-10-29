"""Ollama fault localization: Ollama names the line the bug starts on and the
line it ends on, and that one region goes to the remote model as IR4 under a
single <FILL_ME>.

No network: Ollama and the Colab endpoint are faked.
"""

from __future__ import annotations

import io
import json
import subprocess

import pytest

from orchestrator import adapters
from orchestrator.tests.test_remote_and_push import REVENUE
from src.agents import codegen_agent
from src.agents.coding_agent import fault_localization

REVENUE_ISSUE = "calculateTotalRevenue returns the wrong total revenue."
REVENUE_CHUNK = {"file": "Revenue.java", "name": "calculateTotalRevenue", "type": "method", "language": "java",
                 "start_line": 2, "end_line": 8, "code": ""}


@pytest.fixture
def ollama(monkeypatch):
    """Switch Ollama localization back on and fake its reply."""
    monkeypatch.setattr(fault_localization, "LOCALIZER", "ollama")
    prompts: list[str] = []

    def install(reply):
        def fake_call_ollama(prompt, **kwargs):
            prompts.append(prompt)
            assert kwargs["json_mode"] is True
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(codegen_agent, "call_ollama", fake_call_ollama)
        return prompts

    return install


def test_the_region_from_bug_start_to_bug_end_goes_to_the_remote_model_as_one_ir4(tmp_path, monkeypatch, ollama):
    (tmp_path / "Revenue.java").write_text(REVENUE, encoding="utf-8")
    asked = ollama('{"start": 4, "end": 7}')
    sent = []

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data)["prompt"])
        reply = ("for (int i = 0; i < prices.length; i++) {\n            total += prices[i] * quantities[i];\n"
                 "        }\n        return total;\n</s>")
        return io.BytesIO(json.dumps({"repair": reply}).encode())

    monkeypatch.setattr(adapters.urllib.request, "urlopen", fake_urlopen)

    patch = adapters.remote_patch(str(tmp_path), [REVENUE_CHUNK], issue=REVENUE_ISSUE)

    # Ollama sees the whole method by file line number, and the bug report.
    assert "2 |     public int calculateTotalRevenue(int[] prices, int[] quantities) {" in asked[0]
    assert "8 |     }" in asked[0] and "public class Revenue" not in asked[0]
    assert REVENUE_ISSUE in asked[0]
    # Everything from the bug's start to its end is commented out under one <FILL_ME>.
    assert sent == [
        "Repair the bug in this Java code: calculateTotalRevenue returns the wrong total revenue.\n"
        "Return only the missing code.\n\n"
        "    public int calculateTotalRevenue(int[] prices, int[] quantities) {\n"
        "        int total = 0;\n"
        "// buggy code\n"
        "//         for (int i = 0; i <= prices.length; i++) {\n"
        "//             total += prices[i] - quantities[i];\n"
        "//         }\n"
        "//         return total / prices.length;\n"
        "        <FILL_ME>\n"
        "    }"
    ]
    assert "+        for (int i = 0; i < prices.length; i++) {\n" \
           "+            total += prices[i] * quantities[i];\n" in patch
    assert "+        return total;\n" in patch
    subprocess.run(["git", "apply", "--check", "-"], cwd=tmp_path, input=patch, text=True, check=True)


def test_a_buggy_line_the_region_reply_copied_back_is_asked_about_alone(tmp_path, monkeypatch, ollama):
    (tmp_path / "Revenue.java").write_text(REVENUE, encoding="utf-8")
    ollama('{"buggy_lines": [4, 5, 7], "start": 4, "end": 7}')
    sent = []
    # The replies the Colab endpoint gave: the region's leaves the operator as
    # it was; asked about that line alone, it multiplies.
    region_reply = ("for (int i = 0; i < prices.length; i++) {\n            total += prices[i] - quantities[i];\n"
                    "        }\n        return total;")
    line_reply = "total += prices[i] * quantities[i];"

    def fake_urlopen(request, timeout):
        prompt = json.loads(request.data)["prompt"]
        sent.append(prompt)
        reply = line_reply if "//             total +=" in prompt and "//         for" not in prompt else region_reply
        return io.BytesIO(json.dumps({"repair": reply + "\n</s>"}).encode())

    monkeypatch.setattr(adapters.urllib.request, "urlopen", fake_urlopen)

    patch = adapters.remote_patch(str(tmp_path), [REVENUE_CHUNK], issue=REVENUE_ISSUE)

    assert len(sent) == 2
    # The follow-up holds the region's repairs and comments out only line 5.
    assert sent[1].split("\n\n", 1)[1] == (
        "    public int calculateTotalRevenue(int[] prices, int[] quantities) {\n"
        "        int total = 0;\n"
        "        for (int i = 0; i < prices.length; i++) {\n"
        "// buggy code\n"
        "//             total += prices[i] - quantities[i];\n"
        "            <FILL_ME>\n"
        "        }\n"
        "        return total;\n"
        "    }"
    )
    assert "+        for (int i = 0; i < prices.length; i++) {\n" \
           "+            total += prices[i] * quantities[i];\n" in patch
    assert "+        return total;\n" in patch
    subprocess.run(["git", "apply", "--check", "-"], cwd=tmp_path, input=patch, text=True, check=True)


@pytest.mark.parametrize(
    "reply",
    [
        '{"start": 4, "end": 7}',
        # Several regions become one, from the first start to the last end.
        '{"regions": [{"start": 4, "end": 5}, {"start": 7, "end": 7}]}',
        # A buggy line past "end" still widens the region to reach it.
        '{"buggy_lines": [4, 5, 7], "start": 4, "end": 5}',
        # The signature and the closing brace are clipped off: only the body is repaired.
        '{"start": 2, "end": 8}',
    ],
)
def test_ollama_gives_one_repair_site(tmp_path, ollama, reply):
    (tmp_path / "Revenue.java").write_text(REVENUE, encoding="utf-8")
    ollama(reply)

    _, sites = adapters._remote_sites(str(tmp_path), [REVENUE_CHUNK], REVENUE_ISSUE)

    expected = (2, 7) if '"start": 2' in reply else (3, 7)
    assert sites == [expected]


@pytest.mark.parametrize(
    "reply",
    [codegen_agent.OllamaError("Cannot reach Ollama"), '{"regions": []}', "not json", '{"start": 40, "end": 41}'],
)
def test_without_a_usable_ollama_answer_the_heuristic_sites_are_used(tmp_path, ollama, reply):
    (tmp_path / "Revenue.java").write_text(REVENUE, encoding="utf-8")
    ollama(reply)

    _, sites = adapters._remote_sites(str(tmp_path), [REVENUE_CHUNK], REVENUE_ISSUE)

    assert sites == adapters._repair_sites(REVENUE.splitlines(), 1, 2, 7, REVENUE_ISSUE)


def test_regions_are_read_loosely_then_kept_inside_the_code_and_merged():
    lines = REVENUE.splitlines()
    reply = 'Sure: {"regions": [{"start": 9, "end": 7}, {"start": 8, "end": 8}, {"start": 40}, [3, 3]]}'

    pairs = fault_localization.parse_regions(reply)
    regions = fault_localization.clean_regions(pairs, [(2, 8)], lines)

    assert pairs == [(9, 7), (8, 8), (40, 40), (3, 3)]
    # 7-9 is reversed and runs past the method; 8 is only "}"; 40 is outside it.
    assert regions == [(3, 3), (7, 8)]


def test_the_region_stays_inside_one_function(ollama):
    lines = ["def f():", "    a = 1", "    return a", "", "def g():", "    b = 2", "    return b"]
    ollama('{"regions": [{"start": 2, "end": 2}, {"start": 3, "end": 3}, {"start": 6, "end": 6}]}')

    # Lines 2-3 of f(); line 6 is in g(), which one IR4 hole cannot reach.
    assert fault_localization.ollama_localize(lines, [(1, 3), (5, 7)], "wrong value", "m.py") == (2, 3, (2, 3))


def test_the_standalone_localizer_builds_its_ir4_from_the_ollama_region(tmp_path, ollama):
    (tmp_path / "m.py").write_text("def f(a, b):\n    x = a - b\n    y = 1\n    return x\n", encoding="utf-8")
    ollama('{"start": 2, "end": 4}')

    region = fault_localization.localize_relevant_file(
        str(tmp_path), "m.py", "f returns the wrong value",
        [{"file": "m.py", "type": "function", "start_line": 1, "end_line": 4}],
    )

    assert (region.start_line, region.end_line) == (2, 4)
    assert region.ir4 == "def f(a, b):\n#     x = a - b\n#     y = 1\n#     return x\n<FILL_ME>"
