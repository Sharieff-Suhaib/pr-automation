"""Test Recommender.

Answers: *which tests should be run to validate this repair?*

Asks the LLM (the same local Ollama server codegen_agent.py talks to) which
tests should be executed, given the issue and the relevant code. Falls back to
a naive heuristic -- one test per function defined in the code, plus an
issue-derived edge case -- when Ollama is unreachable.
"""

from __future__ import annotations

import re

from src.agents.codegen_agent  import DEFAULT_MODEL, OllamaError, call_ollama

SYSTEM_PROMPT = (
    "You are a QA engineer. Based on the issue and the relevant code, list the "
    "tests that should be executed to validate a fix. Reply with one test name "
    "per line, each a valid Python identifier starting with 'test_'. "
    "No commentary, no numbering, no markdown."
)

_FUNC_DEF_RE = re.compile(r"\b(?:def|function)\s+(\w+)\s*\(")
_TEST_NAME_RE = re.compile(r"(test_\w+)")
_WORD_RE = re.compile(r"[a-zA-Z]+")


def build_prompt(issue_text: str, code: str) -> str:
    """Assemble the brief handed to the LLM."""
    parts = ["### ISSUE", issue_text.strip()]
    if code and code.strip():
        parts += ["\n### RELEVANT CODE", code.strip()]
    parts += [
        "\n### INSTRUCTION",
        "Suggest the tests that should be executed to validate a fix for this issue.",
    ]
    return "\n".join(parts)


def _parse_llm_response(raw: str) -> list[str]:
    tests = []
    for line in raw.splitlines():
        match = _TEST_NAME_RE.search(line)
        if match:
            tests.append(f"{match.group(1)}()")
    return _dedupe(tests)


def _fallback(issue_text: str, code: str) -> list[str]:
    tests = [f"test_{name}()" for name in _FUNC_DEF_RE.findall(code or "")]

    keywords = [w.lower() for w in _WORD_RE.findall(issue_text or "") if len(w) > 3]
    if keywords:
        tests.append(f"test_{'_'.join(keywords[:3])}()")

    if not tests:
        tests.append("test_reported_issue()")

    return _dedupe(tests)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recommend(issue_text: str, code: str = "", model: str = DEFAULT_MODEL) -> list[str]:
    """Recommend tests to validate a repair for this issue."""
    prompt = build_prompt(issue_text, code)
    try:
        raw = call_ollama(prompt, model=model, system=SYSTEM_PROMPT)
    except OllamaError:
        return _fallback(issue_text, code)

    return _parse_llm_response(raw) or _fallback(issue_text, code)
