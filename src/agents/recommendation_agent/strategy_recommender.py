"""Repair Strategy Recommender.

Answers: *what kind of fix does this issue call for?*

Given the issue, the relevant code, and any similar historical bugs, asks an
LLM (the same local Ollama server codegen_agent.py talks to) to name the bug
type and recommend a repair strategy. Falls back to the keyword-based
classifier in strategy_agent.py when Ollama is unreachable or the response
can't be parsed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.agents.codegen_agent import DEFAULT_MODEL, OllamaError, call_ollama
from src.agents.strategy_agent import STRATEGIES, classify

SYSTEM_PROMPT = (
    "You are a senior software engineer diagnosing a bug report. Reply with "
    'strict JSON only, of the form {"bug_type": "<short label>", '
    '"strategy": "<one or two sentence repair strategy>"}. '
    "No markdown, no commentary, no extra keys."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class StrategyRecommendation:
    """The diagnosed bug type and the strategy recommended to repair it."""

    bug_type: str
    strategy: str
    source: str = "llm"  # "llm" | "keyword_fallback"


def _bug_text(bug) -> tuple[str, str]:
    """Similar bugs may arrive as bug_recommender.SimilarBug or plain dicts."""
    if isinstance(bug, dict):
        return bug.get("issue", ""), bug.get("fix", bug.get("fix_summary", ""))
    return getattr(bug, "issue", ""), getattr(bug, "fix", getattr(bug, "fix_summary", ""))


def build_prompt(issue_text: str, code: str, similar_bugs: list) -> str:
    """Assemble the diagnosis brief handed to the LLM."""
    parts = ["### ISSUE", issue_text.strip()]
    if code and code.strip():
        parts += ["\n### RELEVANT CODE", code.strip()]
    if similar_bugs:
        parts.append("\n### SIMILAR PAST BUGS")
        for bug in similar_bugs:
            bug_issue, bug_fix = _bug_text(bug)
            parts.append(f"- {bug_issue} -> fixed by: {bug_fix}")
    parts += [
        "\n### INSTRUCTION",
        "Identify the bug type and recommend a suitable repair strategy.",
    ]
    return "\n".join(parts)


def _parse_llm_response(raw: str) -> StrategyRecommendation | None:
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        bug_type = str(data["bug_type"]).strip()
        strategy = str(data["strategy"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not bug_type or not strategy:
        return None
    return StrategyRecommendation(bug_type=bug_type, strategy=strategy, source="llm")


def _fallback(issue_text: str) -> StrategyRecommendation:
    bug_type = classify(issue_text)
    return StrategyRecommendation(bug_type=bug_type, strategy=STRATEGIES[bug_type], source="keyword_fallback")


def recommend(
    issue_text: str,
    code: str = "",
    similar_bugs: list | None = None,
    model: str = DEFAULT_MODEL,
) -> StrategyRecommendation:
    """Recommend a bug type + repair strategy for the issue."""
    prompt = build_prompt(issue_text, code, similar_bugs or [])
    try:
        raw = call_ollama(prompt, model=model, system=SYSTEM_PROMPT)
    except OllamaError:
        return _fallback(issue_text)

    return _parse_llm_response(raw) or _fallback(issue_text)
