"""Tool Recommender.

Answers: *which development tools does this repair need?*

Simple, explainable rule-based suggestions keyed by language. Deliberately not
LLM-based -- there's no ambiguity to resolve here, just a lookup.
"""

from __future__ import annotations

_TOOLS_BY_LANGUAGE: dict[str, list[str]] = {
    "python": ["Pytest", "Ruff", "Git"],
    "javascript": ["Jest", "ESLint", "Git"],
    "typescript": ["Jest", "ESLint", "TypeScript Compiler", "Git"],
    "java": ["JUnit", "Checkstyle", "Git"],
    "kotlin": ["JUnit", "ktlint", "Git"],
    "go": ["go test", "golangci-lint", "Git"],
    "rust": ["cargo test", "clippy", "Git"],
    "ruby": ["RSpec", "RuboCop", "Git"],
    "c": ["Unity", "cppcheck", "Git"],
    "cpp": ["Google Test", "cppcheck", "Git"],
    "c_sharp": ["NUnit", "dotnet format", "Git"],
    "php": ["PHPUnit", "PHP_CodeSniffer", "Git"],
}

_DEFAULT_TOOLS = ["Git"]


def recommend(language: str) -> list[str]:
    """Recommend tools for the given language, defaulting to Git-only when unknown."""
    return list(_TOOLS_BY_LANGUAGE.get((language or "").lower().strip(), _DEFAULT_TOOLS))
