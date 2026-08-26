"""Smoke test for the Code Generation Agent.

Sends a few buggy snippets to CodeLlama-7B via Ollama and prints the fixes.

Run:
    ollama serve                    # if it is not already running
    ollama pull codellama:7b
    python scripts/test_codegen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.codegen_agent import DEFAULT_MODEL, OllamaError, fix_code  # noqa: E402

CASES: list[tuple[str, str, str]] = [
    (
        "empty list crash",
        "def get_first(items):\n    return items[0]\n",
        "get_first raises IndexError when items is empty",
    ),
    (
        "off by one",
        "def total(numbers):\n    result = 0\n    for i in range(len(numbers) - 1):\n        result += numbers[i]\n    return result\n",
        "total() skips the last number",
    ),
    (
        "no issue text",
        "def average(values):\n    return sum(values) / len(values) + 1\n",
        "",
    ),
]


def main() -> int:
    print(f"model: {DEFAULT_MODEL}\n")
    for name, buggy, issue in CASES:
        print("=" * 60)
        print(f"CASE: {name}")
        print("=" * 60)
        print("--- buggy ---")
        print(buggy.strip())
        try:
            fixed = fix_code(buggy, issue or None)
        except OllamaError as exc:
            print(f"\n[!!] {exc}")
            return 1
        print("\n--- fixed ---")
        print(fixed or "[empty response]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
