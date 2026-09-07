"""Run a real Phase 1 patch-generation check.

Start Ollama and pull the configured code model before running this file:
    python -m src.agents.coding_agent.phase_1_test

It can also be run directly from the repository root:
    python src/agents/coding_agent/phase_1_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Direct script execution adds this file's directory—not the repository root—to
# sys.path. Make the top-level ``src`` package importable in either invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.agents.coding_agent import generate_patch
from src.agents.coding_agent.code_generator import PatchGenerationError


def main() -> int:
    """Generate and print a patch for one small, known bug."""
    try:
       patch = generate_patch(
    issue="Calculating an average with no values raises ZeroDivisionError; return -1 instead.",
    relevant_code="""File: stats.py

def average(values):
    return sum(values) / len(values)
""",
    similar_bugs=[],
    strategy="Check whether the input collection is empty before dividing by its length.",
    tools=["pytest"],
    tests=["tests/test_stats.py"],
)
    except PatchGenerationError as error:
        print(f"Patch generation failed: {error}")
        return 1

    print("Generated patch:\n")
    print(patch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
