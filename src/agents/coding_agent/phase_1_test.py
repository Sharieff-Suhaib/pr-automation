"""Run a real Phase 1 patch-generation check.

Start Ollama and pull the configured code model before running this file:
    python -m src.agents.coding_agent.phase_1_test
"""

from __future__ import annotations

from src.agents.coding_agent import generate_patch
from src.agents.coding_agent.code_generator import PatchGenerationError


def main() -> int:
    """Generate and print a patch for one small, known bug."""
    try:
        patch = generate_patch(
            issue="Looking up a missing user raises KeyError; return None instead.",
            relevant_code="""File: users.py

def get_user(users, name):
    return users[name]
""",
            similar_bugs=[],
            strategy="Validate that the user exists before lookup.",
            tools=["pytest"],
            tests=["tests/test_users.py"],
        )
    except PatchGenerationError as error:
        print(f"Patch generation failed: {error}")
        return 1

    print("Generated patch:\n")
    print(patch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
