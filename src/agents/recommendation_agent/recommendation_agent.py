"""Recommendation Agent.

Orchestrates the full recommendation pipeline:

    Issue -> Similar Bugs -> Repair Strategy -> Recommended Tools -> Recommended Tests

Each stage lives in its own module so it can be tested and swapped
independently; this file just wires them together and formats the result.

CLI demo:
    python -m src.agents.recommendation_agent.recommendation_agent
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.recommendation_agent.bug_recommender import (
    DEFAULT_TOP_K,
    SimilarBug,
    recommend_similar_bugs,
)
from src.agents.recommendation_agent.strategy_recommender import StrategyRecommendation
from src.agents.recommendation_agent.strategy_recommender import recommend as recommend_strategy
from src.agents.recommendation_agent.test_recommender import recommend as recommend_tests
from src.agents.recommendation_agent.tool_recommender import recommend as recommend_tools


@dataclass
class Recommendation:
    """The combined output of the recommendation pipeline for one issue."""

    similar_bugs: list[SimilarBug]
    strategy: StrategyRecommendation
    tools: list[str]
    tests: list[str]


def recommend(
    issue_text: str,
    code: str = "",
    language: str = "python",
    top_k: int = DEFAULT_TOP_K,
) -> Recommendation:
    """Run Issue -> Similar Bugs -> Repair Strategy -> Tools -> Tests for one issue."""
    similar_bugs = recommend_similar_bugs(issue_text, top_k=top_k)
    strategy = recommend_strategy(issue_text, code=code, similar_bugs=similar_bugs)
    tools = recommend_tools(language)
    tests = recommend_tests(issue_text, code=code)

    return Recommendation(similar_bugs=similar_bugs, strategy=strategy, tools=tools, tests=tests)


def format_report(rec: Recommendation) -> str:
    """Render a Recommendation as a plain-text report."""
    lines = ["SIMILAR BUGS"]
    if rec.similar_bugs:
        for bug in rec.similar_bugs:
            lines += [
                f'"{bug.issue}"',
                f"Similarity: {bug.similarity}",
                f'Previous Fix: "{bug.fix}"',
                "",
            ]
    else:
        lines += ["(no similar bugs found)", ""]

    lines += [
        "REPAIR STRATEGY",
        f"Bug Type: {rec.strategy.bug_type}",
        f"Strategy: {rec.strategy.strategy}",
        "",
        "RECOMMENDED TOOLS",
        *[f"- {tool}" for tool in rec.tools],
        "",
        "RECOMMENDED TESTS",
        *[f"- {test}" for test in rec.tests],
    ]
    return "\n".join(lines)


def _demo() -> None:
    issue = "Login fails when username is empty"
    code = (
        "def login(username, password):\n"
        "    user = get_user(username)\n"
        "    return user.check_password(password)\n"
    )
    print(format_report(recommend(issue, code=code, language="python")))


if __name__ == "__main__":
    _demo()
