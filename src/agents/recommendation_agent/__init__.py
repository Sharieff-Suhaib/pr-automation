"""Recommendation Agent package.

Given a GitHub issue (and optionally the relevant code), produces:

    Similar Bugs -> Repair Strategy -> Recommended Tools -> Recommended Tests
"""

from src.agents.recommendation_agent.recommendation_agent import Recommendation, format_report, recommend

__all__ = ["Recommendation", "format_report", "recommend"]
