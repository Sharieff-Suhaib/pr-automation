"""Similar Bug Recommendation Agent.

Answers: *which historical bugs resemble this issue, and how were they fixed?*

Backed by recommendation_agent.bug_recommender: a FAISS index (falling back to
lexical similarity when embeddings aren't available) over data/bugs.json.
STAGE D can extend that corpus with the Feedback Agent's store of past
repairs -- that second source is what closes the loop in the architecture:
successful repairs become precedent for future issues.
"""

from __future__ import annotations

from src.agents.recommendation_agent.bug_recommender import recommend_similar_bugs
from src.agents.state import AgentState, SimilarBug, trace_entry

DEFAULT_TOP_K = 3


def recommend(state: AgentState, top_k: int = DEFAULT_TOP_K) -> AgentState:
    """Retrieve historical bug-fix pairs similar to the current issue."""
    issue = state["issue"]

    matches = recommend_similar_bugs(issue.as_text(), top_k=top_k)
    bugs = [
        SimilarBug(issue=match.issue, fix_summary=match.fix, similarity=match.similarity, source=match.source)
        for match in matches
    ]

    return AgentState(
        similar_bugs=bugs,
        trace=[
            trace_entry(
                "similar_bug_agent",
                f"{len(bugs)} similar bug(s)",
                query=issue.title,
            )
        ],
    )
