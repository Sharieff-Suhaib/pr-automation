"""Similar Bug Recommendation Agent.

Answers: *which historical bugs resemble this issue, and how were they fixed?*

STAGE A: stub.
STAGE D will back this with a FAISS index over two sources:
    1. CommitPackFT bug-fix pairs  -> cold-start corpus
    2. the Feedback Agent's store  -> repairs this system made previously

That second source is what closes the loop in the architecture: successful
repairs become precedent for future issues.
"""

from __future__ import annotations

from src.agents.state import AgentState, SimilarBug, trace_entry

DEFAULT_TOP_K = 3


def recommend(state: AgentState, top_k: int = DEFAULT_TOP_K) -> AgentState:
    """Retrieve historical bug-fix pairs similar to the current issue."""
    issue = state["issue"]

    # --- STAGE D will replace this with FAISS similarity search ---
    bugs = [
        SimilarBug(
            issue="[STUB] placeholder historical bug",
            fix_summary="[STUB] placeholder fix summary",
            similarity=0.0,
            source="corpus",
        )
    ][:top_k]
    # --------------------------------------------------------------

    return AgentState(
        similar_bugs=bugs,
        trace=[
            trace_entry(
                "similar_bug_agent",
                f"[STUB] {len(bugs)} similar bug(s)",
                query=issue.title,
            )
        ],
    )
