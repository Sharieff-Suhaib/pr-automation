"""Repository Recommendation Agent.

Answers: *which files and functions in this repository are relevant to the issue?*

STAGE A: stub returning placeholder chunks so the graph runs end to end.
STAGE B/C will replace `recommend` internals with:
    clone (GitPython) -> AST chunk into functions/classes -> embed
    -> FAISS search against the issue text -> top-k chunks

The signature and return keys are already final, so downstream agents and the
localization evaluation do not change when the internals land.
"""

from __future__ import annotations

from src.agents.state import AgentState, CodeChunk, trace_entry

# How many chunks to hand downstream. Small: the repair model's context is the
# binding constraint, so precision matters more than recall here.
DEFAULT_TOP_K = 5


def recommend(state: AgentState, top_k: int = DEFAULT_TOP_K) -> AgentState:
    """Rank repository code chunks by relevance to the issue."""
    issue = state["issue"]

    # --- STAGE C will replace this block with FAISS retrieval ---
    placeholder = CodeChunk(
        file_path="src/example_module.py",
        name="example_function",
        kind="function",
        source="def example_function(items):\n    return items[0]\n",
        start_line=1,
        end_line=2,
        score=0.0,
    )
    chunks = [placeholder][:top_k]
    # ------------------------------------------------------------

    files = list(dict.fromkeys(chunk.file_path for chunk in chunks))
    return AgentState(
        relevant_files=files,
        relevant_chunks=chunks,
        trace=[
            trace_entry(
                "repository_agent",
                f"[STUB] {len(chunks)} chunk(s) across {len(files)} file(s)",
                issue=issue.title,
                files=files,
            )
        ],
    )
