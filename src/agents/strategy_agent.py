"""Repair Strategy Recommendation Agent.

Answers: *what kind of fix does this issue call for?*

Classifies the issue into a repair category, which is injected into the repair
prompt to steer generation. `recommend` below delegates to
recommendation_agent.strategy_recommender, a structured LLM prompt; the
keyword-based `classify`/STRATEGIES in this module are its fallback when the
LLM is unavailable or returns an unparseable label.
"""

from __future__ import annotations

from src.agents.state import AgentState, trace_entry

# Categories from the project brief, with the guidance handed to the repair model.
STRATEGIES: dict[str, str] = {
    "null_reference": "Add null/None validation before dereferencing.",
    "boundary_error": "Correct the boundary condition or index range.",
    "logic_error": "Fix the conditional logic or operator.",
    "input_validation": "Validate and sanitize the input before use.",
    "performance": "Optimize the algorithm or data structure.",
    "exception_handling": "Add or correct exception handling.",
    "unknown": "Analyze the code and apply the minimal correct fix.",
}

# Cheap lexical priors. Deliberately conservative: a wrong strategy actively
# misleads the repair model, so ambiguous issues fall through to "unknown".
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("null_reference", ("none", "null", "nonetype", "attributeerror")),
    ("boundary_error", ("index", "range", "off by one", "out of bounds", "indexerror", "empty")),
    ("input_validation", ("invalid input", "validation", "malformed", "sanitize")),
    ("performance", ("slow", "performance", "timeout", "memory", "optimize")),
    ("exception_handling", ("exception", "traceback", "crash", "raises", "error handling")),
    ("logic_error", ("incorrect", "wrong", "unexpected", "should return", "logic")),
]


def classify(text: str) -> str:
    """Pick a repair category from issue text using keyword priors."""
    lowered = text.lower()
    for strategy, keywords in _KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return strategy
    return "unknown"


def recommend(state: AgentState) -> AgentState:
    """Recommend a repair strategy for the issue."""
    # Deferred import: strategy_recommender imports `classify`/STRATEGIES from
    # this module for its own fallback, so importing it at module load time
    # here would be circular.
    from src.agents.recommendation_agent.strategy_recommender import recommend as recommend_strategy

    issue_text = state["issue"].as_text()
    code = "\n\n".join(chunk.source for chunk in state.get("relevant_chunks", []))
    result = recommend_strategy(issue_text, code=code, similar_bugs=state.get("similar_bugs", []))

    return AgentState(
        repair_strategy=result.bug_type,
        strategy_rationale=result.strategy,
        trace=[trace_entry("strategy_agent", f"strategy={result.bug_type} (source={result.source})")],
    )
