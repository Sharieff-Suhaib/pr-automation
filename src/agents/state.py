"""Shared state for the Agent-SWE workflow.

Every agent is a pure function `AgentState -> partial AgentState`. LangGraph
merges each returned dict into the running state, so an agent only returns the
keys it actually produced. That keeps agents independently testable: you can
call any one of them with a hand-built state and assert on its output without
running the graph.

The state doubles as the audit trail for the report — `trace` records what each
agent did, which is what you need when writing up why a repair succeeded or
failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict


@dataclass
class Issue:
    """The GitHub issue under repair."""

    title: str
    body: str = ""
    number: int | None = None
    repo_url: str | None = None

    def as_text(self) -> str:
        """Flattened form used for embedding and prompting."""
        return f"{self.title}\n\n{self.body}".strip()


@dataclass
class CodeChunk:
    """A function/class/method extracted from the repository.

    `start_line`/`end_line` are 1-indexed and inclusive so they can feed
    FaultLocation directly. `start_byte`/`end_byte` are what patch application
    needs to splice a repaired function back into its file.
    """

    file_path: str
    name: str
    kind: Literal["function", "method", "class", "module"]
    source: str
    start_line: int
    end_line: int
    start_byte: int = 0
    end_byte: int = 0
    score: float = 0.0  # similarity to the issue, filled by the retriever

    def qualified_name(self) -> str:
        return f"{self.file_path}::{self.name}"


@dataclass
class SimilarBug:
    """A historical bug-fix retrieved as precedent for the current issue."""

    issue: str
    fix_summary: str
    similarity: float
    buggy_code: str = ""
    fixed_code: str = ""
    source: str = "corpus"  # "corpus" (CommitPackFT) | "feedback" (past repairs)


@dataclass
class CandidatePatch:
    """One generated repair, plus the outcome of testing it."""

    code: str
    target_chunk: str  # qualified_name of the chunk this replaces
    attempt: int = 0
    passed: bool | None = None
    test_output: str = ""


@dataclass
class TestReport:
    """Result of executing a patch against the project's tests."""

    tests_passed: int = 0
    tests_failed: int = 0
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""

    @property
    def all_passed(self) -> bool:
        return self.tests_failed == 0 and not self.errors and self.tests_passed > 0


def append(left: list, right: list) -> list:
    """Reducer so concurrent nodes append to a list instead of overwriting it."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    """The blackboard all agents read from and write to.

    `total=False` means every key is optional — nodes early in the graph have
    not produced the later keys yet.
    """

    # --- Input ---
    issue: Issue
    repo_url: str
    repo_path: str

    # --- Repository Recommendation Agent ---
    relevant_files: list[str]
    relevant_chunks: list[CodeChunk]

    # --- Similar Bug Recommendation Agent ---
    similar_bugs: list[SimilarBug]

    # --- Strategy / Tool / Test Case Recommendation Agents ---
    repair_strategy: str
    strategy_rationale: str
    recommended_tools: list[str]
    recommended_tests: list[str]

    # --- Code Generation Agent ---
    candidate_patches: list[CandidatePatch]

    # --- Testing Agent ---
    test_report: TestReport

    # --- Reflection Agent ---
    reflection: str
    should_retry: bool

    # --- Control ---
    attempt: int
    max_attempts: int
    status: Literal["running", "solved", "failed", "exhausted"]

    # --- Bookkeeping ---
    # `append` reducer: each agent adds its own entries without clobbering others.
    trace: Annotated[list[dict[str, Any]], append]

    # Ablation switch: when False the recommendation sections are withheld from
    # the repair prompt, giving the "Issue -> LLM -> Patch" control arm.
    use_recommendations: bool


def new_state(
    issue: Issue,
    repo_url: str = "",
    repo_path: str = "",
    max_attempts: int = 3,
    use_recommendations: bool = True,
) -> AgentState:
    """Build the initial state for one repair run."""
    return AgentState(
        issue=issue,
        repo_url=repo_url or (issue.repo_url or ""),
        repo_path=repo_path,
        attempt=0,
        max_attempts=max_attempts,
        status="running",
        trace=[],
        use_recommendations=use_recommendations,
    )


def trace_entry(agent: str, summary: str, **details: Any) -> dict[str, Any]:
    """One structured line in the audit trail."""
    return {"agent": agent, "summary": summary, **details}
