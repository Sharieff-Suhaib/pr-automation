"""Thin wrappers over the agents built by Members 1-3.

All the awkward integration lives here so `graph.py` stays readable:

* Member 1's repository agent sits in `src/agents/repository-agent/` -- a
  hyphenated directory, so it is not an importable package. Its modules also
  import each other flatly (`from language_detector import ...`), so the
  directory itself has to go on `sys.path`.
* Member 2's recommendation agent uses absolute `src.` imports, so the project
  root has to go on `sys.path` too.
* Members 2 and 3 return dataclasses; the graph state carries plain JSON-safe
  dicts, so they are converted here.

Every wrapper raises `AdapterError` on failure. Nodes catch it, record the
message in `state["errors"]`, and let the workflow continue with whatever it
has -- one unavailable dependency should degrade the run, not kill it.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../pr-automation
REPO_AGENT_DIR = PROJECT_ROOT / "src" / "agents" / "repository-agent"
REPOSITORIES_DIR = PROJECT_ROOT / "repositories"
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"
TEST_ENVS_DIR = PROJECT_ROOT / ".test_envs"
# How many patched working copies and cached test environments survive a cleanup.
KEEP_WORKSPACES = int(os.environ.get("AGENT_SWE_KEEP_WORKSPACES", "20"))
KEEP_TEST_ENVS = int(os.environ.get("AGENT_SWE_KEEP_TEST_ENVS", "5"))

DEFAULT_TOP_K = 5
# Sampling temperature per attempt; later attempts explore more. Past the end, the last value.
ATTEMPT_TEMPERATURES = (0.0, 0.4, 0.8)


class AdapterError(RuntimeError):
    """A Member 1-3 agent could not complete its part of the run."""


def _ensure_import_paths() -> None:
    """Put the project root and Member 1's flat module directory on sys.path."""
    for path in (str(PROJECT_ROOT), str(REPO_AGENT_DIR)):
        if path not in sys.path:
            sys.path.append(path)


# --------------------------------------------------------------------------
# Member 1 -- Repository Agent
# --------------------------------------------------------------------------

_embedder = None  # the sentence-transformers model is slow to load; load it once


def _get_embedder():
    global _embedder
    if _embedder is None:
        from embeddings import CodeEmbedder  # noqa: PLC0415 -- needs sys.path set up first

        _embedder = CodeEmbedder()
    return _embedder


def resolve_repository(repo_url: str) -> str:
    """Return a local path for `repo_url`, cloning it when it is a remote URL.

    A local directory is used as-is, which is what makes offline demos possible.
    """
    if not repo_url or not repo_url.strip():
        raise AdapterError("No repository URL or path was provided.")

    candidate = Path(repo_url).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())

    _ensure_import_paths()
    try:
        from clone_repo import clone_repository  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Repository agent is unavailable: {error}") from error

    REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)
    _discard_incomplete_clone(repo_url)

    try:
        return clone_repository(repo_url, base_path=str(REPOSITORIES_DIR))
    except Exception as error:  # GitPython raises several unrelated types
        raise AdapterError(f"Could not clone {repo_url}: {error}") from error


def _discard_incomplete_clone(repo_url: str) -> None:
    """Remove a leftover clone directory that has no `.git` inside it.

    `clone_repository` returns early when the destination exists, so an
    interrupted clone would otherwise be reused forever as an empty repository.
    """
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]

    destination = REPOSITORIES_DIR / name
    if destination.is_dir() and not (destination / ".git").exists():
        shutil.rmtree(destination, ignore_errors=True)


def analyze_repository(repo_url: str, issue: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Clone/locate the repo, index its code, and retrieve the chunks matching `issue`.

    Returns `{"repo_path", "relevant_code", "relevant_files", "language",
    "indexed_chunks"}`, where `relevant_code` holds Member 1's node dicts
    (file, language, type, name, start_line, end_line, code, distance).
    """
    repo_path = resolve_repository(repo_url)

    _ensure_import_paths()
    try:
        from code_parser import extract_code_nodes, get_source_files  # noqa: PLC0415
        from vector_store import VectorStore  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Repository agent is unavailable: {error}") from error

    source_files = get_source_files(repo_path)

    nodes: list[dict[str, Any]] = []
    for source_file in source_files:
        try:
            nodes.extend(extract_code_nodes(source_file, repo_path))
        except Exception:
            # One unparseable file must not abandon the whole repository.
            continue

    # Classes are dropped: their bodies duplicate the methods already indexed,
    # which would otherwise crowd out the real matches.
    chunks = [node for node in nodes if node.get("type") != "class"]
    if not chunks:
        return {
            "repo_path": repo_path,
            "relevant_code": [],
            "relevant_files": [],
            "language": "",
            "indexed_chunks": 0,
        }

    embedder = _get_embedder()
    embeddings = embedder.embed_nodes(chunks)

    store = VectorStore(embeddings.shape[1])
    store.add(embeddings, chunks)

    results = store.search(embedder.embed_query(issue), top_k=min(top_k, len(chunks)))

    return {
        "repo_path": repo_path,
        "relevant_code": results,
        "relevant_files": _unique([result["file"] for result in results]),
        "language": results[0]["language"] if results else "",
        "indexed_chunks": len(chunks),
    }


# --------------------------------------------------------------------------
# Member 2 -- Recommendation Agent
# --------------------------------------------------------------------------


def recommend(issue: str, code: str = "", language: str = "python", top_k: int = 3) -> dict[str, Any]:
    """Run Member 2's pipeline and flatten its dataclasses into plain dicts."""
    _ensure_import_paths()
    try:
        from src.agents.recommendation_agent import recommend as run_recommendation  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Recommendation agent is unavailable: {error}") from error

    result = run_recommendation(issue, code=code, language=language or "python", top_k=top_k)

    return {
        "similar_bugs": [
            {
                "issue": bug.issue,
                "fix": bug.fix,
                "similarity": bug.similarity,
                "source": bug.source,
            }
            for bug in result.similar_bugs
        ],
        "bug_type": result.strategy.bug_type,
        "strategy": result.strategy.strategy,
        "strategy_source": result.strategy.source,
        "tools": list(result.tools),
        "tests": list(result.tests),
    }


# --------------------------------------------------------------------------
# Member 3 -- Coding Agent (patch generation)
# --------------------------------------------------------------------------


def generate_patch(
    issue: str,
    relevant_code: str,
    similar_bugs: list[dict[str, Any]],
    strategy: str,
    tools: list[str],
    tests: list[str],
    backend: str = "ollama",
    stub_patch_path: str = "",
    repo_path: str = "",
    relevant_chunks: list[dict[str, Any]] | None = None,
    feedback: str = "",
    attempt: int = 1,
) -> tuple[str, str]:
    """Return `(unified_diff, source)` for the repair.

    `backend="stub"` reads a fixture diff from `stub_patch_path` instead of
    calling the model. It exists so the whole graph can be exercised on a
    machine without Ollama -- the returned patch is a checked-in fixture, not a
    generated repair, and the caller records that in `patch_source`.

    With the model, the function mode runs first: the model returns corrected
    functions and the diff is computed locally, so it always applies (source
    `ollama-function`). If that yields nothing usable, the model is asked to
    write the diff itself (source `ollama`).

    `feedback` (retry loop) says why the previous attempt was rejected. The
    diff mode's interface has no slot for it, so it is appended to the issue.
    From `attempt` 2 on, the function mode samples at a rising temperature
    (`ATTEMPT_TEMPERATURES`): at temperature 0 a local model tends to return
    the rejected patch verbatim.
    """
    if backend == "stub":
        return _load_stub_patch(stub_patch_path), "stub"

    _ensure_import_paths()
    try:
        from src.agents.coding_agent import generate_patch as run_codegen  # noqa: PLC0415
        from src.agents.coding_agent.code_generator import PatchGenerationError  # noqa: PLC0415
        from src.agents.coding_agent.function_patch import generate_function_patch  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Coding agent is unavailable: {error}") from error

    function_error = ""
    if repo_path and relevant_chunks:
        try:
            patch = generate_function_patch(
                issue=issue,
                repo_path=repo_path,
                chunks=relevant_chunks,
                similar_bugs=[f"{bug['issue']} -> {bug['fix']}" for bug in similar_bugs],
                strategy=strategy,
                tests=tests,
                feedback=feedback,
                temperature=ATTEMPT_TEMPERATURES[min(attempt, len(ATTEMPT_TEMPERATURES)) - 1],
            )
            return patch, "ollama-function"
        except PatchGenerationError as error:
            function_error = str(error)

    try:
        patch = run_codegen(
            issue=f"{issue}\n\nPREVIOUS ATTEMPT:\n{feedback}" if feedback else issue,
            relevant_code=relevant_code,
            similar_bugs=[f"{bug['issue']} -> {bug['fix']}" for bug in similar_bugs],
            strategy=strategy,
            tools=tools,
            tests=tests,
        )
    except PatchGenerationError as error:
        if function_error:
            raise AdapterError(f"function mode: {function_error}; diff mode: {error}") from error
        raise AdapterError(str(error)) from error

    return patch, "ollama"


def _load_stub_patch(stub_patch_path: str) -> str:
    path = Path(stub_patch_path) if stub_patch_path else Path()
    if not path.is_file():
        raise AdapterError(f"Stub patch file not found: {stub_patch_path or '(not set)'}")
    return path.read_text(encoding="utf-8")


def format_relevant_code(relevant_code: list[dict[str, Any]], limit: int = 3) -> str:
    """Render retrieved chunks for the repair prompt, keeping the file paths visible.

    The model has to emit `--- a/<file>` headers, so the path of each chunk is
    stated right above its code. Test files are left out whenever any source
    chunk was retrieved: shown a test, the model tends to rewrite the test
    instead of the code (and small models then loop on repeated test hunks).
    """
    source_chunks = [chunk for chunk in relevant_code if not _is_test_file(chunk["file"])]
    blocks = []
    for chunk in (source_chunks or relevant_code)[:limit]:
        blocks.append(
            f"File: {chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']}, "
            f"{chunk['type']} {chunk['name']})\n{chunk['code']}"
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Testing Agent (baseline run, patched run, before/after verdict)
# --------------------------------------------------------------------------


def prepare_test_environment(repo_path: str, enabled: bool = True) -> dict[str, Any]:
    """Clean up old workspaces and environments, then build or reuse this repo's test environment.

    Returns `TestEnvironment.to_dict()` plus `removed`, the directories deleted.
    """
    _ensure_import_paths()
    try:
        from src.agents.testing_agent.environment import (  # noqa: PLC0415
            cleanup_directories,
            prepare_environment,
        )
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Testing agent is unavailable: {error}") from error

    removed = cleanup_directories(WORKSPACES_DIR, keep=KEEP_WORKSPACES)
    removed += cleanup_directories(TEST_ENVS_DIR, keep=KEEP_TEST_ENVS)
    environment = prepare_environment(repo_path, TEST_ENVS_DIR, enabled=enabled).to_dict()
    return {**environment, "removed": removed}


def run_baseline_tests(repo_path: str, language: str = "", python: str | None = None) -> dict[str, Any]:
    """Run the suite on a throwaway copy of the unpatched repository."""
    _ensure_import_paths()
    try:
        from src.agents.testing_agent import run_baseline  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Testing agent is unavailable: {error}") from error

    return run_baseline(repo_path, WORKSPACES_DIR, language=language, python=python)


def prepare_reproduction(
    repo_path: str,
    issue: str,
    relevant_code: list[dict[str, Any]],
    language: str = "",
    backend: str = "ollama",
    stub_repro_path: str = "",
    python: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write and validate a test reproducing the issue; return `(reproduction, baseline)`.

    `backend="stub"` replays the test at `stub_repro_path` instead of calling the
    model, mirroring `generate_patch`. The baseline is always returned, with the
    reproduction test included only when it was accepted.
    """
    _ensure_import_paths()
    try:
        from src.agents.testing_agent import (  # noqa: PLC0415
            ollama_writer,
            prepare_reproduction as run_reproduction,
            stub_writer,
        )
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Testing agent is unavailable: {error}") from error

    write = stub_writer(stub_repro_path) if backend == "stub" else ollama_writer()
    reproduction, baseline = run_reproduction(
        repo_path, issue, relevant_code, WORKSPACES_DIR, write, language=language, python=python
    )
    reproduction["source_of"] = "stub" if backend == "stub" else "ollama"
    return reproduction, baseline


def apply_and_test(
    repo_path: str,
    patch: str,
    language: str = "",
    baseline: dict[str, Any] | None = None,
    reproduction: dict[str, Any] | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Apply `patch` to a fresh copy of the repo, run the suite, and compare with `baseline`.

    The original clone is never modified: every run gets its own timestamped
    working copy under `workspaces/`. Recommended test *names* are suggestions,
    not pytest node ids, so the whole suite is run. An accepted `reproduction`
    test is added to the copy, as it was for the baseline.
    """
    _ensure_import_paths()
    try:
        from src.agents.testing_agent import evaluate_patch  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Testing agent is unavailable: {error}") from error

    accepted = bool(reproduction) and reproduction.get("status") == "accepted"
    return evaluate_patch(
        repo_path,
        patch,
        WORKSPACES_DIR,
        language=language,
        baseline=baseline,
        extra_files={reproduction["path"]: reproduction["source"]} if accepted else None,
        reproduction_tests=reproduction["tests"] if accepted else None,
        python=python,
    )


def attempt_feedback(attempt: dict[str, Any], earlier_patches: list[str] | None = None) -> str:
    """Why a tested attempt was rejected, phrased for the coding agent's next try."""
    _ensure_import_paths()
    from src.agents.testing_agent.feedback import build_feedback  # noqa: PLC0415

    return build_feedback(
        attempt["attempt"],
        attempt["patch"],
        attempt["patch_status"],
        attempt["test_result"],
        attempt["test_verdict"],
        patch_error=attempt.get("patch_error", ""),
        repeated=bool(attempt["patch"].strip()) and attempt["patch"] in (earlier_patches or []),
    )


# The Reflection Agent's model: None uses the coding agent's Ollama model. Tests
# replace it with a fake callable (prompt -> reply).
REFLECTION_LLM = None


def reflect(
    issue: str,
    relevant_code: str,
    attempt: dict[str, Any],
    strategy: str = "",
    previous_attempts: list[dict[str, Any]] | None = None,
    max_attempts: int = 3,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run the Reflection Agent on one tested attempt; returns `ReflectionResult.to_dict()`
    plus `feedback`, the text the coding agent receives next."""
    _ensure_import_paths()
    try:
        from src.agents.reflection_agent import ReflectionAgent  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Reflection agent is unavailable: {error}") from error

    result = ReflectionAgent(llm=REFLECTION_LLM, use_llm=use_llm).reflect(
        issue=issue,
        relevant_code=relevant_code,
        generated_patch=attempt.get("patch", ""),
        test_results=attempt,
        repair_strategy=strategy,
        previous_attempts=previous_attempts,
        attempt=attempt.get("attempt", 1),
        max_attempts=max_attempts,
    )
    return {**result.to_dict(), "feedback": result.as_feedback()}


def is_better_attempt(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    """Whether `candidate` should replace `current` as the attempt to report."""
    _ensure_import_paths()
    from src.agents.testing_agent.feedback import is_better  # noqa: PLC0415

    return is_better(candidate, current)


def _is_test_file(file_path: str) -> bool:
    name = Path(file_path).name
    return name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts"))


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def ollama_host() -> str:
    """The Ollama endpoint the coding agent will talk to (for status reporting)."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")
