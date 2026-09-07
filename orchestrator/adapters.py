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
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../pr-automation
REPO_AGENT_DIR = PROJECT_ROOT / "src" / "agents" / "repository-agent"
REPOSITORIES_DIR = PROJECT_ROOT / "repositories"
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"

DEFAULT_TOP_K = 5


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
) -> tuple[str, str]:
    """Return `(unified_diff, source)` for the repair.

    `backend="stub"` reads a fixture diff from `stub_patch_path` instead of
    calling the model. It exists so the whole graph can be exercised on a
    machine without Ollama -- the returned patch is a checked-in fixture, not a
    generated repair, and the caller records that in `patch_source`.
    """
    if backend == "stub":
        return _load_stub_patch(stub_patch_path), "stub"

    _ensure_import_paths()
    try:
        from src.agents.coding_agent import generate_patch as run_codegen  # noqa: PLC0415
        from src.agents.coding_agent.code_generator import PatchGenerationError  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Coding agent is unavailable: {error}") from error

    try:
        patch = run_codegen(
            issue=issue,
            relevant_code=relevant_code,
            similar_bugs=[f"{bug['issue']} -> {bug['fix']}" for bug in similar_bugs],
            strategy=strategy,
            tools=tools,
            tests=tests,
        )
    except PatchGenerationError as error:
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
    stated right above its code.
    """
    blocks = []
    for chunk in relevant_code[:limit]:
        blocks.append(
            f"File: {chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']}, "
            f"{chunk['type']} {chunk['name']})\n{chunk['code']}"
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Member 3 -- Testing Agent (apply the patch, run the suite)
# --------------------------------------------------------------------------


def apply_and_test(repo_path: str, patch: str, language: str = "") -> dict[str, Any]:
    """Copy the repo, apply `patch` to the copy, and run its native test suite.

    The original clone is never modified: `apply_patch` refuses to write into an
    existing destination, so every run gets its own timestamped working copy.
    """
    _ensure_import_paths()
    try:
        from src.agents.coding_agent.patch_manager import apply_patch  # noqa: PLC0415
        from src.agents.coding_agent.tester import run_tests  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Coding agent is unavailable: {error}") from error

    working_repo = _new_workspace(repo_path)
    patch_result = apply_patch(repo_path, working_repo, patch)

    if patch_result.status != "APPLIED":
        return {
            "working_repo": patch_result.working_repo,
            "changed_files": [],
            "patch_status": patch_result.status,
            "test_result": {
                "status": "SKIPPED",
                "passed": 0,
                "failed": 0,
                "errors": [patch_result.error or "The patch could not be applied."],
                "output": "",
            },
        }

    # Recommended test *names* are suggestions, not pytest node ids, so the whole
    # suite is run instead; passing them through would make pytest fail to collect.
    test_result = run_tests(patch_result.working_repo, language=language or None)

    return {
        "working_repo": patch_result.working_repo,
        "changed_files": patch_result.changed_files,
        "patch_status": patch_result.status,
        "test_result": test_result.to_dict(),
    }


def run_baseline_tests(repo_path: str, language: str = "") -> dict[str, Any]:
    """Run the suite on the unpatched repository, for when no patch was produced."""
    _ensure_import_paths()
    try:
        from src.agents.coding_agent.tester import run_tests  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AdapterError(f"Coding agent is unavailable: {error}") from error

    return run_tests(repo_path, language=language or None).to_dict()


def _new_workspace(repo_path: str) -> Path:
    """A fresh, non-existent directory outside the source repository."""
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return WORKSPACES_DIR / f"{Path(repo_path).name}-{stamp}"


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
