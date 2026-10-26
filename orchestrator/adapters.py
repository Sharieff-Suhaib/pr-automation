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

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_AGENT_DIR = PROJECT_ROOT / "src" / "agents" / "repository-agent"
REPOSITORIES_DIR = PROJECT_ROOT / "repositories"
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"
TEST_ENVS_DIR = PROJECT_ROOT / ".test_envs"

KEEP_WORKSPACES = int(os.environ.get("AGENT_SWE_KEEP_WORKSPACES", "20"))
KEEP_TEST_ENVS = int(os.environ.get("AGENT_SWE_KEEP_TEST_ENVS", "5"))

DEFAULT_TOP_K = 5

ATTEMPT_TEMPERATURES = (0.4, 0.6, 0.8)

REPAIR_MODEL_URL = os.environ.get(
    "REPAIR_MODEL_URL",
    "https://marsupial-abnormal-unselect.ngrok-free.dev",
).rstrip("/")

REMOTE_TIMEOUT_SECONDS = (
    float(os.environ["REPAIR_MODEL_TIMEOUT"])
    if os.environ.get("REPAIR_MODEL_TIMEOUT")
    else None
)

BUILD_FILE_LANGUAGES = (
    ("pom.xml", "java"),
    ("build.gradle", "java"),
    ("build.gradle.kts", "java"),
    ("package.json", "javascript"),
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
    ("CMakeLists.txt", "cpp"),
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("setup.cfg", "python"),
    ("pytest.ini", "python"),
    ("tox.ini", "python"),
    ("requirements.txt", "python"),
)


class AdapterError(RuntimeError):
    """A Member 1-3 agent could not complete its part of the run."""


def _ensure_import_paths() -> None:
    """Add project and legacy flat-import directories used by the agents."""

    for path in (
        str(PROJECT_ROOT),
        str(REPO_AGENT_DIR),
        str(PROJECT_ROOT / "src" / "agents" / "coding_agent"),
    ):
        if path not in sys.path:
            sys.path.append(path)


# --------------------------------------------------------------------------
# Member 1 -- Repository Agent
# --------------------------------------------------------------------------

_embedder = None


def _get_embedder():
    global _embedder

    if _embedder is None:
        from embeddings import CodeEmbedder

        _embedder = CodeEmbedder()

    return _embedder


def resolve_repository(repo_url: str) -> str:
    """Return a local path for `repo_url`, cloning it when it is remote."""

    if not repo_url or not repo_url.strip():
        raise AdapterError("No repository URL or path was provided.")

    candidate = Path(repo_url).expanduser()

    if candidate.is_dir():
        return str(candidate.resolve())

    _ensure_import_paths()

    try:
        from clone_repo import clone_repository
    except ImportError as error:
        raise AdapterError(f"Repository agent is unavailable: {error}") from error

    REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)
    _discard_incomplete_clone(repo_url)

    try:
        print(f"Cloning {repo_url} into {REPOSITORIES_DIR}...")
        return clone_repository(repo_url, base_path=str(REPOSITORIES_DIR))
    except Exception as error:
        raise AdapterError(f"Could not clone {repo_url}: {error}") from error


def _discard_incomplete_clone(repo_url: str) -> None:
    """Remove a leftover clone directory that has no `.git` inside it."""

    name = repo_url.rstrip("/").split("/")[-1]

    if name.endswith(".git"):
        name = name[:-4]

    destination = REPOSITORIES_DIR / name

    if destination.is_dir() and not (destination / ".git").exists():
        shutil.rmtree(destination, ignore_errors=True)


def analyze_repository(
    repo_url: str,
    issue: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Clone or locate a repository and retrieve relevant source-code chunks."""

    repo_path = resolve_repository(repo_url)
    search_query = issue.strip()

    if not search_query:
        search_query = (
            "Find potentially faulty, incomplete, suspicious, or buggy "
            "code in this repository. Prioritize TODO comments, commented "
            "out code, missing return statements, incorrect conditions, "
            "unhandled errors, and incomplete functions."
        )

    _ensure_import_paths()

    try:
        from code_parser import extract_code_nodes, get_source_files
        from vector_store import VectorStore
    except ImportError as error:
        raise AdapterError(
            f"Repository agent is unavailable: {error}"
        ) from error

    source_files = get_source_files(repo_path)
    nodes: list[dict[str, Any]] = []

    for source_file in source_files:
        try:
            nodes.extend(
                extract_code_nodes(
                    source_file,
                    repo_path,
                )
            )
        except Exception:
            continue

    chunks = [
        node
        for node in nodes
        if node.get("type") != "class"
    ]

    if not chunks:
        return {
            "repo_path": repo_path,
            "relevant_code": [],
            "relevant_files": [],
            "language": "",
            "indexed_chunks": 0,
            "issue": search_query,
        }

    embedder = _get_embedder()
    embeddings = embedder.embed_nodes(chunks)

    store = VectorStore(embeddings.shape[1])
    store.add(embeddings, chunks)

    results = store.search(
        embedder.embed_query(search_query),
        top_k=min(top_k, len(chunks)),
    )

    return {
        "repo_path": repo_path,
        "relevant_code": results,
        "relevant_files": _unique(
            [
                result["file"]
                for result in results
                if result.get("file")
            ]
        ),
        "language": _repository_language(
            repo_path,
            results,
        ),
        "indexed_chunks": len(chunks),
        "issue": search_query,
    }


# --------------------------------------------------------------------------
# Member 2 -- Recommendation Agent
# --------------------------------------------------------------------------


def recommend(
    issue: str,
    code: str = "",
    language: str = "python",
    top_k: int = 3,
) -> dict[str, Any]:
    """Run Member 2's pipeline and flatten its dataclasses into plain dicts."""

    _ensure_import_paths()

    try:
        print(f"Running recommendation agent for issue: {issue}...")
        from src.agents.recommendation_agent import recommend as run_recommendation
    except ImportError as error:
        raise AdapterError(
            f"Recommendation agent is unavailable: {error}"
        ) from error

    result = run_recommendation(
        issue,
        code=code,
        language=language or "python",
        top_k=top_k,
    )

    if result is None:
        raise AdapterError("Recommendation agent returned no result.")

    print(f"Recommendation result: {result}")

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
# Member 3 -- Coding Agent
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
    """Return ``(unified_diff, source)`` for the repair.

    Order:

    1. Remote IR4/OR2 model.
    2. Function mode for Python and Java.
    3. Python fault localization.
    4. Generic diff mode.
    """

    if backend == "stub":
        return _load_stub_patch(stub_patch_path), "stub"

    _ensure_import_paths()

    try:
        from src.agents.coding_agent import generate_patch as run_codegen
        from src.agents.coding_agent.code_generator import PatchGenerationError
        from src.agents.coding_agent.function_patch import (
            generate_function_patch,
        )
    except ImportError as error:
        raise AdapterError(
            f"Coding agent is unavailable: {error}"
        ) from error

    temperature = ATTEMPT_TEMPERATURES[
        min(max(attempt, 1), len(ATTEMPT_TEMPERATURES)) - 1
    ]

    chunks = relevant_chunks or []
    languages = [
        str(chunk.get("language", "")).lower()
        for chunk in chunks
    ]

    # Prefer Java when retrieved chunks include Java source; otherwise use the
    # first detected language. This avoids sending Java methods to Python AST.
    language = (
        "java"
        if "java" in languages
        else languages[0]
        if languages
        else ""
    )

    remote_error = ""

    if REPAIR_MODEL_URL and repo_path and chunks:
        try:
            return remote_patch(
                repo_path,
                chunks,
                temperature,
            ), "remote-ir4"
        except PatchGenerationError as error:
            remote_error = str(error)
            print(
                "Remote model unusable; trying local fallback: "
                f"{remote_error[:200]}",
                flush=True,
            )
        except Exception as error:
            remote_error = f"Remote repair failed: {error}"
            print(
                f"{remote_error[:250]}; trying local fallback",
                flush=True,
            )

    # Python AST and Java tree-sitter function repair. The model returns
    # corrected functions and the diff is computed locally.
    function_error = ""

    # A reply that loops on one line is aborted by Ollama. One retry with a
    # higher temperature usually gets through.
    for heat in (
        (temperature, min(temperature + 0.3, 1.0))
        if repo_path and chunks
        else ()
    ):
        try:
            patch = generate_function_patch(
                issue=issue,
                repo_path=repo_path,
                chunks=chunks,
                similar_bugs=[
                    f"{bug['issue']} -> {bug['fix']}"
                    for bug in similar_bugs
                ],
                strategy=strategy,
                tests=tests,
                feedback=feedback,
                temperature=heat,
            )

            return patch, "ollama-function"

        except PatchGenerationError as error:
            function_error = str(error)
        except Exception as error:
            function_error = (
                f"Function-based repair failed: {error}"
            )

        if "repeat limit" not in function_error:
            break

    if language == "java":
        # Diff mode does not reliably produce usable Java patches. Fail fast
        # so the retry loop can try again instead.
        combined = [
            part
            for part in (
                remote_error,
                function_error,
            )
            if part
        ]

        raise AdapterError(
            "; ".join(combined)
            or "No Java function could be repaired."
        )

    # Repository-level generator for Python and other languages.
    fault_error = ""

    if repo_path:
        try:
            from src.agents.coding_agent import (
                generate_patch_from_repository,
            )

            patch = generate_patch_from_repository(
                repo_url=repo_path,
                issue=issue,
                similar_bugs=[
                    f"{bug['issue']} -> {bug['fix']}"
                    for bug in similar_bugs
                ],
                strategy=strategy,
                tools=tools,
                tests=tests,
                top_k=max(3, min(8, len(chunks) or 4)),
            )

            return patch, "ollama-fault-localization"

        except Exception as error:
            fault_error = str(error)

    # Generic diff generation is the final fallback.
    try:
        patch = run_codegen(
            issue=(
                f"{issue}\n\nPREVIOUS ATTEMPT:\n{feedback}"
                if feedback
                else issue
            ),
            relevant_code=relevant_code,
            similar_bugs=[
                f"{bug['issue']} -> {bug['fix']}"
                for bug in similar_bugs
            ],
            strategy=strategy,
            tools=tools,
            tests=tests,
        )
    except Exception as error:
        combined = [
            part
            for part in (
                remote_error,
                function_error,
                fault_error,
                str(error),
            )
            if part
        ]

        raise AdapterError(
            "; ".join(combined)
            if combined
            else str(error)
        ) from error

    return patch, "ollama"


def remote_patch(
    repo_path: str,
    chunks: list[dict[str, Any]],
    temperature: float = 0.0,
) -> str:
    """Repair the top-ranked Python or Java function with the remote model."""

    from src.agents.coding_agent.code_generator import (
        MAX_PATCH_TOKENS,
        PatchGenerationError,
    )
    from src.agents.coding_agent.function_patch import make_diff

    target = next(
        (
            chunk
            for chunk in chunks
            if chunk.get("language") in ("python", "java")
            and chunk.get("type") in ("function", "method")
            and chunk["end_line"]
            > chunk["start_line"]
            + (chunk["language"] == "java")
            and not _is_test_file(chunk["file"])
        ),
        None,
    )

    if target is None:
        raise PatchGenerationError(
            "No Python or Java function was retrieved "
            "for the remote model."
        )

    language = target["language"]
    file_path = target["file"]

    try:
        original = (
            Path(repo_path) / file_path
        ).read_text(encoding="utf-8")
    except OSError as error:
        raise PatchGenerationError(
            f"Cannot read {file_path}: {error}"
        ) from error

    lines = original.splitlines()

    head = target["start_line"] - 1
    start = target["start_line"]
    end = target["end_line"] - (language == "java")

    prompt = _ir4(
        lines,
        head,
        start,
        end,
        target["end_line"],
        file_path,
        language,
    )

    payload = {
        "language": language,
        "prompt": prompt,
        "max_new_tokens": min(MAX_PATCH_TOKENS, 2048),
        "temperature": temperature,
    }

    request = urllib.request.Request(
        f"{REPAIR_MODEL_URL}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "1",
        },
    )

    try:
        print(
            f"POST {REPAIR_MODEL_URL}/generate: "
            f"language={language}, "
            f"{target['name']} in {file_path}, "
            f"prompt {len(prompt)} chars, "
            f"max_new_tokens={payload['max_new_tokens']}, "
            f"temperature={temperature}; "
            "waiting for the reply...",
            flush=True,
        )

        started = time.monotonic()

        with urllib.request.urlopen(
            request,
            timeout=REMOTE_TIMEOUT_SECONDS,
        ) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

        if isinstance(response_data, str):
            reply = response_data
        elif isinstance(response_data, dict):
            reply = next(
                (
                    response_data[key]
                    for key in (
                        "response",
                        "generated_text",
                        "text",
                        "output",
                    )
                    if isinstance(response_data.get(key), str)
                ),
                None,
            )

            if reply is None:
                raise ValueError(
                    "Model API JSON must contain a string field named "
                    "response, generated_text, text, or output; "
                    f"received keys: {list(response_data.keys())}"
                )
        else:
            raise ValueError(
                "Unexpected model API response type: "
                f"{type(response_data).__name__}"
            )

    except urllib.error.HTTPError as error:
        detail = error.read().decode(
            "utf-8",
            errors="replace",
        )[:300]

        raise PatchGenerationError(
            f"Remote model returned HTTP {error.code}: {detail}"
        ) from error

    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        KeyError,
    ) as error:
        raise PatchGenerationError(
            f"Remote model at {REPAIR_MODEL_URL} failed: {error}"
        ) from error

    print(
        f"Remote model replied in "
        f"{time.monotonic() - started:.0f}s: "
        f"{reply[:300]!r}",
        flush=True,
    )

    fixed = reply.strip("\n").rstrip().splitlines()

    if not fixed:
        raise PatchGenerationError(
            "The remote model returned an empty repair: it ended "
            "generation at once, which suggests the IR4 prompt does "
            "not match the format the adapter was trained on."
        )

    expected = next(
        (
            len(line) - len(line.lstrip())
            for line in lines[start:end]
            if line.strip()
        ),
        0,
    )

    if (
        len(fixed[0]) - len(fixed[0].lstrip())
        == expected - 1
    ):
        fixed[0] = " " + fixed[0]

    repaired = "\n".join(
        lines[:start]
        + fixed
        + lines[end:]
    ) + (
        "\n"
        if original.endswith("\n")
        else ""
    )

    if repaired == original:
        raise PatchGenerationError(
            "The remote model returned the function unchanged."
        )

    if language == "python":
        try:
            compile(
                repaired,
                file_path,
                "exec",
            )
        except SyntaxError as error:
            raise PatchGenerationError(
                f"The repaired {file_path} is not valid Python: "
                f"{error.msg} (line {error.lineno})."
            ) from error

    return make_diff(
        file_path,
        original,
        repaired,
    )


def _ir4(
    lines: list[str],
    head: int,
    start: int,
    end: int,
    stop: int,
    file_path: str,
    language: str,
) -> str:
    """Render the IR4 prompt for the selected function body."""

    if language == "java":
        package = str(
            PROJECT_ROOT / "src" / "repairLlama" / "src"
        )

        if package not in sys.path:
            sys.path.append(package)

        from repairllama.representation.ir4 import (
            describe_region,
            render_ir4,
        )

        return render_ir4(
            lines,
            describe_region(
                lines,
                start + 1,
                end,
                file_path=file_path,
            ),
        )

    commented = [
        f"# {line}" if line else "#"
        for line in lines[start:end]
    ]

    return "\n".join(
        [
            *lines[head:start],
            "# BUGGY CODE",
            *commented,
            "<FILL_ME>",
            *lines[end:stop],
        ]
    )


def _load_stub_patch(stub_patch_path: str) -> str:
    path = (
        Path(stub_patch_path)
        if stub_patch_path
        else Path()
    )

    if not path.is_file():
        raise AdapterError(
            "Stub patch file not found: "
            f"{stub_patch_path or '(not set)'}"
        )

    return path.read_text(encoding="utf-8")


def format_relevant_code(
    relevant_code: list[dict[str, Any]],
    limit: int = 3,
) -> str:
    """Render retrieved chunks for the repair prompt."""

    source_chunks = [
        chunk
        for chunk in relevant_code
        if not _is_test_file(chunk["file"])
    ]

    blocks = []

    for chunk in (
        source_chunks or relevant_code
    )[:limit]:
        blocks.append(
            f"File: {chunk['file']} "
            f"(lines {chunk['start_line']}-{chunk['end_line']}, "
            f"{chunk['type']} {chunk['name']})\n"
            f"{chunk['code']}"
        )

    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Testing Agent
# --------------------------------------------------------------------------


def prepare_test_environment(
    repo_path: str,
    enabled: bool = True,
) -> dict[str, Any]:
    """Clean up old workspaces and prepare the repository test environment."""

    _ensure_import_paths()

    try:
        from src.agents.testing_agent.environment import (
            cleanup_directories,
            prepare_environment,
        )
    except ImportError as error:
        raise AdapterError(
            f"Testing agent is unavailable: {error}"
        ) from error

    removed = cleanup_directories(
        WORKSPACES_DIR,
        keep=KEEP_WORKSPACES,
    )

    removed += cleanup_directories(
        TEST_ENVS_DIR,
        keep=KEEP_TEST_ENVS,
    )

    environment = prepare_environment(
        repo_path,
        TEST_ENVS_DIR,
        enabled=enabled,
    ).to_dict()

    return {
        **environment,
        "removed": removed,
    }


def run_baseline_tests(
    repo_path: str,
    language: str = "",
    python: str | None = None,
) -> dict[str, Any]:
    """Run the suite on a throwaway copy of the repository."""

    _ensure_import_paths()

    try:
        from src.agents.testing_agent import run_baseline
    except ImportError as error:
        raise AdapterError(
            f"Testing agent is unavailable: {error}"
        ) from error

    return run_baseline(
        repo_path,
        WORKSPACES_DIR,
        language=language,
        python=python,
    )


def prepare_reproduction(
    repo_path: str,
    issue: str,
    relevant_code: list[dict[str, Any]],
    language: str = "",
    backend: str = "ollama",
    stub_repro_path: str = "",
    python: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write and validate a test reproducing the issue."""

    _ensure_import_paths()

    try:
        from src.agents.testing_agent import (
            ollama_writer,
            prepare_reproduction as run_reproduction,
            stub_writer,
        )
    except ImportError as error:
        raise AdapterError(
            f"Testing agent is unavailable: {error}"
        ) from error

    write = (
        stub_writer(stub_repro_path)
        if backend == "stub"
        else ollama_writer()
    )

    reproduction, baseline = run_reproduction(
        repo_path,
        issue,
        relevant_code,
        WORKSPACES_DIR,
        write,
        language=language,
        python=python,
    )

    reproduction["source_of"] = (
        "stub"
        if backend == "stub"
        else "ollama"
    )

    return reproduction, baseline


def apply_and_test(
    repo_path: str,
    patch: str,
    language: str = "",
    baseline: dict[str, Any] | None = None,
    reproduction: dict[str, Any] | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    """Apply a patch to a fresh copy of the repository and run the suite."""

    _ensure_import_paths()

    try:
        from src.agents.testing_agent import evaluate_patch
    except ImportError as error:
        raise AdapterError(
            f"Testing agent is unavailable: {error}"
        ) from error

    accepted = (
        bool(reproduction)
        and reproduction.get("status") == "accepted"
    )

    return evaluate_patch(
        repo_path,
        patch,
        WORKSPACES_DIR,
        language=language,
        baseline=baseline,
        extra_files=(
            {
                reproduction["path"]: reproduction["source"]
            }
            if accepted
            else None
        ),
        reproduction_tests=(
            reproduction["tests"]
            if accepted
            else None
        ),
        python=python,
    )


def attempt_feedback(
    attempt: dict[str, Any],
    earlier_patches: list[str] | None = None,
) -> str:
    """Explain why a tested attempt was rejected."""

    _ensure_import_paths()

    from src.agents.testing_agent.feedback import build_feedback

    return build_feedback(
        attempt["attempt"],
        attempt["patch"],
        attempt["patch_status"],
        attempt["test_result"],
        attempt["test_verdict"],
        patch_error=attempt.get("patch_error", ""),
        repeated=(
            bool(attempt["patch"].strip())
            and attempt["patch"] in (earlier_patches or [])
        ),
    )


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
    """Run the Reflection Agent on one tested attempt."""

    _ensure_import_paths()

    try:
        from src.agents.reflection_agent import ReflectionAgent
    except ImportError as error:
        raise AdapterError(
            f"Reflection agent is unavailable: {error}"
        ) from error

    result = ReflectionAgent(
        llm=REFLECTION_LLM,
        use_llm=use_llm,
    ).reflect(
        issue=issue,
        relevant_code=relevant_code,
        generated_patch=attempt.get("patch", ""),
        test_results=attempt,
        repair_strategy=strategy,
        previous_attempts=previous_attempts,
        attempt=attempt.get("attempt", 1),
        max_attempts=max_attempts,
    )

    return {
        **result.to_dict(),
        "feedback": result.as_feedback(),
    }


def is_better_attempt(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
) -> bool:
    """Whether `candidate` should replace `current`."""

    _ensure_import_paths()

    from src.agents.testing_agent.feedback import is_better

    return is_better(candidate, current)


# --------------------------------------------------------------------------
# Commit and push
# --------------------------------------------------------------------------


def commit_and_push(
    repo_path: str,
    patch: str,
    issue: str = "",
) -> dict[str, str]:
    """Commit `patch` on a new branch and push that branch."""

    from orchestrator.github_issues import load_token

    if (
        Path(
            _git(
                repo_path,
                "rev-parse",
                "--show-toplevel",
            )
        ).resolve()
        != Path(repo_path).resolve()
    ):
        raise AdapterError(
            f"{repo_path} is not the root of its own git repository; "
            "nothing to push."
        )

    token = load_token()
    remote = _git(
        repo_path,
        "remote",
        "get-url",
        "origin",
    )

    push_url = (
        remote.replace(
            "https://",
            f"https://x-access-token:{token}@",
            1,
        )
        if token and remote.startswith("https://")
        else remote
    )

    branch = (
        "agent-swe/fix-"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )

    title = (
        issue.strip().splitlines()
        or ["repair"]
    )[0][:72]

    with tempfile.TemporaryDirectory() as tmp:
        worktree = str(Path(tmp) / "worktree")

        _git(
            repo_path,
            "worktree",
            "add",
            "-b",
            branch,
            worktree,
            "HEAD",
        )

        try:
            _git(
                worktree,
                "apply",
                "--whitespace=nowarn",
                stdin=patch,
            )

            _git(worktree, "add", "-A")

            _git(
                worktree,
                "commit",
                "-m",
                (
                    f"Fix: {title}\n\n"
                    "Generated by Agent-SWE, approved by a reviewer."
                ),
            )

            commit = _git(
                worktree,
                "rev-parse",
                "HEAD",
            )

            _git(
                worktree,
                "push",
                push_url,
                f"HEAD:refs/heads/{branch}",
                secret=token,
            )
        finally:
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "remove",
                    "--force",
                    worktree,
                ],
                cwd=repo_path,
                capture_output=True,
            )

    return {
        "branch": branch,
        "commit": commit,
        "remote": remote,
    }


def _git(
    cwd: str,
    *args: str,
    stdin: str | None = None,
    secret: str = "",
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
    )

    if result.returncode:
        message = (
            result.stderr
            or result.stdout
        ).strip()

        if secret:
            message = message.replace(secret, "***")

        raise AdapterError(
            f"git {args[0]} failed: {message}"
        )

    return result.stdout.strip()


def _repository_language(
    repo_path: str,
    results: list[dict[str, Any]],
) -> str:
    """Determine the language used for repair and test execution."""

    languages = [
        result["language"]
        for result in results
        if result.get("language")
    ]

    if not languages:
        return ""

    root = Path(repo_path)

    for marker, language in BUILD_FILE_LANGUAGES:
        if (
            language in languages
            and (root / marker).exists()
        ):
            return language

    counts = Counter(languages)

    return max(
        languages,
        key=lambda language: (
            counts[language],
            language == "python",
            -languages.index(language),
        ),
    )


def _is_test_file(file_path: str) -> bool:
    name = Path(file_path).name

    return name.startswith("test_") or name.endswith(
        (
            "_test.py",
            "Test.java",
            ".test.js",
            ".spec.js",
            ".test.ts",
            ".spec.ts",
        )
    )


is_test_file = _is_test_file


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def ollama_host() -> str:
    """Return the Ollama endpoint used by the coding agent."""

    return os.environ.get(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )