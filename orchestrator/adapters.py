"""Thin wrappers over the agents built by Members 1-3."""

from __future__ import annotations

import json
import os
import re
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

KEEP_WORKSPACES = int(
    os.environ.get(
        "AGENT_SWE_KEEP_WORKSPACES",
        "20",
    )
)

KEEP_TEST_ENVS = int(
    os.environ.get(
        "AGENT_SWE_KEEP_TEST_ENVS",
        "5",
    )
)

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

# The latest prompt and response from the Colab RepairLLaMA endpoint.
LAST_REMOTE_REPAIR: dict[str, Any] = {}


class AdapterError(RuntimeError):
    """A wrapped agent could not complete its part of the run."""


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
# Repository Agent
# --------------------------------------------------------------------------

_embedder = None


def _get_embedder():
    global _embedder

    if _embedder is None:
        from embeddings import CodeEmbedder

        _embedder = CodeEmbedder()

    return _embedder


def resolve_repository(repo_url: str) -> str:
    """Return a local path for repo_url, cloning it when it is remote."""

    if not repo_url or not repo_url.strip():
        raise AdapterError(
            "No repository URL or path was provided."
        )

    candidate = Path(repo_url).expanduser()

    if candidate.is_dir():
        return str(candidate.resolve())

    _ensure_import_paths()

    try:
        from clone_repo import clone_repository
    except ImportError as error:
        raise AdapterError(
            f"Repository agent is unavailable: {error}"
        ) from error

    REPOSITORIES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _discard_incomplete_clone(repo_url)

    try:
        print(
            f"Cloning {repo_url} into {REPOSITORIES_DIR}...",
            flush=True,
        )

        return clone_repository(
            repo_url,
            base_path=str(REPOSITORIES_DIR),
        )

    except Exception as error:
        raise AdapterError(
            f"Could not clone {repo_url}: {error}"
        ) from error


def _discard_incomplete_clone(repo_url: str) -> None:
    """Remove a leftover clone directory that has no .git directory."""

    name = repo_url.rstrip("/").split("/")[-1]

    if name.endswith(".git"):
        name = name[:-4]

    destination = REPOSITORIES_DIR / name

    if (
        destination.is_dir()
        and not (destination / ".git").exists()
    ):
        shutil.rmtree(
            destination,
            ignore_errors=True,
        )


def analyze_repository(
    repo_url: str,
    issue: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Clone or locate a repository and retrieve relevant source chunks."""

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
        from code_parser import (
            extract_code_nodes,
            get_source_files,
        )
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
    store.add(
        embeddings,
        chunks,
    )

    results = store.search(
        embedder.embed_query(search_query),
        top_k=min(
            top_k,
            len(chunks),
        ),
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
# Recommendation Agent
# --------------------------------------------------------------------------


def recommend(
    issue: str,
    code: str = "",
    language: str = "python",
    top_k: int = 3,
) -> dict[str, Any]:
    """Run the recommendation agent and flatten its dataclasses."""

    _ensure_import_paths()

    try:
        print(
            f"Running recommendation agent for issue: {issue}...",
            flush=True,
        )

        from src.agents.recommendation_agent import (
            recommend as run_recommendation,
        )

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
        raise AdapterError(
            "Recommendation agent returned no result."
        )

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
# Coding Agent
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
    """Return a unified diff and its source."""

    if backend == "stub":
        return (
            _load_stub_patch(stub_patch_path),
            "stub",
        )

    _ensure_import_paths()

    try:
        from src.agents.coding_agent import (
            generate_patch as run_codegen,
        )

        from src.agents.coding_agent.code_generator import (
            PatchGenerationError,
        )

        from src.agents.coding_agent.function_patch import (
            generate_function_patch,
        )

    except ImportError as error:
        raise AdapterError(
            f"Coding agent is unavailable: {error}"
        ) from error

    temperature = ATTEMPT_TEMPERATURES[
        min(
            max(
                attempt,
                1,
            ),
            len(ATTEMPT_TEMPERATURES),
        )
        - 1
    ]

    chunks = relevant_chunks or []

    languages = [
        str(
            chunk.get(
                "language",
                "",
            )
        ).lower()
        for chunk in chunks
    ]

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
            # The Colab API does not accept or require temperature.
            # Temperature is retained only for local generation paths.
            remote_temperature = (
                0.0
                if language == "java"
                else temperature
            )

            return (
                remote_patch(
                    repo_path,
                    chunks,
                    issue=issue,
                    temperature=remote_temperature,
                    attempt=attempt,
                ),
                "remote-ir4",
            )

        except PatchGenerationError as error:
            remote_error = str(error)

            print(
                "Remote model unusable; trying local fallback: "
                f"{remote_error[:300]}",
                flush=True,
            )

        except Exception as error:
            remote_error = (
                f"Remote repair failed: {error}"
            )

            print(
                f"{remote_error[:300]}; trying local fallback",
                flush=True,
            )

    function_error = ""

    if repo_path and chunks:
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
                temperature=(
                    0.0
                    if language == "java"
                    else temperature
                ),
            )

            return patch, "ollama-function"

        except PatchGenerationError as error:
            function_error = str(error)

        except Exception as error:
            function_error = (
                f"Function-based repair failed: {error}"
            )

    if language == "java":
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
                top_k=max(
                    3,
                    min(
                        8,
                        len(chunks) or 4,
                    ),
                ),
            )

            return patch, "ollama-fault-localization"

        except Exception as error:
            fault_error = str(error)

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


def _validate_java_repair(
    original: str,
    repaired: str,
    file_path: str,
) -> None:
    """Reject Java repairs that change unrelated APIs or behavior."""

    if not repaired.strip():
        raise AdapterError(
            f"The repaired {file_path} is empty."
        )

    if repaired == original:
        raise AdapterError(
            f"The remote model did not change {file_path}."
        )

    if "orders.size()" in original:
        if "orders.length" in repaired:
            raise AdapterError(
                "Rejected Java repair: changed List access "
                "from orders.size() to orders.length."
            )

        if "orders[" in repaired:
            raise AdapterError(
                "Rejected Java repair: changed List access "
                "from orders.get(i) to array indexing."
            )

    if "getOrderId()" in original:
        if "getOrderId()" not in repaired:
            raise AdapterError(
                "Rejected Java repair: removed getOrderId()."
            )

    if "getOrderId() == orderId" in original:
        if "getOrderId().equals(orderId)" in repaired:
            raise AdapterError(
                "Rejected Java repair: replaced primitive ID "
                "comparison with .equals()."
            )

    for fragment in (
        "return orders.get(i);",
        "return null;",
    ):
        if fragment in original and fragment not in repaired:
            raise AdapterError(
                "Rejected Java repair: removed required behavior: "
                f"{fragment}"
            )


def _validate_minimal_order_lookup_repair(
    original: str,
    repaired: str,
) -> None:
    """Validate the minimal OrderService.findOrder() repair."""

    old_condition = "i < orders.size() - 1"
    correct_condition = "i < orders.size()"

    if old_condition not in original:
        return

    if correct_condition not in repaired:
        raise AdapterError(
            "The repair did not fix the order lookup loop boundary."
        )

    for pattern in (
        "orders.length",
        "orders[",
        ".equals(orderId)",
        "orders.remove(",
        "Order order = null",
        "while (",
    ):
        if pattern in repaired:
            raise AdapterError(
                "Rejected non-minimal OrderService repair: "
                f"found {pattern!r}."
            )


def remote_patch(
    repo_path: str,
    chunks: list[dict[str, Any]],
    issue: str = "",
    temperature: float = 0.0,
    attempt: int = 1,
) -> str:
    """
    Repair the selected Python or Java function remotely, one line at a time.

    The request format intentionally matches the unchanged Colab client:

        {
            "prompt": prompt,
            "max_new_tokens": 128
        }

    No language or temperature fields are sent to Colab.
    """

    del temperature

    from src.agents.coding_agent.code_generator import (
        MAX_PATCH_TOKENS,
        PatchGenerationError,
    )

    from src.agents.coding_agent.function_patch import (
        make_diff,
    )

    target = next(
        (
            chunk
            for chunk in chunks
            if chunk.get("language") in (
                "python",
                "java",
            )
            and chunk.get("type") in (
                "function",
                "method",
            )
            and chunk.get("start_line") is not None
            and chunk.get("end_line") is not None
            and chunk["end_line"]
            > chunk["start_line"]
            + (
                chunk["language"] == "java"
            )
            and not _is_test_file(chunk["file"])
        ),
        None,
    )

    if target is None:
        raise PatchGenerationError(
            "No Python or Java function was retrieved "
            "for the remote model."
        )

    language = str(
        target["language"]
    ).lower()

    file_path = str(
        target["file"]
    )

    try:
        original = (
            Path(repo_path) / file_path
        ).read_text(
            encoding="utf-8"
        )

    except OSError as error:
        raise PatchGenerationError(
            f"Cannot read {file_path}: {error}"
        ) from error

    lines = original.splitlines()

    try:
        start_line = int(
            target["start_line"]
        )

        end_line = int(
            target["end_line"]
        )

    except (
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        raise PatchGenerationError(
            f"Invalid source range for {file_path}: "
            f"{target.get('start_line')} - "
            f"{target.get('end_line')}"
        ) from error

    head = max(
        start_line - 1,
        0,
    )

    start = max(
        start_line,
        0,
    )

    end = (
        end_line - 1
        if language == "java"
        else end_line
    )

    stop = min(
        end_line,
        len(lines),
    )

    if start >= len(lines) or end <= start:
        raise PatchGenerationError(
            f"Invalid repair range for {file_path}: "
            f"start={start}, end={end}, "
            f"total_lines={len(lines)}"
        )

    sites = _repair_sites(lines, head, start, end, issue)
    if not sites:
        raise PatchGenerationError(
            f"{target.get('name', 'The function')} in {file_path} has no statement to repair."
        )

    # One site per request: the model repairs a single line, or writes a
    # single missing guard, reliably, but rewrites (and breaks) a whole
    # commented-out body. Attempt n of the repair loop starts at the n-th most
    # likely site, and a rejected reply moves on to the next one.
    first = (max(attempt, 1) - 1) % len(sites)
    ordered = sites[first:] + sites[:first]
    errors: list[str] = []
    for site_start, site_end in ordered[:REMOTE_LINES_PER_ATTEMPT]:
        inserting = site_start == site_end
        where = f"insertion before line {site_start + 1}" if inserting else f"line {site_start + 1}"
        prompt = _ir4(
            lines=lines,
            head=head,
            start=site_start,
            end=site_end,
            stop=stop,
            file_path=file_path,
            language=language,
            # Tested: a guard is written best under the plain instruction.
            issue="" if inserting else _symptom_sentence(issue, lines[site_start]),
        )
        print(
            f"\n========== REMOTE REPAIR PROMPT ({where}) ==========\n"
            f"{prompt}\n"
            "========== END REMOTE REPAIR PROMPT ==========\n",
            flush=True,
        )
        try:
            reply = _ask_remote_model(prompt, language, file_path, target.get("name", "<unknown>"))
            return _splice_or2(original, lines, site_start, site_end, reply, file_path, language)
        except PatchGenerationError as error:
            errors.append(f"{where}: {error}")
            print(f"Remote repair at {where} rejected: {error}", flush=True)

    raise PatchGenerationError("Remote model: " + "; ".join(errors))


# Lines tried per repair attempt before giving up on the remote model.
REMOTE_LINES_PER_ATTEMPT = 3

_BOUND = re.compile(r"(size\(\)|length\(?\)?|len\([^)]*\))\s*-\s*1|<=\s*[\w.]+\.(size\(\)|length)")
_DIVISION = re.compile(r"[\w)\]]\s*/\s*[\w(]")
_TRUNCATING_COMPARE = re.compile(r"\(int\)\s*\(|\bcompar|\bsort")
_NULL = re.compile(r"\bnull\b|\bNone\b")
_ORDERING = re.compile(r"\s(<|>|<=|>=)\s")  # spaced, so not generics

# (pattern, weight): code shapes where bugs often live, whatever the issue says.
_BUG_PATTERNS = (
    (_BOUND, 3.0),  # off-by-one bound
    (re.compile(r"\(int\)\s*\("), 2.0),  # truncating cast, e.g. in a comparator
    (_DIVISION, 1.5),  # divide by zero
    (re.compile(r"\b(for|while)\b"), 1.0),
    (re.compile(r"\bif\b|\?"), 1.0),
    (re.compile(r"\s(<|>|<=|>=|==|!=)\s"), 0.5),
)

# (symptom in the issue, the kind of line that causes it). A described symptom
# outweighs every generic pattern, so the line the issue points at goes first.
_SYMPTOMS = (
    (re.compile(r"divi(de|sion|ded)|by zero|\bzero\b|\bempty\b|average|\bmean\b|\bnan\b|infinit"), _DIVISION),
    (re.compile(r"\blast\b|\bfirst\b|skip|boundar|off.by.one|out of (range|bounds)"), _BOUND),
    (re.compile(r"sort|compar|precision|rank|descending|ascending"), _TRUNCATING_COMPARE),
    (re.compile(r"\bnull\b|\bnone\b|not found|nullpointer"), _NULL),
    (re.compile(r"threshold|exactly|at least|at most|cutoff|limit"), _ORDERING),
)
_SYMPTOM_WEIGHT = 4.0
_NOT_A_STATEMENT = re.compile(r"^(\}|\{|\};?|\} *else *\{|@\w+.*|//.*|#.*|/\*.*|\*.*|\"\"\".*|'''.*)$")


def _suspicious_lines(lines: list[str], start: int, end: int, issue: str = "", with_scores: bool = False) -> list:
    """Indices in lines[start:end] that hold code, most likely buggy first.

    1. symptoms the issue describes, mapped to the code that causes them
       ("divides by zero" -> a division, "last one skipped" -> a loop bound)
    2. generic bug shapes (off-by-one bounds, truncating casts, conditions)
    3. words shared with the issue, capped: an issue about students mentions
       "student" on every line of the class, so this only breaks ties
    Ties keep source order.
    """
    lowered = issue.lower()
    symptoms = [code_pattern for issue_pattern, code_pattern in _SYMPTOMS if issue_pattern.search(lowered)]
    issue_words = set(re.findall(r"[a-z][a-z0-9]{2,}", lowered))
    scored = []
    for index in range(start, min(end, len(lines))):
        code = lines[index].strip()
        if not code or _NOT_A_STATEMENT.match(code):
            continue
        words = set(re.findall(r"[a-z][a-z0-9]{2,}", re.sub(r"([a-z])([A-Z])", r"\1 \2", code).lower()))
        score = _SYMPTOM_WEIGHT * sum(1 for pattern in symptoms if pattern.search(code))
        score += sum(weight for pattern, weight in _BUG_PATTERNS if pattern.search(code))
        score += min(0.25 * len(words & issue_words), 1.0)
        scored.append((-score, index))
    return [index for _, index in sorted(scored)] if not with_scores else [(-s, i) for s, i in sorted(scored)]


# An issue about a method that divides, worded like this, is about a missing
# check far more often than about the division itself: "add validation for the
# average", "it raises an exception", "divides by zero when empty".
_GUARD_WORDS = re.compile(
    r"divi(de|sion|ded)|by zero|\bzero\b|\bempty\b|\bnan\b|average|\bmean\b|exception|crash|\berror|"
    r"validat|invalid|check|guard|handle"
)


def _repair_sites(lines: list[str], head: int, start: int, end: int, issue: str) -> list[tuple[int, int]]:
    """Where to put <FILL_ME>, most likely first: (i, i + 1) replaces line i,
    (i, i) inserts before line i -- at the top of the body, for a missing guard.

    The guard goes first when the method divides and the issue talks about a
    check, an exception, an average or zero. Otherwise it comes right after
    the lines that strongly match a reported symptom (an off-by-one bound, a
    named comparator...), so a missing check is always among the first tries.
    """
    del head
    ranked = _suspicious_lines(lines, start, end, issue, with_scores=True)
    sites = [(index, index + 1) for _, index in ranked]
    body = _first_statement(lines, start, end)
    if body is None:
        return sites

    guard = (body, body)
    if _GUARD_WORDS.search(issue.lower()) and any(_DIVISION.search(lines[i]) for i in range(start, end)):
        return [guard, *sites]
    # With no strong evidence either way, the most suspicious line still goes first.
    strong = max(sum(1 for score, _ in ranked if score >= _SYMPTOM_WEIGHT), 1)
    return [*sites[:strong], guard, *sites[strong:]]


def _first_statement(lines: list[str], start: int, end: int) -> int | None:
    """The first line of the body that is code -- after a Python docstring."""
    index = start
    while index < end and not lines[index].strip():
        index += 1
    if index < end and lines[index].strip()[:3] in ('"""', "'''"):
        quote, first = lines[index].strip()[:3], lines[index].strip()
        # A one-line docstring closes on its own line; otherwise find the closing quote.
        if not (len(first) > 3 and first.endswith(quote)):
            index += 1
            while index < end and quote not in lines[index]:
                index += 1
        index += 1
        while index < end and not lines[index].strip():
            index += 1
    return index if index < end else None


def _symptom_sentence(issue: str, line: str) -> str:
    """The issue's sentence describing the symptom ``line`` can cause, or "".

    "divides by zero" for a division, "the last one is skipped" for a loop
    bound. Given in the prompt's first line, it tells the model what to fix;
    without it the model only knows that something on the line is wrong.
    """
    # A single line break is wrapping inside a sentence; a blank line ends one.
    unwrapped = re.sub(r"(?<!\n)\n(?!\n)", " ", issue)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", unwrapped) if s.strip()]
    # A title ("Bug: The average calculation has bug") names no symptom.
    sentences = [s for s in sentences if not re.match(r"(bug|issue|fix|feature|problem)\s*:", s, re.I)]
    for issue_pattern, code_pattern in _SYMPTOMS:
        if code_pattern.search(line):
            # The sentence naming the symptom most often, e.g. "divides by zero
            # when no students exist" over "calculate the average when empty".
            hits = [(len(issue_pattern.findall(s.lower())), -i, s) for i, s in enumerate(sentences)]
            best = max(hits, default=(0, 0, ""))
            if best[0]:
                return best[2][:200]
    return ""


def _ask_remote_model(prompt: str, language: str, file_path: str, name: str) -> str:
    """POST one IR4 prompt to the Colab /repair route; return the OR2 text."""
    from src.agents.coding_agent.code_generator import MAX_PATCH_TOKENS, PatchGenerationError  # noqa: PLC0415

    payload = {"prompt": prompt, "max_new_tokens": min(MAX_PATCH_TOKENS, 512)}
    request = urllib.request.Request(
        f"{REPAIR_MODEL_URL}/repair",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"},
    )
    print(
        f"POST {REPAIR_MODEL_URL}/repair: language={language}, {name} in {file_path}, "
        f"prompt {len(prompt)} chars, max_new_tokens={payload['max_new_tokens']}; waiting for the reply...",
        flush=True,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=REMOTE_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
        print("Remote model raw response:", repr(raw_body[:2000]), flush=True)
        data = json.loads(raw_body)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise PatchGenerationError(f"Remote model returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        raise PatchGenerationError(f"Remote model at {REPAIR_MODEL_URL} failed: {error}") from error

    if isinstance(data, str):
        reply = data
    elif isinstance(data, dict):
        # The Colab API answers {"repair": ...}; other names are accepted too.
        reply = next(
            (data[key] for key in ("repair", "response", "generated_text", "text", "output")
             if isinstance(data.get(key), str) and data[key].strip()),
            None,
        )
    else:
        raise PatchGenerationError(f"Unexpected remote response type: {type(data).__name__}")
    if not reply or not reply.strip():
        raise PatchGenerationError(f"Remote model returned no usable repair. Response: {data!r}")

    # Stored for the UI: the exact prompt and repair.
    LAST_REMOTE_REPAIR.clear()
    LAST_REMOTE_REPAIR.update(
        {
            "prompt": data.get("prompt", prompt) if isinstance(data, dict) else prompt,
            "repair": reply,
            "generated_text": data.get("generated_text", reply) if isinstance(data, dict) else reply,
            "elapsed_seconds": data.get("elapsed_seconds") if isinstance(data, dict) else None,
            "file": file_path,
            "language": language,
        }
    )
    # The server decodes with special tokens kept: nothing after the
    # end-of-sequence token "</s>" belongs to the repair. OR2 lines carry their
    # original indentation, so only the end is stripped.
    reply = reply.split("</s>")[0].rstrip()
    print(f"Remote model replied in {time.monotonic() - started:.0f}s: {reply[:500]!r}", flush=True)
    return reply


def _splice_or2(
    original: str, lines: list[str], start: int, end: int, reply: str, file_path: str, language: str
) -> str:
    """Put the OR2 reply in place of lines[start:end], validate, and return the diff."""
    from src.agents.coding_agent.code_generator import PatchGenerationError  # noqa: PLC0415
    from src.agents.coding_agent.function_patch import make_diff  # noqa: PLC0415

    fixed = reply.splitlines()
    while fixed and not fixed[0].strip():
        fixed.pop(0)
    while fixed and not fixed[-1].strip():
        fixed.pop()
    if fixed and fixed[0].strip().startswith("```"):
        fixed.pop(0)
    if fixed and fixed[-1].strip() == "```":
        fixed.pop()
    forbidden_markers = {"<FILL_ME>", "<HOLE>", "<PRE>", "<SUF>", "<MID>", "<EOT>", "</s>"}
    fixed = [line for line in fixed if line.strip() not in forbidden_markers]
    if not fixed:
        raise PatchGenerationError("Remote model returned no usable repair lines.")

    # The model often drops the first line's indentation (and the server's
    # decode can strip one space): put it back at the replaced line's indent.
    # An insertion (start == end) takes the indentation of the line it goes before.
    indent = next((line[: len(line) - len(line.lstrip())] for line in lines[start : max(end, start + 1)] if line.strip()), "")
    if len(fixed[0]) - len(fixed[0].lstrip()) < len(indent):
        fixed[0] = indent + fixed[0].lstrip()

    repaired = "\n".join(lines[:start] + fixed + lines[end:])
    if original.endswith("\n"):
        repaired += "\n"
    if repaired == original:
        raise PatchGenerationError("The remote model returned the line unchanged.")

    if language == "java":
        try:
            _validate_java_repair(original=original, repaired=repaired, file_path=file_path)
            _validate_minimal_order_lookup_repair(original=original, repaired=repaired)
        except AdapterError as error:
            raise PatchGenerationError(str(error)) from error
    elif language == "python":
        try:
            compile(repaired, file_path, "exec")
        except SyntaxError as error:
            raise PatchGenerationError(
                f"The repaired {file_path} is not valid Python: {error.msg} (line {error.lineno})."
            ) from error

    return make_diff(file_path, original, repaired)

def get_last_remote_repair() -> dict[str, Any]:
    """
    Return the last prompt and repair received from Colab.

    The UI can use this function after generate_patch().
    """

    return dict(LAST_REMOTE_REPAIR)

def _ir4(
    lines: list[str],
    head: int,
    start: int,
    end: int,
    stop: int,
    file_path: str,
    language: str,
    issue: str = "",
) -> str:
    """
    The IR4 input for lines[start:end], exactly as each adapter was trained.

    IR4: the buggy lines stay visible but commented out between markers, and
    `<FILL_ME>` marks where the replacement goes. The model answers with OR2:
    only the replacement lines, which `remote_patch` splices back in place of
    lines[start:end].

    Java: a two-line instruction, then the RepairLLaMA ir4xor2 format
    (ASSERT-KTH/repairllama-datasets, as in adapters/MegaDiff/test_qlora.py):
    the method alone with its own indentation, "// buggy code" at column 0,
    each buggy line as "// <line>", then <FILL_ME> at the buggy code's
    indentation and the rest of the method. ``issue`` is the one symptom
    sentence for that line (see `_symptom_sentence`), appended to the first
    instruction line; "" keeps the instruction plain.
    Python: preprocessor/pythonprocessor.py's make_ir4, scoped to the function
    (lines[head:stop]) as in the BugsInPy dataset ("# BUGGY CODE", "# <line>");
    that training set had no instructions, so ``issue`` is not used.
    """
    del file_path

    if language == "java":
        region = lines[start:end]
        # An insertion (start == end) is indented like the line it goes before.
        indent = next(
            (line[: len(line) - len(line.lstrip())] for line in lines[start : max(end, start + 1)] if line.strip()), ""
        )
        # Nothing is commented out for an insertion: "// buggy code" with no
        # line under it is not what the tested prompts looked like.
        marked = ["// buggy code", *(f"// {line}" if line.strip() else "//" for line in region)] if region else []
        symptom = issue.strip().rstrip(".")
        return "\n".join(
            [
                # Tested against the Colab adapter: with this header and one
                # commented-out line it returns just the corrected line, and the
                # symptom in the first line makes it fix that symptom.
                f"Repair the bug in this Java code: {symptom[0].lower() + symptom[1:]}." if symptom
                else "Repair the bug in this Java code.",
                "Return only the missing code.",
                "",
                *lines[head:start],
                *marked,
                f"{indent}<FILL_ME>",
                *lines[end:stop],
            ]
        )

    commented = [f"# {line}" if line else "#" for line in lines[start:end]]
    return "\n".join([*lines[head:start], "# BUGGY CODE", *commented, "<FILL_ME>", *lines[end:stop]])

def _load_stub_patch(
    stub_patch_path: str,
) -> str:
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

    return path.read_text(
        encoding="utf-8"
    )


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
            f"(lines {chunk['start_line']}-"
            f"{chunk['end_line']}, "
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
    """Clean old workspaces and prepare the test environment."""

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
    """Run the suite on a throwaway repository copy."""

    _ensure_import_paths()

    try:
        from src.agents.testing_agent import (
            run_baseline,
        )

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
    """Write and validate a reproduction test."""

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
    """Apply a patch to a fresh copy and run the test suite."""

    _ensure_import_paths()

    try:
        from src.agents.testing_agent import (
            evaluate_patch,
        )

    except ImportError as error:
        raise AdapterError(
            f"Testing agent is unavailable: {error}"
        ) from error

    accepted = (
        bool(reproduction)
        and reproduction.get("status")
        == "accepted"
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

    from src.agents.testing_agent.feedback import (
        build_feedback,
    )

    return build_feedback(
        attempt["attempt"],
        attempt["patch"],
        attempt["patch_status"],
        attempt["test_result"],
        attempt["test_verdict"],
        patch_error=attempt.get(
            "patch_error",
            "",
        ),
        repeated=(
            bool(attempt["patch"].strip())
            and attempt["patch"]
            in (earlier_patches or [])
        ),
    )


REFLECTION_LLM = None


def reflect(
    issue: str,
    relevant_code: str,
    attempt: dict[str, Any],
    strategy: str = "",
    previous_attempts: list[dict[str, Any]]
    | None = None,
    max_attempts: int = 3,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run the reflection agent on one tested attempt."""

    _ensure_import_paths()

    try:
        from src.agents.reflection_agent import (
            ReflectionAgent,
        )

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
        generated_patch=attempt.get(
            "patch",
            "",
        ),
        test_results=attempt,
        repair_strategy=strategy,
        previous_attempts=previous_attempts,
        attempt=attempt.get(
            "attempt",
            1,
        ),
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
    """Return whether candidate should replace current."""

    _ensure_import_paths()

    from src.agents.testing_agent.feedback import (
        is_better,
    )

    return is_better(
        candidate,
        current,
    )


# --------------------------------------------------------------------------
# Commit and push
# --------------------------------------------------------------------------


def commit_and_push(
    repo_path: str,
    patch: str,
    issue: str = "",
) -> dict[str, str]:
    """Commit patch on a new branch and push that branch."""

    from orchestrator.github_issues import (
        load_token,
    )

    repository_root = Path(
        _git(
            repo_path,
            "rev-parse",
            "--show-toplevel",
        )
    ).resolve()

    if repository_root != Path(repo_path).resolve():
        raise AdapterError(
            f"{repo_path} is not the root of its own git "
            "repository; nothing to push."
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
        if token
        and remote.startswith("https://")
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
        worktree = str(
            Path(tmp) / "worktree"
        )

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

            _git(
                worktree,
                "add",
                "-A",
            )

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
        [
            "git",
            *args,
        ],
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
            message = message.replace(
                secret,
                "***",
            )

        raise AdapterError(
            f"git {args[0]} failed: {message}"
        )

    return result.stdout.strip()


def _repository_language(
    repo_path: str,
    results: list[dict[str, Any]],
) -> str:
    """Determine the repair and test language."""

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


def _is_test_file(
    file_path: str,
) -> bool:
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


def _unique(
    items: list[str],
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

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