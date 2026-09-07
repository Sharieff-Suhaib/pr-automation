"""Generate a minimal repair patch with a code-capable Ollama model.

This module is deliberately independent of the other agents.  The workflow can
pass plain strings and lists to :func:`generate_patch`, then Phase 2 can apply
the returned unified diff in a copied repository.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agents.coding_agent.code_rewriter import (
    RewriteError,
    TargetChunk,
    build_rewrite_prompt,
    clean_code_block,
    diff_from_replacement,
    select_targets,
)
from src.agents.coding_agent.diff_validator import (
    DiffError,
    describe_diff_error,
    extract_diff,
    normalize_diff,
    validate_unified_diff,
)


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AGENT_SWE_CODEGEN_MODEL", "codellama:7b")
REQUEST_TIMEOUT_SECONDS = 300

# One first attempt plus two corrections. Small models rarely produce a valid
# diff on the first try but usually fix a specific, quoted complaint; beyond
# three attempts they tend to repeat themselves, so the extra minutes buy
# nothing.
DEFAULT_MAX_ATTEMPTS = 3

# How many files one issue may repair. A repository that implements the same
# function in several languages needs one patch per implementation; a normal
# repository retrieves one relevant file and this cap never binds.
DEFAULT_MAX_TARGETS = 4

SYSTEM_PROMPT = """You are a software repair agent.
Return only a standard unified diff patch. Do not include explanations or Markdown."""

REWRITE_SYSTEM_PROMPT = """You are a software repair agent.
You are given one function. You reply with the corrected version of that
function and nothing else: no explanation, no diff, no markdown fence."""

_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class PatchGenerationError(RuntimeError):
    """Raised when a model cannot provide a usable patch."""


def build_repair_prompt(
    issue: Any,
    relevant_code: Any,
    similar_bugs: Any,
    strategy: Any,
    tools: Any,
    tests: Any,
) -> str:
    """Create the repair brief sent to the LLM.

    ``str`` is used intentionally for each input.  This lets the function work
    with the recommendation agents' dataclasses as well as simple dictionaries
    or strings while those agent interfaces are still evolving.
    """
    return f"""You are a software repair agent.

ISSUE:
{_format_value(issue)}

RELEVANT CODE:
{_format_value(relevant_code)}

SIMILAR BUGS:
{_format_value(similar_bugs)}

RECOMMENDED STRATEGY:
{_format_value(strategy)}

RECOMMENDED TOOLS:
{_format_value(tools)}

RECOMMENDED TESTS:
{_format_value(tests)}

Generate a minimal code patch that fixes the issue.
Do not modify unrelated code.
Return only a standard unified diff beginning with --- and +++.
"""


def generate_patch(
    issue: Any,
    relevant_code: Any,
    similar_bugs: Any,
    strategy: Any,
    tools: Any,
    tests: Any,
    repo_path: str | Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    chunks: Any = None,
    language: str = "",
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> str:
    """Ask the configured LLM for a minimal unified-diff repair patch.

    Two strategies, chosen by what the caller can supply:

    * **Rewrite** (preferred, used when ``chunks`` and ``repo_path`` are
      given): the model rewrites one function and the diff is computed from
      the file on disk.  The model never writes diff syntax, so it cannot get
      it wrong -- see :mod:`code_rewriter` for why that matters.
    * **Diff** (the fallback): the model writes the unified diff itself.  Each
      attempt is checked against the diff parser and, when ``repo_path`` is
      known, ``git apply --check``; a failure is quoted back to the model,
      which corrects a specific complaint far more reliably than it responds
      to the original instruction repeated.
    """
    if chunks and repo_path is not None:
        try:
            return generate_patch_by_rewrite(
                issue,
                chunks,
                similar_bugs=similar_bugs,
                strategy=strategy,
                tools=tools,
                tests=tests,
                repo_path=repo_path,
                language=language,
                max_attempts=max_attempts,
                max_targets=max_targets,
            )
        except (RewriteError, PatchGenerationError) as error:
            # Fall through to asking for a diff: a repository layout this
            # cannot handle is better served by the model's own diff than by
            # no patch at all.
            log_note = f"rewrite strategy failed ({error}); asking for a diff instead"
            print(f"[coding_agent] {log_note}")

    prompt = build_repair_prompt(issue, relevant_code, similar_bugs, strategy, tools, tests)
    attempts = max(1, max_attempts)
    complaint = ""

    for attempt in range(1, attempts + 1):
        request = prompt if attempt == 1 else _correction_prompt(prompt, complaint)
        response = _call_ollama(request)

        try:
            patch = _clean_patch(response, repo_path)
        except PatchGenerationError as error:
            complaint = str(error)
        else:
            complaint = _reject_reason(patch, repo_path)
            if not complaint:
                return patch

        if attempt < attempts:
            # The next request repeats the task with this exact complaint.
            continue

    raise PatchGenerationError(
        f"The LLM did not return an applicable unified diff after "
        f"{attempts} attempt(s). Last problem: {complaint}"
    )


def generate_patch_by_rewrite(
    issue: Any,
    chunks: Any,
    *,
    similar_bugs: Any = None,
    strategy: Any = None,
    tools: Any = None,
    tests: Any = None,
    repo_path: str | Path,
    language: str = "",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> str:
    """Rewrite the retrieved functions and diff them against the files on disk.

    One region per file, up to ``max_targets`` files, concatenated into a
    single multi-file patch.  Each file's diff is computed and verified on its
    own, so one file the model cannot repair does not cost the others: the
    failures are logged and the successful diffs are still returned.
    """
    targets = select_targets(chunks, language, max_targets=max_targets)
    patches: list[str] = []
    failures: list[str] = []

    for target in targets:
        try:
            patches.append(
                _rewrite_one(
                    issue,
                    target,
                    similar_bugs=similar_bugs,
                    strategy=strategy,
                    tools=tools,
                    tests=tests,
                    repo_path=repo_path,
                    max_attempts=max_attempts,
                )
            )
        except PatchGenerationError as error:
            failures.append(f"{target.file}: {error}")
            print(f"[coding_agent] could not repair {target.describe()} -- {error}")

    if not patches:
        raise PatchGenerationError(
            "No file could be repaired. " + " | ".join(failures)
        )

    combined = "".join(patches)
    complaint = _reject_reason(combined, repo_path)
    if complaint:
        # Each part applied on its own, so a combined failure means two of them
        # touch the same file. Fall back to the best single patch.
        print(f"[coding_agent] combined patch rejected ({complaint}); using the first file only")
        return patches[0]
    if failures:
        print(f"[coding_agent] repaired {len(patches)} file(s); {len(failures)} could not be repaired")
    return combined


def _rewrite_one(
    issue: Any,
    target: TargetChunk,
    *,
    similar_bugs: Any,
    strategy: Any,
    tools: Any,
    tests: Any,
    repo_path: str | Path,
    max_attempts: int,
) -> str:
    """Rewrite one function, retrying with the reason the last answer failed."""
    prompt = build_rewrite_prompt(issue, target, similar_bugs, strategy, tools, tests)
    attempts = max(1, max_attempts)
    complaint = ""

    for attempt in range(1, attempts + 1):
        request = prompt if attempt == 1 else _rewrite_correction_prompt(prompt, complaint)
        response = _call_ollama(request, system=REWRITE_SYSTEM_PROMPT)

        try:
            # Indentation is aligned inside diff_from_replacement, which has
            # the file and therefore knows how deeply the code is nested.
            replacement = clean_code_block(response, target.language)
            patch = diff_from_replacement(repo_path, target, replacement)
        except RewriteError as error:
            complaint = str(error)
            continue

        complaint = _reject_reason(patch, repo_path)
        if not complaint:
            return patch

    raise PatchGenerationError(
        f"could not turn the rewrite of {target.describe()} into an applicable "
        f"patch after {attempts} attempt(s). Last problem: {complaint}"
    )


def _rewrite_correction_prompt(original_prompt: str, complaint: str) -> str:
    """Repeat the rewrite request, saying what was wrong with the last answer."""
    return f"""{original_prompt}

Your previous answer was rejected:

{complaint}

Reply with only the corrected code, complete and correctly indented.
"""


def _reject_reason(patch: str, repo_path: str | Path | None) -> str:
    """Return why ``patch`` is unusable, or an empty string if it is fine."""
    try:
        validate_unified_diff(patch)
    except DiffError as error:
        return describe_diff_error(error, patch)

    if repo_path is None:
        return ""

    # Parsing only proves the diff is well formed; it can still describe lines
    # that are not in the file. Git is the authority on that.
    from src.agents.coding_agent.patch_manager import check_patch  # noqa: PLC0415

    applies, error = check_patch(repo_path, patch)
    if applies:
        return ""
    return (
        f"git apply rejected the patch: {error}\n"
        "The context lines must match the file exactly, and the line numbers "
        "in each @@ header must be the real ones."
    )


def _format_value(value: Any) -> str:
    """Make lists readable in the prompt without imposing data-model imports."""
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value) or "None"
    return str(value)


def _call_ollama(
    prompt: str, model: str = DEFAULT_MODEL, system: str = SYSTEM_PROMPT
) -> str:
    """Send one non-streaming request to the local Ollama server."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise PatchGenerationError(f"Ollama returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise PatchGenerationError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start Ollama and pull {model}."
        ) from exc

    answer = body.get("response", "")
    if not answer.strip():
        raise PatchGenerationError("The LLM returned an empty response.")
    return answer


def _clean_patch(response: str, repo_path: str | Path | None = None) -> str:
    """Remove a Markdown fence and any prose around the diff, then repair it."""
    match = _FENCE_RE.search(response.strip())
    patch = (match.group(1) if match else response).strip()
    # Models introduce a patch ("Here is the fix:") and explain it afterwards.
    patch = extract_diff(patch)
    # Decorated headers ("--- stats.py (original)"), miscounted @@ headers and
    # invented directories ("src/Auth.java" for a root-level file) are the
    # mistakes small models make constantly, and all three are fixable from
    # the patch body and the repository -- cheaper and more reliable than
    # another request. Only headers are rewritten; changed lines are untouched.
    patch = normalize_diff(patch, repo_path=repo_path)
    if not patch.startswith(("--- ", "diff --git ")):
        raise PatchGenerationError(
            "The response contains no unified diff: it must begin with "
            "'--- a/<path>' and contain a '@@' hunk header."
        )
    return patch + "\n"


def _correction_prompt(original_prompt: str, complaint: str) -> str:
    """Repeat the task, quoting exactly what was wrong with the last answer."""
    return f"""{original_prompt}

Your previous response was rejected:

{complaint}

Reply with only a unified diff, and fix that specific problem.
Rules:
- File headers are paths only: "--- a/src/Auth.java", never a description.
- Every line inside a hunk starts with " ", "+" or "-".
- The counts in "@@ -start,count +start,count @@" must match the number of
  lines you actually write.
- Copy context lines exactly as they appear in the file shown above.
For example:
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,2 +1,4 @@
 def example():
-    return broken_value
+    if broken_value is None:
+        return None
+    return broken_value
Do not return a rewritten function by itself.
"""
