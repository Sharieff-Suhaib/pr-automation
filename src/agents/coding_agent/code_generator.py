"""Generate a minimal repair patch with a code-capable Ollama model.

This module is deliberately independent of the other agents.  The workflow can
pass plain strings and lists to :func:`generate_patch`, then Phase 2 can apply
the returned unified diff in a copied repository.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_CODEGEN_DIR = Path(__file__).resolve().parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from fault_localization import SuspiciousRegion, run_fault_localization


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AGENT_SWE_CODEGEN_MODEL", "codellama:7b")
REQUEST_TIMEOUT_SECONDS = 300
# A minimal repair is a few hundred tokens. Without a cap, a small model that
# starts repeating hunks keeps going until the request times out.
MAX_PATCH_TOKENS = int(os.environ.get("AGENT_SWE_CODEGEN_MAX_TOKENS", "1024"))

SYSTEM_PROMPT = """You are a software repair agent.
Return only a standard unified diff patch. Do not include explanations or Markdown."""

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
) -> str:
    """Ask the configured LLM for a minimal unified-diff repair patch."""
    prompt = build_repair_prompt(issue, relevant_code, similar_bugs, strategy, tools, tests)
    response = _call_ollama(prompt)
    try:
        return _clean_patch(response)
    except PatchGenerationError:
        # Smaller local models sometimes repair the code correctly but ignore
        # the requested diff format. Give the same task one focused retry
        # before treating it as an unusable patch.
        response = _call_ollama(_diff_format_retry_prompt(prompt))
        return _clean_patch(response)


def generate_patch_from_repository(
    repo_url: str,
    issue: str = "",
    similar_bugs: Any = None,
    strategy: Any = None,
    tools: Any = None,
    tests: Any = None,
    top_k: int = 4,
) -> str:
    """Localize suspicious regions, then generate a patch for those regions.

    This is the repository-level entry point for callers that do not already
    have relevant code. Fault localization is deliberately performed before
    contacting Ollama so the model receives the ranked source regions and their
    IR4 representations rather than an unbounded repository dump.
    """
    if not repo_url.strip():
        raise PatchGenerationError("A repository URL or path is required.")

    regions = run_fault_localization(
        repo_url=repo_url,
        issue=issue,
        top_k=top_k,
    )
    if not regions:
        raise PatchGenerationError(
            "Fault localization found no suspicious regions to repair."
        )

    relevant_code = format_fault_localization_results(regions)
    return generate_patch(
        issue=issue,
        relevant_code=relevant_code,
        similar_bugs=similar_bugs,
        strategy=strategy,
        tools=tools,
        tests=tests,
    )


def format_fault_localization_results(
    regions: list[SuspiciousRegion],
) -> str:
    """Format localized regions as bounded, file-labelled model context."""
    sections: list[str] = []
    for index, region in enumerate(regions, start=1):
        sections.append(
            f"""REGION {index}
FILE: {region.file}
SUSPICIOUS LINES: {region.start_line}-{region.end_line}
SCORE: {region.score:.4f}

ORIGINAL SUSPICIOUS CODE:
{region.original_code}

IR4 REPRESENTATION:
{region.ir4}"""
        )
    return "\n\n".join(sections)


def _format_value(value: Any) -> str:
    """Make lists readable in the prompt without imposing data-model imports."""
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value) or "None"
    return str(value)


def _call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0,
) -> str:
    """Send one non-streaming request to the local Ollama server."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": MAX_PATCH_TOKENS},
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
    except TimeoutError as exc:
        # A read timeout is raised as a bare socket timeout, not a URLError.
        raise PatchGenerationError(
            f"Ollama did not answer within {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc

    patch = body.get("response", "")
    if not patch.strip():
        raise PatchGenerationError("The LLM returned an empty patch.")
    if body.get("done_reason") == "length":
        # A diff cut off mid-hunk cannot apply, so report why instead of passing it on.
        raise PatchGenerationError(
            f"The LLM's patch was cut off at {MAX_PATCH_TOKENS} tokens; "
            "it was probably repeating itself."
        )
    return patch


def _clean_patch(response: str) -> str:
    """Remove an optional Markdown fence and reject non-diff model output."""
    match = _FENCE_RE.search(response.strip())
    patch = (match.group(1) if match else response).strip()
    # Models occasionally put one sentence before an otherwise valid diff.
    diff_start = patch.find("--- ")
    if diff_start > 0:
        patch = patch[diff_start:]
    if not patch.startswith("--- ") or "\n+++ " not in patch:
        raise PatchGenerationError(
            "The LLM did not return a standard unified diff patch."
        )
    return patch + "\n"


def _diff_format_retry_prompt(original_prompt: str) -> str:
    """Repeat the task with an example of the only acceptable response form."""
    return f"""{original_prompt}

Your previous response format was invalid. Reply with only a unified diff.
It must start with file headers and contain a hunk header, for example:
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
