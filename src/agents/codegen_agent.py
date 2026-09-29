"""Code Generation Agent.

Answers: *what does the corrected code look like?*

STAGE A: the simplest useful version — buggy code in, corrected code out, with
a local CodeLlama-7B served by Ollama doing the work. No retrieval, no similar
bugs, no strategy hints; those arrive as `extras` sections once the upstream
recommendation agents stop being stubs.

Ollama is called over its plain HTTP API with `urllib`, so the agent adds no
dependency to requirements.txt and works whether or not the training stack
(torch/transformers) is installed.

CLI:
    python -m src.agents.codegen_agent --code-file buggy.py
    python -m src.agents.codegen_agent --code "def f(x): return x[0]" --issue "crashes on []"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from src.agents.state import AgentState, CandidatePatch, trace_entry

# Overridable so the same code runs against a remote Ollama host or a different
# model without editing the module.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AGENT_SWE_CODEGEN_MODEL", "codellama:7b")

# Deterministic by default: at this stage we want a repeatable single patch, not
# a diverse candidate pool.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_PREDICT = 512
REQUEST_TIMEOUT = 300  # a 7B model on CPU is slow; a short timeout just hides that

SYSTEM_PROMPT = (
    "You are an expert Python engineer specialized in automated program repair. "
    "You are given buggy code. You reply with the corrected code only — no "
    "explanations, no commentary, no markdown fences."
)

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)


class OllamaError(RuntimeError):
    """Ollama is unreachable, or the requested model is not pulled."""


def build_prompt(buggy_code: str, issue: str | None = None) -> str:
    """Minimal repair brief: the code, optionally the symptom, one instruction."""
    parts = ["### BUGGY CODE", buggy_code.strip()]
    if issue and issue.strip():
        parts += ["\n### ISSUE", issue.strip()]
    parts += [
        "\n### INSTRUCTION",
        "Fix the bug and output the complete corrected code. Output code only.",
        "\n### FIXED CODE",
    ]
    return "\n".join(parts)


def clean_output(text: str) -> str:
    """Pull plain code out of whatever the model wrapped it in.

    CodeLlama fences its answers most of the time even when told not to, and
    often adds a sentence before the block — so a fenced block, when present,
    wins over the surrounding prose.
    """
    text = text.strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1)
    text = re.sub(r"^###\s*FIXED CODE\s*\n", "", text)
    return text.strip()


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> str:
    """One non-streaming completion from the local Ollama server."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST} ({exc.reason}). "
            "Start it with `ollama serve` and pull the model with "
            f"`ollama pull {model}`."
        ) from exc
    return body.get("response", "")


def fix_code(buggy_code: str, issue: str | None = None, model: str = DEFAULT_MODEL) -> str:
    """Buggy code in, corrected code out. The whole agent in one call."""
    raw = call_ollama(build_prompt(buggy_code, issue), model=model)
    return clean_output(raw)


def generate(state: AgentState, model: str = DEFAULT_MODEL) -> AgentState:
    """Agent-graph node: repair the first relevant chunk and record the patch."""
    chunks = state.get("relevant_chunks", [])
    if not chunks:
        return AgentState(
            candidate_patches=[],
            trace=[trace_entry("codegen_agent", "no relevant chunks to repair")],
        )

    target = chunks[0]
    issue = state.get("issue")
    attempt = state.get("attempt", 0)

    fixed = fix_code(target.source, issue.as_text() if issue else None, model=model)
    patch = CandidatePatch(code=fixed, target_chunk=target.qualified_name(), attempt=attempt)

    return AgentState(
        candidate_patches=[patch],
        trace=[
            trace_entry(
                "codegen_agent",
                f"1 candidate for {target.qualified_name()} via {model}",
                model=model,
                attempt=attempt,
            )
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fix with CodeLlama via Ollama")
    parser.add_argument("--code", default=None, help="Buggy code as an inline string")
    parser.add_argument("--code-file", default=None, help="File containing the buggy code")
    parser.add_argument("--issue", default=None, help="Optional bug description")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    buggy_code = args.code
    if args.code_file:
        buggy_code = Path(args.code_file).read_text(encoding="utf-8")
    if not buggy_code:
        raise SystemExit("No buggy code provided. Use --code or --code-file.")

    try:
        fixed = fix_code(buggy_code, args.issue, model=args.model)
    except OllamaError as exc:
        print(f"[codegen] {exc}")
        return 1

    print("=" * 60)
    print("BUGGY CODE")
    print("=" * 60)
    print(buggy_code.strip())
    print("\n" + "=" * 60)
    print(f"CORRECTED CODE  (model={args.model})")
    print("=" * 60)
    print(fixed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())