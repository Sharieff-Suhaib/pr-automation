"""Repair-specific code representation (RepairLLaMA-inspired).

Two ideas from the RepairLLaMA paper are kept, in simplified form:

1. A *repair-specific representation* rather than raw code completion: the model
   sees a structured brief (task, issue, code, fault location, instruction).
2. *Fault localization signal in the input*: the suspicious lines are annotated
   inline with `>>>` markers and restated as an explicit line range, so the model
   learns to attend to the buggy region instead of rewriting everything.

The builder is composed of small `_render_*` section helpers plus an `extras`
channel, so later Agent-SWE stages (similar bugs, repair strategy, repository
context, relevant tests, tool recommendations) can add sections without changing
the training/inference call sites.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.dataset.loader import FaultLocation, RepairExample

SYSTEM_PROMPT = (
    "You are an expert Python software engineer specialized in automated program "
    "repair. You are given a buggy Python function together with a bug report and "
    "optional fault localization hints. You reply with the corrected code only."
)

_NO_ISSUE = "No issue description was provided. Infer the defect from the code itself."
_NO_LOCATION = "Not available. Inspect the whole snippet to locate the defect."

# Section order in the rendered prompt. Sections that render to None are dropped,
# so optional context never leaves an empty heading behind.
_OPTIONAL_EXTRA_SECTIONS: tuple[tuple[str, str], ...] = (
    ("similar_bugs", "SIMILAR BUGS"),
    ("repair_strategy", "REPAIR STRATEGY"),
    ("repository_context", "REPOSITORY CONTEXT"),
    ("relevant_tests", "RELEVANT TESTS"),
    ("tool_recommendations", "TOOL RECOMMENDATIONS"),
)


def annotate_fault_lines(code: str, location: FaultLocation | None) -> str:
    """Prefix each line with its number, marking the suspicious range with `>>>`.

    Without a location, lines are still numbered so the model can reason about
    the line range mentioned in the issue text.

        1     def get_first(items):
    >>> 2         return items[0]
    """
    lines = code.splitlines() or [""]
    width = len(str(len(lines)))
    rendered = []
    for number, line in enumerate(lines, start=1):
        in_fault = location is not None and location.start_line <= number <= location.end_line
        marker = ">>>" if in_fault else "   "
        rendered.append(f"{marker} {number:>{width}} | {line}")
    return "\n".join(rendered)


def _render_extra(value: Any) -> str | None:
    """Normalize an extras value (str / list / dict) into prompt text."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Sequence):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(f"- {item}" for item in items) or None
    return str(value).strip() or None


def build_repair_prompt(
    issue: str | None,
    buggy_code: str,
    fault_location: FaultLocation | None = None,
    extras: dict[str, Any] | None = None,
    language: str = "Python",
) -> str:
    """Render the user-side repair brief.

    This is the single place the input representation is defined; training and
    inference both call it so the model never sees a format it was not trained on.
    """
    location_text = (
        f"Lines {fault_location.start_line} to {fault_location.end_line} "
        f"(marked with >>> above)."
        if fault_location
        else _NO_LOCATION
    )

    sections: list[tuple[str, str]] = [
        ("TASK", f"Fix the bug in the following {language} code."),
        ("ISSUE", (issue or "").strip() or _NO_ISSUE),
        ("BUGGY CODE", annotate_fault_lines(buggy_code, fault_location)),
        ("FAULT LOCATION", location_text),
    ]

    # Future Agent-SWE context slots, appended only when populated.
    extras = extras or {}
    for key, heading in _OPTIONAL_EXTRA_SECTIONS:
        rendered = _render_extra(extras.get(key))
        if rendered:
            sections.append((heading, rendered))

    sections.append(
        (
            "INSTRUCTION",
            f"Analyze the bug and output the complete corrected {language} code. "
            f"Do not include explanations, comments about the fix, or markdown fences.",
        )
    )

    body = "\n\n".join(f"### {heading}\n{content}" for heading, content in sections)
    return f"{body}\n\n### FIX\n"


def build_target(fixed_code: str) -> str:
    """The completion the model is trained to produce."""
    return fixed_code.strip()


def build_messages(example: RepairExample, include_answer: bool) -> list[dict[str, str]]:
    """Chat-format the example for Qwen2.5-Coder-Instruct.

    `include_answer=True` produces a full training conversation; False produces
    the inference prompt that stops right before the assistant's turn.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_repair_prompt(
                issue=example.issue,
                buggy_code=example.buggy_code,
                fault_location=example.fault_location,
                extras=example.extras,
            ),
        },
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": build_target(example.fixed_code)})
    return messages


def format_for_training(example: RepairExample, tokenizer=None) -> str:
    """Full training text: prompt + target.

    Uses the tokenizer's chat template when available (correct for Qwen
    Instruct models); otherwise falls back to a plain-text rendering so the
    module stays usable without transformers installed.
    """
    messages = build_messages(example, include_answer=True)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return _plain_text_fallback(messages)


def format_for_inference(
    issue: str | None,
    buggy_code: str,
    fault_location: FaultLocation | None = None,
    extras: dict[str, Any] | None = None,
    tokenizer=None,
) -> str:
    """Prompt text ending where the model should start writing the fix."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_repair_prompt(issue, buggy_code, fault_location, extras),
        },
    ]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return _plain_text_fallback(messages) + "\n<|assistant|>\n"


def _plain_text_fallback(messages: list[dict[str, str]]) -> str:
    """Chat rendering used when no tokenizer/chat template is available."""
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
