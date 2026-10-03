"""Length filtering, measured in tokens rather than characters.

A prompt that does not fit the model's context is wasted training signal, so
examples are filtered on the *tokenizer's* count, not on line or character
counts.  The counter is pluggable:

``HeuristicTokenCounter`` (default)
    A dependency-free approximation of a BPE code tokenizer: identifiers,
    numbers, operators and indentation runs are counted as tokens, with long
    identifiers split the way subword vocabularies split them.  Deterministic,
    fast, and available with no model downloaded — which is what makes
    ``prepare-data`` runnable offline.

``HuggingFaceTokenCounter``
    The real tokenizer for the configured base model.  Loads lazily and only
    when explicitly asked for (``data.use_model_tokenizer``), because
    constructing it may download files.

Any object with ``count(text) -> int`` works, so a project can drop in its
own.  The filter reports the full length distribution over everything it
measured, not just what survived — that is the distribution you need in order
to choose a sensible ``max_input_tokens``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from repairllama.data.models import (
    Rejection,
    RejectionReason,
    TokenStats,
    TrainingRecord,
)
from repairllama.utils.logging import get_logger

__all__ = [
    "TokenCounter",
    "HeuristicTokenCounter",
    "HuggingFaceTokenCounter",
    "TokenLengths",
    "TokenizerFilterOptions",
    "TokenizerFilterResult",
    "TokenizerFilter",
    "get_token_counter",
]

log = get_logger("data.tokenizer_filter")


class TokenCounter(Protocol):
    """Anything that can count tokens in a string."""

    name: str

    def count(self, text: str) -> int:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# counters
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"""
    [A-Za-z_$][A-Za-z_$0-9]*   # identifiers and keywords
    | \d+(?:\.\d+)?[fFdDlL]?   # numbers
    | "(?:\\.|[^"\\])*"        # string literals
    | '(?:\\.|[^'\\])*'        # char literals
    | \n                       # newlines cost a token
    | [^\sA-Za-z_$0-9]         # operators and punctuation
    """,
    re.VERBOSE,
)

# Roughly how many characters of an identifier a BPE vocabulary covers per
# token; long camelCase names cost more than one token.
_CHARS_PER_SUBWORD = 5


class HeuristicTokenCounter:
    """Approximate a code tokenizer without loading one.

    Counts lexical tokens, splitting long identifiers and string literals into
    subword-sized pieces.  It is an estimate — within roughly ±15% of a
    CodeLlama tokenizer on Java in practice — and it is deterministic, which
    matters more here: the same corpus always yields the same dataset.
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        total = 0
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            if token == "\n":
                total += 1
            elif len(token) <= _CHARS_PER_SUBWORD:
                total += 1
            else:
                total += -(-len(token) // _CHARS_PER_SUBWORD)  # ceil division
        # Leading indentation is tokenised separately by most code tokenizers.
        total += sum(1 for line in text.split("\n") if line[:1] in {" ", "\t"})
        return total


class HuggingFaceTokenCounter:
    """Count with the real tokenizer of a HuggingFace checkpoint.

    ``transformers`` is imported inside the constructor so that importing this
    module stays free.  ``local_files_only`` defaults to True: a dataset build
    should not silently pull files off the network.
    """

    def __init__(
        self,
        tokenizer_name: str,
        *,
        local_files_only: bool = True,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "transformers is not installed; install the 'train' extra or use "
                "the heuristic token counter"
            ) from exc

        self.name = tokenizer_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])


def get_token_counter(
    tokenizer_name: Optional[str] = None,
    *,
    use_model_tokenizer: bool = False,
    local_files_only: bool = True,
    fallback: bool = True,
) -> TokenCounter:
    """Pick a counter, falling back to the heuristic one.

    With ``use_model_tokenizer=False`` (the default) the heuristic counter is
    returned without touching ``transformers`` at all.  With it enabled, a
    failure to load the real tokenizer falls back with a warning unless
    ``fallback=False``.
    """
    if not use_model_tokenizer or not tokenizer_name:
        return HeuristicTokenCounter()
    try:
        counter = HuggingFaceTokenCounter(
            tokenizer_name, local_files_only=local_files_only
        )
        log.info("counting tokens with the %s tokenizer", tokenizer_name)
        return counter
    except Exception as exc:  # noqa: BLE001 - any load failure is recoverable
        if not fallback:
            raise
        log.warning(
            "could not load tokenizer %s (%s); falling back to the heuristic counter",
            tokenizer_name,
            exc,
        )
        return HeuristicTokenCounter()


# --------------------------------------------------------------------------- #
# the filtering stage
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenLengths:
    """Measured lengths of one record."""

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class TokenizerFilterOptions:
    """Length limits, in tokens."""

    max_input_tokens: int = 1024
    max_output_tokens: int = 512
    max_total_tokens: Optional[int] = None
    min_output_tokens: int = 0
    histogram_buckets: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("token limits must be >= 1")


@dataclass
class TokenizerFilterResult:
    """Records that fit, those that did not, and the length distributions."""

    kept: List[TrainingRecord] = field(default_factory=list)
    rejections: List[Rejection] = field(default_factory=list)
    stats: Dict[str, TokenStats] = field(default_factory=dict)

    @property
    def received(self) -> int:
        return len(self.kept) + len(self.rejections)

    def counts_by_reason(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.reason.value] = counts.get(rejection.reason.value, 0) + 1
        return counts


class TokenizerFilter:
    """Measures records and drops those that exceed the configured limits."""

    stage_name = "tokenizer_filter"

    def __init__(
        self,
        counter: Optional[TokenCounter] = None,
        options: Optional[TokenizerFilterOptions] = None,
    ) -> None:
        self.counter = counter or HeuristicTokenCounter()
        self.options = options or TokenizerFilterOptions()

    def measure(self, record: TrainingRecord) -> TokenLengths:
        return TokenLengths(
            input_tokens=self.counter.count(record.input),
            output_tokens=self.counter.count(record.output),
        )

    def inspect(self, lengths: TokenLengths) -> Optional[Tuple[RejectionReason, str]]:
        """Return why a record of these lengths is dropped, or None."""
        options = self.options
        if lengths.input_tokens > options.max_input_tokens:
            return (
                RejectionReason.INPUT_TOO_LONG,
                f"{lengths.input_tokens} > {options.max_input_tokens} input tokens",
            )
        if lengths.output_tokens > options.max_output_tokens:
            return (
                RejectionReason.OUTPUT_TOO_LONG,
                f"{lengths.output_tokens} > {options.max_output_tokens} output tokens",
            )
        if lengths.output_tokens < options.min_output_tokens:
            return (
                RejectionReason.OUTPUT_TOO_LONG,
                f"{lengths.output_tokens} < {options.min_output_tokens} output tokens",
            )
        if options.max_total_tokens is not None and lengths.total > options.max_total_tokens:
            return (
                RejectionReason.INPUT_TOO_LONG,
                f"{lengths.total} > {options.max_total_tokens} total tokens",
            )
        return None

    def run(self, records: Iterable[TrainingRecord]) -> TokenizerFilterResult:
        """Measure every record, keep those that fit, report the distribution."""
        result = TokenizerFilterResult()
        measured_inputs: List[int] = []
        measured_outputs: List[int] = []
        measured_totals: List[int] = []

        for record in records:
            lengths = self.measure(record)
            measured_inputs.append(lengths.input_tokens)
            measured_outputs.append(lengths.output_tokens)
            measured_totals.append(lengths.total)

            verdict = self.inspect(lengths)
            annotated = record.with_tokens(lengths.input_tokens, lengths.output_tokens)
            if verdict is None:
                result.kept.append(annotated)
                continue
            reason, detail = verdict
            result.rejections.append(
                Rejection(
                    bug_id=record.bug_id,
                    reason=reason,
                    detail=detail,
                    stage=self.stage_name,
                )
            )

        buckets = self.options.histogram_buckets
        result.stats = {
            "input": TokenStats.from_values(measured_inputs, buckets),
            "output": TokenStats.from_values(measured_outputs, buckets),
            "total": TokenStats.from_values(measured_totals, buckets),
            "input_kept": TokenStats.from_values(
                [r.input_tokens or 0 for r in result.kept], buckets
            ),
        }
        log.debug(
            "tokenizer_filter(%s): kept %d of %d",
            getattr(self.counter, "name", "custom"),
            len(result.kept),
            result.received,
        )
        return result


def filter_by_length(
    records: Iterable[TrainingRecord],
    counter: Optional[TokenCounter] = None,
    options: Optional[TokenizerFilterOptions] = None,
) -> TokenizerFilterResult:
    """Convenience wrapper around :class:`TokenizerFilter`."""
    return TokenizerFilter(counter, options).run(records)
