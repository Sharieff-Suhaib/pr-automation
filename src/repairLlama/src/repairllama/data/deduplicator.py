"""Duplicate removal.

Mined corpora repeat themselves: the same fix appears in forks, in
cherry-picks, and across releases of the same file.  Training on those repeats
inflates the apparent dataset size and, worse, leaks identical examples across
the train/test boundary — so deduplication runs *before* splitting.

Three strategies, from strictest to loosest:

``exact``
    The buggy and fixed text must match byte-for-byte.

``normalized`` (default)
    Comments are stripped and whitespace collapsed before comparing, so
    reindented or re-commented copies of the same fix collapse together.

``input_only``
    Keyed on the buggy side alone.  This catches the harmful case of one
    buggy input with several different fixes, which trains the model on
    contradictory targets.

Deduplication is order-deterministic: the first occurrence is kept and later
ones are recorded as duplicates of it, so a stable input order gives a stable
dataset.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from repairllama.data.cleaner import strip_comments
from repairllama.data.models import BugFixPair, Rejection, RejectionReason
from repairllama.utils.logging import get_logger

__all__ = [
    "DedupStrategy",
    "DedupResult",
    "Deduplicator",
    "normalize_code",
    "fingerprint",
    "deduplicate",
]

log = get_logger("data.deduplicator")

_WHITESPACE_RE = re.compile(r"\s+")

STRATEGIES = ("exact", "normalized", "input_only")
DedupStrategy = str


def normalize_code(source: str) -> str:
    """Comment-free, whitespace-collapsed form used for fuzzy matching.

    Whitespace inside string literals is preserved (the literals are kept),
    so two snippets differing only in a message string stay distinct.
    """
    return _WHITESPACE_RE.sub(" ", strip_comments(source)).strip()


def fingerprint(pair: BugFixPair, strategy: DedupStrategy = "normalized") -> str:
    """Stable hash identifying ``pair`` under ``strategy``."""
    if strategy == "exact":
        payload = f"{pair.buggy_code}\x00{pair.fixed_code}"
    elif strategy == "normalized":
        payload = f"{normalize_code(pair.buggy_code)}\x00{normalize_code(pair.fixed_code)}"
    elif strategy == "input_only":
        region = f"{pair.suspicious_start}:{pair.suspicious_end}"
        payload = f"{normalize_code(pair.buggy_code)}\x00{region}"
    else:
        raise ValueError(f"unknown dedup strategy {strategy!r}; expected one of {STRATEGIES}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class DedupResult:
    """Survivors, duplicates, and which sample each duplicate mirrored."""

    kept: List[BugFixPair] = field(default_factory=list)
    rejections: List[Rejection] = field(default_factory=list)
    duplicate_of: Dict[str, str] = field(default_factory=dict)

    @property
    def received(self) -> int:
        return len(self.kept) + len(self.rejections)

    @property
    def duplicates(self) -> int:
        return len(self.rejections)

    def counts_by_reason(self) -> Dict[str, int]:
        return {RejectionReason.DUPLICATE.value: self.duplicates} if self.duplicates else {}


class Deduplicator:
    """Removes repeated bug/fix pairs, keeping the first occurrence."""

    stage_name = "deduplicate"

    def __init__(self, strategy: DedupStrategy = "normalized") -> None:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown dedup strategy {strategy!r}; expected one of {STRATEGIES}"
            )
        self.strategy = strategy
        self._seen: Dict[str, str] = {}

    def reset(self) -> None:
        self._seen.clear()

    def check(self, pair: BugFixPair) -> Optional[str]:
        """Return the bug_id this pair duplicates, or None if it is new.

        Records the pair as seen either way, so repeated calls behave like a
        single pass over the corpus.
        """
        key = fingerprint(pair, self.strategy)
        original = self._seen.get(key)
        if original is None:
            self._seen[key] = pair.bug_id
            return None
        return original

    def run(self, pairs: Iterable[BugFixPair]) -> DedupResult:
        result = DedupResult()
        for pair in pairs:
            original = self.check(pair)
            if original is None:
                result.kept.append(pair)
                continue
            result.duplicate_of[pair.bug_id] = original
            result.rejections.append(
                Rejection(
                    bug_id=pair.bug_id,
                    reason=RejectionReason.DUPLICATE,
                    detail=f"duplicate of {original} ({self.strategy})",
                    stage=self.stage_name,
                )
            )
        log.debug(
            "deduplicate(%s): kept %d of %d",
            self.strategy,
            len(result.kept),
            result.received,
        )
        return result


def deduplicate(
    pairs: Iterable[BugFixPair], strategy: DedupStrategy = "normalized"
) -> DedupResult:
    """Convenience wrapper around :class:`Deduplicator`."""
    return Deduplicator(strategy).run(pairs)
