"""Tokenizer loading and the special-token setup the representation needs.

Three things have to be right before a CodeLlama tokenizer is usable for
supervised fine-tuning on IR4 × OR2 examples:

*A pad token.*  Llama-family tokenizers ship without one, and batching without
a pad token fails at the first multi-example batch.  :func:`ensure_pad_token`
reuses the EOS token by default (the standard choice — with a correct
attention mask, padding is masked out anyway) and says what it did.

*Padding side.*  Causal LMs want right padding for training and left padding
for batched generation, so it is configurable rather than assumed.

*The representation's markers.*  ``<FILL_ME>`` is native to CodeLlama's
infilling vocabulary, but a different base model may tokenize it as five
unrelated pieces.  :func:`ensure_representation_tokens` adds whatever is
missing and reports the count, because the caller must then resize the model's
embedding matrix.

Nothing here downloads: loading is ``local_files_only`` unless the caller
explicitly allows otherwise, and a missing tokenizer raises
:class:`~repairllama.model.loader.ModelNotAvailableError` with instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from repairllama.model.loader import ModelError
from repairllama.utils.logging import get_logger

__all__ = [
    "TokenizerError",
    "load_tokenizer",
    "ensure_pad_token",
    "ensure_representation_tokens",
    "describe_tokenizer",
    "log_tokenizer_summary",
]

log = get_logger("model.tokenizer")


class TokenizerError(ModelError):
    """Raised when a tokenizer cannot be loaded or configured.

    A subclass of :class:`~repairllama.model.loader.ModelError` so callers can
    handle every model-layer failure in one place.
    """


@dataclass(frozen=True)
class TokenizerChanges:
    """What :func:`load_tokenizer` had to change to make the tokenizer usable."""

    pad_token_added: bool = False
    pad_token_source: str = ""
    special_tokens_added: int = 0
    added_tokens: tuple = ()
    padding_side: str = ""

    @property
    def needs_embedding_resize(self) -> bool:
        """True when the model's embedding matrix must grow to match."""
        return self.special_tokens_added > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pad_token_added": self.pad_token_added,
            "pad_token_source": self.pad_token_source,
            "special_tokens_added": self.special_tokens_added,
            "added_tokens": list(self.added_tokens),
            "padding_side": self.padding_side,
            "needs_embedding_resize": self.needs_embedding_resize,
        }


def load_tokenizer(
    source: str,
    *,
    revision: Optional[str] = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    padding_side: str = "right",
    use_fast: bool = True,
    extra_tokens: Sequence[str] = (),
    add_pad_token: bool = True,
):
    """Load a tokenizer from a local directory or a cached checkpoint.

    Args:
        source: a local directory, or a HuggingFace repo id already in the
            local cache.  Nothing is downloaded unless ``local_files_only`` is
            False.
        padding_side: ``right`` for training, ``left`` for batched generation.
        extra_tokens: representation markers (``<FILL_ME>``, …) to guarantee
            exist as single tokens.

    Returns:
        ``(tokenizer, changes)`` — the changes say whether the caller must
        resize the model's embeddings.
    """
    if padding_side not in {"left", "right"}:
        raise TokenizerError(
            f"padding_side must be 'left' or 'right', got {padding_side!r}"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerError(
            "transformers is required to load a tokenizer. Install the model "
            'stack with `pip install -e ".[train]"`.'
        ) from exc

    from repairllama.model.loader import ModelNotAvailableError, describe_missing_model

    log.info("loading tokenizer from %s", source)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
        if _looks_like_a_missing_checkpoint(exc):
            raise ModelNotAvailableError(
                describe_missing_model(source, "tokenizer", exc)
            ) from exc
        raise TokenizerError(f"could not load the tokenizer from {source}: {exc}") from exc

    tokenizer.padding_side = padding_side
    pad_added, pad_source = (False, "")
    if add_pad_token:
        pad_added, pad_source = ensure_pad_token(tokenizer)
    added_count, added = ensure_representation_tokens(tokenizer, extra_tokens)

    changes = TokenizerChanges(
        pad_token_added=pad_added,
        pad_token_source=pad_source,
        special_tokens_added=added_count,
        added_tokens=tuple(added),
        padding_side=padding_side,
    )
    return tokenizer, changes


def _looks_like_a_missing_checkpoint(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "not a local folder",
        "is not the path to a directory",
        "does not appear to have a file named",
        "cannot find the requested files",
        "localentrynotfound",
        "offlinemode",
        "connection error",
        "couldn't connect",
        "no such file or directory",
        "we couldn't connect",
        "incorrect path_or_model_id",
        "repository not found",
        "repo id must be in the form",
    )
    return any(marker in text for marker in markers)


def ensure_pad_token(tokenizer) -> tuple:
    """Give the tokenizer a pad token if it has none.

    Prefers the EOS token, then UNK; only as a last resort does it add a new
    ``<pad>`` token (which would require resizing the model's embeddings).
    Returns ``(added, source)``.
    """
    if tokenizer.pad_token is not None:
        return (False, "")
    for attribute in ("eos_token", "unk_token"):
        candidate = getattr(tokenizer, attribute, None)
        if candidate:
            tokenizer.pad_token = candidate
            log.info("tokenizer had no pad token; reusing %s (%r)", attribute, candidate)
            return (True, attribute)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    log.warning(
        "tokenizer had no pad, eos or unk token; added '<pad>' — the model's "
        "embeddings must be resized to match"
    )
    return (True, "new")


def ensure_representation_tokens(tokenizer, tokens: Sequence[str]) -> tuple:
    """Add any of ``tokens`` the tokenizer does not already encode as one id.

    Returns ``(count_added, added_tokens)``.  A non-zero count means the
    model's embedding matrix must be resized before use.
    """
    missing: List[str] = []
    for token in tokens:
        if not token:
            continue
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            missing.append(token)
    if not missing:
        return (0, [])
    added = tokenizer.add_special_tokens({"additional_special_tokens": missing})
    log.info("added %d special token(s) to the tokenizer: %s", added, ", ".join(missing))
    return (added, missing)


def describe_tokenizer(tokenizer) -> Dict[str, Any]:
    """The facts worth printing about a loaded tokenizer."""
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "total_vocab_size": len(tokenizer),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "is_fast": getattr(tokenizer, "is_fast", None),
    }


def log_tokenizer_summary(tokenizer, changes: Optional[TokenizerChanges] = None) -> None:
    """Log vocabulary size, padding setup and anything that was changed."""
    info = describe_tokenizer(tokenizer)
    log.info(
        "tokenizer: %s | vocab %s | max length %s | padding %s",
        info["class"],
        info["total_vocab_size"],
        info["model_max_length"],
        info["padding_side"],
    )
    log.info(
        "special tokens: pad=%r eos=%r bos=%r",
        info["pad_token"],
        info["eos_token"],
        info["bos_token"],
    )
    if changes and changes.needs_embedding_resize:
        log.warning(
            "%d token(s) were added; call model.resize_token_embeddings(len(tokenizer))",
            changes.special_tokens_added,
        )
