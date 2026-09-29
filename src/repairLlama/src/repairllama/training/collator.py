"""Turning IR4 × OR2 rows into batches the model can be trained on.

A row from ``prepare-data`` is ``{"input": <IR4 prompt>, "output": <OR2 hunk>}``.
Training a causal LM on that pair means concatenating them and predicting only
the completion:

    input_ids = [BOS] prompt_tokens completion_tokens [EOS]
    labels    = -100  ... -100      completion_tokens [EOS]

**The prompt is masked out of the loss.**  Without that mask the model spends
most of its capacity learning to reproduce buggy Java — the prompt is far
longer than the fix — and the signal that matters, the replacement hunk, is
drowned out.  ``-100`` is torch's ignore index for cross-entropy.

Padding is dynamic (to the longest sequence in the batch, optionally rounded
up to a multiple of 8 for tensor cores), because padding every batch to 1024
would waste most of the compute on a corpus whose median example is far
shorter.

Over-long examples
------------------
``prepare-data`` already drops examples over the token budget, but a different
tokenizer at training time can produce longer sequences.  Rather than fail a
run mid-epoch, the collator truncates by ``truncation``:

``left`` (default)
    Drop tokens from the *start* of the prompt.  The end of the prompt holds
    the marked buggy region and the fill slot, which is what the completion
    depends on; the earliest context lines are the most expendable.
``right``
    Drop from the end of the completion — cheap, but it teaches the model to
    stop mid-hunk, so it is not the default.
``error``
    Raise, for a run that would rather fail than train on truncated data.

Every truncation is counted (:attr:`CompletionCollator.truncated`) and logged
once, so a silently-degraded run is visible in the training log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from repairllama.utils.logging import get_logger

__all__ = [
    "CollatorError",
    "CompletionCollator",
    "IGNORE_INDEX",
    "encode_example",
]

log = get_logger("training.collator")

#: torch's cross-entropy ignore index.
IGNORE_INDEX = -100


class CollatorError(ValueError):
    """Raised when an example cannot be encoded for training."""


def encode_example(
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    max_length: int = 1024,
    truncation: str = "left",
    add_bos: bool = True,
    add_eos: bool = True,
) -> Dict[str, List[int]]:
    """Encode one prompt/completion pair with the prompt masked out.

    Returns ``{"input_ids", "attention_mask", "labels"}`` as plain lists;
    padding happens later, per batch.
    """
    if truncation not in {"left", "right", "error"}:
        raise CollatorError(
            f"truncation must be 'left', 'right' or 'error', got {truncation!r}"
        )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    bos = getattr(tokenizer, "bos_token_id", None) if add_bos else None
    eos = getattr(tokenizer, "eos_token_id", None) if add_eos else None
    if bos is not None:
        prompt_ids = [bos, *prompt_ids]
    if eos is not None:
        # The EOS is part of the target: the model must learn where a hunk ends.
        completion_ids = [*completion_ids, eos]

    total = len(prompt_ids) + len(completion_ids)
    if total > max_length:
        if truncation == "error":
            raise CollatorError(
                f"example is {total} tokens, over max_length {max_length}; "
                "raise training.max_length or re-run prepare-data with a lower "
                "representation.max_input_tokens"
            )
        if truncation == "left":
            keep = max_length - len(completion_ids)
            if keep <= 0:
                # The completion alone overflows: keep its head and no prompt.
                completion_ids = completion_ids[: max_length - 1] + (
                    [eos] if eos is not None else []
                )
                prompt_ids = []
            else:
                # Keep the BOS, then the *last* tokens of the prompt.
                head = prompt_ids[:1] if bos is not None else []
                prompt_ids = head + prompt_ids[len(prompt_ids) - (keep - len(head)) :]
        else:  # right
            completion_ids = completion_ids[: max_length - len(prompt_ids)]

    input_ids = [*prompt_ids, *completion_ids]
    labels = [*([IGNORE_INDEX] * len(prompt_ids)), *completion_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


@dataclass
class CompletionCollator:
    """Batches ``{"input", "output"}`` rows, masking the prompt from the loss.

    Usable directly as a ``data_collator`` for ``transformers.Trainer``.
    """

    tokenizer: Any
    max_length: int = 1024
    prompt_key: str = "input"
    completion_key: str = "output"
    truncation: str = "left"
    pad_to_multiple_of: Optional[int] = 8
    add_bos: bool = True
    add_eos: bool = True
    truncated: int = field(default=0, init=False)
    seen: int = field(default=0, init=False)
    _warned: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.max_length < 8:
            raise CollatorError(f"max_length must be >= 8, got {self.max_length}")
        if self.tokenizer.pad_token_id is None:
            raise CollatorError(
                "the tokenizer has no pad token; load it through "
                "repairllama.model.load_tokenizer, which guarantees one"
            )

    # -- encoding ----------------------------------------------------------- #
    def encode(self, row: Mapping[str, Any]) -> Dict[str, List[int]]:
        """Encode one row, counting truncations."""
        try:
            prompt = row[self.prompt_key]
            completion = row[self.completion_key]
        except (KeyError, TypeError) as exc:
            raise CollatorError(
                f"row is missing {self.prompt_key!r}/{self.completion_key!r}; "
                f"got keys {sorted(row) if hasattr(row, 'keys') else type(row).__name__}"
            ) from exc

        encoded = encode_example(
            self.tokenizer,
            prompt,
            completion,
            max_length=self.max_length,
            truncation=self.truncation,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
        )
        self.seen += 1
        raw_length = len(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        ) + len(self.tokenizer.encode(completion, add_special_tokens=False))
        if raw_length + 2 > self.max_length:
            self.truncated += 1
            if not self._warned:
                self._warned = True
                log.warning(
                    "at least one example exceeds max_length=%d and is being "
                    "truncated (%s); the count is reported in the training summary",
                    self.max_length,
                    self.truncation,
                )
        return encoded

    # -- batching ----------------------------------------------------------- #
    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        if not rows:
            raise CollatorError("cannot collate an empty batch")
        encoded = [
            row if "input_ids" in row else self.encode(row)  # type: ignore[operator]
            for row in rows
        ]

        longest = max(len(item["input_ids"]) for item in encoded)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            longest = ((longest + multiple - 1) // multiple) * multiple
        longest = min(longest, self.max_length)

        pad_id = self.tokenizer.pad_token_id
        left = getattr(self.tokenizer, "padding_side", "right") == "left"

        input_ids, attention, labels = [], [], []
        for item in encoded:
            gap = longest - len(item["input_ids"])
            pads_ids = [pad_id] * gap
            pads_mask = [0] * gap
            pads_labels = [IGNORE_INDEX] * gap
            if left:
                input_ids.append(pads_ids + item["input_ids"])
                attention.append(pads_mask + item["attention_mask"])
                labels.append(pads_labels + item["labels"])
            else:
                input_ids.append(item["input_ids"] + pads_ids)
                attention.append(item["attention_mask"] + pads_mask)
                labels.append(item["labels"] + pads_labels)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # -- reporting ----------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        return {
            "examples_encoded": self.seen,
            "examples_truncated": self.truncated,
            "truncation": self.truncation,
            "max_length": self.max_length,
        }
