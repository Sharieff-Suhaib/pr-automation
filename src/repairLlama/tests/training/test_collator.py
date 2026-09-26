"""Batching: prompt masking, truncation, padding."""

from __future__ import annotations

import pytest

from repairllama.training.collator import (
    IGNORE_INDEX,
    CollatorError,
    CompletionCollator,
    encode_example,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _row(prompt: str = "public int f() { <FILL_ME> }", completion: str = "return a ;") -> dict:
    return {"input": prompt, "output": completion}


# --------------------------------------------------------------------------- #
# masking — the property the loss depends on
# --------------------------------------------------------------------------- #
def test_prompt_tokens_are_masked_out_of_the_loss(tiny_tokenizer) -> None:
    encoded = encode_example(tiny_tokenizer, "public int f ( )", "return a ;")
    labels = encoded["labels"]
    ids = encoded["input_ids"]

    assert len(labels) == len(ids)
    masked = [index for index, label in enumerate(labels) if label == IGNORE_INDEX]
    assert masked, "the prompt must be masked"
    # every unmasked label matches its input id: the model is scored on the
    # completion it is meant to produce, and on nothing else
    for index, label in enumerate(labels):
        if label != IGNORE_INDEX:
            assert label == ids[index]


def test_only_the_completion_is_scored(tiny_tokenizer) -> None:
    prompt = "public int f ( ) { <FILL_ME> }"
    completion = "return a ;"
    encoded = encode_example(tiny_tokenizer, prompt, completion)

    scored = [label for label in encoded["labels"] if label != IGNORE_INDEX]
    expected = tiny_tokenizer.encode(completion, add_special_tokens=False)
    assert scored[: len(expected)] == expected


def test_eos_is_part_of_the_target(tiny_tokenizer) -> None:
    """The model must learn where a hunk ends, so EOS is scored."""
    encoded = encode_example(tiny_tokenizer, "a", "b")
    assert encoded["labels"][-1] == tiny_tokenizer.eos_token_id
    assert encoded["input_ids"][-1] == tiny_tokenizer.eos_token_id


def test_bos_starts_the_sequence(tiny_tokenizer) -> None:
    encoded = encode_example(tiny_tokenizer, "a", "b")
    assert encoded["input_ids"][0] == tiny_tokenizer.bos_token_id
    assert encoded["labels"][0] == IGNORE_INDEX


def test_special_tokens_can_be_disabled(tiny_tokenizer) -> None:
    encoded = encode_example(tiny_tokenizer, "a", "b", add_bos=False, add_eos=False)
    assert encoded["input_ids"][0] != tiny_tokenizer.bos_token_id
    assert encoded["input_ids"][-1] != tiny_tokenizer.eos_token_id


def test_attention_mask_covers_the_whole_sequence(tiny_tokenizer) -> None:
    encoded = encode_example(tiny_tokenizer, "a b c", "d")
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])


# --------------------------------------------------------------------------- #
# truncation
# --------------------------------------------------------------------------- #
def test_left_truncation_keeps_the_completion_whole(tiny_tokenizer) -> None:
    prompt = "public class " * 200
    completion = "return a ;"
    encoded = encode_example(
        tiny_tokenizer, prompt, completion, max_length=32, truncation="left"
    )
    assert len(encoded["input_ids"]) <= 32
    scored = [label for label in encoded["labels"] if label != IGNORE_INDEX]
    expected = tiny_tokenizer.encode(completion, add_special_tokens=False)
    assert scored == [*expected, tiny_tokenizer.eos_token_id]


def test_left_truncation_keeps_the_end_of_the_prompt(tiny_tokenizer) -> None:
    """The fill slot is at the end of the prompt, so the end is what matters."""
    prompt = "int " * 100 + "<FILL_ME>"
    encoded = encode_example(
        tiny_tokenizer, prompt, "return a ;", max_length=24, truncation="left"
    )
    fill_id = tiny_tokenizer.encode("<FILL_ME>", add_special_tokens=False)[0]
    assert fill_id in encoded["input_ids"]


def test_right_truncation_cuts_the_completion(tiny_tokenizer) -> None:
    encoded = encode_example(
        tiny_tokenizer, "int " * 20, "return a ; " * 20, max_length=32, truncation="right"
    )
    assert len(encoded["input_ids"]) <= 32


def test_truncation_can_be_an_error(tiny_tokenizer) -> None:
    with pytest.raises(CollatorError, match="over max_length"):
        encode_example(
            tiny_tokenizer, "int " * 100, "return a ;", max_length=16, truncation="error"
        )


def test_unknown_truncation_is_rejected(tiny_tokenizer) -> None:
    with pytest.raises(CollatorError, match="truncation must be"):
        encode_example(tiny_tokenizer, "a", "b", truncation="middle")


def test_short_examples_are_untouched(tiny_tokenizer) -> None:
    encoded = encode_example(tiny_tokenizer, "int a ;", "return a ;", max_length=1024)
    assert len(encoded["input_ids"]) < 1024


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
def test_batch_shapes_and_dtypes(tiny_tokenizer) -> None:
    collator = CompletionCollator(tiny_tokenizer, max_length=64)
    batch = collator([_row(), _row("int g ( )", "return b ;")])

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].dtype == torch.long


def test_padding_is_masked_everywhere(tiny_tokenizer) -> None:
    collator = CompletionCollator(tiny_tokenizer, max_length=64, pad_to_multiple_of=None)
    batch = collator([_row("a", "b"), _row("int " * 10, "return a ; ")])

    pad_positions = batch["attention_mask"] == 0
    assert pad_positions.any(), "the shorter row must be padded"
    assert (batch["labels"][pad_positions] == IGNORE_INDEX).all()
    assert (batch["input_ids"][pad_positions] == tiny_tokenizer.pad_token_id).all()


def test_padding_is_dynamic_not_max_length(tiny_tokenizer) -> None:
    """Padding every batch to max_length would waste most of the compute."""
    collator = CompletionCollator(tiny_tokenizer, max_length=512, pad_to_multiple_of=None)
    batch = collator([_row("a", "b")])
    assert batch["input_ids"].shape[1] < 32


def test_pad_to_multiple_of(tiny_tokenizer) -> None:
    collator = CompletionCollator(tiny_tokenizer, max_length=128, pad_to_multiple_of=8)
    batch = collator([_row("a", "b")])
    assert batch["input_ids"].shape[1] % 8 == 0


def test_batch_never_exceeds_max_length(tiny_tokenizer) -> None:
    collator = CompletionCollator(tiny_tokenizer, max_length=16)
    batch = collator([_row("int " * 100, "return a ;"), _row("a", "b")])
    assert batch["input_ids"].shape[1] <= 16


def test_left_padding_is_honoured(tiny_tokenizer) -> None:
    original = tiny_tokenizer.padding_side
    tiny_tokenizer.padding_side = "left"
    try:
        collator = CompletionCollator(tiny_tokenizer, max_length=64, pad_to_multiple_of=None)
        batch = collator([_row("a", "b"), _row("int " * 10, "return a ;")])
        assert batch["attention_mask"][0][0] == 0
    finally:
        tiny_tokenizer.padding_side = original


def test_empty_batch_is_an_error(tiny_tokenizer) -> None:
    with pytest.raises(CollatorError, match="empty batch"):
        CompletionCollator(tiny_tokenizer)([])


def test_missing_keys_are_reported(tiny_tokenizer) -> None:
    with pytest.raises(CollatorError, match="missing"):
        CompletionCollator(tiny_tokenizer)([{"prompt": "a"}])


def test_a_tokenizer_without_a_pad_token_is_rejected() -> None:
    class _NoPadTokenizer:
        pad_token_id = None

    with pytest.raises(CollatorError, match="no pad token"):
        CompletionCollator(_NoPadTokenizer())


def test_max_length_must_be_sane(tiny_tokenizer) -> None:
    with pytest.raises(CollatorError, match="max_length"):
        CompletionCollator(tiny_tokenizer, max_length=2)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_truncations_are_counted(tiny_tokenizer) -> None:
    collator = CompletionCollator(tiny_tokenizer, max_length=16)
    collator([_row("int " * 100, "return a ;"), _row("a", "b")])
    stats = collator.stats()
    assert stats["examples_encoded"] == 2
    assert stats["examples_truncated"] == 1
    assert stats["max_length"] == 16


def test_nothing_is_reported_truncated_when_it_fits(tiny_tokenizer) -> None:
    collator = CompletionCollator(tiny_tokenizer, max_length=512)
    collator([_row(), _row()])
    assert collator.stats()["examples_truncated"] == 0
