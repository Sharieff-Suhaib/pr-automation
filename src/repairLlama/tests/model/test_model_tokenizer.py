"""Tokenizer loading, pad-token handling and representation tokens."""

from __future__ import annotations

from pathlib import Path

import pytest

from repairllama.model.loader import ModelNotAvailableError
from repairllama.model.tokenizer import (
    TokenizerError,
    describe_tokenizer,
    ensure_pad_token,
    ensure_representation_tokens,
    load_tokenizer,
    log_tokenizer_summary,
)

pytest.importorskip("transformers")


def test_load_from_a_local_directory(tiny_tokenizer_dir: Path) -> None:
    tokenizer, changes = load_tokenizer(str(tiny_tokenizer_dir))
    assert tokenizer is not None
    assert changes.padding_side == "right"


def test_a_pad_token_is_guaranteed(tiny_tokenizer_dir: Path) -> None:
    """Llama tokenizers ship without one, and batching needs it."""
    tokenizer, changes = load_tokenizer(str(tiny_tokenizer_dir))
    assert tokenizer.pad_token is not None
    assert changes.pad_token_added is True
    assert changes.pad_token_source == "eos_token"


def test_pad_token_is_left_alone_when_present(tiny_tokenizer_dir: Path) -> None:
    tokenizer, _ = load_tokenizer(str(tiny_tokenizer_dir))
    added, source = ensure_pad_token(tokenizer)
    assert added is False and source == ""


def test_padding_side_is_configurable(tiny_tokenizer_dir: Path) -> None:
    tokenizer, _ = load_tokenizer(str(tiny_tokenizer_dir), padding_side="left")
    assert tokenizer.padding_side == "left"


def test_an_invalid_padding_side_is_rejected(tiny_tokenizer_dir: Path) -> None:
    with pytest.raises(TokenizerError, match="padding_side"):
        load_tokenizer(str(tiny_tokenizer_dir), padding_side="middle")


def test_representation_tokens_are_added_when_missing(tiny_tokenizer_dir: Path) -> None:
    tokenizer, changes = load_tokenizer(
        str(tiny_tokenizer_dir), extra_tokens=("<FILL_ME>", "<NEW_MARKER>")
    )
    assert changes.special_tokens_added >= 1
    assert changes.needs_embedding_resize is True
    assert len(tokenizer.encode("<NEW_MARKER>", add_special_tokens=False)) == 1


def test_tokens_already_in_the_vocabulary_are_not_re_added(tiny_tokenizer_dir: Path) -> None:
    tokenizer, _ = load_tokenizer(str(tiny_tokenizer_dir))
    added, names = ensure_representation_tokens(tokenizer, ["<FILL_ME>"])
    assert added >= 0
    second, _ = ensure_representation_tokens(tokenizer, ["<FILL_ME>"])
    assert second == 0


def test_no_extra_tokens_means_no_resize(tiny_tokenizer_dir: Path) -> None:
    _, changes = load_tokenizer(str(tiny_tokenizer_dir), extra_tokens=())
    assert changes.special_tokens_added == 0
    assert changes.needs_embedding_resize is False


def test_fill_token_is_a_single_token_after_loading(tiny_tokenizer_dir: Path) -> None:
    tokenizer, _ = load_tokenizer(str(tiny_tokenizer_dir), extra_tokens=("<FILL_ME>",))
    assert len(tokenizer.encode("<FILL_ME>", add_special_tokens=False)) == 1


def test_missing_tokenizer_explains_how_to_configure_it(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError) as info:
        load_tokenizer(str(tmp_path / "absent"))
    message = str(info.value)
    assert "downloading is disabled" in message
    assert "huggingface-cli download" in message


def test_describe_reports_the_vocabulary_and_padding(tiny_tokenizer_dir: Path) -> None:
    tokenizer, _ = load_tokenizer(str(tiny_tokenizer_dir))
    info = describe_tokenizer(tokenizer)
    assert info["total_vocab_size"] >= 1
    assert info["padding_side"] == "right"
    assert info["pad_token"] is not None
    assert info["eos_token"] == "</s>"


def test_summary_warns_about_a_needed_resize(
    tiny_tokenizer_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer, changes = load_tokenizer(
        str(tiny_tokenizer_dir), extra_tokens=("<ANOTHER_MARKER>",)
    )
    with caplog.at_level("WARNING", logger="repairllama.model.tokenizer"):
        log_tokenizer_summary(tokenizer, changes)
    assert any("resize_token_embeddings" in record.message for record in caplog.records)


def test_changes_serialise(tiny_tokenizer_dir: Path) -> None:
    _, changes = load_tokenizer(str(tiny_tokenizer_dir), extra_tokens=("<X_MARKER>",))
    payload = changes.to_dict()
    assert payload["needs_embedding_resize"] is True
    assert payload["padding_side"] == "right"
