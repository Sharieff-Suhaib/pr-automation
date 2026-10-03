"""Builds a tiny Llama checkpoint on disk, so the loader can be tested for real.

Downloading CodeLlama-7B in a test suite is not an option, and mocking
``from_pretrained`` would test the mock rather than the loader.  Instead these
helpers construct a ~200k-parameter Llama with the same architecture (so
``q_proj``/``v_proj`` exist for LoRA to target) plus a small word-level
tokenizer, and save both in the standard layout.  Everything is local; nothing
touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

VOCAB = [
    "<unk>", "<s>", "</s>", "<FILL_ME>",
    "public", "class", "int", "return", "if", "else", "for", "while",
    "a", "b", "x", "y", "(", ")", "{", "}", ";", "+", "-", ">", "<", "=",
    "//", "buggy", "lines", "start", "here", "end", "value", "max", "min",
]


def build_tokenizer(path: Path) -> Path:
    """Save a small word-level fast tokenizer at ``path``."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {token: index for index, token in enumerate(VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=512,
    )
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(path))
    return path


def build_model(path: Path, *, vocab_size: int = len(VOCAB)) -> Path:
    """Save a tiny ``LlamaForCausalLM`` at ``path``."""
    from transformers import LlamaConfig, LlamaForCausalLM

    # Big enough that a LoRA adapter is a small fraction of the whole (the
    # paper's r=8 on q_proj/v_proj is ~0.06% of a 7B model; on a 2-layer toy
    # it would be 8%, which is not a meaningful test), still tiny to load.
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def build_checkpoint(path: Path) -> Path:
    """Save a tiny model *and* tokenizer into one directory."""
    build_model(path)
    build_tokenizer(path)
    return path


def build_adapter(path: Path, checkpoint: Path, target_modules: Sequence[str] = ("q_proj",)) -> Path:
    """Train nothing, but save a valid PEFT adapter next to ``checkpoint``."""
    import peft
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True)
    config = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=list(target_modules), task_type="CAUSAL_LM"
    )
    wrapped = peft.get_peft_model(model, config)
    path.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(str(path))
    # PEFT writes into <path>/<adapter_name>; hand back the directory that
    # actually holds adapter_config.json.
    nested = path / "default"
    return nested if (nested / "adapter_config.json").is_file() else path
