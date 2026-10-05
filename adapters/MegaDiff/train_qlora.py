"""
QLoRA fine-tuning of CodeLlama-7B on RepairLLaMA (ir4xor2).

pip install -U torch transformers peft bitsandbytes accelerate datasets

python train_qlora.py --output_dir ./codellama-7b-repairllama-ir4xor2
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "codellama/CodeLlama-7b-hf"
DATASET = "ASSERT-KTH/repairllama-datasets"
CONFIG = "ir4xor2"


def fim_ids(tokenizer):
    def get(attr, tok):
        val = getattr(tokenizer, attr, None)
        return val if val is not None else tokenizer.convert_tokens_to_ids(tok)

    return (
        get("prefix_id", "▁<PRE>"),
        get("suffix_id", "▁<SUF>"),
        get("middle_id", "▁<MID>"),
        get("eot_id", "▁<EOT>"),
    )


def build_prompt_ids(tokenizer, text):
    """ir4 input: buggy function with `<FILL_ME>` -> CodeLlama infilling prompt."""
    pre, suf, mid, _ = fim_ids(tokenizer)
    prefix, suffix = text.split("<FILL_ME>", 1)
    return (
        [tokenizer.bos_token_id, pre]
        + tokenizer.encode(prefix, add_special_tokens=False)
        + [suf]
        + tokenizer.encode(suffix, add_special_tokens=False)
        + [mid]
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--output_dir", default="./codellama-7b-repairllama-ir4xor2")
    p.add_argument("--max_length", type=int, default=1280)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=500)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    # parse_known_args ignores the `-f kernel.json` flag Jupyter/Colab passes
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    # ---------------- tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    _, _, _, eot = fim_ids(tokenizer)

    # ---------------- data ----------------
    ds = load_dataset(DATASET, CONFIG)
    train_ds, eval_ds = ds["train"], ds["test"]
    if args.max_train_samples:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.max_eval_samples, len(eval_ds))))

    def tokenize(example):
        prompt = build_prompt_ids(tokenizer, example["input"])
        target = tokenizer.encode(example["output"], add_special_tokens=False) + [eot]
        input_ids = prompt + target
        labels = [-100] * len(prompt) + target  # loss only on the fix
        return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}

    def has_fill(example):
        return "<FILL_ME>" in example["input"]

    cols = train_ds.column_names
    train_ds = train_ds.filter(has_fill).map(tokenize, remove_columns=cols, num_proc=4)
    eval_ds = eval_ds.filter(has_fill).map(tokenize, remove_columns=cols, num_proc=4)
    train_ds = train_ds.filter(lambda x: x["length"] <= args.max_length)
    eval_ds = eval_ds.filter(lambda x: x["length"] <= args.max_length)
    print(f"train: {len(train_ds)}  eval: {len(eval_ds)}")

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    # ---------------- model (4-bit) ----------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------- train ----------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float in [0, 1) = ratio of total steps
        weight_decay=0.0,
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
