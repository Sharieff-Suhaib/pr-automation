"""
Test the QLoRA-tuned CodeLlama-7B on RepairLLaMA ir4xor2 (exact-match on the test split),
or on your own buggy function containing `<FILL_ME>`.

python test_qlora.py --adapter ./codellama-7b-repairllama-ir4xor2 --num_samples 100
python test_qlora.py --adapter ./codellama-7b-repairllama-ir4xor2 --no_adapter      # base-model baseline
python test_qlora.py --adapter ./codellama-7b-repairllama-ir4xor2 --code_file bug.java
"""
import argparse
import json

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "codellama/CodeLlama-7b-hf"
DATASET = "ASSERT-KTH/repairllama-datasets"
CONFIG = "ir4xor2"

DEMO = """\tpublic SMSController(IDataStore s, TournamentController t) {
\t\t_tournament = t;
\t\t_sender = new SMSSender(s, this);
\t\t_parser = new SMSParser(s, this);
\t\tSMSReceiver receiver = new SMSReceiver(this);
\t\t_sendThread = new Thread(receiver);
// buggy code
// \t\t_sendThread.run();
\t\t<FILL_ME>
\t}
"""


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
    pre, suf, mid, _ = fim_ids(tokenizer)
    prefix, suffix = text.split("<FILL_ME>", 1)
    return (
        [tokenizer.bos_token_id, pre]
        + tokenizer.encode(prefix, add_special_tokens=False)
        + [suf]
        + tokenizer.encode(suffix, add_special_tokens=False)
        + [mid]
    )


def normalize(code):
    return "".join(code.split())


def load_model(args):
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, dtype=dtype, device_map="auto"
    )
    if not args.no_adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_fix(model, tokenizer, buggy, args):
    _, _, _, eot = fim_ids(tokenizer)
    ids = build_prompt_ids(tokenizer, buggy)[-args.max_input_length:]
    input_ids = torch.tensor([ids], device=model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        eos_token_id=[eot, tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
    )
    return [tokenizer.decode(seq[len(ids):], skip_special_tokens=True) for seq in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--adapter", default="./codellama-7b-repairllama-ir4xor2")
    p.add_argument("--no_adapter", action="store_true", help="evaluate the base model only")
    p.add_argument("--code_file", default=None, help="file with a buggy function containing <FILL_ME>")
    p.add_argument("--demo", action="store_true", help="run on a built-in example")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_input_length", type=int, default=1024)
    p.add_argument("--show", type=int, default=5, help="how many examples to print")
    p.add_argument("--save", default="predictions.jsonl")
    # parse_known_args ignores the `-f kernel.json` flag Jupyter/Colab passes
    args, _ = p.parse_known_args()

    model, tokenizer = load_model(args)

    # ---------- single input ----------
    if args.code_file or args.demo:
        buggy = open(args.code_file).read() if args.code_file else DEMO
        assert "<FILL_ME>" in buggy, "input must contain <FILL_ME> where the fix goes"
        fixes = generate_fix(model, tokenizer, buggy, args)
        for i, fix in enumerate(fixes):
            print(f"===== candidate {i + 1} =====\n{fix}")
        print("===== patched function =====")
        print(buggy.replace("<FILL_ME>", fixes[0].strip("\n")))
        return

    # ---------- test split ----------
    ds = load_dataset(DATASET, CONFIG, split="test")
    ds = ds.select(range(min(args.num_samples, len(ds))))

    exact, total = 0, 0
    with open(args.save, "w") as f:
        for i, ex in enumerate(ds):
            if "<FILL_ME>" not in ex["input"]:
                continue
            preds = generate_fix(model, tokenizer, ex["input"], args)
            hit = any(normalize(pr) == normalize(ex["output"]) for pr in preds)
            exact += hit
            total += 1
            f.write(json.dumps({"input": ex["input"], "target": ex["output"], "predictions": preds, "exact_match": hit}) + "\n")

            if i < args.show:
                print(f"\n========== #{i} {'✅' if hit else '❌'} ==========")
                print("--- buggy ---\n" + ex["input"])
                print("--- expected ---\n" + ex["output"])
                print("--- predicted ---\n" + preds[0])
            print(f"[{total}/{len(ds)}] exact match: {exact / total:.2%}", end="\r")

    print(f"\n\nExact match @{args.num_beams}: {exact}/{total} = {exact / max(total, 1):.2%}")
    print(f"Predictions saved to {args.save}")


if __name__ == "__main__":
    main()
