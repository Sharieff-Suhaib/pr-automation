"""Generate candidate patches from the fine-tuned repair model.

CLI:
    python -m src.inference.generate_patch \
        --issue "get_first crashes on an empty list" \
        --code-file buggy.py --start-line 2 --end-line 2 --candidates 3

    # or read a JSON example: {"issue": ..., "buggy_code": ..., "fault_location": ...}
    python -m src.inference.generate_patch --example-file bug.json

    # skip the adapter to compare against the untuned base model
    python -m src.inference.generate_patch --no-adapter --code-file buggy.py

Python:
    repairer = PatchGenerator.from_config(load_config())
    patches = repairer.generate(issue="...", buggy_code="...")
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.dataset.loader import FaultLocation
from src.representation.repair_prompt import format_for_inference
from src.training.config import Config, GenerationConfig, load_config
from src.training.model_loader import (
    detect_device,
    load_base_model,
    load_model_with_adapter,
    load_tokenizer,
)

# Strip a ```python ... ``` wrapper if the model emits one despite instructions.
_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def clean_generated_code(text: str) -> str:
    """Normalize raw model output into plain code."""
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1)
    # Drop a stray leading header the model may echo from the prompt format.
    text = re.sub(r"^###\s*FIX\s*\n", "", text)
    return text.strip()


class PatchGenerator:
    """Loads the model once and generates candidate patches for many bugs."""

    def __init__(self, model, tokenizer, generation: GenerationConfig, max_seq_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.generation = generation
        self.max_seq_length = max_seq_length

    @classmethod
    def from_config(cls, config: Config, use_adapter: bool = True) -> "PatchGenerator":
        """Build a generator from a Config, with or without the LoRA adapter."""
        tokenizer = load_tokenizer(config.model)
        if use_adapter:
            model = load_model_with_adapter(config.model, config.generation.adapter_dir)
        else:
            print("[generate] running the BASE model without a LoRA adapter")
            model = load_base_model(config.model, for_training=False)
            model.eval()
        return cls(model, tokenizer, config.generation, config.model.max_seq_length)

    def build_prompt(
        self,
        issue: str | None,
        buggy_code: str,
        fault_location: FaultLocation | None = None,
        extras: dict | None = None,
    ) -> str:
        """Same representation used at training time — this must not diverge."""
        return format_for_inference(
            issue=issue,
            buggy_code=buggy_code,
            fault_location=fault_location,
            extras=extras,
            tokenizer=self.tokenizer,
        )

    def generate(
        self,
        issue: str | None,
        buggy_code: str,
        fault_location: FaultLocation | None = None,
        extras: dict | None = None,
        num_candidates: int | None = None,
    ) -> list[str]:
        """Return N cleaned candidate patches for one bug."""
        import torch

        num_candidates = num_candidates or self.generation.num_candidates
        prompt = self.build_prompt(issue, buggy_code, fault_location, extras)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Sampling is required for diverse candidates; a single candidate uses
        # greedy decoding by default for reproducibility.
        do_sample = self.generation.do_sample or num_candidates > 1
        gen_kwargs = {
            "max_new_tokens": self.generation.max_new_tokens,
            "num_return_sequences": num_candidates,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.generation.temperature
            gen_kwargs["top_p"] = self.generation.top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Slice off the prompt so only the newly generated fix remains.
        prompt_length = inputs["input_ids"].shape[1]
        return [
            clean_generated_code(
                self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            )
            for output in outputs
        ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate patches")
    parser.add_argument("--issue", default=None, help="Bug report / issue description")
    parser.add_argument("--code", default=None, help="Buggy code as an inline string")
    parser.add_argument("--code-file", default=None, help="File containing the buggy code")
    parser.add_argument("--example-file", default=None, help="JSON file with a full example")
    parser.add_argument("--start-line", type=int, default=None, help="Fault location start line")
    parser.add_argument("--end-line", type=int, default=None, help="Fault location end line")
    parser.add_argument("--candidates", type=int, default=None, help="Number of candidate patches")
    parser.add_argument(
        "--no-adapter", action="store_true", help="Use the base model without the LoRA adapter"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config()

    issue, buggy_code, location = args.issue, args.code, None

    if args.example_file:
        record = json.loads(Path(args.example_file).read_text(encoding="utf-8"))
        issue = issue or record.get("issue")
        buggy_code = buggy_code or record.get("buggy_code")
        raw_location = record.get("fault_location")
        if isinstance(raw_location, dict):
            location = FaultLocation(
                start_line=int(raw_location["start_line"]), end_line=int(raw_location["end_line"])
            )
    if args.code_file:
        buggy_code = Path(args.code_file).read_text(encoding="utf-8")
    if args.start_line is not None:
        location = FaultLocation(
            start_line=args.start_line, end_line=args.end_line or args.start_line
        )

    if not buggy_code:
        raise SystemExit("No buggy code provided. Use --code, --code-file, or --example-file.")

    generator = PatchGenerator.from_config(config, use_adapter=not args.no_adapter)
    patches = generator.generate(
        issue=issue,
        buggy_code=buggy_code,
        fault_location=location,
        num_candidates=args.candidates,
    )

    print(f"\n[generate] device={detect_device()}  candidates={len(patches)}")
    for index, patch in enumerate(patches, start=1):
        print("\n" + "=" * 70)
        print(f"CANDIDATE PATCH {index}/{len(patches)}")
        print("=" * 70)
        print(patch)
    print("=" * 70)


if __name__ == "__main__":
    main()
