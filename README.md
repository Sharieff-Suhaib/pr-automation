# Agent-SWE — Program Repair Model (Milestone 1)

The fine-tuned program-repair component of Agent-SWE: a QLoRA-fine-tuned
`Qwen2.5-Coder-1.5B-Instruct` that learns

```
bug description + buggy code + repair context  ->  corrected code
```

Only this component is implemented. Fault localization, recommendation agents,
test validation, reflection/retry, and the multi-agent orchestration come later.

## Design notes (RepairLLaMA-inspired)

- **Repair-specific representation** rather than plain code completion: the model
  sees a structured brief (`### TASK / ### ISSUE / ### BUGGY CODE / ### FAULT
  LOCATION / ### INSTRUCTION / ### FIX`).
- **Fault localization in the input**: suspicious lines are numbered and marked
  inline with `>>>`, then restated as an explicit line range.
- **Buggy-to-fixed learning** via supervised fine-tuning on bug-fix pairs.
- **Parameter-efficient tuning**: 4-bit NF4 QLoRA where CUDA is available,
  standard LoRA elsewhere. The base model stays frozen.
- **Candidate patch generation**: N sampled candidates per bug, scored best-of-N.

## Layout

| Path | Purpose |
| --- | --- |
| [data/raw/sample_bugs.jsonl](data/raw/sample_bugs.jsonl) | 15 tiny Python bug-fix examples to prove the pipeline |
| [src/dataset/loader.py](src/dataset/loader.py) | JSONL -> `RepairExample`; tolerates missing optional fields |
| [src/dataset/preprocess.py](src/dataset/preprocess.py) | Split + render into trainer-ready records |
| [src/representation/repair_prompt.py](src/representation/repair_prompt.py) | The repair representation (single source of truth) |
| [src/training/config.py](src/training/config.py) | All settings, every one env-overridable |
| [src/training/model_loader.py](src/training/model_loader.py) | Model/tokenizer loading, quantization, LoRA attach |
| [src/training/train.py](src/training/train.py) | QLoRA fine-tuning via TRL `SFTTrainer` |
| [src/inference/generate_patch.py](src/inference/generate_patch.py) | Load base + adapter, generate candidate patches |
| [src/evaluation/evaluate.py](src/evaluation/evaluate.py) | Exact / AST / syntax scoring over the holdout |
| [scripts/test_pipeline.py](scripts/test_pipeline.py) | Smoke test — run this before any training |

## Setup

Python 3.10–3.12 (torch has no 3.14 wheels yet).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

```bash
# 1. Data stages only — no model download, runs anywhere
python scripts/test_pipeline.py

# 2. Full smoke test: tokenization, model load, one forward+backward pass
python scripts/test_pipeline.py --full

# 3. Two-step training run to confirm the loop works end to end
AGENT_SWE_MAX_STEPS=2 python -m src.training.train

# 4. Real (still small) fine-tune
python -m src.training.train

# 5. Generate a patch for unseen buggy code
python -m src.inference.generate_patch \
  --issue "Crashes with ZeroDivisionError when the list is empty" \
  --code "def mean(xs):
    return sum(xs) / len(xs)" \
  --start-line 2 --end-line 2 --candidates 3

# 6. Score the holdout set (add --no-adapter for the untuned baseline)
python -m src.evaluation.evaluate
```

## Configuration

Every setting in `src/training/config.py` reads an environment variable:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_SWE_BASE_MODEL` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Base model |
| `AGENT_SWE_TRAIN_FILE` | `data/raw/sample_bugs.jsonl` | Training JSONL |
| `AGENT_SWE_EVAL_SPLIT` | `0.2` | Holdout fraction when no eval file is given |
| `AGENT_SWE_OUTPUT_DIR` | `outputs/repair-lora` | Adapter destination |
| `AGENT_SWE_EPOCHS` / `AGENT_SWE_LR` | `3` / `2e-4` | Schedule |
| `AGENT_SWE_BATCH_SIZE` / `AGENT_SWE_GRAD_ACCUM` | `1` / `8` | Effective batch = 8 |
| `AGENT_SWE_MAX_SEQ_LEN` | `1024` | Truncation length |
| `AGENT_SWE_LORA_R` / `_ALPHA` / `_DROPOUT` | `16` / `32` / `0.05` | LoRA capacity |
| `AGENT_SWE_LOAD_IN_4BIT` | `true` | Auto-disabled without CUDA + bitsandbytes |
| `AGENT_SWE_MAX_STEPS` | `-1` | Cap optimizer steps (use for smoke runs) |
| `AGENT_SWE_NUM_CANDIDATES` | `1` | Candidate patches per bug |

## Hardware

| Environment | Mode |
| --- | --- |
| CUDA GPU (Colab T4, Kaggle P100) | 4-bit NF4 QLoRA — the intended path |
| Apple Silicon (MPS) | fp16 LoRA; bitsandbytes unavailable, fallback is automatic |
| CPU | fp32 LoRA; smoke tests only |

## Data format

Only `buggy_code` and `fixed_code` are required; everything else is optional.

```json
{
  "id": "example_001",
  "issue": "Function crashes when the input list is empty",
  "buggy_code": "def get_first(items):\n    return items[0]",
  "fixed_code": "def get_first(items):\n    if not items:\n        return None\n    return items[0]",
  "fault_location": {"start_line": 2, "end_line": 2}
}
```

`build_repair_prompt` already reserves optional sections for the later Agent-SWE
stages — `similar_bugs`, `repair_strategy`, `repository_context`,
`relevant_tests`, `tool_recommendations` — which render only when populated.
