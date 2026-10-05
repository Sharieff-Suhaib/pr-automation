from __future__ import annotations

import gc
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Prevent tokenizer multiprocessing crashes on Windows.
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

# Reduce CUDA memory fragmentation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================================
# Project setup
# ============================================================================

# Expected layout:
#
# pr-automation/
# └── src/
#     └── agents/
#         └── coding_agent/
#             └── fault_localization.py
#
# parents[3] points to pr-automation/

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from orchestrator.adapters import analyze_repository


# ============================================================================
# Configuration
# ============================================================================

# CodeLlama 7B only.
# Do not replace this with another model.
MODEL_ID = (
    "codellama/CodeLlama-7b-Instruct-hf"
)

HF_HOME = Path(
    os.getenv(
        "HF_HOME",
        r"D:\huggingface",
    )
)

HF_CACHE_DIR = Path(
    os.getenv(
        "HF_HUB_CACHE",
        str(HF_HOME / "hub"),
    )
)

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

FILL_TOKEN = "<FILL_ME>"

DEFAULT_TOP_K = 4

MAX_NEW_TOKENS = int(
    os.getenv(
        "REPAIR_MAX_NEW_TOKENS",
        "512",
    )
)

MAX_INPUT_TOKENS = int(
    os.getenv(
        "REPAIR_MAX_INPUT_TOKENS",
        "4096",
    )
)

# Use 4-bit quantization on CUDA to reduce VRAM usage.
# The model remains CodeLlama 7B.
USE_4BIT = os.getenv(
    "CODELLAMA_USE_4BIT",
    "1",
).lower() not in {
    "0",
    "false",
    "no",
}


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SuspiciousRegion:
    file: str
    start_line: int
    end_line: int
    score: float
    original_code: str
    ir4: str
    or2: str


# ============================================================================
# CodeLlama 7B model
# ============================================================================

class CodeLlamaRepairModel:
    """
    CodeLlama 7B Instruct repair model.

    CUDA:
        CodeLlama 7B is loaded with 4-bit quantization by default.

    CPU:
        CodeLlama 7B is loaded in float32. This requires substantial RAM.
    """

    def __init__(
        self,
        token: str | None = HF_TOKEN,
    ) -> None:
        self.model_id = MODEL_ID
        self.token = token

        # Ensure the configured model cannot accidentally be changed.
        if self.model_id != (
            "codellama/CodeLlama-7b-Instruct-hf"
        ):
            raise RuntimeError(
                "This script supports CodeLlama 7B only."
            )

        HF_HOME.mkdir(
            parents=True,
            exist_ok=True,
        )

        HF_CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Model: {self.model_id}",
            flush=True,
        )

        print(
            f"Cache directory: {HF_CACHE_DIR}",
            flush=True,
        )

        print(
            f"PyTorch version: {torch.__version__}",
            flush=True,
        )

        print(
            f"CUDA available: {torch.cuda.is_available()}",
            flush=True,
        )

        self._print_memory_information()

        print(
            "Loading CodeLlama tokenizer...",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.token,
            cache_dir=str(HF_CACHE_DIR),
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        print(
            "CodeLlama tokenizer loaded.",
            flush=True,
        )

        self.model = self._load_model()

        self.model.eval()

        print(
            "CodeLlama 7B loaded successfully.",
            flush=True,
        )

    def _print_memory_information(self) -> None:
        """
        Print available system and GPU memory.
        """

        try:
            import psutil

            memory = psutil.virtual_memory()

            print(
                "Available system RAM: "
                f"{memory.available / (1024 ** 3):.2f} GB",
                flush=True,
            )

        except ImportError:
            print(
                "psutil is not installed. "
                "RAM information is unavailable.",
                flush=True,
            )

        if not torch.cuda.is_available():
            return

        try:
            for index in range(
                torch.cuda.device_count()
            ):
                properties = (
                    torch.cuda.get_device_properties(index)
                )

                total_vram_gb = (
                    properties.total_memory
                    / (1024 ** 3)
                )

                free_bytes, total_bytes = (
                    torch.cuda.mem_get_info(index)
                )

                free_vram_gb = (
                    free_bytes
                    / (1024 ** 3)
                )

                print(
                    f"GPU {index}: {properties.name}",
                    flush=True,
                )

                print(
                    f"Total VRAM: {total_vram_gb:.2f} GB",
                    flush=True,
                )

                print(
                    f"Free VRAM: {free_vram_gb:.2f} GB",
                    flush=True,
                )

        except Exception as error:
            print(
                f"Could not inspect GPU memory: {error}",
                flush=True,
            )

    def _load_model(self):
        """
        Load CodeLlama 7B.

        CUDA uses 4-bit quantization by default.
        CPU uses float32.
        """

        if torch.cuda.is_available():
            return self._load_cuda_model()

        return self._load_cpu_model()

    def _load_cuda_model(self):
        """
        Load CodeLlama 7B on NVIDIA CUDA.

        This is still the CodeLlama 7B model. 4-bit quantization only
        changes the weight representation to reduce VRAM usage.
        """

        print(
            "Loading CodeLlama 7B on CUDA...",
            flush=True,
        )

        try:
            if USE_4BIT:
                print(
                    "Using 4-bit quantization.",
                    flush=True,
                )

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    token=self.token,
                    cache_dir=str(HF_CACHE_DIR),
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            else:
                print(
                    "Using float16 without quantization.",
                    flush=True,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    token=self.token,
                    cache_dir=str(HF_CACHE_DIR),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            print(
                "CodeLlama 7B CUDA model loaded.",
                flush=True,
            )

            return model

        except Exception as error:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            gc.collect()

            raise RuntimeError(
                "Failed to load CodeLlama 7B on CUDA.\n"
                "Check that CUDA, PyTorch, accelerate, and "
                "bitsandbytes are installed correctly.\n"
                f"Original error: {error}"
            ) from error

    def _load_cpu_model(self):
        """
        Load CodeLlama 7B on CPU.

        CodeLlama 7B in float32 can require more than 30 GB of RAM.
        If available RAM is too low, loading is stopped before the native
        PyTorch code can crash the process.
        """

        print(
            "CUDA is not available.",
            flush=True,
        )

        print(
            "CodeLlama 7B will be loaded on CPU.",
            flush=True,
        )

        print(
            "This requires substantial system RAM.",
            flush=True,
        )

        try:
            import psutil

            available_ram_gb = (
                psutil.virtual_memory().available
                / (1024 ** 3)
            )

            print(
                f"Available RAM before model load: "
                f"{available_ram_gb:.2f} GB",
                flush=True,
            )

            if available_ram_gb < 32:
                raise MemoryError(
                    "At least approximately 32 GB of available RAM "
                    "is recommended for CodeLlama 7B CPU loading. "
                    f"Only {available_ram_gb:.2f} GB is available."
                )

        except ImportError:
            print(
                "psutil is not installed. "
                "Skipping CPU memory safety check.",
                flush=True,
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.token,
                cache_dir=str(HF_CACHE_DIR),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

            model.to(
                torch.device("cpu")
            )

            print(
                "CodeLlama 7B CPU model loaded.",
                flush=True,
            )

            return model

        except Exception as error:
            gc.collect()

            raise RuntimeError(
                "Failed to load CodeLlama 7B on CPU. "
                "Insufficient RAM is the most likely cause.\n"
                f"Original error: {error}"
            ) from error

    def _build_prompt(
        self,
        issue: str,
        file_path: str,
        original_code: str,
        ir4_input: str,
    ) -> str:
        """
        Build a CodeLlama repair prompt.
        """

        system_message = (
            "You are an automated program repair system. "
            "Return only corrected source code."
        )

        user_message = f"""
Repair the suspicious code according to the issue.

Issue:
{issue}

File:
{file_path}

Original suspicious code:
{original_code}

IR4 input:
{ir4_input}

Output requirements:
1. Return only the replacement code.
2. Return code for the suspicious region only.
3. Do not return the complete file.
4. Do not return a unified diff.
5. Do not include explanations.
6. Do not include Markdown code fences.
7. Do not include labels such as OR2 OUTPUT.
8. Preserve the original programming language.
9. Preserve appropriate indentation.
10. Fix only the reported issue.
11. If the region is a complete function, return the corrected function.
"""

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            except Exception as error:
                print(
                    f"Chat template unavailable: {error}",
                    flush=True,
                )

        return (
            f"{system_message}\n\n"
            f"{user_message}\n\n"
            "### Response:\n"
        )

    def _get_input_device(self) -> torch.device:
        """
        Get the device used by the model input embeddings.

        This works with device_map='auto'.
        """

        try:
            return (
                self.model
                .get_input_embeddings()
                .weight
                .device
            )

        except Exception:
            if torch.cuda.is_available():
                return torch.device("cuda")

            return torch.device("cpu")

    def generate_or2(
        self,
        issue: str,
        file_path: str,
        original_code: str,
        ir4_input: str,
    ) -> str:
        """
        Generate the OR2 replacement code.
        """

        prompt = self._build_prompt(
            issue=issue,
            file_path=file_path,
            original_code=original_code,
            ir4_input=ir4_input,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        input_device = self._get_input_device()

        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[1]
        )

        print(
            "Generating OR2 with CodeLlama 7B...",
            flush=True,
        )

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        generated_tokens = generated[
            0,
            input_length:,
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return clean_or2_response(response)


# ============================================================================
# Text and path utilities
# ============================================================================

def normalize_path(
    value: str | Path,
) -> str:
    """
    Normalize Windows and Unix path separators.
    """

    return str(value).replace(
        "\\",
        "/",
    ).lstrip("./")


def tokenize(
    text: str,
) -> set[str]:
    """
    Extract meaningful tokens from the issue description.
    """

    ignored_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "where",
        "when",
        "into",
        "should",
        "could",
        "would",
        "have",
        "has",
        "does",
        "are",
        "was",
        "were",
        "bug",
        "buggy",
        "code",
        "issue",
        "function",
        "method",
        "file",
        "repository",
    }

    words = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        text.lower(),
    )

    return {
        word
        for word in words
        if len(word) > 2
        and word not in ignored_words
    }


def source_line_tokens(
    line: str,
) -> set[str]:
    """
    Extract source-code tokens from a line.
    """

    return set(
        re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            line.lower(),
        )
    )


def get_comment_prefix(
    file_path: str,
) -> str:
    """
    Return the comment prefix for a file type.
    """

    suffix = Path(
        file_path
    ).suffix.lower()

    if suffix in {
        ".py",
        ".rb",
        ".sh",
        ".yaml",
        ".yml",
    }:
        return "#"

    if suffix in {
        ".html",
        ".xml",
    }:
        return "<!--"

    return "//"


def clean_or2_response(
    response: str,
) -> str:
    """
    Remove labels and Markdown code fences from model output.
    """

    text = response.strip()

    labels = [
        "OR2 OUTPUT:",
        "OR2 OUTPUT",
        "FIXED CHUNK:",
        "FIXED CODE:",
        "REPLACEMENT CODE:",
        "ANSWER:",
    ]

    for label in labels:
        if text.upper().startswith(label):
            text = text[
                len(label):
            ].strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if (
            lines
            and lines[0].strip().startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    text = text.replace(
        FILL_TOKEN,
        "",
    ).strip()

    return text


# ============================================================================
# Fault-localization scoring
# ============================================================================

def heuristic_score(
    line: str,
) -> float:
    """
    Rank suspicious source lines.
    """

    stripped = line.strip()
    lowered = stripped.lower()

    if not stripped:
        return 0.0

    score = 0.0

    if "todo" in lowered:
        score += 3.0

    if "fixme" in lowered:
        score += 3.0

    if "bug" in lowered:
        score += 3.0

    if lowered == "pass":
        score += 2.0

    if stripped.startswith(
        (
            "#",
            "//",
        )
    ):
        if re.search(
            r"\b(if|else|for|while|return|try|except|catch|def|class)\b",
            lowered,
        ):
            score += 3.0

    if re.search(
        r"\breturn\s+(none|null|nil|false|0)\b",
        lowered,
    ):
        score += 1.0

    if "not in" in lowered:
        score += 0.5

    if "==" in lowered:
        score += 0.5

    if "!=" in lowered:
        score += 0.5

    return score


def score_lines(
    lines: list[str],
    issue: str,
    allowed_ranges: list[tuple[int, int]],
) -> list[float]:
    """
    Score lines within repository-agent-selected code ranges.
    """

    issue_tokens = tokenize(issue)
    scores: list[float] = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        is_allowed = any(
            start <= line_number <= end
            for start, end in allowed_ranges
        )

        if not is_allowed:
            scores.append(0.0)
            continue

        code_tokens = source_line_tokens(
            line
        )

        matching_tokens = (
            code_tokens.intersection(
                issue_tokens
            )
        )

        score = float(
            len(matching_tokens) * 2
        )

        score += heuristic_score(line)

        scores.append(score)

    return scores


# ============================================================================
# Repository-agent result handling
# ============================================================================

def get_relevant_files(
    repository_result: dict[str, Any],
) -> list[str]:
    """
    Get relevant files returned by the repository agent.
    """

    result: list[str] = []

    for file_path in repository_result.get(
        "relevant_files",
        [],
    ):
        file_path = str(
            file_path
        ).strip()

        if (
            file_path
            and file_path not in result
        ):
            result.append(file_path)

    if result:
        return result

    for node in repository_result.get(
        "relevant_code",
        [],
    ):
        if not isinstance(node, dict):
            continue

        file_path = str(
            node.get(
                "file",
                "",
            )
        ).strip()

        if (
            file_path
            and file_path not in result
        ):
            result.append(file_path)

    return result


def get_node_ranges_for_file(
    relevant_code: list[dict[str, Any]],
    file_path: str,
    line_count: int,
) -> list[tuple[int, int]]:
    """
    Get function and method ranges for a file.
    """

    target = normalize_path(
        file_path
    )

    function_ranges: list[
        tuple[int, int]
    ] = []

    fallback_ranges: list[
        tuple[int, int]
    ] = []

    for node in relevant_code:
        if not isinstance(node, dict):
            continue

        node_file = normalize_path(
            str(
                node.get(
                    "file",
                    "",
                )
            )
        )

        same_file = (
            node_file == target
            or Path(node_file).name
            == Path(target).name
        )

        if not same_file:
            continue

        start_line = node.get(
            "start_line"
        )

        end_line = node.get(
            "end_line"
        )

        if not isinstance(
            start_line,
            int,
        ):
            continue

        if not isinstance(
            end_line,
            int,
        ):
            continue

        start_line = max(
            1,
            start_line,
        )

        end_line = min(
            line_count,
            end_line,
        )

        if start_line > end_line:
            continue

        node_type = str(
            node.get(
                "type",
                "",
            )
        ).lower()

        if node_type in {
            "class",
            "interface",
            "struct",
            "module",
        }:
            fallback_ranges.append(
                (
                    start_line,
                    end_line,
                )
            )
        else:
            function_ranges.append(
                (
                    start_line,
                    end_line,
                )
            )

    if function_ranges:
        return function_ranges

    if fallback_ranges:
        return fallback_ranges

    return [
        (
            1,
            line_count,
        )
    ]


def resolve_file_inside_repository(
    repo_path: str,
    file_path: str,
) -> Path:
    """
    Resolve a repository file safely.
    """

    repository_root = Path(
        repo_path
    ).resolve()

    candidate = Path(
        file_path
    )

    possible_paths = [
        candidate,
        repository_root / file_path,
        repository_root / normalize_path(file_path),
    ]

    for possible_path in possible_paths:
        try:
            resolved = possible_path.resolve()

            if not resolved.is_file():
                continue

            resolved.relative_to(
                repository_root
            )

            return resolved

        except (
            FileNotFoundError,
            ValueError,
        ):
            continue

    raise FileNotFoundError(
        f"Could not find '{file_path}' inside "
        f"repository '{repository_root}'"
    )


# ============================================================================
# Suspicious-region selection and IR4
# ============================================================================

def select_suspicious_region(
    file_path: str,
    lines: list[str],
    scores: list[float],
    allowed_ranges: list[tuple[int, int]],
) -> tuple[int, int, float]:
    """
    Select the complete function or method containing the highest-scoring line.
    """

    if not lines:
        raise ValueError(
            f"File is empty: {file_path}"
        )

    candidates: list[int] = []

    for index in range(
        len(lines)
    ):
        line_number = index + 1

        if any(
            start <= line_number <= end
            for start, end in allowed_ranges
        ):
            candidates.append(index)

    if not candidates:
        candidates = list(
            range(
                len(lines)
            )
        )

    best_index = max(
        candidates,
        key=lambda index: scores[index],
    )

    best_score = scores[
        best_index
    ]

    for start, end in allowed_ranges:
        if start <= best_index + 1 <= end:
            return (
                start,
                end,
                best_score,
            )

    return (
        best_index + 1,
        best_index + 1,
        best_score,
    )


def build_ir4(
    lines: list[str],
    start_line: int,
    end_line: int,
    file_path: str,
) -> str:
    """
    Comment out the suspicious region and insert <FILL_ME>.
    """

    if start_line < 1:
        raise ValueError(
            "start_line must be at least 1"
        )

    if end_line > len(lines):
        raise ValueError(
            f"end_line {end_line} exceeds "
            f"{len(lines)} source lines"
        )

    if start_line > end_line:
        raise ValueError(
            "start_line must be <= end_line"
        )

    comment_prefix = get_comment_prefix(
        file_path
    )

    prefix = lines[
        :start_line - 1
    ]

    buggy_lines = lines[
        start_line - 1:end_line
    ]

    suffix = lines[
        end_line:
    ]

    commented_buggy_lines: list[str] = []

    for line in buggy_lines:
        if comment_prefix == "<!--":
            commented_buggy_lines.append(
                f"<!-- BUGGY CODE: {line} -->"
            )
        else:
            commented_buggy_lines.append(
                f"{comment_prefix} BUGGY CODE: {line}"
            )

    return "\n".join(
        prefix
        + commented_buggy_lines
        + [FILL_TOKEN]
        + suffix
    )


# ============================================================================
# Per-file localization
# ============================================================================

def localize_relevant_file(
    repo_path: str,
    file_path: str,
    issue: str,
    relevant_code: list[dict[str, Any]],
    repair_model: CodeLlamaRepairModel,
) -> SuspiciousRegion:
    """
    Localize one suspicious region and generate OR2 with CodeLlama 7B.
    """

    actual_path = resolve_file_inside_repository(
        repo_path=repo_path,
        file_path=file_path,
    )

    source = actual_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = source.splitlines()

    if not lines:
        raise ValueError(
            f"Relevant file is empty: {file_path}"
        )

    allowed_ranges = get_node_ranges_for_file(
        relevant_code=relevant_code,
        file_path=file_path,
        line_count=len(lines),
    )

    scores = score_lines(
        lines=lines,
        issue=issue,
        allowed_ranges=allowed_ranges,
    )

    start_line, end_line, score = (
        select_suspicious_region(
            file_path=file_path,
            lines=lines,
            scores=scores,
            allowed_ranges=allowed_ranges,
        )
    )

    original_code = "\n".join(
        lines[
            start_line - 1:end_line
        ]
    )

    ir4 = build_ir4(
        lines=lines,
        start_line=start_line,
        end_line=end_line,
        file_path=file_path,
    )

    print(
        f"Generating OR2 for {file_path}...",
        flush=True,
    )

    or2 = repair_model.generate_or2(
        issue=issue,
        file_path=file_path,
        original_code=original_code,
        ir4_input=ir4,
    )

    if not or2.strip():
        raise RuntimeError(
            f"CodeLlama returned empty OR2 "
            f"for {file_path}"
        )

    return SuspiciousRegion(
        file=file_path,
        start_line=start_line,
        end_line=end_line,
        score=score,
        original_code=original_code,
        ir4=ir4,
        or2=or2,
    )


# ============================================================================
# Main fault-localization pipeline
# ============================================================================

def run_fault_localization(
    repo_url: str,
    issue: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> list[SuspiciousRegion]:
    """
    Run repository-agent fault localization and CodeLlama repair generation.
    """

    print(
        f"Repository: {repo_url}",
        flush=True,
    )

    repository_result = analyze_repository(
        repo_url=repo_url,
        issue=issue,
        top_k=top_k,
    )

    repo_path = str(
        repository_result.get(
            "repo_path",
            "",
        )
    )

    relevant_code = repository_result.get(
        "relevant_code",
        [],
    )

    relevant_files = get_relevant_files(
        repository_result
    )

    effective_issue = str(
        repository_result.get(
            "issue",
            issue,
        )
    ).strip()

    if not effective_issue:
        effective_issue = (
            "Find suspicious, incomplete, or faulty code "
            "in the repository."
        )

    print()
    print(
        "REPOSITORY AGENT RESULT",
        flush=True,
    )

    print(
        "=======================",
        flush=True,
    )

    print(
        f"Repository path: {repo_path}",
        flush=True,
    )

    print(
        "Indexed chunks: "
        f"{repository_result.get('indexed_chunks', 0)}",
        flush=True,
    )

    print(
        f"Relevant files: {relevant_files}",
        flush=True,
    )

    if not repo_path:
        print(
            "The repository agent did not return a "
            "repository path.",
            flush=True,
        )

        return []

    if not relevant_files:
        print(
            "The repository agent returned no relevant files.",
            flush=True,
        )

        return []

    # CodeLlama 7B is loaded only after relevant files are found.
    repair_model = CodeLlamaRepairModel()

    results: list[SuspiciousRegion] = []

    for file_path in relevant_files:
        try:
            result = localize_relevant_file(
                repo_path=repo_path,
                file_path=file_path,
                issue=effective_issue,
                relevant_code=relevant_code,
                repair_model=repair_model,
            )

            results.append(result)

        except Exception as error:
            print(
                f"Could not localize or repair "
                f"'{file_path}': {error}",
                flush=True,
            )

    return results


# ============================================================================
# Output
# ============================================================================

def print_results(
    results: list[SuspiciousRegion],
) -> None:
    """
    Print fault-localization and CodeLlama results.
    """

    if not results:
        print()
        print(
            "No relevant files or suspicious code was found."
        )

        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            "=" * 80
        )

        print(
            f"FAULT-LOCALIZATION RESULT {index}"
        )

        print(
            "=" * 80
        )

        print(
            f"File: {result.file}"
        )

        print(
            f"Suspicious lines: "
            f"{result.start_line}-{result.end_line}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print()
        print(
            "ORIGINAL SUSPICIOUS CODE"
        )

        print(
            "-----------------------"
        )

        print(
            result.original_code
        )

        print()
        print(
            "IR4 INPUT"
        )

        print(
            "---------"
        )

        print(
            result.ir4
        )

        print()
        print(
            "OR2 OUTPUT"
        )

        print(
            "----------"
        )

        print(
            result.or2
        )


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    repo_url = (
        "https://github.com/Sharieff-Suhaib/dummy_repo.git"
    )

    issue = """
The login function incorrectly accepts hard-coded credentials.
It should validate credentials using the repository's user store
instead of comparing the username and password directly with
literal strings.
"""

    results = run_fault_localization(
        repo_url=repo_url,
        issue=issue,
        top_k=4,
    )

    print_results(results)


if __name__ == "__main__":
    main()