"""Run candidate patches against unit tests and score functional correctness.

SAFETY
------
Generated code is executed in a separate short-lived process with a wall-clock
timeout, a scratch working directory, and a minimal environment. That is process
*isolation*, not sandboxing: the child still runs with your user's permissions
and can reach the filesystem and network. It is appropriate for curated
benchmark data (HumanEvalFix), and NOT appropriate for code generated from
scraped or untrusted inputs — that needs a container or gVisor-style sandbox.

This module is deliberately independent of any dataset so the later Agent-SWE
"Testing and Validation" stage can reuse it directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Prepended to every program. HumanEval solutions routinely rely on typing and
# math without importing them, and the reference `declaration` supplies those
# imports separately. Providing them here is deliberate leniency: a model should
# fail on a wrong *fix*, not on a forgotten import line.
DEFAULT_PREAMBLE = (
    "import math\n"
    "import re\n"
    "import itertools\n"
    "import collections\n"
    "from typing import List, Tuple, Dict, Set, Any, Optional, Union\n"
)


@dataclass
class ExecutionResult:
    """Outcome of running one candidate against its tests."""

    passed: bool
    status: str  # "passed" | "failed" | "timeout" | "error"
    detail: str = ""

    def as_dict(self) -> dict:
        return {"passed": self.passed, "status": self.status, "detail": self.detail}


def build_test_program(
    candidate_code: str,
    test_code: str,
    entry_point: str,
    imports: str = "",
    test_setup: str = "",
    preamble: str = DEFAULT_PREAMBLE,
) -> str:
    """Assemble a self-contained script: candidate + tests + the check() call.

    HumanEvalPack's `test` field defines `def check(candidate)` but never calls
    it, so the invocation has to be appended here.
    """
    parts = [preamble, imports, test_setup, candidate_code, test_code, f"check({entry_point})\n"]
    return "\n\n".join(part for part in parts if part and part.strip())


def run_program(program: str, timeout: float = 15.0) -> ExecutionResult:
    """Execute a program in a subprocess and classify the outcome."""
    with tempfile.TemporaryDirectory(prefix="agent_swe_exec_") as workdir:
        script_path = os.path.join(workdir, "candidate_test.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(program)

        # Minimal environment: no inherited PYTHONPATH, no venv surprises.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": workdir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }

        try:
            completed = subprocess.run(
                [sys.executable, script_path],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # Usually an infinite loop introduced by the "fix".
            return ExecutionResult(False, "timeout", f"exceeded {timeout}s")
        except Exception as exc:  # could not even spawn the process
            return ExecutionResult(False, "error", f"{type(exc).__name__}: {exc}")

        if completed.returncode == 0:
            return ExecutionResult(True, "passed")

        # Last stderr line is the useful one (the assertion or exception).
        stderr = (completed.stderr or "").strip()
        last_line = stderr.splitlines()[-1] if stderr else f"exit code {completed.returncode}"
        status = "failed" if "AssertionError" in stderr else "error"
        return ExecutionResult(False, status, last_line[:300])


def check_candidate(
    candidate_code: str,
    test_code: str,
    entry_point: str,
    imports: str = "",
    test_setup: str = "",
    timeout: float = 15.0,
) -> ExecutionResult:
    """Convenience wrapper: assemble the program and run it."""
    if not candidate_code.strip():
        return ExecutionResult(False, "error", "empty candidate")
    program = build_test_program(
        candidate_code, test_code, entry_point, imports=imports, test_setup=test_setup
    )
    return run_program(program, timeout=timeout)


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex paper (Chen et al., 2021).

    pass@k = 1 - C(n-c, k) / C(n, k) — the probability that at least one of k
    candidates drawn from n samples is correct. Reduces to `c/n` when k == 1.
    Computed as a running product to avoid overflow on large binomials.
    """
    if num_samples < k:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {num_samples}")
    if num_samples - num_correct < k:
        return 1.0
    # prod_{i=n-c+1}^{n} (1 - k/i)
    product = 1.0
    for i in range(num_samples - num_correct + 1, num_samples + 1):
        product *= 1.0 - k / i
    return 1.0 - product
