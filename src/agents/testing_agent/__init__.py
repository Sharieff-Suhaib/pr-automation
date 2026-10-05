"""Testing Agent: decide whether a patch fixed the bug without breaking anything.

See `verdict.py` for the solved / unverified / failed rules, and
`reproduction.py` for the test written to reproduce the issue.
"""

from .agent import evaluate_patch, run_baseline, run_suite
from .feedback import build_feedback, is_better
from .junit import parse_junit_report, parse_junit_xml
from .reproduction import (
    REPRO_FILE,
    ReproductionError,
    ollama_writer,
    prepare_reproduction,
    stub_writer,
)
from .verdict import TestVerdict, compare_runs, not_tested

__all__ = [
    "evaluate_patch",
    "run_baseline",
    "run_suite",
    "build_feedback",
    "is_better",
    "parse_junit_report",
    "parse_junit_xml",
    "REPRO_FILE",
    "ReproductionError",
    "ollama_writer",
    "prepare_reproduction",
    "stub_writer",
    "TestVerdict",
    "compare_runs",
    "not_tested",
]
