"""Utilities for generating software-repair patches."""

from .code_generator import generate_patch, generate_patch_from_repository
from .fault_localization import SuspiciousRegion, run_fault_localization
from .tester import detect_framework, run_tests, select_test_command

__all__ = [
    "generate_patch",
    "generate_patch_from_repository",
    "SuspiciousRegion",
    "run_fault_localization",
    "run_tests",
    "detect_framework",
    "select_test_command",
]
