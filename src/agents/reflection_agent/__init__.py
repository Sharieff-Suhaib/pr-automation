"""Reflection Agent: turn a failed repair attempt into guidance for the next one."""

from .agent import (
    FAILURE_TYPES,
    MAX_REPAIR_ATTEMPTS,
    ReflectionAgent,
    ReflectionResult,
    classify_failure,
    failed_tests,
    parse_analysis,
)

__all__ = [
    "FAILURE_TYPES",
    "MAX_REPAIR_ATTEMPTS",
    "ReflectionAgent",
    "ReflectionResult",
    "classify_failure",
    "failed_tests",
    "parse_analysis",
]
