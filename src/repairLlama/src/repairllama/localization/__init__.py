"""Fault localization: which region of the file to repair.

Planned modules:
    perfect.py     ground-truth localization from the reference diff
    stacktrace.py  parse JUnit stack traces into file/line candidates
    spectrum.py    coverage-based suspiciousness ranking
    regions.py     the ``BugRegion`` record shared by every strategy
"""
