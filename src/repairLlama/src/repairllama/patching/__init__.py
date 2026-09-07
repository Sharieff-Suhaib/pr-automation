"""Splicing a generated hunk back into real source files.

Planned modules:
    applier.py     apply a candidate to a file, honouring the patch strategy
    indentation.py re-indent generated code to match its insertion site
    validator.py   cheap syntactic sanity checks before compilation
    diffing.py     unified-diff rendering of an applied patch
"""
