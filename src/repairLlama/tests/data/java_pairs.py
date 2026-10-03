"""Java corpora fixtures for the data-pipeline tests.

The helpers here build :class:`BugFixPair` objects and raw corpus files, so
each test can state the shape it needs (a duplicate, a multi-method change, a
test file, a huge diff) without repeating Java source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from repairllama.data.models import BugFixPair

# --------------------------------------------------------------------------- #
# single-method class, one buggy line (line 5)
# --------------------------------------------------------------------------- #
CALCULATOR_BUGGY = """\
public class Calculator {
    // returns the larger of two values
    public int max(int a, int b) {
        if (a > b) {
            return b;
        }
        return b;
    }
}
"""

CALCULATOR_FIXED = CALCULATOR_BUGGY.replace(
    "        if (a > b) {\n            return b;",
    "        if (a > b) {\n            return a;",
)

# --------------------------------------------------------------------------- #
# two methods; the fix touches both (rejected as not-single-function)
# --------------------------------------------------------------------------- #
TWO_METHODS_BUGGY = """\
public class Pair {
    public int first(int[] xs) {
        return xs[1];
    }

    public int last(int[] xs) {
        return xs[xs.length];
    }
}
"""

TWO_METHODS_FIXED = """\
public class Pair {
    public int first(int[] xs) {
        return xs[0];
    }

    public int last(int[] xs) {
        return xs[xs.length - 1];
    }
}
"""

# --------------------------------------------------------------------------- #
# no method at all: a field initialiser change
# --------------------------------------------------------------------------- #
FIELD_ONLY_BUGGY = """\
public class Config {
    public static final int LIMIT = 10;
    public static final String NAME = "config";
}
"""

FIELD_ONLY_FIXED = FIELD_ONLY_BUGGY.replace("LIMIT = 10", "LIMIT = 100")

# --------------------------------------------------------------------------- #
# a JUnit test class
# --------------------------------------------------------------------------- #
TEST_CLASS_BUGGY = """\
public class CalculatorTest {
    @Test
    public void testMax() {
        assertEquals(2, new Calculator().max(1, 2));
    }
}
"""

TEST_CLASS_FIXED = TEST_CLASS_BUGGY.replace("assertEquals(2,", "assertEquals(3,")

# --------------------------------------------------------------------------- #
# a method whose braces live inside comments and strings
# --------------------------------------------------------------------------- #
TRICKY_BRACES_BUGGY = """\
public class Tricky {
    public String render(int n) {
        // a closing brace in a comment: }
        String template = "value: { }";
        return template + n + '}';
    }
}
"""

TRICKY_BRACES_FIXED = TRICKY_BRACES_BUGGY.replace(
    "return template + n + '}';", "return template + (n + 1) + '}';"
)


def big_diff_pair(bug_id: str = "big", changed_lines: int = 40) -> BugFixPair:
    """A pair whose fix rewrites ``changed_lines`` lines of one method."""
    body_buggy = "\n".join(f"        total += {i};" for i in range(changed_lines))
    body_fixed = "\n".join(f"        total -= {i};" for i in range(changed_lines))
    template = "public class Big {{\n    public int run() {{\n        int total = 0;\n{}\n        return total;\n    }}\n}}\n"
    return BugFixPair(
        bug_id=bug_id,
        buggy_code=template.format(body_buggy),
        fixed_code=template.format(body_fixed),
        project="big",
    )


def calculator_pair(
    bug_id: str = "calc-1",
    project: Optional[str] = "calculator",
    file_path: Optional[str] = "src/main/java/Calculator.java",
    **kwargs: Any,
) -> BugFixPair:
    """The one-line Calculator bug (buggy line 5)."""
    return BugFixPair(
        bug_id=bug_id,
        buggy_code=CALCULATOR_BUGGY,
        fixed_code=CALCULATOR_FIXED,
        project=project,
        file_path=file_path,
        **kwargs,
    )


def variant_pair(bug_id: str, marker: int, project: str = "variants") -> BugFixPair:
    """A distinct one-line bug, so a corpus can have many unique pairs."""
    buggy = (
        "public class Variant%d {\n"
        "    public int value(int a) {\n"
        "        return a - %d;\n"
        "    }\n"
        "}\n" % (marker, marker)
    )
    fixed = buggy.replace(f"return a - {marker};", f"return a + {marker};")
    return BugFixPair(
        bug_id=bug_id,
        buggy_code=buggy,
        fixed_code=fixed,
        project=project,
        file_path=f"src/main/java/Variant{marker}.java",
    )


def corpus(count: int = 12, projects: int = 4) -> List[BugFixPair]:
    """A small corpus of unique pairs spread over several projects."""
    return [
        variant_pair(f"bug-{index:03d}", index, project=f"project-{index % projects}")
        for index in range(count)
    ]


def as_records(pairs: Iterable[BugFixPair]) -> List[Dict[str, Any]]:
    return [pair.to_dict() for pair in pairs]


def write_jsonl_corpus(path: Path, pairs: Iterable[BugFixPair]) -> Path:
    """Write pairs as a JSONL corpus and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict()) + "\n")
    return path


def write_directory_corpus(
    root: Path,
    pairs: Iterable[BugFixPair],
    buggy_name: str = "buggy.java",
    fixed_name: str = "fixed.java",
    with_metadata: bool = False,
) -> Path:
    """Write pairs as ``<root>/<bug_id>/{buggy,fixed}.java`` and return root."""
    for pair in pairs:
        directory = root / pair.bug_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / buggy_name).write_text(pair.buggy_code, encoding="utf-8")
        (directory / fixed_name).write_text(pair.fixed_code, encoding="utf-8")
        if with_metadata:
            (directory / "metadata.json").write_text(
                json.dumps({"project": pair.project, "metadata": {"origin": "test"}}),
                encoding="utf-8",
            )
    return root
