"""Java bug/fix fixtures for the representation tests.

Each :class:`Sample` records the buggy unit, the fixed unit, the suspicious
region (1-based, inclusive) and the exact replacement text expected as the
OR2 target.  Line numbers in the comments below are the real ones, so a test
failure can be read against the source without counting lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Sample:
    name: str
    buggy: str
    fixed: str
    start: int
    end: int
    target: str

    @property
    def buggy_lines(self) -> List[str]:
        return self.buggy.splitlines()

    @property
    def fixed_lines(self) -> List[str]:
        return self.fixed.splitlines()

    @property
    def target_lines(self) -> List[str]:
        return self.target.splitlines() if self.target else []

    @property
    def region(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @property
    def original_region(self) -> str:
        return "\n".join(self.buggy_lines[self.start - 1 : self.end])


# --------------------------------------------------------------------------- #
# 1. one-line bug: wrong variable returned on line 4
# --------------------------------------------------------------------------- #
ONE_LINE_BUGGY = """\
public class Calculator {
    public int max(int a, int b) {
        if (a > b) {
            return b;
        }
        return b;
    }
}
"""

ONE_LINE_FIXED = """\
public class Calculator {
    public int max(int a, int b) {
        if (a > b) {
            return a;
        }
        return b;
    }
}
"""

ONE_LINE = Sample(
    name="one_line",
    buggy=ONE_LINE_BUGGY,
    fixed=ONE_LINE_FIXED,
    start=4,
    end=4,
    target="            return a;",
)

# Same bug, but the "fix" also rewrites line 6 — outside the marked region.
ONE_LINE_FIXED_WITH_OUTSIDE_EDIT = """\
public class Calculator {
    public int max(int a, int b) {
        if (a > b) {
            return a;
        }
        return Math.min(a, b);
    }
}
"""


# --------------------------------------------------------------------------- #
# 2. multi-line bug: off-by-one guard and accumulation, lines 4-5
# --------------------------------------------------------------------------- #
MULTI_LINE_BUGGY = """\
public class Sum {
    public int sum(int[] values) {
        int total = 0;
        for (int i = 0; i <= values.length; i++) {
            total += values[i];
        }
        return total;
    }
}
"""

MULTI_LINE_FIXED = """\
public class Sum {
    public int sum(int[] values) {
        int total = 0;
        for (int i = 0; i < values.length; i++) {
            total = total + values[i];
        }
        return total;
    }
}
"""

MULTI_LINE = Sample(
    name="multi_line",
    buggy=MULTI_LINE_BUGGY,
    fixed=MULTI_LINE_FIXED,
    start=4,
    end=5,
    target=(
        "        for (int i = 0; i < values.length; i++) {\n"
        "            total = total + values[i];"
    ),
)


# --------------------------------------------------------------------------- #
# 3. nested if/else: wrong grade on line 6, region spans the whole branch 5-7
# --------------------------------------------------------------------------- #
NESTED_IF_BUGGY = """\
public class Grade {
    public String classify(int score) {
        if (score >= 90) {
            return "A";
        } else if (score >= 80) {
            return "C";
        } else {
            if (score >= 70) {
                return "C";
            } else {
                return "F";
            }
        }
    }
}
"""

NESTED_IF_FIXED = """\
public class Grade {
    public String classify(int score) {
        if (score >= 90) {
            return "A";
        } else if (score >= 80) {
            return "B";
        } else {
            if (score >= 70) {
                return "C";
            } else {
                return "F";
            }
        }
    }
}
"""

NESTED_IF = Sample(
    name="nested_if_else",
    buggy=NESTED_IF_BUGGY,
    fixed=NESTED_IF_FIXED,
    start=5,
    end=7,
    target=(
        "        } else if (score >= 80) {\n"
        '            return "B";\n'
        "        } else {"
    ),
)


# --------------------------------------------------------------------------- #
# 4. method with nested loops: wrong bound on the inner loop, line 5
# --------------------------------------------------------------------------- #
LOOPS_BUGGY = """\
public class Matrix {
    public int sum(int[][] grid) {
        int total = 0;
        for (int row = 0; row < grid.length; row++) {
            for (int col = 0; col < grid.length; col++) {
                total += grid[row][col];
            }
        }
        return total;
    }
}
"""

LOOPS_FIXED = """\
public class Matrix {
    public int sum(int[][] grid) {
        int total = 0;
        for (int row = 0; row < grid.length; row++) {
            for (int col = 0; col < grid[row].length; col++) {
                total += grid[row][col];
            }
        }
        return total;
    }
}
"""

LOOPS = Sample(
    name="loops",
    buggy=LOOPS_BUGGY,
    fixed=LOOPS_FIXED,
    start=5,
    end=5,
    target="            for (int col = 0; col < grid[row].length; col++) {",
)


# --------------------------------------------------------------------------- #
# 5. region at the very beginning: the signature on line 1
# --------------------------------------------------------------------------- #
NEAR_START_BUGGY = """\
static int guard(int value) {
    if (value < 0) {
        return 0;
    }
    return value;
}
"""

NEAR_START_FIXED = """\
public static int guard(final int value) {
    if (value < 0) {
        return 0;
    }
    return value;
}
"""

NEAR_START = Sample(
    name="near_start",
    buggy=NEAR_START_BUGGY,
    fixed=NEAR_START_FIXED,
    start=1,
    end=1,
    target="public static int guard(final int value) {",
)


# --------------------------------------------------------------------------- #
# 6. region at the very end: lines 5-6, through the last line of the unit
# --------------------------------------------------------------------------- #
NEAR_END_BUGGY = """\
public int last(int[] xs) {
    if (xs.length == 0) {
        return -1;
    }
    return xs[xs.length];
}
"""

NEAR_END_FIXED = """\
public int last(int[] xs) {
    if (xs.length == 0) {
        return -1;
    }
    return xs[xs.length - 1];
}
"""

NEAR_END = Sample(
    name="near_end",
    buggy=NEAR_END_BUGGY,
    fixed=NEAR_END_FIXED,
    start=5,
    end=5,
    target="    return xs[xs.length - 1];",
)

# The same unit with the region running through the final line (6 of 6).
NEAR_END_THROUGH_LAST = Sample(
    name="near_end_through_last",
    buggy=NEAR_END_BUGGY,
    fixed=NEAR_END_FIXED,
    start=5,
    end=6,
    target="    return xs[xs.length - 1];\n}",
)


# --------------------------------------------------------------------------- #
# extra: the fix deletes the region outright (line 4)
# --------------------------------------------------------------------------- #
DELETION_BUGGY = """\
public class Counter {
    public int count(int[] xs) {
        int total = 0;
        total = 0;
        for (int x : xs) {
            total += x;
        }
        return total;
    }
}
"""

DELETION_FIXED = """\
public class Counter {
    public int count(int[] xs) {
        int total = 0;
        for (int x : xs) {
            total += x;
        }
        return total;
    }
}
"""

DELETION = Sample(
    name="deletion",
    buggy=DELETION_BUGGY,
    fixed=DELETION_FIXED,
    start=4,
    end=4,
    target="",
)


# --------------------------------------------------------------------------- #
# extra: tab-indented source, to prove indentation is preserved byte-for-byte
# --------------------------------------------------------------------------- #
TABS_BUGGY = "public class Tabs {\n\tpublic int id(int x) {\n\t\treturn -x;\n\t}\n}\n"
TABS_FIXED = "public class Tabs {\n\tpublic int id(int x) {\n\t\treturn x;\n\t}\n}\n"

TABS = Sample(
    name="tabs",
    buggy=TABS_BUGGY,
    fixed=TABS_FIXED,
    start=3,
    end=3,
    target="\t\treturn x;",
)


ALL_SAMPLES = [
    ONE_LINE,
    MULTI_LINE,
    NESTED_IF,
    LOOPS,
    NEAR_START,
    NEAR_END,
    NEAR_END_THROUGH_LAST,
    DELETION,
    TABS,
]


def long_source(total_lines: int = 60, bug_line: int = 30) -> str:
    """A long synthetic method, for exercising the context window."""
    lines = ["public class Long {", "    public int run() {", "        int total = 0;"]
    while len(lines) < total_lines - 3:
        index = len(lines) + 1
        marker = "BUG" if index == bug_line else "ok"
        lines.append(f"        total += {index}; // {marker}")
    lines += ["        return total;", "    }", "}"]
    return "\n".join(lines[:total_lines]) + "\n"
