"""Standalone test for fault localization and Ollama patch generation.

Requirements:

    ollama serve
    ollama pull codellama:7b

Run:

    python test_repair_pipeline.py

Optional environment variables:

    AGENT_SWE_CODEGEN_MODEL=codellama:7b
    OLLAMA_HOST=http://localhost:11434
    AGENT_SWE_CODEGEN_MAX_TOKENS=1024
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

DEFAULT_MODEL = os.environ.get(
    "AGENT_SWE_CODEGEN_MODEL",
    "codellama:7b",
)

REQUEST_TIMEOUT_SECONDS = 300

MAX_PATCH_TOKENS = int(
    os.environ.get(
        "AGENT_SWE_CODEGEN_MAX_TOKENS",
        "1024",
    )
)

SYSTEM_PROMPT = """You are a software repair agent.
Return only a standard unified diff patch.
Do not include explanations or Markdown.
"""

FENCE_RE = re.compile(
    r"```(?:diff|patch)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Exceptions and data structures
# ---------------------------------------------------------------------------

class PatchGenerationError(RuntimeError):
    """Raised when patch generation or validation fails."""


@dataclass
class SuspiciousRegion:
    """A source region considered suspicious by fault localization."""

    file: str
    start_line: int
    end_line: int
    score: float
    original_code: str
    ir4: str


# ---------------------------------------------------------------------------
# Fault localization
# ---------------------------------------------------------------------------

def run_fault_localization(
    repo_url: str,
    issue: str,
    top_k: int = 4,
) -> list[SuspiciousRegion]:
    """Perform a small keyword-based fault-localization pass.

    This is intentionally simple so this entire test can run from one file.
    It scans Python files and ranks lines containing issue-related terms.
    """

    repository = Path(repo_url)

    if not repository.exists():
        raise PatchGenerationError(
            f"Repository does not exist: {repository}"
        )

    issue_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", issue)
        if len(word) > 2
    }

    regions: list[SuspiciousRegion] = []

    for source_file in repository.rglob("*.py"):
        if any(
            ignored in source_file.parts
            for ignored in {".git", "__pycache__", ".venv", "venv"}
        ):
            continue

        try:
            source = source_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        lines = source.splitlines()

        for index, line in enumerate(lines, start=1):
            line_words = {
                word.lower()
                for word in re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]+",
                    line,
                )
            }

            matching_words = line_words & issue_words

            score = len(matching_words) / max(len(issue_words), 1)

            # Also prioritize suspicious implementation patterns.
            if "return" in line:
                score += 0.15

            if "lower" in line or "strip" in line:
                score += 0.10

            if score <= 0:
                continue

            relative_file = source_file.relative_to(repository)

            ir4 = build_simple_ir4(line)

            regions.append(
                SuspiciousRegion(
                    file=str(relative_file),
                    start_line=index,
                    end_line=index,
                    score=min(score, 1.0),
                    original_code=line,
                    ir4=ir4,
                )
            )

    regions.sort(
        key=lambda region: region.score,
        reverse=True,
    )

    return regions[:top_k]


def build_simple_ir4(line: str) -> str:
    """Create a small intermediate representation for the model prompt."""

    stripped = line.strip()

    if stripped.startswith("return "):
        return f"RETURN({stripped[7:]})"

    if stripped.startswith("if "):
        return f"IF({stripped[3:].rstrip(':')})"

    if "=" in stripped:
        left, right = stripped.split("=", 1)
        return f"ASSIGN({left.strip()}, {right.strip()})"

    return f"CODE({stripped})"


def format_fault_localization_results(
    regions: list[SuspiciousRegion],
) -> str:
    """Format localized source regions for the code-generation prompt."""

    sections: list[str] = []

    for index, region in enumerate(regions, start=1):
        sections.append(
            f"""REGION {index}
FILE: {region.file}
SUSPICIOUS LINES: {region.start_line}-{region.end_line}
SCORE: {region.score:.4f}

ORIGINAL SUSPICIOUS CODE:
{region.original_code}

IR4 REPRESENTATION:
{region.ir4}"""
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Prompt and Ollama communication
# ---------------------------------------------------------------------------

def format_value(value: Any) -> str:
    """Format strings, lists, and optional values for the prompt."""

    if value is None:
        return "None"

    if isinstance(value, (list, tuple)):
        return "\n".join(
            f"- {item}"
            for item in value
        ) or "None"

    return str(value)


def build_repair_prompt(
    issue: Any,
    relevant_code: Any,
    similar_bugs: Any,
    strategy: Any,
    tools: Any,
    tests: Any,
) -> str:
    """Create the repair prompt sent to Ollama."""

    return f"""You are a software repair agent.

ISSUE:
{format_value(issue)}

RELEVANT CODE:
{format_value(relevant_code)}

SIMILAR BUGS:
{format_value(similar_bugs)}

RECOMMENDED STRATEGY:
{format_value(strategy)}

RECOMMENDED TOOLS:
{format_value(tools)}

RECOMMENDED TESTS:
{format_value(tests)}

Generate a minimal code patch that fixes the issue.
Do not modify unrelated code.
Return only a standard unified diff beginning with --- and +++.
"""


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
) -> str:
    """Send a non-streaming generation request to Ollama."""

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": MAX_PATCH_TOKENS,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            body = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        raise PatchGenerationError(
            f"Ollama returned HTTP {exc.code}."
        ) from exc

    except urllib.error.URLError as exc:
        raise PatchGenerationError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            f"Start Ollama and pull {model}."
        ) from exc

    except TimeoutError as exc:
        raise PatchGenerationError(
            f"Ollama did not answer within "
            f"{REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc

    response_text = body.get("response", "")

    if not isinstance(response_text, str):
        raise PatchGenerationError(
            "Ollama returned a non-string response."
        )

    if not response_text.strip():
        raise PatchGenerationError(
            "The LLM returned an empty patch."
        )

    if body.get("done_reason") == "length":
        raise PatchGenerationError(
            f"The LLM response was cut off at "
            f"{MAX_PATCH_TOKENS} tokens."
        )

    return response_text


# ---------------------------------------------------------------------------
# Patch generation and validation
# ---------------------------------------------------------------------------

def clean_patch(response: str) -> str:
    """Remove Markdown formatting and validate unified-diff structure."""

    response = response.strip()

    match = FENCE_RE.search(response)

    if match:
        patch = match.group(1).strip()
    else:
        patch = response

    diff_start = patch.find("--- ")

    if diff_start > 0:
        patch = patch[diff_start:]

    if not patch.startswith("--- "):
        raise PatchGenerationError(
            "The LLM did not return a patch beginning with '--- '."
        )

    if "\n+++ " not in patch:
        raise PatchGenerationError(
            "The LLM patch does not contain a '+++ ' file header."
        )

    if "\n@@ " not in patch:
        raise PatchGenerationError(
            "The LLM patch does not contain a unified-diff hunk."
        )

    return patch.rstrip() + "\n"


def diff_retry_prompt(original_prompt: str) -> str:
    """Ask the model again if the first response is not a valid diff."""

    return f"""{original_prompt}

Your previous response format was invalid.
Reply with only a unified diff.

The response must look like this:

--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,2 +1,2 @@
 def example():
-    return broken_value
+    return fixed_value

Do not include Markdown, explanations, or a rewritten function.
"""


def generate_patch(
    issue: Any,
    relevant_code: Any,
    similar_bugs: Any,
    strategy: Any,
    tools: Any,
    tests: Any,
) -> str:
    """Generate and validate a repair patch with Ollama."""

    prompt = build_repair_prompt(
        issue=issue,
        relevant_code=relevant_code,
        similar_bugs=similar_bugs,
        strategy=strategy,
        tools=tools,
        tests=tests,
    )

    response = call_ollama(prompt)

    try:
        return clean_patch(response)

    except PatchGenerationError:
        retry_response = call_ollama(
            diff_retry_prompt(prompt)
        )
        return clean_patch(retry_response)


def generate_patch_from_repository(
    repo_url: str,
    issue: str,
    similar_bugs: Any = None,
    strategy: Any = None,
    tools: Any = None,
    tests: Any = None,
    top_k: int = 4,
) -> str:
    """Localize suspicious code and generate an Ollama repair patch."""

    if not repo_url.strip():
        raise PatchGenerationError(
            "A repository URL or path is required."
        )

    regions = run_fault_localization(
        repo_url=repo_url,
        issue=issue,
        top_k=top_k,
    )

    if not regions:
        raise PatchGenerationError(
            "Fault localization found no suspicious regions."
        )

    relevant_code = format_fault_localization_results(regions)

    print("\nFault-localization results:\n")
    print(relevant_code)

    return generate_patch(
        issue=issue,
        relevant_code=relevant_code,
        similar_bugs=similar_bugs,
        strategy=strategy,
        tools=tools,
        tests=tests,
    )


# ---------------------------------------------------------------------------
# Repository and patch helpers
# ---------------------------------------------------------------------------

def run_command(
    command: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and return its completed process."""

    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def initialize_git_repository(repository: Path) -> None:
    """Initialize a temporary Git repository."""

    run_command(["git", "init"], cwd=repository)

    run_command(
        ["git", "add", "."],
        cwd=repository,
    )

    run_command(
        [
            "git",
            "-c",
            "user.name=Repair Test",
            "-c",
            "user.email=repair@example.com",
            "commit",
            "-m",
            "Initial broken repository",
        ],
        cwd=repository,
    )


def verify_patch_can_apply(
    repository: Path,
    patch_text: str,
) -> Path:
    """Write the patch and verify it with git apply --check."""

    patch_file = repository / "generated-repair.patch"

    patch_file.write_text(
        patch_text,
        encoding="utf-8",
    )

    check = run_command(
        [
            "git",
            "apply",
            "--check",
            str(patch_file),
        ],
        cwd=repository,
        check=False,
    )

    if check.returncode != 0:
        raise PatchGenerationError(
            "The generated patch cannot be applied:\n"
            f"{check.stderr}\n\n"
            f"Patch:\n{patch_text}"
        )

    return patch_file


def apply_patch(
    repository: Path,
    patch_file: Path,
) -> None:
    """Apply a previously validated patch."""

    run_command(
        [
            "git",
            "apply",
            str(patch_file),
        ],
        cwd=repository,
    )


# ---------------------------------------------------------------------------
# Manual verification
# ---------------------------------------------------------------------------

def create_sample_repository(base_directory: Path) -> Path:
    """Create a small repository containing intentional bugs."""

    repository = base_directory / "sample_repository"
    repository.mkdir()

    users_file = repository / "users.py"

    users_file.write_text(
        '''"""Simple user directory with intentional bugs."""


class UserDirectory:
    def __init__(self, users=None):
        self.users = list(users or [])

    def add_user(self, user):
        self.users.append(user)

    def find_by_email(self, email):
        """Find a user by email, ignoring letter case."""
        email = email.strip().lower()

        for user in self.users:
            if user["email"] == email:
                return user

        return None

    def active_users(self):
        """Return only active users."""
        return [
            user
            for user in self.users
            if user["active"]
        ]

    def count_active_users(self):
        """Count active users."""
        return len(self.users)

    def search_by_name(self, query):
        """Find users by name without case sensitivity."""
        query = query.strip().lower()

        return [
            user
            for user in self.users
            if query in user["name"]
        ]

    def deactivate_user(self, email):
        user = self.find_by_email(email)

        if user is None:
            return False

        user["active"] = False
        return True

    def get_user_name(self, email):
        user = self.find_by_email(email)

        if user is None:
            return None

        return user["name"]

    def user_exists(self, email):
        return self.find_by_email(email) is not None
''',
        encoding="utf-8",
    )

    initialize_git_repository(repository)

    return repository


def verify_repaired_file(repository: Path) -> None:
    """Perform simple manual checks after applying the generated patch."""

    users_file = repository / "users.py"
    source = users_file.read_text(encoding="utf-8")

    expected_changes = [
        "if user['email'].lower() == email",
        "if user[\"email\"].lower() == email",
        "if user['active']",
        "if query in user['name'].lower()",
    ]

    found_change = any(
        expected in source
        for expected in expected_changes
    )

    if not found_change:
        print(
            "Warning: the patch applied, but the expected repair "
            "patterns were not detected."
        )

    print("\nPatched users.py:\n")
    print(source)

    print("Manual verification completed.")


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the complete fault-localization and patch-generation test."""

    print("Starting repair pipeline test...")
    print(f"Ollama host: {OLLAMA_HOST}")
    print(f"Ollama model: {DEFAULT_MODEL}")

    try:
        with tempfile.TemporaryDirectory() as directory:
            temporary_directory = Path(directory)

            repository = create_sample_repository(
                temporary_directory
            )

            issue = (
                "Fix case-insensitive email matching, count only active "
                "users, and make name search case-insensitive."
            )

            patch = generate_patch_from_repository(
                repo_url=str(repository),
                issue=issue,
                similar_bugs=[
                    "Stored email values are not normalized.",
                    "Counters include inactive records.",
                    "Search compares names with incorrect casing.",
                ],
                strategy=(
                    "Make the smallest possible changes to normalize "
                    "stored values during comparisons."
                ),
                tools=[
                    "git apply --check",
                    "git diff",
                ],
                tests=[
                    "Email lookup should ignore case.",
                    "Inactive users should not be counted.",
                    "Name search should ignore case.",
                ],
                top_k=4,
            )

            print("\nGenerated patch:\n")
            print(patch)

            patch_file = verify_patch_can_apply(
                repository=repository,
                patch_text=patch,
            )

            print("Patch passed git apply --check.")

            apply_patch(
                repository=repository,
                patch_file=patch_file,
            )

            print("Patch applied successfully.")

            verify_repaired_file(repository)

            print("\nFinal Git diff:\n")

            diff = run_command(
                ["git", "diff"],
                cwd=repository,
            )

            print(diff.stdout)

    except PatchGenerationError as exc:
        print(f"\nRepair pipeline failed:\n{exc}")
        return 1

    except FileNotFoundError as exc:
        print(
            "\nRequired command was not found:"
            f" {exc.filename}"
        )
        print("Make sure Git and Ollama are installed.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("\nCommand failed:")
        print(" ".join(exc.cmd))
        print(exc.stdout or "")
        print(exc.stderr or "")
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    print("\nRepair pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())