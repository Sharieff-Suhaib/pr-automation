"""Fetch a repository's issues from the GitHub REST API.

A Python port of `scripts/issue.js`, so the web UI can offer the repository's
real issues instead of asking the user to paste one. Same endpoint, same
`Authorization: Bearer <GITHUB_TOKEN>` header; `urllib` is used rather than
`requests` to keep the dependency list unchanged.

The token is read on the server and never sent to the browser.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

GITHUB_API = "https://api.github.com"
API_VERSION = "2026-03-10"
REQUEST_TIMEOUT = 20
DEFAULT_LIMIT = 30

# GitHub owner and repository names: letters, digits, dot, dash, underscore.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


class GitHubError(RuntimeError):
    """The issues could not be fetched, with a message worth showing the user."""


def load_token() -> str:
    """Read GITHUB_TOKEN from the environment, falling back to the .env file.

    Unauthenticated requests still work for public repositories, just at a much
    lower rate limit, so an empty token is not an error here.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token or not ENV_FILE.is_file():
        return token

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "GITHUB_TOKEN":
            return value.strip().strip('"').strip("'")
    return ""


def parse_repo(repo_url: str) -> tuple[str, str]:
    """Pull `(owner, repo)` out of a GitHub URL or an `owner/repo` shorthand.

    Accepts the forms people actually paste: the plain repository URL, the
    `.git` clone URL, and a deep link such as `.../issues` or `.../pull/3`.
    """
    text = (repo_url or "").strip()
    if not text:
        raise GitHubError("Enter a GitHub repository URL first.")

    # The same field accepts a local directory for offline runs; saying so beats
    # letting "C:/code/repo" through and returning a puzzling 404.
    if Path(text).is_dir():
        raise GitHubError("That is a local path. Issues can only be listed for a GitHub repository.")

    if "github.com" in text:
        path = urllib.parse.urlparse(text if "//" in text else f"https://{text}").path
    else:
        path = text  # already an owner/repo shorthand

    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        raise GitHubError(f"Not a GitHub repository URL: {repo_url}")

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not _NAME_RE.match(owner) or not _NAME_RE.match(repo):
        raise GitHubError(f"Not a GitHub repository URL: {repo_url}")
    return owner, repo


def fetch_issues(repo_url: str, state: str = "open", limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """List a repository's issues, newest first.

    Returns `{"owner", "repo", "issues"}`. Pull requests are filtered out: the
    issues endpoint returns them too, marked by a `pull_request` key, and they
    are not repair targets.
    """
    owner, repo = parse_repo(repo_url)

    query = urllib.parse.urlencode({"state": state, "per_page": max(1, min(limit, 100))})
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "agent-swe-orchestrator",
    }
    token = load_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        f"{GITHUB_API}/repos/{owner}/{repo}/issues?{query}",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise GitHubError(_http_message(error, owner, repo, token)) from error
    except (urllib.error.URLError, OSError) as error:
        raise GitHubError(f"Could not reach the GitHub API: {error}") from error
    except ValueError as error:
        raise GitHubError(f"GitHub returned a response that is not JSON: {error}") from error

    issues = [_summarise(item) for item in payload if "pull_request" not in item]
    return {"owner": owner, "repo": repo, "issues": issues}


def _summarise(item: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields the UI shows and the workflow consumes."""
    return {
        "number": item.get("number"),
        "title": item.get("title") or "",
        "body": item.get("body") or "",
        "state": item.get("state") or "",
        "url": item.get("html_url") or "",
        "labels": [label.get("name", "") for label in item.get("labels") or []],
        "comments": item.get("comments", 0),
    }


def _http_message(error: urllib.error.HTTPError, owner: str, repo: str, token: str) -> str:
    """Turn a GitHub error status into something a user can act on."""
    try:
        detail = json.loads(error.read().decode("utf-8")).get("message", "")
    except (ValueError, OSError):
        detail = ""

    if error.code == 401:
        return "GitHub rejected the token in .env (401). Check GITHUB_TOKEN."
    if error.code == 404:
        missing = f"{owner}/{repo} was not found (404)."
        return missing if token else f"{missing} Private repositories need a GITHUB_TOKEN in .env."
    if error.code == 403 and "rate limit" in detail.lower():
        return "GitHub rate limit reached. Add a GITHUB_TOKEN to .env to raise it."
    return f"GitHub returned HTTP {error.code}{f': {detail}' if detail else '.'}"


def issue_text(issue: dict[str, Any]) -> str:
    """Flatten an issue into the single string the workflow takes as input."""
    return f"{issue.get('title', '')}\n\n{issue.get('body', '')}".strip()
