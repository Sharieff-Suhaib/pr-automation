"""FastAPI surface over the workflow.

    GET  /         the single-page UI in static/index.html
    POST /solve    run the full pipeline on one issue; pauses for review before pushing
    POST /review   approve (commit + push) or deny a paused run
    GET  /issues   list a GitHub repository's open issues, for the UI's picker
    GET  /health   liveness plus whether Ollama is reachable
    GET  /sample   the bundled offline demo request, used by the UI

Run it:
    python -m uvicorn orchestrator.api:app --reload
UI:
    http://127.0.0.1:8000/
Swagger UI:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from orchestrator import adapters
from orchestrator.github_issues import DEFAULT_LIMIT, GitHubError, fetch_issues
from orchestrator.manager_agent import resume_review, run_status, solve_issue, start_run
from orchestrator.state import DEFAULT_MAX_ATTEMPTS
from orchestrator.run_demo import SAMPLE_ISSUE, SAMPLE_PATCH, SAMPLE_REPO, SAMPLE_REPRO

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"

app = FastAPI(
    title="Agent-SWE Orchestrator",
    version="1.0.0",
    description=(
        "Multi-agent GitHub issue repair. One LangGraph workflow chains the "
        "Repository Agent, the Recommendation Agent, the Code Generation Agent "
        "and the Testing Agent."
    ),
)

class SolveRequest(BaseModel):
    """One repair request: which repository, and what is wrong with it."""

    repo_url: str = Field(
        ...,
        description=(
            "Repository URL to clone, or a local directory path"
        ),
    )

    issue: str = Field(
        ...,
        description="The GitHub issue text",
    )

    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="How many code chunks to retrieve",
    )

    codegen_backend: str = Field(
        "ngrok",
        description=(
            "'ngrok' tries the remote repair model first and falls back "
            "to Ollama; 'stub' replays a fixture diff"
        ),
    )

    stub_patch_path: str = Field(
        "",
        description=(
            "Fixture diff, used only when codegen_backend='stub'"
        ),
    )

    reproduce: bool = Field(
        True,
        description=(
            "Write a test reproducing the issue before patching"
        ),
    )

    max_attempts: int = Field(
        DEFAULT_MAX_ATTEMPTS,
        ge=1,
        le=10,
        description=(
            "Patches to try before giving up"
        ),
    )

    isolated_env: bool = Field(
        True,
        description=(
            "Run tests in a cached venv with the repo's dependencies"
        ),
    )

    stub_repro_path: str = Field(
        "",
        description=(
            "Fixture reproduction test, used only when "
            "codegen_backend='stub'"
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "repo_url": (
                        "https://github.com/"
                        "Sharieff-Suhaib/dummy_repo.git"
                    ),
                    "issue": (
                        "get_user crashes with a KeyError "
                        "when the username is not registered"
                    ),
                    "top_k": 5,
                    "codegen_backend": "ngrok",
                }
            ]
        }
    }

class SolveResponse(BaseModel):
    """The report produced by one full pass through the graph."""

    status: str
    repo_url: str
    issue: str
    repo_path: str = ""
    language: str = ""
    relevant_files: list[str] = []
    relevant_code: list[dict[str, Any]] = []
    similar_bugs: list[dict[str, Any]] = []
    bug_type: str = ""
    strategy: str = ""
    tools: list[str] = []
    tests: list[str] = []
    test_env: dict[str, Any] = {}
    reproduction: dict[str, Any] = {}
    patch: str = ""
    patch_source: str = "none"
    patch_status: str = "SKIPPED"
    changed_files: list[str] = []
    working_repo: str = ""
    test_status: str = "SKIPPED"
    test_result: dict[str, Any] = {}
    baseline_result: dict[str, Any] = {}
    test_verdict: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    reflection: dict[str, Any] = {}
    reflections: list[dict[str, Any]] = []
    review: dict[str, Any] = {}  # {"pending": True, "thread_id"} until POST /review decides
    push: dict[str, Any] = {}
    errors: list[str] = []
    trace: list[dict[str, Any]] = []
    running: bool = False  # only GET /runs/{thread_id}: the run is still going
    current: list[str] = []  # ...and the agent it is in


def _validate(request: SolveRequest) -> None:
    if not request.issue.strip():
        raise HTTPException(status_code=422, detail="issue must not be empty")
    if not request.repo_url.strip():
        raise HTTPException(status_code=422, detail="repo_url must not be empty")


def _options(request: SolveRequest) -> dict[str, Any]:
    return request.model_dump(exclude={"repo_url", "issue"})


@app.post("/runs", summary="Start a repair run in the background")
def start(request: SolveRequest) -> dict[str, str]:
    """Return at once with the run's `thread_id`; poll GET /runs/{thread_id} for progress."""
    _validate(request)
    return {"thread_id": start_run(request.repo_url, request.issue, **_options(request))}


@app.get("/runs/{thread_id}", response_model=SolveResponse, summary="Progress or result of a run")
def status(thread_id: str) -> dict[str, Any]:
    try:
        return run_status(thread_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="No run with that thread_id") from error


@app.post("/solve", response_model=SolveResponse, summary="Repair one GitHub issue")
def solve(request: SolveRequest) -> dict[str, Any]:
    """Run Repository -> Recommendation -> Code Generation -> Testing on one issue.

    Blocks until the run ends; the UI uses POST /runs instead. Individual agents
    degrade rather than fail: anything that could not run is reported in
    `errors`, and the rest of the report is still filled in.
    """
    _validate(request)
    return solve_issue(
        repo_url=request.repo_url,
        issue=request.issue,
        top_k=request.top_k,
        codegen_backend=request.codegen_backend,
        stub_patch_path=request.stub_patch_path,
        reproduce=request.reproduce,
        stub_repro_path=request.stub_repro_path,
        max_attempts=request.max_attempts,
        isolated_env=request.isolated_env,
    )


class ReviewRequest(BaseModel):
    thread_id: str = Field(..., description="`review.thread_id` from the /solve response")
    approve: bool = Field(..., description="True commits and pushes the patch; False ends the run")


@app.post("/review", response_model=SolveResponse, summary="Approve or deny a patch awaiting review")
def review(request: ReviewRequest) -> dict[str, Any]:
    """Resume a run paused at the human review: approve pushes a new branch, deny ends it."""
    try:
        return resume_review(request.thread_id, request.approve)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="No run is awaiting review with that thread_id") from error


@app.get("/issues", summary="List a GitHub repository's issues")
def issues(
    repo_url: str = Query(..., description="GitHub repository URL, or an owner/repo shorthand"),
    state: str = Query("open", pattern="^(open|closed|all)$"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=100),
) -> dict[str, Any]:
    """Return the repository's issues so the UI can offer them in a picker.

    Pull requests are filtered out. The GITHUB_TOKEN stays on the server; only
    the issue summaries reach the browser.
    """
    try:
        return fetch_issues(repo_url, state=state, limit=limit)
    except GitHubError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(INDEX_HTML)


@app.get("/sample", summary="The bundled offline demo request")
def sample() -> dict[str, Any]:
    """Return a ready-made request against the bundled sample repository.

    The UI's offline checkbox calls this, so the demo needs neither a network
    nor a running model.
    """
    return {
        "repo_url": str(SAMPLE_REPO),
        "issue": SAMPLE_ISSUE,
        "codegen_backend": "stub",
        "stub_patch_path": str(SAMPLE_PATCH),
        "stub_repro_path": str(SAMPLE_REPRO),
    }

@app.get(
    "/health",
    summary="Service and model-backend status",
)
def health() -> dict[str, Any]:
    """Report API, ngrok, and Ollama availability."""

    return {
        "status": "ok",
        "repair_model": _repair_model_status(),
        "ollama": _ollama_status(),
    }


def _repair_model_status() -> dict[str, Any]:
    """Probe the remote ngrok repair server."""

    host = adapters.REPAIR_MODEL_URL

    if not host:
        return {
            "reachable": False,
            "host": "",
            "detail": (
                "REPAIR_MODEL_URL is not configured"
            ),
        }

    # Prefer a normal health endpoint when available.
    for endpoint in ("/health", "/"):
        try:
            with urllib.request.urlopen(
                f"{host}{endpoint}",
                timeout=3,
            ) as response:
                body = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

            return {
                "reachable": True,
                "host": host,
                "endpoint": endpoint,
                "response": body[:500],
            }

        except urllib.error.HTTPError as error:
            # Try the next endpoint for 404/405 responses.
            if error.code in (404, 405):
                continue

            return {
                "reachable": False,
                "host": host,
                "endpoint": endpoint,
                "detail": f"HTTP {error.code}",
            }

        except (
            urllib.error.URLError,
            OSError,
            ValueError,
        ) as error:
            return {
                "reachable": False,
                "host": host,
                "endpoint": endpoint,
                "detail": str(error),
            }

    return {
        "reachable": False,
        "host": host,
        "detail": (
            "The remote server does not expose /health or /"
        ),
    }


def _ollama_status() -> dict[str, Any]:
    """Probe Ollama, which is used only as a fallback."""

    host = adapters.ollama_host()

    try:
        with urllib.request.urlopen(
            f"{host}/api/tags",
            timeout=3,
        ) as response:
            models = json.loads(
                response.read().decode("utf-8")
            ).get("models", [])

        return {
            "reachable": True,
            "host": host,
            "fallback": True,
            "models": [
                model.get("name")
                for model in models
                if isinstance(model, dict)
            ],
        }

    except urllib.error.HTTPError as error:
        return {
            "reachable": False,
            "host": host,
            "fallback": True,
            "detail": f"HTTP {error.code}",
        }

    except (
        urllib.error.URLError,
        OSError,
        ValueError,
    ) as error:
        return {
            "reachable": False,
            "host": host,
            "fallback": True,
            "detail": str(error),
        }