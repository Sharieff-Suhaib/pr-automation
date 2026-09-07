"""FastAPI surface over the workflow.

    GET  /         the single-page UI in static/index.html
    POST /solve    run the full pipeline on one issue
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
from orchestrator.manager_agent import solve_issue
from orchestrator.run_demo import SAMPLE_ISSUE, SAMPLE_PATCH, SAMPLE_REPO

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

    repo_url: str = Field(..., description="Repository URL to clone, or a local directory path")
    issue: str = Field(..., description="The GitHub issue text")
    top_k: int = Field(5, ge=1, le=20, description="How many code chunks to retrieve")
    codegen_backend: str = Field(
        "ollama",
        description="'ollama' generates a patch with the model; 'stub' replays a fixture diff",
    )
    stub_patch_path: str = Field("", description="Fixture diff, used only when codegen_backend='stub'")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "repo_url": "https://github.com/Sharieff-Suhaib/dummy_repo.git",
                    "issue": "get_user crashes with a KeyError when the username is not registered",
                    "top_k": 5,
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
    patch: str = ""
    patch_source: str = "none"
    patch_status: str = "SKIPPED"
    changed_files: list[str] = []
    working_repo: str = ""
    test_status: str = "SKIPPED"
    test_result: dict[str, Any] = {}
    errors: list[str] = []
    trace: list[dict[str, Any]] = []


@app.post("/solve", response_model=SolveResponse, summary="Repair one GitHub issue")
def solve(request: SolveRequest) -> dict[str, Any]:
    """Run Repository -> Recommendation -> Code Generation -> Testing on one issue.

    Individual agents degrade rather than fail: anything that could not run is
    reported in `errors`, and the rest of the report is still filled in.
    """
    if not request.issue.strip():
        raise HTTPException(status_code=422, detail="issue must not be empty")
    if not request.repo_url.strip():
        raise HTTPException(status_code=422, detail="repo_url must not be empty")

    return solve_issue(
        repo_url=request.repo_url,
        issue=request.issue,
        top_k=request.top_k,
        codegen_backend=request.codegen_backend,
        stub_patch_path=request.stub_patch_path,
    )


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
    }


@app.get("/health", summary="Service and model-backend status")
def health() -> dict[str, Any]:
    """Report whether the API is up and whether the patch model is reachable."""
    return {"status": "ok", "ollama": _ollama_status()}


def _ollama_status() -> dict[str, Any]:
    """Probe the Ollama host so a failed /solve run is easy to explain."""
    host = adapters.ollama_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        return {"reachable": True, "host": host, "models": [model["name"] for model in models]}
    except (urllib.error.URLError, OSError, ValueError) as error:
        return {"reachable": False, "host": host, "detail": str(error)}
