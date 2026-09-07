# Orchestrator (Member 4)

One LangGraph workflow that chains the three agents built by Members 1-3, plus
the FastAPI endpoint that drives it.

```
POST /solve
    |
    v
  START
    |
    v
  repository_agent      Member 1  clone/locate repo -> tree-sitter parse -> FAISS retrieval
    |                             writes: repo_path, relevant_files, relevant_code, language
    v
  recommendation_agent  Member 2  similar bugs -> repair strategy -> tools -> tests
    |                             writes: similar_bugs, bug_type, strategy, tools, tests
    v
  coding_agent          Member 3  builds the repair brief -> unified diff patch
    |                             writes: patch, patch_source
    v
  testing_agent         Member 3  copy repo -> git apply -> run the suite
    |                             writes: working_repo, changed_files, patch_status, test_result
    v
   END
```

## Files

| File | Role |
| --- | --- |
| `state.py` | `AgentState` — the shared blackboard every node reads and writes |
| `graph.py` | The four nodes and the `StateGraph` wiring them START -> END |
| `adapters.py` | All integration with Members 1-3 (import plumbing, dataclass -> dict) |
| `manager_agent.py` | Entry point: takes issue + repo URL, runs the graph, shapes the report |
| `api.py` | FastAPI app — serves the UI plus `POST /solve`, `GET /issues`, `GET /sample`, `GET /health` |
| `github_issues.py` | Lists a repository's issues from the GitHub API (port of `scripts/issue.js`) |
| `run_demo.py` | CLI runner; prints a summary and saves JSON to `outputs/` |
| `static/index.html` | The web UI — one page, plain HTML/CSS/JS, no build step |
| `sample_repo/` | A tiny buggy repository (2 of its 4 tests fail) for offline demos |
| `fixtures/` | The diff that repairs `sample_repo`, replayed by the `stub` backend |

## Running it

Every command below is run from the `pr-automation/` directory, with Python
3.10-3.12. Install the dependencies once:

```bash
cd pr-automation
pip install -r requirements.txt
```

### Start the website

```bash
python -m uvicorn orchestrator.api:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/> in a browser. Stop the server with `Ctrl+C`.

| URL | What it is |
| --- | --- |
| <http://127.0.0.1:8000/> | The web UI |
| <http://127.0.0.1:8000/docs> | Swagger / OpenAPI docs |
| <http://127.0.0.1:8000/health> | Whether Ollama is reachable and which models are pulled |
| <http://127.0.0.1:8000/issues?repo_url=...> | The repository's open issues, as JSON |

During development, add `--reload` to restart the server whenever a Python file
changes. `static/index.html` is read from disk on every request, so UI edits
only need a browser refresh either way:

```bash
python -m uvicorn orchestrator.api:app --reload
```

If port 8000 is already in use, pick another one with `--port 8010`. On Windows,
if `python` is not on your PATH, use `py -3` in place of `python`.

### Use the website

**Against a real GitHub repository:**

1. Paste the repository URL and click **Load issues**. The server calls the
   GitHub API and fills a dropdown with the repository's open issues; pull
   requests are filtered out.
2. Pick an issue. Its title and body are copied into the issue box, which stays
   editable so the text can be trimmed before the run.
3. Click **Run workflow**.

This path needs Ollama running with a model pulled (see below), otherwise the
run completes without a patch.

**Offline, with no network and no model:**

1. Tick **Offline demo** — it fills the form from `/sample` with the bundled
   sample repository and the fixture patch.
2. Click **Run workflow**.

Either way the report renders below: relevant files, similar bugs, strategy,
tools, tests, the colourised diff, the test run, and the per-agent trace.

The first run takes around 35 seconds while the embedding model loads; later
runs reuse it and are quicker. The page shows *Running...* until the whole
workflow finishes, because `/solve` returns only once the graph is done.

### Command line

**Offline demo** — no network, no model, proves the whole graph works:

```bash
python -m orchestrator.run_demo --offline
```

It ends with `WORKFLOW STATUS: SOLVED`: the patch applies and `sample_repo`
goes from 2 failing tests to 4 passing ones.

**Real run** — clones the repository and generates the patch with the model:

```bash
python -m orchestrator.run_demo --repo-url https://github.com/Sharieff-Suhaib/dummy_repo.git --issue "Login fails when the username is empty"
```

**Call the API directly** — with the server running:

```bash
curl -X POST http://127.0.0.1:8000/solve -H "Content-Type: application/json" -d "{\"repo_url\": \"https://github.com/Sharieff-Suhaib/dummy_repo.git\", \"issue\": \"Login fails when the username is empty\"}"
```

In PowerShell, `curl` is an alias for `Invoke-WebRequest` and takes different
arguments, so call `curl.exe` explicitly or use:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/solve -Method Post -ContentType "application/json" -Body '{"repo_url": "orchestrator/sample_repo", "issue": "get_user crashes on a missing username"}'
```

## The `/solve` contract

Request:

```json
{
  "repo_url": "https://github.com/owner/repo.git",
  "issue": "get_user crashes with a KeyError when the username is not registered",
  "top_k": 5,
  "codegen_backend": "ollama",
  "stub_patch_path": ""
}
```

`repo_url` also accepts a local directory path, which is how the offline demo
runs without a network. `codegen_backend` defaults to `"ollama"`; set it to
`"stub"` with a `stub_patch_path` to replay a fixture diff instead of calling
the model.

Response (abbreviated):

```json
{
  "status": "solved",
  "relevant_files": ["users.py"],
  "similar_bugs": [{"issue": "...", "fix": "...", "similarity": 0.62}],
  "bug_type": "null_reference",
  "strategy": "Add null/None validation before dereferencing.",
  "tools": ["Pytest", "Ruff", "Git"],
  "tests": ["test_missing_user_returns_none()"],
  "patch": "--- a/users.py\n+++ b/users.py\n@@ ...",
  "patch_status": "APPLIED",
  "changed_files": ["users.py"],
  "test_status": "PASS",
  "test_result": {"status": "PASS", "passed": 4, "failed": 0, "errors": [], "output": "..."},
  "errors": [],
  "trace": [{"agent": "repository_agent", "summary": "5 relevant chunk(s) from 6 indexed"}]
}
```

`test_status` is the plain `PASS` / `FAIL` / `SKIPPED` string; `test_result`
keeps the counts and the runner output behind it. `trace` is the audit trail —
one entry per node, in execution order.

## Listing GitHub issues

`GET /issues?repo_url=<url>&state=open&limit=30` returns the repository's
issues, and the **Load issues** button in the UI calls it. `repo_url` accepts a
plain repository URL, a `.git` clone URL, a deep link such as `.../issues`, or
an `owner/repo` shorthand.

Reads are unauthenticated by default, which GitHub rate-limits to 60 requests
an hour and which cannot see private repositories. Put a token in
`pr-automation/.env` to lift both limits:

```
GITHUB_TOKEN=ghp_your_token_here
```

The token is read on the server by `github_issues.load_token()` and is never
sent to the browser — only the issue summaries are. `GITHUB_TOKEN` in the
environment takes precedence over the `.env` file.

## Requires Ollama for real patches

The Code Generation Agent and parts of the Recommendation Agent call a local
Ollama server:

```bash
ollama serve
ollama pull codellama:7b
```

Without it the workflow still completes: the Recommendation Agent falls back to
its keyword classifier and test heuristic, the Code Generation Agent reports
`Cannot reach Ollama ...` in `errors` and returns no patch, and the Testing
Agent runs the suite on the unmodified repository so the report still carries a
real baseline. `GET /health` says whether Ollama is reachable.

## Notes

* Nodes never raise. A failing agent records its message in `errors` and
  returns empty results, so one unavailable dependency degrades a run instead
  of ending it.
* The cloned repository is never modified. Each run copies it into
  `workspaces/<repo>-<timestamp>/` and patches the copy.
* Recommended test *names* are suggestions, not pytest node ids, so the Testing
  Agent runs the full suite rather than passing them to pytest.
* `outputs/` and `workspaces/` are gitignored; `outputs/latest.json` is always
  the most recent run.
