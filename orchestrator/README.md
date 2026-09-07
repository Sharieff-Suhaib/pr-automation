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
| `api.py` | FastAPI app — serves the UI plus `POST /solve`, `GET /sample`, `GET /health` |
| `run_demo.py` | CLI runner; prints a summary and saves JSON to `outputs/` |
| `static/index.html` | The web UI — one page, plain HTML/CSS/JS, no build step |
| `sample_repo/` | A tiny buggy repository (2 of its 4 tests fail) for offline demos |
| `fixtures/` | The diff that repairs `sample_repo`, replayed by the `stub` backend |

## Running it

Install once (Python 3.10-3.12):

```bash
pip install -r requirements.txt
```

**Offline demo** — no network, no model, proves the whole graph works:

```bash
python -m orchestrator.run_demo --offline
```

It ends with `WORKFLOW STATUS: SOLVED`: the patch applies and `sample_repo`
goes from 2 failing tests to 4 passing ones.

**Real run** — clones the repository and generates the patch with the model:

```bash
python -m orchestrator.run_demo \
    --repo-url https://github.com/Sharieff-Suhaib/dummy_repo.git \
    --issue "Login fails when the username is empty"
```

**Web UI / API** — start the server once and both are available:

```bash
python -m uvicorn orchestrator.api:app --reload
```

* UI: <http://127.0.0.1:8000/>
* Swagger: <http://127.0.0.1:8000/docs>

The UI is one static page that posts to `/solve` and renders the report:
relevant files, similar bugs, strategy, tools, tests, the colourised diff, the
test run and the trace. Tick **Offline demo** to fill the form from `/sample`
and run the bundled repository with the fixture patch — useful for a live
walkthrough when Ollama is not running.

```bash
curl -X POST http://127.0.0.1:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/Sharieff-Suhaib/dummy_repo.git",
       "issue": "Login fails when the username is empty"}'
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
