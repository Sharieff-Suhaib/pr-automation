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
  reproduction_agent              model writes a test for the issue -> kept only if it
    |                             fails on the unpatched code; that run is the baseline
    |                             writes: reproduction, baseline_result
    v
  coding_agent          Member 3  builds the repair brief -> unified diff patch
    |                             writes: patch, patch_source
    v
  testing_agent                   reuse (or run) the baseline -> git apply -> suite again -> compare
    |                             writes: working_repo, changed_files, patch_status,
    |                                     baseline_result, test_result, test_verdict
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
| `fixtures/` | The diff that repairs `sample_repo` and a test reproducing its bug, both replayed by the `stub` backend |

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
goes from 2 failing tests to 4 passing ones (2 fixed, 0 broken).

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
  "stub_patch_path": "",
  "reproduce": true,
  "stub_repro_path": "",
  "max_attempts": 3,
  "isolated_env": true
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
  "test_result": {"status": "PASS", "passed": 4, "failed": 0, "errors": [], "output": "...", "cases": {"test_users::test_known_user_is_returned": "passed"}},
  "baseline_result": {"status": "FAIL", "passed": 2, "failed": 2, "...": "..."},
  "test_verdict": {
    "status": "solved",
    "reason": "2 test(s) went from failing to passing and none of the passing tests broke.",
    "fail_to_pass": ["test_users::test_missing_user_returns_none", "test_users::test_login_rejects_unknown_user"],
    "pass_to_pass": ["..."], "pass_to_fail": [], "fail_to_fail": [], "added": {}, "per_test": true
  },
  "errors": [],
  "trace": [{"agent": "repository_agent", "summary": "5 relevant chunk(s) from 6 indexed"}]
}
```

`status` is one of:

| `status` | Meaning |
| --- | --- |
| `solved` | At least one test went from failing to passing, and no passing test broke |
| `unverified` | Nothing broke, but no test shows the fix (e.g. no test covers the bug) |
| `failed` | A test broke, the patch did not apply, or there was no patch |

It comes from `test_verdict`, which compares `baseline_result` (the suite on the
unpatched repository) with `test_result` (the suite on the patched copy) test by
test. Per-test results come from pytest's `--junitxml` report; other runners
fall back to comparing the overall PASS/FAIL of the two runs (`per_test: false`).
`test_status` is the plain `PASS` / `FAIL` / `SKIPPED` string of the patched
run. `trace` is the audit trail —
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
Agent still runs the suite on the unmodified repository, so `baseline_result`
carries real numbers. `GET /health` says whether Ollama is reachable.

## Test environment and cleanup

Before any test runs, `environment_agent` prepares the interpreter:

* A repository that declares dependencies (`requirements*.txt`, `pyproject.toml`
  `[project]` dependencies and test/dev extras, `setup.cfg`) gets a virtual
  environment with them plus pytest, cached in `.test_envs/` under a hash of
  those files: built once (seconds to minutes), then reused until they change.
* The repository itself is never installed (`-e .` lines are skipped), so tests
  import the patched working copy; it and its `src/` go on `PYTHONPATH`.
* No dependency files, `isolated_env: false` / `--no-isolated-env`, or a failed
  install (no network, bad pin) -> tests use this project's interpreter, as
  before, and `test_env` in the report says why.

Each run first deletes all but the newest 20 directories in `workspaces/` and
the newest 5 environments in `.test_envs/` (`AGENT_SWE_KEEP_WORKSPACES`,
`AGENT_SWE_KEEP_TEST_ENVS`). Stored test output is capped at its last 20,000
characters.

This is isolation of *dependencies*, not a security sandbox: the repository's
tests still run on this machine with your permissions.

## Test stages

A patched copy is tested cheapest stage first, and the run stops at the first
stage that rules the patch out, so a bad patch (and so a retry) fails in seconds:

| Stage | Runs | Stops when |
| --- | --- | --- |
| `syntax` | compiles the changed Python files | one does not compile |
| `reproduction` | only `test_issue_reproduction.py` | it still fails (cannot be `solved`) |
| `related` | test files named after, or importing, the changed files | a passing test breaks |
| `full` | the whole suite | -- the final verdict |

A partial stage is compared only with the baseline results of the files it ran,
so the tests it skipped never count as broken. `test_result.stages` and
`test_verdict.stage` record how far a patch got. Non-pytest projects go straight
from `syntax` to `full`.

## The Reflection Agent

`src/agents/reflection_agent/` runs after every failed attempt (never after a
solved one) and never edits code. It reads the issue, the faulty code, the
failed patch, the before/after test comparison with failure messages, the
strategy and earlier attempts, and returns:

```json
{"status": "retry", "failure_type": "runtime_error", "root_cause": "...",
 "patch_analysis": "...", "failed_tests": ["..."], "suggested_changes": ["..."],
 "repair_guidance": "...", "confidence": 0.8, "attempt": 1, "source": "llm"}
```

`failure_type` (syntax_error, compilation_error, test_failure, runtime_error,
logic_error, regression, patch_application_error, unknown) is classified from
the test results. The analysis comes from the coding agent's Ollama model; if
it is unreachable or returns unusable JSON, a rule-based analysis is used
(`source: "rules"`; always with the `stub` backend). Its feedback -- starting
"This is a repair retry. The previous patch failed validation..." -- leads the
coding agent's next prompt, followed by the testing agent's per-test feedback.
After the last attempt it still runs once with `status: max_retries`, so the
report explains the final failure, and then the run ends. Progress is logged as
`[ReflectionAgent] ...` lines.

## The retry loop

When an attempt is not `solved`, `testing_agent` routes to `reflection_agent`
and then back to `coding_agent` (up to `max_attempts`, default 3, or
`AGENT_SWE_MAX_REPAIR_ATTEMPTS`; the `stub` backend never retries since it
replays the same fixture). The coding agent gets `test_feedback`: the verdict,
the reproduction tests that still fail, the tests the patch broke, their
failure messages, and the rejected patch -- flagged when it repeats an earlier
one. Retries sample at a rising temperature (0, 0.4, 0.8) so they are not
forced to return the same answer. The baseline and reproduction test are
reused, so a retry costs one model call and one suite run.

The report's top-level fields show the solved attempt, or the best one when
none was solved (verdict first, then fewest broken tests, then most fixed).
`attempts` lists every attempt with its patch and outcome.

## The reproduction test

When a repository has no test that exercises the bug, comparing the suite before
and after a patch can only ever say `unverified`. So before the patch is
written, the Reproduction Agent asks the model for `test_issue_reproduction.py`
and checks it against the unpatched code:

1. It must be valid Python, contain `test_...` functions, and import the code
   under test rather than paste its own copy.
2. Run on a copy of the unpatched repository, at least one of its tests must
   fail for a real reason: an assertion or an exception from the code under
   test, not a `NameError` or `ImportError` in the test itself.
3. Tests that pass on the unpatched code are removed, because they reproduce
   nothing, and one may even assert the buggy behaviour, which would make a
   correct fix look like a regression.
4. A rejected file goes back to the model with the reason, once.

An accepted test is added to both copies (never the original repository), so it
appears in the before/after comparison, and `solved` additionally requires at
least one of its tests to pass after the patch. `reproduction.status` is
`accepted`, `rejected` or `skipped` (model unreachable, non-Python repository,
or `reproduce: false`). With `codegen_backend: "stub"`, the test at
`stub_repro_path` is replayed instead of calling the model.

## Notes

* Nodes never raise. A failing agent records its message in `errors` and
  returns empty results, so one unavailable dependency degrades a run instead
  of ending it.
* The cloned repository is never modified. The baseline runs in a temporary
  copy that is deleted afterwards; the patched copy is kept in
  `workspaces/<repo>-<timestamp>/` for inspection.
* The Testing Agent's own tests: `python -m pytest src/agents/testing_agent/tests`.
* Recommended test *names* are suggestions, not pytest node ids, so the Testing
  Agent runs the full suite rather than passing them to pytest.
* `outputs/` and `workspaces/` are gitignored; `outputs/latest.json` is always
  the most recent run.
