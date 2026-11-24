# Supervised RLM v1

`rlm.supervised_rlm.SessionRuntime` runs the design with Root, Child, and Supervisor sessions.
Root commits a versioned plan and dispatches task IDs. Each Child submits REPL variable names.
Supervisor uses an independent REPL to read a scoped, frozen candidate, its variables and
trace, and the original task input. Accepted results and ledger updates share a commit record.
REVISE resumes the existing Child session. PLAN_ISSUE returns to Root for an explicit plan
revision, and REVIEW_INCOMPLETE leaves the task unaccepted.

## Local verification

Use a Linux host or Pod with `libseccomp.so.2` and Python 3.11+. The opt-in runtime gives
each role a persistent Python subprocess with an allowlisted syscall filter, separate input
and namespace, bounded execution time, a per-cell tool call limit, and role-specific JSON tools. Supported artifact
values are JSON-compatible. Worker modules are preloaded from the Python standard library.

```bash
uv sync --group dev --group test
uv run pytest -q tests/test_supervised_v1.py tests/test_supervised.py tests/test_rlm_query.py
uv run python -m examples.supervised_v1_smoke \
  --base-url http://10.100.1.114:8001/v1 --model Qwen/Qwen3-32B \
  --out /tmp/supervised-rlm-v1-smoke
```

The live smoke test uses the full-list sums of `[4, 7]` and `[8, 9]`. It expects two
accepted receipts and the final answer `28`. The scripted end-to-end test separately
asserts rejection followed by repair in the same Child session and acceptance of v2.

## Runtime API

```python
from rlm.supervised_rlm import QwenClient, RunConfig, SessionRuntime

config = RunConfig()
with QwenClient("http://10.100.1.114:8001/v1", "Qwen/Qwen3-32B", config) as client:
    runtime = SessionRuntime(client, "/tmp/my-supervised-run")
    outcome = runtime.solve("Analyze both groups and compare their totals", {"A": [4, 7], "B": [8, 9]}, config)
    print(outcome.status, outcome.final_text)
```

Use an empty output directory for each run. For separate initialization, call
`start_run(question, context, config)` and then `run(handle.run_id)`. The outcome records the
status and reason when a run ends without a final answer. Root and Supervisor use Qwen
thinking, and Child uses non-thinking mode. The Qwen adapter uses the service's
`/tokenize` endpoint to check the configured context window before model calls.

## Experiment records

- `run.json`, `plans/`, `task-ledger.json`, and `accepted-receipts.json` capture the run
  contract, plan versions, task lifecycle, and valid results.
- `artifacts/`, `submissions/`, `reviews/`, and `commits/` capture stored inputs,
  frozen candidate references, verdicts, and published acceptance records.
- `wire/` preserves each request and response. `agents/` preserves each role's messages
  and Python cells. `events.jsonl` and `metrics.json` record execution and aggregate usage.
  Missing provider usage and unevaluated tool semantics are represented explicitly.

Task IDs serve as stable tracking keys without imposing dependency edges. Changing a
previously accepted task creates a new task version and explicitly supersedes its earlier
receipt. Natural-language acceptance criteria are evaluated through Supervisor review.
The runtime checks available output schemas, task coverage, and receipt consistency
deterministically. Use benchmark ground truth for independent end-answer scoring.

The earlier [supervised OOLONG pilot](supervised-oolong.md) remains available for reproducing
its two historical modes.
