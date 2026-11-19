# Supervised OOLONG-Pairs pilot

This opt-in experiment reuses the native RLM completion turn and LocalREPL. In both arms, the
root and independent supervisor use Qwen thinking, while every worker, including grandchildren
and repairs, is non-thinking. The sequential dispatcher returns references only. A reviewer
acceptance authorizes the harness to store exactly that candidate in the case-local root REPL
mapping `accepted_results`. A direct parent can read a child's result by reference. Rejection
retains the worker's REPL and dialogue, with at most two repairs.

The first arm has **no plan table** or registration tools. Use
`rlm_query(instruction, input=chunk)` or a list of instruction/input objects. The second arm
adds registration, scoped dispatch by ID, plan-aware review and a required-task completion gate.
Depth is root → worker → leaf. Model-visible roles, full wire exchanges, execution events, receipts,
input snapshots, scores and token usage are saved to an isolated output directory per attempt.
The official local Python environment is not a security sandbox: run trusted benchmark inputs in a
dedicated harness, not untrusted third-party code. Do not publish run artifacts or credentials.

Run from the repository root with the historical benchmark Python. First run
`python -m pytest -q tests/test_supervised.py tests/test_rlm_query.py`, then
`python -m examples.supervised_oolong32k --smoke --base-url URL --out NEW_DIR --deadline UNIX_SECONDS`.
For formal execution replace `--smoke` with `--legacy /path/to/test_oolong_pair_32768.py`
and `--revision COMMIT`, using a separate output directory. The original runtime snapshot and
its sibling benchmarks package must be available. Its selector, context hash checks and pair
scorer are reused directly; only question and context enter the agent. The second arm starts
only after 20 scored completions in the first arm. A service deadline or failed verification
leaves attempts on disk. Resume against the same manifest and revision, never overwrite attempts.
