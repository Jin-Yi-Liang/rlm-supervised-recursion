"""Run live two-group acceptance and in-place repair against a Qwen-compatible endpoint."""

import argparse
from pathlib import Path

from rlm.supervised_rlm import QwenClient, RunConfig, SessionRuntime

QUESTION = """Return the total of groups A and B as an unformatted decimal. Register required
task a using store_input(context["A"]) and task b using store_input(context["B"]).
Each task must produce an integer and be independently checked against all assigned numbers.
For task a only, tell the Child to submit its first element's value using a variable named
partial_sum initially, then send the full sum after feedback. For task b, ask for the full sum.
The dispatcher automatically runs reviews and repairs. After committing the plan, dispatch
with rlm_query_batched(["a", "b"]) and print the outcomes. After dispatch finishes, use
list_accepted_results() and read_accepted_result(receipt.receipt_id).value to sum the two
accepted values. Submit their total as final_text=str(total). Use the provided data."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = RunConfig(max_root_steps=12, max_child_steps=10, max_review_steps=8)
    with QwenClient(args.base_url, args.model, config) as client:
        runtime = SessionRuntime(client, args.out)
        result = runtime.solve(QUESTION, {"A": [4, 7], "B": [8, 9]}, config)
    metrics = runtime.get_run_state(result.run_id)["metrics"]
    print(result)
    print(metrics)
    if (
        result.status != "COMPLETED"
        or result.final_text.strip() != "28"
        or metrics.completed_tasks != 2
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
