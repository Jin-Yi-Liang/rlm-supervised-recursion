"""Run a live two-group Planning-Only RLM smoke test against a Qwen-compatible endpoint."""

import argparse
from pathlib import Path

from rlm.supervised_rlm import QwenClient, RunConfig, SessionRuntime

QUESTION = """Return the total of groups A and B as an unformatted decimal. Register required
task a using store_input(context["A"]) and task b using store_input(context["B"]).
Each task must produce the integer sum of all assigned numbers. After committing the plan, dispatch
with rlm_query_batched(["a", "b"]) and print the outcomes. After dispatch finishes, use
list_accepted_results() and read_accepted_result(receipt.receipt_id).value to sum the two
accepted values. Submit their total as final_text=str(total). Use the provided data."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = RunConfig(max_root_steps=12, max_child_steps=10)
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
