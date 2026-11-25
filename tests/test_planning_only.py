import json
from collections import defaultdict
from threading import Lock

import httpx
import pytest
from openai import OpenAI

from examples.planning_only_case12 import DeepSeekFlashClient
from rlm.supervised_rlm import QwenClient, RunConfig, SessionRuntime
from rlm.supervised_rlm.client import ModelReply
from rlm.supervised_rlm.metrics import (
    LLMCallRecord,
    MetricsCollector,
    TokenUsage,
    ToolCallRecord,
)
from rlm.supervised_rlm.repl import REPLRuntime
from rlm.supervised_rlm.schemas import AgentRole, ContractError, TaskStatus


def block(code):
    return f"```repl\n{code.strip()}\n```"


class ScriptedModel:
    model = "scripted"

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.cursors = defaultdict(int)
        self.calls = []
        self.lock = Lock()

    def count_tokens(self, messages, thinking):
        return 250

    def complete(self, messages, thinking, timeout, request_dir):
        meta = json.loads((request_dir / "meta.json").read_text())
        key = (meta["role"], meta["task_id"])
        assert thinking and meta["thinking"]
        with self.lock:
            index = self.cursors[key]
            self.cursors[key] += 1
            self.calls.append((meta, list(messages)))
        answer = self.responses[key][index]
        return ModelReply(block(answer), TokenUsage(250, 50, 300, "PROVIDER"), "stop")


def register(run):
    draft = run.plan.create_plan("Compute sums")
    for name, values in {"a": [1, 2], "b": [4, 8]}.items():
        run.plan.add_task(
            draft, name, f"Sum {name}", run.root.store_input(values), f"All of {name}"
        )
        run.plan.set_task_acceptance(draft, name, ["Return the full sum"], "integer")
    return draft


def test_plan_drafts_are_independent_and_versioned(tmp_path):
    run = SessionRuntime(ScriptedModel(), tmp_path)
    run.start_run("Sum two lists", [])
    draft = register(run)
    copied = run.plan.copy_plan(draft, "Reworded")
    with pytest.raises(ContractError, match="DUPLICATE_TASK"):
        run.plan.add_task(draft, "a", "Duplicate", run.root.store_input([1]), "rows")
    run.plan.update_task(copied, "a", instruction="Changed")
    assert run.plan.commit_plan(draft).plan_version == 1
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        run.plan.commit_plan(copied)
    assert run.plan.spec("a").instruction == "Sum a"
    revision = run.plan.begin_plan_revision(1, "Correct scope")
    run.plan.update_task(revision, "a", scope="All of first shard")
    assert run.plan.commit_plan(revision).plan_version == 2
    assert run.plan.spec("a").task_version == 2
    assert run.plan.spec("b").task_version == 1
    assert run.plan.state().pending_task_ids == ("a", "b")
    assert not run.root.validate_plan_completion().valid


def test_metrics_missing_usage_and_tool_protocol_are_distinct(tmp_path):
    metrics = MetricsCollector(tmp_path, "run")
    metrics.record_llm_call(
        LLMCallRecord(
            "request", "root", AgentRole.ROOT, "fake", 20, False, None, TokenUsage(), None
        )
    )
    flags = [
        (False, False, False, False, None),
        (True, True, False, False, None),
        (True, True, True, True, True),
        (True, True, True, True, False),
    ]
    for index, args in enumerate(flags):
        metrics.record_tool_call(
            ToolCallRecord(str(index), "root", AgentRole.ROOT, "test", *args, 1, None)
        )
    result = metrics.finish_run("FAILED", [])
    assert (result.input_tokens, result.output_tokens, result.token_usage_coverage) == (
        None,
        None,
        0,
    )
    assert result.llm_calls == result.llm_failures == 1
    assert (
        result.tool_calls,
        result.tool_protocol_valid,
        result.tool_arguments_valid,
        result.tool_execution_success,
    ) == (4, 3, 2, 1)
    assert result.tool_semantic_evaluated == 0


def test_worker_disallows_file_access_and_keeps_namespace(tmp_path):
    metrics = MetricsCollector(tmp_path, "run")
    with REPLRuntime(AgentRole.CHILD, "test", [1, 2], {}, metrics, 5) as repl:
        result, _ = repl.execute("x = sum(context)\nprint(x)", "first", lambda: False)
        assert result.success and result.stdout.strip() == "3"
        result, _ = repl.execute('open("/etc/hostname").read()', "denied", lambda: False)
        assert result.error_type == "NameError"
        result, _ = repl.execute("assert x == 3", "last", lambda: False)
        assert result.success


def test_global_budget_produces_incomplete_metrics(tmp_path):
    model = ScriptedModel({("ROOT", "root"): ["print(42)"]})
    runtime = SessionRuntime(model, tmp_path)
    outcome = runtime.solve("Add 4 and 2", [4, 2], RunConfig(max_model_calls=1))
    assert outcome.status == "INCOMPLETE"
    assert runtime.get_run_state(outcome.run_id)["metrics"].llm_calls == 1


def test_repl_stops_a_cell_that_repeatedly_calls_host_tools(tmp_path):
    from rlm.supervised_rlm.repl import REPLError

    metrics = MetricsCollector(tmp_path, "run")
    with (
        REPLRuntime(
            AgentRole.ROOT,
            "root",
            [],
            {"pending": lambda: [1]},
            metrics,
            5,
            max_tool_calls=4,
        ) as repl,
        pytest.raises(REPLError, match="exceeded its 4 tool calls"),
    ):
        repl.execute("while pending():\n    pass", "loop", lambda: False)
    assert sum(record.tool_name == "pending" for record in metrics.tools) == 4


def test_result_receipt_is_immutable_idempotent_and_superseded_by_revision(tmp_path):
    run = SessionRuntime(ScriptedModel(), tmp_path)
    run.start_run("Collect group a", [])
    draft = register(run)
    run.plan.set_task_acceptance(draft, "a", ["Collect all values"], "array", expected_count=4)
    run.plan.commit_plan(draft)
    run.plan.transition("a", TaskStatus.RUNNING)
    intent = run.submissions.prepare_submission(1, "result")
    values = [1, 2, 3, 4]
    namespace = [{"name": "result", "available": True, "value": values}]
    task = run.plan.spec("a")
    receipt = run.submissions.finalize_submission(task, intent, namespace)
    assert run.submissions.finalize_submission(task, intent, namespace) == receipt
    assert run.dispatcher.rlm_query("a") == receipt
    commit = json.loads((tmp_path / "commits" / f"{receipt.receipt_id}.json").read_text())
    assert commit["task"]["status"] == "ACCEPTED"
    assert commit["receipt"]["submission_id"] == intent.ack.submission_id
    assert len(list((tmp_path / "commits").glob("*.json"))) == 1

    values[0] = 999
    read = run.submissions.read_accepted_result
    page = read(receipt.receipt_id, limit=2)
    assert page.value == [1, 2] and page.truncated and page.next_offset == 2
    assert read(receipt.receipt_id, offset=2, limit=2).value == [3, 4]
    assert read(receipt.receipt_id, path="$[0]").value == 1

    revision = run.plan.begin_plan_revision(1, "Use the original context")
    run.plan.update_task(revision, "a", input_ref=run.global_task.context_refs[0])
    run.plan.commit_plan(revision)
    assert run.submissions.list_accepted_results() == []
    with pytest.raises(ContractError, match="PERMISSION_DENIED"):
        read(receipt.receipt_id)
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        run.submissions.finalize_submission(task, intent, namespace)
    assert not run.root.validate_plan_completion().valid


def test_failed_commit_never_publishes_a_receipt(tmp_path, monkeypatch):
    from rlm.supervised_rlm import submission

    model = ScriptedModel({("CHILD", "a"): ['result = sum(context)\nsubmit_candidate("result")']})
    run = SessionRuntime(model, tmp_path)
    run.start_run("Sum groups", [])
    run.plan.commit_plan(register(run))
    write_json = submission.write_json

    def fail_commit(path, value):
        if path.parent.name == "commits":
            raise OSError("Simulated storage failure")
        write_json(path, value)

    monkeypatch.setattr(submission, "write_json", fail_commit)
    outcome = run.dispatcher.dispatch_registered("a")
    assert outcome.status == TaskStatus.FAILED and outcome.receipt is None
    assert "Simulated storage failure" in outcome.reason
    assert run.submissions.list_accepted_results() == []
    assert not run.root.validate_plan_completion().valid


def test_plan_batch_validation_and_root_sequenced_dependency(tmp_path):
    model = ScriptedModel(
        {
            ("ROOT", "root"): [
                """
d = create_plan("Sum both groups")
for key in ["a", "b"]:
    add_task(d, key, "Sum the assigned numbers", store_input(context[key]), key)
    set_task_acceptance(d, key, ["Include each number once"], "object", ["sum"], 1)
commit_plan(d)
""",
                'submit_root_answer(final_text="premature")',
                """
outcomes = rlm_query_batched(["a", "b"])
assert [item.task_id for item in outcomes] == ["a", "b"]
raise ValueError("The accepted Child receipts survive this Root cell error")
""",
                """
state = get_plan_state()
assert state.completed_task_ids == ["a", "b"]
values = [read_accepted_result(item.receipt_id).value["sum"] for item in list_accepted_results()]
draft = begin_plan_revision(state.plan_version, "Add aggregate over completed groups")
add_task(draft, "total", "Sum group sums", store_input(values), "Both groups")
set_task_acceptance(draft, "total", ["Sum both totals"], "integer")
assert commit_plan(draft).plan_version == 2
assert not validate_plan_completion().valid
receipt = rlm_query("total")
submit_root_answer(final_text=str(read_accepted_result(receipt.receipt_id).value))
""",
            ],
            ("CHILD", "a"): [
                'partial = context[0]\nprint(partial)\nsubmit_candidate("missing")',
                'assert partial == 1\nsubmit_candidate("partial")',
                'wrong = {"other": partial}\nsubmit_candidate("wrong")',
                'wrong = {"sum": partial, "extra": 0}\nsubmit_candidate("wrong")',
                'partial += context[1]\nsubmit_candidate("partial")\nraise ValueError("rollback")',
                'assert partial == 3\nresult = {"sum": partial}\nsubmit_candidate("result", ["context"])',
            ],
            ("CHILD", "b"): [
                'assert "partial" not in globals()\nresult = {"sum": sum(context)}\nsubmit_candidate("result")',
            ],
            ("CHILD", "total"): [
                'assert context == [3, 12]\nresult = sum(context)\nsubmit_candidate("result")',
            ],
        }
    )
    run = SessionRuntime(model, tmp_path)
    result = run.solve("Add the sums from the two groups", {"a": [1, 2], "b": [4, 8]})
    assert result.status == "COMPLETED", result
    assert result.final_text == "15"
    assert [r.candidate_version for r in run.submissions.list_accepted_results()] == [1, 1, 1]
    assert run.plan.records["a"].attempt_count == 1

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["llm_calls"] == 12
    assert metrics["roles"]["ROOT"]["llm_calls"] == 4
    assert metrics["roles"]["CHILD"]["llm_calls"] == 8
    assert metrics["roles"]["SUPERVISOR"]["llm_calls"] == 0
    assert metrics["candidates_submitted"] == metrics["completed_tasks"] == 3
    assert metrics["candidates_invalid"] == 4
    assert metrics["candidates_aborted"] == 1
    assert metrics["candidate_submission_attempts"] == 8
    assert metrics["output_format_compliance_rate"] == 3 / 7
    assert metrics["token_usage_coverage"] == 1
    child_calls = [(meta, history) for meta, history in model.calls if meta["task_id"] == "a"]
    assert len({meta["session_id"] for meta, _ in child_calls}) == 1
    for (_, history), feedback in zip(
        child_calls[1:],
        (
            "Unknown submitted variable",
            "Expected object",
            "Missing required fields",
            "Expected 1 items",
            "Submission aborted",
        ),
        strict=True,
    ):
        assert feedback in history[-1]["content"]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    states = [
        item["data"]["status"]
        for item in events
        if item["event"] == "task_state" and item["data"]["task_id"] == "a"
    ]
    assert states == ["RUNNING", "SUBMITTED", "ACCEPTED"]
    assert {meta["role"] for meta, _ in model.calls} == {"ROOT", "CHILD"}
    assert not (tmp_path / "reviews").exists()


@pytest.mark.parametrize(
    "limits,reason,steps",
    [
        ({"max_child_steps": 1}, "Child step budget exhausted", 1),
        ({"max_tokens": 4096}, "Global model calls or token budget exhausted", 0),
    ],
)
def test_failed_child_has_no_result_and_blocks_completion(tmp_path, limits, reason, steps):
    configuration = RunConfig(**limits)
    model = ScriptedModel({("CHILD", "a"): ["result = sum(context)"]})
    run = SessionRuntime(model, tmp_path)
    handle = run.start_run("Sum groups", [], configuration)
    run.plan.commit_plan(register(run))
    outcome = run.dispatcher.dispatch_registered("a")
    assert outcome.status == TaskStatus.FAILED and outcome.receipt is None
    assert reason in outcome.reason
    assert not run.root.validate_plan_completion().valid
    with pytest.raises(ContractError, match="PLAN_INCOMPLETE"):
        run.finalize_run(handle.run_id, final_text="premature")
    assert run.dispatcher.dispatch_registered("a") == outcome
    metrics = run.get_run_state(handle.run_id)["metrics"]
    assert metrics.step_limits_reached == steps
    assert metrics.failed_tasks == 1 and metrics.completed_tasks == 0
    assert not list(tmp_path.glob("commits/*.json"))


@pytest.mark.parametrize("backend_name", ["qwen", "deepseek"])
def test_thinking_on_http_wire_in_root_single_and_batch_children(tmp_path, backend_name):
    config = RunConfig()
    base = "https://example.test/v1"
    requests = []
    counters = defaultdict(int)
    lock = Lock()
    root_code = """
d = create_plan("Compute")
for name in ["single", "batched"]:
    add_task(d, name, "Sum", store_input(context), name)
    set_task_acceptance(d, name, ["Add everything"], "integer")
commit_plan(d)
single = rlm_query("single")
batch = rlm_query_batched(["batched"])
assert batch[0].status == "ACCEPTED"
submit_root_answer(final_text=str(sum(read_accepted_result(r.receipt_id).value for r in [single, batch[0].receipt])))
"""

    def transport(request):
        payload = json.loads(request.content)
        if request.url.path == "/tokenize":
            assert payload["chat_template_kwargs"]["enable_thinking"] is True
            return httpx.Response(200, json={"count": 250})

        brief = json.loads(payload["messages"][1]["content"])
        task = brief.get("task", {}).get("task_id", "root")
        with lock:
            index = counters[task]
            counters[task] += 1
            requests.append((task, payload))
        if backend_name == "deepseek":
            assert payload["thinking"] == {"type": "enabled"}
            assert "chat_template_kwargs" not in payload
        else:
            assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert payload["max_tokens"] == 4096
        assert payload["temperature"] == 0.6 and payload["top_p"] == 0.95

        content, finish = block(root_code), "stop"
        if task != "root":
            if index == 0:
                content = None if task == "single" else "```repl\nbad = True"
                finish = "length"
            else:
                assert "without a complete REPL block" in payload["messages"][-1]["content"]
                content = block(
                    'assert "bad" not in globals()\nresult = sum(context)\nsubmit_candidate("result")'
                )
        usage = {
            "prompt_tokens": 250,
            "completion_tokens": 50,
            "total_tokens": 300,
            "completion_tokens_details": {"reasoning_tokens": 10},
        }
        if backend_name == "deepseek":
            usage.update(prompt_cache_hit_tokens=200, prompt_cache_miss_tokens=50)
        else:
            usage["prompt_tokens_details"] = {"cached_tokens": 200}
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "created": 0,
                "model": payload["model"],
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": block("bad = True"),
                        },
                        "finish_reason": finish,
                    }
                ],
                "usage": usage,
            },
        )

    def count_tokens(messages, thinking):
        assert messages and thinking
        return 250

    client = (
        DeepSeekFlashClient(config, api_key="mock", encode=count_tokens)
        if backend_name == "deepseek"
        else QwenClient(base, "mock-qwen", config)
    )
    with client:
        client.sdk.close()
        client.http = httpx.Client(transport=httpx.MockTransport(transport))
        client.sdk = OpenAI(base_url=client.base_url, api_key="mock", http_client=client.http)
        outcome = SessionRuntime(client, tmp_path).solve("Add results", [4, 5], config)
    assert outcome.status == "COMPLETED", outcome
    assert outcome.final_text == "18"
    assert counters == {"root": 1, "single": 2, "batched": 2}
    assert all(
        "reasoning_content" not in message
        for _, payload in requests
        for message in payload["messages"]
    )
    for wire in (tmp_path / "wire").iterdir():
        assert {p.name for p in wire.iterdir()} == {
            "meta.json",
            "request.json",
            "response.json",
            "model-output.txt",
        }
        assert json.loads((wire / "meta.json").read_text())["thinking"] is True
        raw = json.loads((wire / "response.json").read_text())
        assert raw["choices"][0]["message"]["reasoning_content"] == block("bad = True")
        assert (wire / "model-output.txt").read_text() == (
            raw["choices"][0]["message"]["content"] or ""
        )
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["length_finishes"] == metrics["roles"]["CHILD"]["length_finishes"] == 2
    assert metrics["reasoning_tokens"] == 50
    assert metrics["prompt_cache_hit_tokens"] == 1000
    assert metrics["prompt_cache_miss_tokens"] == 250
