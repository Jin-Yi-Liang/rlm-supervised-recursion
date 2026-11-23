from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from threading import Lock

import pytest

from rlm.supervised_rlm import RunConfig, SessionRuntime
from rlm.supervised_rlm.client import ModelReply
from rlm.supervised_rlm.metrics import (
    LLMCallRecord,
    MetricsCollector,
    TokenUsage,
    ToolCallRecord,
)
from rlm.supervised_rlm.repl import REPLRuntime
from rlm.supervised_rlm.schemas import AgentRole, ContractError, ReviewDecision, TaskStatus
from rlm.supervised_rlm.state_broker import StateBroker
from rlm.supervised_rlm.submission import SubmissionRegistry


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
        import json

        meta = json.loads((request_dir / "meta.json").read_text())
        key = (
            meta["role"],
            meta["task_id"],
            meta["candidate_version"] if meta["role"] == "SUPERVISOR" else None,
        )
        assert thinking == (meta["role"] != "CHILD")
        with self.lock:
            index = self.cursors[key]
            self.cursors[key] += 1
            self.calls.append((meta, list(messages)))
        answer = self.responses[key][index]
        if isinstance(answer, Exception):
            raise answer
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


def test_snapshot_paging_immutability_and_review_gate(tmp_path):
    run = SessionRuntime(ScriptedModel(), tmp_path)
    run.start_run("Sum two groups", [])
    run.plan.commit_plan(register(run))
    run.plan.transition("a", TaskStatus.RUNNING)
    registry = SubmissionRegistry(run.folder, run.plan.artifacts, run.metrics)
    intent = registry.prepare_submission(1, "total", ["data"])
    data = [1, 2, 3, 4]
    entries = [
        {"name": "total", "type_name": "int", "size_bytes": 1, "available": True, "value": 3},
        {"name": "data", "type_name": "list", "size_bytes": 12, "available": True, "value": data},
        {"name": "helper", "type_name": "function", "size_bytes": 0, "available": False},
    ]
    snapshot = registry.finalize_submission(run.plan.spec("a"), intent, entries, ())
    data[0] = 999
    broker = StateBroker(run.plan.artifacts, run.plan, snapshot)
    assert broker.list_child_variables().names() == ("total", "data", "helper")
    first = broker.inspect_child_variable("data", limit=2)
    assert first.value == [1, 2] and first.truncated and first.next_offset == 2
    first.value[0] = 555
    assert broker.read_artifact(first.evidence_ref, offset=2, limit=2).value == [3, 4]
    assert broker.inspect_child_variable("data", path="$[0]").value == 1
    assert broker.read_task_input().value == [1, 2]
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        run.gate.validate_verdict(broker, "a", 2, "ACCEPT", "Wrong version", [first.evidence_ref])
    with pytest.raises(ContractError, match="PERMISSION_DENIED"):
        run.gate.validate_verdict(broker, "a", 1, "ACCEPT", "Fabricated", ["unknown"])
    with pytest.raises(ContractError, match="MISSING_EVIDENCE"):
        run.gate.validate_verdict(broker, "a", 1, "ACCEPT", "No evidence", [])
    result = broker.inspect_child_variable("total")
    verdict = run.gate.validate_verdict(
        broker, "a", 1, "ACCEPT", "Checked sum", [result.evidence_ref]
    )
    broker.close()
    with pytest.raises(ContractError, match="PERMISSION_DENIED"):
        broker.inspect_child_variable("data")
    run.plan.transition("a", TaskStatus.SUBMITTED, candidate_version=1)
    run.plan.transition("a", TaskStatus.REVIEWING)
    receipt = run.gate.commit_accepted_candidate(snapshot, verdict)
    assert run.gate.commit_accepted_candidate(snapshot, verdict) == receipt
    assert run.gate.list_accepted_results() == [receipt]
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        run.gate.commit_accepted_candidate(
            snapshot, replace(verdict, decision=ReviewDecision.REVISE)
        )
    revision = run.plan.begin_plan_revision(1, "Recheck a against different authoritative data")
    run.plan.update_task(revision, "a", input_ref=run.root.store_input([2, 3]))
    run.plan.commit_plan(revision)
    assert run.gate.list_accepted_results() == []
    with pytest.raises(ContractError, match="PERMISSION_DENIED"):
        run.gate.read_accepted_result(receipt.receipt_id)
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        run.gate.commit_accepted_candidate(snapshot, verdict)


def test_end_to_end_repair_same_session_and_measured_events(tmp_path):
    model = ScriptedModel(
        {
            ("ROOT", "root", None): [
                """
d = create_plan("Sum both groups")
for key in ["a", "b"]:
    add_task(d, key, "Sum the assigned numbers", store_input(context[key]), key)
    set_task_acceptance(d, key, ["Include each number once"], "integer")
commit_plan(d)
""",
                'submit_root_answer(final_text="premature")',
                'outcomes = rlm_query_batched(["a", "b"])\nraise AssertionError("Failed after dispatch")',
                """
assert validate_plan_completion().valid
total = sum(read_accepted_result(item.receipt_id).value for item in list_accepted_results())
submit_root_answer(final_text=str(total))
""",
            ],
            ("CHILD", "a", None): [
                'submit_candidate("missing")',
                'partial_sum = context[0]\nsubmit_candidate("partial_sum")\nraise ValueError("test rollback")',
                'assert partial_sum == 1\nsubmit_candidate("partial_sum", ["context"])',
                'assert partial_sum == 1\ntotal = partial_sum + context[1]\nsubmit_candidate("total", ["context"])',
            ],
            ("CHILD", "b", None): [
                'total = sum(context)\nsubmit_candidate("total", ["context"])',
            ],
            ("SUPERVISOR", "a", 1): [
                """
found = inspect_child_variable("partial_sum")
original = read_task_input()
assert found.value == 1 and sum(original.value) == 3
submit_review("a", 1, "REVISE", "Only first row processed", [found.evidence_ref, original.evidence_ref], "Include the second row")
""",
            ],
            ("SUPERVISOR", "a", 2): [
                """
assert "found" not in globals() and "partial_sum" not in globals()
found = inspect_child_variable("total")
assert found.value == sum(read_task_input().value)
submit_review("a", 2, "ACCEPT", "Recomputed input sum", [found.evidence_ref])
""",
            ],
            ("SUPERVISOR", "b", 1): [
                """
found = inspect_child_variable("total")
assert found.value == sum(read_task_input().value)
submit_review("b", 1, "ACCEPT", "Recomputed input sum", [found.evidence_ref])
""",
            ],
        }
    )
    run = SessionRuntime(model, tmp_path)
    result = run.solve("Add the sums from the two groups", {"a": [1, 2], "b": [4, 8]})
    assert result.status == "COMPLETED", result
    assert result.final_text == "15"
    assert [item.candidate_version for item in run.gate.list_accepted_results()] == [2, 1]
    metrics = run.get_run_state(result.run_id)["metrics"]
    assert metrics.llm_calls == 12
    assert metrics.roles["ROOT"].llm_calls == 4
    assert metrics.roles["CHILD"].llm_calls == 5
    assert metrics.roles["SUPERVISOR"].llm_calls == 3
    assert metrics.candidates_submitted == 3
    assert metrics.candidates_aborted == 2
    assert (
        metrics.candidates_rejected
        == metrics.repair_attempts
        == metrics.repaired_tasks_accepted
        == 1
    )
    assert metrics.candidates_accepted == metrics.completed_tasks == 2
    assert metrics.token_usage_coverage == 1
    root_calls = [messages for meta, messages in model.calls if meta["role"] == "ROOT"]
    assert "Authoritative progress after this cell:" in root_calls[-1][-1]["content"]
    assert '"pending_task_ids": []' in root_calls[-1][-1]["content"]
    child_calls = [
        entry for entry in model.calls if entry[0]["task_id"] == "a" and entry[0]["role"] == "CHILD"
    ]
    assert len({item["session_id"] for item, _ in child_calls}) == 1
    assert "Include the second row" in child_calls[-1][1][-1]["content"]
    for meta, messages in model.calls:
        if meta["role"] == "SUPERVISOR":
            assert messages[0]["role"] == "system" and len(messages) == 2
            assert "partial_sum + context[1]" not in messages[1]["content"]


@pytest.mark.parametrize(
    "decision,status",
    [
        ("PLAN_ISSUE", TaskStatus.PLAN_ISSUE),
        ("REVIEW_INCOMPLETE", TaskStatus.REVIEW_INCOMPLETE),
    ],
)
def test_review_failures_never_publish(tmp_path, decision, status):
    model = ScriptedModel(
        {
            ("CHILD", "a", None): ['total = sum(context)\nsubmit_candidate("total")'],
            ("SUPERVISOR", "a", 1): [
                f'submit_review("a", 1, "{decision}", "Unable to verify scope", [], "Clarify the scope")',
            ],
        }
    )
    run = SessionRuntime(model, tmp_path)
    handle = run.start_run("Sum groups", [])
    run.plan.commit_plan(register(run))
    result = run.dispatcher.dispatch_registered("a")
    assert result.status == status and result.receipt is None
    assert run.gate.list_accepted_results() == []
    assert run.plan.plan.plan_version == 1
    assert run.abort_run(handle.run_id, "Test cancellation").status == "CANCELLED"
    metrics = run.get_run_state(handle.run_id)["metrics"]
    assert metrics.candidates_rejected == metrics.candidates_accepted == 0


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
    model = ScriptedModel({("ROOT", "root", None): ["print(42)"]})
    runtime = SessionRuntime(model, Path(tmp_path))
    outcome = runtime.solve("Add 4 and 2", [4, 2], RunConfig(max_model_calls=1))
    assert outcome.status == "INCOMPLETE"
    assert runtime.get_run_state(outcome.run_id)["metrics"].llm_calls == 1


def test_plan_issue_returns_to_root_and_revision_binds_original_input(tmp_path):
    model = ScriptedModel(
        {
            ("ROOT", "root", None): [
                """
assert context == "original"
try:
    store_input(context_ref)
except ValueError as error:
    assert "already an artifact handle" in str(error)
else:
    raise AssertionError("An artifact handle must not be wrapped as data")
draft = create_plan("Inspect the original input")
add_task(draft, "read", "Return the complete original text", store_input("wrong"), "complete input")
set_task_acceptance(draft, "read", ["Return the original text"], "string")
commit_plan(draft)
print(rlm_query_batched(["read"]))
""",
                """
state = get_plan_state()
assert state.tasks[0].status == "PLAN_ISSUE"
revision = begin_plan_revision(state.plan_version, "Use the authoritative original data")
update_task(revision, "read", input_ref=context_ref)
assert commit_plan(revision).plan_version == 2
receipt = rlm_query("read")
submit_root_answer(final_text=read_accepted_result(receipt.receipt_id).value)
""",
            ],
            ("CHILD", "read", None): [
                'result = context\nsubmit_candidate("result")',
                'result = context\nsubmit_candidate("result")',
            ],
            ("SUPERVISOR", "read", 1): [
                """
source = read_task_input()
assert source.value == "wrong"
submit_review("read", 1, "PLAN_ISSUE", "Wrong assigned input", [source.evidence_ref],
              "Use the original text reference")
""",
            ],
            ("SUPERVISOR", "read", 2): [
                """
source = read_task_input()
result = inspect_child_variable("result")
assert source.value == result.value == "original"
submit_review("read", 2, "ACCEPT", "Original text verified", [source.evidence_ref, result.evidence_ref])
""",
            ],
        }
    )
    run = SessionRuntime(model, tmp_path)
    outcome = run.solve("Return the text of the original input", "original")
    assert outcome.status == "COMPLETED"
    assert outcome.final_text == "original"
    assert run.plan.spec("read").task_version == 2
    assert run.gate.list_accepted_results()[0].candidate_version == 2
    assert run.get_run_state(outcome.run_id)["metrics"].plan_issues == 1
