import json

import pytest

from rlm.core.supervised import ReviewFailure, SupervisedRLM


def cell(code):
    return "```repl\n" + code + "\n```"


class Scripted:
    def __init__(self, steps, limit=100000):
        self.steps = iter(steps)
        self.limit = limit
        self.requests = []

    def fits(self, messages):
        return len(messages[-1]["content"]) <= self.limit

    def chat(self, role, task, messages):
        self.requests.append((role, task, messages.copy()))
        if role == "supervisor":
            packet = json.loads(messages[-1]["content"])
            if packet["mode"] == "SEGMENT":
                decision = "SEGMENT_NOTES"
            else:
                expected, decision = next(self.steps)
                assert expected == role
            return json.dumps(
                {
                    "decision": decision,
                    "task_id": task,
                    "candidate_version": packet["candidate_version"],
                    "evidence_refs": ["turn:1"],
                    "finding": "check the input",
                    "requested_action": "" if decision == "ACCEPT" else "fix sum",
                }
            )
        expected, code = next(self.steps)
        assert expected == role
        return cell(code)


@pytest.mark.parametrize("planned", [False, True])
def test_nested_receipt_after_same_repl_repair(tmp_path, planned):
    outer = (
        (
            "register_plan([dict(task_id='outer', instruction='sum', input=context, scope='all', "
            "acceptance=['sum'])]); receipt = rlm_query('outer')"
        )
        if planned
        else ("receipt = rlm_query('sum', input=context)")
    )
    inner = (
        (
            "register_plan([dict(task_id='leaf', instruction='add', input=context, scope='all', "
            "acceptance=['sum'])]); nested = rlm_query('leaf')"
        )
        if planned
        else "nested = rlm_query('add', input=context)"
    )
    service = Scripted(
        [
            ("root", outer),
            ("sub", inner),
            ("sub", "saved = sum(context); answer['content'] = 0; answer['ready'] = True"),
            ("supervisor", "REVISE"),
            ("sub", "answer['content'] = saved; answer['ready'] = True"),
            ("supervisor", "ACCEPT"),
            (
                "sub",
                "answer['content'] = read_result(nested['result_ref']); answer['ready'] = True",
            ),
            ("supervisor", "ACCEPT"),
            (
                "root",
                "answer['content'] = accepted_results[receipt['result_ref']]; answer['ready'] = True",
            ),
        ]
    )
    agent = SupervisedRLM(service, tmp_path, planned=planned)
    assert agent.solve("add", [4, 7]) == "11"
    assert agent.reviews == 3 and agent.rejections == 1
    assert len(agent.receipts) == len(agent.accepted) == 2
    assert (tmp_path / "task-table.json").exists() == planned
    assert agent.commit_candidate(agent.receipts[0]["task_id"], 2, 11) == agent.receipts[0]
    assert len(agent.receipts) == 2
    repaired = service.requests[4][2]
    assert "fix sum" in repaired[-1]["content"]
    assert "accepted" in service.requests[-1][2][-1]["content"]


def test_rejected_result_never_commits(tmp_path):
    turns = [("root", "receipt = rlm_query('check', input=context)")]
    for _ in range(3):
        turns.extend(
            [("sub", "answer['content'] = 0; answer['ready'] = True"), ("supervisor", "REVISE")]
        )
    turns.append(("root", "answer['content'] = receipt['status']; answer['ready'] = True"))
    agent = SupervisedRLM(Scripted(turns), tmp_path)
    assert agent.solve("test", "text") == "failed"
    assert len(agent.accepted) == 0


def test_batch_and_plan_gate(tmp_path):
    service = Scripted(
        [
            ("root", "answer['content']='premature'; answer['ready']=True"),
            (
                "root",
                "register_plan([dict(task_id=key, instruction='echo', input=key, scope=key, "
                "acceptance=['echo']) for key in ['first','second']]); "
                "receipts = rlm_query(['first','second'])",
            ),
            ("sub", "answer['content']=context; answer['ready']=True"),
            ("supervisor", "ACCEPT"),
            ("sub", "answer['content']=context; answer['ready']=True"),
            ("supervisor", "ACCEPT"),
            (
                "root",
                "answer['content'] = [read_result(r['result_ref']) for r in receipts]; answer['ready']=True",
            ),
        ]
    )
    assert SupervisedRLM(service, tmp_path, planned=True).solve("echo", "") == "['first', 'second']"


def test_oversize_review_covers_all_segments(tmp_path):
    service = Scripted(
        [
            ("root", "r = rlm_query('echo', input=context)"),
            ("sub", "answer['content']=1; answer['ready']=True"),
            ("supervisor", "ACCEPT"),
            ("root", "answer['content']=1; answer['ready']=True"),
        ],
        limit=4000,
    )
    agent = SupervisedRLM(service, tmp_path)
    assert agent.solve("echo", "z" * 10000) == "1"
    packets = [
        json.loads(m[-1]["content"]) for role, _, m in service.requests if role == "supervisor"
    ]
    pieces = [p["data"] for p in packets if p["mode"] == "SEGMENT"]
    assert len(pieces) > 1 and pieces[0]["start"] == 0
    assert all(
        left["end"] == right["start"] for left, right in zip(pieces, pieces[1:], strict=False)
    )
    assert pieces[-1]["end"] == pieces[0]["total_chars"]
    assert packets[-1]["mode"] == "FINAL_FROM_SEGMENTS"


def test_incomplete_review_halts_without_commit(tmp_path):
    service = Scripted(
        [
            ("root", "rlm_query('test', input=context)"),
            ("sub", "answer['content']=1; answer['ready']=True"),
            ("supervisor", "REVIEW_INCOMPLETE"),
        ]
    )
    agent = SupervisedRLM(service, tmp_path)
    with pytest.raises(ReviewFailure):
        agent.solve("test", "")
    assert not agent.accepted


def test_native_rebound_answer_preserves_typed_payload(tmp_path):
    service = Scripted(
        [
            ("root", "receipt = rlm_query('sum', input=context)"),
            ("sub", "answer = {'content': {'sum': sum(context)}, 'ready': True}"),
            ("supervisor", "ACCEPT"),
            (
                "root",
                "answer = {'content': read_result(receipt['result_ref'])['sum'], 'ready': True}",
            ),
        ]
    )
    agent = SupervisedRLM(service, tmp_path)
    assert agent.solve("sum", [4, 7]) == "11"
    assert agent.accepted[agent.receipts[0]["result_ref"]] == {"sum": 11}
