import copy
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from rlm.core import supervised_prompts as prompts
from rlm.core.rlm import RLM
from rlm.environments.local_repl import LocalREPL
from rlm.utils.parsing import format_iteration


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class ReviewFailure(RuntimeError):
    pass


class Accepted(Mapping):
    def __init__(self):
        self._values = {}

    def __getitem__(self, key):
        return copy.deepcopy(self._values[key])

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class TaskREPL(LocalREPL):
    def execute_code(self, code):
        self.candidate = None
        result = super().execute_code(code)
        if result.final_answer is not None:
            value = self.candidate
            if value is None:
                value = self.locals["answer"].get("content")
            result.candidate_value = copy.deepcopy(value)
        return result

    def _restore_scaffold(self):
        super()._restore_scaffold()
        for name in ("llm_query", "llm_query_batched", "rlm_query_batched", "rlm_query"):
            self.globals.pop(name, None)
            self.locals.pop(name, None)
        if hasattr(self, "task_tools"):
            self.globals.update(self.task_tools)

    def _capture_answer(self, content):
        self.candidate = copy.deepcopy(content)
        super()._capture_answer(content)


class SupervisedRLM(RLM):
    """Opt-in sequential RLM using native completion turns and persistent LocalREPLs."""

    def __init__(self, backend, folder, planned=False, max_iterations=20):
        self.service = backend
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.planned = planned
        self.max_iterations = max_iterations
        self.accepted = Accepted()
        self.receipts = []
        self.tasks = {}
        self.serial = 0
        self.reviews = 0
        self.rejections = 0
        self.fatal = None

    def event(self, kind, **fields):
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encode({"event": kind, **fields}) + "\n")

    def state(self):
        return {
            key: {name: val for name, val in spec.items() if name != "input"}
            for key, spec in self.tasks.items()
        }

    def update(self, task, status):
        self.tasks[task]["status"] = status
        if self.planned:
            dump(self.folder / "task-table.json", self.state())
        self.event("status", task=task, status=status)

    def register(self, parent, specs, replacement=None):
        if not isinstance(specs, list) or not specs:
            raise ValueError("Provide a nonempty list of task specifications.")
        additions = {}
        for raw in specs:
            spec = json.loads(encode(raw))
            key = spec["task_id"]
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[\w-]{1,100}", key)
                or key == "root"
                or key in self.tasks
                or key in additions
            ):
                raise ValueError(f"Invalid or duplicate task ID: {key!r}")
            if not spec["instruction"] or "input" not in spec:
                raise ValueError("Task needs instruction and input.")
            if self.planned and (not spec.get("acceptance") or not spec.get("scope")):
                raise ValueError("Provide scope and acceptance criteria.")
            additions[key] = {
                **spec,
                "parent": parent,
                "status": "registered",
                "required": spec.get("required", True),
            }
        if replacement and not any(item["required"] for item in additions.values()):
            raise ValueError("Replacement must cover the required obligation.")
        for key, spec in additions.items():
            dump(self.folder / "tasks" / key / "input.json", spec["input"])
            spec["input_ref"] = f"tasks/{key}/input.json"
            self.tasks[key] = spec
            self.update(key, "registered")
        return list(additions)

    def revise(self, owner, old, reason, specs):
        if owner != "root" or not reason or self.tasks[old]["status"] in {"accepted", "superseded"}:
            raise ValueError("Only root can replace unfinished tasks with a reason.")
        keys = self.register(self.tasks[old]["parent"], specs, replacement=old)
        self.tasks[old].update(replacements=keys, revision_reason=reason)
        self.update(old, "superseded")
        return keys

    def commit_candidate(self, task, version, candidate):
        ref = f"accepted/{task}/v{version}.json"
        if ref in self.accepted:
            return next(item for item in self.receipts if item["result_ref"] == ref)
        if self.tasks[task]["status"] == "accepted":
            raise ReviewFailure("Attempted double acceptance")
        value = json.loads(encode(candidate))
        dump(self.folder / ref, value)
        self.accepted._values[ref] = value
        receipt = dict(status="accepted", task_id=task, candidate_version=version, result_ref=ref)
        self.tasks[task]["result_ref"] = ref
        self.update(task, "accepted")
        self.receipts.append(receipt)
        self.event("accepted", **receipt)
        return receipt

    def dispatch(self, owner, depth, request, input=None):
        batch = isinstance(request, list)
        items = request if batch else [request]
        if batch and input is not None:
            raise ValueError("Supply each batch item's own input.")
        replies = []
        for item in items:
            if self.planned:
                key = item
                if input is not None or key not in self.tasks or self.tasks[key]["parent"] != owner:
                    raise ValueError("Dispatch a task registered under this parent.")
            else:
                self.serial += 1
                key = f"sub-{self.serial:06d}"
                spec = item if isinstance(item, dict) else {"instruction": item, "input": input}
                if spec["input"] is None:
                    spec = {**spec, "input": ""}
                self.register(owner, [{**spec, "task_id": key}])
            spec = self.tasks[key]
            if spec["status"] == "accepted":
                replies.append(next(rec for rec in self.receipts if rec["task_id"] == key))
                continue
            if spec["status"] != "registered":
                raise ValueError("Task is not dispatchable.")
            self.update(key, "running")
            try:
                replies.append(self.node(key, depth + 1, spec["instruction"], spec["input"]))
            except BaseException as error:
                self.fatal = error
                raise
        return replies if batch else replies[0]

    def review(self, task, version, candidate, record):
        self.reviews += 1
        system = prompts.REVIEW + ("\n" + prompts.PLAN_REVIEW if self.planned else "")
        payload = {
            "root_question": self.question,
            "assignment": self.tasks[task],
            "visible_execution": record,
            "candidate": candidate,
        }
        if self.planned:
            payload["plan"] = self.state()
        transcript = encode(payload)

        def messages(mode, data):
            return [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": encode(
                        dict(mode=mode, task_id=task, candidate_version=version, data=data)
                    ),
                },
            ]

        def inspect(mode, data, permitted):
            convo = messages(mode, data)
            for attempt in range(2):
                raw = self.service.chat("supervisor", task, convo)
                cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw).strip()
                try:
                    decision = json.loads(cleaned)
                    action = decision["decision"]
                    if action in {"ACCEPT", "SEGMENT_NOTES", "REVIEW_INCOMPLETE"}:
                        decision.setdefault("requested_action", "")
                    if (
                        action not in permitted
                        or decision["task_id"] != task
                        or type(decision["candidate_version"]) is not int
                        or decision["candidate_version"] != version
                        or not isinstance(decision["evidence_refs"], list)
                        or (action == "ACCEPT" and not decision["evidence_refs"])
                        or not isinstance(decision["finding"], str)
                        or not decision["finding"].strip()
                        or not isinstance(decision["requested_action"], str)
                        or (
                            action in {"REVISE", "PLAN_ISSUE"}
                            and not decision["requested_action"].strip()
                        )
                    ):
                        raise ValueError("invalid reviewer envelope")
                except (KeyError, TypeError, ValueError) as error:
                    self.event("invalid_review", task=task, version=version, error=str(error))
                    if attempt:
                        raise ReviewFailure(
                            f"Invalid supervisor output for {task}: {error}"
                        ) from error
                    convo = [
                        *convo,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": "Return one valid review object. Include decision, task_id, integer "
                            "candidate_version, evidence_refs, finding and requested_action. Use the task ID "
                            "and version in the user payload. Do not approve without evidence.",
                        },
                    ]
                    continue
                self.event("review", mode=mode, **decision)
                return decision

        allowed = {"ACCEPT", "REVISE", "REVIEW_INCOMPLETE"}
        if self.planned:
            allowed.add("PLAN_ISSUE")
        if self.service.fits(messages("WHOLE", payload)):
            return inspect("WHOLE", payload, allowed)
        segments = []
        pending = [(0, transcript)]
        while pending:
            start, part = pending.pop()
            data = {
                "start": start,
                "end": start + len(part),
                "total_chars": len(transcript),
                "text": part,
            }
            if not self.service.fits(messages("SEGMENT", data)):
                half = len(part) // 2
                if not half:
                    raise ReviewFailure("Cannot fit review segment")
                pending.extend([(start + half, part[half:]), (start, part[:half])])
                continue
            notes = inspect("SEGMENT", data, {"SEGMENT_NOTES"})
            segments.append({"start": start, "end": start + len(part), "notes": notes})
        final = {"total_chars": len(transcript), "segments": segments}
        if not self.service.fits(messages("FINAL_FROM_SEGMENTS", final)):
            raise ReviewFailure("All segment notes cannot fit; not accepting partial review")
        return inspect("FINAL_FROM_SEGMENTS", final, allowed)

    def node(self, task, depth, instruction, data):
        root = depth == 0
        role = "root" if root else "sub"
        system = "\n".join(
            [
                prompts.ROOT if root else prompts.CHILD,
                prompts.REPL,
                (prompts.PLAN if self.planned else prompts.FREE) if depth < 2 else prompts.LEAF,
            ]
        )
        tools = {"read_result": self.accepted.__getitem__}
        if root:
            tools["accepted_results"] = self.accepted
        if depth < 2:
            tools["rlm_query"] = lambda request, input=None: self.dispatch(
                task, depth, request, input
            )
            if self.planned:
                tools["register_plan"] = lambda specs: self.register(task, specs)
                tools["task_state"] = lambda: copy.deepcopy(self.state())
                if root:
                    tools["revise_plan"] = lambda old, reason, specs: self.revise(
                        task, old, reason, specs
                    )
        repl = TaskREPL(context_payload=data)
        repl.task_tools = tools
        repl._restore_scaffold()
        brief = {
            "instruction": instruction,
            "task_id": task,
            "may_delegate": depth < 2,
            "context_type": type(data).__name__,
            "context_length": len(data) if hasattr(data, "__len__") else None,
        }
        if self.planned and not root:
            brief.update(scope=self.tasks[task]["scope"], acceptance=self.tasks[task]["acceptance"])
        history = [
            {"role": "system", "content": system},
            {"role": "user", "content": encode(brief)},
        ]
        record = history.copy()
        handler = SimpleNamespace(
            completion=lambda messages: self.service.chat(role, task, messages)
        )
        seen, version = 0, 0
        try:
            for turn in range(1, self.max_iterations + 1):
                if root and seen < len(self.receipts):
                    notice = {
                        "role": "user",
                        "content": "New accepted receipts: " + encode(self.receipts[seen:]),
                    }
                    history.append(notice)
                    record.append(notice)
                    self.event("root_notified", receipts=self.receipts[seen:])
                    seen = len(self.receipts)
                iteration = self._completion_turn(history, handler, repl)
                if self.fatal:
                    raise self.fatal
                visible = {
                    "turn": turn,
                    "assistant": iteration.response,
                    "blocks": [
                        {
                            "code": block.code,
                            "stdout": block.result.stdout,
                            "stderr": block.result.stderr,
                            "final_answer": block.result.final_answer,
                        }
                        for block in iteration.code_blocks
                    ],
                }
                record.append(visible)
                self.event("turn", task=task, **visible)
                history.extend(format_iteration(iteration))
                submission = next(
                    (
                        block.result
                        for block in iteration.code_blocks
                        if block.result.final_answer is not None
                    ),
                    None,
                )
                if submission is None:
                    if not iteration.code_blocks:
                        reminder = {
                            "role": "user",
                            "content": "Use a fenced repl block to inspect or submit.",
                        }
                        history.append(reminder)
                        record.append(reminder)
                    continue
                candidate = submission.candidate_value
                repl.locals["answer"]["ready"] = False
                if root:
                    missing = [
                        key
                        for key, spec in self.tasks.items()
                        if spec["required"] and spec["status"] not in {"accepted", "superseded"}
                    ]
                    if self.planned and (not self.tasks or missing):
                        history.append(
                            {"role": "user", "content": "Plan incomplete: " + str(missing)}
                        )
                        continue
                    if seen < len(self.receipts):
                        history.append(
                            {
                                "role": "user",
                                "content": "Read new receipts before final submission.",
                            }
                        )
                        continue
                    return str(candidate)
                version += 1
                dump(self.folder / "tasks" / task / f"candidate-{version}.json", candidate)
                self.update(task, "reviewing")
                verdict = self.review(task, version, candidate, record)
                decision = verdict["decision"]
                if decision == "ACCEPT":
                    return self.commit_candidate(task, version, candidate)
                if decision == "REVIEW_INCOMPLETE":
                    raise ReviewFailure(f"Incomplete review of {task}")
                if decision == "PLAN_ISSUE":
                    self.update(task, "plan_issue")
                    return {"status": "plan_issue", "task_id": task, "issue": verdict["finding"]}
                self.rejections += 1
                if version == 3:
                    break
                feedback = {
                    "role": "user",
                    "content": "Candidate rejected; repair in this REPL: " + encode(verdict),
                }
                history.append(feedback)
                record.append(feedback)
                self.update(task, "repairing")
            if root:
                raise RuntimeError("Root did not finish within the iteration limit")
            self.update(task, "failed")
            return {"status": "failed", "task_id": task, "reason": "repair or iteration limit"}
        finally:
            repl.cleanup()

    def solve(self, question, context):
        self.question = question
        return self.node("root", 0, question, context)
