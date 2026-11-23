"""Versioned plans and authoritative task transitions."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from uuid import uuid4

from rlm.supervised_rlm.schemas import (
    AcceptanceSpec,
    ContractError,
    GlobalTaskSpec,
    OutputSpec,
    PlanChangeRecord,
    PlanDraft,
    PlanReceipt,
    PlanSpec,
    PlanState,
    TaskDraft,
    TaskRecord,
    TaskSpec,
    TaskStatus,
    ValidationReport,
)
from rlm.supervised_rlm.storage import ArtifactStore, strings, text, write_json

OUTPUT_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "any": object,
}
TRANSITIONS = {
    TaskStatus.PLANNED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.SUBMITTED, TaskStatus.FAILED},
    TaskStatus.SUBMITTED: {TaskStatus.REVIEWING},
    TaskStatus.REVIEWING: {
        TaskStatus.REPAIRING,
        TaskStatus.PLAN_ISSUE,
        TaskStatus.REVIEW_INCOMPLETE,
    },
    TaskStatus.REPAIRING: {TaskStatus.RUNNING, TaskStatus.FAILED},
}


class PlanManager:
    def __init__(
        self,
        task: GlobalTaskSpec,
        folder: Path,
        artifacts: ArtifactStore,
        max_tasks: int,
    ):
        self.task = task
        self.folder = folder
        self.artifacts = artifacts
        self.max_tasks = max_tasks
        self.plan = PlanSpec(task.run_id, 0, task.objective, ())
        self.drafts: dict[str, PlanDraft] = {}
        self.records: dict[str, TaskRecord] = {}
        self.lock = RLock()

    def create_plan(self, goal: str) -> str:
        if self.plan.plan_version:
            raise ContractError("VERSION_CONFLICT", "Use begin_plan_revision for an existing plan")
        draft_id = uuid4().hex
        self.drafts[draft_id] = PlanDraft(
            draft_id,
            self.task.run_id,
            text(goal, "goal"),
            0,
            "Initial plan",
        )
        return draft_id

    def draft(self, draft_id: str) -> PlanDraft:
        if not isinstance(draft_id, str) or draft_id not in self.drafts:
            raise ContractError("PLAN_NOT_FOUND", "Unknown plan draft")
        return self.drafts[draft_id]

    def copy_plan(self, draft_id: str, goal: str | None = None) -> str:
        original = self.draft(draft_id)
        key = uuid4().hex
        self.drafts[key] = replace(
            original,
            draft_id=key,
            goal=original.goal if goal is None else text(goal, "goal"),
            tasks=deepcopy(original.tasks),
        )
        return key

    def begin_plan_revision(self, expected_version: int, reason: str = "Update task plan") -> str:
        if (
            type(expected_version) is not int
            or not expected_version
            or expected_version != self.plan.plan_version
        ):
            raise ContractError("VERSION_CONFLICT", "Plan has a different current version")
        key = uuid4().hex
        self.drafts[key] = PlanDraft(
            key,
            self.task.run_id,
            self.plan.goal,
            expected_version,
            text(reason, "reason"),
            {
                task.task_id: TaskDraft(
                    task.task_id,
                    task.instruction,
                    task.scope,
                    task.input_refs,
                    task.required,
                    task.acceptance,
                    task.output_schema,
                )
                for task in self.plan.tasks
            },
        )
        return key

    def input_ref(self, ref_id: str) -> str:
        ref = self.artifacts.require(ref_id)
        if ref.source_namespace != "root":
            raise ContractError("PERMISSION_DENIED", "Task input must be a Root input reference")
        return ref_id

    def add_task(
        self,
        draft_id: str,
        task_id: str,
        instruction: str,
        input_ref: str,
        scope: str,
        required: bool = True,
    ) -> None:
        draft = self.draft(draft_id)
        task_id = text(task_id, "task_id", 80)
        if task_id == "root" or task_id in draft.tasks or task_id in self.records:
            raise ContractError("DUPLICATE_TASK", "Use a new task ID")
        if type(required) is not bool or len(draft.tasks) >= self.max_tasks:
            raise ContractError("INVALID_ARGUMENT", "Invalid required flag or task limit exceeded")
        draft.tasks[task_id] = TaskDraft(
            task_id,
            text(instruction, "instruction"),
            text(scope, "scope"),
            (self.input_ref(input_ref),),
            required,
        )

    def draft_task(self, draft_id: str, task_id: str) -> TaskDraft:
        draft = self.draft(draft_id)
        if not isinstance(task_id, str) or task_id not in draft.tasks:
            raise ContractError("TASK_NOT_FOUND", "Unknown task in draft")
        return draft.tasks[task_id]

    def update_task(
        self,
        draft_id: str,
        task_id: str,
        instruction: str | None = None,
        input_ref: str | None = None,
        scope: str | None = None,
        required: bool | None = None,
    ) -> None:
        task = self.draft_task(draft_id, task_id)
        if required is not None and type(required) is not bool:
            raise ContractError("INVALID_ARGUMENT", "required must be bool")
        self.drafts[draft_id].tasks[task_id] = replace(
            task,
            instruction=task.instruction
            if instruction is None
            else text(instruction, "instruction"),
            input_refs=task.input_refs if input_ref is None else (self.input_ref(input_ref),),
            scope=task.scope if scope is None else text(scope, "scope"),
            required=task.required if required is None else required,
        )

    def set_task_acceptance(
        self,
        draft_id: str,
        task_id: str,
        checks: list[str],
        output_type: str = "object",
        required_fields: list[str] | None = None,
        expected_count: int | None = None,
    ) -> None:
        task = self.draft_task(draft_id, task_id)
        conditions = strings(checks, "checks")
        if not isinstance(output_type, str) or output_type not in OUTPUT_TYPES:
            raise ContractError(
                "INVALID_ARGUMENT", f"Choose output_type from {', '.join(OUTPUT_TYPES)}"
            )
        if expected_count is not None and (
            type(expected_count) is not int
            or expected_count < 0
            or output_type not in {"array", "object", "string", "any"}
        ):
            raise ContractError("INVALID_ARGUMENT", "expected_count requires a sized result")
        fields = (
            ()
            if required_fields is None
            else strings(required_fields, "required_fields", allow_empty=True)
        )
        if fields and output_type != "object":
            raise ContractError("INVALID_ARGUMENT", "required_fields requires an object result")
        task.acceptance = AcceptanceSpec(conditions, expected_count)
        task.output_schema = OutputSpec(output_type, fields)

    def remove_task(self, draft_id: str, task_id: str) -> None:
        self.draft_task(draft_id, task_id)
        del self.drafts[draft_id].tasks[task_id]

    def discard_plan_draft(self, draft_id: str) -> None:
        self.draft(draft_id)
        del self.drafts[draft_id]

    def commit_plan(self, draft_id: str) -> PlanReceipt:
        with self.lock:
            draft = self.draft(draft_id)
            old = self.plan
            if draft.expected_version != old.plan_version:
                raise ContractError("VERSION_CONFLICT", "Draft is based on a stale plan")
            if not draft.tasks or not any(task.required for task in draft.tasks.values()):
                raise ContractError("PLAN_INVALID", "Plan requires at least one required task")
            if len(set(self.records) | set(draft.tasks)) > self.max_tasks:
                raise ContractError("PLAN_INVALID", "Run task budget exhausted")
            new_version = old.plan_version + 1
            before = {task.task_id: task for task in old.tasks}
            records = self.records.copy()
            tasks, superseded = [], []
            changed = set(before) - draft.tasks.keys()
            for item in draft.tasks.values():
                if item.acceptance is None or item.output_schema is None:
                    raise ContractError(
                        "PLAN_INVALID", f"Missing acceptance criteria for {item.task_id}"
                    )
                for ref_id in item.input_refs:
                    self.input_ref(ref_id)
                previous = before.get(item.task_id)
                spec = TaskSpec(
                    item.task_id,
                    item.instruction,
                    item.scope,
                    item.input_refs,
                    item.acceptance,
                    item.output_schema,
                    item.required,
                    previous.task_version if previous else 1,
                )
                if spec != previous:
                    changed.add(item.task_id)
                    record = records.get(item.task_id)
                    if previous:
                        self.check_plan_change(record)
                        spec = replace(spec, task_version=spec.task_version + 1)
                    if record and record.accepted_receipt_id:
                        superseded.append((record.accepted_receipt_id, spec.task_id))
                    records[item.task_id] = TaskRecord(
                        old.run_id,
                        item.task_id,
                        new_version,
                        spec.task_version,
                        candidate_version=record.candidate_version if record else 0,
                        required=spec.required,
                    )
                else:
                    records[item.task_id] = replace(records[item.task_id], plan_version=new_version)
                tasks.append(spec)

            for task_id in before.keys() - draft.tasks.keys():
                record = records[task_id]
                self.check_plan_change(record)
                if record.accepted_receipt_id:
                    superseded.append((record.accepted_receipt_id, ""))
                status = TaskStatus.SUPERSEDED if record.is_accepted() else TaskStatus.CANCELLED
                records[task_id] = replace(record, status=status, plan_version=new_version)

            removed_required = any(
                before[task_id].required
                and (task_id not in draft.tasks or not draft.tasks[task_id].required)
                for task_id in changed & before.keys()
            )
            if removed_required:
                replacements = [
                    task.task_id for task in tasks if task.required and task.task_id in changed
                ]
                if not replacements:
                    raise ContractError(
                        "PLAN_INVALID",
                        "Replacing required coverage needs a new or revised required task",
                    )
                superseded = [
                    (receipt, target or replacements[0]) for receipt, target in superseded
                ]

            new_plan = PlanSpec(old.run_id, new_version, draft.goal, tuple(tasks))
            change = PlanChangeRecord(
                uuid4().hex,
                old.plan_version,
                new_version,
                draft.reason,
                tuple(sorted(changed)),
                tuple(superseded),
            )
            write_json(
                self.folder / "plans" / f"v{new_version}.json",
                {
                    "plan": new_plan,
                    "change": change,
                    "records": records,
                },
            )
            self.plan, self.records = new_plan, records
            del self.drafts[draft_id]
            return PlanReceipt(old.run_id, new_version, f"plans/v{new_version}.json")

    def check_plan_change(self, record: TaskRecord) -> None:
        if record.status in {
            TaskStatus.RUNNING,
            TaskStatus.REPAIRING,
            TaskStatus.SUBMITTED,
            TaskStatus.REVIEWING,
        }:
            raise ContractError("TASK_BUSY", "Cannot change an active task")

    def spec(self, task_id: str) -> TaskSpec:
        for task in self.plan.tasks:
            if task.task_id == task_id:
                return task
        raise ContractError("TASK_NOT_FOUND", "Dispatch a task registered in the current plan")

    def transition(self, task_id: str, status: TaskStatus, **changes) -> TaskRecord:
        with self.lock:
            item = self.records[task_id]
            if status not in TRANSITIONS.get(item.status, ()):
                raise ContractError("INVALID_TRANSITION", f"Cannot move {item.status} to {status}")
            updated = replace(item, status=status, **changes)
            self.records[task_id] = updated
            return updated

    def state(self) -> PlanState:
        current = [self.records[task.task_id] for task in self.plan.tasks]
        return PlanState(
            self.plan.run_id,
            self.plan.plan_version,
            self.plan.goal,
            tuple(task.task_id for task in current if task.is_accepted()),
            tuple(task.task_id for task in current if not task.is_accepted()),
            tuple(task.accepted_receipt_id for task in current if task.is_accepted()),
            tuple(f"{task.task_id}: {task.last_error}" for task in current if task.last_error),
        )

    def validate_completion(self) -> ValidationReport:
        missing = [
            task.task_id
            for task in self.plan.tasks
            if task.required and not self.records[task.task_id].is_accepted()
        ]
        active = [
            record.task_id
            for record in self.records.values()
            if record.status
            in {
                TaskStatus.RUNNING,
                TaskStatus.REPAIRING,
                TaskStatus.SUBMITTED,
                TaskStatus.REVIEWING,
            }
        ]
        errors = []
        if not self.plan.plan_version:
            errors.append("Register the plan before finalization")
        if missing:
            errors.append(f"Required tasks without accepted results: {', '.join(missing)}")
        if active:
            errors.append(f"Active tasks: {', '.join(active)}")
        return ValidationReport(not errors, tuple(errors))
