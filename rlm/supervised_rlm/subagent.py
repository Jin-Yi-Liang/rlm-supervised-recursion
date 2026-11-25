"""Dispatch registered tasks with persistent Child sessions."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns

from rlm.supervised_rlm import prompts
from rlm.supervised_rlm.client import AgentLoop
from rlm.supervised_rlm.plan_manager import PlanManager
from rlm.supervised_rlm.repl import REPLRuntime
from rlm.supervised_rlm.schemas import (
    AgentRole,
    ContractError,
    TaskDispatchOutcome,
    TaskSpec,
    TaskStatus,
)
from rlm.supervised_rlm.submission import SubmissionRegistry


class SubAgentDispatcher:
    def __init__(self, loop: AgentLoop, plan: PlanManager, submissions: SubmissionRegistry):
        self.loop, self.plan, self.submissions = loop, plan, submissions

    def transition(self, task_id: str, status: TaskStatus, **changes):
        record = self.plan.transition(task_id, status, **changes)
        self.loop.metrics.record_task_state(record)
        return record

    def dispatch_registered(self, task_id: str) -> TaskDispatchOutcome:
        if not isinstance(task_id, str):
            raise ContractError("INVALID_ARGUMENT", "Task ID must be a string")
        with self.plan.lock:
            task = self.plan.spec(task_id)
            record = self.plan.records[task_id]
            if record.is_accepted():
                return TaskDispatchOutcome(
                    task_id, record.status, self.submissions.get_receipt(task_id), None
                )
            if record.status != TaskStatus.PLANNED:
                return TaskDispatchOutcome(
                    task_id, record.status, None, record.last_error or "Task is not dispatchable"
                )
            self.transition(task_id, TaskStatus.RUNNING, attempt_count=record.attempt_count + 1)

        try:
            return self.run_child(task, record.candidate_version)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            status = self.plan.records[task_id].status
            if status not in (TaskStatus.RUNNING, TaskStatus.SUBMITTED):
                raise
            self.transition(task_id, TaskStatus.FAILED, last_error=message)
            return TaskDispatchOutcome(task_id, TaskStatus.FAILED, None, message)

    def dispatch_batch(self, task_ids: list[str]) -> list[TaskDispatchOutcome]:
        if not isinstance(task_ids, list) or any(not isinstance(task, str) for task in task_ids):
            raise ContractError("INVALID_ARGUMENT", "Provide a list of task IDs")
        if (
            not task_ids
            or len(task_ids) != len(set(task_ids))
            or len(task_ids) > self.loop.config.max_tasks
        ):
            raise ContractError(
                "INVALID_ARGUMENT",
                "Batch requires distinct registered task IDs within the task limit",
            )
        with ThreadPoolExecutor(max_workers=self.loop.config.max_parallel_children) as executor:
            return list(executor.map(self.batch_item, task_ids))

    def batch_item(self, task_id: str) -> TaskDispatchOutcome:
        try:
            return self.dispatch_registered(task_id)
        except ContractError as error:
            return TaskDispatchOutcome(task_id, TaskStatus.FAILED, None, str(error))

    def rlm_query(self, task_id: str):
        outcome = self.dispatch_registered(task_id)
        if not outcome.receipt:
            raise ContractError(
                "TASK_NOT_DISPATCHABLE", f"{task_id}: {outcome.status}: {outcome.reason}"
            )
        return outcome.receipt

    def run_child(self, task: TaskSpec, version: int) -> TaskDispatchOutcome:
        started = perf_counter_ns()
        task_id = task.task_id
        data = self.plan.artifacts.value(self.plan.artifacts.require(task.input_refs[0]))
        session = self.loop.start(
            AgentRole.CHILD,
            task_id,
            prompts.REPL + "\n" + prompts.CHILD,
            {
                "task": task,
                "global_objective": self.plan.task.objective[:2500],
                "input_type": type(data).__name__,
                "input_length": len(data) if isinstance(data, (str, list, dict)) else None,
            },
        )
        pending = None

        def submit_candidate(
            result_var: str, evidence_vars: list[str] | None = None, summary: str = ""
        ):
            nonlocal pending
            if pending is not None:
                raise ContractError("INVALID_ARGUMENT", "Only one submission per cell")
            pending = self.submissions.prepare_submission(
                version + 1, result_var, evidence_vars, summary
            )
            return pending.ack

        def finish_cell(result, namespace):
            nonlocal pending
            intent, pending = pending, None
            if not intent:
                return None, ""
            if not result.success:
                self.submissions.record_submission(task_id, intent, "ABORTED", result.error_message)
                return None, "Submission aborted because the Python cell failed."
            try:
                return self.submissions.finalize_submission(task, intent, namespace), ""
            except ContractError as error:
                self.submissions.record_submission(task_id, intent, "INVALID", str(error))
                return None, str(error)

        try:
            with REPLRuntime(
                session.role,
                task_id,
                data,
                {"submit_candidate": submit_candidate},
                self.loop.metrics,
                self.loop.config.execution_timeout,
                max_tool_calls=self.loop.config.max_cell_tool_calls,
            ) as repl:
                session.candidate_version = version + 1
                for step in range(1, self.loop.config.max_child_steps + 1):
                    receipt = self.loop.turn(
                        session, repl, step, lambda: pending.names if pending else (), finish_cell
                    )
                    if receipt is not None:
                        return TaskDispatchOutcome(task_id, TaskStatus.ACCEPTED, receipt, None)
            detail = "Child step budget exhausted"
            self.loop.metrics.record_step_limit(session.role, task_id)
            self.transition(task_id, TaskStatus.FAILED, last_error=detail)
            return TaskDispatchOutcome(task_id, TaskStatus.FAILED, None, detail)
        finally:
            if pending:
                self.submissions.record_submission(task_id, pending, "ABORTED")
            self.loop.metrics.record_phase(session.role, task_id, started)
