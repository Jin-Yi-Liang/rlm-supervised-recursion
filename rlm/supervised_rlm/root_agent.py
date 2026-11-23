"""Root planning, progress queries and the final answer gate."""

from dataclasses import replace
from time import perf_counter_ns

from rlm.supervised_rlm import prompts
from rlm.supervised_rlm.client import AgentLoop
from rlm.supervised_rlm.plan_manager import PlanManager
from rlm.supervised_rlm.repl import REPLRuntime
from rlm.supervised_rlm.review_gate import ReviewGate
from rlm.supervised_rlm.schemas import AgentRole, ContractError, RunOutcome, ValidationReport
from rlm.supervised_rlm.storage import serialize, text
from rlm.supervised_rlm.subagent import SubAgentDispatcher


class RootAgent:
    def __init__(
        self,
        loop: AgentLoop,
        plan: PlanManager,
        gate: ReviewGate,
        dispatcher: SubAgentDispatcher,
    ):
        self.loop, self.plan, self.gate, self.dispatcher = loop, plan, gate, dispatcher
        self.task = plan.task

    def store_input(self, value: object, name: str = "input") -> str:
        if isinstance(value, str) and value in self.plan.artifacts.refs:
            raise ContractError(
                "INVALID_ARGUMENT",
                "That value is already an artifact handle. Pass the input_ref to add_task or "
                "update_task, or pass the actual data in context to store_input.",
            )
        return self.plan.artifacts.freeze(value, text(name, "name", 120), "root").ref_id

    def store_final(self, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ContractError("INVALID_ARGUMENT", "The final answer must be nonempty text")
        return self.plan.artifacts.freeze(value, "final_answer", "root", namespace="final").ref_id

    def get_task(self, task_id: str):
        if not isinstance(task_id, str) or task_id not in self.plan.records:
            raise ContractError("TASK_NOT_FOUND", "Unknown task ID")
        return self.plan.records[task_id]

    def get_plan_state(self, include_tasks: bool = True):
        if type(include_tasks) is not bool:
            raise ContractError("INVALID_ARGUMENT", "include_tasks must be bool")
        result = self.plan.state()
        if not include_tasks:
            return result
        return replace(result, tasks=tuple(self.active_records()))

    def list_pending_tasks(self):
        return [self.plan.records[task_id] for task_id in self.plan.state().pending_task_ids]

    def get_global_task(self, include_full: bool = False):
        if type(include_full) is not bool:
            raise ContractError("INVALID_ARGUMENT", "include_full must be bool")
        return (
            self.task if include_full else replace(self.task, objective=self.task.objective[:2500])
        )

    def validate_plan_completion(self) -> ValidationReport:
        report = self.plan.validate_completion()
        if not report.valid:
            return report
        for task_id in self.plan.state().completed_task_ids:
            receipt = self.gate.get_receipt(task_id)
            record = self.plan.records[task_id]
            if (
                receipt is None
                or receipt.run_id != self.task.run_id
                or receipt.result_ref.task_id != task_id
                or receipt.candidate_version != record.candidate_version
                or receipt.review_id != record.latest_review_id
            ):
                return ValidationReport(False, (f"Invalid result receipt for {task_id}",))
        return report

    def prepare_final_answer(
        self, final_answer_ref: str | None, final_text: str | None
    ) -> RunOutcome:
        if (final_answer_ref is None) == (final_text is None):
            raise ContractError("INVALID_ARGUMENT", "Supply exactly one final answer or reference")
        report = self.validate_plan_completion()
        if not report.valid:
            raise ContractError("PLAN_INCOMPLETE", "; ".join(report.errors))
        if final_answer_ref is not None:
            ref = self.plan.artifacts.require(final_answer_ref)
            if ref.source_namespace != "final":
                raise ContractError("PERMISSION_DENIED", "Use a reference from store_final")
            final_text = self.plan.artifacts.value(ref)
        elif not isinstance(final_text, str) or not final_text.strip():
            raise ContractError("INVALID_ARGUMENT", "The final answer must be nonempty text")
        return RunOutcome(self.task.run_id, "COMPLETED", final_answer_ref, final_text)

    def run(self) -> RunOutcome:
        started = perf_counter_ns()
        pending = None
        last_progress = self.plan.state()

        def submit_root_answer(final_answer_ref: str | None = None, final_text: str | None = None):
            nonlocal pending
            if pending is not None:
                raise ContractError("INVALID_ARGUMENT", "Submit one final answer per cell")
            pending = self.prepare_final_answer(final_answer_ref, final_text)

        def abort_run(reason: str):
            nonlocal pending
            if pending:
                raise ContractError("INVALID_ARGUMENT", "A run outcome is already pending")
            pending = RunOutcome(self.task.run_id, "INCOMPLETE", reason=text(reason, "reason"))

        def finish_cell(result, snapshot):
            nonlocal pending, last_progress
            outcome, pending = pending, None
            current = self.plan.state()
            notice = ""
            if current != last_progress:
                summary = {
                    "plan_version": current.plan_version,
                    "accepted": [
                        {"task_id": receipt.task_id, "receipt_id": receipt.receipt_id}
                        for receipt in self.gate.list_accepted_results()
                    ],
                    "pending_task_ids": current.pending_task_ids,
                    "open_issues": current.open_issues,
                }
                notice = "Authoritative progress after this cell: " + serialize(summary)
                last_progress = current
            if outcome and result.success and outcome.status == "COMPLETED":
                report = self.validate_plan_completion()
                if not report.valid:
                    return None, "; ".join(report.errors)
                if outcome.final_answer_ref is None:
                    outcome = replace(
                        outcome, final_answer_ref=self.store_final(outcome.final_text)
                    )
            return (outcome if result.success else None), notice

        tools = {
            name: getattr(self.plan, name)
            for name in (
                "create_plan",
                "add_task",
                "set_task_acceptance",
                "commit_plan",
                "copy_plan",
                "begin_plan_revision",
                "update_task",
                "remove_task",
                "discard_plan_draft",
            )
        } | {
            "get_plan_state": self.get_plan_state,
            "get_task": self.get_task,
            "list_pending_tasks": self.list_pending_tasks,
            "get_global_task": self.get_global_task,
            "store_input": self.store_input,
            "store_final": self.store_final,
            "rlm_query": self.dispatcher.rlm_query,
            "rlm_query_batched": self.dispatcher.dispatch_batch,
            "list_accepted_results": self.gate.list_accepted_results,
            "read_accepted_result": self.gate.read_accepted_result,
            "validate_plan_completion": self.validate_plan_completion,
            "get_run_metrics": lambda: self.loop.metrics.get_run_metrics(
                "RUNNING", self.active_records()
            ),
            "submit_root_answer": submit_root_answer,
            "finalize_run": submit_root_answer,
            "abort_run": abort_run,
        }
        ref = self.plan.artifacts.require(self.task.context_refs[0])
        session = self.loop.start(
            AgentRole.ROOT,
            "root",
            prompts.REPL + "\n" + prompts.ROOT,
            {
                "goal": self.task.objective,
                "run_id": self.task.run_id,
                "context_ref": ref.ref_id,
                "context_type": ref.value_type,
                "context_size_bytes": ref.size_bytes,
            },
        )
        try:
            with REPLRuntime(
                session.role,
                session.task_id,
                self.plan.artifacts.value(ref),
                tools,
                self.loop.metrics,
                self.loop.config.execution_timeout,
                context_ref=ref.ref_id,
            ) as repl:
                for step in range(1, self.loop.config.max_root_steps + 1):
                    outcome = self.loop.turn(session, repl, step, lambda: False, finish_cell)
                    if outcome:
                        return outcome
            return RunOutcome(self.task.run_id, "INCOMPLETE", reason="Root step budget exhausted")
        finally:
            self.loop.metrics.record_phase(session.role, session.task_id, started)

    def active_records(self):
        return [self.plan.records[task.task_id] for task in self.plan.plan.tasks]
