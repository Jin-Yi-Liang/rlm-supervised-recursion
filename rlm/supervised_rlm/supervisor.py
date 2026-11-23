"""Independent agent loop over frozen child evidence."""

from time import perf_counter_ns
from uuid import uuid4

from rlm.supervised_rlm import prompts
from rlm.supervised_rlm.client import AgentLoop
from rlm.supervised_rlm.metrics import ReviewEvent, elapsed
from rlm.supervised_rlm.repl import REPLError, REPLRuntime
from rlm.supervised_rlm.review_gate import ReviewGate
from rlm.supervised_rlm.schemas import (
    AgentRole,
    BudgetExhausted,
    ContractError,
    ReviewDecision,
    ReviewSubmissionAck,
    ReviewVerdict,
)
from rlm.supervised_rlm.state_broker import StateBroker
from rlm.supervised_rlm.storage import write_json


class SupervisorAgent:
    def __init__(self, loop: AgentLoop, gate: ReviewGate):
        self.loop, self.gate = loop, gate

    def review(self, broker: StateBroker) -> ReviewVerdict:
        started = perf_counter_ns()
        scope = broker.scope
        pending = None

        def submit_review(
            task_id: str,
            candidate_version: int,
            decision: str,
            finding: str,
            evidence_refs: list[str],
            requested_action: str = "",
        ) -> ReviewSubmissionAck:
            nonlocal pending
            if pending:
                raise ContractError("INVALID_ARGUMENT", "Only one review per Python execution")
            pending = self.gate.validate_verdict(
                broker,
                task_id,
                candidate_version,
                decision,
                finding,
                evidence_refs,
                requested_action,
            )
            return ReviewSubmissionAck(True, pending.review_id)

        def finish_cell(result, snapshot):
            nonlocal pending
            verdict, pending = pending, None
            return (verdict if result.success else None), ""

        tools = {
            name: getattr(broker, name)
            for name in (
                "list_child_variables",
                "inspect_child_variable",
                "read_artifact",
                "read_candidate",
                "get_child_trace",
                "read_child_execution",
                "get_task_contract",
                "get_global_task",
                "get_plan_context",
                "read_task_input",
            )
        } | {"submit_review": submit_review}
        brief = {
            "global_task": broker.get_global_task(),
            "task": broker.get_task_contract(),
            "candidate": broker.read_candidate(),
            "review_scope": {
                "task_id": scope.task_id,
                "candidate_version": scope.candidate_version,
            },
            "plan_context": broker.get_plan_context(),
        }
        session = self.loop.start(
            AgentRole.SUPERVISOR,
            scope.task_id,
            prompts.REPL + "\n" + prompts.REVIEW,
            brief,
            scope.candidate_version,
        )
        verdict = None
        failure = "Supervisor did not submit a valid decision within its review steps"
        try:
            with REPLRuntime(
                session.role,
                session.task_id,
                None,
                tools,
                self.loop.metrics,
                self.loop.config.execution_timeout,
            ) as repl:
                for step in range(1, self.loop.config.max_review_steps + 1):
                    verdict = self.loop.turn(session, repl, step, lambda: False, finish_cell)
                    if verdict:
                        break
        except (BudgetExhausted, REPLError) as error:
            failure = str(error)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            self.loop.metrics.event("supervisor_failure", {"scope": scope, "reason": failure})
        finally:
            broker.close()
            self.loop.metrics.record_phase(session.role, session.task_id, started)
        if verdict is None:
            verdict = ReviewVerdict(
                uuid4().hex,
                scope.run_id,
                scope.task_id,
                scope.candidate_version,
                ReviewDecision.REVIEW_INCOMPLETE,
                failure,
                (),
                "",
                "INCOMPLETE",
            )
        write_json(self.loop.folder / "reviews" / f"{verdict.review_id}.json", verdict)
        self.loop.metrics.record_review(
            ReviewEvent(
                scope.task_id,
                scope.candidate_version,
                verdict.review_id,
                verdict.decision,
                elapsed(started),
            )
        )
        return verdict
