"""Session entry point for the supervised RLM framework."""

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from rlm.supervised_rlm.client import AgentLoop, Backend
from rlm.supervised_rlm.metrics import MetricsCollector
from rlm.supervised_rlm.plan_manager import PlanManager
from rlm.supervised_rlm.review_gate import ReviewGate
from rlm.supervised_rlm.root_agent import RootAgent
from rlm.supervised_rlm.schemas import (
    BudgetExhausted,
    GlobalTaskSpec,
    RunConfig,
    RunHandle,
    RunOutcome,
    TaskStatus,
)
from rlm.supervised_rlm.storage import ArtifactStore, text, write_json
from rlm.supervised_rlm.subagent import SubAgentDispatcher


class SessionRuntime:
    """One run per output directory. The existing unsupervised RLM API stays independent."""

    def __init__(self, backend: Backend, folder: str | Path):
        self.backend = backend
        self.folder = Path(folder).resolve()
        self.handle: RunHandle | None = None
        self.outcome: RunOutcome | None = None

    def start_run(
        self, user_request: str, context: object, config: RunConfig | None = None
    ) -> RunHandle:
        if self.handle is not None:
            raise RuntimeError("This session already owns a run")
        config = config or RunConfig()
        objective = text(user_request, "user_request", 32_768)
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(self.folder.iterdir()):
            raise FileExistsError(f"Run directory already contains data: {self.folder}")
        run_id = uuid4().hex
        self.metrics = MetricsCollector(self.folder, run_id)
        artifacts = ArtifactStore(self.folder, run_id)
        context_ref = artifacts.freeze(context, "context", "root")
        request_ref = artifacts.freeze(user_request, "user_request", "root")
        self.global_task = GlobalTaskSpec(
            run_id, objective, request_ref.ref_id, (context_ref.ref_id,)
        )
        self.plan = PlanManager(self.global_task, self.folder, artifacts, config.max_tasks)
        self.loop = AgentLoop(self.folder, self.backend, config, self.metrics)
        self.gate = ReviewGate(self.folder, self.plan, self.metrics)
        self.dispatcher = SubAgentDispatcher(self.loop, self.plan, self.gate)
        self.root = RootAgent(self.loop, self.plan, self.gate, self.dispatcher)
        self.handle = RunHandle(run_id)
        write_json(self.folder / "run.json", {"global_task": self.global_task, "config": config})
        return self.handle

    def require_run(self, run_id: str) -> None:
        if self.handle is None or self.handle.run_id != run_id:
            raise ValueError("Unknown run ID")

    def get_global_task(self, run_id: str, include_full: bool = False) -> GlobalTaskSpec:
        self.require_run(run_id)
        return self.root.get_global_task(include_full)

    def get_run_state(self, run_id: str) -> dict:
        self.require_run(run_id)
        status = self.outcome.status if self.outcome else "RUNNING"
        return {
            "run_id": run_id,
            "status": status,
            "plan": self.plan.state(),
            "metrics": self.metrics.get_run_metrics(status, self.root.active_records()),
        }

    def finish_run(self, outcome: RunOutcome) -> RunOutcome:
        if self.outcome is None:
            self.outcome = outcome
            write_json(self.folder / "outcome.json", outcome)
            write_json(self.folder / "task-ledger.json", self.plan.records)
            write_json(self.folder / "accepted-receipts.json", self.gate.list_accepted_results())
            self.metrics.finish_run(outcome.status, self.root.active_records())
        return self.outcome

    def finalize_run(
        self,
        run_id: str,
        final_answer_ref: str | None = None,
        final_text: str | None = None,
    ) -> RunOutcome:
        self.require_run(run_id)
        if self.outcome is not None:
            return self.outcome
        outcome = self.root.prepare_final_answer(final_answer_ref, final_text)
        if outcome.final_answer_ref is None:
            outcome = replace(outcome, final_answer_ref=self.root.store_final(outcome.final_text))
        return self.finish_run(outcome)

    def abort_run(self, run_id: str, reason: str) -> RunOutcome:
        self.require_run(run_id)
        if self.outcome is not None:
            return self.outcome
        reason = text(reason, "reason")
        with self.plan.lock:
            for task_id, record in self.plan.records.items():
                if record.status == TaskStatus.PLANNED:
                    self.dispatcher.transition(task_id, TaskStatus.CANCELLED, last_error=reason)
        return self.finish_run(RunOutcome(run_id, "CANCELLED", reason=reason))

    def run(self, run_id: str) -> RunOutcome:
        self.require_run(run_id)
        if self.outcome is not None:
            return self.outcome
        try:
            outcome = self.root.run()
        except KeyboardInterrupt:
            self.abort_run(run_id, "Interrupted")
            raise
        except BudgetExhausted as error:
            outcome = RunOutcome(run_id, "INCOMPLETE", reason=str(error))
        except Exception as error:
            self.metrics.event(
                "run_failure", {"error_type": type(error).__name__, "message": str(error)}
            )
            outcome = RunOutcome(run_id, "FAILED", reason=f"{type(error).__name__}: {error}")
        return self.finish_run(outcome)

    def solve(self, question: str, context: object, config: RunConfig | None = None) -> RunOutcome:
        handle = self.start_run(question, context, config)
        return self.run(handle.run_id)
