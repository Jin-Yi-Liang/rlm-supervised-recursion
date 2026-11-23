"""Collect measured requests, tool attempts and review decisions."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter_ns
from uuid import uuid4

from rlm.supervised_rlm.schemas import AgentRole, ReviewDecision, TaskRecord
from rlm.supervised_rlm.storage import serialize, write_json


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    source: str = "UNAVAILABLE"


@dataclass(frozen=True)
class LLMCallRecord:
    request_id: str
    task_id: str
    agent_role: AgentRole
    model: str
    duration_ms: float
    success: bool
    finish_reason: str | None
    usage: TokenUsage
    candidate_version: int | None
    reasoning_present: bool = False


@dataclass(frozen=True)
class ToolCallRecord:
    tool_call_id: str
    task_id: str
    agent_role: AgentRole
    tool_name: str
    protocol_valid: bool
    tool_known: bool
    arguments_valid: bool
    executed: bool
    execution_success: bool | None
    duration_ms: float
    error_code: str | None
    semantic_correct: bool | None = None


@dataclass(frozen=True)
class CandidateEvent:
    task_id: str
    candidate_version: int
    submission_id: str
    status: str
    duration_ms: float


@dataclass(frozen=True)
class ReviewEvent:
    task_id: str
    candidate_version: int
    review_id: str
    decision: ReviewDecision
    duration_ms: float


@dataclass(frozen=True)
class RoleMetricsSnapshot:
    llm_calls: int
    llm_failures: int
    input_tokens: int | None
    output_tokens: int | None
    token_usage_coverage: float | None
    llm_duration_ms: float
    agent_duration_ms: float


@dataclass(frozen=True)
class RunMetricsSnapshot:
    run_id: str
    run_status: str
    wall_time_ms: float
    llm_calls: int
    llm_failures: int
    input_tokens: int | None
    output_tokens: int | None
    token_usage_coverage: float | None
    tool_calls: int
    tool_protocol_valid: int
    tool_arguments_valid: int
    tool_executed: int
    tool_execution_success: int
    tool_semantic_evaluated: int
    tool_semantic_correct: int
    candidates_submitted: int
    candidates_aborted: int
    candidates_rejected: int
    candidates_accepted: int
    plan_issues: int
    review_incomplete: int
    repair_attempts: int
    repaired_tasks_accepted: int
    repair_success_rate: float | None
    completed_tasks: int
    total_tasks: int
    roles: dict[str, RoleMetricsSnapshot]


def elapsed(started: int) -> float:
    return (perf_counter_ns() - started) / 1_000_000


class MetricsCollector:
    def __init__(self, folder: Path, run_id: str):
        self.folder = folder
        self.run_id = run_id
        self.started = perf_counter_ns()
        self.finished_ms: float | None = None
        self.llm: list[LLMCallRecord] = []
        self.tools: list[ToolCallRecord] = []
        self.candidates: dict[tuple[str, int, str], CandidateEvent] = {}
        self.reviews: dict[str, ReviewEvent] = {}
        self.repairs: set[tuple[str, str]] = set()
        self.accepted: set[tuple[str, int]] = set()
        self.phase_times: Counter = Counter()
        self.seen: set[str] = set()
        self.lock = RLock()

    def event(self, kind: str, fields: object, identity: str | None = None) -> None:
        with self.lock:
            identity = identity or uuid4().hex
            if identity in self.seen:
                return
            with (self.folder / "events.jsonl").open("a", encoding="utf-8") as stream:
                print(
                    serialize(
                        {
                            "event_id": identity,
                            "run_id": self.run_id,
                            "event": kind,
                            "utc": datetime.now(UTC).isoformat(),
                            "data": fields,
                        }
                    ),
                    file=stream,
                )
            self.seen.add(identity)

    def record_llm_call(self, record: LLMCallRecord) -> None:
        with self.lock:
            if record.request_id not in self.seen:
                self.event("llm", record, record.request_id)
                self.llm.append(record)

    def record_tool_call(self, record: ToolCallRecord) -> None:
        with self.lock:
            if record.tool_call_id not in self.seen:
                self.event("tool", record, record.tool_call_id)
                self.tools.append(record)

    def record_candidate(self, record: CandidateEvent) -> None:
        with self.lock:
            key = (record.submission_id, record.candidate_version, record.status)
            if key not in self.candidates:
                self.event("candidate", record)
                self.candidates[key] = record

    def record_review(self, record: ReviewEvent) -> None:
        with self.lock:
            if record.review_id not in self.reviews:
                self.event("review", record, record.review_id)
                self.reviews[record.review_id] = record

    def record_task_state(self, record: TaskRecord) -> None:
        self.event("task_state", record)

    def record_repair(self, task_id: str, review_id: str, candidate_version: int) -> None:
        with self.lock:
            key = (task_id, review_id)
            if key not in self.repairs:
                self.event(
                    "repair_started",
                    {
                        "task_id": task_id,
                        "review_id": review_id,
                        "rejected_version": candidate_version,
                    },
                )
                self.repairs.add(key)

    def record_acceptance(self, task_id: str, candidate_version: int) -> None:
        with self.lock:
            self.accepted.add((task_id, candidate_version))

    def record_phase(self, role: AgentRole, task_id: str, started: int) -> None:
        with self.lock:
            duration = elapsed(started)
            self.phase_times[role] += duration
            self.event(
                "agent_phase_finished", {"role": role, "task_id": task_id, "duration_ms": duration}
            )

    def role_summary(self, calls: list[LLMCallRecord], duration: float = 0) -> RoleMetricsSnapshot:
        known_input = [
            call.usage.input_tokens for call in calls if call.usage.input_tokens is not None
        ]
        known_output = [
            call.usage.output_tokens for call in calls if call.usage.output_tokens is not None
        ]
        measured = sum(
            call.usage.input_tokens is not None and call.usage.output_tokens is not None
            for call in calls
        )
        return RoleMetricsSnapshot(
            len(calls),
            sum(not call.success for call in calls),
            sum(known_input) if known_input else None,
            sum(known_output) if known_output else None,
            measured / len(calls) if calls else None,
            sum(call.duration_ms for call in calls),
            duration,
        )

    def get_role_metrics(self, role: AgentRole) -> RoleMetricsSnapshot:
        with self.lock:
            return self.role_summary(
                [item for item in self.llm if item.agent_role == role], self.phase_times[role]
            )

    def get_task_metrics(self, task_id: str) -> dict:
        with self.lock:
            return {
                "task_id": task_id,
                "roles": {
                    role: self.role_summary(
                        [
                            item
                            for item in self.llm
                            if item.agent_role == role and item.task_id == task_id
                        ]
                    )
                    for role in AgentRole
                },
            }

    def get_run_metrics(self, status: str, records: list[TaskRecord]) -> RunMetricsSnapshot:
        with self.lock:
            usage = self.role_summary(self.llm)
            decisions = Counter(review.decision for review in self.reviews.values())
            candidates = Counter(item.status for item in self.candidates.values())
            revised = {
                review.task_id
                for review in self.reviews.values()
                if review.decision == ReviewDecision.REVISE
            }
            repaired = revised & {task_id for task_id, _ in self.accepted}
            semantic = [
                item.semantic_correct for item in self.tools if item.semantic_correct is not None
            ]
            return RunMetricsSnapshot(
                self.run_id,
                status,
                self.finished_ms if self.finished_ms is not None else elapsed(self.started),
                usage.llm_calls,
                usage.llm_failures,
                usage.input_tokens,
                usage.output_tokens,
                usage.token_usage_coverage,
                len(self.tools),
                sum(item.protocol_valid for item in self.tools),
                sum(item.arguments_valid for item in self.tools),
                sum(item.executed for item in self.tools),
                sum(item.execution_success is True for item in self.tools),
                len(semantic),
                sum(semantic),
                candidates["SUBMITTED"],
                candidates["ABORTED"],
                decisions[ReviewDecision.REVISE],
                len(self.accepted),
                decisions[ReviewDecision.PLAN_ISSUE],
                decisions[ReviewDecision.REVIEW_INCOMPLETE],
                len(self.repairs),
                len(repaired),
                len(repaired) / len(revised) if revised and self.finished_ms is not None else None,
                sum(record.is_accepted() for record in records),
                len(records),
                {role: self.get_role_metrics(role) for role in AgentRole},
            )

    def finish_run(self, status: str, records: list[TaskRecord]) -> RunMetricsSnapshot:
        self.finished_ms = elapsed(self.started)
        snapshot = self.get_run_metrics(status, records)
        self.export_run(snapshot, self.folder / "metrics.json")
        return snapshot

    def export_run(self, snapshot: RunMetricsSnapshot, path: Path) -> None:
        values = asdict(snapshot)
        values["tool_semantic_accuracy"] = (
            snapshot.tool_semantic_correct / snapshot.tool_semantic_evaluated
            if snapshot.tool_semantic_evaluated
            else None
        )
        values["tool_protocol_rate"] = (
            snapshot.tool_protocol_valid / snapshot.tool_calls if snapshot.tool_calls else None
        )
        values["tool_argument_rate"] = (
            snapshot.tool_arguments_valid / snapshot.tool_calls if snapshot.tool_calls else None
        )
        values["tool_execution_rate"] = (
            snapshot.tool_execution_success / snapshot.tool_executed
            if snapshot.tool_executed
            else None
        )
        verdicts = snapshot.candidates_rejected + snapshot.candidates_accepted
        values["rejection_rate"] = snapshot.candidates_rejected / verdicts if verdicts else None
        write_json(path, values)
