"""Typed contracts owned by the supervised runtime."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar


class TaskStatus(StrEnum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    SUBMITTED = "SUBMITTED"
    REVIEWING = "REVIEWING"
    REPAIRING = "REPAIRING"
    ACCEPTED = "ACCEPTED"
    PLAN_ISSUE = "PLAN_ISSUE"
    REVIEW_INCOMPLETE = "REVIEW_INCOMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"


class ReviewDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    PLAN_ISSUE = "PLAN_ISSUE"
    REVIEW_INCOMPLETE = "REVIEW_INCOMPLETE"


class AgentRole(StrEnum):
    ROOT = "ROOT"
    CHILD = "CHILD"
    SUPERVISOR = "SUPERVISOR"


class ErrorCode(StrEnum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INVALID_TOOL_PROTOCOL = "INVALID_TOOL_PROTOCOL"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    PLAN_INVALID = "PLAN_INVALID"
    PLAN_NOT_FOUND = "PLAN_NOT_FOUND"
    PLAN_INCOMPLETE = "PLAN_INCOMPLETE"
    DUPLICATE_TASK = "DUPLICATE_TASK"
    TASK_BUSY = "TASK_BUSY"
    TASK_NOT_DISPATCHABLE = "TASK_NOT_DISPATCHABLE"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    SUBMISSION_MISSING = "SUBMISSION_MISSING"
    ARTIFACT_TOO_LARGE = "ARTIFACT_TOO_LARGE"
    UNSUPPORTED_ARTIFACT = "UNSUPPORTED_ARTIFACT"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    OUTPUT_SCHEMA_MISMATCH = "OUTPUT_SCHEMA_MISMATCH"


T = TypeVar("T")


@dataclass(frozen=True)
class ToolResponse(Generic[T]):
    ok: bool
    data: T | None = None
    code: ErrorCode | None = None
    message: str = ""
    retryable: bool = False


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = ErrorCode(code)
        super().__init__(f"{code}: {message}")


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class RunConfig:
    max_root_steps: int = 24
    max_child_steps: int = 20
    max_review_steps: int = 10
    max_revisions: int = 2
    max_tasks: int = 64
    max_model_calls: int = 480
    max_tokens: int = 2_000_000
    context_window: int = 32768
    max_output_tokens: int = 4096
    max_parallel_children: int = 4
    execution_timeout: float = 45
    deadline: float | None = None

    def __post_init__(self):
        for name in (
            "max_root_steps",
            "max_child_steps",
            "max_review_steps",
            "max_tasks",
            "max_model_calls",
            "max_tokens",
            "context_window",
            "max_output_tokens",
            "max_parallel_children",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_revisions) is not int or self.max_revisions < 0:
            raise ValueError("max_revisions must be a nonnegative integer")
        if not 0 < self.max_output_tokens < self.context_window or self.execution_timeout <= 0:
            raise ValueError("Invalid model context or execution limits")


@dataclass(frozen=True)
class FinalAcceptanceSpec:
    required_task_policy: str = "all_current_required_tasks"
    checks: tuple[str, ...] = ("Cover the user objective and reconcile accepted results",)


@dataclass(frozen=True)
class GlobalTaskSpec:
    run_id: str
    objective: str
    user_request_ref: str
    context_refs: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    final_acceptance: FinalAcceptanceSpec = FinalAcceptanceSpec()


@dataclass(frozen=True)
class AcceptanceSpec:
    checks: tuple[str, ...]
    expected_count: int | None = None


@dataclass(frozen=True)
class OutputSpec:
    type_name: str
    required_fields: tuple[str, ...] = ()


@dataclass
class TaskDraft:
    task_id: str
    instruction: str
    scope: str
    input_refs: tuple[str, ...]
    required: bool
    acceptance: AcceptanceSpec | None = None
    output_schema: OutputSpec | None = None


@dataclass
class PlanDraft:
    draft_id: str
    run_id: str
    goal: str
    expected_version: int
    reason: str
    tasks: dict[str, TaskDraft] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    scope: str
    input_refs: tuple[str, ...]
    acceptance: AcceptanceSpec
    output_schema: OutputSpec
    required: bool
    task_version: int


@dataclass(frozen=True)
class PlanSpec:
    run_id: str
    plan_version: int
    goal: str
    tasks: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class PlanReceipt:
    run_id: str
    plan_version: int
    plan_id: str


@dataclass(frozen=True)
class PlanChangeRecord:
    change_id: str
    from_version: int
    to_version: int
    reason: str
    affected_task_ids: tuple[str, ...]
    superseded_receipts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TaskRecord:
    run_id: str
    task_id: str
    plan_version: int
    task_version: int
    status: TaskStatus = TaskStatus.PLANNED
    attempt_count: int = 0
    candidate_version: int = 0
    latest_review_id: str | None = None
    accepted_receipt_id: str | None = None
    last_error: str | None = None
    required: bool = True

    def is_accepted(self) -> bool:
        return self.status == TaskStatus.ACCEPTED


@dataclass(frozen=True)
class PlanState:
    run_id: str
    plan_version: int
    goal: str
    completed_task_ids: tuple[str, ...]
    pending_task_ids: tuple[str, ...]
    accepted_receipt_ids: tuple[str, ...]
    open_issues: tuple[str, ...]
    tasks: tuple[TaskRecord, ...] = ()


@dataclass(frozen=True)
class ArtifactRef:
    ref_id: str
    run_id: str
    task_id: str
    candidate_version: int
    source_namespace: str
    variable_name: str
    value_type: str
    size_bytes: int
    content_hash: str
    encoding: str
    snapshot_id: str


@dataclass(frozen=True)
class ArtifactView:
    name: str
    path: str
    offset: int
    limit: int
    truncated: bool
    next_offset: int | None
    value: object
    evidence_ref: str


@dataclass(frozen=True)
class VariableInfo:
    name: str
    type_name: str
    size_bytes: int
    available: bool
    snapshot_id: str
    evidence_ref: str | None


@dataclass(frozen=True)
class VariableCatalog:
    entries: tuple[VariableInfo, ...]
    snapshot_id: str

    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.entries)


@dataclass(frozen=True)
class SubmissionAck:
    status: str
    submission_id: str
    candidate_version: int


@dataclass(frozen=True)
class CandidateManifest:
    submission_id: str
    run_id: str
    task_id: str
    task_version: int
    candidate_version: int
    result_ref: ArtifactRef
    evidence_refs: tuple[ArtifactRef, ...]
    snapshot_id: str
    summary: str
    status: str
    idempotency_key: str


@dataclass(frozen=True)
class ReviewScope:
    run_id: str
    task_id: str
    candidate_version: int
    supervisor_id: str
    snapshot_id: str


@dataclass(frozen=True)
class ReviewVerdict:
    review_id: str
    run_id: str
    task_id: str
    candidate_version: int
    decision: ReviewDecision
    finding: str
    evidence_refs: tuple[str, ...]
    requested_action: str
    review_status: str = "VALIDATED"


@dataclass(frozen=True)
class ReviewSubmissionAck:
    accepted_by_gate: bool
    review_id: str


@dataclass(frozen=True)
class ResultReceipt:
    receipt_id: str
    run_id: str
    task_id: str
    candidate_version: int
    review_id: str
    result_ref: ArtifactRef
    status: str = "ACCEPTED"

    def is_accepted(self) -> bool:
        return self.status == "ACCEPTED"


@dataclass(frozen=True)
class TaskDispatchOutcome:
    task_id: str
    status: TaskStatus
    receipt: ResultReceipt | None
    reason: str | None


@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    success: bool
    stdout: str
    stderr: str
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True)
class TraceRecord:
    execution_id: str
    step: int
    code: str
    result: ExecutionResult


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunHandle:
    run_id: str


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: str
    final_answer_ref: str | None = None
    final_text: str | None = None
    reason: str | None = None
    metrics_ref: str = "metrics.json"
