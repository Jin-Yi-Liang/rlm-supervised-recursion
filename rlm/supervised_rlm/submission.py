"""Validate submissions and publish result receipts for the active plan."""

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

from rlm.supervised_rlm.metrics import CandidateEvent, MetricsCollector, elapsed
from rlm.supervised_rlm.plan_manager import OUTPUT_TYPES, PlanManager
from rlm.supervised_rlm.schemas import (
    CandidateManifest,
    ContractError,
    ResultReceipt,
    SubmissionAck,
    TaskSpec,
    TaskStatus,
)
from rlm.supervised_rlm.storage import strings, write_json


@dataclass(frozen=True)
class SubmissionIntent:
    ack: SubmissionAck
    result_var: str
    evidence_vars: tuple[str, ...]
    summary: str
    started: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.result_var, *self.evidence_vars)))


class SubmissionRegistry:
    def __init__(self, folder: Path, plan: PlanManager, metrics: MetricsCollector):
        self.folder, self.plan, self.metrics = folder, plan, metrics
        self.receipts: dict[str, ResultReceipt] = {}

    def prepare_submission(
        self,
        version: int,
        result_var: str,
        evidence_vars: list[str] | None = None,
        summary: str = "",
    ) -> SubmissionIntent:
        evidence = (
            ()
            if evidence_vars is None
            else strings(evidence_vars, "evidence_vars", allow_empty=True)
        )
        if not isinstance(result_var, str) or not result_var.isidentifier():
            raise ContractError("INVALID_ARGUMENT", "result_var must be a Python variable name")
        if any(not value.isidentifier() for value in evidence):
            raise ContractError("INVALID_ARGUMENT", "Evidence must contain variable names")
        if not isinstance(summary, str) or len(summary) > 1200:
            raise ContractError("INVALID_ARGUMENT", "Use a short submission summary")
        ack = SubmissionAck("PENDING", uuid4().hex, version)
        return SubmissionIntent(
            ack, result_var, tuple(dict.fromkeys(evidence)), summary, perf_counter_ns()
        )

    def record_submission(
        self, task_id: str, intent: SubmissionIntent, status: str, error: str | None = None
    ) -> None:
        self.metrics.record_candidate(
            CandidateEvent(
                task_id,
                intent.ack.candidate_version,
                intent.ack.submission_id,
                status,
                elapsed(intent.started),
                error,
            )
        )

    def finalize_submission(
        self, task: TaskSpec, intent: SubmissionIntent, namespace: list | None
    ) -> ResultReceipt:
        if not isinstance(namespace, list) or any(not isinstance(item, dict) for item in namespace):
            raise ContractError("SUBMISSION_MISSING", "No snapshot from the successful execution")
        entries = {item["name"]: item for item in namespace}
        for name in intent.names:
            if name not in entries:
                raise ContractError("SUBMISSION_MISSING", f"Unknown submitted variable {name!r}")
            if not entries[name]["available"]:
                raise ContractError("UNSUPPORTED_ARTIFACT", f"Cannot freeze {name!r}")
        self.validate_output(task, entries[intent.result_var]["value"])

        ack = intent.ack
        task_id = task.task_id
        with self.plan.lock:
            record = self.plan.records[task_id]
            if record.task_version != task.task_version:
                raise ContractError("VERSION_CONFLICT", "Submission refers to a stale task version")
            if record.is_accepted():
                previous = self.receipts[record.accepted_receipt_id]
                if previous.submission_id == ack.submission_id:
                    return previous
            if record.status != TaskStatus.RUNNING:
                raise ContractError("INVALID_TRANSITION", "Only a running task can submit a result")
            if ack.candidate_version != record.candidate_version + 1:
                raise ContractError("VERSION_CONFLICT", "Submission version is out of date")

            refs = {
                name: self.plan.artifacts.freeze(
                    entries[name]["value"],
                    name,
                    task_id,
                    ack.candidate_version,
                    ack.submission_id,
                    "child",
                )
                for name in intent.names
            }
            manifest = CandidateManifest(
                ack.submission_id,
                record.run_id,
                task_id,
                task.task_version,
                ack.candidate_version,
                refs[intent.result_var],
                tuple(refs[name] for name in intent.evidence_vars),
                ack.submission_id,
                intent.summary,
                "SUBMITTED",
            )
            write_json(
                self.folder / "submissions" / f"{ack.submission_id}.json", {"manifest": manifest}
            )
            submitted = self.plan.transition(
                task_id, TaskStatus.SUBMITTED, candidate_version=ack.candidate_version
            )
            self.metrics.record_task_state(submitted)
            self.record_submission(task_id, intent, "SUBMITTED")

            receipt = ResultReceipt(
                uuid4().hex,
                record.run_id,
                task_id,
                ack.candidate_version,
                ack.submission_id,
                task.task_version,
                manifest.result_ref,
            )
            updated = replace(
                submitted,
                status=TaskStatus.ACCEPTED,
                accepted_receipt_id=receipt.receipt_id,
                last_error=None,
            )
            # Publish the receipt and ledger update in one record, after its artifacts exist.
            write_json(
                self.folder / "commits" / f"{receipt.receipt_id}.json",
                {"receipt": receipt, "task": updated},
            )
            self.receipts[receipt.receipt_id] = receipt
            self.plan.records[task_id] = updated
            self.metrics.record_acceptance(task_id, ack.candidate_version)
            self.metrics.record_task_state(updated)
            return receipt

    def validate_output(self, task: TaskSpec, value: object) -> None:
        schema = task.output_schema
        target = OUTPUT_TYPES[schema.type_name]
        if schema.type_name != "any" and (
            not isinstance(value, target) or (type(value) is bool and schema.type_name != "boolean")
        ):
            raise ContractError("OUTPUT_SCHEMA_MISMATCH", f"Expected {schema.type_name}")
        if schema.required_fields and not set(schema.required_fields) <= value.keys():
            raise ContractError(
                "OUTPUT_SCHEMA_MISMATCH", f"Missing required fields: {schema.required_fields}"
            )
        expected = task.acceptance.expected_count
        if expected is not None and (
            not isinstance(value, (dict, list, str)) or len(value) != expected
        ):
            raise ContractError("OUTPUT_SCHEMA_MISMATCH", f"Expected {expected} items")

    def list_accepted_results(self) -> list[ResultReceipt]:
        with self.plan.lock:
            return [self.receipts[key] for key in self.plan.state().accepted_receipt_ids]

    def get_receipt(self, task_id: str) -> ResultReceipt | None:
        with self.plan.lock:
            record = self.plan.records.get(task_id)
            return (
                self.receipts[record.accepted_receipt_id]
                if record and record.is_accepted()
                else None
            )

    def read_accepted_result(
        self,
        receipt_id: str,
        path: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ):
        with self.plan.lock:
            if (
                not isinstance(receipt_id, str)
                or receipt_id not in self.plan.state().accepted_receipt_ids
            ):
                raise ContractError(
                    "PERMISSION_DENIED", "Reference does not identify an active accepted receipt"
                )
            return self.plan.artifacts.read(
                self.receipts[receipt_id].result_ref, path, offset, limit
            )
