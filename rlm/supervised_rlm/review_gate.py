"""Enforce review scope and publish one atomic acceptance record."""

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from rlm.supervised_rlm.metrics import MetricsCollector
from rlm.supervised_rlm.plan_manager import PlanManager
from rlm.supervised_rlm.schemas import (
    ContractError,
    ResultReceipt,
    ReviewDecision,
    ReviewVerdict,
    TaskStatus,
)
from rlm.supervised_rlm.state_broker import StateBroker
from rlm.supervised_rlm.storage import strings, text, write_json
from rlm.supervised_rlm.submission import CandidateSnapshot


class ReviewGate:
    def __init__(self, folder: Path, plan: PlanManager, metrics: MetricsCollector):
        self.folder, self.plan, self.metrics = folder, plan, metrics
        self.receipts: dict[str, ResultReceipt] = {}
        self.commits: dict[tuple[str, int], ResultReceipt] = {}

    def validate_verdict(
        self,
        broker: StateBroker,
        task_id: str,
        candidate_version: int,
        decision: str,
        finding: str,
        evidence_refs: list[str],
        requested_action: str = "",
    ) -> ReviewVerdict:
        broker.require_active()
        scope = broker.scope
        if (
            task_id != scope.task_id
            or type(candidate_version) is not int
            or candidate_version != scope.candidate_version
        ):
            raise ContractError(
                "VERSION_CONFLICT", "Review must address the current task and candidate version"
            )
        try:
            choice = ReviewDecision(decision)
        except (TypeError, ValueError) as error:
            raise ContractError("INVALID_ARGUMENT", "Invalid review decision") from error
        finding = text(finding, "finding", 2400)
        if not isinstance(requested_action, str) or len(requested_action) > 2400:
            raise ContractError("INVALID_ARGUMENT", "Invalid requested action")
        refs = strings(evidence_refs, "evidence_refs", allow_empty=True)
        if not set(refs) <= broker.issued:
            raise ContractError(
                "PERMISSION_DENIED", "Review must cite evidence obtained by reading this snapshot"
            )
        if choice == ReviewDecision.ACCEPT and not refs:
            raise ContractError("MISSING_EVIDENCE", "ACCEPT requires read evidence")
        if (
            choice in (ReviewDecision.REVISE, ReviewDecision.PLAN_ISSUE)
            and not requested_action.strip()
        ):
            raise ContractError("INVALID_ARGUMENT", "Provide a concrete requested action")
        return ReviewVerdict(
            uuid4().hex,
            scope.run_id,
            task_id,
            candidate_version,
            choice,
            finding,
            refs,
            requested_action,
        )

    def commit_accepted_candidate(
        self, snapshot: CandidateSnapshot, verdict: ReviewVerdict
    ) -> ResultReceipt:
        manifest = snapshot.manifest
        key = (manifest.task_id, manifest.candidate_version)
        with self.plan.lock:
            record = self.plan.records[manifest.task_id]
            if (
                verdict.run_id != manifest.run_id
                or verdict.task_id != manifest.task_id
                or verdict.candidate_version != manifest.candidate_version
                or verdict.decision != ReviewDecision.ACCEPT
                or not verdict.evidence_refs
                or record.task_version != manifest.task_version
                or record.candidate_version != manifest.candidate_version
            ):
                raise ContractError(
                    "VERSION_CONFLICT", "Acceptance does not match the current submitted candidate"
                )
            if key in self.commits:
                receipt = self.commits[key]
                if receipt.review_id == verdict.review_id:
                    return receipt
                raise ContractError("VERSION_CONFLICT", "Candidate already has an accepted review")
            if record.status != TaskStatus.REVIEWING:
                raise ContractError(
                    "INVALID_TRANSITION", "Only a candidate under review can be accepted"
                )
            receipt = ResultReceipt(
                uuid4().hex,
                manifest.run_id,
                manifest.task_id,
                manifest.candidate_version,
                verdict.review_id,
                manifest.result_ref,
            )
            updated = replace(
                record,
                status=TaskStatus.ACCEPTED,
                accepted_receipt_id=receipt.receipt_id,
                latest_review_id=verdict.review_id,
                last_error=None,
            )
            # The result payload is already immutable. Publish the receipt and ledger record together.
            write_json(
                self.folder / "commits" / f"{receipt.receipt_id}.json",
                {"receipt": receipt, "task": updated},
            )
            self.receipts[receipt.receipt_id] = self.commits[key] = receipt
            self.plan.records[manifest.task_id] = updated
            self.metrics.record_acceptance(*key)
            self.metrics.record_task_state(updated)
            return receipt

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
