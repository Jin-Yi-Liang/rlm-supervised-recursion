"""Candidate-version-scoped evidence reads."""

from dataclasses import replace
from uuid import uuid4

from rlm.supervised_rlm.plan_manager import PlanManager
from rlm.supervised_rlm.schemas import ContractError, ReviewScope, VariableCatalog
from rlm.supervised_rlm.storage import ArtifactStore, make_view, page
from rlm.supervised_rlm.submission import CandidateSnapshot


class StateBroker:
    def __init__(self, artifacts: ArtifactStore, plan: PlanManager, snapshot: CandidateSnapshot):
        self.artifacts = artifacts
        self.plan = plan
        self.snapshot = snapshot
        manifest = snapshot.manifest
        self.scope = ReviewScope(
            manifest.run_id,
            manifest.task_id,
            manifest.candidate_version,
            uuid4().hex,
            manifest.snapshot_id,
        )
        self.active = True
        self.issued: set[str] = set()
        self.refs = {ref.ref_id: ref for ref in snapshot.refs}
        self.variables = {item.name: item for item in snapshot.variables}
        self.trace_ref = uuid4().hex
        self.input_ref = uuid4().hex

    def require_active(self):
        if not self.active:
            raise ContractError("PERMISSION_DENIED", "Review scope is closed")

    def close(self):
        self.active = False

    def list_child_variables(self, include_all: bool = True) -> VariableCatalog:
        self.require_active()
        if type(include_all) is not bool:
            raise ContractError("INVALID_ARGUMENT", "include_all must be bool")
        catalog = self.snapshot.variables
        if not include_all:
            manifest = self.snapshot.manifest
            refs = {ref.ref_id for ref in (manifest.result_ref, *manifest.evidence_refs)}
            catalog = tuple(item for item in catalog if item.evidence_ref in refs)
        return VariableCatalog(catalog, self.scope.snapshot_id)

    def inspect_child_variable(
        self,
        name: str,
        path: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ):
        self.require_active()
        item = self.variables.get(name) if isinstance(name, str) else None
        if not item:
            raise ContractError("ARTIFACT_NOT_FOUND", "Variable does not exist in this snapshot")
        if item.evidence_ref is None:
            raise ContractError("UNSUPPORTED_ARTIFACT", "Variable could not be frozen")
        return self.read_artifact(item.evidence_ref, offset, limit, path)

    def read_artifact(
        self,
        ref_id: str,
        offset: int = 0,
        limit: int = 1000,
        path: str | None = None,
    ):
        self.require_active()
        if not isinstance(ref_id, str) or ref_id not in self.refs:
            raise ContractError(
                "PERMISSION_DENIED", "Reference belongs to another scope or is unavailable"
            )
        result = self.artifacts.read(self.refs[ref_id], path, offset, limit)
        self.issued.add(ref_id)
        return result

    def read_candidate(self):
        self.require_active()
        return self.snapshot.manifest

    def get_child_trace(self, start: int = 0, limit: int = 50):
        self.require_active()
        page(start, limit, maximum=50)
        trace = [
            {
                "execution_id": record.execution_id,
                "step": record.step,
                "code": record.code[:4000],
                "result": replace(
                    record.result,
                    stdout=record.result.stdout[:1000],
                    stderr=record.result.stderr[:1000],
                ),
                "preview_truncated": len(record.code) > 4000
                or max(len(record.result.stdout), len(record.result.stderr)) > 1000,
            }
            for record in self.snapshot.trace
        ]
        result = make_view(trace, "child_trace", self.trace_ref, None, start, limit)
        if result.value:
            self.issued.add(self.trace_ref)
        return result

    def read_child_execution(
        self, execution_id: str, field: str, offset: int = 0, limit: int = 1000
    ):
        self.require_active()
        if field not in ("code", "stdout", "stderr", "error_message"):
            raise ContractError("INVALID_ARGUMENT", "Select code, stdout, stderr or error_message")
        for record in self.snapshot.trace:
            if record.execution_id == execution_id:
                value = record.code if field == "code" else getattr(record.result, field)
                result = make_view(value, field, self.trace_ref, None, offset, limit)
                self.issued.add(self.trace_ref)
                return result
        raise ContractError("ARTIFACT_NOT_FOUND", "Execution is not in this candidate's trace")

    def get_task_contract(self):
        self.require_active()
        return self.plan.spec(self.scope.task_id)

    def get_global_task(self, include_full: bool = False):
        self.require_active()
        if type(include_full) is not bool:
            raise ContractError("INVALID_ARGUMENT", "include_full must be bool")
        return (
            self.plan.task
            if include_full
            else replace(self.plan.task, objective=self.plan.task.objective[:2500])
        )

    def get_plan_context(self):
        self.require_active()
        return self.plan.state()

    def read_task_input(self, offset: int = 0, limit: int = 1000, path: str | None = None):
        self.require_active()
        ref = self.artifacts.require(self.get_task_contract().input_refs[0])
        result = replace(self.artifacts.read(ref, path, offset, limit), evidence_ref=self.input_ref)
        self.issued.add(self.input_ref)
        return result
