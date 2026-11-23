"""Stage submission intents and publish a snapshot after a successful cell."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

from rlm.supervised_rlm.metrics import CandidateEvent, MetricsCollector, elapsed
from rlm.supervised_rlm.plan_manager import OUTPUT_TYPES
from rlm.supervised_rlm.schemas import (
    ArtifactRef,
    CandidateManifest,
    ContractError,
    SubmissionAck,
    TaskSpec,
    TraceRecord,
    VariableInfo,
)
from rlm.supervised_rlm.storage import ArtifactStore, strings, write_json


@dataclass(frozen=True)
class SubmissionIntent:
    ack: SubmissionAck
    result_var: str
    evidence_vars: tuple[str, ...]
    summary: str
    started: int


@dataclass(frozen=True)
class CandidateSnapshot:
    manifest: CandidateManifest
    variables: tuple[VariableInfo, ...]
    refs: tuple[ArtifactRef, ...]
    trace: tuple[TraceRecord, ...]


class SubmissionRegistry:
    def __init__(
        self,
        folder: Path,
        artifacts: ArtifactStore,
        metrics: MetricsCollector,
    ):
        self.folder, self.artifacts, self.metrics = folder, artifacts, metrics

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

    def abort_submission(self, task_id: str, intent: SubmissionIntent) -> None:
        self.metrics.record_candidate(
            CandidateEvent(
                task_id,
                intent.ack.candidate_version,
                intent.ack.submission_id,
                "ABORTED",
                elapsed(intent.started),
            )
        )

    def finalize_submission(
        self,
        task: TaskSpec,
        intent: SubmissionIntent,
        namespace: list,
        trace: tuple[TraceRecord, ...],
    ) -> CandidateSnapshot:
        if not isinstance(namespace, list) or any(not isinstance(item, dict) for item in namespace):
            raise ContractError("SUBMISSION_MISSING", "No snapshot from the successful execution")
        entries = {item["name"]: item for item in namespace}
        requested = (intent.result_var, *intent.evidence_vars)
        for name in requested:
            if name not in entries:
                raise ContractError("SUBMISSION_MISSING", f"Unknown submitted variable {name!r}")
            if not entries[name]["available"]:
                raise ContractError("UNSUPPORTED_ARTIFACT", f"Cannot freeze {name!r}")
        self.validate_output(task, entries[intent.result_var]["value"])
        ack = intent.ack
        refs, catalog = {}, []
        for name, entry in entries.items():
            ref = (
                self.artifacts.freeze(
                    entry["value"],
                    name,
                    task.task_id,
                    ack.candidate_version,
                    ack.submission_id,
                    "child",
                )
                if entry["available"]
                else None
            )
            if ref:
                refs[name] = ref
            catalog.append(
                VariableInfo(
                    name,
                    entry["type_name"],
                    entry["size_bytes"],
                    bool(ref),
                    ack.submission_id,
                    ref.ref_id if ref else None,
                )
            )
        manifest = CandidateManifest(
            ack.submission_id,
            self.artifacts.run_id,
            task.task_id,
            task.task_version,
            ack.candidate_version,
            refs[intent.result_var],
            tuple(refs[name] for name in intent.evidence_vars),
            ack.submission_id,
            intent.summary,
            "SUBMITTED",
            ack.submission_id,
        )
        result = CandidateSnapshot(manifest, tuple(catalog), tuple(refs.values()), trace)
        write_json(self.folder / "submissions" / f"{manifest.submission_id}.json", result)
        self.metrics.record_candidate(
            CandidateEvent(
                task.task_id,
                ack.candidate_version,
                ack.submission_id,
                "SUBMITTED",
                elapsed(intent.started),
            )
        )
        return result

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
