"""JSON persistence for artifacts and structured run records."""

import ast
import hashlib
import json
import re
from dataclasses import is_dataclass
from pathlib import Path
from uuid import uuid4

from rlm.supervised_rlm.schemas import ArtifactRef, ArtifactView, ContractError

PATH_COMPONENT = re.compile(r"""\.([a-zA-Z_]\w*)|\[(-?\d+|'[^']*'|"[^"]*")\]""")
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


def serialize(value: object) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = {key: getattr(value, key) for key in value.__dataclass_fields__}
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=serialize_object)


def serialize_object(value: object) -> dict:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: getattr(value, key) for key in value.__dataclass_fields__}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    write_text(path, serialize(value))


def write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(data)
    temporary.replace(path)


def text(value: object, name: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ContractError("INVALID_ARGUMENT", f"{name} must contain 1 to {limit} characters")
    return value


def strings(values: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(values, list) or len(values) > 128 or (not values and not allow_empty):
        raise ContractError("INVALID_ARGUMENT", f"{name} must be a list of strings")
    return tuple(text(value, name) for value in values)


def page(offset: int, limit: int, maximum: int = 8192) -> None:
    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= maximum:
        raise ContractError(
            "INVALID_ARGUMENT", f"Offset must be nonnegative and limit in 1..{maximum}"
        )


class ArtifactStore:
    def __init__(self, folder: Path, run_id: str):
        self.folder = folder
        self.run_id = run_id
        self.refs: dict[str, ArtifactRef] = {}

    def freeze(
        self,
        value: object,
        name: str,
        task_id: str,
        candidate_version: int = 0,
        snapshot_id: str = "",
        namespace: str = "root",
    ) -> ArtifactRef:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        except (TypeError, ValueError, RecursionError) as error:
            raise ContractError("UNSUPPORTED_ARTIFACT", str(error)) from error
        if len(encoded) > MAX_ARTIFACT_BYTES:
            raise ContractError("ARTIFACT_TOO_LARGE", "Artifact exceeds 16 MiB")
        ref = ArtifactRef(
            uuid4().hex,
            self.run_id,
            task_id,
            candidate_version,
            namespace,
            name,
            type(value).__name__,
            len(encoded),
            hashlib.sha256(encoded).hexdigest(),
            "json",
            snapshot_id,
        )
        write_text(self.folder / "artifacts" / f"{ref.ref_id}.json", encoded.decode())
        self.refs[ref.ref_id] = ref
        return ref

    def require(self, ref_id: str) -> ArtifactRef:
        if not isinstance(ref_id, str) or ref_id not in self.refs:
            raise ContractError("ARTIFACT_NOT_FOUND", "Unknown artifact reference")
        return self.refs[ref_id]

    def value(self, ref: ArtifactRef) -> object:
        return json.loads((self.folder / "artifacts" / f"{ref.ref_id}.json").read_bytes())

    def read(
        self,
        ref: ArtifactRef,
        path: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> ArtifactView:
        return make_view(self.value(ref), ref.variable_name, ref.ref_id, path, offset, limit)


def make_view(
    value: object,
    name: str,
    ref_id: str,
    path: str | None,
    offset: int,
    limit: int,
) -> ArtifactView:
    page(offset, limit)
    path = "$" if path is None else path
    if not isinstance(path, str) or not path.startswith("$"):
        raise ContractError("INVALID_ARGUMENT", "Use a $-rooted artifact path")
    position = 1
    try:
        while position < len(path):
            match = PATH_COMPONENT.match(path, position)
            if not match:
                raise ValueError("Invalid artifact path component")
            key = match[1] if match[1] is not None else ast.literal_eval(match[2])
            value = value[key]
            position = match.end()
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ContractError("ARTIFACT_NOT_FOUND", str(error)) from error

    if isinstance(value, (list, str, dict)):
        total = len(value)
        if isinstance(value, dict):
            value = dict(list(value.items())[offset : offset + limit])
        else:
            value = value[offset : offset + limit]
        next_offset = offset + limit if offset + limit < total else None
    else:
        if offset:
            raise ContractError("INVALID_ARGUMENT", "A scalar has no nonzero offset")
        next_offset = None
    if len(serialize(value).encode()) > MAX_ARTIFACT_BYTES:
        raise ContractError("INVALID_ARGUMENT", "Select a smaller artifact page")
    return ArtifactView(
        name, path, offset, limit, next_offset is not None, next_offset, value, ref_id
    )
