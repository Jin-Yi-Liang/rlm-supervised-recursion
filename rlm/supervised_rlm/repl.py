"""Persistent agent REPL process and validated host tool transport."""

import inspect
import json
import os
import select
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from time import monotonic, perf_counter_ns
from uuid import uuid4

from rlm.supervised_rlm.metrics import MetricsCollector, ToolCallRecord, elapsed
from rlm.supervised_rlm.schemas import AgentRole, ContractError, ExecutionResult, ToolResponse
from rlm.supervised_rlm.storage import serialize

MAX_MESSAGE = 64 * 1024 * 1024


class REPLError(RuntimeError):
    pass


class REPLRuntime:
    def __init__(
        self,
        role: AgentRole,
        task_id: str,
        context: object,
        tools: dict[str, Callable],
        metrics: MetricsCollector,
        timeout: float,
        *,
        context_ref: str | None = None,
        max_tool_calls: int = 1024,
    ):
        self.role = role
        self.task_id = task_id
        self.metrics = metrics
        self.tools = tools
        self.timeout = timeout
        self.max_tool_calls = max_tool_calls
        self.buffer = bytearray()
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-u", str(Path(__file__).with_name("repl_worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd="/",
            start_new_session=True,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.read_fd, self.write_fd = self.process.stdout.fileno(), self.process.stdin.fileno()
        os.set_blocking(self.write_fd, False)
        try:
            if self.receive(monotonic() + timeout) != {"kind": "ready"}:
                raise REPLError("Restricted Python worker did not initialize")
            self.send(
                {"context": context, "context_ref": context_ref, "tools": list(tools)},
                monotonic() + timeout,
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()

    def receive(self, deadline: float) -> dict:
        while (end := self.buffer.find(b"\n")) < 0:
            if len(self.buffer) >= MAX_MESSAGE:
                raise REPLError("Worker message exceeds size limit")
            remaining = deadline - monotonic()
            if remaining <= 0 or not select.select([self.read_fd], [], [], remaining)[0]:
                raise REPLError("Python execution timed out")
            chunk = os.read(self.read_fd, 65536)
            if not chunk:
                raise REPLError("Restricted Python worker stopped")
            self.buffer.extend(chunk)
        packet = bytes(self.buffer[:end])
        del self.buffer[: end + 1]
        try:
            value = json.loads(packet)
        except (ValueError, RecursionError) as error:
            raise REPLError("Malformed Python worker response") from error
        if not isinstance(value, dict):
            raise REPLError("Invalid Python worker message")
        return value

    def send(self, packet: object, deadline: float) -> None:
        payload = memoryview((serialize(packet) + "\n").encode())
        while payload:
            remaining = deadline - monotonic()
            if remaining <= 0 or not select.select([], [self.write_fd], [], remaining)[1]:
                raise REPLError("Python worker did not read its reply")
            try:
                size = os.write(self.write_fd, payload)
            except BrokenPipeError as error:
                raise REPLError("Restricted Python worker stopped") from error
            payload = payload[size:]

    def call_tool(self, packet: dict, tools: dict[str, Callable]) -> ToolResponse:
        started = perf_counter_ns()
        name = packet.get("tool")
        args, kwargs = packet.get("args"), packet.get("kwargs")
        protocol = isinstance(name, str) and isinstance(args, list) and isinstance(kwargs, dict)
        tool = tools.get(name) if protocol else None
        valid = executed = success = False
        error_code = "INVALID_TOOL_PROTOCOL"
        if protocol:
            error_code = "UNKNOWN_TOOL"
        try:
            if tool is None:
                raise ContractError(error_code, f"Invalid call to {name!r}")
            signature = inspect.signature(tool)
            try:
                signature.bind(*args, **kwargs)
            except TypeError as error:
                call_shape = signature.replace(
                    parameters=[
                        parameter.replace(annotation=inspect.Parameter.empty)
                        for parameter in signature.parameters.values()
                    ],
                    return_annotation=inspect.Signature.empty,
                )
                raise ContractError(
                    "INVALID_ARGUMENT", f"{error}. Use {name}{call_shape}"
                ) from error
            valid = executed = True
            result = tool(*args, **kwargs)
            success, error_code = True, None
            return ToolResponse(True, data=result)
        except ContractError as error:
            error_code = error.code
            if error.code in {"INVALID_ARGUMENT", "PERMISSION_DENIED", "UNKNOWN_TOOL"}:
                valid, executed = False, False
            return ToolResponse(False, code=error.code, message=str(error))
        finally:
            if name != "has_pending_candidate":
                self.metrics.record_tool_call(
                    ToolCallRecord(
                        uuid4().hex,
                        self.task_id,
                        self.role,
                        name if isinstance(name, str) else "",
                        protocol,
                        tool is not None,
                        valid,
                        executed,
                        success if executed else None,
                        elapsed(started),
                        error_code,
                    )
                )

    def execute(
        self, code: str, execution_id: str, pending: Callable
    ) -> tuple[ExecutionResult, list | None]:
        tools = {**self.tools, "has_pending_candidate": pending}
        deadline = monotonic() + self.timeout
        self.send({"code": code, "execution_id": execution_id}, deadline)
        calls = 0
        while True:
            packet = self.receive(deadline)
            if packet.get("kind") == "tool":
                name = packet.get("tool")
                if name != "has_pending_candidate":
                    calls += 1
                    if calls > self.max_tool_calls:
                        raise REPLError(
                            f"Python cell exceeded its {self.max_tool_calls} tool calls"
                        )
                start = monotonic()
                reply = self.call_tool(packet, tools)
                if name in ("rlm_query", "rlm_query_batched") and name in self.tools:
                    # Delegated model calls have their own runtime budgets.
                    deadline += monotonic() - start
                self.send(reply, deadline)
                continue
            if packet.get("kind") != "execution" or packet.get("execution_id") != execution_id:
                raise REPLError("Invalid execution reply")
            try:
                return (
                    ExecutionResult(
                        execution_id,
                        packet["success"],
                        packet["stdout"],
                        packet["stderr"],
                        packet["error_type"],
                        packet["error_message"],
                    ),
                    packet["snapshot"],
                )
            except KeyError as error:
                raise REPLError("Incomplete execution reply") from error
