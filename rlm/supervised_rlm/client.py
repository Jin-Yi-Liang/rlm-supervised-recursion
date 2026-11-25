"""Qwen-compatible OpenAI client and shared, budgeted agent loop."""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import Protocol
from uuid import uuid4

import httpx
from openai import OpenAI

from rlm.supervised_rlm.metrics import (
    LLMCallRecord,
    MetricsCollector,
    TokenUsage,
    ToolCallRecord,
    elapsed,
)
from rlm.supervised_rlm.repl import REPLRuntime
from rlm.supervised_rlm.schemas import AgentRole, BudgetExhausted, RunConfig, TraceRecord
from rlm.supervised_rlm.storage import serialize, write_json, write_text
from rlm.utils.parsing import find_code_blocks

Message = dict[str, str]


@dataclass(frozen=True)
class ModelReply:
    content: str
    usage: TokenUsage = TokenUsage()
    finish_reason: str | None = None
    reasoning_present: bool = False


class Backend(Protocol):
    model: str

    def count_tokens(self, messages: list[Message], thinking: bool) -> int: ...

    def complete(
        self, messages: list[Message], thinking: bool, timeout: float, request_dir: Path
    ) -> ModelReply: ...


class QwenClient:
    def __init__(self, base_url: str, model: str, config: RunConfig, *, api_key: str = "EMPTY"):
        self.model = model
        self.config = config
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(trust_env=False)
        self.sdk = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=self.http,
            max_retries=0,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.sdk.close()

    def count_tokens(self, messages: list[Message], thinking: bool) -> int:
        response = self.http.post(
            self.base_url.removesuffix("/v1") + "/tokenize",
            json={
                "model": self.model,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["count"]

    def generation_options(self, thinking: bool) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": thinking}, "top_k": 20}

    def complete(
        self,
        messages: list[Message],
        thinking: bool,
        timeout: float,
        request_dir: Path,
    ) -> ModelReply:
        options = self.generation_options(thinking)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.config.max_output_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 0.0,
        }
        write_json(request_dir / "request.json", payload | options)
        try:
            response = self.sdk.chat.completions.create(
                **payload, timeout=timeout, extra_body=options
            )
        except Exception as error:
            detail = str(error).replace(self.sdk.api_key, "[REDACTED]")
            write_json(
                request_dir / "error.json", {"type": type(error).__name__, "message": detail}
            )
            raise RuntimeError(f"{type(error).__name__}: {detail}") from None
        raw = response.model_dump()
        write_json(request_dir / "response.json", raw)
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning_present = bool(
            getattr(choice.message, "reasoning_content", None)
            or getattr(choice.message, "reasoning", None)
        )
        if "</think>" in content:
            thought, content = content.split("</think>", 1)
            reasoning_present |= bool(thought.removeprefix("<think>").strip())
        elif content.lstrip().startswith("<think>"):
            reasoning_present, content = True, ""
        return ModelReply(
            content.strip(),
            TokenUsage.from_provider(raw.get("usage")),
            choice.finish_reason,
            reasoning_present,
        )


@dataclass
class AgentSession:
    role: AgentRole
    task_id: str
    session_id: str
    messages: list[Message]
    candidate_version: int | None = None
    history: list[TraceRecord] = field(default_factory=list)


class AgentLoop:
    def __init__(
        self,
        folder: Path,
        backend: Backend,
        config: RunConfig,
        metrics: MetricsCollector,
    ):
        self.folder, self.backend, self.config, self.metrics = folder, backend, config, metrics
        self.calls = self.tokens = 0
        self.budget_lock = Lock()

    def time_left(self) -> float:
        if self.config.deadline is None:
            return 600
        remaining = self.config.deadline - time.time()
        if remaining <= 2:
            raise BudgetExhausted("Run deadline reached")
        return min(600, remaining - 1)

    def window(self, session: AgentSession, thinking: bool) -> tuple[list[Message], int]:
        messages = session.messages.copy()
        maximum = self.config.context_window - self.config.max_output_tokens
        count = self.backend.count_tokens(messages, thinking)
        if count <= maximum:
            return messages, count
        # Preserve the initial contract and the latest turns. The entire transcript remains on disk.
        marker = {
            "role": "user",
            "content": "Earlier turns are archived. Your Python variables and current task state persist.",
        }
        tail = messages[2:]
        while len(tail) > 2:
            tail = tail[2:]
            reduced = [*messages[:2], marker, *tail]
            count = self.backend.count_tokens(reduced, thinking)
            if count <= maximum:
                return reduced, count
        raise BudgetExhausted("The agent contract and latest turns exceed the model context window")

    def call(self, session: AgentSession) -> ModelReply:
        self.time_left()
        thinking = True
        messages, count = self.window(session, thinking)
        reserved = count + self.config.max_output_tokens
        with self.budget_lock:
            if (
                self.calls >= self.config.max_model_calls
                or self.tokens + reserved > self.config.max_tokens
            ):
                raise BudgetExhausted("Global model calls or token budget exhausted")
            self.calls += 1
            sequence = self.calls
            self.tokens += reserved
        request_id = uuid4().hex
        location = self.folder / "wire" / f"{sequence:05d}-{session.role.lower()}"
        write_json(
            location / "meta.json",
            {
                "request_id": request_id,
                "role": session.role,
                "task_id": session.task_id,
                "session_id": session.session_id,
                "candidate_version": session.candidate_version,
                "prompt_tokens_checked": count,
                "thinking": thinking,
            },
        )
        result = None
        started = perf_counter_ns()
        try:
            result = self.backend.complete(messages, thinking, self.time_left(), location)
            write_text(location / "model-output.txt", result.content)
            return result
        finally:
            usage = result.usage if result else TokenUsage()
            with self.budget_lock:
                if usage.total_tokens is not None:
                    self.tokens += usage.total_tokens - reserved
            self.metrics.record_llm_call(
                LLMCallRecord(
                    request_id,
                    session.task_id,
                    session.role,
                    self.backend.model,
                    elapsed(started),
                    result is not None,
                    result.finish_reason if result else None,
                    usage,
                    session.candidate_version,
                    result.reasoning_present if result else False,
                )
            )

    def message(self, session: AgentSession, role: str, content: str) -> None:
        item = {"role": role, "content": content}
        session.messages.append(item)
        with (self.folder / "agents" / f"{session.session_id}.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            print(serialize({"message": item}), file=stream)

    def start(
        self, role: AgentRole, task_id: str, system: str, brief: object, version: int | None = None
    ):
        session = AgentSession(role, task_id, uuid4().hex, [], version)
        (self.folder / "agents").mkdir(exist_ok=True)
        self.metrics.event(
            "agent_started",
            {
                "role": role,
                "task_id": task_id,
                "session_id": session.session_id,
                "candidate_version": version,
            },
        )
        self.message(session, "system", system)
        self.message(session, "user", serialize(brief))
        return session

    def tool_protocol(self, session: AgentSession, valid: bool, error: str | None = None) -> None:
        if not valid:
            self.metrics.record_tool_call(
                ToolCallRecord(
                    uuid4().hex,
                    session.task_id,
                    session.role,
                    "execute_python",
                    False,
                    False,
                    False,
                    False,
                    None,
                    0,
                    error,
                )
            )

    def turn(
        self,
        session: AgentSession,
        repl: REPLRuntime,
        step: int,
        pending_variables: Callable,
        finish_cell: Callable,
    ) -> object:
        reply = self.call(session)
        self.message(session, "assistant", reply.content)
        code_response = re.sub(r"(?m)^```python[ \t]*$", "```repl", reply.content)
        blocks = find_code_blocks(code_response)
        outputs = []
        outcome = None
        if not blocks:
            if "```" in reply.content:
                self.tool_protocol(session, False, "INVALID_TOOL_PROTOCOL")
            if reply.finish_reason == "length":
                outputs.append(
                    "The response reached the token limit without a complete REPL block."
                )
            outputs.append("Use a fenced repl block with valid Python and the provided tools.")
        for code in blocks[:8]:
            execution_id = uuid4().hex
            started = perf_counter_ns()
            result = None
            try:
                result, snapshot = repl.execute(code, execution_id, pending_variables)
            finally:
                self.metrics.record_tool_call(
                    ToolCallRecord(
                        execution_id,
                        session.task_id,
                        session.role,
                        "execute_python",
                        True,
                        True,
                        True,
                        True,
                        result.success if result else False,
                        elapsed(started),
                        result.error_type if result else "REPL_FAILURE",
                    )
                )
            trace = TraceRecord(execution_id, step, code, result)
            session.history.append(trace)
            with (self.folder / "agents" / f"{session.session_id}.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                print(serialize({"execution": trace}), file=stream)
            output = (result.stdout + result.stderr)[:4200]
            if not result.success:
                output += f"\n{result.error_type}: {result.error_message}"
            outcome, detail = finish_cell(result, snapshot)
            outputs.append(output + ("\n" + detail if detail else ""))
            if outcome is not None:
                break
        self.message(
            session,
            "user",
            "\n".join(outputs)[:9000] or "Execution completed without printed output.",
        )
        return outcome
