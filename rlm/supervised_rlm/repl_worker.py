"""Standalone Python worker. It receives JSON and exposes only role-scoped RPC."""

import builtins
import collections
import contextlib
import ctypes
import datetime
import decimal
import errno
import fractions
import functools
import heapq
import io
import itertools
import json
import math
import operator
import re
import resource
import statistics
import string
import sys

MODULES = {
    module.__name__: module
    for module in (
        collections,
        datetime,
        decimal,
        fractions,
        functools,
        heapq,
        itertools,
        json,
        math,
        operator,
        re,
        statistics,
        string,
    )
}
INPUT = sys.stdin
OUTPUT = sys.stdout
MAX_MESSAGE = 48 * 1024 * 1024
MAX_STDIO = 64 * 1024
MAX_SNAPSHOT = 24 * 1024 * 1024


def restrict_process():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    library = ctypes.CDLL("libseccomp.so.2")
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_load.argtypes = library.seccomp_release.argtypes = [ctypes.c_void_p]
    filt = library.seccomp_init(0x00050000 | errno.EPERM)
    if not filt:
        raise RuntimeError("Could not create Linux syscall filter")
    try:
        for name in (
            "read",
            "write",
            "close",
            "fstat",
            "lseek",
            "mmap",
            "mprotect",
            "munmap",
            "mremap",
            "brk",
            "madvise",
            "futex",
            "clock_gettime",
            "clock_getres",
            "gettimeofday",
            "getrandom",
            "getpid",
            "gettid",
            "rt_sigreturn",
            "rt_sigaction",
            "rt_sigprocmask",
            "sigaltstack",
            "exit",
            "exit_group",
        ):
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number < 0 or library.seccomp_rule_add(filt, 0x7FFF0000, number, 0):
                raise RuntimeError(f"Could not allow worker syscall {name}")
        if library.seccomp_load(filt):
            raise RuntimeError("Could not activate Linux syscall filter")
    finally:
        library.seccomp_release(filt)


class View(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def is_accepted(self):
        return self.get("status") == "ACCEPTED"

    def names(self):
        return tuple(item["name"] for item in self["entries"])


class BoundedOutput(io.StringIO):
    def write(self, value):
        remaining = MAX_STDIO - self.tell()
        if remaining > 0:
            super().write(value[:remaining])
        return len(value)


class ToolError(ValueError):
    pass


def send(message):
    OUTPUT.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
    OUTPUT.flush()


def receive():
    line = INPUT.readline(MAX_MESSAGE + 1)
    if not line:
        return None
    if not line.endswith("\n"):
        raise ValueError("Oversized worker message")
    return json.loads(line)


def view(value):
    if isinstance(value, dict):
        return View((key, view(item)) for key, item in value.items())
    if isinstance(value, list):
        return [view(item) for item in value]
    return value


def proxy(name):
    def call(*args, **kwargs):
        send({"kind": "tool", "tool": name, "args": args, "kwargs": kwargs})
        response = receive()
        if not response["ok"]:
            raise ToolError(response["message"])
        return view(response["data"])

    return call


def import_module(name, globals=None, locals=None, fromlist=(), level=0):
    if not level and name in MODULES:
        return MODULES[name]
    raise ImportError(f"Module {name!r} is unavailable in the restricted REPL")


def freeze_namespace(namespace, protected):
    values = []
    used = 0
    for name, value in namespace.items():
        if name in protected and name != "context":
            continue
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            size = len(encoded)
        except (TypeError, ValueError, RecursionError):
            values.append(
                {
                    "name": name,
                    "type_name": type(value).__name__,
                    "size_bytes": 0,
                    "available": False,
                }
            )
            continue
        available = size <= 16 * 1024 * 1024 and used + size <= MAX_SNAPSHOT
        entry = {
            "name": name,
            "type_name": type(value).__name__,
            "size_bytes": size,
            "available": available,
        }
        if available:
            entry["value"] = json.loads(encoded)
            used += size
        values.append(entry)
    return values


def main():
    INPUT.reconfigure(encoding="utf-8")
    OUTPUT.reconfigure(encoding="utf-8")
    restrict_process()
    send({"kind": "ready"})
    config = receive()
    protected = {name: proxy(name) for name in config["tools"]}
    allowed = vars(builtins).copy()
    for name in ("open", "exit", "quit", "help", "breakpoint", "input"):
        allowed.pop(name, None)
    allowed["__import__"] = import_module
    protected.update(context=config["context"], __builtins__=allowed)
    namespace = protected.copy()
    while (message := receive()) is not None:
        out, err = BoundedOutput(), BoundedOutput()
        error_type = error_message = None
        snapshot = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(message["code"], "<supervised-repl>", "exec"), namespace)
                if proxy("has_pending_candidate")():
                    snapshot = freeze_namespace(namespace, protected)
        except BaseException as error:
            error_type, error_message = type(error).__name__, str(error)[:4096]
        finally:
            namespace.update(protected)
        send(
            {
                "kind": "execution",
                "execution_id": message["execution_id"],
                "success": error_type is None,
                "stdout": out.getvalue(),
                "stderr": err.getvalue(),
                "error_type": error_type,
                "error_message": error_message,
                "snapshot": snapshot,
            }
        )


if __name__ == "__main__":
    main()
