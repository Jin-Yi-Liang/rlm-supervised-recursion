"""Run the supervised variants using the historical 32K selector and scorer."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from rlm.core.supervised import ReviewFailure, SupervisedRLM, dump


class StopRun(RuntimeError):
    pass


class Wire:
    def __init__(self, base_url, model, folder, deadline):
        self.base = base_url.rstrip("/")
        self.model = model
        self.folder = folder
        self.deadline = deadline
        self.http = httpx.Client(trust_env=False)
        self.sequence = 0
        self.usage = {}

    def post(self, url, payload):
        remaining = self.deadline - time.time()
        if remaining <= 30:
            raise StopRun("Service deadline approaching; preserve attempt")
        try:
            response = self.http.post(url, json=payload, timeout=min(remaining - 10, 3600))
            response.raise_for_status()
        except httpx.HTTPError as error:
            detail = (
                error.response.text[:1200]
                if isinstance(error, httpx.HTTPStatusError)
                else str(error)
            )
            raise StopRun(f"Endpoint error: {detail}") from error
        return response.json()

    def fits(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        count = self.post(self.base.removesuffix("/v1") + "/tokenize", body)["count"]
        return count + 4096 <= 32768

    def chat(self, role, task, messages):
        self.sequence += 1
        directory = self.folder / "wire" / f"{self.sequence:05d}-{role}"
        thinking = role in ("root", "supervisor")
        body = dict(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=0.0,
            chat_template_kwargs={"enable_thinking": thinking},
        )
        dump(directory / "request.json", body)
        dump(directory / "meta.json", {"role": role, "task": task, "started": time.time()})
        try:
            response = self.post(self.base + "/chat/completions", body)
        except StopRun as error:
            dump(directory / "error.json", {"error": str(error)})
            raise
        dump(directory / "response.json", response)
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if "</think>" in content:
            prefix, content = content.split("</think>", 1)
            reasoning = reasoning or prefix.replace("<think>", "").strip()
        if not thinking and reasoning:
            raise StopRun("Sub-Agent emitted nonempty reasoning with thinking disabled")
        entry = self.usage.setdefault(
            role,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "missing_usage": 0,
                "with_reasoning": 0,
            },
        )
        entry["requests"] += 1
        entry["with_reasoning"] += int(bool(reasoning))
        usage = response.get("usage") or {}
        entry["missing_usage"] += int(
            not all(key in usage for key in ("prompt_tokens", "completion_tokens"))
        )
        for key in ("prompt_tokens", "completion_tokens"):
            entry[key] += usage.get(key, 0)
        dump(self.folder / "usage.json", self.usage)
        if choice.get("finish_reason") == "length" and role == "supervisor":
            raise ReviewFailure("Supervisor output truncated; cannot commit")
        return content.strip()


def legacy_cases(script):
    spec = importlib.util.spec_from_file_location("historical_oolong32768", script)
    module = importlib.util.module_from_spec(spec)
    sys.path.append(str(script.parent))
    spec.loader.exec_module(module)
    cases, _ = module.selected_cases(
        SimpleNamespace(
            dataset="oolong_pairs_32768",
            case_id=None,
            pairs_index=module.PAIRS_INDEX,
            pairs_gold=module.PAIRS_GOLD,
        )
    )
    if {int(row["id"]) for row, _, _ in cases} != set(range(1, 21)) or len(cases) != 20:
        raise StopRun("Historical selector did not produce the expected 20 unique IDs")
    return sorted(cases, key=lambda item: int(item[0]["id"])), module.score_pairs


def run_case(args, directory, mode, question, context, row=None, score=None):
    directory.mkdir(parents=True, exist_ok=False)
    dump(directory / "model-input.json", {"question": question, "context": context})
    wire = Wire(args.base_url, args.model, directory, args.deadline)
    runner = SupervisedRLM(wire, directory, planned=mode == "planned")
    started = time.time()
    result = {"status": "running", "mode": mode, "started": started}
    dump(directory / "result.json", result)
    try:
        answer = runner.solve(question, context)
        result.update(
            answer=answer, score=score(row, answer) if score else None, status="completed"
        )
        if row is None and (
            answer.strip() != "11"
            or not runner.receipts
            or not all(
                wire.usage.get(role, {}).get("with_reasoning", 0) for role in ("root", "supervisor")
            )
        ):
            raise ReviewFailure("Synthetic smoke did not demonstrate the required roles/output")
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        if (
            not isinstance(error, RuntimeError)
            or isinstance(error, (StopRun, ReviewFailure))
            or row is None
        ):
            result["halt"] = True
    finally:
        result.update(
            elapsed_seconds=time.time() - started,
            reviews=runner.reviews,
            rejections=runner.rejections,
            accepted=len(runner.receipts),
            usage=wire.usage,
        )
        dump(directory / "result.json", result)
        wire.http.close()
    return result


def execute(args):
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        question = (
            "Delegate once to a child: add the two input numbers using Python. "
            "Pass the input list, retrieve the accepted result by reference, then submit "
            "only the decimal sum without formatting."
        )
        return run_case(args, args.out / "smoke", "no-plan", question, [4, 7])
    cases, score = legacy_cases(args.legacy.resolve())
    manifest = args.out / "manifest.json"
    config = {
        "base_url": args.base_url,
        "model": args.model,
        "order": ["no-plan", "planned"],
        "thinking": {"root": True, "sub": False, "supervisor": True},
        "case_ids": list(range(1, 21)),
        "depth": 3,
        "iterations": 20,
        "repairs": 2,
        "legacy_sha256": hashlib.sha256(args.legacy.read_bytes()).hexdigest(),
        "context_hashes": sorted({row["context_sha256"] for row, _, _ in cases}),
        "revision": args.revision,
    }
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise StopRun("Manifest changed; use a new output directory")
    dump(manifest, config)
    for mode in config["order"]:
        results = []
        for row, context, _ in cases:
            case_id = int(row["id"])
            case = args.out / mode / f"case-{case_id:02}"
            previous = sorted(case.glob("attempt-*/result.json"))
            finished = next(
                (
                    item
                    for path in previous
                    if (item := json.loads(path.read_text()))["status"] == "completed"
                ),
                None,
            )
            if finished is None:
                remaining = args.deadline - time.time()
                estimate = max(900, results[-1]["elapsed_seconds"] if results else 900)
                if remaining < estimate + 300:
                    return {
                        "status": "paused_deadline",
                        "mode": mode,
                        "next_case": case_id,
                        "remaining": remaining,
                    }
                attempt = case / f"attempt-{len(list(case.glob('attempt-*'))) + 1:03}"
                dump(
                    args.out / "progress.json",
                    {"status": "running", "mode": mode, "case": case_id, "attempt": attempt.name},
                )
                finished = run_case(args, attempt, mode, row["question"], context, row, score)
                print(mode, case_id, finished["status"], finished.get("error", ""), flush=True)
            results.append(finished)
            scored = [item["score"] for item in results if item["status"] == "completed"]
            dump(
                args.out / mode / "summary.json",
                {
                    "attempted": len(results),
                    "completed": len(scored),
                    "planned": 20,
                    "mean_pair_f1_completed": sum(x["pair_f1"] for x in scored) / len(scored)
                    if scored
                    else None,
                    "mean_exact_completed": sum(x["exact_match"] for x in scored) / len(scored)
                    if scored
                    else None,
                    "cases": [
                        {k: v for k, v in item.items() if k not in ("score", "answer")}
                        | (
                            {
                                "pair_f1": item["score"]["pair_f1"],
                                "exact_match": item["score"]["exact_match"],
                            }
                            if item["status"] == "completed"
                            else {}
                        )
                        for item in results
                    ],
                },
            )
            if finished.get("halt"):
                return {
                    "status": "paused_failure",
                    "mode": mode,
                    "case": case_id,
                    "error": finished["error"],
                }
        if len(scored) != 20:
            return {"status": "incomplete", "mode": mode, "completed": len(scored)}
    return {"status": "completed", "arms": 2, "cases_per_arm": 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--revision", default="working-tree")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and not args.legacy:
        parser.error("--legacy must point to the original 32K script")
    os.umask(0o077)
    result = execute(args)
    dump(args.out / "progress.json", result)
    print({k: v for k, v in result.items() if k not in ("score", "answer")}, flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
