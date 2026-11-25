"""Run OOLONG-Pairs 32K case 12 with the historical selector and DeepSeek Flash."""

import argparse
import importlib.util
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from rlm.supervised_rlm import QwenClient, RunConfig
from rlm.supervised_rlm.client import Message

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
METHOD = "planning-only-thinking"
CASE_ID = 12


class DeepSeekFlashClient(QwenClient):
    def __init__(
        self, config: RunConfig, *, api_key: str, encode: Callable[[list[Message], bool], int]
    ):
        super().__init__(BASE_URL, MODEL, config, api_key=api_key)
        self.encode = encode

    def count_tokens(self, messages: list[Message], thinking: bool) -> int:
        return self.encode(messages, thinking)

    def generation_options(self, thinking: bool) -> dict:
        return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-runner", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.out is None:
        parser.error("--out is required for model execution")

    from tokenizers import Tokenizer

    historical = load_module("historical_oolong_runner", args.historical_runner)
    cases, sources, legacy = historical.load_legacy(args.legacy)
    (case,) = (item for item in cases if int(item[0]["id"]) == CASE_ID)
    row, context, _ = case
    reference = args.historical_runner.parent / "case-12" / "model-input.json"
    previous_input = json.loads(reference.read_text())
    if previous_input != {"question": row["question"], "context": context}:
        raise ValueError("Case 12 question or context differs from the saved baseline input")

    encoder_path = args.tokenizer_dir / "encoding.py"
    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    encoder = load_module("deepseek_oolong_encoding", encoder_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count_tokens(messages: list[Message], thinking: bool) -> int:
        prompt = encoder.encode_messages(messages, thinking_mode="thinking" if thinking else "chat")
        return len(tokenizer.encode(prompt, add_special_tokens=False).ids)

    if args.dry_run:
        checked = count_tokens([{"role": "user", "content": context[:1000]}], True)
        print(f"Case {CASE_ID} matches baseline, context SHA-256 {row['context_sha256']}")
        print(f"Thinking tokenizer sample: {checked} tokens")
        return 0

    config = RunConfig(
        max_root_steps=20,
        max_child_steps=20,
        max_parallel_children=1,
        max_output_tokens=4096,
        deadline=time.time() + 3600,
    )
    historical.METHOD = METHOD
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    request = {
        "base_url": BASE_URL,
        "model": MODEL,
        "dataset": historical.DATASET,
        "method": METHOD,
        "case_ids": [CASE_ID],
        "thinking": {"root": True, "child": True},
        "token_counter": "DeepSeek-V4.1-Flash-1017 local prompt encoder and tokenizer",
    }
    with DeepSeekFlashClient(
        config, api_key=os.environ["DEEPSEEK_API_KEY"], encode=count_tokens
    ) as client:
        historical.snapshot(
            args.out,
            [case],
            [*sources, Path(__file__).resolve(), encoder_path, tokenizer_path],
            args.legacy,
            config,
            request,
        )
        result, _ = historical.run_case(
            args.out / "case-12", row, context, config, client, legacy.score_pairs
        )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
