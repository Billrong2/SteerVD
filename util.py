#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from openai import AzureOpenAI

ROOT = Path(__file__).resolve().parent
DEFAULT_KEYS = ROOT / "Keys"
DEFAULT_PRO = "gpt-5.4-pro"
MIN_TOKENS = 16
JUDGE_INSTRUCTIONS = "Judge only the meaning of another model's answer. Return exactly one label: YES, NO, or UNKNOWN. YES means the answer says the code is vulnerable. NO means it says not vulnerable. UNKNOWN means ambiguous or no final verdict."

def load_azure_config(keys_path: Path = DEFAULT_KEYS) -> dict[str, str]:
    raw = Path(keys_path).read_text(encoding="utf-8")
    model = re.search(r"^Model:\s*(.+)$", raw, flags=re.M)
    key = re.search(r"^Key:\s*(.+)$", raw, flags=re.M)
    url = next((line.strip() for line in raw.splitlines() if line.strip().startswith("https://")), None)
    if not (model and key and url):
        raise ValueError(f"Could not parse model/key/url from {keys_path}.")
    parsed = urlparse(url)
    return {"model": model.group(1).strip(), "api_key": key.group(1).strip(), "api_version": parse_qs(parsed.query).get("api-version", ["2025-04-01-preview"])[0].strip(), "azure_endpoint": f"{parsed.scheme}://{parsed.netloc}"}

def call_llm(prompt: str, *, keys_path: Path = DEFAULT_KEYS, model: str | None = None, instructions: str | None = None, max_output_tokens: int = 32, temperature: float | None = None, top_p: float | None = None) -> str:
    cfg = load_azure_config(keys_path)
    client = AzureOpenAI(api_key=cfg["api_key"], api_version=cfg["api_version"], azure_endpoint=cfg["azure_endpoint"])
    req = {"model": (model or cfg["model"]).strip(), "input": str(prompt), "instructions": instructions, "max_output_tokens": max(MIN_TOKENS, int(max_output_tokens))}
    if temperature is not None: req["temperature"] = float(temperature)
    if top_p is not None: req["top_p"] = float(top_p)
    return getattr(client.responses.create(**req), "output_text", "") or ""

def _label(text: str) -> str:
    text = str(text or "").upper()
    for label in ("YES", "NO", "UNKNOWN"):
        if re.search(rf"\b{label}\b", text):
            return label
    return "UNKNOWN"

def _retry_prompt(answer: str, previous: str) -> str:
    return f"Your previous response was not parseable. Return exactly one token: YES, NO, or UNKNOWN.\n\nAnswer to judge:\n<<<BEGIN>>>\n{answer}\n<<<END>>>\n\nPrevious response:\n<<<BEGIN>>>\n{previous}\n<<<END>>>"

def _judge(answer: str, *, keys_path: Path, model: str, prompt: str, attempts: int, temperature: float | None, top_p: float | None, max_output_tokens: int) -> str:
    for _ in range(max(1, int(attempts))):
        output = call_llm(prompt, keys_path=keys_path, model=model, instructions=JUDGE_INSTRUCTIONS, max_output_tokens=max_output_tokens, temperature=temperature, top_p=top_p)
        label = _label(output)
        if label != "UNKNOWN":
            return label
        prompt = _retry_prompt(answer, output)
    return "UNKNOWN"

def LLM_annotation(input_string: str, *, keys_path: Path = DEFAULT_KEYS, mini_model: str | None = None, pro_model: str = DEFAULT_PRO, judge_max_attempts: int = 2, max_output_tokens: int = MIN_TOKENS, mini_temperature: float | None = 0.2, mini_top_p: float | None = 1.0) -> str:
    cfg = load_azure_config(keys_path)
    answer = str(input_string)
    mini_model = (mini_model or cfg["model"]).strip()
    prompt = f"Return only YES, NO, or UNKNOWN.\n\nAnswer to judge:\n<<<BEGIN>>>\n{answer}\n<<<END>>>"
    vote_a = _judge(answer, keys_path=keys_path, model=mini_model, prompt=prompt, attempts=judge_max_attempts, temperature=mini_temperature, top_p=mini_top_p, max_output_tokens=max_output_tokens)
    vote_b = _judge(answer, keys_path=keys_path, model=mini_model, prompt=prompt, attempts=judge_max_attempts, temperature=mini_temperature, top_p=mini_top_p, max_output_tokens=max_output_tokens)
    if vote_a == vote_b:
        return vote_a
    meta = f"Two mini judges disagreed. Return only YES, NO, or UNKNOWN.\nMini vote A: {vote_a}\nMini vote B: {vote_b}\n\nAnswer to judge:\n<<<BEGIN>>>\n{answer}\n<<<END>>>"
    return _judge(answer, keys_path=keys_path, model=pro_model, prompt=meta, attempts=judge_max_attempts, temperature=None, top_p=None, max_output_tokens=max_output_tokens)

def _rewrite_jsonl(input_path: Path, output_path: Path, row_fn) -> Path:
    tmp = output_path.with_name(output_path.name + ".tmp")
    with Path(input_path).open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(json.dumps(row_fn(json.loads(line)), ensure_ascii=False) + "\n")
    tmp.replace(output_path)
    return output_path

def _count_rows(path: Path) -> int:
    with Path(path).open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def _show_progress(prefix: str, done: int, total: int) -> None:
    total = max(1, total)
    pct = done / total
    if sys.stderr.isatty():
        width = 24
        fill = min(width, int(width * pct))
        end = "\n" if done >= total else "\r"
        print(f"{prefix} [{'#' * fill}{'.' * (width - fill)}] {done}/{total} {pct:6.2%}", end=end, file=sys.stderr, flush=True)
    elif done == 1 or done >= total or done % 100 == 0:
        print(f"{prefix} {done}/{total} ({pct:6.2%})", file=sys.stderr, flush=True)

def clean_jsonl(input_path: Path, *, output_path: Path | None = None, drop_keys: tuple[str, ...] = ("steering_debug", "head_subset_calibration")) -> Path:
    output_path = output_path or input_path.with_name(f"{input_path.stem}.clean.jsonl")
    return _rewrite_jsonl(input_path, output_path, lambda row: {k: v for k, v in row.items() if k not in drop_keys})

def annotate_jsonl(input_path: Path, *, output_path: Path | None = None, text_field: str = "generated_completion", answer_key: str = "answer", in_place: bool = False, **kwargs: object) -> Path:
    output_path = Path(input_path) if in_place else (output_path or input_path.with_name(f"{input_path.stem}.llm_annotated.jsonl"))
    total = _count_rows(input_path)
    count = {"n": 0}
    prefix = f"[annotate {Path(input_path).name}]"
    def add_answer(row: dict) -> dict:
        count["n"] += 1
        row[answer_key] = LLM_annotation(str(row.get(text_field) or ""), **kwargs)
        _show_progress(prefix, count["n"], total)
        return row
    return _rewrite_jsonl(input_path, output_path, add_answer)

def main() -> int:
    p = argparse.ArgumentParser(description="Azure GPT call helper and reusable LLM judge.")
    p.add_argument("--keys-path", type=Path, default=DEFAULT_KEYS)
    p.add_argument("--prompt"); p.add_argument("--prompt-file", type=Path); p.add_argument("--instructions"); p.add_argument("--model")
    p.add_argument("--temperature", type=float); p.add_argument("--top-p", type=float); p.add_argument("--max-output-tokens", type=int, default=32)
    p.add_argument("--judge-text"); p.add_argument("--judge-file", type=Path); p.add_argument("--mini-model"); p.add_argument("--pro-model", default=DEFAULT_PRO)
    p.add_argument("--judge-max-attempts", type=int, default=2); p.add_argument("--annotate-jsonl", nargs="*", type=Path); p.add_argument("--clean-jsonl", nargs="*", type=Path)
    p.add_argument("--text-field", default="generated_completion"); p.add_argument("--output-suffix", default=".llm_annotated.jsonl"); p.add_argument("--clean-suffix", default=".clean.jsonl")
    p.add_argument("--answer-key", default="answer"); p.add_argument("--in-place", action="store_true")
    args = p.parse_args()
    if args.clean_jsonl:
        outputs = [str(clean_jsonl(path, output_path=path.with_name(f"{path.stem}{args.clean_suffix}"))) for path in args.clean_jsonl]
        print(json.dumps(outputs, indent=2)); return 0
    if args.annotate_jsonl:
        outputs = [str(annotate_jsonl(path, output_path=path.with_name(f"{path.stem}{args.output_suffix}"), text_field=args.text_field, answer_key=args.answer_key, in_place=args.in_place, keys_path=args.keys_path, mini_model=args.mini_model, pro_model=args.pro_model, judge_max_attempts=args.judge_max_attempts, max_output_tokens=args.max_output_tokens)) for path in args.annotate_jsonl]
        print(json.dumps(outputs, indent=2)); return 0
    if args.judge_text is not None or args.judge_file is not None:
        text = args.judge_text if args.judge_text is not None else Path(args.judge_file).read_text(encoding="utf-8")
        print(LLM_annotation(text, keys_path=args.keys_path, mini_model=args.mini_model, pro_model=args.pro_model, judge_max_attempts=args.judge_max_attempts, max_output_tokens=args.max_output_tokens)); return 0
    prompt = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding="utf-8")
    print(call_llm(prompt, keys_path=args.keys_path, model=args.model, instructions=args.instructions, max_output_tokens=args.max_output_tokens, temperature=args.temperature, top_p=args.top_p))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
