#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency at runtime
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parent

from models import ModelRunner
from paths import model_dir_name, resolve_artifact_path
DEFAULT_LOCAL_OUTPUT_DIR = Path("primevul")
DEFAULT_CODE_FIELDS = "func_before,func,code"
DEFAULT_LABEL_FIELDS = "target,vul,label"
DEFAULT_ID_FIELDS = "idx,id,commit_id,cve_id"
DEFAULT_PROTOCOL = "native"
DEFAULT_INSTRUCTION = (
    "You are a security reviewer. Analyze the following function for security vulnerabilities. "
    "Respond with exactly one label on the first line: VULNERABLE or SAFE. "
    "On the second line, give one short sentence explaining the key evidence."
)
PRIMEVUL_SYS_INST = "You are a security expert that is good at static program analysis."
PRIMEVUL_PROMPT_INST = """Please analyze the following code:
```
{func}
```
Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""
PRIMEVUL_PROMPT_INST_COT = """Please analyze the following code:
```
{func}
```
Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Make sure to include one of the options above "explicitly" (EXPLICITLY!!!) in your response.
Let's think step-by-step.
"""


def _normalize_protocol(raw: Optional[str]) -> str:
    text = str(raw or DEFAULT_PROTOCOL).strip().lower()
    aliases = {
        "revd": "revd_cot",
        "revd_cot": "revd_cot",
        "primevul": "primevul_std",
        "primevul_std": "primevul_std",
        "primevul_cot": "primevul_cot",
        "native": "native",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported protocol: {raw!r}")
    return aliases[text]


def _default_instruction_for_protocol(protocol: str) -> str:
    normalized = _normalize_protocol(protocol)
    if normalized == "native":
        return DEFAULT_INSTRUCTION
    if normalized == "primevul_std":
        return f"{PRIMEVUL_SYS_INST}\n\n{PRIMEVUL_PROMPT_INST}".strip()
    if normalized in {"primevul_cot", "revd_cot"}:
        return f"{PRIMEVUL_SYS_INST}\n\n{PRIMEVUL_PROMPT_INST_COT}".strip()
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def _render_primevul_prompt(template: str, *, code: str, language: str) -> str:
    rendered = str(template).format(func=code)
    return rendered.replace("```\n", f"```{language}\n", 1)


def _build_prompt_text(
    *,
    protocol: str,
    code: str,
    language: str,
    instruction_override: Optional[str],
    answer_prefix: str,
) -> str:
    normalized = _normalize_protocol(protocol)
    if instruction_override is not None:
        base = str(instruction_override)
        return f"{base}\n\n```{language}\n{code}\n```{answer_prefix or ''}"

    if normalized == "native":
        return f"{DEFAULT_INSTRUCTION}\n\n```{language}\n{code}\n```{answer_prefix or ''}"

    if normalized == "primevul_std":
        return f"{PRIMEVUL_SYS_INST}\n\n{_render_primevul_prompt(PRIMEVUL_PROMPT_INST, code=code, language=language)}{answer_prefix or ''}"

    if normalized in {"primevul_cot", "revd_cot"}:
        return f"{PRIMEVUL_SYS_INST}\n\n{_render_primevul_prompt(PRIMEVUL_PROMPT_INST_COT, code=code, language=language)}{answer_prefix or ''}"

    raise ValueError(f"Unsupported protocol: {protocol!r}")


def _build_generation_prompt(
    *,
    model: ModelRunner,
    protocol: str,
    code: str,
    language: str,
    instruction_override: Optional[str],
    answer_prefix: str,
) -> str:
    normalized = _normalize_protocol(protocol)
    if instruction_override is None and normalized in {"primevul_std", "primevul_cot", "revd_cot"}:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            user_content = (
                _render_primevul_prompt(PRIMEVUL_PROMPT_INST, code=code, language=language)
                if normalized == "primevul_std"
                else _render_primevul_prompt(PRIMEVUL_PROMPT_INST_COT, code=code, language=language)
            )
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": PRIMEVUL_SYS_INST},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            return f"{prompt}{answer_prefix or ''}"

    return _build_prompt_text(
        protocol=protocol,
        code=code,
        language=language,
        instruction_override=instruction_override,
        answer_prefix=answer_prefix,
    )


def _parse_csv_fields(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _parse_gpu_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    parts = re.split(r"[+,]", cleaned)
    ids: List[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"Invalid GPU id '{part}' in --gpu-ids={raw!r}.")
        ids.append(int(part))
    return ids or None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_field(row: Dict[str, Any], candidates: Sequence[str], *, required: bool) -> Any:
    lowered = {str(key).lower(): key for key in row.keys()}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is None:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    if required:
        raise KeyError(f"None of the candidate fields {list(candidates)!r} were present in row keys {list(row.keys())!r}.")
    return None


def _normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1 if int(value) != 0 else 0

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "vulnerable", "vul", "buggy", "unsafe"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "safe", "clean", "benign", "non-vulnerable", "non_vulnerable"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def _parse_prediction_label(text: str) -> Optional[int]:
    raw = str(text or "").strip()
    if not raw:
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    probe = "\n".join(lines[:3]) if lines else raw
    upper = probe.upper()

    first_line = lines[0].upper() if lines else upper
    if re.fullmatch(r"(?:\(?1\)?[:.]?\s*)?YES\b.*", first_line):
        return 1
    if re.fullmatch(r"(?:\(?2\)?[:.]?\s*)?NO\b.*", first_line):
        return 0
    if re.search(r"\bSAFE\b", first_line):
        return 0
    if re.search(r"\bVULNERABLE\b", first_line):
        return 1

    if re.search(r"\bYES\b", upper):
        return 1
    if re.search(r"\bNO\b", upper):
        return 0
    if re.search(r"\bNOT\s+VULNERABLE\b", upper):
        return 0
    if re.search(r"\bNO\s+VULNERABILIT(?:Y|IES)\b", upper):
        return 0
    if re.search(r"\bSAFE\b", upper) or re.search(r"\bBENIGN\b", upper):
        return 0
    if re.search(r"\bVULNERABLE\b", upper) or re.search(r"\bUNSAFE\b", upper):
        return 1
    return None


def _records_from_json_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    raise ValueError("Unsupported JSON dataset structure; expected a list of records or a dict with data/records/items/examples.")


def _load_local_records(path: Path, dataset_format: str) -> List[Dict[str, Any]]:
    fmt = dataset_format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix in {".parquet", ".pq"}:
            fmt = "parquet"
        else:
            raise ValueError(f"Could not infer dataset format from {path}. Use --dataset-format explicitly.")

    if fmt == "jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(dict(item))
        return rows

    if fmt == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _records_from_json_payload(payload)

    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    if fmt == "parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError("Reading parquet requires pandas + pyarrow or fastparquet.") from exc
        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")

    raise ValueError(f"Unsupported dataset format: {fmt!r}")


def _load_hf_records(dataset_name: str, split: str, cache_dir: Optional[Path]) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        raise RuntimeError("Loading a Hugging Face dataset requires the `datasets` package.") from exc

    dataset = load_dataset(str(dataset_name), split=str(split), cache_dir=str(cache_dir) if cache_dir else None)
    return [dict(row) for row in dataset]


def _load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset_path is not None:
        rows = _load_local_records(args.dataset_path, args.dataset_format)
    else:
        rows = _load_hf_records(args.hf_dataset, args.hf_split, args.hf_cache_dir)

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(rows)
    if args.offset:
        rows = rows[int(args.offset) :]
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    return rows


def _default_run_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _run_generation(
    *,
    model: ModelRunner,
    code: str,
    protocol: str,
    instruction: Optional[str],
    language: str,
    answer_prefix: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    do_sample: bool,
) -> Dict[str, Any]:
    prompt = _build_generation_prompt(
        model=model,
        protocol=protocol,
        code=code,
        language=language,
        instruction_override=instruction,
        answer_prefix=answer_prefix,
    )

    overrides: Dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "do_sample": bool(do_sample),
    }
    if top_k is not None:
        overrides["top_k"] = top_k

    model._current_code_snippet = code
    model._current_vocab_tokens = []
    try:
        return model._generate_with_attn(prompt, overrides=overrides)
    finally:
        model._current_code_snippet = ""
        model._current_vocab_tokens = []


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def _recover_from_cuda_oom(model: ModelRunner) -> None:
    model._steering_runtime = None
    model._last_steering_runtime = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_error_record(
    *,
    row_idx: int,
    sample_id: str,
    sample_run: int,
    sample_seed: int,
    variant: str,
    gold_label: int,
    started: float,
    exc: BaseException,
) -> Dict[str, Any]:
    return {
        "sample_index": int(row_idx),
        "sample_id": str(sample_id),
        "sample_run": int(sample_run),
        "sample_seed": int(sample_seed),
        "variant": str(variant),
        "gold_label": int(gold_label),
        "pred_label": None,
        "is_correct": False,
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "generated_completion": "",
        "steering_debug": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _compute_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    parsed = [row for row in records if row.get("pred_label") is not None]
    tp = fp = tn = fn = 0
    for row in records:
        gold = int(row["gold_label"])
        pred = row.get("pred_label")
        if pred is None:
            if gold == 1:
                fn += 1
            else:
                fp += 1
            continue
        pred = int(pred)
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    coverage = len(parsed) / total if total else 0.0

    return {
        "n_total": int(total),
        "n_parsed": int(len(parsed)),
        "coverage": float(coverage),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _write_summary(
    *,
    output_root: Path,
    run_manifest: Dict[str, Any],
    per_variant: Dict[str, List[Dict[str, Any]]],
    filename: str,
) -> Dict[str, Any]:
    summary = {
        "run": run_manifest,
        "by_variant": {
            variant: _compute_metrics(records)
            for variant, records in per_variant.items()
        },
    }
    (output_root / filename).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_existing_predictions(
    predictions_path: Path,
    variants: Sequence[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], set[Tuple[int, str, int]]]:
    per_variant: Dict[str, List[Dict[str, Any]]] = {str(variant): [] for variant in variants}
    completed: set[Tuple[int, str, int]] = set()
    if not predictions_path.exists():
        return per_variant, completed

    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            variant = str(row.get("variant") or "")
            if variant not in per_variant:
                per_variant[variant] = []
            per_variant[variant].append(dict(row))
            completed.add(
                (
                    int(row.get("sample_index", -1)),
                    variant,
                    int(row.get("sample_run", -1)),
                )
            )
    return per_variant, completed


def _preview_rows(
    *,
    rows: Sequence[Dict[str, Any]],
    code_fields: Sequence[str],
    label_fields: Sequence[str],
    id_fields: Sequence[str],
    protocol: str,
    language: str,
    instruction: Optional[str],
    answer_prefix: str,
    preview_count: int,
) -> None:
    count = min(max(0, int(preview_count)), len(rows))
    for idx in range(count):
        row = rows[idx]
        code = str(_pick_field(row, code_fields, required=True))
        label = _normalize_label(_pick_field(row, label_fields, required=True))
        sample_id = _pick_field(row, id_fields, required=False)
        if sample_id is None:
            sample_id = f"row_{idx:05d}"
        print(f"\n=== Preview {idx + 1}/{count} :: id={sample_id} :: gold={label} ===")
        prompt = _build_prompt_text(
            protocol=protocol,
            code=code,
            language=language,
            instruction_override=instruction,
            answer_prefix=answer_prefix,
        )
        prompt_lines = prompt.splitlines()
        print("\n".join(prompt_lines[:30]))
        if len(prompt_lines) > 30:
            print("...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline vulnerability classification on PrimeVul-style datasets."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-path", type=Path, default=None, help="Local dataset file (.jsonl/.json/.csv/.parquet).")
    source.add_argument("--hf-dataset", type=str, default=None, help="Optional Hugging Face dataset name.")
    parser.add_argument("--hf-split", type=str, default="test", help="Split used with --hf-dataset.")
    parser.add_argument("--hf-cache-dir", type=Path, default=None, help="Optional HF datasets cache directory.")
    parser.add_argument("--dataset-format", choices=["auto", "jsonl", "json", "csv", "parquet"], default="auto")
    parser.add_argument("--code-field", type=str, default=DEFAULT_CODE_FIELDS)
    parser.add_argument("--label-field", type=str, default=DEFAULT_LABEL_FIELDS)
    parser.add_argument("--id-field", type=str, default=DEFAULT_ID_FIELDS)
    parser.add_argument("--language", type=str, default="c")
    parser.add_argument(
        "--protocol",
        choices=["native", "primevul_std", "primevul_cot", "revd_cot"],
        default=DEFAULT_PROTOCOL,
        help="Prompt/task setting. `revd_cot` reproduces the public CoT baseline prompt used by ReVD.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--preview-only", action="store_true", help="Load records and print a few formatted examples without running a model.")
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--variant", choices=["baseline"], default="baseline")
    parser.add_argument("--instruction", type=str, default=None, help="Optional custom instruction. If omitted, selected by --protocol.")
    parser.add_argument("--answer-prefix", type=str, default="\n")
    parser.add_argument("--model-name", type=str, default=None, help="HF model name. Defaults to ModelRunner's built-in default.")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF model cache directory.")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", choices=["on", "off"], default="off")
    parser.add_argument("--samples-per-snippet", type=int, default=1, help="Number of generations to run per snippet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOCAL_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--resume", choices=["on", "off"], default="off")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.protocol = _normalize_protocol(args.protocol)
    samples_per_snippet = max(1, int(args.samples_per_snippet))

    code_fields = _parse_csv_fields(args.code_field)
    label_fields = _parse_csv_fields(args.label_field)
    id_fields = _parse_csv_fields(args.id_field)
    selected_instruction = args.instruction
    if not code_fields:
        raise ValueError("--code-field must specify at least one candidate field.")
    if not label_fields:
        raise ValueError("--label-field must specify at least one candidate field.")

    rows = _load_records(args)
    if not rows:
        raise RuntimeError("No dataset rows were loaded.")

    if args.preview_only:
        _preview_rows(
            rows=rows,
            code_fields=code_fields,
            label_fields=label_fields,
            id_fields=id_fields,
            protocol=args.protocol,
            language=args.language,
            instruction=selected_instruction,
            answer_prefix=args.answer_prefix,
            preview_count=args.preview_count,
        )
        return 0

    model = ModelRunner()
    resolved_model_name = args.model_name or model.model_name
    run_name = args.run_name or _default_run_name()
    output_root = resolve_artifact_path(PROJECT_ROOT, args.output_dir) / model_dir_name(resolved_model_name) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    max_devices = None
    if gpu_ids is not None:
        max_devices = min(len(gpu_ids), visible_gpus) if visible_gpus > 0 else None
    elif args.gpus is not None and visible_gpus > 0:
        max_devices = min(max(1, int(args.gpus)), visible_gpus)

    variants = ["baseline"]

    try:
        model.login_hf()
    except Exception as exc:
        print(f"[WARN] Hugging Face login failed; continuing without auth: {exc}")
    model.config(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        use_4bit=bool(args.use_4bit),
        max_new_tokens=max(1, int(args.max_new_tokens)),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=args.top_k,
        max_devices=max_devices,
    )
    model.build()

    run_manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_path": str(args.dataset_path) if args.dataset_path is not None else None,
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "dataset_format": args.dataset_format,
        "limit": args.limit,
        "offset": int(args.offset),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "language": args.language,
        "protocol": args.protocol,
        "instruction": selected_instruction or _default_instruction_for_protocol(args.protocol),
        "samples_per_snippet": int(samples_per_snippet),
        "variant": "baseline",
        "variants": variants,
        "model_name": model.model_name,
    }
    predictions_path = output_root / "predictions.jsonl"
    if args.resume == "on":
        per_variant, completed_records = _load_existing_predictions(predictions_path, variants)
    else:
        per_variant = {variant: [] for variant in variants}
        completed_records = set()

    run_manifest["resume"] = bool(args.resume == "on")
    run_manifest["existing_predictions"] = int(sum(len(records) for records in per_variant.values()))
    run_manifest["checkpoint_every"] = max(1, int(args.checkpoint_every))
    (output_root / "run_config.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    iterator: Iterable[Any] = enumerate(rows)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(rows), desc="primevul-eval")

    checkpoint_every = max(1, int(args.checkpoint_every))
    new_records = 0
    try:
        sink_mode = "a" if args.resume == "on" else "w"
        with predictions_path.open(sink_mode, encoding="utf-8") as sink:
            for row_idx, row in iterator:
                code = str(_pick_field(row, code_fields, required=True))
                gold_label = _normalize_label(_pick_field(row, label_fields, required=True))
                sample_id = _pick_field(row, id_fields, required=False)
                if sample_id is None:
                    sample_id = f"row_{row_idx:05d}"
                sample_id = str(sample_id)

                for variant in variants:
                    needed_runs = [
                        sample_run
                        for sample_run in range(samples_per_snippet)
                        if (int(row_idx), str(variant), int(sample_run)) not in completed_records
                    ]
                    if not needed_runs:
                        continue
                    pending_runs = list(needed_runs)
                    try:
                        for sample_run in needed_runs:
                            sample_seed = int(args.seed) + int(row_idx) * 1000 + sample_run
                            _seed_all(sample_seed)
                            started = time.perf_counter()
                            result = _run_generation(
                                model=model,
                                code=code,
                                protocol=args.protocol,
                                instruction=selected_instruction,
                                language=args.language,
                                answer_prefix=args.answer_prefix,
                                max_new_tokens=max(1, int(args.max_new_tokens)),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=args.top_k,
                                do_sample=(args.do_sample == "on"),
                            )
                            completion = str(result.get("generated_completion", "") or "")
                            pred_label = _parse_prediction_label(completion)
                            result["elapsed_sec"] = round(time.perf_counter() - started, 4)
                            record = {
                                "sample_index": int(row_idx),
                                "sample_id": sample_id,
                                "sample_run": int(sample_run),
                                "sample_seed": int(sample_seed),
                                "variant": variant,
                                "gold_label": int(gold_label),
                                "pred_label": None if pred_label is None else int(pred_label),
                                "is_correct": bool(pred_label == gold_label) if pred_label is not None else False,
                                "elapsed_sec": float(result.get("elapsed_sec", 0.0)),
                                "generated_completion": completion,
                                "steering_debug": result.get("steering_debug"),
                            }
                            per_variant[variant].append(record)
                            completed_records.add((int(row_idx), str(variant), int(sample_run)))
                            sink.write(json.dumps(record) + "\n")
                            sink.flush()
                            new_records += 1
                            pending_runs.remove(sample_run)
                            if new_records % checkpoint_every == 0:
                                _write_summary(
                                    output_root=output_root,
                                    run_manifest=run_manifest,
                                    per_variant=per_variant,
                                    filename="summary.partial.json",
                                )
                    except Exception as exc:
                        if not _is_cuda_oom(exc):
                            raise
                        print(
                            f"[WARN] CUDA OOM at row={row_idx} sample_id={sample_id} "
                            f"variant={variant}; marking remaining runs as failed and continuing."
                        )
                        _recover_from_cuda_oom(model)
                        for sample_run in pending_runs:
                            sample_seed = int(args.seed) + int(row_idx) * 1000 + sample_run
                            error_record = _make_error_record(
                                row_idx=row_idx,
                                sample_id=sample_id,
                                sample_run=sample_run,
                                sample_seed=sample_seed,
                                variant=variant,
                                gold_label=gold_label,
                                started=time.perf_counter(),
                                exc=exc,
                            )
                            per_variant[variant].append(error_record)
                            completed_records.add((int(row_idx), str(variant), int(sample_run)))
                            sink.write(json.dumps(error_record) + "\n")
                            sink.flush()
                            new_records += 1
                        _write_summary(
                            output_root=output_root,
                            run_manifest=run_manifest,
                            per_variant=per_variant,
                            filename="summary.partial.json",
                        )
                        continue
    except Exception as exc:
        _write_summary(
            output_root=output_root,
            run_manifest=run_manifest,
            per_variant=per_variant,
            filename="summary.partial.json",
        )
        error_payload = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (output_root / "last_error.json").write_text(json.dumps(error_payload, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        model.free()

    summary = _write_summary(
        output_root=output_root,
        run_manifest=run_manifest,
        per_variant=per_variant,
        filename="summary.json",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nPredictions written to {predictions_path}")
    print(f"Summary written to {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
