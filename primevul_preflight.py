#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from primevul_eval import (
    _build_prompt_text,
    _default_instruction_for_protocol,
    _load_local_records,
    _normalize_label,
    _normalize_protocol,
    _parse_csv_fields,
    _pick_field,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset/prompt preflight for PrimeVul runs. Loops the dataset but does not run generation."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-format", choices=["auto", "jsonl", "json", "csv", "parquet"], default="auto")
    parser.add_argument("--protocol", choices=["native", "primevul_std", "primevul_cot", "revd_cot"], default="revd_cot")
    parser.add_argument("--language", type=str, default="c")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--answer-prefix", type=str, default="")
    parser.add_argument("--code-fields", type=str, default="func_before,func,code")
    parser.add_argument("--label-fields", type=str, default="target,vul,label")
    parser.add_argument("--id-fields", type=str, default="idx,id,commit_id,cve_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--error-sample-limit", type=int, default=25)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/primevul_preflight"))
    return parser.parse_args()


def _default_run_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _truncate(text: Any, limit: int = 160) -> str:
    raw = str(text or "")
    return raw if len(raw) <= limit else raw[: limit - 3] + "..."


def main() -> int:
    args = _parse_args()
    protocol = _normalize_protocol(args.protocol)
    instruction = args.instruction or _default_instruction_for_protocol(protocol)
    code_fields = _parse_csv_fields(args.code_fields)
    label_fields = _parse_csv_fields(args.label_fields)
    id_fields = _parse_csv_fields(args.id_fields)

    rows = _load_local_records(args.dataset_path, args.dataset_format)
    if args.offset:
        rows = rows[int(args.offset) :]
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]

    run_name = args.run_name or _default_run_name()
    output_root = (PROJECT_ROOT / args.output_dir / run_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    valid = 0
    failures: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    hashes: List[str] = []
    code_lengths: List[int] = []
    prompt_lengths: List[int] = []
    null_byte_rows = 0
    empty_code_rows = 0
    label_counts: Counter[int] = Counter()

    for row_idx, row in enumerate(rows):
        try:
            code = str(_pick_field(row, code_fields, required=True))
            if "\x00" in code:
                null_byte_rows += 1
            if not code.strip():
                empty_code_rows += 1
                raise ValueError("Empty code after stripping whitespace.")
            gold_label = _normalize_label(_pick_field(row, label_fields, required=True))
            sample_id = _pick_field(row, id_fields, required=False)
            sample_id = f"row_{row_idx:05d}" if sample_id is None else str(sample_id)
            prompt = _build_prompt_text(
                protocol=protocol,
                code=code,
                language=args.language,
                instruction_override=args.instruction,
                answer_prefix=args.answer_prefix,
            )
        except Exception as exc:
            if len(failures) < max(1, int(args.error_sample_limit)):
                failures.append(
                    {
                        "row_index": int(row_idx + int(args.offset)),
                        "row_keys": sorted(str(k) for k in row.keys()),
                        "sample_id_guess": str(row.get("idx") or row.get("id") or row.get("commit_id") or ""),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            continue

        valid += 1
        sample_ids.append(sample_id)
        if "hash" in row:
            hashes.append(str(row.get("hash")))
        code_lengths.append(len(code))
        prompt_lengths.append(len(prompt))
        label_counts[int(gold_label)] += 1

    duplicate_sample_ids = len(sample_ids) - len(set(sample_ids))
    duplicate_hashes = len(hashes) - len(set(hashes))

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "dataset_format": args.dataset_format,
        "protocol": protocol,
        "language": args.language,
        "offset": int(args.offset),
        "limit": args.limit,
        "n_total_rows": int(total),
        "n_valid_rows": int(valid),
        "n_failed_rows": int(total - valid),
        "n_null_byte_rows": int(null_byte_rows),
        "n_empty_code_rows": int(empty_code_rows),
        "duplicate_sample_ids": int(duplicate_sample_ids),
        "duplicate_hashes": int(duplicate_hashes),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "code_length_stats": {
            "min": min(code_lengths) if code_lengths else 0,
            "median": statistics.median(code_lengths) if code_lengths else 0,
            "max": max(code_lengths) if code_lengths else 0,
        },
        "prompt_length_stats": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "median": statistics.median(prompt_lengths) if prompt_lengths else 0,
            "max": max(prompt_lengths) if prompt_lengths else 0,
        },
        "failure_examples": failures,
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
