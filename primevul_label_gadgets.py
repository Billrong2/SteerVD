#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency at runtime
    tqdm = None

from models import ModelRunner
from primevul_eval import (
    _is_cuda_oom,
    _normalize_protocol,
    _parse_gpu_ids,
    _parse_prediction_label,
    _recover_from_cuda_oom,
    _run_generation,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "Source" / "primevul_test_vulnerable_snippets"
DEFAULT_PAIRED_DATASET_PATH = PROJECT_ROOT / "Source" / "primevul_test_paired.jsonl"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PROTOCOL = "native"
LABEL_FILENAME = "gadget_label.json"
LABEL_ERROR_FILENAME = "gadget_label_error.json"
PAIRED_GROUND_TRUTH_FILENAME = "paired_ground_truth.json"
PREDICTIONS_FILENAME = "gadget_predictions.jsonl"
STATE_FILENAME = "gadget_label_state.json"


@dataclass(frozen=True)
class GadgetTarget:
    snippet_idx: str
    snippet_dir: Path
    gadget_dir: Path
    gadget_index: int
    api_call_name: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield dict(payload)


def _numeric_name_key(path: Path) -> Tuple[int, str]:
    parsed = _safe_int(path.name)
    if parsed is None:
        return (1, path.name)
    return (0, f"{parsed:020d}")


def _gadget_key(path: Path) -> Tuple[int, str]:
    prefix = path.name.split("__", 1)[0]
    index_text = prefix.replace("gadget_", "", 1)
    parsed = _safe_int(index_text)
    if parsed is None:
        return (1, path.name)
    return (0, f"{parsed:06d}")


def _normalize_target(value: Any) -> Optional[int]:
    parsed = _safe_int(value)
    if parsed is None:
        return None
    return 1 if parsed != 0 else 0


def _normalize_lines(text: str) -> List[str]:
    return [line.rstrip() for line in str(text or "").splitlines()]


def _normalize_statement_text(text: str) -> str:
    statement = str(text or "")
    statement = re.sub(r"//.*$", "", statement).strip()
    statement = " ".join(statement.split())
    return statement


def _compute_changed_lines(vulnerable_func: str, fixed_func: str) -> Tuple[List[int], List[int]]:
    vulnerable_lines = _normalize_lines(vulnerable_func)
    fixed_lines = _normalize_lines(fixed_func)
    matcher = SequenceMatcher(a=vulnerable_lines, b=fixed_lines, autojunk=False)
    vulnerable_changed: set[int] = set()
    fixed_changed: set[int] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        vulnerable_changed.update(range(i1 + 1, i2 + 1))
        fixed_changed.update(range(j1 + 1, j2 + 1))
    return sorted(vulnerable_changed), sorted(fixed_changed)


def _load_paired_rows(path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(path))


def _group_exact_opposite(rows: Sequence[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if len(rows) != 2:
        return None
    normalized = [_normalize_target(row.get("target")) for row in rows]
    if sorted(normalized) != [0, 1]:
        return None
    vulnerable_row = next(row for row in rows if _normalize_target(row.get("target")) == 1)
    fixed_row = next(row for row in rows if _normalize_target(row.get("target")) == 0)
    return dict(vulnerable_row), dict(fixed_row)


def _build_pair_indexes(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    by_project_commit: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    by_project_commit_bigvul: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    by_bigvul: Dict[int, List[Dict[str, Any]]] = {}

    for raw in rows:
        row = dict(raw)
        idx_value = _safe_int(row.get("idx"))
        if idx_value is not None:
            by_idx[idx_value] = row

        project = str(row.get("project") or "")
        commit_id = str(row.get("commit_id") or "")
        if project and commit_id:
            by_project_commit.setdefault((project, commit_id), []).append(row)

        big_vul_idx = _safe_int(row.get("big_vul_idx"))
        if big_vul_idx is not None:
            by_bigvul.setdefault(big_vul_idx, []).append(row)
            if project and commit_id:
                by_project_commit_bigvul.setdefault((project, commit_id, big_vul_idx), []).append(row)

    exact_project_commit: Dict[Tuple[str, str], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for key, group in by_project_commit.items():
        match = _group_exact_opposite(group)
        if match is not None:
            exact_project_commit[key] = match

    exact_project_commit_bigvul: Dict[Tuple[str, str, int], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for key, group in by_project_commit_bigvul.items():
        match = _group_exact_opposite(group)
        if match is not None:
            exact_project_commit_bigvul[key] = match

    exact_bigvul: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for key, group in by_bigvul.items():
        match = _group_exact_opposite(group)
        if match is not None:
            exact_bigvul[key] = match

    return {
        "by_idx": by_idx,
        "exact_project_commit": exact_project_commit,
        "exact_project_commit_bigvul": exact_project_commit_bigvul,
        "exact_bigvul": exact_bigvul,
    }


def _resolve_pair(row: Dict[str, Any], pair_indexes: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    source_row = row
    idx_value = _safe_int(row.get("idx"))
    if idx_value is not None:
        source_row = dict(pair_indexes["by_idx"].get(idx_value) or row)

    target = _normalize_target(source_row.get("target"))
    project = str(source_row.get("project") or row.get("project") or "")
    commit_id = str(source_row.get("commit_id") or row.get("commit_id") or "")
    big_vul_idx = _safe_int(source_row.get("big_vul_idx"))

    if big_vul_idx is not None and project and commit_id:
        match = pair_indexes["exact_project_commit_bigvul"].get((project, commit_id, big_vul_idx))
        if match is not None:
            vulnerable_row, fixed_row = match
            return vulnerable_row, fixed_row, "project_commit_big_vul"

    if project and commit_id:
        match = pair_indexes["exact_project_commit"].get((project, commit_id))
        if match is not None:
            vulnerable_row, fixed_row = match
            return vulnerable_row, fixed_row, "project_commit"

    if big_vul_idx is not None:
        match = pair_indexes["exact_bigvul"].get(big_vul_idx)
        if match is not None:
            vulnerable_row, fixed_row = match
            return vulnerable_row, fixed_row, "big_vul_idx"

    if target is None:
        return None, None, "missing_target"
    return None, None, "unavailable"


def _build_paired_ground_truth(row: Dict[str, Any], pair_indexes: Dict[str, Any]) -> Dict[str, Any]:
    vulnerable_row, fixed_row, resolution = _resolve_pair(row, pair_indexes)
    if vulnerable_row is None or fixed_row is None:
        return {
            "paired_available": False,
            "pair_resolution": resolution,
            "vulnerable_idx": _safe_int(row.get("idx")),
            "fixed_idx": None,
            "big_vul_idx": _safe_int(row.get("big_vul_idx")),
            "commit_id": row.get("commit_id"),
            "cwe": row.get("cwe"),
            "fixed_func": None,
            "vulnerable_changed_lines": [],
            "fixed_changed_lines": [],
            "vulnerable_changed_statements": [],
        }

    vulnerable_func = str(vulnerable_row.get("func") or "")
    fixed_func = str(fixed_row.get("func") or "")
    vulnerable_changed_lines, fixed_changed_lines = _compute_changed_lines(vulnerable_func, fixed_func)
    vulnerable_func_lines = vulnerable_func.splitlines()
    vulnerable_changed_statements = [
        vulnerable_func_lines[line_no - 1]
        for line_no in vulnerable_changed_lines
        if 1 <= int(line_no) <= len(vulnerable_func_lines)
    ]
    return {
        "paired_available": True,
        "pair_resolution": resolution,
        "vulnerable_idx": _safe_int(vulnerable_row.get("idx")),
        "fixed_idx": _safe_int(fixed_row.get("idx")),
        "big_vul_idx": _safe_int(vulnerable_row.get("big_vul_idx")),
        "commit_id": vulnerable_row.get("commit_id"),
        "cwe": vulnerable_row.get("cwe"),
        "fixed_func": fixed_func,
        "vulnerable_changed_lines": vulnerable_changed_lines,
        "fixed_changed_lines": fixed_changed_lines,
        "vulnerable_changed_statements": vulnerable_changed_statements,
    }


def _compute_patch_overlap(gadget_lines: Sequence[Any], paired_ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    if not paired_ground_truth.get("paired_available"):
        return {
            "paired_available": False,
            "patch_overlap_lines": [],
            "patch_overlap_count": 0,
            "patch_overlap_ratio": 0.0,
            "hits_patch": False,
            "patch_support_label": "unknown",
        }

    gadget_line_set = {int(value) for value in gadget_lines if _safe_int(value) is not None}
    changed_line_set = {
        int(value)
        for value in (paired_ground_truth.get("vulnerable_changed_lines") or [])
        if _safe_int(value) is not None
    }
    overlap = sorted(gadget_line_set & changed_line_set)
    overlap_count = len(overlap)
    denominator = max(1, len(gadget_line_set))
    return {
        "paired_available": True,
        "patch_overlap_lines": overlap,
        "patch_overlap_count": overlap_count,
        "patch_overlap_ratio": round(float(overlap_count) / float(denominator), 6),
        "hits_patch": overlap_count > 0,
        "patch_support_label": "supports_vulnerable" if overlap_count > 0 else "no_patch_overlap",
    }


def _compute_statement_match_support(code_gadget: str, paired_ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    if not paired_ground_truth.get("paired_available"):
        return {
            "statement_match_count": 0,
            "statement_match_ratio": 0.0,
            "matched_gadget_statements": [],
            "matched_vulnerable_changed_statements": [],
            "hits_changed_statement": False,
            "heuristic_gadget_label": "unknown",
            "heuristic_label_source": "unavailable_pair",
        }

    changed_statements = [
        _normalize_statement_text(value)
        for value in (paired_ground_truth.get("vulnerable_changed_statements") or [])
    ]
    changed_statements = [value for value in changed_statements if value]
    gadget_statements = [_normalize_statement_text(value) for value in str(code_gadget or "").splitlines()]
    gadget_statements = [value for value in gadget_statements if value]

    matched_gadget_statements: List[str] = []
    matched_changed_statements: List[str] = []
    for gadget_statement in gadget_statements:
        for changed_statement in changed_statements:
            if (
                gadget_statement == changed_statement
                or gadget_statement in changed_statement
                or changed_statement in gadget_statement
            ):
                if gadget_statement not in matched_gadget_statements:
                    matched_gadget_statements.append(gadget_statement)
                if changed_statement not in matched_changed_statements:
                    matched_changed_statements.append(changed_statement)

    match_count = len(matched_gadget_statements)
    denominator = max(1, len(gadget_statements))
    hits_changed_statement = match_count > 0
    return {
        "statement_match_count": match_count,
        "statement_match_ratio": round(float(match_count) / float(denominator), 6),
        "matched_gadget_statements": matched_gadget_statements,
        "matched_vulnerable_changed_statements": matched_changed_statements,
        "hits_changed_statement": hits_changed_statement,
        "heuristic_gadget_label": "vulnerable" if hits_changed_statement else "safe",
        "heuristic_label_source": "changed_statement_string_match" if hits_changed_statement else "no_changed_statement_match",
    }


def _load_snippet_context(snippet_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row = dict(_load_json(snippet_dir / "row.json"))
    status = dict(_load_json(snippet_dir / "status.json"))
    return row, status


def _discover_targets(input_root: Path, *, resume: bool) -> Tuple[List[GadgetTarget], int]:
    targets: List[GadgetTarget] = []
    skipped_existing = 0
    for snippet_dir in sorted((path for path in input_root.iterdir() if path.is_dir()), key=_numeric_name_key):
        status_path = snippet_dir / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = _load_json(status_path)
        except Exception:
            continue
        if not isinstance(status, dict) or status.get("status") != "ok":
            continue

        for gadget_dir in sorted(
            (path for path in snippet_dir.iterdir() if path.is_dir() and path.name.startswith("gadget_")),
            key=_gadget_key,
        ):
            gadget_json_path = gadget_dir / "gadget.json"
            code_gadget_path = gadget_dir / "code_gadget.c"
            if not gadget_json_path.is_file() or not code_gadget_path.is_file():
                continue
            if resume and (gadget_dir / LABEL_FILENAME).is_file():
                skipped_existing += 1
                continue
            try:
                gadget = _load_json(gadget_json_path)
            except Exception:
                continue
            if not isinstance(gadget, dict):
                continue
            targets.append(
                GadgetTarget(
                    snippet_idx=str(status.get("dataset_idx") or snippet_dir.name),
                    snippet_dir=snippet_dir,
                    gadget_dir=gadget_dir,
                    gadget_index=int(gadget.get("gadget_index", 0)),
                    api_call_name=str(gadget.get("api_call_name") or "unknown_call"),
                )
            )
    return targets, skipped_existing


def _iter_existing_label_records(input_root: Path) -> Iterator[Dict[str, Any]]:
    for snippet_dir in sorted((path for path in input_root.iterdir() if path.is_dir()), key=_numeric_name_key):
        for gadget_dir in sorted(
            (path for path in snippet_dir.iterdir() if path.is_dir() and path.name.startswith("gadget_")),
            key=_gadget_key,
        ):
            label_path = gadget_dir / LABEL_FILENAME
            if not label_path.is_file():
                continue
            try:
                payload = _load_json(label_path)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield dict(payload)


def _ensure_manifest_for_resume(input_root: Path, predictions_path: Path) -> None:
    if predictions_path.exists():
        return
    for record in _iter_existing_label_records(input_root):
        _append_jsonl(predictions_path, record)


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label exported PrimeVul code gadgets with a baseline model.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--paired-dataset-path", type=Path, default=DEFAULT_PAIRED_DATASET_PATH)
    parser.add_argument("--label-mode", choices=["heuristic_only", "model"], default="heuristic_only")
    parser.add_argument("--protocol", choices=["native", "primevul_std", "primevul_cot", "revd_cot"], default=DEFAULT_PROTOCOL)
    parser.add_argument("--language", type=str, default="c")
    parser.add_argument("--answer-prefix", type=str, default="\n")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--resume", choices=["on", "off"], default="on")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many unlabeled gadget targets after resume filtering.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many unlabeled gadget targets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    paired_dataset_path = args.paired_dataset_path.resolve()
    resume = args.resume == "on"
    protocol = _normalize_protocol(args.protocol)
    checkpoint_every = max(1, int(args.checkpoint_every))

    predictions_path = input_root / PREDICTIONS_FILENAME
    state_path = input_root / STATE_FILENAME
    if resume:
        _ensure_manifest_for_resume(input_root, predictions_path)
    else:
        if predictions_path.exists():
            predictions_path.unlink()

    paired_rows = _load_paired_rows(paired_dataset_path)
    pair_indexes = _build_pair_indexes(paired_rows)

    targets, skipped_existing = _discover_targets(input_root, resume=resume)
    offset = max(0, int(args.offset))
    if offset:
        targets = targets[offset:]
    if args.limit is not None:
        targets = targets[: max(0, int(args.limit))]

    total_targets = len(targets)
    state: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_root": str(input_root),
        "paired_dataset_path": str(paired_dataset_path),
        "label_mode": str(args.label_mode),
        "model_name": str(args.model_name),
        "protocol": str(protocol),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k) if args.top_k is not None else None,
        "resume": bool(resume),
        "eligible_total": int(total_targets + skipped_existing),
        "target_total": int(total_targets),
        "skipped_existing": int(skipped_existing),
        "processed": 0,
        "labeled": 0,
        "error": 0,
        "last_snippet_idx": None,
        "last_gadget_relpath": None,
    }

    if total_targets <= 0:
        _write_state(state_path, state)
        print(
            f"[gadget-label] nothing to do target_total=0 skipped_existing={skipped_existing} input_root={input_root}",
            flush=True,
        )
        return 0

    model: Optional[ModelRunner] = None
    if args.label_mode == "model":
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        max_devices = None
        if gpu_ids is not None:
            max_devices = min(len(gpu_ids), visible_gpus) if visible_gpus > 0 else None
        elif args.gpus is not None and visible_gpus > 0:
            max_devices = min(max(1, int(args.gpus)), visible_gpus)

        model = ModelRunner()
        try:
            model.login_hf()
        except Exception as exc:
            print(f"[WARN] Hugging Face login failed; continuing without auth: {exc}", flush=True)
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

    progress = None
    if tqdm is not None:
        progress = tqdm(total=total_targets, desc="gadget-label", unit="gadget")
        progress.set_postfix(
            labeled=0,
            skipped_existing=skipped_existing,
            error=0,
            snippet="-",
            gadget="-",
            call="-",
        )

    snippet_cache: Dict[Path, Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = {}

    try:
        for index, target in enumerate(targets, start=1):
            row: Dict[str, Any]
            status: Dict[str, Any]
            paired_ground_truth: Dict[str, Any]
            if target.snippet_dir not in snippet_cache:
                row, status = _load_snippet_context(target.snippet_dir)
                paired_ground_truth = _build_paired_ground_truth(row, pair_indexes)
                _write_json(target.snippet_dir / PAIRED_GROUND_TRUTH_FILENAME, paired_ground_truth)
                snippet_cache[target.snippet_dir] = (row, status, paired_ground_truth)
            row, status, paired_ground_truth = snippet_cache[target.snippet_dir]

            gadget_json = dict(_load_json(target.gadget_dir / "gadget.json"))
            code_gadget = (target.gadget_dir / "code_gadget.c").read_text(encoding="utf-8")
            overlap = _compute_patch_overlap(gadget_json.get("line_sequence") or [], paired_ground_truth)
            statement_support = _compute_statement_match_support(code_gadget, paired_ground_truth)

            if progress is not None:
                progress.set_postfix(
                    labeled=int(state["labeled"]),
                    skipped_existing=skipped_existing,
                    error=int(state["error"]),
                    snippet=target.snippet_idx,
                    gadget=target.gadget_index,
                    call=target.api_call_name,
                )

            state["last_snippet_idx"] = str(target.snippet_idx)
            state["last_gadget_relpath"] = str(target.gadget_dir.relative_to(input_root))

            completion = ""
            try:
                if args.label_mode == "model":
                    assert model is not None
                    result = _run_generation(
                        model=model,
                        code=code_gadget,
                        protocol=protocol,
                        instruction=None,
                        language=args.language,
                        answer_prefix=args.answer_prefix,
                        max_new_tokens=max(1, int(args.max_new_tokens)),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=args.top_k,
                        do_sample=False,
                    )
                    completion = str(result.get("generated_completion", "") or "")
                    pred_label = _parse_prediction_label(completion)
                    if pred_label is None:
                        raise ValueError("Could not parse model output into VULNERABLE or SAFE.")
                    pred_label_text = "VULNERABLE" if int(pred_label) == 1 else "SAFE"
                    pred_text = completion
                    model_name = str(model.model_name)
                    protocol_name = str(protocol)
                else:
                    pred_label_text = (
                        "VULNERABLE"
                        if statement_support.get("heuristic_gadget_label") == "vulnerable"
                        else "SAFE"
                    )
                    pred_text = None
                    model_name = None
                    protocol_name = None

                label_payload = {
                    "pred_label": pred_label_text,
                    "pred_text": pred_text,
                    "label_mode": str(args.label_mode),
                    "model_name": model_name,
                    "protocol": protocol_name,
                    "max_new_tokens": int(args.max_new_tokens) if args.label_mode == "model" else None,
                    "temperature": float(args.temperature) if args.label_mode == "model" else None,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "snippet_idx": str(target.snippet_idx),
                    "row_index": _safe_int(status.get("row_index")),
                    "gold_snippet_label": _normalize_target(row.get("target")),
                    "gadget_index": int(gadget_json.get("gadget_index", target.gadget_index)),
                    "api_call_name": str(gadget_json.get("api_call_name") or target.api_call_name),
                    "call_line": _safe_int(gadget_json.get("call_line")),
                    **overlap,
                    **statement_support,
                }
                _write_json(target.gadget_dir / LABEL_FILENAME, label_payload)
                error_path = target.gadget_dir / LABEL_ERROR_FILENAME
                if error_path.exists():
                    error_path.unlink()
                _append_jsonl(predictions_path, label_payload)
                state["labeled"] = int(state["labeled"]) + 1
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _recover_from_cuda_oom(model)
                error_payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "snippet_idx": str(target.snippet_idx),
                    "row_index": _safe_int(status.get("row_index")),
                    "gadget_index": int(gadget_json.get("gadget_index", target.gadget_index)),
                    "api_call_name": str(gadget_json.get("api_call_name") or target.api_call_name),
                    "call_line": _safe_int(gadget_json.get("call_line")),
                    "generated_completion": completion,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "paired_available": bool(overlap.get("paired_available")),
                    "patch_support_label": overlap.get("patch_support_label"),
                    "heuristic_gadget_label": statement_support.get("heuristic_gadget_label"),
                    "heuristic_label_source": statement_support.get("heuristic_label_source"),
                }
                _write_json(target.gadget_dir / LABEL_ERROR_FILENAME, error_payload)
                state["error"] = int(state["error"]) + 1

            state["processed"] = int(state["processed"]) + 1
            if progress is not None:
                progress.update(1)
            if int(state["processed"]) % checkpoint_every == 0:
                _write_state(state_path, state)
                if progress is None:
                    print(
                        f"[gadget-label] processed={state['processed']} labeled={state['labeled']} "
                        f"skipped_existing={skipped_existing} error={state['error']}",
                        flush=True,
                    )
    finally:
        if progress is not None:
            progress.close()
        if model is not None:
            model.free()

    _write_state(state_path, state)
    print(
        f"[gadget-label] done processed={state['processed']} labeled={state['labeled']} "
        f"skipped_existing={skipped_existing} error={state['error']} input_root={input_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
