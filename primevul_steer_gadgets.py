#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency at runtime
    tqdm = None

from models import ModelRunner
from paths import model_dir_name, resolve_artifact_path
from primevul_eval import (
    _build_generation_prompt,
    _compute_metrics,
    _default_instruction_for_protocol,
    _is_cuda_oom,
    _normalize_protocol,
    _parse_gpu_ids,
    _parse_prediction_label,
    _recover_from_cuda_oom,
)
from steering import SteeringConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "Source" / "primevul_test_vulnerable_snippets"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_PROTOCOL = "revd_cot"
DEFAULT_OUTPUT_DIR = Path("primevul_gadget_steered")
LABEL_FILENAME = "gadget_label.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
STATE_FILENAME = "state.json"
RUN_CONFIG_FILENAME = "run_config.json"
SUMMARY_FILENAME = "summary.json"
SUMMARY_PARTIAL_FILENAME = "summary.partial.json"


@dataclass(frozen=True)
class GadgetTarget:
    row_index: int
    snippet_idx: str
    snippet_dir: Path
    gadget_dir: Path
    gadget_relpath: str
    gadget_index: int
    api_call_name: str
    needed_runs: Tuple[int, ...]


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


def _numeric_name_key(path: Path) -> Tuple[int, str]:
    parsed = _safe_int(path.name)
    if parsed is None:
        return (1, path.name)
    return (0, f"{parsed:020d}")


def _gadget_key(path: Path) -> Tuple[int, str]:
    prefix = path.name.split("__", 1)[0]
    parsed = _safe_int(prefix.replace("gadget_", "", 1))
    if parsed is None:
        return (1, path.name)
    return (0, f"{parsed:06d}")


def _iter_existing_predictions(predictions_path: Path) -> Iterator[Dict[str, Any]]:
    if not predictions_path.exists():
        return
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield dict(payload)


def _load_existing_predictions(
    predictions_path: Path,
) -> Tuple[List[Dict[str, Any]], set[Tuple[str, int]]]:
    records: List[Dict[str, Any]] = []
    completed: set[Tuple[str, int]] = set()
    for record in _iter_existing_predictions(predictions_path):
        records.append(record)
        relpath = str(record.get("gadget_relpath") or "")
        sample_run = int(record.get("sample_run", -1))
        if relpath:
            completed.add((relpath, sample_run))
    return records, completed


def _load_snapshot_relpaths(snapshot_path: Path) -> set[str]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    relpaths: list[str] = []
    if isinstance(payload, dict):
        values = payload.get("gadget_relpaths")
        if isinstance(values, list):
            relpaths = [str(value) for value in values]
    elif isinstance(payload, list):
        relpaths = [str(value) for value in payload]
    if not relpaths:
        raise ValueError(f"Snapshot manifest has no gadget_relpaths: {snapshot_path}")
    return {value for value in relpaths if value}


def _normalize_label_filter(value: str) -> str:
    text = str(value or "vulnerable").strip().lower()
    if text not in {"all", "vulnerable", "safe"}:
        raise ValueError(f"Unsupported --label-filter value: {value!r}")
    return text


def _matches_label_filter(label_payload: Dict[str, Any], label_filter: str) -> bool:
    if label_filter == "all":
        return True
    pred_label = str(label_payload.get("pred_label") or "").strip().upper()
    if label_filter == "vulnerable":
        return pred_label == "VULNERABLE"
    return pred_label == "SAFE"


def _discover_targets(
    *,
    input_root: Path,
    label_filter: str,
    completed_records: set[Tuple[str, int]],
    snapshot_relpaths: Optional[set[str]],
    offset: int,
    limit: Optional[int],
    samples_per_gadget: int,
) -> Tuple[List[GadgetTarget], int]:
    targets: List[GadgetTarget] = []
    skipped_completed = 0
    for row_index, snippet_dir in enumerate(
        sorted((path for path in input_root.iterdir() if path.is_dir()), key=_numeric_name_key)
    ):
        status_path = snippet_dir / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = _load_json(status_path)
        except Exception:
            continue
        if not isinstance(status, dict) or status.get("status") != "ok":
            continue

        snippet_idx = str(status.get("dataset_idx") or snippet_dir.name)
        for gadget_dir in sorted(
            (path for path in snippet_dir.iterdir() if path.is_dir() and path.name.startswith("gadget_")),
            key=_gadget_key,
        ):
            label_path = gadget_dir / LABEL_FILENAME
            gadget_json_path = gadget_dir / "gadget.json"
            if not gadget_json_path.is_file():
                continue
            try:
                gadget_payload = _load_json(gadget_json_path)
                label_payload = _load_json(label_path) if label_path.is_file() else {}
            except Exception:
                continue
            if not isinstance(gadget_payload, dict):
                continue
            if not isinstance(label_payload, dict):
                label_payload = {}
            if label_filter != "all" and not label_path.is_file():
                continue
            if not _matches_label_filter(label_payload, label_filter):
                continue
            relpath = str(gadget_dir.relative_to(input_root))
            if snapshot_relpaths is not None and relpath not in snapshot_relpaths:
                continue
            needed_runs = [
                sample_run
                for sample_run in range(samples_per_gadget)
                if (relpath, sample_run) not in completed_records
            ]
            if not needed_runs:
                skipped_completed += samples_per_gadget
                continue
            skipped_completed += samples_per_gadget - len(needed_runs)
            targets.append(
                GadgetTarget(
                    row_index=row_index,
                    snippet_idx=snippet_idx,
                    snippet_dir=snippet_dir,
                    gadget_dir=gadget_dir,
                    gadget_relpath=relpath,
                    gadget_index=int(gadget_payload.get("gadget_index", 0)),
                    api_call_name=str(gadget_payload.get("api_call_name") or "unknown_call"),
                    needed_runs=tuple(int(sample_run) for sample_run in needed_runs),
                )
            )

    offset = max(0, int(offset))
    if offset:
        targets = targets[offset:]
    if limit is not None:
        targets = targets[: max(0, int(limit))]
    return targets, skipped_completed


def _snippet_code_from_dir(snippet_dir: Path, row: Dict[str, Any]) -> str:
    snippet_path = snippet_dir / "snippet.c"
    if snippet_path.is_file():
        return snippet_path.read_text(encoding="utf-8")
    for key in ("func", "func_before", "code"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"Could not locate snippet text for {snippet_dir}")


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    state = dict(payload)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(path, state)


def _write_summary(
    *,
    output_root: Path,
    run_manifest: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    variant: str,
    filename: str,
) -> Dict[str, Any]:
    summary = {
        "run": run_manifest,
        "by_variant": {
            str(variant): _compute_metrics(records),
        },
    }
    _write_json(output_root / filename, summary)
    return summary


def _default_run_name(protocol: str, steer_last_n_layers: int, topk: int) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"qwen25_gadget_steered_{protocol}_s2_l{int(steer_last_n_layers)}_k{int(topk)}_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-gadget steered PrimeVul experiments from exported gadget artifacts.")
    parser.add_argument("--variant", choices=["baseline", "steered"], default="steered")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--label-filter", choices=["all", "vulnerable", "safe"], default="vulnerable")
    parser.add_argument("--language", type=str, default="c")
    parser.add_argument("--protocol", choices=["native", "primevul_std", "primevul_cot", "revd_cot"], default=DEFAULT_PROTOCOL)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--answer-prefix", type=str, default="\n")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--samples-per-gadget", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", choices=["on", "off"], default="off")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--beta-post", type=float, default=0.5)
    parser.add_argument("--beta-bias", type=float, default=0.0)
    parser.add_argument("--steer-last-n-layers", type=int, default=8)
    parser.add_argument("--head-subset-mode", choices=["none", "auto"], default="auto")
    parser.add_argument("--head-subset-topk-per-layer", type=int, default=4)
    parser.add_argument("--head-subset-calib-runs", type=int, default=1)
    parser.add_argument("--head-subset-calib-max-new-tokens", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    snapshot_path = args.snapshot_path.resolve() if args.snapshot_path is not None else None
    snapshot_relpaths = _load_snapshot_relpaths(snapshot_path) if snapshot_path is not None else None
    label_filter = _normalize_label_filter(args.label_filter)
    protocol = _normalize_protocol(args.protocol)
    checkpoint_every = max(1, int(args.checkpoint_every))
    samples_per_gadget = max(1, int(args.samples_per_gadget))

    run_name = args.run_name or _default_run_name(
        protocol=protocol,
        steer_last_n_layers=max(1, int(args.steer_last_n_layers)),
        topk=max(1, int(args.head_subset_topk_per_layer)),
    )
    output_root = (
        resolve_artifact_path(PROJECT_ROOT, args.output_dir)
        / model_dir_name(args.model_name)
        / run_name
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / PREDICTIONS_FILENAME
    state_path = output_root / STATE_FILENAME

    existing_records: List[Dict[str, Any]] = []
    completed_records: set[Tuple[str, int]] = set()
    if args.resume == "on":
        existing_records, completed_records = _load_existing_predictions(predictions_path)
    elif predictions_path.exists():
        predictions_path.unlink()

    targets, skipped_completed = _discover_targets(
        input_root=input_root,
        label_filter=label_filter,
        completed_records=completed_records,
        snapshot_relpaths=snapshot_relpaths,
        offset=int(args.offset),
        limit=args.limit,
        samples_per_gadget=samples_per_gadget,
    )
    pending_runs_total = sum(len(target.needed_runs) for target in targets)

    run_manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": str(args.variant),
        "input_root": str(input_root),
        "snapshot_path": None if snapshot_path is None else str(snapshot_path),
        "snapshot_total": None if snapshot_relpaths is None else int(len(snapshot_relpaths)),
        "label_filter": label_filter,
        "protocol": protocol,
        "instruction": args.instruction or _default_instruction_for_protocol(protocol),
        "language": str(args.language),
        "answer_prefix": str(args.answer_prefix),
        "model_name": str(args.model_name),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": None if args.top_k is None else int(args.top_k),
        "samples_per_gadget": int(samples_per_gadget),
        "seed": int(args.seed),
        "offset": int(args.offset),
        "limit": None if args.limit is None else int(args.limit),
        "resume": bool(args.resume == "on"),
        "existing_predictions": int(len(existing_records)),
        "target_total": int(pending_runs_total),
        "skipped_completed": int(skipped_completed),
        "prior": "code_gadget" if args.variant == "steered" else None,
        "enabled_levels": [2] if args.variant == "steered" else [],
        "n_bins": int(args.n_bins),
        "beta_bias": float(args.beta_bias),
        "beta_post": float(args.beta_post),
        "steer_last_n_layers": int(args.steer_last_n_layers),
        "head_subset_mode": str(args.head_subset_mode),
        "head_subset_topk_per_layer": int(args.head_subset_topk_per_layer),
        "head_subset_calib_runs": int(args.head_subset_calib_runs),
        "head_subset_calib_max_new_tokens": int(args.head_subset_calib_max_new_tokens),
        "checkpoint_every": int(checkpoint_every),
    }
    _write_json(output_root / RUN_CONFIG_FILENAME, run_manifest)

    per_variant: Dict[str, List[Dict[str, Any]]] = {"steered": list(existing_records)}
    state: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "processed": 0,
        "completed_existing": int(len(existing_records)),
        "skipped_completed": int(skipped_completed),
        "target_total": int(pending_runs_total),
        "error": 0,
        "last_snippet_idx": None,
        "last_gadget_relpath": None,
    }
    if pending_runs_total <= 0:
        _write_state(state_path, state)
        _write_summary(
            output_root=output_root,
            run_manifest=run_manifest,
            records=per_variant["steered"],
            variant=str(args.variant),
            filename=SUMMARY_FILENAME,
        )
        print(
            f"[gadget-steer] nothing to do target_total=0 existing={len(existing_records)} output_root={output_root}",
            flush=True,
        )
        return 0

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
    steering_config: Optional[SteeringConfig] = None
    if args.variant == "steered":
        steering_config = SteeringConfig(
            enabled_levels=[2],
            prior="code_gadget",
            n_bins=max(1, int(args.n_bins)),
            beta_bias=float(args.beta_bias),
            beta_post=float(args.beta_post),
            decode_only=True,
            only_first_decode_step=False,
            split_prefill=True,
            recency_mix=True,
            recency_rho=0.2,
            recency_window=64,
            recency_apply_after_prompt=True,
            recency_scope="prefer_generated",
            head_subset_mode=str(args.head_subset_mode),
            head_subset_topk_per_layer=max(1, int(args.head_subset_topk_per_layer)),
            head_subset_calib_runs=max(1, int(args.head_subset_calib_runs)),
            head_subset_calib_max_new_tokens=max(1, int(args.head_subset_calib_max_new_tokens)),
            head_subset_calib_first_decode_only=True,
            head_subset_auto_save=output_root / "head_masks" / "{snippet}.json",
        )
        model.set_steering_config(steering_config)
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

    if model.model is None:
        raise RuntimeError("Model failed to build.")
    if steering_config is not None:
        num_layers = int(getattr(model.model.config, "num_hidden_layers"))
        steer_last_n_layers = max(1, min(int(args.steer_last_n_layers), num_layers))
        steering_config.steer_layer_start = max(0, num_layers - steer_last_n_layers)
        steering_config.steer_layer_end = num_layers - 1

    instruction_text = args.instruction or _default_instruction_for_protocol(protocol)
    snippet_cache: Dict[Path, Tuple[Dict[str, Any], Dict[str, Any], str, str]] = {}

    progress = None
    if tqdm is not None:
        progress = tqdm(total=pending_runs_total, desc="gadget-steer", unit="run")
        progress.set_postfix(
            processed=0,
            error=0,
            snippet="-",
            gadget="-",
            call="-",
        )

    try:
        with predictions_path.open("a" if args.resume == "on" else "w", encoding="utf-8") as sink:
            for target_index, target in enumerate(targets):
                if target.snippet_dir not in snippet_cache:
                    row = dict(_load_json(target.snippet_dir / "row.json"))
                    status = dict(_load_json(target.snippet_dir / "status.json"))
                    snippet_code = _snippet_code_from_dir(target.snippet_dir, row)
                    prompt = _build_generation_prompt(
                        model=model,
                        protocol=protocol,
                        code=snippet_code,
                        language=args.language,
                        instruction_override=args.instruction,
                        answer_prefix=args.answer_prefix,
                    )
                    snippet_cache[target.snippet_dir] = (row, status, snippet_code, prompt)
                row, status, snippet_code, prompt = snippet_cache[target.snippet_dir]

                gadget_payload = dict(_load_json(target.gadget_dir / "gadget.json"))
                label_path = target.gadget_dir / LABEL_FILENAME
                gadget_label = dict(_load_json(label_path)) if label_path.is_file() else {}

                if steering_config is not None:
                    steering_config.code_gadget_artifact_path = target.gadget_dir
                    steering_config.head_mask_inline = None
                    steering_config.head_subset_selected_heads = {}
                    steering_config.head_subset_calibration = {}

                if progress is not None:
                    progress.set_postfix(
                        processed=int(state["processed"]),
                        error=int(state["error"]),
                        snippet=target.snippet_idx,
                        gadget=target.gadget_index,
                        call=target.api_call_name,
                    )

                state["last_snippet_idx"] = str(target.snippet_idx)
                state["last_gadget_relpath"] = str(target.gadget_relpath)

                for sample_run in target.needed_runs:
                    sample_seed = int(args.seed) + int(target_index) * 1000 + int(sample_run)
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)
                    started = time.perf_counter()
                    completion = ""
                    calibration_info: Optional[Dict[str, Any]] = None
                    try:
                        if steering_config is not None and steering_config.head_subset_mode == "auto":
                            calibration_info = model.calibrate_head_subset(
                                code_snippet=snippet_code,
                                instruction=instruction_text,
                                language=args.language,
                                vocab_tokens=[],
                                snippet_name=f"{target.snippet_idx}-g{target.gadget_index}",
                                steering_code_snippet=snippet_code,
                                prompt_text_override=prompt,
                            )
                        model._current_code_snippet = snippet_code
                        model._current_vocab_tokens = []
                        try:
                            result = model._generate_with_attn(
                                prompt,
                                overrides={
                                    "max_new_tokens": max(1, int(args.max_new_tokens)),
                                    "temperature": float(args.temperature),
                                    "top_p": float(args.top_p),
                                    "top_k": args.top_k,
                                    "do_sample": False,
                                    "record_layers": False,
                                    "record_attention": False,
                                },
                            )
                        finally:
                            model._current_code_snippet = ""
                            model._current_vocab_tokens = []
                        completion = str(result.get("generated_completion", "") or "")
                        pred_label = _parse_prediction_label(completion)
                        record = {
                            "sample_index": int(target.row_index),
                            "sample_id": str(target.snippet_idx),
                            "sample_run": int(sample_run),
                            "sample_seed": int(sample_seed),
                            "variant": str(args.variant),
                            "gadget_relpath": str(target.gadget_relpath),
                            "gadget_index": int(gadget_payload.get("gadget_index", target.gadget_index)),
                            "api_call_name": str(gadget_payload.get("api_call_name") or target.api_call_name),
                            "call_line": _safe_int(gadget_payload.get("call_line")),
                            "gadget_label_pred": str(gadget_label.get("pred_label") or ""),
                            "patch_support_label": gadget_label.get("patch_support_label"),
                            "gold_label": 1 if int(row.get("target", 0)) != 0 else 0,
                            "pred_label": None if pred_label is None else int(pred_label),
                            "is_correct": bool(pred_label == (1 if int(row.get("target", 0)) != 0 else 0)) if pred_label is not None else False,
                            "elapsed_sec": round(time.perf_counter() - started, 4),
                            "generated_completion": completion,
                            "steering_debug": result.get("steering_debug") if args.variant == "steered" else None,
                            "head_subset_calibration": calibration_info,
                        }
                    except Exception as exc:
                        if _is_cuda_oom(exc):
                            _recover_from_cuda_oom(model)
                        state["error"] = int(state["error"]) + 1
                        record = {
                            "sample_index": int(target.row_index),
                            "sample_id": str(target.snippet_idx),
                            "sample_run": int(sample_run),
                            "sample_seed": int(sample_seed),
                            "variant": str(args.variant),
                            "gadget_relpath": str(target.gadget_relpath),
                            "gadget_index": int(gadget_payload.get("gadget_index", target.gadget_index)),
                            "api_call_name": str(gadget_payload.get("api_call_name") or target.api_call_name),
                            "call_line": _safe_int(gadget_payload.get("call_line")),
                            "gadget_label_pred": str(gadget_label.get("pred_label") or ""),
                            "patch_support_label": gadget_label.get("patch_support_label"),
                            "gold_label": 1 if int(row.get("target", 0)) != 0 else 0,
                            "pred_label": None,
                            "is_correct": False,
                            "elapsed_sec": round(time.perf_counter() - started, 4),
                            "generated_completion": completion,
                            "steering_debug": None,
                            "head_subset_calibration": calibration_info,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }

                    per_variant["steered"].append(record)
                    completed_records.add((target.gadget_relpath, sample_run))
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                    state["processed"] = int(state["processed"]) + 1

                    if progress is not None:
                        progress.update(1)
                    if int(state["processed"]) % checkpoint_every == 0:
                        _write_state(state_path, state)
                        _write_summary(
                            output_root=output_root,
                            run_manifest=run_manifest,
                            records=per_variant["steered"],
                            variant=str(args.variant),
                            filename=SUMMARY_PARTIAL_FILENAME,
                        )
    finally:
        if progress is not None:
            progress.close()
        model.free()

    _write_state(state_path, state)
    _write_summary(
        output_root=output_root,
        run_manifest=run_manifest,
        records=per_variant["steered"],
        variant=str(args.variant),
        filename=SUMMARY_FILENAME,
    )
    print(
        f"[gadget-steer] done processed={state['processed']} existing={len(existing_records)} "
        f"output_root={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
