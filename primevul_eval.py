#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
from steering import SteeringConfig


DEFAULT_JOERN_CLI_DIR = Path.home() / "bin" / "joern" / "joern-cli"
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


def _variant_list(args: argparse.Namespace) -> List[str]:
    variant = str(args.variant).lower()
    if variant == "both":
        return ["baseline", "steered"]
    return [variant]


def _build_steering_config(args: argparse.Namespace, project_root: Path) -> Optional[SteeringConfig]:
    if not bool(args.steer):
        return None

    joern_cache_dir = resolve_artifact_path(project_root, args.joern_cache_dir)
    head_mask_path = (
        resolve_artifact_path(project_root, args.head_mask_path)
        if args.head_mask_path is not None
        else None
    )
    head_subset_auto_save = (
        resolve_artifact_path(project_root, args.head_subset_auto_save)
        if args.head_subset_auto_save is not None
        else None
    )
    return SteeringConfig(
        enabled_levels=[2],
        prior=args.prior,
        n_bins=max(1, int(args.n_bins)),
        binning="equal_count",
        beta_bias=float(args.beta_bias),
        beta_post=float(args.beta_post),
        lambda_attn=float(args.lambda_attn),
        lambda_mlp=float(args.lambda_mlp),
        alpha_k=float(args.alpha_k),
        alpha_v=float(args.alpha_v),
        bias_cap=None if args.bias_cap is None else float(args.bias_cap),
        gamma_min=float(args.gamma_min),
        gamma_max=float(args.gamma_max),
        eta_min=float(args.eta_min),
        eta_max=float(args.eta_max),
        recency_mix=(args.recency_mix == "on"),
        recency_rho=float(args.recency_rho),
        recency_window=max(1, int(args.recency_window)),
        recency_apply_after_prompt=(args.recency_apply_after_prompt == "on"),
        recency_scope=str(args.recency_scope),
        joern_cli_dir=args.joern_cli_dir,
        joern_cache_dir=joern_cache_dir,
        joern_direction=str(args.joern_direction),
        joern_slice_depth=max(1, int(args.joern_slice_depth)),
        joern_parallelism=max(1, int(args.joern_parallelism)),
        joern_timeout_sec=max(1, int(args.joern_timeout_sec)),
        joern_include_control=(args.joern_include_control == "on"),
        joern_max_hops=args.joern_max_hops,
        head_subset_mode=str(args.head_subset_mode),
        head_mask_path=head_mask_path,
        head_mask_apply_to=str(args.head_mask_apply_to),
        head_mask_debug=bool(args.head_mask_debug),
        head_subset_topk_per_layer=max(1, int(args.head_subset_topk_per_layer)),
        head_subset_calib_runs=max(1, int(args.head_subset_calib_runs)),
        head_subset_calib_max_new_tokens=max(1, int(args.head_subset_calib_max_new_tokens)),
        head_subset_calib_first_decode_only=(args.head_subset_calib_first_decode_only == "on"),
        head_subset_auto_save=head_subset_auto_save,
        collect_head_stats=(args.collect_head_stats == "on"),
        collect_head_stats_first_decode_only=(args.collect_head_stats_first_decode_only == "on"),
    )


def _apply_last_n_layers(model: ModelRunner, cfg: Optional[SteeringConfig], last_n: Optional[int]) -> None:
    if cfg is None or last_n is None or model.model is None:
        return
    num_layers = int(getattr(getattr(model.model, "config", None), "num_hidden_layers", 0))
    if num_layers <= 0:
        return
    n = max(1, min(int(last_n), num_layers))
    cfg.steer_layer_start = max(0, num_layers - n)
    cfg.steer_layer_end = max(0, num_layers - 1)


def _extract_joern_summary(model: ModelRunner) -> Optional[Dict[str, Any]]:
    runtime = getattr(model, "_steering_runtime", None) or getattr(model, "_last_steering_runtime", None)
    manager = getattr(runtime, "manager", None)
    prior = getattr(manager, "prior_provider", None)
    payload = getattr(prior, "joern_payload", None)
    if not isinstance(payload, dict):
        return None

    aggregate = payload.get("aggregate_line_scores") or {}
    top_lines = sorted(
        ((int(line_no), float(score)) for line_no, score in aggregate.items()),
        key=lambda item: (-item[1], item[0]),
    )[:12]
    selected_code_gadget = payload.get("selected_code_gadget")
    return {
        "direction": payload.get("direction"),
        "num_variable_slices": int(payload.get("num_variable_slices", 0)),
        "num_selected_variable_slices": int(payload.get("num_selected_variable_slices", 0)),
        "num_graph_nodes": int(payload.get("num_graph_nodes", 0)),
        "num_graph_edges": int(payload.get("num_graph_edges", 0)),
        "top_lines": top_lines,
        "sink_filter": payload.get("sink_filter"),
        "selected_code_gadget": dict(selected_code_gadget) if isinstance(selected_code_gadget, dict) else None,
        "meta": dict(payload.get("meta") or {}),
    }


def _calibrate_head_subset_for_protocol(
    *,
    model: ModelRunner,
    base_cfg: Optional[SteeringConfig],
    code: str,
    sample_id: str,
    seed_base: int,
    protocol: str,
    instruction: Optional[str],
    language: str,
    answer_prefix: str,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
) -> Dict[str, Any]:
    if base_cfg is None:
        return {"mode": "none", "active_total": 0, "layers_with_heads": 0}
    if str(base_cfg.head_subset_mode).lower() != "auto":
        return {"mode": str(base_cfg.head_subset_mode), "active_total": 0, "layers_with_heads": 0}
    if not any(level in base_cfg.enabled_levels for level in (1, 2)):
        raise RuntimeError("Head subset auto mode requires active attention steering.")

    from steering.backends.common import compute_default_cutoffs

    runs = max(1, int(base_cfg.head_subset_calib_runs))
    calib_max_new = max(1, int(base_cfg.head_subset_calib_max_new_tokens))
    topk = max(1, int(base_cfg.head_subset_topk_per_layer))

    calib_cfg = copy.deepcopy(base_cfg)
    calib_cfg.head_subset_mode = "none"
    calib_cfg.head_mask_path = None
    calib_cfg.head_mask_inline = None
    calib_cfg.head_subset_selected_heads = {}
    calib_cfg.head_subset_calibration = {}
    calib_cfg.collect_head_stats = True
    calib_cfg.collect_head_stats_first_decode_only = bool(base_cfg.head_subset_calib_first_decode_only)
    calib_cfg.beta_bias = 0.0
    calib_cfg.beta_post = 0.0
    calib_cfg.schedule = {}

    agg_sum: Optional[np.ndarray] = None
    agg_count: Optional[np.ndarray] = None
    valid_runs = 0

    for run_idx in range(runs):
        _seed_all(int(seed_base) + int(run_idx))
        _set_active_steering(model, calib_cfg)
        result = _run_generation(
            model=model,
            code=code,
            protocol=protocol,
            instruction=instruction,
            language=language,
            answer_prefix=answer_prefix,
            max_new_tokens=calib_max_new,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=top_k,
            do_sample=True,
        )
        hs = (result.get("steering_debug") or {}).get("head_stats")
        if not hs:
            continue
        run_sum = np.asarray(hs.get("sum", []), dtype=np.float64)
        run_count = np.asarray(hs.get("count", []), dtype=np.float64)
        if run_sum.ndim != 2 or run_count.ndim != 2:
            continue
        if agg_sum is None:
            agg_sum = np.zeros_like(run_sum, dtype=np.float64)
            agg_count = np.zeros_like(run_count, dtype=np.float64)
        if agg_sum.shape != run_sum.shape:
            raise RuntimeError(f"Head stats shape mismatch during calibration: {agg_sum.shape} vs {run_sum.shape}")
        agg_sum += run_sum
        agg_count += run_count
        valid_runs += 1

    if agg_sum is None or agg_count is None or valid_runs == 0:
        raise RuntimeError("Step-4 auto calibration failed: no valid head stats collected.")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.divide(agg_sum, np.maximum(agg_count, 1.0), where=np.maximum(agg_count, 1.0) > 0)

    num_layers, num_heads = mean.shape
    cutoffs = compute_default_cutoffs(num_layers)
    layer_start = cutoffs.l12_start if base_cfg.steer_layer_start is None else int(base_cfg.steer_layer_start)
    layer_end = cutoffs.l12_end if base_cfg.steer_layer_end is None else int(base_cfg.steer_layer_end)
    layer_start = max(0, min(num_layers - 1, layer_start))
    layer_end = max(layer_start, min(num_layers - 1, layer_end))
    k = max(1, min(topk, num_heads))

    mask = np.zeros((num_layers, num_heads), dtype=bool)
    active_heads: Dict[str, List[int]] = {}
    for li in range(layer_start, layer_end + 1):
        candidates = np.where(agg_count[li] > 0)[0]
        if candidates.size == 0:
            continue
        scores = mean[li, candidates]
        order = np.argsort(-scores)
        selected = candidates[order[: min(k, candidates.size)]].astype(int).tolist()
        if not selected:
            continue
        mask[li, selected] = True
        active_heads[str(li)] = selected

    active_total = int(mask.sum())
    if active_total <= 0:
        raise RuntimeError("Step-4 auto calibration selected zero heads; aborting.")

    base_cfg.head_mask_inline = {
        "meta": {
            "model_name": model.model_name,
            "mode": "auto",
            "snippet": sample_id,
        },
        "mask": mask.astype(np.int32).tolist(),
        "active_heads": active_heads,
    }
    base_cfg.head_subset_selected_heads = active_heads
    base_cfg.head_subset_calibration = {
        "mode": "auto",
        "calib_runs_requested": int(runs),
        "calib_runs_valid": int(valid_runs),
        "calib_max_new_tokens": int(calib_max_new),
        "collect_first_decode_only": bool(base_cfg.head_subset_calib_first_decode_only),
        "layer_start": int(layer_start),
        "layer_end": int(layer_end),
        "topk_per_layer": int(k),
        "active_total": int(active_total),
        "layers_with_heads": int(len(active_heads)),
        "auto_save_path": None,
    }

    if base_cfg.head_subset_auto_save:
        save_path = model._resolve_auto_mask_save_path(
            base_cfg.head_subset_auto_save,
            snippet_name=sample_id,
            topk=k,
        )
        payload = {
            "meta": {
                "snippet": sample_id,
                "model_name": model.model_name,
                "mode": "auto",
                "calib_runs_requested": int(runs),
                "calib_runs_valid": int(valid_runs),
                "calib_max_new_tokens": int(calib_max_new),
                "collect_first_decode_only": bool(base_cfg.head_subset_calib_first_decode_only),
                "layer_start": int(layer_start),
                "layer_end": int(layer_end),
                "topk_per_layer": int(k),
                "protocol": str(protocol),
            },
            "shape": [int(num_layers), int(num_heads)],
            "mask": mask.astype(np.int32).tolist(),
            "active_heads": active_heads,
            "agree_mean": mean.tolist(),
            "agree_sum": agg_sum.tolist(),
            "agree_count": agg_count.astype(np.int64).tolist(),
        }
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        base_cfg.head_subset_calibration["auto_save_path"] = str(save_path)
        print(f"[HeadSubset-Auto] saved mask -> {save_path}")

    noop_ok: Optional[bool] = None
    beta_zero = abs(float(base_cfg.beta_bias)) <= 1e-12 and abs(float(base_cfg.beta_post)) <= 1e-12
    if beta_zero:
        cfg_none = copy.deepcopy(base_cfg)
        cfg_none.head_subset_mode = "none"
        cfg_none.head_mask_inline = None
        cfg_none.collect_head_stats = False

        cfg_auto = copy.deepcopy(base_cfg)
        cfg_auto.head_subset_mode = "auto"
        cfg_auto.collect_head_stats = False

        compare_seed = int(seed_base) + 10_000
        _seed_all(compare_seed)
        _set_active_steering(model, cfg_none)
        out_none = _run_generation(
            model=model,
            code=code,
            protocol=protocol,
            instruction=instruction,
            language=language,
            answer_prefix=answer_prefix,
            max_new_tokens=min(calib_max_new, 16),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=top_k,
            do_sample=False,
        )

        _seed_all(compare_seed)
        _set_active_steering(model, cfg_auto)
        out_auto = _run_generation(
            model=model,
            code=code,
            protocol=protocol,
            instruction=instruction,
            language=language,
            answer_prefix=answer_prefix,
            max_new_tokens=min(calib_max_new, 16),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=top_k,
            do_sample=False,
        )
        noop_ok = out_none.get("token_ids_all") == out_auto.get("token_ids_all")
        if not noop_ok:
            raise RuntimeError("Step-4 smoke check failed: auto head subset with beta=0 changed outputs.")
        base_cfg.head_subset_calibration["beta_zero_noop_ok"] = bool(noop_ok)

    return dict(base_cfg.head_subset_calibration)


def _set_active_steering(model: ModelRunner, cfg: Optional[SteeringConfig]) -> None:
    model.steering_config = copy.deepcopy(cfg) if cfg is not None else None
    model._steering_runtime = None
    model._last_steering_runtime = None


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
    if isinstance(exc, torch.OutOfMemoryError):
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
        "joern_summary": None,
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
        description="Run baseline vulnerability classification on PrimeVul-style datasets with future code_gadget steering hooks preserved."
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
    parser.add_argument("--variant", choices=["baseline", "steered", "both"], default="baseline")
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

    parser.add_argument("--steer", action="store_true", help="Enable steering for the steered variant.")
    parser.add_argument(
        "--prior",
        choices=["code_gadget"],
        default="code_gadget",
    )
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--beta-bias", type=float, default=0.0)
    parser.add_argument("--beta-post", type=float, default=0.0)
    parser.add_argument("--lambda-attn", type=float, default=1.0)
    parser.add_argument("--lambda-mlp", type=float, default=1.0)
    parser.add_argument("--alpha-k", type=float, default=0.0)
    parser.add_argument("--alpha-v", type=float, default=0.0)
    parser.add_argument("--bias-cap", type=float, default=None)
    parser.add_argument("--gamma-min", type=float, default=0.0)
    parser.add_argument("--gamma-max", type=float, default=5.0)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--eta-max", type=float, default=5.0)
    parser.add_argument("--recency-mix", choices=["on", "off"], default="on")
    parser.add_argument("--recency-rho", type=float, default=0.2)
    parser.add_argument("--recency-window", type=int, default=64)
    parser.add_argument("--recency-apply-after-prompt", choices=["on", "off"], default="on")
    parser.add_argument("--recency-scope", choices=["prefer_generated", "last_w"], default="prefer_generated")
    parser.add_argument("--steer-last-n-layers", type=int, default=None)
    parser.add_argument("--head-subset-mode", choices=["none", "file", "auto"], default="none")
    parser.add_argument("--head-mask-path", type=Path, default=None)
    parser.add_argument("--head-mask-apply-to", choices=["l1", "l2", "both"], default="both")
    parser.add_argument("--head-mask-debug", action="store_true")
    parser.add_argument("--head-subset-topk-per-layer", type=int, default=4)
    parser.add_argument("--head-subset-calib-runs", type=int, default=3)
    parser.add_argument("--head-subset-calib-max-new-tokens", type=int, default=64)
    parser.add_argument("--head-subset-calib-first-decode-only", choices=["on", "off"], default="on")
    parser.add_argument("--head-subset-auto-save", type=Path, default=None)
    parser.add_argument("--collect-head-stats", choices=["on", "off"], default="off")
    parser.add_argument("--collect-head-stats-first-decode-only", choices=["on", "off"], default="on")
    parser.add_argument("--joern-cli-dir", type=Path, default=DEFAULT_JOERN_CLI_DIR)
    parser.add_argument("--joern-cache-dir", type=Path, default=Path(".cache/joern_slice"))
    parser.add_argument("--joern-direction", choices=["backward", "forward"], default="backward")
    parser.add_argument("--joern-slice-depth", type=int, default=20)
    parser.add_argument("--joern-parallelism", type=int, default=1)
    parser.add_argument("--joern-timeout-sec", type=int, default=180)
    parser.add_argument("--joern-include-control", choices=["on", "off"], default="on")
    parser.add_argument("--joern-max-hops", type=int, default=None)
    parser.add_argument("--resume", choices=["on", "off"], default="off")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.protocol = _normalize_protocol(args.protocol)
    if args.variant in {"steered", "both"} and not args.steer:
        args.steer = True
    if args.steer and args.prior == "code_gadget":
        raise ValueError(
            "Steered code_gadget runs are temporarily blocked pending the final "
            "multi-gadget-to-prior reduction rule."
        )
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

    variants = _variant_list(args)
    base_steering_cfg = _build_steering_config(args, PROJECT_ROOT)

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
    if "steered" in variants and base_steering_cfg is not None:
        _apply_last_n_layers(model, base_steering_cfg, args.steer_last_n_layers)
        model.set_steering_config(copy.deepcopy(base_steering_cfg))
    model.build()
    if "steered" in variants and base_steering_cfg is not None:
        _apply_last_n_layers(model, base_steering_cfg, args.steer_last_n_layers)

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
        "variant": args.variant,
        "variants": variants,
        "model_name": model.model_name,
        "steering_enabled": bool(args.steer),
        "prior": args.prior,
        "joern_direction": args.joern_direction,
        "joern_slice_depth": int(args.joern_slice_depth),
        "joern_timeout_sec": int(args.joern_timeout_sec),
        "steer_last_n_layers": args.steer_last_n_layers,
        "head_subset_mode": args.head_subset_mode,
        "head_subset_topk_per_layer": int(args.head_subset_topk_per_layer),
        "head_subset_calib_runs": int(args.head_subset_calib_runs),
        "head_subset_calib_max_new_tokens": int(args.head_subset_calib_max_new_tokens),
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
                    variant_cfg = copy.deepcopy(base_steering_cfg) if (variant == "steered" and base_steering_cfg is not None) else None
                    pending_runs = list(needed_runs)
                    try:
                        if variant == "steered" and variant_cfg is not None and str(variant_cfg.head_subset_mode).lower() == "auto":
                            _calibrate_head_subset_for_protocol(
                                model=model,
                                base_cfg=variant_cfg,
                                code=code,
                                sample_id=sample_id,
                                seed_base=int(args.seed) + int(row_idx) * 1000 + 500_000,
                                protocol=args.protocol,
                                instruction=selected_instruction,
                                language=args.language,
                                answer_prefix=args.answer_prefix,
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=args.top_k,
                            )
                        for sample_run in needed_runs:
                            _set_active_steering(model, variant_cfg)
                            sample_seed = int(args.seed) + int(row_idx) * 1000 + sample_run
                            if variant == "steered":
                                sample_seed += 1_000_000
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
                            joern_summary = _extract_joern_summary(model) if variant == "steered" else None
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
                                "joern_summary": joern_summary,
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
                            if variant == "steered":
                                sample_seed += 1_000_000
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
