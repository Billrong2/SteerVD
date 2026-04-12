from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import base64
import torch


POOLING_STRATEGIES: Dict[str, str] = {
    "first_layer": "first",
    "last_layer": "last",
    "last4_layers_mean": "last4",
    "all_layers_mean": "all",
}


@dataclass(frozen=True)
class AttentionPostprocessResult:
    attention_by_generated_token: List[Dict[str, Any]]
    pooled_attention_by_generated_token: List[Dict[str, Any]]
    global_pooled_attention_over_prompt: Dict[str, Any]
    layer_prompt_stats: Optional[Dict[str, Any]]


def _renorm_or_copy(vec: torch.Tensor, do: bool) -> torch.Tensor:
    if not do:
        return vec
    s = float(vec.sum().item())
    if s > 1e-12:
        return vec / s
    return vec


def _pool_layer_vectors(layer_vecs: List[torch.Tensor], mode: str) -> torch.Tensor:
    """
    layer_vecs: list of 1D tensors of length k_len (one per layer), already averaged across heads.
    mode: 'first', 'last', 'last4', 'all'
    """
    if len(layer_vecs) == 0:
        raise ValueError("No layer vectors to pool.")
    if mode == "first":
        return layer_vecs[0]
    if mode == "last":
        return layer_vecs[-1]
    if mode == "last4":
        take = min(4, len(layer_vecs))
        return torch.stack(layer_vecs[-take:], dim=0).mean(dim=0)
    if mode == "all":
        return torch.stack(layer_vecs, dim=0).mean(dim=0)
    raise ValueError(f"Unknown pooling mode: {mode}")


def _to_cpu_list_floats(t: torch.Tensor) -> List[float]:
    return [float(x) for x in t.detach().to(torch.float32).cpu().tolist()]


def _to_cpu_list_ints(t: torch.Tensor) -> List[int]:
    return [int(x) for x in t.detach().to(torch.int64).cpu().tolist()]


def postprocess_generation_attentions(
    *,
    attentions: Any,
    tokens_all: List[str],
    prompt_len: int,
    num_generated: int,
    key_scope: str,
    renormalize: bool,
    top_attended_k: int,
    record_layers: bool,
    model: Any,
) -> AttentionPostprocessResult:
    """
    Convert HF `generate(..., output_attentions=True)` tensors into:
    - per-step per-layer top-k diagnostics (optional)
    - per-step pooled attention distributions (always when attentions are present)
    - global pooled distribution over prompt tokens (when key_scope == "prompt")

    This function intentionally preserves the existing `model_output.json` schema.
    """

    pool_names = list(POOLING_STRATEGIES.keys())

    device = model.device
    global_prompt_accum = {name: torch.zeros(prompt_len, dtype=torch.float32, device=device) for name in pool_names}

    layer_prompt_totals = None
    if key_scope == "prompt":
        num_layers_cfg = getattr(model.config, "num_hidden_layers", None)
        if num_layers_cfg is None:
            num_layers_cfg = len(model.model.layers)  # type: ignore[attr-defined]
        layer_prompt_totals = [0.0 for _ in range(int(num_layers_cfg))]

    attn_record: List[Dict[str, Any]] = []
    pooled_record: List[Dict[str, Any]] = []

    if attentions is None:
        attentions = ()

    num_steps = min(num_generated, len(attentions))

    # attentions[step][layer] is typically: [batch=1, n_heads, q_len(=1), k_len]
    for step_idx in range(num_steps):
        abs_idx = prompt_len + step_idx
        gen_tok = tokens_all[abs_idx] if abs_idx < len(tokens_all) else ""

        step_layers = attentions[step_idx]
        layer_vecs: List[torch.Tensor] = []
        layer_summaries: List[Dict[str, Any]] = []

        for layer_idx, layer_tensor in enumerate(step_layers):
            # [batch, n_heads, q_len, k_len] -> [n_heads, q_len, k_len]
            attn = layer_tensor[0]
            # choose the last query position (robust if q_len != 1)
            attn = attn[:, -1, :]  # [n_heads, k_len]
            attn_mean = attn.mean(dim=0).to(torch.float32)  # [k_len]
            layer_vecs.append(attn_mean)

            if layer_prompt_totals is not None and layer_idx < len(layer_prompt_totals):
                layer_prompt_totals[layer_idx] += float(attn_mean[:prompt_len].sum().item())

            if record_layers:
                k_len = int(attn_mean.shape[-1])
                take = min(top_attended_k, k_len)
                top_vals, top_idx = torch.topk(attn_mean, k=take, largest=True, sorted=True)
                top_idx_list = _to_cpu_list_ints(top_idx)
                layer_summaries.append(
                    {
                        "layer": layer_idx,
                        "k_len": k_len,
                        "top_indices": top_idx_list,
                        "top_tokens": [tokens_all[i] for i in top_idx_list],
                        "top_values": _to_cpu_list_floats(top_vals),
                    }
                )

        per_step_pooled: Dict[str, Any] = {
            "generated_step": step_idx,
            "absolute_token_index": abs_idx,
            "token": gen_tok,
            "pools": {},
        }

        for name, mode in POOLING_STRATEGIES.items():
            pooled_vec = _pool_layer_vectors(layer_vecs, mode=mode)  # [k_len]

            if key_scope == "prompt":
                pooled_slice = pooled_vec[:prompt_len]
                pooled_slice = _renorm_or_copy(pooled_slice, renormalize)
                per_step_pooled.setdefault("prompt_scores", {})[name] = _to_cpu_list_floats(pooled_slice)
                global_prompt_accum[name] += pooled_slice

                take = min(top_attended_k, prompt_len)
                vals, idx = torch.topk(pooled_slice, k=take, largest=True, sorted=True)
                idx_list = _to_cpu_list_ints(idx)
                per_step_pooled["pools"][name] = {
                    "key_scope": "prompt",
                    "top_indices": idx_list,
                    "top_tokens": [tokens_all[i] for i in idx_list],
                    "top_values": _to_cpu_list_floats(vals),
                }
            else:
                pooled_vec = _renorm_or_copy(pooled_vec, renormalize)
                per_step_pooled.setdefault("prompt_scores", {})[name] = _to_cpu_list_floats(pooled_vec)
                k_len = int(pooled_vec.shape[-1])
                take = min(top_attended_k, k_len)
                vals, idx = torch.topk(pooled_vec, k=take, largest=True, sorted=True)
                idx_list = _to_cpu_list_ints(idx)
                per_step_pooled["pools"][name] = {
                    "key_scope": "all",
                    "top_indices": idx_list,
                    "top_tokens": [tokens_all[i] for i in idx_list],
                    "top_values": _to_cpu_list_floats(vals),
                }

        pooled_record.append(per_step_pooled)

        if record_layers:
            attn_record.append(
                {
                    "generated_step": step_idx,
                    "absolute_token_index": abs_idx,
                    "token": gen_tok,
                    "layers": layer_summaries,
                }
            )

    global_pooled_over_prompt: Dict[str, Any] = {}
    if key_scope == "prompt" and num_generated > 0:
        prompt_tokens = tokens_all[:prompt_len]
        for name in pool_names:
            mean_vec = global_prompt_accum[name] / float(max(num_steps, 1))
            mean_vec = _renorm_or_copy(mean_vec, renormalize)
            take = min(top_attended_k, prompt_len)
            vals, idx = torch.topk(mean_vec, k=take, largest=True, sorted=True)
            idx_list = _to_cpu_list_ints(idx)
            global_pooled_over_prompt[name] = {
                "prompt_tokens": prompt_tokens,
                "scores": _to_cpu_list_floats(mean_vec),
                "top_indices": idx_list,
                "top_tokens": [prompt_tokens[i] for i in idx_list],
                "top_values": _to_cpu_list_floats(vals),
            }

    layer_prompt_stats = None
    if layer_prompt_totals is not None:
        denom = float(max(num_steps, 1))
        prompt_mass_mean = [val / denom for val in layer_prompt_totals]
        layer_prompt_stats = {
            "num_layers": len(layer_prompt_totals),
            "prompt_mass_per_layer": prompt_mass_mean,
            "prompt_mass_totals": layer_prompt_totals,
            "prompt_length": prompt_len,
            "num_generated_tokens": num_generated,
        }

    return AttentionPostprocessResult(
        attention_by_generated_token=attn_record,
        pooled_attention_by_generated_token=pooled_record,
        global_pooled_attention_over_prompt=global_pooled_over_prompt,
        layer_prompt_stats=layer_prompt_stats,
    )


def extract_full_decode_head_tensors(
    *,
    attentions: Any,
    tokens_all: List[str],
    token_ids_all: List[int],
    prompt_len: int,
    num_generated: int,
) -> Dict[str, Any]:
    """
    Export full decode-time attention tensors as deterministic, compact JSON-ready payload.

    Output schema:
      {
        "schema_version": "record_layers_full_v1",
        "prompt_len": int,
        "num_generated": int,
        "num_layers": int,
        "num_heads": int,
        "token_ids_all": [...],
        "tokens_all": [...],
        "steps": [
          {
            "step_idx": int,
            "abs_token_index": int,
            "generated_token": str,
            "kv_len": int,
            "layers": [
              {
                "layer_idx": int,
                "shape": [num_heads, kv_len],
                "dtype": "fp16",
                "values_b64": "..."
              }, ...
            ]
          }, ...
        ]
      }
    """
    payload: Dict[str, Any] = {
        "schema_version": "record_layers_full_v1",
        "prompt_len": int(prompt_len),
        "num_generated": int(num_generated),
        "num_layers": 0,
        "num_heads": 0,
        "token_ids_all": [int(x) for x in token_ids_all],
        "tokens_all": list(tokens_all),
        "steps": [],
    }
    if attentions is None:
        return payload
    if not isinstance(attentions, (tuple, list)):
        return payload

    num_steps = min(int(num_generated), len(attentions))
    if num_steps <= 0:
        return payload

    for step_idx in range(num_steps):
        step_layers = attentions[step_idx]
        if not isinstance(step_layers, (tuple, list)):
            continue
        abs_idx = prompt_len + step_idx
        generated_token = tokens_all[abs_idx] if abs_idx < len(tokens_all) else ""

        step_entry: Dict[str, Any] = {
            "step_idx": int(step_idx),
            "abs_token_index": int(abs_idx),
            "generated_token": generated_token,
            "kv_len": 0,
            "layers": [],
        }

        for layer_idx, layer_tensor in enumerate(step_layers):
            # expected: [batch=1, n_heads, q_len, kv_len]
            if layer_tensor is None:
                continue
            attn = layer_tensor[0][:, -1, :]  # [n_heads, kv_len]
            attn_fp16 = attn.detach().to(torch.float16).contiguous().cpu().numpy()
            step_entry["kv_len"] = int(attn_fp16.shape[-1])
            if payload["num_heads"] == 0:
                payload["num_heads"] = int(attn_fp16.shape[0])
            layer_entry = {
                "layer_idx": int(layer_idx),
                "shape": [int(attn_fp16.shape[0]), int(attn_fp16.shape[1])],
                "dtype": "fp16",
                "values_b64": base64.b64encode(attn_fp16.tobytes()).decode("ascii"),
            }
            step_entry["layers"].append(layer_entry)

        payload["num_layers"] = max(payload["num_layers"], len(step_entry["layers"]))
        payload["steps"].append(step_entry)

    return payload
