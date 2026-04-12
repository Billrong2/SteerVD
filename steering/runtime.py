from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .config import SteeringConfig
from .manager import SteeringManager
from .pointer import build_pointer_mapping


@dataclass
class SteeringRuntime:
    config: SteeringConfig
    manager: SteeringManager
    prompt_token_ids: Sequence[int]
    prompt_tokens: Sequence[str]
    pointer_mapping: dict
    prompt_len: int
    prompt_attention_mask: Optional[Sequence[int]] = None
    enabled: bool = True
    decode_only: bool = True
    only_first_decode_step: bool = True
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    gating_debug: bool = False
    debug_assert_mask: bool = False
    total_steps: int = 0
    step_index: int = 0
    latest_attention: Optional[torch.Tensor] = None
    steer_calls: int = 0
    blocked_prefill_calls: int = 0
    blocked_q_len: int = 0
    blocked_kv_len: int = 0
    blocked_layer: int = 0
    blocked_disabled: int = 0
    current_step_kv_len: Optional[int] = None
    steered_layers_this_step: Optional[np.ndarray] = None
    agree_rel_this_step: Optional[np.ndarray] = None
    _gate_logged_once: bool = False
    temporal_debug: list[dict] = field(default_factory=list)
    _temporal_logged_keys: set[tuple[int, int]] = field(default_factory=set)
    # Level-3 residual-scale diagnostics.
    residual_calls_attn: int = 0
    residual_calls_mlp: int = 0
    blocked_residual_prefill: int = 0
    blocked_residual_layer: int = 0
    blocked_residual_disabled: int = 0
    blocked_residual_no_steer: int = 0
    residual_debug: list[dict] = field(default_factory=list)
    _residual_logged_once: bool = False
    _residual_debug_keys: set[tuple[int, int, str]] = field(default_factory=set)
    # Step-4 head subset state.
    head_mask_loaded: bool = False
    head_mask_error: Optional[str] = None
    head_mask_active_total: int = 0
    head_mask_active_by_layer: Dict[str, int] = field(default_factory=dict)
    _head_mask_np: Optional[np.ndarray] = None
    _head_mask_shape: Optional[tuple[int, int]] = None
    # Optional offline head-stat collection.
    head_stats_sum: Optional[np.ndarray] = None
    head_stats_count: Optional[np.ndarray] = None
    # Per-level activation accounting for parity audits.
    level1_calls: int = 0
    level2_calls: int = 0
    level4_calls: int = 0
    level5_calls: int = 0
    level1_steps: set[int] = field(default_factory=set)
    level2_steps: set[int] = field(default_factory=set)
    level3_steps: set[int] = field(default_factory=set)
    level4_steps: set[int] = field(default_factory=set)
    level5_steps: set[int] = field(default_factory=set)
    level_event_trace: list[dict] = field(default_factory=list)
    # Level-5 pointer diagnostics.
    pointer_calls_total: int = 0
    pointer_bias_applied_steps: int = 0
    pointer_missing_attention_steps: int = 0
    pointer_beta_zero_steps: int = 0

    def start(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.step_index = 0
        self.steer_calls = 0
        self.blocked_prefill_calls = 0
        self.blocked_q_len = 0
        self.blocked_kv_len = 0
        self.blocked_layer = 0
        self.blocked_disabled = 0
        self.current_step_kv_len = None
        self.steered_layers_this_step = None
        self.agree_rel_this_step = None
        self._gate_logged_once = False
        self.temporal_debug = []
        self._temporal_logged_keys = set()
        self.residual_calls_attn = 0
        self.residual_calls_mlp = 0
        self.blocked_residual_prefill = 0
        self.blocked_residual_layer = 0
        self.blocked_residual_disabled = 0
        self.blocked_residual_no_steer = 0
        self.residual_debug = []
        self._residual_logged_once = False
        self._residual_debug_keys = set()
        self.head_stats_sum = None
        self.head_stats_count = None
        self.level1_calls = 0
        self.level2_calls = 0
        self.level4_calls = 0
        self.level5_calls = 0
        self.level1_steps = set()
        self.level2_steps = set()
        self.level3_steps = set()
        self.level4_steps = set()
        self.level5_steps = set()
        self.level_event_trace = []
        self.pointer_calls_total = 0
        self.pointer_bias_applied_steps = 0
        self.pointer_missing_attention_steps = 0
        self.pointer_beta_zero_steps = 0
        self.manager.init_bins(total_steps)
        self.manager.step(self.step_index)

    def advance(self) -> None:
        self.step_index = min(self.step_index + 1, self.total_steps - 1)
        self.manager.step(self.step_index)

    def _prompt_mask_tensor(self, device: torch.device, key_len: int) -> torch.Tensor:
        """
        Valid-key mask for [0:key_len), using prompt pad-mask for prompt positions and
        ones for generated positions.
        """
        mask = torch.ones(key_len, device=device, dtype=torch.float32)
        if self.prompt_attention_mask is not None:
            prompt_len = min(self.prompt_len, key_len, len(self.prompt_attention_mask))
            if prompt_len > 0:
                prompt_mask = torch.as_tensor(
                    self.prompt_attention_mask[:prompt_len],
                    device=device,
                    dtype=torch.float32,
                )
                mask[:prompt_len] = prompt_mask
        return mask

    def _uniform_over_valid_keys(self, device: torch.device, key_len: int) -> torch.Tensor:
        key_valid = self._prompt_mask_tensor(device, key_len)
        total = key_valid.sum()
        if total <= 0:
            return torch.full((key_len,), 1.0 / max(key_len, 1), device=device, dtype=torch.float32)
        return key_valid / total

    def _record_temporal_debug(self, key_len: int, prior_vec: torch.Tensor, rho_used: float) -> None:
        if len(self.temporal_debug) >= 3:
            return
        key = (self.step_index, int(key_len))
        if key in self._temporal_logged_keys:
            return
        self._temporal_logged_keys.add(key)

        prompt_len = min(self.prompt_len, key_len)
        recent_start = max(0, key_len - max(int(self.config.recency_window), 1))
        prompt_mass = float(prior_vec[:prompt_len].sum().item()) if prompt_len > 0 else 0.0
        recent_mass = float(prior_vec[recent_start:key_len].sum().item()) if key_len > recent_start else 0.0
        generated_mass = float(prior_vec[prompt_len:key_len].sum().item()) if key_len > prompt_len else 0.0
        self.temporal_debug.append(
            {
                "step_index": int(self.step_index),
                "kv_len": int(key_len),
                "rho_used": float(rho_used),
                "prompt_mass": prompt_mass,
                "recent_mass": recent_mass,
                "generated_mass": generated_mass,
            }
        )

    def build_key_prior(self, device: torch.device, key_len: int) -> torch.Tensor:
        """
        Step-2 temporal prior:
        - First decode step: prompt prior only.
        - Later decode steps: (1-rho)*prompt_prior_padded + rho*recency_prior.
        - Uniform prior stays a no-op over valid keys.
        """
        eps = 1e-12
        key_len = int(key_len)
        if key_len <= 0:
            return torch.zeros(0, device=device, dtype=torch.float32)

        # Keep uniform prior as a guaranteed no-op for softmax.
        if self.config.prior == "uniform":
            prior = self._uniform_over_valid_keys(device, key_len)
            self._record_temporal_debug(key_len, prior, rho_used=0.0)
            return prior

        prompt_len = min(self.prompt_len, key_len)
        prompt_prior = torch.zeros(key_len, device=device, dtype=torch.float32)
        prior_vec = self.manager.prior_vector()
        if prompt_len > 0:
            prompt_prior[:prompt_len] = torch.from_numpy(prior_vec[:prompt_len]).to(device=device, dtype=torch.float32)

        key_valid = self._prompt_mask_tensor(device, key_len)
        prompt_prior = prompt_prior * key_valid
        prompt_mass = prompt_prior.sum()
        if prompt_mass > 0:
            prompt_prior = prompt_prior / prompt_mass
        else:
            prompt_prior = self._uniform_over_valid_keys(device, key_len)

        rho = float(np.clip(self.config.recency_rho, 0.0, 1.0))
        use_mix = bool(self.config.recency_mix)
        apply_after_prompt = bool(self.config.recency_apply_after_prompt)
        if (not use_mix) or (apply_after_prompt and key_len == prompt_len):
            prior = prompt_prior
            self._record_temporal_debug(key_len, prior, rho_used=0.0)
            return prior

        # Recency prior: prefer generated keys first, fallback to last-W overall.
        window = max(1, int(self.config.recency_window))
        if str(self.config.recency_scope).lower() == "prefer_generated":
            start = max(prompt_len, key_len - window)
            if start >= key_len:
                start = max(0, key_len - window)
        else:
            start = max(0, key_len - window)

        recent = torch.zeros(key_len, device=device, dtype=torch.float32)
        recent[start:key_len] = 1.0
        recent = recent * key_valid
        recent_mass = recent.sum()
        if recent_mass > 0:
            recent = recent / recent_mass
        else:
            recent = self._uniform_over_valid_keys(device, key_len)

        prior = (1.0 - rho) * prompt_prior + rho * recent
        prior = prior * key_valid
        norm = prior.sum()
        if norm > eps:
            prior = prior / norm
        else:
            prior = self._uniform_over_valid_keys(device, key_len)
        self._record_temporal_debug(key_len, prior, rho_used=rho)
        return prior

    def prior_tensor(self, device: torch.device, key_len: int) -> torch.Tensor:
        prior_vec = self.build_key_prior(device=device, key_len=key_len)
        return prior_vec.view(1, 1, 1, -1)

    # ---- Step-4: steerable head subset ---------------------------------
    @staticmethod
    def _normalize_model_name(name: Optional[str]) -> str:
        return str(name or "").strip().lower()

    def _validate_head_mask_model(self, payload: Any, source_desc: str) -> None:
        if not isinstance(payload, dict):
            return

        meta = payload.get("meta")
        payload_model = None
        if isinstance(meta, dict):
            payload_model = meta.get("model_name")
        if payload_model is None:
            payload_model = payload.get("model_name")

        if not payload_model:
            return

        active_model = self._normalize_model_name(getattr(self.config, "model_name", None))
        source_model = self._normalize_model_name(str(payload_model))
        if active_model and source_model and active_model != source_model:
            raise ValueError(
                "Head mask model mismatch: "
                f"mask={payload_model!r}, active={getattr(self.config, 'model_name', None)!r}, "
                f"source={source_desc}"
            )

    def _parse_head_mask_payload(self, payload: Any, num_layers: int, num_heads: int) -> np.ndarray:
        if isinstance(payload, dict) and "mask" in payload:
            mask_data = payload["mask"]
        elif isinstance(payload, dict) and "active_heads" in payload:
            mask = np.zeros((num_layers, num_heads), dtype=bool)
            active = payload["active_heads"]
            if isinstance(active, dict):
                for layer_key, heads in active.items():
                    li = int(layer_key)
                    if li < 0 or li >= num_layers:
                        continue
                    for h in heads:
                        hi = int(h)
                        if 0 <= hi < num_heads:
                            mask[li, hi] = True
                return mask
            raise ValueError("head_mask active_heads must be a dict[layer -> list[head]].")
        elif isinstance(payload, list):
            mask_data = payload
        else:
            raise ValueError("Unsupported head mask format. Use {'mask': [[...], ...]} or {'active_heads': {...}}.")

        mask_np = np.asarray(mask_data)
        if mask_np.shape != (num_layers, num_heads):
            raise ValueError(
                f"Head mask shape mismatch: expected {(num_layers, num_heads)}, got {tuple(mask_np.shape)}"
            )
        if mask_np.dtype != np.bool_:
            mask_np = mask_np.astype(np.float32) > 0.5
        return mask_np

    def _ensure_head_mask(self, num_layers: int, num_heads: int) -> None:
        if self.config.head_subset_mode == "none":
            self._head_mask_np = None
            self._head_mask_shape = None
            self.head_mask_loaded = False
            self.head_mask_error = None
            self.head_mask_active_total = 0
            self.head_mask_active_by_layer = {}
            return

        if self.config.head_subset_mode not in {"file", "auto"}:
            self.head_mask_error = f"Unknown head_subset_mode={self.config.head_subset_mode!r}"
            raise ValueError(self.head_mask_error)

        if self._head_mask_np is not None and self._head_mask_shape == (num_layers, num_heads):
            return

        source_desc = "inline"
        if self.config.head_mask_inline is not None:
            payload = self.config.head_mask_inline
        else:
            if not self.config.head_mask_path:
                self.head_mask_error = (
                    f"head_subset_mode={self.config.head_subset_mode!r} requires "
                    "head_mask_inline (auto calibration) or head_mask_path."
                )
                raise ValueError(self.head_mask_error)

            path = Path(self.config.head_mask_path)
            if not path.is_file():
                self.head_mask_error = f"Head mask file not found: {path}"
                raise FileNotFoundError(self.head_mask_error)

            payload = json.loads(path.read_text(encoding="utf-8"))
            source_desc = str(path)

        self._validate_head_mask_model(payload, source_desc=source_desc)
        mask_np = self._parse_head_mask_payload(payload, num_layers=num_layers, num_heads=num_heads)
        self._head_mask_np = mask_np
        self._head_mask_shape = (num_layers, num_heads)
        self.head_mask_loaded = True
        self.head_mask_error = None
        self.head_mask_active_total = int(mask_np.sum())
        self.head_mask_active_by_layer = {
            str(i): int(mask_np[i].sum()) for i in range(num_layers) if int(mask_np[i].sum()) > 0
        }
        if self.config.head_mask_debug:
            print(
                f"[HeadMask] loaded source={source_desc} shape={mask_np.shape} "
                f"active_total={self.head_mask_active_total}"
            )

    def _head_mask_applies(self, level: str) -> bool:
        mode = str(self.config.head_mask_apply_to).lower()
        if mode == "both":
            return True
        return mode == level

    def get_head_mask_vector(
        self,
        *,
        layer_idx: int,
        num_layers: int,
        num_heads: int,
        device: torch.device,
        level: str,
        ignore_apply_to: bool = False,
    ) -> Optional[torch.Tensor]:
        if self.config.head_subset_mode == "none":
            return None
        if (not ignore_apply_to) and (not self._head_mask_applies(level)):
            return None

        self._ensure_head_mask(num_layers=num_layers, num_heads=num_heads)
        if self._head_mask_np is None:
            return None
        if layer_idx < 0 or layer_idx >= self._head_mask_np.shape[0]:
            return None
        row = self._head_mask_np[layer_idx]
        return torch.as_tensor(row, device=device, dtype=torch.float32)

    def get_head_mask_tensor(
        self,
        *,
        layer_idx: int,
        num_layers: int,
        num_heads: int,
        device: torch.device,
        level: str,
    ) -> Optional[torch.Tensor]:
        row = self.get_head_mask_vector(
            layer_idx=layer_idx,
            num_layers=num_layers,
            num_heads=num_heads,
            device=device,
            level=level,
            ignore_apply_to=False,
        )
        if row is None:
            return None
        return row.view(1, num_heads, 1, 1)

    # ---- Optional offline head-stat collection --------------------------
    def _ensure_head_stats_arrays(self, num_layers: int, num_heads: int) -> None:
        shape = (num_layers, num_heads)
        if self.head_stats_sum is None or self.head_stats_sum.shape != shape:
            self.head_stats_sum = np.zeros(shape, dtype=np.float64)
            self.head_stats_count = np.zeros(shape, dtype=np.int64)

    def maybe_collect_head_stats(
        self,
        *,
        layer_idx: int,
        num_layers: int,
        num_heads: int,
        q_len: int,
        kv_len: int,
        default_layer_start: int,
        default_layer_end: int,
        attn_probs: torch.Tensor,
    ) -> None:
        if not self.config.collect_head_stats:
            return
        if q_len != 1:
            return
        if self.config.collect_head_stats_first_decode_only and kv_len != self.prompt_len:
            return
        layer_start = default_layer_start if self.layer_start is None else self.layer_start
        layer_end = default_layer_end if self.layer_end is None else self.layer_end
        if layer_idx < layer_start or layer_idx > layer_end:
            return
        if attn_probs.ndim != 4:
            return
        if attn_probs.shape[1] != num_heads:
            return

        self._ensure_head_stats_arrays(num_layers=num_layers, num_heads=num_heads)
        prior = self.build_key_prior(device=attn_probs.device, key_len=kv_len).to(attn_probs.dtype)
        # agree[b,h] = sum_k P_last[b,h,k] * p_all[k]
        agree = (attn_probs[:, :, -1, :] * prior.view(1, 1, kv_len)).sum(dim=-1)
        agree_mean = agree.mean(dim=0).detach().to(torch.float64).cpu().numpy()
        self.head_stats_sum[layer_idx, :num_heads] += agree_mean[:num_heads]
        self.head_stats_count[layer_idx, :num_heads] += 1

    def head_stats_payload(self) -> Optional[Dict[str, Any]]:
        if self.head_stats_sum is None or self.head_stats_count is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.divide(
                self.head_stats_sum,
                np.maximum(self.head_stats_count, 1),
                where=np.maximum(self.head_stats_count, 1) > 0,
            )
        return {
            "shape": [int(self.head_stats_sum.shape[0]), int(self.head_stats_sum.shape[1])],
            "sum": self.head_stats_sum.tolist(),
            "count": self.head_stats_count.tolist(),
            "mean": mean.tolist(),
        }

    # ---- Decode-step state for residual scaling -------------------------
    def _ensure_step_state(self, num_layers: int) -> None:
        n = max(1, int(num_layers))
        if self.steered_layers_this_step is None or int(self.steered_layers_this_step.shape[0]) != n:
            self.steered_layers_this_step = np.zeros((n,), dtype=np.bool_)
        if self.agree_rel_this_step is None or int(self.agree_rel_this_step.shape[0]) != n:
            self.agree_rel_this_step = np.ones((n,), dtype=np.float32)

    def begin_decode_step(self, *, q_len: int, kv_len: int, num_layers: int) -> None:
        if q_len != 1:
            return
        self._ensure_step_state(num_layers)
        if self.current_step_kv_len is None or int(self.current_step_kv_len) != int(kv_len):
            self.current_step_kv_len = int(kv_len)
            self.steered_layers_this_step.fill(False)
            self.agree_rel_this_step.fill(1.0)

    def mark_layer_steered(self, *, layer_idx: int, num_layers: int) -> None:
        self._ensure_step_state(num_layers)
        if 0 <= int(layer_idx) < int(self.steered_layers_this_step.shape[0]):
            self.steered_layers_this_step[int(layer_idx)] = True

    def set_layer_agreement(self, *, layer_idx: int, num_layers: int, agree_rel: float) -> None:
        self._ensure_step_state(num_layers)
        if 0 <= int(layer_idx) < int(self.agree_rel_this_step.shape[0]):
            self.agree_rel_this_step[int(layer_idx)] = float(agree_rel)

    def _residual_band(self, *, default_layer_start: int, default_layer_end: int) -> tuple[int, int]:
        if self.config.residual_scale_layer_start is not None:
            start = int(self.config.residual_scale_layer_start)
        elif self.layer_start is not None:
            start = int(self.layer_start)
        else:
            start = int(default_layer_start)

        if self.config.residual_scale_layer_end is not None:
            end = int(self.config.residual_scale_layer_end)
        elif self.layer_end is not None:
            end = int(self.layer_end)
        else:
            end = int(default_layer_end)

        if end < start:
            end = start
        return start, end

    def should_apply_residual(
        self,
        *,
        layer_idx: int,
        seq_len: int,
        default_layer_start: int,
        default_layer_end: int,
    ) -> bool:
        if not self.enabled or not bool(self.config.residual_scale):
            self.blocked_residual_disabled += 1
            return False
        if int(seq_len) != 1:
            self.blocked_residual_prefill += 1
            return False

        layer_start, layer_end = self._residual_band(
            default_layer_start=default_layer_start,
            default_layer_end=default_layer_end,
        )
        if int(layer_idx) < layer_start or int(layer_idx) > layer_end:
            self.blocked_residual_layer += 1
            return False

        mode = str(self.config.residual_scale_mode).lower()
        if mode == "amplifier":
            self._ensure_step_state(max(layer_end + 1, layer_idx + 1))
            if not bool(self.steered_layers_this_step[int(layer_idx)]):
                self.blocked_residual_no_steer += 1
                return False

        if self.config.residual_scale_debug and not self._residual_logged_once:
            coeffs = self.coeffs()
            print(
                f"[ResidualScale] mode={mode} band={layer_start}..{layer_end} "
                f"lambdas=({float(coeffs.lambda_attn):.4f},{float(coeffs.lambda_mlp):.4f})"
            )
            self._residual_logged_once = True
        return True

    def residual_lambdas(
        self,
        *,
        layer_idx: int,
        num_layers: int,
        coeffs: Any,
    ) -> tuple[float, float]:
        mode = str(self.config.residual_scale_mode).lower()
        if mode == "paired":
            delta = float(np.clip(float(self.config.lambda_attn_delta), -0.2, 0.2))
            return 1.0 + delta, 1.0 - delta

        if mode == "agreement_gate":
            self._ensure_step_state(num_layers)
            agree_rel = 1.0
            if 0 <= int(layer_idx) < int(self.agree_rel_this_step.shape[0]):
                agree_rel = float(self.agree_rel_this_step[int(layer_idx)])
            cap = max(0.0, float(self.config.lambda_attn_cap))
            alpha = float(self.config.lambda_attn_alpha)
            gain = float(np.clip(agree_rel - 1.0, 0.0, cap))
            lambda_attn = 1.0 + alpha * gain
            lambda_mlp = float(coeffs.lambda_mlp)
            self._record_residual_debug(
                layer_idx=layer_idx,
                mode=mode,
                agree_rel=agree_rel,
                lambda_attn=lambda_attn,
                lambda_mlp=lambda_mlp,
            )
            return lambda_attn, lambda_mlp

        # static / amplifier share the same lambda source; amplifier only changes apply-gating.
        lambda_attn = float(coeffs.lambda_attn)
        lambda_mlp = float(coeffs.lambda_mlp)
        self._record_residual_debug(
            layer_idx=layer_idx,
            mode=mode,
            agree_rel=None,
            lambda_attn=lambda_attn,
            lambda_mlp=lambda_mlp,
        )
        return lambda_attn, lambda_mlp

    def _record_residual_debug(
        self,
        *,
        layer_idx: int,
        mode: str,
        agree_rel: Optional[float],
        lambda_attn: float,
        lambda_mlp: float,
    ) -> None:
        if not self.config.residual_scale_debug:
            return
        if len(self.residual_debug) >= 12:
            return
        key = (int(self.step_index), int(layer_idx), str(mode))
        if key in self._residual_debug_keys:
            return
        self._residual_debug_keys.add(key)
        entry: Dict[str, Any] = {
            "step_index": int(self.step_index),
            "layer_idx": int(layer_idx),
            "mode": str(mode),
            "lambda_attn": float(lambda_attn),
            "lambda_mlp": float(lambda_mlp),
        }
        if agree_rel is not None:
            entry["agree_rel"] = float(agree_rel)
        self.residual_debug.append(entry)

    def mark_residual_call(self, kind: str) -> None:
        self.level3_steps.add(int(self.step_index))
        if len(self.level_event_trace) < 4096:
            self.level_event_trace.append(
                {
                    "step_index": int(self.step_index),
                    "level": 3,
                    "kind": str(kind),
                }
            )
        if kind == "attn":
            self.residual_calls_attn += 1
        elif kind == "mlp":
            self.residual_calls_mlp += 1

    def mark_level_call(self, level: int) -> None:
        li = int(level)
        if len(self.level_event_trace) < 4096:
            self.level_event_trace.append(
                {
                    "step_index": int(self.step_index),
                    "level": li,
                }
            )
        if li == 1:
            self.level1_calls += 1
            self.level1_steps.add(int(self.step_index))
        elif li == 2:
            self.level2_calls += 1
            self.level2_steps.add(int(self.step_index))
        elif li == 4:
            self.level4_calls += 1
            self.level4_steps.add(int(self.step_index))
        elif li == 5:
            self.level5_calls += 1
            self.level5_steps.add(int(self.step_index))

    def level_call_counts_payload(self) -> Dict[str, Any]:
        return {
            "l1_calls": int(self.level1_calls),
            "l2_calls": int(self.level2_calls),
            "l4_calls": int(self.level4_calls),
            "l5_calls": int(self.level5_calls),
            "l1_active_steps": int(len(self.level1_steps)),
            "l2_active_steps": int(len(self.level2_steps)),
            "l3_active_steps": int(len(self.level3_steps)),
            "l4_active_steps": int(len(self.level4_steps)),
            "l5_active_steps": int(len(self.level5_steps)),
        }

    def level_event_trace_payload(self) -> list[dict]:
        return list(self.level_event_trace)

    def coeffs(self):
        return self.manager.coeffs()

    def should_apply_l12(
        self,
        *,
        layer_idx: int,
        q_len: int,
        kv_len: int,
        default_layer_start: int,
        default_layer_end: int,
    ) -> bool:
        reason = "apply"
        if not self.enabled:
            self.blocked_disabled += 1
            reason = "disabled"
            self._maybe_log_gate_decision(layer_idx, q_len, kv_len, reason)
            return False
        layer_start = default_layer_start if self.layer_start is None else self.layer_start
        layer_end = default_layer_end if self.layer_end is None else self.layer_end
        if layer_idx < layer_start or layer_idx > layer_end:
            self.blocked_layer += 1
            reason = "layer"
            self._maybe_log_gate_decision(layer_idx, q_len, kv_len, reason)
            return False
        if self.decode_only and q_len != 1:
            self.blocked_prefill_calls += 1
            self.blocked_q_len += 1
            reason = "q_len"
            self._maybe_log_gate_decision(layer_idx, q_len, kv_len, reason)
            return False
        if self.only_first_decode_step and kv_len != self.prompt_len:
            self.blocked_kv_len += 1
            reason = "kv_len"
            self._maybe_log_gate_decision(layer_idx, q_len, kv_len, reason)
            return False
        self._maybe_log_gate_decision(layer_idx, q_len, kv_len, reason)
        return True

    def _maybe_log_gate_decision(self, layer_idx: int, q_len: int, kv_len: int, reason: str) -> None:
        if not self.gating_debug or self._gate_logged_once:
            return
        print(
            f"[SteeringGate] layer={layer_idx} q_len={q_len} kv_len={kv_len} "
            f"prompt_len={self.prompt_len} decision={reason}"
        )
        self._gate_logged_once = True


def create_runtime(
    config: SteeringConfig,
    prompt_token_ids: Sequence[int],
    prompt_tokens: Sequence[str],
    code_text: str,
    vocab_tokens: Sequence[dict],
    prompt_text: str = "",
    prompt_attention_mask: Optional[Sequence[int]] = None,
) -> SteeringRuntime:
    manager = SteeringManager(
        config=config,
        prompt_tokens=prompt_tokens,
        code_text=code_text,
        vocab_tokens=vocab_tokens,
        prompt_text=prompt_text,
    )
    pointer_mapping = build_pointer_mapping(prompt_token_ids)
    return SteeringRuntime(
        config=config,
        manager=manager,
        prompt_token_ids=prompt_token_ids,
        prompt_tokens=prompt_tokens,
        pointer_mapping=pointer_mapping,
        prompt_len=len(prompt_token_ids),
        prompt_attention_mask=prompt_attention_mask,
        enabled=config.steering_enabled,
        decode_only=config.decode_only,
        only_first_decode_step=config.only_first_decode_step,
        layer_start=config.steer_layer_start,
        layer_end=config.steer_layer_end,
        gating_debug=config.gating_debug,
        debug_assert_mask=config.debug_assert_mask,
    )
