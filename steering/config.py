from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


@dataclass
class ScheduleEntry:
    beta_bias: float = 0.0
    beta_post: float = 0.0
    lambda_attn: float = 1.0
    lambda_mlp: float = 1.0
    alpha_k: float = 0.0
    alpha_v: float = 0.0
    beta_ptr: float = 0.0


@dataclass
class SteeringConfig:
    enabled_levels: Sequence[int] = field(default_factory=list)
    prior: str = "code_gadget"
    n_bins: int = 8
    binning: str = "equal_count"
    beta_bias: float = 0.0
    beta_post: float = 0.0
    lambda_attn: float = 1.0
    lambda_mlp: float = 1.0
    alpha_k: float = 0.0
    alpha_v: float = 0.0
    beta_ptr: float = 0.0
    bias_cap: Optional[float] = None
    gamma_min: float = 0.0
    gamma_max: float = 5.0
    eta_min: float = 0.0
    eta_max: float = 5.0
    model_name: Optional[str] = None
    schedule_json: Optional[Path] = None
    schedule: Dict[int, ScheduleEntry] = field(default_factory=dict)
    # Step-1 safety defaults for non-destructive L1/L2 steering.
    steering_enabled: bool = True
    decode_only: bool = True
    only_first_decode_step: bool = False
    split_prefill: bool = True
    steer_layer_start: Optional[int] = None
    steer_layer_end: Optional[int] = None
    gating_debug: bool = False
    debug_assert_mask: bool = False
    # Step-2 temporal decode prior: prompt prior + recency prior.
    recency_mix: bool = True
    recency_rho: float = 0.2
    recency_window: int = 64
    recency_apply_after_prompt: bool = True
    recency_scope: str = "prefer_generated"
    # Step-4 steerable-head subset configuration.
    head_subset_mode: str = "none"  # "none" | "file" | "auto"
    head_mask_path: Optional[Path] = None
    head_mask_apply_to: str = "both"  # "l1" | "l2" | "both"
    head_mask_debug: bool = False
    head_subset_topk_per_layer: int = 4
    head_subset_calib_runs: int = 3
    head_subset_calib_max_new_tokens: int = 64
    head_subset_calib_first_decode_only: bool = True
    head_subset_auto_save: Optional[Path] = None
    # In-memory mask produced by auto calibration (same shape as file mask).
    head_mask_inline: Optional[Any] = None
    head_subset_selected_heads: Dict[str, list[int]] = field(default_factory=dict)
    head_subset_calibration: Dict[str, Any] = field(default_factory=dict)
    # Joern-backed code-gadget extraction configuration.
    joern_cli_dir: Optional[Path] = None
    joern_cache_dir: Optional[Path] = None
    joern_slice_depth: int = 20
    joern_parallelism: int = 1
    joern_timeout_sec: int = 180
    joern_max_hops: Optional[int] = None
    # Optional offline head-stat collection.
    collect_head_stats: bool = False
    collect_head_stats_first_decode_only: bool = True
    # Improved Level-3 residual scaling controls.
    residual_scale: bool = False
    residual_scale_mode: str = "static"  # "static" | "amplifier" | "paired" | "agreement_gate"
    lambda_attn_delta: float = 0.0
    residual_scale_layer_start: Optional[int] = None
    residual_scale_layer_end: Optional[int] = None
    residual_scale_debug: bool = False
    lambda_attn_alpha: float = 0.05
    lambda_attn_cap: float = 4.0
    agreement_scope: str = "selected_heads"  # "selected_heads" | "all_heads"

    def load_schedule(self) -> None:
        if not self.schedule_json:
            return
        data = json.loads(Path(self.schedule_json).read_text(encoding="utf-8"))
        for bin_idx, payload in data.items():
            self.schedule[int(bin_idx)] = ScheduleEntry(
                beta_bias=payload.get("beta_bias", self.beta_bias),
                beta_post=payload.get("beta_post", self.beta_post),
                lambda_attn=payload.get("lambda_attn", self.lambda_attn),
                lambda_mlp=payload.get("lambda_mlp", self.lambda_mlp),
                alpha_k=payload.get("alpha_k", self.alpha_k),
                alpha_v=payload.get("alpha_v", self.alpha_v),
                beta_ptr=payload.get("beta_ptr", self.beta_ptr),
            )

    def coeff_for_bin(self, bin_idx: int) -> ScheduleEntry:
        return self.schedule.get(
            bin_idx,
            ScheduleEntry(
                beta_bias=self.beta_bias,
                beta_post=self.beta_post,
                lambda_attn=self.lambda_attn,
                lambda_mlp=self.lambda_mlp,
                alpha_k=self.alpha_k,
                alpha_v=self.alpha_v,
                beta_ptr=self.beta_ptr,
            ),
        )
