from __future__ import annotations

import math
from typing import Callable, Optional, TYPE_CHECKING

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen3,
    repeat_kv as repeat_kv_qwen3,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen3_moe,
    repeat_kv as repeat_kv_qwen3_moe,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen3_next,
    repeat_kv as repeat_kv_qwen3_next,
)

from ..levels import level1_bias, level2_post, level4_scale
from .common import LayerCutoffs, compute_default_cutoffs, get_decoder_layers

if TYPE_CHECKING:
    from ..config import SteeringConfig
    from ..runtime import SteeringRuntime


def _agreement_rel_layer(
    *,
    runtime: "SteeringRuntime",
    attn_probs: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    num_heads: int,
    kv_len: int,
    device: torch.device,
) -> Optional[float]:
    if attn_probs.ndim != 4:
        return None
    if attn_probs.shape[1] != num_heads:
        return None

    prior = runtime.build_key_prior(device=device, key_len=kv_len).to(attn_probs.dtype)
    agree = (attn_probs[:, :, -1, :] * prior.view(1, 1, kv_len)).sum(dim=-1) * float(kv_len)
    if agree.numel() == 0:
        return None

    scope = str(runtime.config.agreement_scope).lower()
    if scope == "selected_heads":
        mask = runtime.get_head_mask_vector(
            layer_idx=layer_idx,
            num_layers=num_layers,
            num_heads=num_heads,
            device=device,
            level="l1",
            ignore_apply_to=True,
        )
        if mask is not None and float(mask.sum().item()) > 0.0:
            selected = mask.view(-1) > 0
            if bool(selected.any().item()):
                agree = agree[:, selected]

    return float(agree.mean().item())


class Qwen3SteeringAttention(nn.Module):
    """
    Qwen3 full-attention wrapper with L1/L2/L4 steering and attention telemetry.
    """

    def __init__(
        self,
        base_attn: Qwen3Attention,
        runtime_getter: Callable[[], Optional["SteeringRuntime"]],
        layer_index: int,
        config: "SteeringConfig",
        cutoffs: LayerCutoffs,
    ) -> None:
        super().__init__()
        self.base = base_attn
        self.runtime_getter = runtime_getter
        self.layer_index = layer_index
        self.config = config
        self.cutoffs = cutoffs

        self.config_hf = base_attn.config
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj
        self.q_norm = base_attn.q_norm
        self.k_norm = base_attn.k_norm
        self.layer_idx = getattr(base_attn, "layer_idx", layer_index)
        self.head_dim = int(base_attn.head_dim)
        self.num_heads = int(getattr(base_attn.config, "num_attention_heads", 0))
        self.num_key_value_heads = int(getattr(base_attn.config, "num_key_value_heads", 0))
        self.num_key_value_groups = int(getattr(base_attn, "num_key_value_groups", 1))
        self.scaling = float(base_attn.scaling)
        self.attention_dropout = float(base_attn.attention_dropout)
        self.hidden_size = int(getattr(base_attn.config, "hidden_size", 0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        runtime = self.runtime_getter()
        output_attentions = bool(kwargs.get("output_attentions", False))

        input_shape = hidden_states.shape[:-1]
        bsz = hidden_states.shape[0]
        q_len = hidden_states.shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        target_device = hidden_states.device
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)
        if cache_position is not None and cache_position.device != target_device:
            cache_position = cache_position.to(target_device)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qwen3(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv_qwen3(key_states, self.num_key_value_groups)
        value_states = repeat_kv_qwen3(value_states, self.num_key_value_groups)

        if runtime and 4 in self.config.enabled_levels and self.layer_index >= self.cutoffs.l4_start:
            prior_vec = runtime.prior_tensor(key_states.device, key_states.shape[-2]).view(-1)
            key_states, value_states = level4_scale(
                key_states,
                value_states,
                prior_vec,
                self.config.alpha_k,
                self.config.alpha_v,
                self.config.gamma_min,
                self.config.gamma_max,
                self.config.eta_min,
                self.config.eta_max,
            )
            runtime.mark_level_call(4)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        num_layers = int(getattr(self.config_hf, "num_hidden_layers", self.layer_index + 1))
        if runtime:
            runtime.begin_decode_step(q_len=q_len, kv_len=attn_weights.shape[-1], num_layers=num_layers)

        if runtime and 1 in self.config.enabled_levels and runtime.should_apply_l12(
            layer_idx=self.layer_index,
            q_len=q_len,
            kv_len=attn_weights.shape[-1],
            default_layer_start=self.cutoffs.l12_start,
            default_layer_end=self.cutoffs.l12_end,
        ):
            base_attn_weights = attn_weights
            bias = runtime.prior_tensor(attn_weights.device, attn_weights.shape[-1])
            steered_attn_weights = level1_bias(attn_weights, bias, runtime.coeffs().beta_bias, cap=self.config.bias_cap)
            head_mask = runtime.get_head_mask_tensor(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                device=attn_weights.device,
                level="l1",
            )
            if head_mask is not None:
                assert head_mask.shape[1] == attn_weights.shape[1], (
                    f"Head mask/query mismatch at layer {self.layer_index}: "
                    f"mask_heads={head_mask.shape[1]}, query_heads={attn_weights.shape[1]}"
                )
                attn_weights = base_attn_weights + (steered_attn_weights - base_attn_weights) * head_mask.to(attn_weights.dtype)
                heads_active = float(head_mask.sum().item()) > 0.0
            else:
                attn_weights = steered_attn_weights
                heads_active = True
            runtime.steer_calls += 1
            runtime.mark_level_call(1)
            if abs(float(runtime.coeffs().beta_bias)) > 0.0 and heads_active:
                runtime.mark_layer_steered(layer_idx=self.layer_index, num_layers=num_layers)

        effective_mask = None
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            effective_mask = attention_mask

        if runtime and runtime.debug_assert_mask and effective_mask is not None:
            mask_blocked = effective_mask <= -1e4
            if mask_blocked.any():
                if mask_blocked.shape != attn_weights.shape:
                    mask_blocked = mask_blocked.expand_as(attn_weights)
                masked_logits = attn_weights.masked_select(mask_blocked)
                if masked_logits.numel() > 0:
                    assert torch.all(masked_logits < -1e4), "Masked logits became finite after steering bias."

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout if self.training else 0.0, training=self.training)

        if runtime:
            mean_heads = attn_probs.mean(dim=1)
            runtime.latest_attention = mean_heads[:, -1, :].detach()
            runtime.maybe_collect_head_stats(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                q_len=q_len,
                kv_len=attn_probs.shape[-1],
                default_layer_start=self.cutoffs.l12_start,
                default_layer_end=self.cutoffs.l12_end,
                attn_probs=attn_probs,
            )

        if runtime and 2 in self.config.enabled_levels and runtime.should_apply_l12(
            layer_idx=self.layer_index,
            q_len=q_len,
            kv_len=attn_probs.shape[-1],
            default_layer_start=self.cutoffs.l12_start,
            default_layer_end=self.cutoffs.l12_end,
        ):
            base_attn_probs = attn_probs
            scale = runtime.prior_tensor(attn_probs.device, attn_probs.shape[-1])
            steered_attn_probs = level2_post(attn_probs, scale, runtime.coeffs().beta_post)
            head_mask = runtime.get_head_mask_tensor(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                device=attn_probs.device,
                level="l2",
            )
            if head_mask is not None:
                assert head_mask.shape[1] == attn_probs.shape[1], (
                    f"Head mask/query mismatch at layer {self.layer_index}: "
                    f"mask_heads={head_mask.shape[1]}, query_heads={attn_probs.shape[1]}"
                )
                attn_probs = base_attn_probs + (steered_attn_probs - base_attn_probs) * head_mask.to(attn_probs.dtype)
                heads_active = float(head_mask.sum().item()) > 0.0
            else:
                attn_probs = steered_attn_probs
                heads_active = True
            runtime.steer_calls += 1
            runtime.mark_level_call(2)
            if abs(float(runtime.coeffs().beta_post)) > 0.0 and heads_active:
                runtime.mark_layer_steered(layer_idx=self.layer_index, num_layers=num_layers)

        if (
            runtime
            and q_len == 1
            and 3 in self.config.enabled_levels
            and str(runtime.config.residual_scale_mode).lower() == "agreement_gate"
        ):
            agree_rel = _agreement_rel_layer(
                runtime=runtime,
                attn_probs=attn_probs,
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                kv_len=attn_probs.shape[-1],
                device=attn_probs.device,
            )
            if agree_rel is not None:
                runtime.set_layer_agreement(layer_idx=self.layer_index, num_layers=num_layers, agree_rel=agree_rel)

        attn_probs_out = attn_probs.to(value_states.dtype)
        attn_output = torch.matmul(attn_probs_out, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            return attn_output, None
        return attn_output, attn_probs_out


class Qwen3MoeSteeringAttention(nn.Module):
    """
    Qwen3-MoE full-attention wrapper with L1/L2/L4 steering and attention telemetry.
    """

    def __init__(
        self,
        base_attn: Qwen3MoeAttention,
        runtime_getter: Callable[[], Optional["SteeringRuntime"]],
        layer_index: int,
        config: "SteeringConfig",
        cutoffs: LayerCutoffs,
    ) -> None:
        super().__init__()
        self.base = base_attn
        self.runtime_getter = runtime_getter
        self.layer_index = layer_index
        self.config = config
        self.cutoffs = cutoffs

        self.config_hf = base_attn.config
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj
        self.q_norm = base_attn.q_norm
        self.k_norm = base_attn.k_norm
        self.layer_idx = getattr(base_attn, "layer_idx", layer_index)
        self.head_dim = int(base_attn.head_dim)
        self.num_heads = int(getattr(base_attn.config, "num_attention_heads", 0))
        self.num_key_value_heads = int(getattr(base_attn.config, "num_key_value_heads", 0))
        self.num_key_value_groups = int(getattr(base_attn, "num_key_value_groups", 1))
        self.scaling = float(base_attn.scaling)
        self.attention_dropout = float(base_attn.attention_dropout)
        self.hidden_size = int(getattr(base_attn.config, "hidden_size", 0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        runtime = self.runtime_getter()
        output_attentions = bool(kwargs.get("output_attentions", False))

        input_shape = hidden_states.shape[:-1]
        bsz = hidden_states.shape[0]
        q_len = hidden_states.shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        target_device = hidden_states.device
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)
        if cache_position is not None and cache_position.device != target_device:
            cache_position = cache_position.to(target_device)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qwen3_moe(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv_qwen3_moe(key_states, self.num_key_value_groups)
        value_states = repeat_kv_qwen3_moe(value_states, self.num_key_value_groups)

        if runtime and 4 in self.config.enabled_levels and self.layer_index >= self.cutoffs.l4_start:
            prior_vec = runtime.prior_tensor(key_states.device, key_states.shape[-2]).view(-1)
            key_states, value_states = level4_scale(
                key_states,
                value_states,
                prior_vec,
                self.config.alpha_k,
                self.config.alpha_v,
                self.config.gamma_min,
                self.config.gamma_max,
                self.config.eta_min,
                self.config.eta_max,
            )
            runtime.mark_level_call(4)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        num_layers = int(getattr(self.config_hf, "num_hidden_layers", self.layer_index + 1))
        if runtime:
            runtime.begin_decode_step(q_len=q_len, kv_len=attn_weights.shape[-1], num_layers=num_layers)

        if runtime and 1 in self.config.enabled_levels and runtime.should_apply_l12(
            layer_idx=self.layer_index,
            q_len=q_len,
            kv_len=attn_weights.shape[-1],
            default_layer_start=self.cutoffs.l12_start,
            default_layer_end=self.cutoffs.l12_end,
        ):
            base_attn_weights = attn_weights
            bias = runtime.prior_tensor(attn_weights.device, attn_weights.shape[-1])
            steered_attn_weights = level1_bias(attn_weights, bias, runtime.coeffs().beta_bias, cap=self.config.bias_cap)
            head_mask = runtime.get_head_mask_tensor(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                device=attn_weights.device,
                level="l1",
            )
            if head_mask is not None:
                assert head_mask.shape[1] == attn_weights.shape[1], (
                    f"Head mask/query mismatch at layer {self.layer_index}: "
                    f"mask_heads={head_mask.shape[1]}, query_heads={attn_weights.shape[1]}"
                )
                attn_weights = base_attn_weights + (steered_attn_weights - base_attn_weights) * head_mask.to(attn_weights.dtype)
                heads_active = float(head_mask.sum().item()) > 0.0
            else:
                attn_weights = steered_attn_weights
                heads_active = True
            runtime.steer_calls += 1
            runtime.mark_level_call(1)
            if abs(float(runtime.coeffs().beta_bias)) > 0.0 and heads_active:
                runtime.mark_layer_steered(layer_idx=self.layer_index, num_layers=num_layers)

        effective_mask = None
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            effective_mask = attention_mask

        if runtime and runtime.debug_assert_mask and effective_mask is not None:
            mask_blocked = effective_mask <= -1e4
            if mask_blocked.any():
                if mask_blocked.shape != attn_weights.shape:
                    mask_blocked = mask_blocked.expand_as(attn_weights)
                masked_logits = attn_weights.masked_select(mask_blocked)
                if masked_logits.numel() > 0:
                    assert torch.all(masked_logits < -1e4), "Masked logits became finite after steering bias."

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout if self.training else 0.0, training=self.training)

        if runtime:
            mean_heads = attn_probs.mean(dim=1)
            runtime.latest_attention = mean_heads[:, -1, :].detach()
            runtime.maybe_collect_head_stats(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                q_len=q_len,
                kv_len=attn_probs.shape[-1],
                default_layer_start=self.cutoffs.l12_start,
                default_layer_end=self.cutoffs.l12_end,
                attn_probs=attn_probs,
            )

        if runtime and 2 in self.config.enabled_levels and runtime.should_apply_l12(
            layer_idx=self.layer_index,
            q_len=q_len,
            kv_len=attn_probs.shape[-1],
            default_layer_start=self.cutoffs.l12_start,
            default_layer_end=self.cutoffs.l12_end,
        ):
            base_attn_probs = attn_probs
            scale = runtime.prior_tensor(attn_probs.device, attn_probs.shape[-1])
            steered_attn_probs = level2_post(attn_probs, scale, runtime.coeffs().beta_post)
            head_mask = runtime.get_head_mask_tensor(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                device=attn_probs.device,
                level="l2",
            )
            if head_mask is not None:
                assert head_mask.shape[1] == attn_probs.shape[1], (
                    f"Head mask/query mismatch at layer {self.layer_index}: "
                    f"mask_heads={head_mask.shape[1]}, query_heads={attn_probs.shape[1]}"
                )
                attn_probs = base_attn_probs + (steered_attn_probs - base_attn_probs) * head_mask.to(attn_probs.dtype)
                heads_active = float(head_mask.sum().item()) > 0.0
            else:
                attn_probs = steered_attn_probs
                heads_active = True
            runtime.steer_calls += 1
            runtime.mark_level_call(2)
            if abs(float(runtime.coeffs().beta_post)) > 0.0 and heads_active:
                runtime.mark_layer_steered(layer_idx=self.layer_index, num_layers=num_layers)

        if (
            runtime
            and q_len == 1
            and 3 in self.config.enabled_levels
            and str(runtime.config.residual_scale_mode).lower() == "agreement_gate"
        ):
            agree_rel = _agreement_rel_layer(
                runtime=runtime,
                attn_probs=attn_probs,
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                kv_len=attn_probs.shape[-1],
                device=attn_probs.device,
            )
            if agree_rel is not None:
                runtime.set_layer_agreement(layer_idx=self.layer_index, num_layers=num_layers, agree_rel=agree_rel)

        attn_probs_out = attn_probs.to(value_states.dtype)
        attn_output = torch.matmul(attn_probs_out, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            return attn_output, None
        return attn_output, attn_probs_out


class Qwen3NextSteeringAttention(nn.Module):
    """
    Qwen3-Next full-attention wrapper with L1/L2/L4 steering and attention telemetry.

    Note: Qwen3-Next linear-attention layers are intentionally untouched.
    """

    def __init__(
        self,
        base_attn: Qwen3NextAttention,
        runtime_getter: Callable[[], Optional["SteeringRuntime"]],
        layer_index: int,
        config: "SteeringConfig",
        cutoffs: LayerCutoffs,
        full_layer_rank: Optional[int] = None,
        full_layer_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base = base_attn
        self.runtime_getter = runtime_getter
        self.layer_index = layer_index
        self.config = config
        self.cutoffs = cutoffs
        self.full_layer_rank = full_layer_rank
        self.full_layer_count = full_layer_count

        self.config_hf = base_attn.config
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj
        self.q_norm = base_attn.q_norm
        self.k_norm = base_attn.k_norm
        self.layer_idx = getattr(base_attn, "layer_idx", layer_index)
        self.head_dim = int(base_attn.head_dim)
        self.num_heads = int(getattr(base_attn.config, "num_attention_heads", 0))
        self.num_key_value_heads = int(getattr(base_attn.config, "num_key_value_heads", 0))
        self.num_key_value_groups = int(getattr(base_attn, "num_key_value_groups", 1))
        self.scaling = float(base_attn.scaling)
        self.attention_dropout = float(base_attn.attention_dropout)
        self.hidden_size = int(getattr(base_attn.config, "hidden_size", 0))

    def _allow_l4(self) -> bool:
        # For Qwen3-Next mixed stacks, decide L4 band over full-attention layers to avoid
        # dropping L4 entirely when late layers are linear_attention.
        if self.full_layer_rank is not None and self.full_layer_count is not None and self.full_layer_count > 0:
            full_l4_start = int(self.full_layer_count * 0.8)
            return int(self.full_layer_rank) >= full_l4_start
        return self.layer_index >= self.cutoffs.l4_start

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        runtime = self.runtime_getter()
        output_attentions = bool(kwargs.get("output_attentions", False))

        input_shape = hidden_states.shape[:-1]
        bsz = hidden_states.shape[0]
        q_len = hidden_states.shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        target_device = hidden_states.device
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)
        if cache_position is not None and cache_position.device != target_device:
            cache_position = cache_position.to(target_device)

        q_proj = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_proj, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qwen3_next(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv_qwen3_next(key_states, self.num_key_value_groups)
        value_states = repeat_kv_qwen3_next(value_states, self.num_key_value_groups)

        if runtime and 4 in self.config.enabled_levels and self._allow_l4():
            prior_vec = runtime.prior_tensor(key_states.device, key_states.shape[-2]).view(-1)
            key_states, value_states = level4_scale(
                key_states,
                value_states,
                prior_vec,
                self.config.alpha_k,
                self.config.alpha_v,
                self.config.gamma_min,
                self.config.gamma_max,
                self.config.eta_min,
                self.config.eta_max,
            )
            runtime.mark_level_call(4)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        num_layers = int(getattr(self.config_hf, "num_hidden_layers", self.layer_index + 1))
        if runtime:
            runtime.begin_decode_step(q_len=q_len, kv_len=attn_weights.shape[-1], num_layers=num_layers)

        if runtime and 1 in self.config.enabled_levels and runtime.should_apply_l12(
            layer_idx=self.layer_index,
            q_len=q_len,
            kv_len=attn_weights.shape[-1],
            default_layer_start=self.cutoffs.l12_start,
            default_layer_end=self.cutoffs.l12_end,
        ):
            base_attn_weights = attn_weights
            bias = runtime.prior_tensor(attn_weights.device, attn_weights.shape[-1])
            steered_attn_weights = level1_bias(attn_weights, bias, runtime.coeffs().beta_bias, cap=self.config.bias_cap)
            head_mask = runtime.get_head_mask_tensor(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                device=attn_weights.device,
                level="l1",
            )
            if head_mask is not None:
                assert head_mask.shape[1] == attn_weights.shape[1], (
                    f"Head mask/query mismatch at layer {self.layer_index}: "
                    f"mask_heads={head_mask.shape[1]}, query_heads={attn_weights.shape[1]}"
                )
                attn_weights = base_attn_weights + (steered_attn_weights - base_attn_weights) * head_mask.to(attn_weights.dtype)
                heads_active = float(head_mask.sum().item()) > 0.0
            else:
                attn_weights = steered_attn_weights
                heads_active = True
            runtime.steer_calls += 1
            runtime.mark_level_call(1)
            if abs(float(runtime.coeffs().beta_bias)) > 0.0 and heads_active:
                runtime.mark_layer_steered(layer_idx=self.layer_index, num_layers=num_layers)

        effective_mask = None
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            effective_mask = attention_mask

        if runtime and runtime.debug_assert_mask and effective_mask is not None:
            mask_blocked = effective_mask <= -1e4
            if mask_blocked.any():
                if mask_blocked.shape != attn_weights.shape:
                    mask_blocked = mask_blocked.expand_as(attn_weights)
                masked_logits = attn_weights.masked_select(mask_blocked)
                if masked_logits.numel() > 0:
                    assert torch.all(masked_logits < -1e4), "Masked logits became finite after steering bias."

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout if self.training else 0.0, training=self.training)

        if runtime:
            mean_heads = attn_probs.mean(dim=1)
            runtime.latest_attention = mean_heads[:, -1, :].detach()
            runtime.maybe_collect_head_stats(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                q_len=q_len,
                kv_len=attn_probs.shape[-1],
                default_layer_start=self.cutoffs.l12_start,
                default_layer_end=self.cutoffs.l12_end,
                attn_probs=attn_probs,
            )

        if runtime and 2 in self.config.enabled_levels and runtime.should_apply_l12(
            layer_idx=self.layer_index,
            q_len=q_len,
            kv_len=attn_probs.shape[-1],
            default_layer_start=self.cutoffs.l12_start,
            default_layer_end=self.cutoffs.l12_end,
        ):
            base_attn_probs = attn_probs
            scale = runtime.prior_tensor(attn_probs.device, attn_probs.shape[-1])
            steered_attn_probs = level2_post(attn_probs, scale, runtime.coeffs().beta_post)
            head_mask = runtime.get_head_mask_tensor(
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                device=attn_probs.device,
                level="l2",
            )
            if head_mask is not None:
                assert head_mask.shape[1] == attn_probs.shape[1], (
                    f"Head mask/query mismatch at layer {self.layer_index}: "
                    f"mask_heads={head_mask.shape[1]}, query_heads={attn_probs.shape[1]}"
                )
                attn_probs = base_attn_probs + (steered_attn_probs - base_attn_probs) * head_mask.to(attn_probs.dtype)
                heads_active = float(head_mask.sum().item()) > 0.0
            else:
                attn_probs = steered_attn_probs
                heads_active = True
            runtime.steer_calls += 1
            runtime.mark_level_call(2)
            if abs(float(runtime.coeffs().beta_post)) > 0.0 and heads_active:
                runtime.mark_layer_steered(layer_idx=self.layer_index, num_layers=num_layers)

        if (
            runtime
            and q_len == 1
            and 3 in self.config.enabled_levels
            and str(runtime.config.residual_scale_mode).lower() == "agreement_gate"
        ):
            agree_rel = _agreement_rel_layer(
                runtime=runtime,
                attn_probs=attn_probs,
                layer_idx=self.layer_index,
                num_layers=num_layers,
                num_heads=self.num_heads,
                kv_len=attn_probs.shape[-1],
                device=attn_probs.device,
            )
            if agree_rel is not None:
                runtime.set_layer_agreement(layer_idx=self.layer_index, num_layers=num_layers, agree_rel=agree_rel)

        attn_probs_out = attn_probs.to(value_states.dtype)
        attn_output = torch.matmul(attn_probs_out, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            return attn_output, None
        return attn_output, attn_probs_out


def patch_qwen3_decoder_layer(
    layer: Qwen3DecoderLayer,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
    layer_idx: int,
    cutoffs: LayerCutoffs,
) -> None:
    num_layers = int(getattr(getattr(layer.self_attn, "config_hf", None) or getattr(layer.self_attn, "config", None), "num_hidden_layers", 80))

    def steered_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        runtime = runtime_getter()
        coeffs = runtime.coeffs() if runtime else None

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = hidden_states.to(dtype=residual.dtype, device=residual.device, non_blocking=True)
        if runtime and 3 in config.enabled_levels and runtime.should_apply_residual(
            layer_idx=layer_idx,
            seq_len=hidden_states.shape[1],
            default_layer_start=cutoffs.l12_start,
            default_layer_end=cutoffs.l12_end,
        ):
            lambda_attn, _ = runtime.residual_lambdas(layer_idx=layer_idx, num_layers=num_layers, coeffs=coeffs)
            hidden_states = (hidden_states.float() * float(lambda_attn)).to(
                dtype=residual.dtype,
                device=residual.device,
                non_blocking=True,
            )
            runtime.mark_residual_call("attn")
        residual = residual.to(hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states.to(dtype=residual.dtype, device=residual.device, non_blocking=True)
        if runtime and 3 in config.enabled_levels and runtime.should_apply_residual(
            layer_idx=layer_idx,
            seq_len=hidden_states.shape[1],
            default_layer_start=cutoffs.l12_start,
            default_layer_end=cutoffs.l12_end,
        ):
            _, lambda_mlp = runtime.residual_lambdas(layer_idx=layer_idx, num_layers=num_layers, coeffs=coeffs)
            hidden_states = (hidden_states.float() * float(lambda_mlp)).to(
                dtype=residual.dtype,
                device=residual.device,
                non_blocking=True,
            )
            runtime.mark_residual_call("mlp")
        residual = residual.to(hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        hidden_states = residual + hidden_states
        return hidden_states

    layer.forward = steered_forward  # type: ignore[method-assign]


def patch_qwen3_moe_decoder_layer(
    layer: Qwen3MoeDecoderLayer,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
    layer_idx: int,
    cutoffs: LayerCutoffs,
) -> None:
    num_layers = int(getattr(getattr(layer.self_attn, "config_hf", None) or getattr(layer.self_attn, "config", None), "num_hidden_layers", 80))

    def steered_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        runtime = runtime_getter()
        coeffs = runtime.coeffs() if runtime else None

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = hidden_states.to(dtype=residual.dtype, device=residual.device, non_blocking=True)
        if runtime and 3 in config.enabled_levels and runtime.should_apply_residual(
            layer_idx=layer_idx,
            seq_len=hidden_states.shape[1],
            default_layer_start=cutoffs.l12_start,
            default_layer_end=cutoffs.l12_end,
        ):
            lambda_attn, _ = runtime.residual_lambdas(layer_idx=layer_idx, num_layers=num_layers, coeffs=coeffs)
            hidden_states = (hidden_states.float() * float(lambda_attn)).to(
                dtype=residual.dtype,
                device=residual.device,
                non_blocking=True,
            )
            runtime.mark_residual_call("attn")
        residual = residual.to(hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states.to(dtype=residual.dtype, device=residual.device, non_blocking=True)
        if runtime and 3 in config.enabled_levels and runtime.should_apply_residual(
            layer_idx=layer_idx,
            seq_len=hidden_states.shape[1],
            default_layer_start=cutoffs.l12_start,
            default_layer_end=cutoffs.l12_end,
        ):
            _, lambda_mlp = runtime.residual_lambdas(layer_idx=layer_idx, num_layers=num_layers, coeffs=coeffs)
            hidden_states = (hidden_states.float() * float(lambda_mlp)).to(
                dtype=residual.dtype,
                device=residual.device,
                non_blocking=True,
            )
            runtime.mark_residual_call("mlp")
        residual = residual.to(hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        hidden_states = residual + hidden_states
        return hidden_states

    layer.forward = steered_forward  # type: ignore[method-assign]


def patch_qwen3_next_decoder_layer(
    layer: Qwen3NextDecoderLayer,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
    layer_idx: int,
    cutoffs: LayerCutoffs,
    num_layers: int,
) -> None:
    def steered_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        runtime = runtime_getter()
        coeffs = runtime.coeffs() if runtime else None

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        if getattr(layer, "layer_type", None) == "linear_attention":
            hidden_states = layer.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif getattr(layer, "layer_type", None) == "full_attention":
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown Qwen3-Next layer type at index {layer_idx}: {getattr(layer, 'layer_type', None)!r}")

        hidden_states = hidden_states.to(dtype=residual.dtype, device=residual.device, non_blocking=True)
        if runtime and 3 in config.enabled_levels and runtime.should_apply_residual(
            layer_idx=layer_idx,
            seq_len=hidden_states.shape[1],
            default_layer_start=cutoffs.l12_start,
            default_layer_end=cutoffs.l12_end,
        ):
            lambda_attn, _ = runtime.residual_lambdas(layer_idx=layer_idx, num_layers=num_layers, coeffs=coeffs)
            hidden_states = (hidden_states.float() * float(lambda_attn)).to(
                dtype=residual.dtype,
                device=residual.device,
                non_blocking=True,
            )
            runtime.mark_residual_call("attn")
        residual = residual.to(hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states.to(dtype=residual.dtype, device=residual.device, non_blocking=True)
        if runtime and 3 in config.enabled_levels and runtime.should_apply_residual(
            layer_idx=layer_idx,
            seq_len=hidden_states.shape[1],
            default_layer_start=cutoffs.l12_start,
            default_layer_end=cutoffs.l12_end,
        ):
            _, lambda_mlp = runtime.residual_lambdas(layer_idx=layer_idx, num_layers=num_layers, coeffs=coeffs)
            hidden_states = (hidden_states.float() * float(lambda_mlp)).to(
                dtype=residual.dtype,
                device=residual.device,
                non_blocking=True,
            )
            runtime.mark_residual_call("mlp")
        residual = residual.to(hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        hidden_states = residual + hidden_states
        return hidden_states

    layer.forward = steered_forward  # type: ignore[method-assign]


def install_qwen3_steering(
    model,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
) -> None:
    layers = get_decoder_layers(model)
    cutoffs = compute_default_cutoffs(len(layers))
    if len(layers) == 0:
        raise ValueError("Qwen3 steering install failed: model has no decoder layers.")

    first = layers[0]
    if isinstance(first, Qwen3DecoderLayer):
        for idx, layer in enumerate(layers):
            if not isinstance(layer, Qwen3DecoderLayer):
                raise ValueError("Mixed layer classes detected in Qwen3 model; refusing to install steering.")
            layer.self_attn = Qwen3SteeringAttention(
                layer.self_attn,
                runtime_getter=runtime_getter,
                layer_index=idx,
                config=config,
                cutoffs=cutoffs,
            )
            patch_qwen3_decoder_layer(layer, runtime_getter, config, idx, cutoffs)
        return

    if isinstance(first, Qwen3MoeDecoderLayer):
        for idx, layer in enumerate(layers):
            if not isinstance(layer, Qwen3MoeDecoderLayer):
                raise ValueError("Mixed layer classes detected in Qwen3-MoE model; refusing to install steering.")
            layer.self_attn = Qwen3MoeSteeringAttention(
                layer.self_attn,
                runtime_getter=runtime_getter,
                layer_index=idx,
                config=config,
                cutoffs=cutoffs,
            )
            patch_qwen3_moe_decoder_layer(layer, runtime_getter, config, idx, cutoffs)
        return

    if isinstance(first, Qwen3NextDecoderLayer):
        full_count = 0
        linear_count = 0
        full_layer_indices = [i for i, lyr in enumerate(layers) if str(getattr(lyr, "layer_type", "")) == "full_attention"]
        full_index_to_rank = {idx: rank for rank, idx in enumerate(full_layer_indices)}
        full_layer_total = len(full_layer_indices)
        num_layers = len(layers)
        for idx, layer in enumerate(layers):
            if not isinstance(layer, Qwen3NextDecoderLayer):
                raise ValueError("Mixed layer classes detected in Qwen3-Next model; refusing to install steering.")

            layer_type = str(getattr(layer, "layer_type", ""))
            if layer_type == "full_attention":
                if not hasattr(layer, "self_attn"):
                    raise ValueError(f"Qwen3-Next full_attention layer {idx} missing self_attn.")
                layer.self_attn = Qwen3NextSteeringAttention(
                    layer.self_attn,
                    runtime_getter=runtime_getter,
                    layer_index=idx,
                    config=config,
                    cutoffs=cutoffs,
                    full_layer_rank=full_index_to_rank.get(idx),
                    full_layer_count=full_layer_total,
                )
                full_count += 1
            elif layer_type == "linear_attention":
                linear_count += 1
            else:
                raise ValueError(f"Unsupported Qwen3-Next layer_type={layer_type!r} at layer {idx}.")

            patch_qwen3_next_decoder_layer(layer, runtime_getter, config, idx, cutoffs, num_layers)

        if full_count <= 0:
            raise ValueError("Qwen3-Next steering install failed: found zero full_attention layers.")
        print(f"[Steering] Qwen3-Next patched layers: full_attention={full_count}, linear_attention={linear_count}")
        return

    raise ValueError(
        "Unsupported Qwen3 decoder layer class for steering: "
        f"{first.__class__.__name__}."
    )
