from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch
from torch import nn
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2DecoderLayer,
    apply_rotary_emb,
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


class DeepseekV2SteeringAttention(nn.Module):
    """
    DeepSeek-V2 eager-attention wrapper with attention steering, key/value scaling, and telemetry.
    """

    def __init__(
        self,
        base_attn: DeepseekV2Attention,
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

        cls_name = base_attn.__class__.__name__.lower()
        if "flash" in cls_name:
            raise ValueError(
                "DeepSeek-V2 steering requires eager attention implementation; found flash attention class "
                f"{base_attn.__class__.__name__!r}."
            )

        self.config_hf = base_attn.config
        self.layer_idx = getattr(base_attn, "layer_idx", layer_index)
        self.attention_dropout = float(base_attn.attention_dropout)
        self.hidden_size = int(base_attn.hidden_size)
        self.num_heads = int(base_attn.num_heads)

        self.q_lora_rank = base_attn.q_lora_rank
        self.qk_rope_head_dim = int(base_attn.qk_rope_head_dim)
        self.kv_lora_rank = int(base_attn.kv_lora_rank)
        self.v_head_dim = int(base_attn.v_head_dim)
        self.qk_nope_head_dim = int(base_attn.qk_nope_head_dim)
        self.qk_head_dim = int(base_attn.qk_head_dim)
        self.scaling = float(base_attn.scaling)

        self.q_proj = getattr(base_attn, "q_proj", None)
        self.q_a_proj = getattr(base_attn, "q_a_proj", None)
        self.q_a_layernorm = getattr(base_attn, "q_a_layernorm", None)
        self.q_b_proj = getattr(base_attn, "q_b_proj", None)
        self.kv_a_proj_with_mqa = base_attn.kv_a_proj_with_mqa
        self.kv_a_layernorm = base_attn.kv_a_layernorm
        self.kv_b_proj = base_attn.kv_b_proj
        self.o_proj = base_attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        runtime = self.runtime_getter()
        output_attentions = bool(kwargs.get("output_attentions", False))

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            if self.q_proj is None:
                raise RuntimeError("DeepSeek-V2 attention missing q_proj with q_lora_rank=None.")
            q = self.q_proj(hidden_states)
        else:
            if self.q_a_proj is None or self.q_a_layernorm is None or self.q_b_proj is None:
                raise RuntimeError("DeepSeek-V2 attention missing q_lora projection modules.")
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        q = q.view(query_shape).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_nope = self.kv_b_proj(self.kv_a_layernorm(k_nope)).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if position_embeddings is None:
            raise ValueError("DeepSeek-V2 attention requires position_embeddings for RoPE.")
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))

        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
        q_len = int(attn_weights.shape[-2])
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
            if attention_mask.size(-2) != q_len:
                raise ValueError(
                    f"DeepSeek-V2 attention mask query length mismatch: expected {q_len}, got {attention_mask.size(-2)}"
                )
            if attention_mask.size(-1) != attn_weights.shape[-1]:
                raise ValueError(
                    "DeepSeek-V2 attention mask key length mismatch: "
                    f"expected {attn_weights.shape[-1]}, got {attention_mask.size(-1)}"
                )
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
        attn_probs = nn.functional.dropout(
            attn_probs,
            p=self.attention_dropout if self.training else 0.0,
            training=self.training,
        )

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
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be {(batch_size, self.num_heads, q_len, self.v_head_dim)}, "
                f"got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            return attn_output, None
        return attn_output, attn_probs_out


def patch_deepseek_v2_decoder_layer(
    layer: DeepseekV2DecoderLayer,
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
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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


def install_deepseek_v2_steering(
    model,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
) -> None:
    layers = get_decoder_layers(model)
    cutoffs = compute_default_cutoffs(len(layers))
    if len(layers) == 0:
        raise ValueError("DeepSeek-V2 steering install failed: model has no decoder layers.")

    for idx, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            raise ValueError(f"DeepSeek-V2 layer {idx} missing self_attn.")
        layer.self_attn = DeepseekV2SteeringAttention(
            layer.self_attn,
            runtime_getter=runtime_getter,
            layer_index=idx,
            config=config,
            cutoffs=cutoffs,
        )
        patch_deepseek_v2_decoder_layer(layer, runtime_getter, config, idx, cutoffs)
