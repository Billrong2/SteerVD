from __future__ import annotations

import math
from typing import Callable, Optional, TYPE_CHECKING

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv,
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


class Qwen2SteeringAttention(nn.Module):
    """
    Qwen2/Qwen2.5 attention wrapper with L1/L2/L4 steering + telemetry.

    Supports the current Transformers signature (position_embeddings/past_key_values/cache_position).
    """

    def __init__(
        self,
        base_attn: Qwen2Attention,
        runtime_getter: Callable[[], Optional["SteeringRuntime"]],
        layer_index: int,
        config: "SteeringConfig",
        cutoffs: LayerCutoffs | None = None,
    ) -> None:
        super().__init__()
        self.base = base_attn
        self.runtime_getter = runtime_getter
        self.layer_index = layer_index
        self.config = config
        if cutoffs is None:
            num_layers = int(getattr(getattr(base_attn, "config", None), "num_hidden_layers", 80))
            cutoffs = compute_default_cutoffs(num_layers)
        self.cutoffs = cutoffs

        self.config_hf = base_attn.config
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

        self.head_dim = int(base_attn.head_dim)
        self.num_heads = int(getattr(base_attn, "num_heads", getattr(base_attn.config, "num_attention_heads", 0)))
        self.num_key_value_heads = int(
            getattr(base_attn, "num_key_value_heads", getattr(base_attn.config, "num_key_value_heads", self.num_heads))
        )
        self.num_key_value_groups = int(getattr(base_attn, "num_key_value_groups", 1))
        self.scaling = float(getattr(base_attn, "scaling", 1.0 / math.sqrt(max(self.head_dim, 1))))
        self.attention_dropout = float(base_attn.attention_dropout)
        self.hidden_size = int(getattr(base_attn, "hidden_size", getattr(base_attn.config, "hidden_size", 0)))
        self.layer_idx = getattr(base_attn, "layer_idx", layer_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, object | None]:
        runtime = self.runtime_getter()
        past_key_value = kwargs.pop("past_key_values", past_key_value)
        output_attentions = bool(output_attentions or kwargs.get("output_attentions", False))

        bsz, q_len, _ = hidden_states.size()

        target_device = hidden_states.device
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)
        if position_ids is not None and position_ids.device != target_device:
            position_ids = position_ids.to(target_device)
        if cache_position is not None and cache_position.device != target_device:
            cache_position = cache_position.to(target_device)

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and hasattr(past_key_value, "get_usable_length"):
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_embeddings is None:
            cos, sin = self.base.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = position_embeddings
            if cos.device != target_device:
                cos = cos.to(target_device)
            if sin.device != target_device:
                sin = sin.to(target_device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

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

        num_layers = int(getattr(self.config_hf, "num_hidden_layers", self.layer_index + 1))
        if runtime:
            runtime.begin_decode_step(q_len=q_len, kv_len=key_states.shape[-2], num_layers=num_layers)

        # Decode-only steering does not need prompt-prefill attention tensors.
        # Use SDPA here so long C/C++ functions do not materialize an O(L^2)
        # attention probability tensor before the first decode step.
        if q_len != 1 and not output_attentions and (runtime is None or runtime.decode_only):
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

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

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            return attn_output, None, past_key_value
        return attn_output, attn_probs_out, past_key_value


def patch_decoder_layer(
    layer: Qwen2DecoderLayer,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
    layer_idx: int,
    cutoffs: LayerCutoffs | None = None,
) -> None:
    attn = getattr(layer, "self_attn", None)
    cfg = getattr(attn, "config_hf", None) or getattr(attn, "config", None)
    num_layers = int(getattr(cfg, "num_hidden_layers", 80))
    if cutoffs is None:
        cutoffs = compute_default_cutoffs(num_layers)

    def steered_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        runtime = runtime_getter()
        coeffs = runtime.coeffs() if runtime else None
        cache_obj = past_key_value if past_key_value is not None else past_key_values

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache_obj,
            output_attentions=output_attentions,
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

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    layer.forward = steered_forward  # type: ignore[method-assign]


def install_qwen2_steering(
    model,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
) -> None:
    layers = get_decoder_layers(model)
    cutoffs = compute_default_cutoffs(len(layers))
    for idx, layer in enumerate(layers):
        layer.self_attn = Qwen2SteeringAttention(
            layer.self_attn,
            runtime_getter=runtime_getter,
            layer_index=idx,
            config=config,
            cutoffs=cutoffs,
        )
        patch_decoder_layer(layer, runtime_getter, config, idx, cutoffs=cutoffs)
