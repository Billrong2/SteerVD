from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerCutoffs:
    """
    Default layer ranges for steering stages.

    Default behavior:
    - attention steering: last 4 layers
    - key/value scaling: deeper layers
    - residual scaling: middle-to-late layers

    Residual and key/value scaling remain scaled by model depth to preserve prior behavior.
    """

    l12_start: int
    l12_end: int
    l4_start: int
    l3_start: int
    l3_end: int


def compute_default_cutoffs(num_layers: int) -> LayerCutoffs:
    if num_layers <= 0:
        return LayerCutoffs(l12_start=0, l12_end=0, l4_start=0, l3_start=0, l3_end=0)

    def clamp_idx(idx: int) -> int:
        return max(0, min(num_layers - 1, idx))

    # Default attention-steering band: only steer the final 4 layers.
    l12_start = clamp_idx(num_layers - 4)
    l12_end = clamp_idx(num_layers - 1)
    l4_start = clamp_idx(int(num_layers * 0.75))

    l3_start = clamp_idx(int(num_layers * 0.50))
    l3_end = clamp_idx(int(num_layers * 0.7875))
    if l3_end < l3_start:
        l3_end = l3_start

    return LayerCutoffs(
        l12_start=l12_start,
        l12_end=l12_end,
        l4_start=l4_start,
        l3_start=l3_start,
        l3_end=l3_end,
    )


def get_decoder_layers(model: Any):
    """
    Return decoder layers for common HF causal LMs
    (LLaMA/CodeLlama, Qwen2.*, Qwen3*, DeepSeek-V2).
    """

    # LlamaForCausalLM / Qwen2ForCausalLM expose: model.model.layers
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise AttributeError(
            "Unsupported model structure: expected `model.model.layers` to exist for steering backends."
        )
    return layers
