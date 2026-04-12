from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence, TYPE_CHECKING

import torch
from transformers.generation.logits_process import LogitsProcessor

if TYPE_CHECKING:
    from .runtime import SteeringRuntime


class PointerBiasProcessor(LogitsProcessor):
    def __init__(self, runtime: "SteeringRuntime"):
        self.runtime = runtime

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        runtime = self.runtime
        runtime.pointer_calls_total += 1
        coeffs = runtime.coeffs()
        beta = coeffs.beta_ptr
        if beta == 0.0:
            runtime.pointer_beta_zero_steps += 1
            runtime.advance()
            return scores
        if runtime.latest_attention is None:
            runtime.pointer_missing_attention_steps += 1
            raise RuntimeError(
                "Level-5 pointer steering requires latest_attention, but it is missing at decode step "
                f"{runtime.step_index}."
            )

        attention_vector = runtime.latest_attention  # [batch, k_len]
        bias = torch.zeros_like(scores)
        for vocab_id, positions in runtime.pointer_mapping.items():
            mass = attention_vector[:, positions].sum(dim=1)
            bias[:, vocab_id] += beta * mass
        scores = scores + bias
        runtime.pointer_bias_applied_steps += 1
        runtime.mark_level_call(5)
        runtime.advance()
        return scores


class StepAdvanceProcessor(LogitsProcessor):
    def __init__(self, runtime: "SteeringRuntime"):
        self.runtime = runtime

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.runtime.advance()
        return scores


def build_pointer_mapping(prompt_tokens: Sequence[int]) -> Dict[int, Sequence[int]]:
    mapping = defaultdict(list)
    for idx, token_id in enumerate(prompt_tokens):
        mapping[token_id].append(idx)
    return mapping
