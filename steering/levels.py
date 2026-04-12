from __future__ import annotations

import torch


def level1_bias(attn_scores: torch.Tensor, prior: torch.Tensor, beta: float, eps: float = 1e-6, cap: float | None = None):
    if beta == 0.0:
        return attn_scores
    # Compute in fp32 for numerical stability, then cast back to logits dtype.
    bias = beta * torch.log((prior + eps).to(torch.float32))
    if cap is not None:
        bias = torch.clamp(bias, min=-cap, max=cap)
    return attn_scores + bias.to(attn_scores.dtype)


def level2_post(attn_probs: torch.Tensor, prior: torch.Tensor, beta: float, eps: float = 1e-6):
    if beta == 0.0:
        return attn_probs
    weights = torch.pow(prior + eps, beta)
    attn = attn_probs * weights
    attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
    return attn


def level3_residual(attn_resid: torch.Tensor, mlp_resid: torch.Tensor, lam_attn: float, lam_mlp: float):
    return attn_resid * lam_attn, mlp_resid * lam_mlp


def level4_scale(k: torch.Tensor, v: torch.Tensor, prior: torch.Tensor, alpha_k: float, alpha_v: float,
                 gamma_min: float, gamma_max: float, eta_min: float, eta_max: float):
    if alpha_k != 0.0:
        gamma = torch.clamp(1.0 + alpha_k * prior, gamma_min, gamma_max)
        k = k * gamma[..., None]
    if alpha_v != 0.0:
        eta = torch.clamp(1.0 + alpha_v * prior, eta_min, eta_max)
        v = v * eta[..., None]
    return k, v
