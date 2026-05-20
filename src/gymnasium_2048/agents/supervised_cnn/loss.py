from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    legal_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distill teacher probabilities while excluding illegal actions."""
    if logits.shape != target_probs.shape or logits.shape != legal_mask.shape:
        raise ValueError("logits, target_probs, and legal_mask must have matching shapes")

    temperature = max(float(temperature), 1e-6)
    mask = legal_mask.bool()
    valid = mask.any(dim=-1)

    masked_logits = logits / temperature
    masked_logits = masked_logits.masked_fill(~mask, -1.0e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)

    targets = target_probs.float().masked_fill(~mask, 0.0)
    target_sums = targets.sum(dim=-1, keepdim=True)
    mask_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform_targets = mask.float() / mask_counts.float()
    targets = torch.where(target_sums > 1.0e-8, targets / target_sums.clamp_min(1.0e-8), uniform_targets)

    per_sample = -(targets * log_probs).sum(dim=-1) * (temperature**2)
    per_sample = torch.where(valid, per_sample, torch.zeros_like(per_sample))

    if sample_weight is not None:
        weights = sample_weight.float().to(per_sample.device)
        per_sample = per_sample * weights
        denom = (weights * valid.float()).sum().clamp_min(1.0)
    else:
        denom = valid.float().sum().clamp_min(1.0)
    return per_sample.sum() / denom
