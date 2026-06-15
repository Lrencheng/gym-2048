from __future__ import annotations

import torch
from torch.nn import functional as F


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    kind: str = "huber",
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shapes")
    if kind == "huber":
        return F.smooth_l1_loss(predictions.float(), targets.float())
    if kind == "mse":
        return F.mse_loss(predictions.float(), targets.float())
    raise ValueError(f"unknown regression loss: {kind!r}")
