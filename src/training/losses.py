"""Quantile(Pinball) Loss.

10/50/90 분위를 동시에 학습한다. 분위 교차(q10 > q50 > q90)가 생기면
신뢰구간 폭 기반 기권 로직이 무너지므로 교차 페널티를 옵션으로 둔다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    pred:      (B, ..., Q) 분위별 예측
    target:    (B, ...) 실제값 — 마지막에 Q 축이 없다
    quantiles: (Q,)
    """
    if target.dim() == pred.dim() - 1:
        target = target.unsqueeze(-1)
    errors = target - pred                      # (B, ..., Q)
    q = quantiles.view(*([1] * (errors.dim() - 1)), -1).to(errors)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def crossing_penalty(pred: torch.Tensor) -> torch.Tensor:
    """분위가 단조증가하지 않는 만큼만 벌점. 이미 정렬돼 있으면 0."""
    diffs = pred[..., 1:] - pred[..., :-1]
    return torch.relu(-diffs).mean()


class QuantileLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9), crossing_weight: float = 0.0):
        super().__init__()
        q = torch.as_tensor(quantiles, dtype=torch.float32)
        if not torch.all(q[1:] > q[:-1]):
            raise ValueError("quantiles 는 오름차순이어야 한다")
        self.register_buffer("quantiles", q)
        self.crossing_weight = crossing_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pinball_loss(pred, target, self.quantiles)
        if self.crossing_weight > 0:
            loss = loss + self.crossing_weight * crossing_penalty(pred)
        return loss
