"""Quantile 출력 헤드 — 10/50/90 분위.

**분위 단조성을 구조로 보장한다.** 첫 분위를 그대로 내고, 이후는 softplus 증분을 누적한다:
    q10 = base
    q50 = q10 + softplus(d1)
    q90 = q50 + softplus(d2)

손실 페널티(crossing_penalty)만으로는 교차가 남을 수 있는데, 교차가 생기면
src/trading/signal.py 의 QuantilePrediction 이 예외를 던져 백테스트가 통째로 죽는다.
구조로 막으면 그 사고 자체가 불가능해진다. 페널티는 보조로 유지한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileHead(nn.Module):
    def __init__(self, d_model: int, n_quantiles: int = 3, hidden: int | None = None,
                 dropout: float = 0.1):
        super().__init__()
        if n_quantiles < 1:
            raise ValueError("분위는 1개 이상이어야 한다")
        self.n_quantiles = n_quantiles
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) → (B, Q). 항상 q[0] <= q[1] <= ... 를 만족한다."""
        raw = self.net(x)
        if self.n_quantiles == 1:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)
