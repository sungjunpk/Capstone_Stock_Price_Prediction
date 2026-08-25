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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    """softplus(x) = y 를 만족하는 x. 헤드 bias 초기화용."""
    return math.log(math.expm1(max(y, 1e-6)))


class QuantileHead(nn.Module):
    """분위 헤드.

    ⚠️ **초기화가 성능을 좌우한다.** 기본 초기화로 두면 softplus(0)=0.693 이라
    초기 분위 간격이 0.69 인데 5일 수익률의 전체 분위 폭은 0.124 다.
    출력이 타깃보다 9배 커서, 모델이 신호를 배우기 전에 스케일 줄이기부터 해야 한다.
    실측: 이 상태로 학습하면 무조건부 분위수 기준선보다 5% 나쁜 결과가 나왔다.

    init_quantiles 로 학습 데이터의 무조건부 분위수를 넣으면 **거기서 출발**한다.
    즉 초기 모델이 곧 기준선이고, 학습은 그 위에서 개선만 하면 된다.
    """

    def __init__(self, d_model: int, n_quantiles: int = 3, hidden: int | None = None,
                 dropout: float = 0.1, init_quantiles: tuple[float, ...] | None = None):
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
        self._init_output(init_quantiles)

    def _init_output(self, init_quantiles: tuple[float, ...] | None) -> None:
        last = self.net[-1]
        # 가중치를 아주 작게(0은 아니게) 시작한다.
        #   - 크면: 초기 출력이 기준선에서 멀어져 스케일 맞추기부터 해야 한다
        #   - 정확히 0이면: d(out)/d(pooled) = W = 0 이라 **인코더까지 경사가 안 간다**
        # std 1e-4 면 초기 출력 편차가 0.001 수준이라 기준선과 사실상 같으면서 경사는 흐른다.
        nn.init.normal_(last.weight, std=1e-4)

        if init_quantiles is None:
            # 스케일 정보가 없으면 최소한 softplus(0)=0.693 문제만이라도 없앤다
            bias = [0.0] + [_inv_softplus(0.02)] * (self.n_quantiles - 1)
        else:
            if len(init_quantiles) != self.n_quantiles:
                raise ValueError(
                    f"init_quantiles 길이 {len(init_quantiles)} != 분위 수 {self.n_quantiles}"
                )
            q = sorted(init_quantiles)
            bias = [q[0]] + [_inv_softplus(q[i + 1] - q[i]) for i in range(len(q) - 1)]
        with torch.no_grad():
            last.bias.copy_(torch.tensor(bias, dtype=last.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) → (B, Q). 항상 q[0] <= q[1] <= ... 를 만족한다."""
        raw = self.net(x)
        if self.n_quantiles == 1:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)
