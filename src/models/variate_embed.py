"""변수 임베딩 — 시계열 하나를 통째로 토큰 하나로.

시간축을 잘라 토큰을 만드는 `patch_embed.py` 와 정반대다. lookback 전체를 한 번에
투영해 **변수 하나당 토큰 하나**를 만든다. 어텐션이 시점이 아니라 변수 사이에서 걸린다.

두 곳에서 쓴다:
  - 매크로 경로 (exog_mode="variate_token") — TimeXer 의 외생변수 처리
  - 종목 경로 (endog_mode="variate")        — iTransformer 의 내생 백본

토큰이 변수를 가리키므로 어텐션 가중치가 "어느 **변수**를 봤나" 가 된다.
패치 방식에서는 "어느 **시점**을 봤나" 라서 해석 근거로 쓰기 어려웠다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VariateEmbedding(nn.Module):
    def __init__(self, lookback: int, n_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        # 투영은 변수끼리 공유한다. 변수를 구분하는 것은 아래 var 임베딩이다 —
        # 변수마다 별도 투영을 두면 파라미터가 n_vars 배가 되고 과적합한다.
        self.proj = nn.Linear(lookback, d_model)
        self.var = nn.Parameter(torch.zeros(1, n_vars, d_model))
        nn.init.trunc_normal_(self.var, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, C, d)"""
        if x.shape[-1] != self.n_vars:
            raise ValueError(f"변수 수 불일치: {x.shape[-1]} != {self.n_vars}")
        return self.dropout(self.proj(x.transpose(1, 2)) + self.var)
