"""글로벌-로컬 크로스어텐션.

국내 종목 시퀀스가 Query, 매크로/글로벌 지표 시퀀스가 Key/Value.
"이 종목의 이 시점 표현이 매크로의 어느 구간과 관련 있는가"를 학습한다.

방향이 중요하다: 매크로가 Query 가 되면 출력이 매크로 시퀀스 모양이 되어
종목별 예측을 낼 수 없다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """query: (B, N, d) 종목 / memory: (B, M, d) 매크로 → (B, N, d)"""
        q = self.norm_q(query)
        kv = self.norm_kv(memory)
        out, w = self.attn(q, kv, kv, need_weights=need_weights, average_attn_weights=True)
        x = query + self.dropout(out)
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x, w
