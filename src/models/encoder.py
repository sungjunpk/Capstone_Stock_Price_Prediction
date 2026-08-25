"""Pre-LN Transformer 인코더.

Pre-LN 을 쓰는 이유: Post-LN 은 warmup 에 민감하고 깊어지면 학습이 불안정하다.
데이터가 35만 샘플이라 학습을 여러 번 돌릴 여유가 없어 안정성을 택했다.

채널 독립 처리: (B, C, N, d) 를 (B*C, N, d) 로 접어 한 번에 처리한다.
채널끼리 어텐션하지 않는다 — 채널 간 결합은 VSN 이 담당한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d) 또는 (B, C, N, d). 후자는 채널을 배치로 접어 처리."""
        folded = x.dim() == 4
        if folded:
            b, c, n, d = x.shape
            x = x.reshape(b * c, n, d)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if folded:
            x = x.reshape(b, c, n, d)
        return x
