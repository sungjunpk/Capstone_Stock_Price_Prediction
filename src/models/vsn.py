"""TFT 변수 선택 네트워크 (Variable Selection Network).

피처 중요도를 자동 학습한다. 이 프로젝트에서 VSN 은 성능뿐 아니라
**해석가능성 리포트의 근거**라서 가중치를 반드시 밖으로 내보낸다.

static / dynamic 을 분리하고, static 문맥 벡터로 dynamic 선택을 조건화한다(TFT 원논문 구조).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """TFT 의 기본 블록. GLU 게이트 + 잔차 연결.

    context 가 주어지면 더해서 조건화한다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        context_size: int | None = None,
    ):
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ctx = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, output_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)
        self.skip = (
            nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h = self.fc2(torch.nn.functional.elu(h))
        h = self.dropout(h)
        a, b = self.gate(h).chunk(2, dim=-1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))


class StaticVSN(nn.Module):
    """범주형 static covariate → 임베딩 → 가중 선택 → 문맥 벡터."""

    def __init__(self, vocab_sizes: dict[str, int], d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.names = list(vocab_sizes)
        self.embeds = nn.ModuleList(
            nn.Embedding(vocab_sizes[n], d_model) for n in self.names
        )
        n_vars = len(self.names)
        self.select = GatedResidualNetwork(
            d_model * n_vars, hidden, n_vars, dropout=dropout
        )
        self.transforms = nn.ModuleList(
            GatedResidualNetwork(d_model, hidden, d_model, dropout=dropout)
            for _ in self.names
        )

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """codes: (B, n_vars) int64 → (context (B,d), weights (B,n_vars))"""
        embs = [emb(codes[:, i]) for i, emb in enumerate(self.embeds)]
        flat = torch.cat(embs, dim=-1)
        weights = torch.softmax(self.select(flat), dim=-1)

        transformed = torch.stack(
            [t(e) for t, e in zip(self.transforms, embs, strict=True)], dim=1
        )  # (B, n_vars, d)
        context = (weights.unsqueeze(-1) * transformed).sum(dim=1)
        return context, weights


class DynamicVSN(nn.Module):
    """채널별 시퀀스 표현 → 채널 가중 선택 → 하나의 시퀀스.

    입력이 (B, C, N, d) 로 채널이 살아 있어야 가중할 대상이 있다.
    그래서 인코더를 채널 독립으로 둔 것이다.
    """

    def __init__(self, n_vars: int, d_model: int, hidden: int, dropout: float,
                 context_size: int | None = None):
        super().__init__()
        self.n_vars = n_vars
        self.select = GatedResidualNetwork(
            d_model * n_vars, hidden, n_vars, dropout=dropout, context_size=context_size
        )
        self.transforms = nn.ModuleList(
            GatedResidualNetwork(d_model, hidden, d_model, dropout=dropout)
            for _ in range(n_vars)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, N, d) → (out (B,N,d), weights (B,N,C))"""
        b, c, n, d = x.shape
        if c != self.n_vars:
            raise ValueError(f"채널 수 불일치: {c} != {self.n_vars}")

        # (B, N, C*d) — 시점별로 모든 채널을 보고 가중치를 정한다
        flat = x.permute(0, 2, 1, 3).reshape(b, n, c * d)
        ctx = context.unsqueeze(1).expand(b, n, -1) if context is not None else None
        weights = torch.softmax(self.select(flat, ctx), dim=-1)  # (B, N, C)

        transformed = torch.stack(
            [t(x[:, i]) for i, t in enumerate(self.transforms)], dim=-2
        )  # (B, N, C, d)
        out = (weights.unsqueeze(-1) * transformed).sum(dim=-2)
        return out, weights
