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


class GroupedGRN(nn.Module):
    """채널마다 독립 가중치를 갖는 GRN 을 **한 번의 배치 연산**으로 처리한다.

    채널별 nn.Linear 를 파이썬 루프로 돌면 채널 수만큼 커널이 따로 뜬다.
    17채널 × 레이어 4개 = 68회 호출이라 GPU 가 놀고 오버헤드만 쌓인다.
    가중치를 (C, in, out) 텐서로 두고 einsum 으로 한 번에 민다.
    """

    def __init__(self, n_vars: int, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.n_vars = n_vars

        def w(i, o):
            t = nn.Parameter(torch.empty(n_vars, i, o))
            nn.init.xavier_uniform_(t)
            return t

        def b(o):
            return nn.Parameter(torch.zeros(n_vars, o))

        self.w1, self.b1 = w(d_model, hidden), b(hidden)
        self.w2, self.b2 = w(hidden, hidden), b(hidden)
        self.wg, self.bg = w(hidden, d_model * 2), b(d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C, d) → (B, N, C, d)"""
        h = torch.einsum("bncd,cdh->bnch", x, self.w1) + self.b1
        h = torch.einsum("bnch,chk->bnck", torch.nn.functional.elu(h), self.w2) + self.b2
        h = self.dropout(h)
        g = torch.einsum("bnch,chk->bnck", h, self.wg) + self.bg
        a, gate = g.chunk(2, dim=-1)
        return self.norm(x + a * torch.sigmoid(gate))


class DynamicVSN(nn.Module):
    """채널별 표현 → 채널 가중 선택 → 하나의 시퀀스.

    TFT 원논문처럼 **시간축 인코더 앞에서** 변수 선택을 한다.
    인코더 뒤에 두면 인코더가 채널 수만큼 반복 실행돼야 해서 비용이 17배가 된다.
    """

    def __init__(self, n_vars: int, d_model: int, hidden: int, dropout: float,
                 context_size: int | None = None):
        super().__init__()
        self.n_vars = n_vars
        self.select = GatedResidualNetwork(
            d_model * n_vars, hidden, n_vars, dropout=dropout, context_size=context_size
        )
        self.transform = GroupedGRN(n_vars, d_model, hidden, dropout)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, N, d) → (out (B,N,d), weights (B,N,C))"""
        b, c, n, d = x.shape
        if c != self.n_vars:
            raise ValueError(f"채널 수 불일치: {c} != {self.n_vars}")

        # (B, N, C, d) — 시점별로 모든 채널을 보고 가중치를 정한다
        xt = x.permute(0, 2, 1, 3)
        ctx = context.unsqueeze(1).expand(b, n, -1) if context is not None else None
        weights = torch.softmax(self.select(xt.reshape(b, n, c * d), ctx), dim=-1)

        transformed = self.transform(xt)                      # (B, N, C, d)
        out = (weights.unsqueeze(-1) * transformed).sum(dim=-2)
        return out, weights
