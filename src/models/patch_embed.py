"""PatchTST 방식 구간 토큰화.

시점별 토큰이 아니라 **구간(기본 5영업일 = 1주) 단위**로 묶어 토큰을 만든다.
lookback 120, patch_len 5, stride 5 → 24토큰. 시퀀스가 24로 짧아져
어텐션 비용이 크게 준다.

채널 독립(channel-independent): 채널을 섞지 않고 각 채널을 따로 토큰화한다.
채널을 섞으면 TFT 변수선택망이 가중할 대상이 사라져 해석가능성 근거가 없어진다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def num_patches(lookback: int, patch_len: int, stride: int) -> int:
    if lookback < patch_len:
        raise ValueError(f"lookback({lookback})이 patch_len({patch_len})보다 작다")
    return (lookback - patch_len) // stride + 1


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        lookback: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = num_patches(lookback, patch_len, stride)

        self.proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, C, N_patch, d_model)"""
        # (B, L, C) → (B, C, L)
        x = x.transpose(1, 2)
        # (B, C, N, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(2) != self.n_patches:
            raise ValueError(
                f"패치 수 불일치: {patches.size(2)} != {self.n_patches}. "
                f"lookback/patch_len/stride 설정을 확인할 것"
            )
        out = self.proj(patches) + self.pos.unsqueeze(1)
        return self.dropout(out)
