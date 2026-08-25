"""RevIN — 윈도우별·채널별 인스턴스 정규화.

비정상(non-stationary) 시계열의 필수 기법. 종목마다, 시기마다 스케일이 다른 것을
윈도우 안에서 표준화하고, 출력에서 되돌린다.

이 프로젝트의 타깃은 이미 수익률이라 가격 역변환은 필요 없다. 대신 윈도우의
변동성으로 타깃을 스케일링하고 출력에서 되돌린다(scale_target) — 변동성 국면 차이를
흡수하는 효과가 있고, "출력 시 역변환"의 의도를 그대로 유지한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_features))
            self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → 정규화된 (B, L, C). 통계는 인스턴스별로 보관한다."""
        # dim=1 → 시간축. 채널별로 따로 통계를 낸다.
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = x.std(dim=1, keepdim=True, unbiased=False).detach()
        out = (x - self._mean) / (self._std + self.eps)
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def denormalize(self, y: torch.Tensor, channel: int = 0) -> torch.Tensor:
        """정규화 공간의 출력을 원래 스케일로 되돌린다.

        y: (B, ...) — 특정 채널 기준으로 예측된 값
        """
        if not hasattr(self, "_mean"):
            raise RuntimeError("forward() 를 먼저 호출해야 통계가 생긴다")
        mean = self._mean[:, 0, channel]
        std = self._std[:, 0, channel]
        shape = (-1,) + (1,) * (y.dim() - 1)
        if self.affine:
            y = (y - self.bias[channel]) / self.weight[channel]
        return y * (std + self.eps).view(shape) + mean.view(shape)

    def scale_of(self, channel: int) -> torch.Tensor:
        """해당 채널의 윈도우 표준편차. 타깃 스케일링에 쓴다. (B,)"""
        if not hasattr(self, "_std"):
            raise RuntimeError("forward() 를 먼저 호출해야 통계가 생긴다")
        return self._std[:, 0, channel] + self.eps
