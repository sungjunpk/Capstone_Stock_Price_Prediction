"""Phase 1 모델 조립.

    dynamic (B,L,C) ─ RevIN ─ patch ─ embed ─→ (B,C,N,d)
                                                 │ 채널 독립 인코더
    static ─ StaticVSN ─→ context (B,d) ─────────┤ (context 로 조건화)
                                                 ↓ DynamicVSN: 채널 가중합
                                              (B,N,d)
    macro (B,L,M) ─ RevIN ─ patch ─ 인코더 ────→ (B,N,d)
                                                 ↓ 크로스어텐션 (Q=종목, KV=매크로)
                                              (B,N,d) ─ pool ─→ (B,d) ─ 헤드 ─→ (B,3)

CLAUDE.md 의 Phase 1 구성 그대로다. Phase 2 는 encoder 만 교체하면 되도록
인코더를 주입 가능한 형태로 두었다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.cross_attention import CrossAttentionBlock
from src.models.encoder import TransformerEncoder
from src.models.patch_embed import PatchEmbedding, num_patches
from src.models.quantile_head import QuantileHead
from src.models.revin import RevIN
from src.models.vsn import DynamicVSN, StaticVSN


@dataclass
class Phase1Config:
    n_dynamic: int
    n_macro: int
    static_vocab: dict[str, int]
    lookback: int = 120
    patch_len: int = 5
    stride: int = 5
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    cross_heads: int = 4
    cross_dropout: float = 0.1
    vsn_hidden: int = 64
    vsn_dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    revin_affine: bool = True
    revin_eps: float = 1e-5
    # 윈도우 변동성으로 출력을 곱하면 상수(무조건부 분위수)조차 표현하기 어려워진다 —
    # 헤드가 1/scale 을 학습해야 하기 때문. 기본은 끈다.
    scale_target: bool = False
    # 학습 데이터의 무조건부 분위수. 헤드 bias 를 여기서 출발시킨다.
    init_quantiles: tuple[float, ...] | None = None
    target_scale_channel: int = 0   # panel 의 첫 피처(ret_1d)를 변동성 기준으로 쓴다

    @classmethod
    def from_config(cls, cfg: dict, *, n_dynamic: int, n_macro: int,
                    static_vocab: dict[str, int]) -> Phase1Config:
        m = cfg["model"]
        return cls(
            n_dynamic=n_dynamic, n_macro=n_macro, static_vocab=static_vocab,
            lookback=int(cfg["features"]["lookback"]),
            patch_len=int(m["patch"]["patch_len"]),
            stride=int(m["patch"]["stride"]),
            d_model=int(m["encoder"]["d_model"]),
            n_heads=int(m["encoder"]["n_heads"]),
            n_layers=int(m["encoder"]["n_layers"]),
            d_ff=int(m["encoder"]["d_ff"]),
            dropout=float(m["encoder"]["dropout"]),
            cross_heads=int(m["cross_attention"]["n_heads"]),
            cross_dropout=float(m["cross_attention"]["dropout"]),
            vsn_hidden=int(m["vsn"]["hidden_size"]),
            vsn_dropout=float(m["vsn"]["dropout"]),
            quantiles=tuple(m["head"]["quantiles"]),
            revin_affine=bool(m["revin"]["affine"]),
            revin_eps=float(m["revin"]["eps"]),
            scale_target=bool(m["revin"].get("scale_target", False)),
        )


@dataclass
class Phase1Output:
    quantiles: torch.Tensor                 # (B, Q) 원래 스케일
    dynamic_weights: torch.Tensor           # (B, N, C) 해석용
    static_weights: torch.Tensor            # (B, n_static) 해석용
    cross_weights: torch.Tensor | None = field(default=None)


class Phase1Model(nn.Module):
    def __init__(self, cfg: Phase1Config):
        super().__init__()
        self.cfg = cfg
        self.n_patches = num_patches(cfg.lookback, cfg.patch_len, cfg.stride)

        # --- 종목 경로
        self.revin_dyn = RevIN(cfg.n_dynamic, cfg.revin_eps, cfg.revin_affine)
        self.patch_dyn = PatchEmbedding(
            cfg.patch_len, cfg.stride, cfg.d_model, cfg.lookback, cfg.dropout
        )
        self.encoder = TransformerEncoder(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout
        )

        # --- 매크로 경로 (별도 인코더 — 성격이 다른 시퀀스라 가중치를 공유하지 않는다)
        self.revin_mac = RevIN(cfg.n_macro, cfg.revin_eps, cfg.revin_affine)
        self.patch_mac = PatchEmbedding(
            cfg.patch_len, cfg.stride, cfg.d_model, cfg.lookback, cfg.dropout
        )
        # 매크로는 VSN 을 안 거치므로 채널을 살려둘 이유가 없다.
        # 채널별로 인코딩한 뒤 합치면 인코더가 13번 도는데, 합친 뒤 한 번 도는 것과
        # 표현력 차이는 없고 비용만 13배다.
        self.macro_merge = nn.Linear(cfg.n_macro * cfg.d_model, cfg.d_model)
        self.encoder_mac = TransformerEncoder(
            cfg.d_model, cfg.cross_heads, max(1, cfg.n_layers - 1), cfg.d_ff, cfg.dropout
        )

        # --- VSN
        self.static_vsn = StaticVSN(
            cfg.static_vocab, cfg.d_model, cfg.vsn_hidden, cfg.vsn_dropout
        )
        self.dynamic_vsn = DynamicVSN(
            cfg.n_dynamic, cfg.d_model, cfg.vsn_hidden, cfg.vsn_dropout,
            context_size=cfg.d_model,
        )

        # --- 결합 + 출력
        self.cross = CrossAttentionBlock(cfg.d_model, cfg.cross_heads, cfg.cross_dropout)
        self.head = QuantileHead(
            cfg.d_model, len(cfg.quantiles), dropout=cfg.cross_dropout,
            init_quantiles=cfg.init_quantiles,
        )

    def forward(
        self,
        dynamic: torch.Tensor,
        macro: torch.Tensor,
        static: torch.Tensor,
        *,
        need_cross_weights: bool = False,
    ) -> Phase1Output:
        """dynamic (B,L,C) / macro (B,L,M) / static (B,n_static) int64"""
        # 종목 경로 — TFT 순서대로 **변수 선택을 인코더 앞에서** 한다.
        # 인코더 뒤에 두면 인코더가 채널 수(17)만큼 반복 실행돼 비용이 17배가 된다.
        x = self.revin_dyn(dynamic)                      # (B,L,C)
        x = self.patch_dyn(x)                            # (B,C,N,d)

        ctx, w_static = self.static_vsn(static)          # (B,d), (B,n_static)
        x, w_dyn = self.dynamic_vsn(x, ctx)              # (B,N,d), (B,N,C)
        x = self.encoder(x)                              # (B,N,d)

        # 매크로 경로 — 채널을 먼저 합치고 한 번만 인코딩
        m = self.revin_mac(macro)
        m = self.patch_mac(m)                            # (B,M,N,d)
        b, n_mac, n, d = m.shape
        m = self.macro_merge(m.permute(0, 2, 1, 3).reshape(b, n, n_mac * d))  # (B,N,d)
        m = self.encoder_mac(m)

        # 결합
        z, w_cross = self.cross(x, m, need_weights=need_cross_weights)
        z = z + ctx.unsqueeze(1)                         # static 문맥을 한 번 더 주입
        pooled = z[:, -1]                                # 마지막 패치 = 가장 최근 구간

        q = self.head(pooled)                            # (B,Q) 정규화 공간

        if self.cfg.scale_target:
            # 윈도우 변동성으로 되돌린다 — RevIN 의 "출력 시 역변환"
            scale = self.revin_dyn.scale_of(self.cfg.target_scale_channel)
            q = q * scale.unsqueeze(-1)

        return Phase1Output(
            quantiles=q, dynamic_weights=w_dyn, static_weights=w_static,
            cross_weights=w_cross,
        )

    def target_scale(self) -> torch.Tensor:
        """손실 계산 시 타깃도 같은 스케일로 맞추고 싶을 때 쓴다. (B,)"""
        return self.revin_dyn.scale_of(self.cfg.target_scale_channel)
